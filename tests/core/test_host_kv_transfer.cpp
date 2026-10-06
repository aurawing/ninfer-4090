#include "core/kvmem/host_kv_archive.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::kvmem;

namespace {
void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}
template <class F> void rejects(F&& operation, const char* message) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    require(rejected, message);
}
std::vector<std::byte> pattern(std::size_t size, unsigned plane, unsigned page, unsigned version) {
    std::vector<std::byte> result(size);
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = std::byte((i * 37 + plane * 19 + page * 71 + version * 43) & 255);
    }
    return result;
}

void read_event_allocation_failure_unwinds(HostArchiveMode mode) {
    LayoutBuilder builder;
    const auto pool_layout=plan_paged_kv_pool(builder,
        {2,4,1,PagedKVPlaneOrder::PageMajor,{{DType::I8,16,2}}});
    const auto archive_layout=plan_host_kv_archive(pool_layout,1,256);
    HostKVArchiveFaultInjection fault;
    fault.fail_read_done_allocation=true;
    bool allocation_failed=false;
    try { HostKVArchive archive(archive_layout,mode,false,fault); }
    catch(const std::bad_alloc&) { allocation_failed=true; }
    require(allocation_failed,"injected archive read-event allocation must unwind as an ordinary bad_alloc");
    // Failed construction must clean up initialized resources, leaving the same
    // validated layout reusable by an ordinary archive owner.
    HostKVArchive retry(archive_layout,mode);
    retry.synchronize();
    require(retry.frontier(0)==0,"archive reconstruction retains an empty frontier");
}

void roundtrip(HostArchiveMode mode, PagedKVPlaneOrder order, bool os_lock = false) {
    LayoutBuilder builder;
    PagedKVPoolSpec spec{8, 8, 1, order, {}};
    for (int layer = 0; layer < 2; ++layer) {
        spec.planes.insert(spec.planes.end(), {{DType::I8, 64, 4}, {DType::I8, 32, 4},
                                             {DType::FP16, 1, 4}, {DType::FP16, 2, 4}});
    }
    const auto pool_layout = plan_paged_kv_pool(builder, spec);
    DeviceBuffer backing(builder.finish(256));
    PagedKVPool pool({backing.p, backing.bytes}, pool_layout);
    auto archive_layout = plan_host_kv_archive(pool, 4, 512);
    require(archive_layout.layers.size() == 2 && archive_layout.logical_pages == 8,
            "archive layer/page shape");
    require(archive_layout.layers[0][3].offset < archive_layout.layers[1][0].offset,
            "archive must use layer then plane then page order");
    HostKVArchive archive(archive_layout, mode, os_lock);
    require(archive.os_locked() == os_lock, "explicit OS lock status");
    if (mode != HostArchiveMode::Auto) { require(archive.mode() == mode, "explicit archive mode"); }
    require(archive.frontier(0) == 0, "new archive frontier");
    if (os_lock) require(archive.committed_bytes() == archive_layout.bytes, "OS lock commits full archive during loading");
    if (archive.mode() == HostArchiveMode::Pageable && !os_lock) {
        require(archive.committed_bytes() == 0, "pageable archive must initially reserve only");
    }
    DeviceContext device;
    // Constructor descriptor copies on legacy stream 0 must finish before nonblocking work.
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::array<std::int32_t, 4> source{5, 2, 3, 7};
    for (std::size_t p = 0; p < 8; ++p) {
        for (std::size_t logical = 0; logical < source.size(); ++logical) {
            const auto bytes = pattern(pool.page_bytes(p), static_cast<unsigned>(p),
                                       static_cast<unsigned>(logical), 0);
            pool.copy_page_from_host(p, source[logical], bytes.data(), device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
        }
    }
    for (std::size_t layer = 0; layer < 2; ++layer) {
        archive.writeback(pool, layer, 0, source, 250, device.stream);
    }
    archive.synchronize();
    const auto full_commit = archive.committed_bytes();
    require(archive.frontier(0) == 250 && archive.frontier(1) == 250, "exact partial frontier");
    const std::array<std::int32_t, 4> destination{1, 6, 4, 0};
    for (std::size_t layer = 0; layer < 2; ++layer) {
        archive.restore_to_pool(pool, layer, 0, destination, device.stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    for (std::size_t p = 0; p < 8; ++p) {
        for (std::size_t page = 0; page < 4; ++page) {
            auto expected = pattern(pool.page_bytes(p), static_cast<unsigned>(p),
                                    static_cast<unsigned>(page), 0);
            auto host = archive.pages(p / 4, p % 4, static_cast<std::uint32_t>(page), 1);
            require(std::equal(host.begin(), host.end(), expected.begin()), "archive byte mirror");
            std::vector<std::byte> actual(expected.size());
            pool.copy_page_to_host(p, destination[page], actual.data(), device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            require(actual == expected, "fragmented device roundtrip");
        }
    }
    // Overwrite a retained page and ensure every plane is updated before publishing Both.
    const std::array<std::int32_t, 1> update{0};
    for (std::size_t p = 0; p < 4; ++p) {
        auto bytes = pattern(pool.page_bytes(p), static_cast<unsigned>(p), 2, 1);
        pool.copy_page_from_host(p, 0, bytes.data(), device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
    }
    archive.writeback(pool, 0, 2, update, 250, device.stream);
    archive.synchronize();
    for (std::size_t p = 0; p < 4; ++p) {
        auto expected = pattern(pool.page_bytes(p), static_cast<unsigned>(p), 2, 1);
        auto bytes = archive.pages(0, p, 2, 1);
        require(std::equal(bytes.begin(), bytes.end(), expected.begin()), "overwrite writeback");
    }
    const auto generation = archive.generation();
    CUDA_CHECK(cudaLaunchHostFunc(device.load_stream, [](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }, nullptr));
    archive.restore_to_pool(pool, 0, 0, destination, device.load_stream);
    archive.trim(130);
    require(cudaStreamQuery(device.load_stream) == cudaSuccess,
            "trim must drain outstanding archive H2D reads");
    require(archive.frontier(0) == 130 && archive.frontier(1) == 130,
            "trim must keep exact token frontier");
    require(archive.generation() == generation + 1, "trim invalidates old transfer generation");
    rejects([&] { (void)archive.pages(0, 0, 3, 1); }, "trimmed page cannot be read");
    auto retained = archive.pages(0, 0, 2, 1);
    const auto expected = pattern(retained.size(), 0, 2, 1);
    require(std::equal(retained.begin(), retained.end(), expected.begin()), "retain partial page");
    if (archive.mode() == HostArchiveMode::Pageable && !os_lock) {
        require(archive.committed_bytes() < full_commit, "trim decommits full trailing OS pages");
    }
    // Trim must drain an outstanding writeback before releasing its destination.
    archive.writeback(pool, 0, 2, update, 130, device.stream);
    archive.trim(0);
    require(archive.frontier(0) == 0, "empty trim");
    if (os_lock) require(archive.committed_bytes() == archive_layout.bytes, "trim must preserve OS locked allocation");
    if (archive.mode() == HostArchiveMode::Pageable && !os_lock) {
        require(archive.committed_bytes() == 0, "empty trim decommits all pageable storage");
    }
    rejects([&] { archive.writeback(pool, 0, 1, update, 128, device.stream); },
            "writeback cannot create a logical hole");
    archive.writeback(pool, 0, 0, update, 12, device.stream);
    archive.synchronize();
    require(archive.frontier(0) == 12, "resume after empty trim");
    rejects([&] { archive.writeback(pool, 0, 1, update, 32, device.stream); },
            "partial frontier extension must rewrite its actual tail page");
    const std::array<std::int32_t, 1> invalid_page{-1};
    rejects([&] { archive.writeback(pool, 0, 0, invalid_page, 12, device.stream); },
            "negative physical page cannot be copied");
    rejects([&] { archive.writeback(pool, 0, 0, update, 513, device.stream); },
            "frontier cannot exceed max_context");
    rejects([&] { archive.trim(513); }, "trim cannot grow frontier");
}
}

int main(int argc,char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { return 77; }
    try {
        for(auto mode : {HostArchiveMode::Pinned,HostArchiveMode::Pageable,HostArchiveMode::Auto})
            read_event_allocation_failure_unwinds(mode);
        if(argc==2 && std::strcmp(argv[1],"--constructor-unwind-only")==0) return 0;
        require(keep_pinned_archive(true, std::uint64_t{4} << 30), "4 GiB boundary admits pin");
        require(!keep_pinned_archive(true, (std::uint64_t{4} << 30) - 1), "headroom rejection");
        require(!keep_pinned_archive(false, std::numeric_limits<std::uint64_t>::max()),
                "failed pin must fall back");
        rejects([] { check_host_archive_admission(1024, std::uint64_t{4} << 30); },
                "initial admission includes archive plus 4 GiB");
        check_host_archive_admission(1024, (std::uint64_t{4} << 30) + 1024);
        try { check_host_archive_admission(1024, std::uint64_t{4} << 30); }
        catch (const std::runtime_error& error) {
            const std::string message = error.what();
            require(message.find("--max-context") != std::string::npos &&
                    message.find("--kv-dtype") != std::string::npos &&
                    message.find("pageable") != std::string::npos, "admission lacks actionable remedies");
        }
        for (auto mode : {HostArchiveMode::Pinned, HostArchiveMode::Pageable, HostArchiveMode::Auto}) {
            roundtrip(mode, PagedKVPlaneOrder::PageMajor);
            roundtrip(mode, PagedKVPlaneOrder::HeadMajor);
        }
        roundtrip(HostArchiveMode::Pageable, PagedKVPlaneOrder::PageMajor, true);
        roundtrip(HostArchiveMode::Auto, PagedKVPlaneOrder::HeadMajor, true);
        rejects([] { HostKVArchive archive({}, HostArchiveMode::Pinned, true); }, "CUDA pin and OS lock must conflict");
        std::cout << "PASS host KV archive modes, byte roundtrip, writeback and trim\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL host KV archive: " << error.what() << '\n';
        return 1;
    }
}

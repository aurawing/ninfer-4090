#include "core/kvmem/host_kv_transfer.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::kvmem;
namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}
template<class F> void rejects(F&& f, const char* message) {
    bool caught = false;
    try { f(); } catch (const std::exception&) { caught = true; }
    require(caught, message);
}
std::vector<std::byte> bytes(std::size_t count, unsigned plane, unsigned page, unsigned version = 0) {
    std::vector<std::byte> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        result[i] = std::byte((i * 37 + plane * 53 + page * 17 + version * 97) & 255);
    }
    return result;
}
void staging_plan_contract() {
    LayoutBuilder builder;
    const auto layout = plan_paged_kv_pool(builder, {1, 4096, 1, PagedKVPlaneOrder::PageMajor,
        {{DType::I8, 256, 4}, {DType::I8, 256, 4},
         {DType::FP16, 4, 4}, {DType::FP16, 4, 4}}});
    DeviceBuffer storage(builder.finish(256));
    PagedKVPool pool({storage.p, storage.bytes}, layout);
    auto host = plan_host_kv_archive(pool, 4, 262144);
    auto plan = plan_host_kv_staging(host, 2048);
    require(plan.maximum_layer_stream_bytes == 264ULL * 1024 * 1024, "real INT8 stream budget");
    require(plan.capacity_bytes == 328ULL * 1024 * 1024, "stage must add one 64 MiB tile");
    rejects([&] { (void)plan_host_kv_staging(host, 2048, plan.capacity_bytes - 1); },
            "undersized stage override must fail");
}
void exercise(HostArchiveMode mode, PagedKVPlaneOrder order) {
    LayoutBuilder builder;
    PagedKVPoolSpec spec{48, 64, 1, order, {}};
    for (int layer = 0; layer < 2; ++layer) {
        spec.planes.insert(spec.planes.end(), {{DType::I8, 16, 2}, {DType::I8, 16, 2},
                                             {DType::FP16, 1, 2}, {DType::FP16, 1, 2}});
    }
    auto pool_layout = plan_paged_kv_pool(builder, spec);
    DeviceBuffer backing(builder.finish(256));
    PagedKVPool pool({backing.p, backing.bytes}, pool_layout);
    DeviceContext device;
    CUDA_CHECK(cudaDeviceSynchronize());
    HostKVArchive archive(plan_host_kv_archive(pool, 4, 4096), mode);
    std::vector<std::int32_t> source(40);
    for (std::size_t page = 0; page < 40; ++page) { source[page] = static_cast<int>(page * 7 % 48); }
    for (std::size_t p = 0; p < 8; ++p) {
        for (std::size_t page = 0; page < source.size(); ++page) {
            auto value = bytes(pool.page_bytes(p), static_cast<unsigned>(p), static_cast<unsigned>(page));
            pool.copy_page_from_host(p, source[page], value.data(), device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
        }
    }
    archive.writeback(pool, 0, 0, source, 2560, device.stream);
    archive.writeback(pool, 1, 0, source, 2560, device.stream);
    archive.synchronize();
    const auto plan = plan_host_kv_staging(archive.layout(), 44);
    DeviceBuffer staging(plan.capacity_bytes);
    PinnedHostBuffer check_a(pool.page_bytes(0)), check_b(pool.page_bytes(1));
    HostKVTransferEngine engine(archive, {staging.p, staging.bytes}, plan);
    require(engine.pinned_ring_bytes() == (mode == HostArchiveMode::Pageable ?
            4 * kHostKVTransferTileBytes : 0), "mode must allocate exactly the required ring");
    // All planes and a different layer are queued before the first consumer waits.
    std::vector<HostKVTransferTicket> first;
    std::size_t offset = 0;
    for (std::size_t p = 0; p < 4; ++p) {
        first.push_back(engine.prefetch(0, p, 20, 20, offset));
        offset += 20 * pool.page_bytes(p);
    }
    auto next_layer = engine.prefetch(1, 0, 0, 1, offset);
    rejects([&] { (void)engine.prefetch(1, 0, 0, 1, 0); },
            "cannot overwrite a live staging segment");
    for (std::size_t p = 0; p < first.size(); ++p) {
        engine.wait(first[p], device.stream);
        auto staged = engine.staged(first[p]);
        PinnedHostBuffer actual(staged.bytes);
        CUDA_CHECK(cudaMemcpyAsync(actual.data(), staged.data, staged.bytes,
                                    cudaMemcpyDeviceToHost, device.stream));
        engine.release(first[p], device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        for (std::size_t i = 0; i < 20; ++i) {
            auto expected = bytes(pool.page_bytes(p), static_cast<unsigned>(p), static_cast<unsigned>(20 + i));
            require(std::memcmp(static_cast<std::byte*>(actual.data()) + i * expected.size(),
                                expected.data(), expected.size()) == 0, "multi-plane staged bytes");
        }
    }
    engine.wait(next_layer, device.stream);
    CUDA_CHECK(cudaMemcpyAsync(check_b.data(), engine.staged(next_layer).data, check_b.size(),
                                cudaMemcpyDeviceToHost, device.stream));
    engine.release(next_layer, device.stream);
    engine.synchronize();
    const auto next_expected = bytes(check_b.size(), 4, 0);
    require(std::memcmp(check_b.data(), next_expected.data(), next_expected.size()) == 0,
            "cross-layer prefetch must preserve the next layer's own bytes");
    // Delay consumption, enqueue a conflicting transfer, and prove the old bytes survived.
    auto a = engine.prefetch(0, 0, 0, 1, 0);
    engine.wait(a, device.stream);
    CUDA_CHECK(cudaLaunchHostFunc(device.stream, [](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }, nullptr));
    CUDA_CHECK(cudaMemcpyAsync(check_a.data(), engine.staged(a).data, check_a.size(),
                                cudaMemcpyDeviceToHost, device.stream));
    engine.release(a, device.stream);
    auto b = engine.prefetch(0, 1, 3, 1, 0);
    engine.wait(b, device.load_stream);
    CUDA_CHECK(cudaMemcpyAsync(check_b.data(), engine.staged(b).data, check_b.size(),
                                cudaMemcpyDeviceToHost, device.load_stream));
    engine.release(b, device.load_stream);
    engine.synchronize();
    auto expected_a = bytes(check_a.size(), 0, 0), expected_b = bytes(check_b.size(), 1, 3);
    require(std::memcmp(check_a.data(), expected_a.data(), expected_a.size()) == 0,
            "release must protect old consumption before overwrite");
    require(std::memcmp(check_b.data(), expected_b.data(), expected_b.size()) == 0,
            "new segment must contain new plane bytes");
    // Enough consecutive copies to reuse every ring slot and event ticket repeatedly.
    for (unsigned repeat = 0; repeat < 48; ++repeat) {
        auto ticket = engine.prefetch(repeat % 2, repeat % 4, repeat % 40, 1, 0);
        engine.wait(ticket, device.stream);
        CUDA_CHECK(cudaMemcpyAsync(check_a.data(), engine.staged(ticket).data,
                                    pool.page_bytes(repeat % 4), cudaMemcpyDeviceToHost, device.stream));
        engine.release(ticket, device.stream);
        engine.synchronize();
        auto expected = bytes(pool.page_bytes(repeat % 4), (repeat % 2) * 4 + repeat % 4, repeat % 40);
        require(std::memcmp(check_a.data(), expected.data(), expected.size()) == 0, "ring reuse bytes");
    }
    const std::array<std::int32_t, 1> changed{0};
    for (std::size_t p = 0; p < 4; ++p) {
        auto value = bytes(pool.page_bytes(p), static_cast<unsigned>(p), 2, 1);
        pool.copy_page_from_host(p, 0, value.data(), device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
    }
    auto writeback = engine.writeback(pool, 0, 2, changed, 2560, device.stream);
    writeback.get();
    for (std::size_t p = 0; p < 4; ++p) {
        auto expected = bytes(pool.page_bytes(p), static_cast<unsigned>(p), 2, 1);
        auto host = archive.pages(0, p, 2, 1);
        require(std::equal(host.begin(), host.end(), expected.begin()), "ring/direct writeback bytes");
    }
    // Capture old archive bytes, then enqueue a delayed GPU producer and overwrite
    // the same host page. FIFO DMA must finish the old read before that writeback.
    auto old_read = engine.prefetch(0, 0, 2, 1, 0);
    engine.wait(old_read, device.load_stream);
    CUDA_CHECK(cudaMemcpyAsync(check_a.data(), engine.staged(old_read).data, check_a.size(),
                                cudaMemcpyDeviceToHost, device.load_stream));
    engine.release(old_read, device.load_stream);
    PinnedHostBuffer produced(pool.page_bytes(0));
    const auto replacement = bytes(produced.size(), 0, 2, 2);
    std::memcpy(produced.data(), replacement.data(), replacement.size());
    CUDA_CHECK(cudaLaunchHostFunc(device.stream, [](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }, nullptr));
    pool.copy_page_from_host(0, 0, produced.data(), device.stream);
    auto delayed_writeback = engine.writeback(pool, 0, 2, changed, 2560, device.stream);
    delayed_writeback.get();
    engine.synchronize();
    const auto old_value = bytes(check_a.size(), 0, 2, 1);
    require(std::memcmp(check_a.data(), old_value.data(), old_value.size()) == 0,
            "queued archive read must survive a later overwrite");
    const auto new_value = archive.pages(0, 0, 2, 1);
    require(std::equal(new_value.begin(), new_value.end(), replacement.begin()),
            "writeback must wait for the producing CUDA stream");
    auto stale = engine.prefetch(0, 0, 0, 1, 0);
    rejects([&] { archive.trim(128); }, "attached archive mutations must go through transfer owner");
    engine.trim(130);
    rejects([&] { engine.wait(stale, device.stream); }, "trim invalidates prior transfer tickets");
    require(archive.frontier(0) == 130 && archive.frontier(1) == 130, "engine trim frontier");
}
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { return 77; }
    try {
        staging_plan_contract();
        for (auto mode : {HostArchiveMode::Pinned, HostArchiveMode::Pageable}) {
            exercise(mode, PagedKVPlaneOrder::PageMajor);
            exercise(mode, PagedKVPlaneOrder::HeadMajor);
        }
        std::cout << "PASS host KV transfer ring/direct, cross-layer prefetch, events and trim\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL host KV transfer engine: " << error.what() << '\n';
        return 1;
    }
}

#include "core/kvmem/host_kv_transfer.h"
#include "core/device.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::kvmem;

// Manual end-to-end transfer benchmark, not an inference benchmark. Synthetic
// bytes use the Qwen3.8-27B INT8 production K/V/FP16-scale page geometry.
int main(int argc, char** argv) try {
    if (argc != 2 || (std::string_view(argv[1]) != "pinned" &&
                      std::string_view(argv[1]) != "pageable")) {
        throw std::invalid_argument("usage: ninfer_host_kv_transfer_bench pinned|pageable");
    }
    const auto mode = std::string_view(argv[1]) == "pinned" ?
        HostArchiveMode::Pinned : HostArchiveMode::Pageable;
    constexpr std::uint32_t logical_pages = 4096, streamed_pages = 2048;
    constexpr std::size_t layers = 16, repeats = 16, warmups = 2;
    DeviceContext device;
    const auto physical_before = available_physical_memory_bytes();
    std::unique_ptr<HostKVArchive> archive;
    {
        LayoutBuilder builder;
        PagedKVPoolSpec spec{logical_pages, logical_pages, 1, PagedKVPlaneOrder::PageMajor, {}};
        for (std::size_t layer = 0; layer < layers; ++layer) {
            spec.planes.insert(spec.planes.end(), {{DType::I8, 256, 4}, {DType::I8, 256, 4},
                                                 {DType::FP16, 4, 4}, {DType::FP16, 4, 4}});
        }
        auto layout = plan_paged_kv_pool(builder, spec);
        DeviceBuffer backing(builder.finish(256));
        PagedKVPool pool({backing.p, backing.bytes}, layout);
        // Initial pool descriptors use stream 0; the measurement stream is nonblocking.
        CUDA_CHECK(cudaDeviceSynchronize());
        archive = std::make_unique<HostKVArchive>(plan_host_kv_archive(pool, 4, 262144), mode);
        std::vector<std::int32_t> ids(logical_pages);
        std::iota(ids.begin(), ids.end(), 0);
        for (std::size_t layer = 0; layer < layers; ++layer) {
            for (std::size_t p = 0; p < 4; ++p) {
                const auto& plane = pool.plane(layer * 4 + p);
                CUDA_CHECK(cudaMemsetAsync(plane.data, static_cast<int>(layer * 4 + p + 1),
                                           plane.bytes(), device.stream));
            }
            archive->writeback(pool, layer, 0, ids, 262144, device.stream);
        }
        archive->synchronize();
    } // Full GPU source is freed; timed reads are exclusively from the host mirror.
    const auto plan = plan_host_kv_staging(archive->layout(), logical_pages - streamed_pages);
    DeviceBuffer staging(plan.capacity_bytes);
    PinnedHostBuffer check(plan.maximum_layer_stream_bytes);
    HostKVTransferEngine engine(*archive, {staging.p, staging.bytes}, plan);
    const auto physical_ready = available_physical_memory_bytes();
    const auto feed_layer = [&](std::size_t layer) {
        std::vector<HostKVTransferTicket> tickets;
        tickets.reserve(6);
        std::size_t offset = 0;
        for (std::size_t p = 0; p < 4; ++p) {
            const auto page_bytes = archive->layout().layers[layer][p].page_bytes;
            const auto per_tile = static_cast<std::uint32_t>(kHostKVTransferTileBytes / page_bytes);
            for (std::uint32_t begin = 0; begin < streamed_pages; begin += per_tile) {
                const auto count = std::min(per_tile, streamed_pages - begin);
                tickets.push_back(engine.prefetch(layer, p, streamed_pages + begin, count, offset));
                offset += count * page_bytes;
            }
        }
        for (auto ticket : tickets) {
            engine.wait(ticket, device.stream);
            engine.release(ticket, device.stream);
        }
        engine.synchronize();
    };
    for (std::size_t i = 0; i < warmups; ++i) { feed_layer(i); }
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < repeats; ++i) { feed_layer(i % layers); }
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    CUDA_CHECK(cudaMemcpyAsync(check.data(), staging.p, check.size(), cudaMemcpyDeviceToHost, device.stream));
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    std::size_t offset = 0;
    for (std::size_t p = 0; p < 4; ++p) {
        const auto bytes = archive->layout().layers.back()[p].page_bytes * streamed_pages;
        const auto expected = static_cast<unsigned char>((layers - 1) * 4 + p + 1);
        const auto* begin = static_cast<const unsigned char*>(check.data()) + offset;
        if (!std::all_of(begin, begin + bytes, [expected](unsigned char value) { return value == expected; })) {
            throw std::runtime_error("staging byte validation failed");
        }
        offset += bytes;
    }
    const auto total = repeats * plan.maximum_layer_stream_bytes;
    std::cout << std::setprecision(12)
              << "{\"mode\":\"" << argv[1] << "\",\"max_context\":262144,\"layers\":" << layers
              << ",\"streamed_pages\":" << streamed_pages << ",\"archive_bytes\":" << archive->layout().bytes
              << ",\"layer_bytes\":" << plan.maximum_layer_stream_bytes << ",\"staging_bytes\":" << plan.capacity_bytes
              << ",\"ring_bytes\":" << engine.pinned_ring_bytes() << ",\"warmups\":" << warmups
              << ",\"repeats\":" << repeats << ",\"transferred_bytes\":" << total << ",\"seconds\":" << seconds
              << ",\"gib_per_second\":" << total / seconds / (1ULL << 30)
              << ",\"physical_before\":" << physical_before << ",\"physical_ready\":" << physical_ready
              << ",\"validated_bytes\":" << check.size() << ",\"byte_validation\":true}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "host KV transfer benchmark: " << error.what() << '\n';
    return 1;
}

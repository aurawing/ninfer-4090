#include "core/device.h"
#include "core/kvmem/host_kv_transfer.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::kvmem;

namespace {
void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
template <class F> void rejects(F&& f) {
    bool caught = false;
    try {
        f();
    } catch (const std::exception&) {
        caught = true;
    }
    require(caught, "invalid index transfer accepted");
}
void exercise(HostKVTransferEngine& engine, DeviceContext& device, std::span<const std::byte> layer,
              std::span<const std::byte> next_layer) {
    PinnedHostBuffer actual(layer.size()), following(next_layer.size());
    auto ticket = engine.prefetch_index(layer, 0);
    rejects([&] { (void)engine.prefetch_index(next_layer, 0); });
    engine.wait(ticket, device.stream);
    auto staged = engine.staged(ticket);
    CUDA_CHECK(cudaLaunchHostFunc(
        device.stream, [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); },
        nullptr));
    CUDA_CHECK(cudaMemcpyAsync(actual.data(), staged.data, actual.size(), cudaMemcpyDeviceToHost,
                               device.stream));
    engine.release(ticket, device.stream);
    // Reusing an overlapping range must wait on the preceding consumer event.
    auto next = engine.prefetch_index(next_layer, 0);
    engine.wait(next, device.stream);
    staged = engine.staged(next);
    CUDA_CHECK(cudaMemcpyAsync(following.data(), staged.data, following.size(),
                               cudaMemcpyDeviceToHost, device.stream));
    engine.release(next, device.stream);
    engine.synchronize();
    require(std::memcmp(actual.data(), layer.data(), layer.size()) == 0, "first layer bytes");
    require(std::memcmp(following.data(), next_layer.data(), next_layer.size()) == 0,
            "next layer bytes");
    rejects([&] { (void)engine.staged(ticket); });
    rejects([&] { (void)engine.prefetch_index({}, 0); });
    rejects([&] { (void)engine.prefetch_index(layer, SIZE_MAX); });
    engine.reset();
    rejects([&] { engine.wait(next, device.stream); });
}
void run(HostArchiveMode mode) {
    LayoutBuilder builder;
    auto layout = plan_paged_kv_pool(
        builder,
        {2,
         64,
         1,
         PagedKVPlaneOrder::PageMajor,
         {{DType::I8, 256, 4}, {DType::I8, 256, 4}, {DType::FP16, 4, 4}, {DType::FP16, 4, 4}}});
    DeviceBuffer backing(builder.finish(256));
    PagedKVPool pool({backing.p, backing.bytes}, layout);
    HostKVArchive archive(plan_host_kv_archive(pool, 4, 4096), mode);
    auto plan = plan_host_kv_staging(archive.layout(), 2);
    DeviceBuffer staging(plan.capacity_bytes);
    DeviceContext device;
    // Real 262K Qwen27B Mean-K layer stride: 4096 pages x 4 heads x D=256 FP16.
    constexpr std::size_t layer_bytes = 4096 * 4 * 256 * 2;
    std::vector<std::byte> pageable(layer_bytes * 2);
    for (std::size_t i = 0; i < pageable.size(); ++i)
        pageable[i] = std::byte((i * 37 + i / layer_bytes * 79) & 255);
    PinnedHostBuffer pinned(mode == HostArchiveMode::Pinned ? pageable.size() : 1);
    const std::byte* source = pageable.data();
    if (mode == HostArchiveMode::Pinned) {
        std::memcpy(pinned.data(), source, pageable.size());
        source = static_cast<const std::byte*>(pinned.data());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    HostKVTransferEngine engine(archive, {staging.p, staging.bytes}, plan);
    require(engine.pinned_ring_bytes() ==
                (mode == HostArchiveMode::Pinned ? 0 : 4 * kHostKVTransferTileBytes),
            "index transfer reuses existing ring");
    if (mode == HostArchiveMode::Pageable) {
        std::vector<std::byte> oversized(kHostKVTransferTileBytes + 1);
        rejects([&] { (void)engine.prefetch_index(oversized, 0); });
    } else {
        rejects([&] { (void)engine.prefetch_index(pageable, 0); });
        void* write_combined = nullptr;
        CUDA_CHECK(cudaHostAlloc(&write_combined, layer_bytes, cudaHostAllocWriteCombined));
        rejects([&] {
            (void)engine.prefetch_index(
                {static_cast<const std::byte*>(write_combined), layer_bytes}, 0);
        });
        CUDA_CHECK(cudaFreeHost(write_combined));
    }
    exercise(engine, device, {source, layer_bytes}, {source + layer_bytes, layer_bytes});
    std::cout << "mode=" << (mode == HostArchiveMode::Pinned ? "pinned" : "pageable")
              << " layer_index_bytes=" << layer_bytes << " h2d_bytes=" << layer_bytes * 2 << '\n';
}
} // namespace
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        return 77;
    try {
        run(HostArchiveMode::Pageable);
        run(HostArchiveMode::Pinned);
        std::cout << "PASS Mean-K layer index transfer\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

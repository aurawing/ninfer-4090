#include "targets/qwen3_6/impl/runtime/tiered_context.h"
#include "core/device.h"
#include <ninfer/ops/gqa_attention.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::targets::qwen3_6;
using namespace ninfer::targets::qwen3_6::detail;

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

// Source storage stays alive until the caller synchronizes this consumer stream.
void upload_on_stream(DeviceBuffer& buffer, const void* source, std::size_t bytes,
                      cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(buffer.p, source, bytes, cudaMemcpyHostToDevice, stream));
}

// A reusable input must retain the preceding consumer's bytes and publish its
// replacement before the following consumer. Pinned storage keeps the controlled
// upload asynchronous so incidental pageable staging cannot hide either edge.
void exercise_ordered_input_uploads() {
    DeviceContext device;
    constexpr std::size_t count = 64 * 4 * 256;
    constexpr auto bytes = count * sizeof(std::uint16_t);
    constexpr std::uint16_t previous = 0x3ef0; // BF16 .46875
    constexpr std::uint16_t replacement = 0x3f00; // BF16 .5
    DeviceBuffer input(bytes), keys(bytes), cache_keys(2 * bytes), snapshots(2 * bytes),
        positions_memory(128 * sizeof(std::int32_t)), table_memory(2 * sizeof(std::int32_t));
    PinnedHostBuffer source(2 * bytes);
    auto* data = static_cast<std::uint16_t*>(source.data());
    std::fill_n(data, count, previous);
    std::fill_n(data + count, count, replacement);
    std::vector<std::int32_t> positions(128);
    for (int token = 0; token < 128; ++token) positions[token] = token;
    const std::int32_t physical_pages[] = {0, 1};
    upload_on_stream(input, data, bytes, device.stream);
    upload_on_stream(positions_memory, positions.data(), positions.size() * sizeof(std::int32_t),
                     device.stream);
    upload_on_stream(table_memory, physical_pages, sizeof(physical_pages), device.stream);
    CUDA_CHECK(cudaMemsetAsync(keys.p, 0, bytes, device.stream));
    Tensor k(keys.p, DType::BF16, {256, 4, 64});
    Tensor v(input.p, DType::BF16, {256, 4, 64});
    Tensor first_positions(positions_memory.p, DType::I32, {64});
    Tensor next_positions(static_cast<std::int32_t*>(positions_memory.p) + 64, DType::I32, {64});
    const PagedKVLayerView cache{
        .k_pages = Tensor(cache_keys.p, DType::BF16, {256, 64, 4, 2}),
        .v_pages = Tensor(snapshots.p, DType::BF16, {256, 64, 4, 2}),
        .block_table = Tensor(table_memory.p, DType::I32, {2}),
        .head_dim = 256,
        .num_kv_heads = 4};
    // Warm the launch before the gate so lazy module initialization cannot drain it.
    ops::gqa_kv_append(k, v, first_positions, cache, device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    struct Gate {
        std::atomic<bool> release{false};
        std::atomic<bool> timed_out{false};
        ~Gate() { release.store(true); }
    } gate;
    // Allocate every CUDA resource before holding the producer/consumer stream.
    CUDA_CHECK(cudaLaunchHostFunc(device.stream, [](void* pointer) {
        auto& state = *static_cast<Gate*>(pointer);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!state.release.load()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                state.timed_out.store(true);
                return;
            }
            std::this_thread::yield();
        }
    }, &gate));
    ops::gqa_kv_append(k, v, first_positions, cache, device.stream);
    upload_on_stream(input, data + count, bytes, device.stream);
    ops::gqa_kv_append(k, v, next_positions, cache, device.stream);
    gate.release.store(true);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    std::vector<std::uint16_t> observed(2 * count);
    snapshots.copy_to_host(observed.data(), 2 * bytes);
    require(!gate.timed_out.load(), "ordered input upload gate must not time out");
    require(std::all_of(observed.begin(), observed.begin() + count,
                        [](auto bits) { return bits == previous; }),
            "input upload must preserve the preceding consumer's source bytes");
    require(std::all_of(observed.begin() + count, observed.end(),
                        [](auto bits) { return bits == replacement; }),
            "input upload must publish replacement bytes to the following consumer");
}

std::uint16_t bf16(float x) {
    auto u = std::bit_cast<std::uint32_t>(x);
    return std::uint16_t((u + 0x7fffU + ((u >> 16) & 1U)) >> 16);
}

float value(std::uint32_t token, std::uint32_t layer) {
    return float(token / 64) * 0.25f + float(layer + 1) * 0.03125f;
}

// Independent FP64 oracle: Q=K=0 gives a uniform causal mean of original BF16 V.
double oracle(std::uint32_t position, std::uint32_t layer) {
    double sum = 0;
    for (std::uint32_t key = 0; key <= position; ++key) sum += value(key, layer);
    return sum / (position + 1);
}

void exercise(bool shadow, DType dtype, HostKVArchiveMode mode = HostKVArchiveMode::Pageable) {
    DeviceContext device;
    LayoutBuilder pool_builder;
    DecoderStateSpec spec;
    spec.capacity                  = shadow ? 320 : 384;
    spec.full_attention_layers     = 16;
    spec.kv_heads                  = 4;
    spec.attention_head_dim        = 256;
    spec.kv_dtype                  = dtype;
    spec.kv_quant_group            = dtype == DType::I8 ? 64 : 0;
    spec.text_physical_page_groups = shadow ? 5 : 6;
    spec.allow_tiered_text_pages   = true;
    spec.linear_attention          = {1, 1, 1, 1, 1, 1};
    const auto layout              = plan_decoder_state(pool_builder, spec).text_kv;
    DeviceBuffer pool_memory(pool_builder.finish(256));
    PagedKVCache cache({pool_memory.p, pool_memory.bytes}, layout);
    auto lease = cache.pool().reserve(shadow ? 5 : 6);
    lease.materialize_pages(shadow ? 5 : 6, device.stream);
    lease.bind_row(0, device.stream);
    TieredKVOptions options;
    options.sink_tokens  = 64;
    options.view_tokens  = 192;
    options.host_archive = mode;
    const auto plan      = plan_tiered_runtime(layout.pool, 320, 3, 64, options, shadow, true);
    DeviceBuffer runtime_memory(plan.bytes);
    DeviceBuffer q_memory(256 * 24 * 64 * 2), k_memory(256 * 4 * 64 * 2), v_memory(k_memory.bytes),
        out_memory(q_memory.bytes), pos_memory(64 * 4);
    q_memory.fill();
    k_memory.fill();
    CUDA_CHECK(cudaDeviceSynchronize());
    TieredContext owner(plan, cache, {runtime_memory.p, runtime_memory.bytes}, device.stream);
    auto bound_ids = std::vector<std::int32_t>(lease.page_ids().begin(), lease.page_ids().end());
    if (!shadow) bound_ids = {bound_ids[5], bound_ids[1], bound_ids[3]};
    owner.bind_pages(bound_ids);
    struct ExpectedPage {
        std::uint32_t logical, valid_tokens;
        std::size_t pool_plane;
        std::vector<std::byte> bytes;
    };
    const auto read_page = [&](std::uint32_t plane, std::uint32_t logical) {
        const auto view = owner.current_view();
        const auto slot = view.blocktable.at(logical);
        require(slot >= 0, "test expected page must be resident");
        const auto id = bound_ids.at(shadow ? logical : std::uint32_t(slot));
        const auto& tensor = cache.pool().plane(plane);
        std::vector<std::byte> bytes(tensor.nb[3]);
        CUDA_CHECK(cudaMemcpy(bytes.data(), static_cast<const std::byte*>(tensor.data) + id * tensor.nb[3],
                              bytes.size(), cudaMemcpyDeviceToHost));
        return bytes;
    };
    const auto expected_pages = [&] {
        std::vector<ExpectedPage> pages;
        const auto view = owner.current_view();
        for (const auto& layer : plan.archive.layers)
            for (const auto& plane : layer)
                for (const auto& page : view.resident)
                    pages.push_back({page.logical_page,
                        std::min(64U, view.frontier - page.logical_page * 64U), plane.pool_plane,
                        read_page(plane.pool_plane, page.logical_page)});
        return pages;
    };
    const auto check_prefix_bytes = [&](const std::vector<ExpectedPage>& pages) {
        for (const auto& page : pages) {
            const auto observed = read_page(page.pool_plane, page.logical);
            const auto& tensor = cache.pool().plane(page.pool_plane);
            for (int head = 0; head < 4; ++head)
                for (std::uint32_t token = 0; token < page.valid_tokens; ++token)
                    for (std::size_t n = 0; n < tensor.nb[1]; ++n) {
                        const auto offset = head * tensor.nb[2] + token * tensor.nb[1] + n;
                        require(observed[offset] == page.bytes[offset],
                                "restored valid KV prefix must match every original layer and plane byte");
                    }
        }
    };
    const auto restore_log = [&](const TieredSnapshot& saved) {
        std::ostringstream log;
        auto* previous = std::clog.rdbuf(log.rdbuf());
        try { owner.restore(saved, device.stream); }
        catch (...) { std::clog.rdbuf(previous); throw; }
        std::clog.rdbuf(previous);
        std::clog << log.str();
        return log.str();
    };
    auto run = [&](std::uint32_t base, std::uint32_t count) {
        const auto width = !shadow && count == 1 ? 4U : count;
        std::vector<std::int32_t> positions(width, 0);
        for (std::uint32_t t = 0; t < count; ++t) positions[t] = base + t;
        upload_on_stream(pos_memory, positions.data(), positions.size() * 4, device.stream);
        owner.begin_block(base, count, device.stream);
        Tensor q(q_memory.p, DType::BF16, {256, 24, int(width)});
        Tensor k(k_memory.p, DType::BF16, {256, 4, int(width)});
        Tensor v(v_memory.p, DType::BF16, {256, 4, int(width)});
        Tensor out(out_memory.p, DType::BF16, {256, 24, int(width)});
        Tensor pos(pos_memory.p, DType::I32, {int(width)});
        for (std::uint32_t layer = 0; layer < 16; ++layer) {
            std::vector<std::uint16_t> input(v.numel());
            for (std::uint32_t t = 0; t < count; ++t)
                std::fill_n(input.begin() + t * 1024, 1024, bf16(value(base + t, layer)));
            // Previous layer's append owns these source bytes until its producer has run.
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            upload_on_stream(v_memory, input.data(), input.size() * 2, device.stream);
            if (shadow) {
                ops::gqa_kv_append(k, v, pos, cache.execution_view(lease).layer_view(layer),
                                   device.stream);
                std::vector<std::uint16_t> expected(out.numel());
                for (std::uint32_t t = 0; t < count; ++t)
                    std::fill_n(expected.begin() + t * 6144, 6144,
                                bf16(float(oracle(base + t, layer))));
                upload_on_stream(out_memory, expected.data(), expected.size() * 2, device.stream);
                owner.shadow_attention(layer, q, pos, 0.0625f, out, device.stream);
                std::vector<std::uint16_t> actual(expected.size());
                out_memory.copy_to_host(actual.data(), actual.size() * 2);
                require(actual == expected, "shadow must preserve the dense output bytes");
            } else {
                owner.attention(layer, q, k, v, pos, 0.0625f, out, device.stream);
                CUDA_CHECK(cudaStreamSynchronize(device.stream));
                std::vector<std::uint16_t> actual(out.numel());
                out_memory.copy_to_host(actual.data(), actual.size() * 2);
                for (std::uint32_t t = 0; t < count; ++t)
                    for (int d = 0; d < 6144; ++d) {
                        const float observed =
                            std::bit_cast<float>(std::uint32_t(actual[t * 6144 + d]) << 16);
                        require(std::abs(observed - oracle(base + t, layer)) < 0.008,
                                "tiered prefill/decode differs from independent uniform-attention "
                                "oracle");
                    }
                for (std::size_t n = std::size_t(count) * 6144; n < actual.size(); ++n)
                    require(actual[n] == 0,
                            "invalid speculative output tail must be exact BF16 zero");
            }
        }
        require(owner.frontier() == base + count,
                "runtime frontier must commit the actual token count");
    };
    run(0, 64);
    run(64, 64);
    run(128, 1);
    const auto checkpoint = owner.capture();
    const auto checkpoint_bytes = expected_pages();
    require(checkpoint.frontier == 129 && checkpoint.archive_frontier == 129,
            "capture must preserve exact boundary before later prefill chunks");
    run(129, 63);
    run(192, 64);
    run(256, 1);
    if (!shadow) {
        const auto partial_checkpoint = owner.capture();
        const auto partial_bytes = expected_pages();
        const auto partial_view = owner.current_view();
        run(257, 1);
        require(restore_log(partial_checkpoint).find("missing_pages=0") != std::string::npos,
                "surviving current partial page restore must schedule zero hydration");
        require(owner.current_view().blocktable == partial_view.blocktable,
                "partial prefix restore must preserve its physical slots");
        check_prefix_bytes(partial_bytes);
    }
    auto wrong_bundle = checkpoint;
    ++wrong_bundle.bundle_identity;
    bool wrong_rejected = false;
    try { owner.restore(wrong_bundle, device.stream); }
    catch (const std::exception&) { wrong_rejected = true; }
    require(wrong_rejected && owner.frontier() == 257, "foreign bundle restore must be rejected before mutation");
    const auto rollback_log = restore_log(checkpoint);
    if (!shadow) {
        std::size_t bytes = 0;
        for (const auto& layer : plan.archive.layers)
            for (const auto& plane : layer) bytes += 2 * plane.page_bytes;
        require(rollback_log.find("missing_pages=2") != std::string::npos &&
                    rollback_log.find("retained_pages=1") != std::string::npos &&
                    rollback_log.find("scheduled_h2d_bytes=" + std::to_string(bytes)) != std::string::npos,
                "rollback must schedule only overwritten pages across all sixteen layers/planes");
    }
    check_prefix_bytes(checkpoint_bytes);
    const auto restored = owner.capture();
    require(restored.view_generation > checkpoint.view_generation &&
                restored.archive_frontier == 129 && owner.current_view().resident.size() == 3,
            "restore must hydrate sink/recent/partial page and advance generation");
    for (std::uint32_t logical = 0; logical < 3; ++logical)
        require(owner.current_view().resident[logical].logical_page == logical,
                "restore must replace later eviction mapping");
    const auto repeated_log = restore_log(checkpoint);
    require(repeated_log.find("missing_pages=0") != std::string::npos &&
                repeated_log.find("scheduled_h2d_bytes=0") != std::string::npos,
            "repeated restore must schedule no archive H2D copies");
    require(owner.capture().view_generation > restored.view_generation,
            "repeated restore must not roll generation backwards");
    run(129, 1);
    run(130, 62);
    run(192, 64);
    run(256, 1); // Both host pages must be streamed into the single-pass decode.
    owner.trim(129, device.stream);
    run(129, 1); // Page 2 was HostOnly: its retained prefix must be restored before A2.
    owner.trim(128, device.stream);
    run(128, 1);
    bool trimmed_rejected = false;
    try { owner.restore(checkpoint, device.stream); }
    catch (const std::exception&) { trimmed_rejected = true; }
    require(trimmed_rejected, "trim below captured frontier must invalidate snapshot even after regrowth");
    owner.reset(device.stream);
    require(owner.frontier() == 0, "reset frontier");
    bool rejected = false;
    try { owner.restore(checkpoint, device.stream); } catch (const std::exception&) { rejected = true; }
    require(rejected, "reset must invalidate snapshots");
    run(0, 4); // Reuse the fixed backing and transfer owner after generation change.
    owner.drain();
    // Whole-page archival includes the unused suffix. A newly assigned logical
    // page must define those bytes without destroying an existing live prefix.
    for (const auto& plane : plan.archive.layers)
        for (const auto& spec : plane) {
            const auto& tensor = cache.pool().plane(spec.pool_plane);
            std::vector<std::byte> page(spec.page_bytes);
            CUDA_CHECK(cudaMemcpy(page.data(),
                                  static_cast<const std::byte*>(tensor.data) +
                                      bound_ids[0] * tensor.nb[3],
                                  page.size(), cudaMemcpyDeviceToHost));
            for (int head = 0; head < 4; ++head)
                for (int token = 4; token < 64; ++token)
                    for (std::size_t n = 0; n < tensor.nb[1]; ++n)
                        require(page[head * tensor.nb[2] + token * tensor.nb[1] + n] == std::byte{},
                                "new logical page suffix must be initialized before whole-page "
                                "archival");
        }
    owner.log_stats();
}

void CUDART_CB delay_compute(void*) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); }

// Pinned H2D jobs can all be submitted while compute is delayed. Every layer's
// D2H job retains a ready-event ticket until its append producer reaches the GPU.
// No readback or stream synchronization is allowed to hide that backlog.
void exercise_queued_writeback_fences() {
    DeviceContext device;
    LayoutBuilder builder;
    DecoderStateSpec spec;
    spec.capacity                  = 320;
    spec.full_attention_layers     = 16;
    spec.kv_heads                  = 4;
    spec.attention_head_dim        = 256;
    spec.text_physical_page_groups = 3;
    spec.allow_tiered_text_pages   = true;
    spec.linear_attention          = {1, 1, 1, 1, 1, 1};
    const auto layout              = plan_decoder_state(builder, spec).text_kv;
    DeviceBuffer pool_memory(builder.finish(256));
    PagedKVCache cache({pool_memory.p, pool_memory.bytes}, layout);
    auto lease = cache.pool().reserve(3);
    lease.materialize_pages(3, device.stream);
    lease.bind_row(0, device.stream);
    TieredKVOptions options;
    options.sink_tokens  = 64;
    options.view_tokens  = 192;
    options.host_archive = HostKVArchiveMode::Pinned;
    const auto plan      = plan_tiered_runtime(layout.pool, 320, 3, 64, options, false, false);
    DeviceBuffer runtime_memory(plan.bytes);
    DeviceBuffer q_memory(256 * 24 * 64 * 2), k_memory(256 * 4 * 64 * 2), v_memory(k_memory.bytes),
        positions_memory(64 * 4), outputs(16ULL * 256 * 24 * 64 * 2);
    q_memory.fill();
    k_memory.fill();
    CUDA_CHECK(cudaDeviceSynchronize());
    TieredContext owner(plan, cache, {runtime_memory.p, runtime_memory.bytes}, device.stream);
    owner.bind_pages(lease.page_ids());
    // Historical values are identical across layers; each block has its own value.
    for (std::uint32_t base = 0; base < 256; base += 64) {
        std::vector<std::uint16_t> values(256 * 4 * 64, bf16(float(base / 64) * 0.25f));
        std::vector<std::int32_t> positions(64);
        for (int t = 0; t < 64; ++t) positions[t] = base + t;
        upload_on_stream(v_memory, values.data(), values.size() * 2, device.stream);
        upload_on_stream(positions_memory, positions.data(), positions.size() * 4, device.stream);
        owner.begin_block(base, 64, device.stream);
        Tensor q(q_memory.p, DType::BF16, {256, 24, 64});
        Tensor k(k_memory.p, DType::BF16, {256, 4, 64});
        Tensor v(v_memory.p, DType::BF16, {256, 4, 64});
        Tensor pos(positions_memory.p, DType::I32, {64});
        Tensor out(outputs.p, DType::BF16, {256, 24, 64});
        for (std::uint32_t layer = 0; layer < 16; ++layer) {
            owner.attention(layer, q, k, v, pos, 0.0625f, out, device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
        }
        owner.drain();
    }
    std::vector<std::uint16_t> fixed_values(256 * 4, bf16(1.0f));
    const std::int32_t position = 256;
    upload_on_stream(v_memory, fixed_values.data(), fixed_values.size() * 2, device.stream);
    upload_on_stream(positions_memory, &position, sizeof(position), device.stream);
    Tensor q(q_memory.p, DType::BF16, {256, 24, 1});
    Tensor k(k_memory.p, DType::BF16, {256, 4, 1});
    Tensor v(v_memory.p, DType::BF16, {256, 4, 1});
    Tensor pos(positions_memory.p, DType::I32, {1});
    owner.begin_block(256, 1, device.stream);
    CUDA_CHECK(cudaLaunchHostFunc(device.stream, delay_compute, nullptr));
    for (std::uint32_t layer = 0; layer < 16; ++layer) {
        auto* data = static_cast<std::uint16_t*>(outputs.p) + layer * 6144;
        Tensor out(data, DType::BF16, {256, 24, 1});
        owner.attention(layer, q, k, v, pos, 0.0625f, out, device.stream);
    }
    // All producer inputs stay allocated and unchanged until every DMA completes.
    owner.drain();
    require(owner.frontier() == 257, "queued writeback stress frontier");
    std::vector<std::uint16_t> actual(16 * 6144);
    outputs.copy_to_host(actual.data(), actual.size() * 2);
    const double expected = (64.0 * (0.0 + 0.25 + 0.5 + 0.75) + 1.0) / 257.0;
    for (auto bits : actual) {
        const auto observed = std::bit_cast<float>(std::uint32_t(bits) << 16);
        require(std::abs(observed - expected) < 0.003,
                "queued sixteen-layer attention differs from independent causal mean");
    }
}
} // namespace

int main(int argc, char** argv) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }
    try {
        exercise_ordered_input_uploads();
        if (argc == 2 && std::strcmp(argv[1], "--input-upload-only") == 0) return 0;
        exercise(false, DType::BF16);
        exercise(false, DType::I8);
        exercise(false, DType::I8, HostKVArchiveMode::Pinned);
        exercise(true, DType::BF16);
        exercise_queued_writeback_fences();
        std::cout << "tiered runtime uniform-attention oracle, trim restore and shadow passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

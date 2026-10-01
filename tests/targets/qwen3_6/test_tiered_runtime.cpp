#include "targets/qwen3_6/impl/runtime/tiered_context.h"
#include "core/device.h"
#include <ninfer/ops/gqa_attention.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <iostream>
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

void exercise(bool shadow, DType dtype) {
    DeviceContext device;
    LayoutBuilder pool_builder;
    DecoderStateSpec spec;
    spec.capacity                  = 320;
    spec.full_attention_layers     = 16;
    spec.kv_heads                  = 4;
    spec.attention_head_dim        = 256;
    spec.kv_dtype                  = dtype;
    spec.kv_quant_group            = dtype == DType::I8 ? 64 : 0;
    spec.text_physical_page_groups = shadow ? 5 : 3;
    spec.allow_tiered_text_pages   = true;
    spec.linear_attention          = {1, 1, 1, 1, 1, 1};
    const auto layout              = plan_decoder_state(pool_builder, spec).text_kv;
    DeviceBuffer pool_memory(pool_builder.finish(256));
    PagedKVCache cache({pool_memory.p, pool_memory.bytes}, layout);
    auto lease = cache.pool().reserve(shadow ? 5 : 3);
    lease.materialize_pages(shadow ? 5 : 3, device.stream);
    lease.bind_row(0, device.stream);
    TieredKVOptions options;
    options.sink_tokens  = 64;
    options.view_tokens  = 192;
    options.host_archive = HostKVArchiveMode::Pageable;
    const auto plan      = plan_tiered_runtime(layout.pool, 320, 3, 64, options, shadow, true);
    DeviceBuffer runtime_memory(plan.bytes);
    DeviceBuffer q_memory(256 * 24 * 64 * 2), k_memory(256 * 4 * 64 * 2), v_memory(k_memory.bytes),
        out_memory(q_memory.bytes), pos_memory(64 * 4);
    q_memory.fill();
    k_memory.fill();
    CUDA_CHECK(cudaDeviceSynchronize());
    TieredContext owner(plan, cache, {runtime_memory.p, runtime_memory.bytes}, device.stream);
    owner.bind_pages(lease.page_ids());
    auto run = [&](std::uint32_t base, std::uint32_t count) {
        const auto width = !shadow && count == 1 ? 4U : count;
        std::vector<std::int32_t> positions(width, 0);
        for (std::uint32_t t = 0; t < count; ++t) positions[t] = base + t;
        pos_memory.copy_from_host(positions.data(), positions.size() * 4);
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
            v_memory.copy_from_host(input.data(), input.size() * 2);
            if (shadow) {
                ops::gqa_kv_append(k, v, pos, cache.execution_view(lease).layer_view(layer),
                                   device.stream);
                std::vector<std::uint16_t> expected(out.numel());
                for (std::uint32_t t = 0; t < count; ++t)
                    std::fill_n(expected.begin() + t * 6144, 6144,
                                bf16(float(oracle(base + t, layer))));
                out_memory.copy_from_host(expected.data(), expected.size() * 2);
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
    for (std::uint32_t base = 0; base < 256; base += 64) run(base, 64);
    run(256, 1); // Both host pages must be streamed into the single-pass decode.
    owner.trim(129, device.stream);
    run(129, 1); // Page 2 was HostOnly: its retained prefix must be restored before A2.
    owner.reset(device.stream);
    require(owner.frontier() == 0, "reset frontier");
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
                                      lease.page_ids()[0] * tensor.nb[3],
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
        v_memory.copy_from_host(values.data(), values.size() * 2);
        positions_memory.copy_from_host(positions.data(), positions.size() * 4);
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
    v_memory.copy_from_host(fixed_values.data(), fixed_values.size() * 2);
    positions_memory.copy_from_host(&position, sizeof(position));
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

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }
    try {
        exercise(false, DType::BF16);
        exercise(false, DType::I8);
        exercise(true, DType::BF16);
        exercise_queued_writeback_fences();
        std::cout << "tiered runtime uniform-attention oracle, trim restore and shadow passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

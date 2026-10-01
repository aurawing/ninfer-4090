#pragma once
#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include <cuda_bf16.h>
#include <cstdint>
#include <math_constants.h>

namespace ninfer::ops {
template <bool Carry = false, bool Finalize = false>
__global__ void
attention_partial_lse_merge_kernel(const float* o, const float* m, const float* l, int rows,
                                   int parts, float* result_o, float* result_m, float* result_l,
                                   bool reset = true, __nv_bfloat16* final_output = nullptr) {
    const int row = int(blockIdx.x), d = int(threadIdx.x);
    extern __shared__ float weights[];
    if (!d) {
        const float prior_l = Carry && !reset ? result_l[row] : 0.0f;
        const float prior_m = prior_l > 0 ? result_m[row] : -CUDART_INF_F;
        float maximum       = prior_m;
        for (int part = 0; part < parts; ++part) {
            const std::int64_t index = row + std::int64_t(rows) * part;
            if (l[index] > 0) maximum = fmaxf(maximum, m[index]);
        }
        const float prior_weight = prior_l > 0 ? expf(prior_m - maximum) : 0.0f;
        if constexpr (Carry) weights[parts] = prior_weight;
        // The maximum and prior denominator are shared before result_l is
        // overwritten. Exponentials are independent; the sum still follows
        // ascending split order in one thread, preserving deterministic math.
        weights[parts + 1] = maximum;
        weights[parts + 2] = prior_l;
        result_m[row]      = maximum;
    }
    __syncthreads();
    for (int part = d; part < parts; part += blockDim.x) {
        const std::int64_t index = row + std::int64_t(rows) * part;
        weights[part]            = l[index] > 0 ? expf(m[index] - weights[parts + 1]) : 0.0f;
    }
    __syncthreads();
    if (!d) {
        float sum = 0.0f;
        if constexpr (Carry) sum = weights[parts + 2] * weights[parts];
        for (int part = 0; part < parts; ++part) {
            const std::int64_t index = row + std::int64_t(rows) * part;
            sum += l[index] > 0 ? l[index] * weights[part] : 0.0f;
        }
        result_l[row] = sum;
        if constexpr (Finalize) weights[parts + 1] = sum;
    }
    __syncthreads();
    float value = 0;
    if constexpr (Carry)
        if (weights[parts] > 0) value = result_o[d + std::int64_t(256) * row] * weights[parts];
    // Ascending part order is fixed; no atomics and no arrival-order reduction.
    for (int part = 0; part < parts; ++part) {
        if (weights[part] > 0)
            value += o[d + std::int64_t(256) * (row + std::int64_t(rows) * part)] * weights[part];
    }
    result_o[d + std::int64_t(256) * row] = value;
    if constexpr (Finalize) {
        const float denominator = weights[parts + 1];
        final_output[d + std::int64_t(256) * row] =
            __float2bfloat16(denominator > 0 ? value / denominator : 0.0f);
    }
}

__global__ void attention_partial_finalize_kernel(const float* o, const float* l,
                                                  std::int64_t elements, __nv_bfloat16* result) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < elements) {
        const float denominator = l[i / 256];
        result[i]               = __float2bfloat16(denominator > 0 ? o[i] / denominator : 0.0f);
    }
}

// One row per 128-thread CTA, one complete warp per 64-coordinate group.
// Each lane owns coordinates lane and lane+32. All lanes execute both H64
// calls with a full mask; no partial-warp synchronization or shared scratch.
__global__ void attention_partial_finalize_rotated_kernel(const float* o, const float* l,
                                                          __nv_bfloat16* result) {
    constexpr unsigned FullMask = 0xffffffffu;
    const int row = int(blockIdx.x), group = int(threadIdx.x) / 32,
              lane = int(threadIdx.x) & 31;
    const std::int64_t i = std::int64_t(row) * 256 + group * 64 + lane;
    const float denominator = l[row];
    // Ignore even NaN O payloads for neutral rows, as the standard finalizer does.
    float x0 = denominator > 0 ? o[i] / denominator : 0.0f;
    float x1 = denominator > 0 ? o[i + 32] / denominator : 0.0f;
    gqa_kv_hadamard64(x0, x1, FullMask);
    x0 = __bfloat162float(__float2bfloat16_rn(x0));
    x1 = __bfloat162float(__float2bfloat16_rn(x1));
    gqa_kv_hadamard64(x0, x1, FullMask);
    result[i] = __float2bfloat16_rn(x0);
    result[i + 32] = __float2bfloat16_rn(x1);
}
} // namespace ninfer::ops

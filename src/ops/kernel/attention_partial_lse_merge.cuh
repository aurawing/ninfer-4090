#pragma once
#include <cuda_bf16.h>
#include <cstdint>
#include <math_constants.h>

namespace ninfer::ops {
__global__ void attention_partial_lse_merge_kernel(const float* o, const float* m, const float* l,
                                                   int rows, int parts, float* result_o,
                                                   float* result_m, float* result_l) {
    const int row = int(blockIdx.x), d = int(threadIdx.x);
    extern __shared__ float weights[];
    if (!d) {
        float maximum = -CUDART_INF_F;
        for (int part = 0; part < parts; ++part) {
            const std::int64_t index = row + std::int64_t(rows) * part;
            if (l[index] > 0)
                maximum = fmaxf(maximum, m[index]);
        }
        float sum = 0;
        for (int part = 0; part < parts; ++part) {
            const std::int64_t index = row + std::int64_t(rows) * part;
            const float weight = l[index] > 0 ? expf(m[index] - maximum) : 0.0f;
            weights[part] = weight;
            sum += l[index] > 0 ? l[index] * weight : 0.0f;
        }
        result_m[row] = maximum;
        result_l[row] = sum;
    }
    __syncthreads();
    float value = 0;
    // Ascending part order is fixed; no atomics and no arrival-order reduction.
    for (int part = 0; part < parts; ++part) {
        if (weights[part] > 0)
            value += o[d + std::int64_t(256) * (row + std::int64_t(rows) * part)] * weights[part];
    }
    result_o[d + std::int64_t(256) * row] = value;
}
__global__ void attention_partial_finalize_kernel(const float* o, const float* l,
                                                  std::int64_t elements, __nv_bfloat16* result) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < elements) {
        const float denominator = l[i / 256];
        result[i] = __float2bfloat16(denominator > 0 ? o[i] / denominator : 0.0f);
    }
}
} // namespace ninfer::ops

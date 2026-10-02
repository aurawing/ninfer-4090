#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cstddef>
namespace ninfer::ops::detail {
__global__ void score_rows(const __nv_bfloat16* q, const __half* means, int d, int h, int kv, int m,
                           int p, int first, int end, bool reset, float* logits, float* stats,
                           int* invalid) {
    int row = blockIdx.x, head = row % h, token = row / h, lane = threadIdx.x;
    __shared__ float reduction[256];
    __shared__ int bad[256];
    float maximum = -INFINITY;
    int error = 0;
    for (std::size_t page = lane; page < std::size_t(p); page += 256) {
        float s = 0;
        const auto* a = q + std::size_t(d) * (head + h * token);
        const auto* b = means + std::size_t(d) * (head / (h / kv) + kv * page);
#pragma unroll 2
        for (int i = 0; i < d; ++i) {
            float x = __bfloat162float(a[i]), y = __half2float(b[i]);
            error |= !isfinite(x) || !isfinite(y);
            s = fmaf(x, y, s);
        }
        s /= sqrtf(float(d));
        error |= !isfinite(s);
        logits[page + std::size_t(p) * row] = s;
        if (page >= std::size_t(first) && page < std::size_t(end))
            maximum = fmaxf(maximum, s);
    }
    reduction[lane] = maximum;
    bad[lane] = error;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (lane < stride) {
            reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
            bad[lane] |= bad[lane + stride];
        }
        __syncthreads();
    }
    maximum = reduction[0];
    error = bad[0];
    // Every warp must read the maximum before reduction[] becomes sum scratch.
    __syncthreads();
    float sum = 0;
    if (!error)
        for (std::size_t page = std::size_t(first) + lane; page < std::size_t(end); page += 256)
            sum += expf(logits[page + std::size_t(p) * row] - maximum);
    reduction[lane] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (lane < stride)
            reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    if (!lane) {
        stats[2 * row] = maximum;
        stats[2 * row + 1] = reduction[0];
        invalid[row] =
            error || !isfinite(reduction[0]) || reduction[0] <= 0 || (!reset && invalid[row]);
    }
}
__global__ void score_pages(int h, int m, int p, int first, int end, int layers, bool reset,
                            const float* logits, const float* stats, const int* rows, float* scores,
                            int* status) {
    std::size_t page = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (page >= std::size_t(p))
        return;
    int invalid = 0;
    for (int row = 0; row < h * m; ++row)
        invalid |= rows[row];
    // Row status is sticky across ordered layers. Only page zero writes the summary;
    // other pages never read that word, avoiding a read/write race within this pass.
    if (!page)
        *status = invalid;
    float s = reset ? 0 : scores[page];
    if (invalid)
        s = 0;
    else if (page >= std::size_t(first) && page < std::size_t(end)) {
        for (int head = 0; head < h; ++head)
            for (int token = 0; token < m; ++token) {
                int row = head + h * token;
                s += expf(logits[page + std::size_t(p) * row] - stats[2 * row]) /
                     stats[2 * row + 1] / float(layers * h);
            }
    }
    scores[page] = s;
}
} // namespace ninfer::ops::detail

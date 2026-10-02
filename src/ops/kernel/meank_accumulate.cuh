#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops::detail {
// One thread per coordinate, one CTA per page/coordinate tile. No reduction,
// atomics, reassociation or inter-page seed mutation. Token ordering is fixed.
__global__ void meank_accumulate_kernel(const __nv_bfloat16* k, const float* seed, int rows,
                                        int offset, int tokens, int pages, __half* means,
                                        float* tail, int* count) {
    int r = int(blockIdx.x) * blockDim.x + threadIdx.x, p = int(blockIdx.y);
    if (r >= rows)
        return;
    if (!pages) {
        tail[r] = 0.f;
        if (!r)
            *count = 0;
        return;
    }
    float sum = p == 0 && offset ? seed[r] : 0.f;
    int lo = max(0, p * 64 - offset), hi = min(tokens, (p + 1) * 64 - offset);
#pragma unroll 2
    for (int t = lo; t < hi; ++t)
        sum = __fadd_rn(sum, __bfloat162float(k[std::size_t(t) * rows + r]));
    int n = hi - lo + (p == 0 ? offset : 0);
    means[std::size_t(p) * rows + r] = __float2half_rn(__fdiv_rn(sum, float(n)));
    if (p == pages - 1) {
        int final_count = (offset + tokens) % 64;
        tail[r] = final_count ? sum : 0.f;
        if (!r)
            *count = final_count;
    }
}
__global__ void meank_retain_tail_kernel(const __nv_bfloat16* k, int rows, int first, int tokens,
                                         __nv_bfloat16* tail) {
    int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    int final_count = (first + tokens) % 64;
    int lo = max(0, tokens - final_count), n = tokens - lo;
    if (i >= rows * n)
        return;
    int t = i / rows, r = i % rows, slot = (first + lo + t) % 64;
    tail[std::size_t(slot) * rows + r] = k[std::size_t(lo + t) * rows + r];
}
} // namespace ninfer::ops::detail

#pragma once

#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"

namespace ninfer::ops {
struct TieredKVPlanes {
    const void* k;
    const void* v;
    const __half* ks;
    const __half* vs;
};
struct TieredKey {
    int absolute, physical;
};

// Prefix is built at frontier F; each query tile clips it further at q_max+1.
__device__ __forceinline__ int tiered_visible_keys(const int2* pages, const int* prefix, int count,
                                                   int limit) {
    int lo = 0, hi = count;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (pages[mid].x * 64 < limit)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (!lo)
        return 0;
    const int last = lo - 1;
    return prefix[last] + min(prefix[last + 1] - prefix[last], limit - pages[last].x * 64);
}
__device__ __forceinline__ TieredKey tiered_key(const int2* pages, const int* prefix, int,
                                                int ordinal) {
    // attention_access_prefix() includes whole pages; only the final frontier
    // page can be short. Compact key ordinal thus selects a list entry in O(1).
    // Position still comes from its ORIGINAL logical ID, never the list index.
    const int index = ordinal / 64;
    const int2 page = pages[index];
    return {page.x * 64 + ordinal - prefix[index], page.y};
}
__device__ __forceinline__ void tiered_i8_store_swz(std::int8_t* tile, int row, int d,
                                                    std::int8_t code) {
    tile[(row * 128 + gqa_small_t_tc_swz(row, d / 2)) * 2 + (d & 1)] = code;
}
template <bool Prefill> __device__ __forceinline__ auto tiered_probability(float value) {
    if constexpr (Prefill)
        return __float2half_rn(value);
    else
        return __float2bfloat16(value);
}
template <bool Prefill>
__device__ __forceinline__ int4 tiered_dequant_v(const std::int8_t* codes, float scale) {
    if constexpr (!Prefill)
        return gqa_kv_dequant_i8x8_from(codes, scale);
    else {
        const int2 raw = load_vec<int2>(codes);
        const auto* values = reinterpret_cast<const std::int8_t*>(&raw);
        unsigned packed[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __half2 pair =
                __floats2half2_rn(float(values[2 * i]) * scale, float(values[2 * i + 1]) * scale);
            packed[i] = *reinterpret_cast<const unsigned*>(&pair);
        }
        return make_int4(int(packed[0]), int(packed[1]), int(packed[2]), int(packed[3]));
    }
}
template <bool Prefill>
__device__ __forceinline__ void tiered_pv_mma(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    if constexpr (Prefill)
        mma_f16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    else
        mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
}
__global__ void tiered_inverse_rotate_fp32(float* output, int rows) {
    const int unit = int(blockIdx.x), lane = int(threadIdx.x);
    if (unit >= rows * 4)
        return;
    const int row = unit / 4, group = unit % 4;
    const std::int64_t base = std::int64_t(row) * 256 + group * 64;
    float x0 = output[base + lane], x1 = output[base + lane + 32];
    gqa_kv_hadamard64(x0, x1);
    output[base + lane] = x0;
    output[base + lane + 32] = x1;
}
} // namespace ninfer::ops

#pragma once

#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"

namespace ninfer::ops {
// Each four-lane quad owns one row's aligned 64-dimension MMA groups: bit 0
// is the fragment component, bits 1/2 are the lane, bits 3..5 the n tile.
// Ascending butterfly bits exactly match gqa_kv_hadamard64's FP32 order.
template <int Fragments>
__device__ __forceinline__ void tiered_inverse_rotate_fragment(float (&acc)[Fragments][4],
                                                               int lid) {
    static_assert(Fragments % 8 == 0);
#pragma unroll
    for (int n = 0; n < Fragments; ++n) {
#pragma unroll
        for (int row = 0; row < 2; ++row) {
            const float a = acc[n][2 * row], b = acc[n][2 * row + 1];
            acc[n][2 * row]     = a + b;
            acc[n][2 * row + 1] = a - b;
        }
    }
#pragma unroll
    for (int offset = 1; offset <= 2; offset <<= 1) {
#pragma unroll
        for (int n = 0; n < Fragments; ++n) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float a = acc[n][c];
                const float b = __shfl_xor_sync(0xffffffffu, a, offset);
                acc[n][c]     = lid & offset ? b - a : a + b;
            }
        }
    }
#pragma unroll
    for (int offset = 1; offset <= 4; offset <<= 1) {
#pragma unroll
        for (int n = 0; n < Fragments; ++n) {
            if (!(n & offset)) {
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const float a = acc[n][c], b = acc[n + offset][c];
                    acc[n][c]          = a + b;
                    acc[n + offset][c] = a - b;
                }
            }
        }
    }
#pragma unroll
    for (int n = 0; n < Fragments; ++n)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[n][c] *= 0.125f;
}

// Expand signed nibbles in registers and issue one aligned 128-bit shared
// store. Each byte's sign contribution is at most 240, so the packed multiply
// cannot carry into an adjacent byte. Dense codec helpers remain unchanged.
__device__ __forceinline__ void tiered_unpack_i4x16(const std::uint8_t* source,
                                                    std::int8_t* destination) {
    const auto raw = load_vec<std::uint64_t>(source);
    auto expand    = [](unsigned word) {
        unsigned lo = word & 0x0f0f0f0fu;
        unsigned hi = (word >> 4) & 0x0f0f0f0fu;
        lo |= (lo & 0x08080808u) * 30u;
        hi |= (hi & 0x08080808u) * 30u;
        return make_uint2(__byte_perm(lo, hi, 0x5140), __byte_perm(lo, hi, 0x7362));
    };
    const auto a = expand(unsigned(raw)), b = expand(unsigned(raw >> 32));
    store_vec(destination, make_int4(int(a.x), int(a.y), int(b.x), int(b.y)));
}

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
    if (!lo) return 0;
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

template <bool Prefill>
__device__ __forceinline__ auto tiered_probability(float value) {
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
        const int2 raw           = load_vec<int2>(codes);
        const auto* values       = reinterpret_cast<const std::int8_t*>(&raw);
        const __half half_scale  = __float2half_rn(scale);
        const __half2 scale_pair = __halves2half2(half_scale, half_scale);
        unsigned packed[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __half2 pair = __hmul2(
                __floats2half2_rn(float(values[2 * i]), float(values[2 * i + 1])), scale_pair);
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
} // namespace ninfer::ops

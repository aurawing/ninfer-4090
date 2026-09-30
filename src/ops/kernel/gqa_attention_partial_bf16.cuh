#pragma once
// Tiered read-only BF16 Tensor Core attention; sparse logical/physical access
// with FP32 split-local numerators and explicit frontier-based key counts.
#include "ops/kernel/gqa_attention_partial_common.cuh"
namespace ninfer::ops {
template <typename Geometry, int TokenTile, int WarpsPerCta>
__launch_bounds__(128, 2) __global__
    void gqa_attention_partial_bf16_kernel(const __nv_bfloat16* q, const std::int32_t* pos,
                                           TieredKVPlanes resident, TieredKVPlanes staging,
                                           int resident_pages, const int2* pages, const int* prefix,
                                           int page_count, int frontier, int full_width,
                                           float scale, float* partial_acc, float* partial_m,
                                           float* partial_l) {
    static_assert(TokenTile >= 1 && TokenTile <= 10);
    static_assert(WarpsPerCta >= 1 && WarpsPerCta <= 4);

    constexpr int Wc = WarpsPerCta;
    constexpr int Br = Wc * 16;
    constexpr int Bc = 32;
    constexpr int D = kGqaHeadDim;
    constexpr int Threads = Wc * 32;
    constexpr int QKNt = Bc / 8;
    constexpr int QKKs = D / 16;
    constexpr int PVNt = D / 8;
    constexpr int PVKs = Bc / 16;
    constexpr float Log2E = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    constexpr int QkvRows = 2 * Bc;

    static_assert(QkvRows >= Br);

    __shared__ __align__(16) __nv_bfloat16 qkv_s[QkvRows * D];
    __shared__ __align__(16) __nv_bfloat16 p_s[Wc * 16 * Bc];
    __shared__ __align__(16) __nv_bfloat16 p_lo[Wc * 16 * Bc];
    __nv_bfloat16* k_s = qkv_s;
    __nv_bfloat16* v_s = qkv_s + Bc * D;

    const int kv_head = int(blockIdx.x), split = int(blockIdx.y), split_count = int(gridDim.y);
    const int tid = int(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int column_begin = int(blockIdx.z) * TokenTile;
    const int valid_tokens = min(TokenTile, full_width - column_begin);
    const int tokens = valid_tokens;
    const int row_count = tokens * Geometry::GroupSize;
    const int RowCount = row_count;
    q += std::int64_t(256) * Geometry::QHeads * column_begin;
    pos += column_begin;
    partial_acc += std::int64_t(256) * Geometry::QHeads * column_begin;
    partial_m += Geometry::QHeads * column_begin;
    partial_l += Geometry::QHeads * column_begin;
    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, full_width)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, full_width)] =
                    0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d = idx - row * D;
            int q_head = 0;
            int token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, full_width)] =
                    0.0f;
            }
        }
    };

    const int limit = min(frontier, pos[valid_tokens - 1] + 1);
    const int visible = tiered_visible_keys(pages, prefix, page_count, limit);
    const int split_start = int(std::int64_t(visible) * split / split_count);
    const int split_end = int(std::int64_t(visible) * (split + 1) / split_count);
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    for (int idx = tid; idx < Br * D; idx += Threads) {
        const int row = idx / D;
        const int d = idx - row * D;
        int q_head = 0;
        int token = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (row < row_count && gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            value = q[gqa_q_index<Geometry>(q_head, d, token)];
        }
        qkv_s[row * D + gqa_small_t_tc_swz(row, d)] = value;
    }
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat = lane >> 3;
    const int a_rin = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;

    const int warp_row0 = warp * 16;
    __nv_bfloat16* p_sw = &p_s[warp * 16 * Bc];

    unsigned af_q[QKKs][4];
#pragma unroll
    for (int k = 0; k < QKKs; ++k) {
        const int arow = warp_row0 + a_rowoff;
        const int acol = k * 16 + a_coloff;
        ldmatrix_x4(af_q[k][0], af_q[k][1], af_q[k][2], af_q[k][3],
                    smem_addr(&qkv_s[arow * D + gqa_small_t_tc_swz(arow, acol)]));
    }
    __syncthreads();
    __shared__ int key_absolute[Bc], key_physical[Bc];
    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[n][i] = 0.0f;
        }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;

        for (int j = tid; j < Bc; j += Threads) {
            const int ordinal = k0 + j;
            if (ordinal >= split_start && ordinal < split_end) {
                const auto key = tiered_key(pages, prefix, page_count, ordinal);
                key_absolute[j] = key.absolute;
                key_physical[j] = key.physical;
            } else {
                key_absolute[j] = 0x7fffffff;
                key_physical[j] = -1;
            }
        }
        __syncthreads();
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
            const int j = chunk / (D / 8), d = (chunk % (D / 8)) * 8;
            auto* kd = &k_s[j * D + gqa_small_t_tc_swz(j, d)];
            auto* vd = &v_s[j * D + gqa_small_t_tc_swz(j, d)];
            if (key_physical[j] >= 0) {
                const bool stage = key_physical[j] >= resident_pages;
                const auto source = stage ? staging : resident;
                const int page = key_physical[j] - (stage ? resident_pages : 0);
                const auto off = gqa_cache_index<Geometry>(page, kv_head, d, key_absolute[j] & 63);
                cp_async<16>(kd, static_cast<const __nv_bfloat16*>(source.k) + off);
                cp_async<16>(vd, static_cast<const __nv_bfloat16*>(source.v) + off);
            } else {
                store_vec(kd, make_int4(0, 0, 0, 0));
                store_vec(vd, make_int4(0, 0, 0, 0));
            }
        }
        ninfer::ops::cp_commit();
        ninfer::ops::cp_wait<0>();
        __syncthreads();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned bf[2];
                const int brow = nt * 8 + b_rin;
                const int bcol = k * 16 + b_koff;
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(&k_s[brow * D + gqa_small_t_tc_swz(brow, bcol)]));
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af_q[k][0],
                         af_q[k][1], af_q[k][2], af_q[k][3], bf[0], bf[1]);
            }
        }

        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head0, token0);
        gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head1, token1);
        const int qabs0 = (row0 < row_count) ? pos[token0] : -1;
        const int qabs1 = (row1 < row_count) ? pos[token1] : -1;

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0 = nt * 8 + 2 * lid;
            const int col1 = col0 + 1;
            const int key0 = key_absolute[col0];
            const int key1 = key_absolute[col1];
            score[nt][0] = (row0 < row_count && k0 + col0 >= split_start && k0 + col0 < split_end &&
                            key0 <= qabs0)
                               ? score[nt][0] * scale
                               : -CUDART_INF_F;
            score[nt][1] = (row0 < row_count && k0 + col1 >= split_start && k0 + col1 < split_end &&
                            key1 <= qabs0)
                               ? score[nt][1] * scale
                               : -CUDART_INF_F;
            score[nt][2] = (row1 < row_count && k0 + col0 >= split_start && k0 + col0 < split_end &&
                            key0 <= qabs1)
                               ? score[nt][2] * scale
                               : -CUDART_INF_F;
            score[nt][3] = (row1 < row_count && k0 + col1 >= split_start && k0 + col1 < split_end &&
                            key1 <= qabs1)
                               ? score[nt][3] * scale
                               : -CUDART_INF_F;
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0 = fmaxf(m0, bm0);
        const float nm1 = fmaxf(m1, bm1);
        const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
        const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);

        float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0 = nt * 8 + 2 * lid;
            const int col1 = col0 + 1;
            const float p00 = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][0] - nm0) * Log2E)
                                  : 0.0f;
            const float p01 = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][1] - nm0) * Log2E)
                                  : 0.0f;
            const float p10 = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][2] - nm1) * Log2E)
                                  : 0.0f;
            const float p11 = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][3] - nm1) * Log2E)
                                  : 0.0f;
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col0)] = __float2bfloat16(p00);
            p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col1)] = __float2bfloat16(p01);
            p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col0)] = __float2bfloat16(p10);
            p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col1)] = __float2bfloat16(p11);
            __nv_bfloat16* lo = &p_lo[warp * 16 * Bc];
            const int indices[]{gid * Bc + gqa_small_t_tc_swz32(gid, col0),
                                gid * Bc + gqa_small_t_tc_swz32(gid, col1),
                                (gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col0),
                                (gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col1)};
            const float values[]{p00, p01, p10, p11};
#pragma unroll
            for (int i = 0; i < 4; ++i)
                lo[indices[i]] = __float2bfloat16(values[i] - __bfloat162float(p_sw[indices[i]]));
        }
        bl0 = warp_sum<4>(bl0, FullMask);
        bl1 = warp_sum<4>(bl1, FullMask);

        l0 = l0 * alpha0 + bl0;
        l1 = l1 * alpha1 + bl1;
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }
        __syncwarp();

#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                            smem_addr(&p_sw[a_rowoff * Bc + gqa_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_s[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                         vf[0], vf[1]);
                ldmatrix_x4(
                    pf[0], pf[1], pf[2], pf[3],
                    smem_addr(
                        &p_lo[(warp * 16 + a_rowoff) * Bc + gqa_small_t_tc_swz32(a_rowoff, pcol)]));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                         vf[0], vf[1]);
            }
        }
        __syncthreads();
    }

    if (lid == 0) {
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            int q_head = 0;
            int token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, full_width)] = m0;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, full_width)] = l0;
        }
        if (row1 < row_count) {
            int q_head = 0;
            int token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, full_width)] = m1;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, full_width)] = l1;
        }
    }

// Preserve native FP32 accumulators. No split-local BF16 rounding.
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0 = n * 8 + 2 * lid, row0 = warp_row0 + gid, row1 = row0 + 8;
        if (row0 < row_count) {
            int q_head = 0, token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head, token);
            const auto off = gqa_partial_acc_index<Geometry>(q_head, d0, token, split, full_width);
            *reinterpret_cast<float2*>(partial_acc + off) = make_float2(acc[n][0], acc[n][1]);
        }
        if (row1 < row_count) {
            int q_head = 0, token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head, token);
            const auto off = gqa_partial_acc_index<Geometry>(q_head, d0, token, split, full_width);
            *reinterpret_cast<float2*>(partial_acc + off) = make_float2(acc[n][2], acc[n][3]);
        }
    }
}
} // namespace ninfer::ops

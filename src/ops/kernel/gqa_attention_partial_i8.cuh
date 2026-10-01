#pragma once
// Tiered read-only Q8 Tensor Core attention. Native decode arithmetic with
// explicit sparse page access, frontier-based key prefixes and FP32 partials.
#include "ops/kernel/gqa_attention_partial_common.cuh"
#include <type_traits>

namespace ninfer::ops {
template <typename Geometry, int TokenTile, int WarpsPerCta, int KeyBlock, bool DynamicArena,
          bool Packed, bool Prefill>
__launch_bounds__(WarpsPerCta * 32, (Prefill || TokenTile >= 8) ? 1 : 2) __global__
    void gqa_attention_partial_i8_kernel(const __nv_bfloat16* q, const std::int32_t* pos,
                                         TieredKVPlanes resident, TieredKVPlanes staging,
                                         int resident_pages, const int2* pages, const int* prefix,
                                         int page_count, int frontier, int full_width, float scale,
                                         float* partial_acc, float* partial_m, float* partial_l) {
    using Narrow                       = std::conditional_t<Prefill, __half, __nv_bfloat16>;
    constexpr int Wc                   = WarpsPerCta;
    constexpr int RowCapacity          = Prefill ? TokenTile : TokenTile * Geometry::GroupSize;
    constexpr int RowTiles             = (RowCapacity + 15) / 16;
    constexpr int Br                   = RowTiles * 16;
    constexpr int Bc                   = KeyBlock;
    constexpr int D                    = kGqaHeadDim;
    constexpr int DB16                 = D / 2;
    constexpr int Threads              = Wc * 32;
    constexpr int Groups               = kGqaKvQuantGroups;
    constexpr int GroupKc              = kGqaKvQuantGroup / 32;
    constexpr int QKKs                 = D / 32;
    constexpr int QKNt                 = Bc / 8;
    constexpr int ConsumerWarpsPerTile = Wc / RowTiles;
    constexpr int PVNtPerWarp          = D / (ConsumerWarpsPerTile * 8);
    constexpr int PVKs                 = Bc / 16;
    constexpr int ProducerThreads      = RowTiles * 32;
    constexpr int VLoaderThreads       = Threads - ProducerThreads;
    constexpr float Log2E              = 1.4426950408889634074f;
    constexpr unsigned FullMask        = 0xffffffffu;

    static_assert(TokenTile >= 1 && TokenTile <= (Prefill ? 64 : 10));
    static_assert(Bc == 32 || Bc == 64);
    static_assert(RowTiles >= 1 && RowTiles <= 4);
    static_assert(Wc % RowTiles == 0);
    static_assert(PVNtPerWarp == 2 || PVNtPerWarp == 4 || PVNtPerWarp == 8 || PVNtPerWarp == 16);
    static_assert(QKKs == Groups * GroupKc);

    // Keep Q in a compact dedicated tile so the producer can reload one
    // 64-dimension group at a time instead of carrying all eight fragments in
    // registers across the whole kernel. The main arena holds K i8, V i8, and
    // V bf16 during the key loop.
    __shared__ __align__(16) std::int8_t q_s[Br * D];
    __shared__ __align__(16) std::int8_t static_r_s[DynamicArena ? 16 : 4 * Bc * D];
    extern __shared__ __align__(16) std::int8_t dynamic_r_s[];
    std::int8_t* r_s     = DynamicArena ? dynamic_r_s : static_r_s;
    std::int8_t* q_i8    = q_s;
    std::int8_t* k_i8    = r_s;
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);
    __nv_bfloat16* k_b16 = reinterpret_cast<__nv_bfloat16*>(k_i8);
    std::int8_t* v_i8    = r_s + Bc * D;
    Narrow* v_bf16       = reinterpret_cast<Narrow*>(r_s + 2 * Bc * D);
    __shared__ __align__(16) Narrow p_s[Br * Bc];
    __shared__ __align__(16) __nv_bfloat16 p_lo[Prefill ? 8 : Br * Bc];
    // Reload scales by group to bound live registers in the QK/PV overlap.
    __shared__ volatile float q_scale_tmp[Br * Groups];
    __shared__ float alpha_s[Br];
    __shared__ __align__(16) __half k_scale_s[Bc * Groups];
    __shared__ __align__(16) __half v_scale_s[Bc * Groups];

    const int query_head  = Prefill ? int(blockIdx.y) : 0;
    const int kv_head     = Prefill ? query_head / Geometry::GroupSize : int(blockIdx.x);
    const int split       = Prefill ? int(blockIdx.z) : int(blockIdx.y);
    const int split_count = Prefill ? int(gridDim.z) : int(gridDim.y);
    const int tid = int(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int column_begin = (Prefill ? int(blockIdx.x) : int(blockIdx.z)) * TokenTile;
    const int valid_tokens = min(TokenTile, full_width - column_begin);
    const int tokens       = valid_tokens;
    const int row_count    = Prefill ? tokens : tokens * Geometry::GroupSize;
    const int RowCount     = row_count;
    q += std::int64_t(256) * Geometry::QHeads * column_begin;
    pos += column_begin;
    partial_acc += std::int64_t(256) * Geometry::QHeads * column_begin;
    partial_m += Geometry::QHeads * column_begin;
    partial_l += Geometry::QHeads * column_begin;
    auto row_to_qt = [&](int row, int& head, int& token) {
        if constexpr (Prefill) {
            head  = query_head;
            token = row;
        } else {
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, head, token);
        }
    };
    auto probability_swizzle = [&](int row, int column) {
        if constexpr (Bc == 64)
            return gqa_small_t_tc_swz(row, column);
        else
            return gqa_small_t_tc_swz32(row, column);
    };
    auto write_neutral = [&]() {
        for (int row = tid; row < RowCount; row += Threads) {
            int q_head = 0;
            int token  = 0;
            row_to_qt(row, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, full_width)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, full_width)] =
                    0.0f;
            }
        }
        for (int idx = tid; idx < RowCount * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            row_to_qt(row, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, full_width)] =
                    0.0f;
            }
        }
    };

    const int limit       = min(frontier, pos[valid_tokens - 1] + 1);
    const int visible     = tiered_visible_keys(pages, prefix, page_count, limit);
    const int split_start = int(std::int64_t(visible) * split / split_count);
    const int split_end   = int(std::int64_t(visible) * (split + 1) / split_count);
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);

    auto issue_kv_tile = [&]<bool FullLoad>(int tile_k0) {
        // A 32/64-key aligned tile never crosses a compact list page. Only the
        // frontier page may be short; split edges below still mask its padding.
        const int2 mapping    = pages[tile_k0 / 64];
        const bool is_staging = mapping.y >= resident_pages;
        const auto source     = is_staging ? staging : resident;
        const int page        = mapping.y - (is_staging ? resident_pages : 0);
        const int page_offset = tile_k0 & 63;
        for (int j = tid; j < Bc; j += Threads) {
            if (FullLoad || (tile_k0 + j >= split_start && tile_k0 + j < split_end)) {
                const auto off =
                    gqa_kv_quant_scale_index<Geometry>(page, kv_head, 0, page_offset + j);
                cp_async<8>(&k_scale_s[j * Groups], source.ks + off);
                cp_async<8>(&v_scale_s[j * Groups], source.vs + off);
            } else {
                store_vec(k_scale_s + j * Groups, make_int2(0, 0));
                store_vec(v_scale_s + j * Groups, make_int2(0, 0));
            }
        }
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
            const int j = chunk / (D / 16), d = (chunk % (D / 16)) * 16;
            auto* kd = k_i8 + j * D + gqa_small_t_tc_swz(j, d / 2) * 2;
            auto* vd = v_i8 + j * D + d;
            if (FullLoad || (tile_k0 + j >= split_start && tile_k0 + j < split_end)) {
                const int offset = page_offset + j;
                if constexpr (Packed) {
                    const auto off = gqa_kv_i4_code_index<Geometry>(page, kv_head, d / 2, offset);
                    tiered_unpack_i4x16(static_cast<const std::uint8_t*>(source.k) + off, kd);
                    tiered_unpack_i4x16(static_cast<const std::uint8_t*>(source.v) + off, vd);
                } else {
                    const auto off = gqa_kv_quant_code_index<Geometry>(page, kv_head, d, offset);
                    cp_async<16, Prefill ? Cache::cg : Cache::ca>(
                        kd, static_cast<const std::int8_t*>(source.k) + off);
                    cp_async<16, Prefill ? Cache::cg : Cache::ca>(
                        vd, static_cast<const std::int8_t*>(source.v) + off);
                }
            } else {
                store_vec(kd, make_int4(0, 0, 0, 0));
                store_vec(vd, make_int4(0, 0, 0, 0));
            }
        }
        cp_commit();
    };
    auto load_tile = [&](int tile_k0) {
        if (tile_k0 >= split_start && tile_k0 + Bc <= split_end)
            issue_kv_tile.template operator()<true>(tile_k0);
        else
            issue_kv_tile.template operator()<false>(tile_k0);
    };
    load_tile(first_tile);

    // One warp owns every complete (row,group), including zero padding. No
    // cross-warp clear/write race; scale storage never aliases probabilities.
    for (int unit = warp; unit < Br * Groups; unit += Wc) {
        const int row = unit / Groups, grp = unit % Groups, d0 = grp * 64 + lane, d1 = d0 + 32;
        if (row >= RowCount) {
            tiered_i8_store_swz(q_i8, row, d0, 0);
            tiered_i8_store_swz(q_i8, row, d1, 0);
            if (!lane) q_scale_tmp[row * Groups + grp] = 0.0f;
            continue;
        }
        int q_head = 0, token = 0;
        row_to_qt(row, q_head, token);
        float x0 =
            row < RowCount ? __bfloat162float(q[gqa_q_index<Geometry>(q_head, d0, token)]) : 0.0f;
        float x1 =
            row < RowCount ? __bfloat162float(q[gqa_q_index<Geometry>(q_head, d1, token)]) : 0.0f;
        if constexpr (Packed) gqa_kv_hadamard64(x0, x1, FullMask);
        const float amax = warp_max(fmaxf(fabsf(x0), fabsf(x1)), FullMask);
        const float qs = amax > 0 ? amax / 127.0f : 0.0f, inv = qs > 0 ? 1.0f / qs : 0.0f;
        tiered_i8_store_swz(q_i8, row, d0, gqa_kv_quant_code(x0, inv));
        tiered_i8_store_swz(q_i8, row, d1, gqa_kv_quant_code(x1, inv));
        if (!lane) q_scale_tmp[row * Groups + grp] = qs;
    }
    ninfer::ops::cp_wait<0>();
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    // Quad lanes read one row/group address (shared-memory broadcast).
    // Keeping all eight scales live across the key loop spills small-T and
    // full/masked prefill variants; reload each group from dedicated storage.
    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }

    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;

    const float scale_l2 = scale * Log2E;
    auto produce_tile    = [&]<bool FullScore>(int kb) {
        const int k0       = first_tile + kb * Bc;
        const int key_base = pages[k0 / 64].x * 64 + (k0 & 63);

        // One warp per row tile produces P and alpha while the remaining warps
        // stream/dequant V.
        if (warp < RowTiles) {
            const int producer_row_base = warp * 16;
            Narrow* p_sw                = &p_s[producer_row_base * Bc];
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = 0.0f;
                score[nt][1] = 0.0f;
                score[nt][2] = 0.0f;
                score[nt][3] = 0.0f;
            }

#pragma unroll
            for (int g = 0; g < Groups; ++g) {
                const float qs0 = q_scale_tmp[(producer_row_base + gid) * Groups + g];
                const float qs1 = q_scale_tmp[(producer_row_base + gid + 8) * Groups + g];
                unsigned af[GroupKc][4];
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int k    = g * GroupKc + kk;
                    const int acol = k * 16 + a_coloff;
                    ldmatrix_x4(
                        af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                        smem_addr(&q_b16[(producer_row_base + a_rowoff) * DB16 +
                                         gqa_small_t_tc_swz(producer_row_base + a_rowoff, acol)]));
                }

#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                    for (int kk = 0; kk < GroupKc; ++kk) {
                        const int k    = g * GroupKc + kk;
                        const int brow = nt * 8 + b_rin;
                        const int bcol = k * 16 + b_koff;
                        unsigned bf[2];
                        ldmatrix_x2(
                            bf[0], bf[1],
                            smem_addr(&k_b16[brow * DB16 + gqa_small_t_tc_swz(brow, bcol)]));
                        mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                               bf[1]);
                    }
                    // keya/keyb depend only on lid, so the eight lanes sharing a lid
                    // read one address: an LDS broadcast, not a dynamic-index shuffle.
                    const int keya  = nt * 8 + 2 * lid;
                    const int keyb  = keya + 1;
                    const float ka  = __half2float(k_scale_s[keya * Groups + g]);
                    const float kb2 = __half2float(k_scale_s[keyb * Groups + g]);
                    score[nt][0] += qs0 * ka * static_cast<float>(c0);
                    score[nt][1] += qs0 * kb2 * static_cast<float>(c1);
                    score[nt][2] += qs1 * ka * static_cast<float>(c2);
                    score[nt][3] += qs1 * kb2 * static_cast<float>(c3);
                }
            }

            const int row0 = producer_row_base + gid;
            const int row1 = row0 + 8;
            int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
            row_to_qt(row0, q_head0, token0);
            row_to_qt(row1, q_head1, token1);
            const int qabs0 = (row0 < RowCount) ? pos[token0] : -1;
            const int qabs1 = (row1 < RowCount) ? pos[token1] : -1;
            float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0 = nt * 8 + 2 * lid;
                const int col1 = col0 + 1;
                const int key0 = key_base + col0;
                const int key1 = key_base + col1;
                score[nt][0]   = (FullScore || (row0 < RowCount && k0 + col0 >= split_start &&
                                                k0 + col0 < split_end && key0 <= qabs0))
                                     ? score[nt][0]
                                     : -CUDART_INF_F;
                score[nt][1]   = (FullScore || (row0 < RowCount && k0 + col1 >= split_start &&
                                                k0 + col1 < split_end && key1 <= qabs0))
                                     ? score[nt][1]
                                     : -CUDART_INF_F;
                score[nt][2]   = (FullScore || (row1 < RowCount && k0 + col0 >= split_start &&
                                                k0 + col0 < split_end && key0 <= qabs1))
                                     ? score[nt][2]
                                     : -CUDART_INF_F;
                score[nt][3]   = (FullScore || (row1 < RowCount && k0 + col1 >= split_start &&
                                                k0 + col1 < split_end && key1 <= qabs1))
                                     ? score[nt][3]
                                     : -CUDART_INF_F;
                bm0            = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1            = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);

            const float nm0    = fmaxf(m0, bm0);
            const float nm1    = fmaxf(m1, bm1);
            const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * scale_l2);
            const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * scale_l2);

            float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0 = nt * 8 + 2 * lid;
                const int col1 = col0 + 1;
                const float p00 =
                    (FullScore || (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F))
                        ? exp2_approx(__fmul_rn(__fsub_rn(score[nt][0], nm0), scale_l2))
                        : 0.0f;
                const float p01 =
                    (FullScore || (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F))
                        ? exp2_approx(__fmul_rn(__fsub_rn(score[nt][1], nm0), scale_l2))
                        : 0.0f;
                const float p10 =
                    (FullScore || (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F))
                        ? exp2_approx(__fmul_rn(__fsub_rn(score[nt][2], nm1), scale_l2))
                        : 0.0f;
                const float p11 =
                    (FullScore || (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F))
                        ? exp2_approx(__fmul_rn(__fsub_rn(score[nt][3], nm1), scale_l2))
                        : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int index0 = gid * Bc + probability_swizzle(gid, col0);
                const int index1 = (gid + 8) * Bc + probability_swizzle(gid + 8, col0);
                if constexpr (Prefill) {
                    p_sw[index0]     = tiered_probability<true>(p00);
                    p_sw[index0 + 1] = tiered_probability<true>(p01);
                    p_sw[index1]     = tiered_probability<true>(p10);
                    p_sw[index1 + 1] = tiered_probability<true>(p11);
                } else {
                    // Adjacent columns remain aligned pairs under either
                    // swizzle. Keep rounded values in registers for residuals;
                    // no shared reload or individual 16-bit stores are needed.
                    const auto p0 = pack_bf16x2(p00, p01), p1 = pack_bf16x2(p10, p11);
                    store_vec(p_sw + index0, p0);
                    store_vec(p_sw + index1, p1);
                    auto* lo = &p_lo[producer_row_base * Bc];
                    store_vec(lo + index0, pack_bf16x2(p00 - __uint_as_float(p0 << 16),
                                                       p01 - __uint_as_float(p0 & 0xffff0000u)));
                    store_vec(lo + index1, pack_bf16x2(p10 - __uint_as_float(p1 << 16),
                                                       p11 - __uint_as_float(p1 & 0xffff0000u)));
                }
            }
            bl0 = warp_sum<4>(bl0, FullMask);
            bl1 = warp_sum<4>(bl1, FullMask);

            l0 = l0 * alpha0 + bl0;
            l1 = l1 * alpha1 + bl1;
            m0 = nm0;
            m1 = nm1;
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else {
            const int loader_tid = tid - ProducerThreads;
#pragma unroll 1
            for (int chunk = loader_tid; chunk < Bc * (D / 8); chunk += VLoaderThreads) {
                const int key_l = chunk / (D / 8);
                const int dc    = chunk - key_l * (D / 8);
                const int d     = dc * 8;
                const int key   = k0 + key_l;
                Narrow* dst     = &v_bf16[key_l * D + gqa_small_t_tc_swz(key_l, d)];
                if (FullScore || (key >= split_start && key < split_end)) {
                    const int grp  = d >> 6;
                    const float vs = __half2float(v_scale_s[key_l * Groups + grp]);
                    store_vec(dst, tiered_dequant_v<Prefill>(&v_i8[key_l * D + d], vs));
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();
    };
    // Specialize only score/mask generation and V dequantization. Keep the
    // identical PV and next-tile pipeline in one copy to bound instruction size.
    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0       = first_tile + kb * Bc;
        const int key_base = pages[k0 / 64].x * 64 + (k0 & 63);
        // Every real query sees this whole tile. Padded rows own initialized
        // zero Q and may compute unused values; their outputs are never stored.
        if (k0 >= split_start && k0 + Bc <= split_end && key_base + Bc - 1 <= pos[0])
            produce_tile.template operator()<true>(kb);
        else
            produce_tile.template operator()<false>(kb);
        const bool has_next = kb + 1 < key_blocks;
        if (has_next) {
            const int next_k0 = k0 + Bc;
            load_tile(next_k0);
        }

        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        Narrow* p_consumer          = &p_s[consumer_row_base * Bc];
        const float alpha0          = alpha_s[consumer_row_base + gid];
        const float alpha1          = alpha_s[consumer_row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            unsigned pf[4], plf[4];
            const int pcol = k * 16 + a_coloff;
            ldmatrix_x4(
                pf[0], pf[1], pf[2], pf[3],
                smem_addr(&p_consumer[a_rowoff * Bc + probability_swizzle(a_rowoff, pcol)]));
            if constexpr (!Prefill) {
                ldmatrix_x4(plf[0], plf[1], plf[2], plf[3],
                            smem_addr(&p_lo[(consumer_row_base + a_rowoff) * Bc +
                                            probability_swizzle(a_rowoff, pcol)]));
            }
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int global_n = consumer_slice * PVNtPerWarp + n;
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_bf16[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
                tiered_pv_mma<Prefill>(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1],
                                       pf[2], pf[3], vf[0], vf[1]);
                if constexpr (!Prefill) {
                    mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], plf[0], plf[1], plf[2],
                             plf[3], vf[0], vf[1]);
                }
            }
        }
        if (has_next) { ninfer::ops::cp_wait<0>(); }
        __syncthreads();
    }

    if constexpr (Packed) {
        static_assert(PVNtPerWarp == 8 || PVNtPerWarp == 16);
        tiered_inverse_rotate_fragment(acc, lid);
    }

    if (warp < RowTiles && lid == 0) {
        const int row0 = warp * 16 + gid;
        const int row1 = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0;
            int token  = 0;
            row_to_qt(row0, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, full_width)] =
                m0 * scale;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, full_width)] = l0;
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token  = 0;
            row_to_qt(row1, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, full_width)] =
                m1 * scale;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, full_width)] = l1;
        }
    }

#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        const int d0                = (consumer_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        const int row0              = consumer_row_base + gid;
        const int row1              = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0;
            int token  = 0;
            row_to_qt(row0, q_head, token);
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d0, token, split, full_width);
            *reinterpret_cast<float2*>(&partial_acc[dst]) = make_float2(acc[n][0], acc[n][1]);
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token  = 0;
            row_to_qt(row1, q_head, token);
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d0, token, split, full_width);
            *reinterpret_cast<float2*>(&partial_acc[dst]) = make_float2(acc[n][2], acc[n][3]);
        }
    }
}
} // namespace ninfer::ops

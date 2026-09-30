#pragma once
// Tiered read-only Q8 Tensor Core attention. Native decode arithmetic with
// explicit sparse page access, frontier-based key prefixes and FP32 partials.
#include "ops/kernel/gqa_attention_partial_common.cuh"
#include <type_traits>
namespace ninfer::ops {
template <typename Geometry, int TokenTile, int WarpsPerCta, int KeyBlock, bool DynamicArena,
          bool Packed, bool Prefill>
__launch_bounds__(WarpsPerCta * 32, 1) __global__
    void gqa_attention_partial_i8_kernel(const __nv_bfloat16* q, const std::int32_t* pos,
                                         TieredKVPlanes resident, TieredKVPlanes staging,
                                         int resident_pages, const int2* pages, const int* prefix,
                                         int page_count, int frontier, int full_width, float scale,
                                         float* partial_acc, float* partial_m, float* partial_l) {
    using Narrow = std::conditional_t<Prefill, __half, __nv_bfloat16>;
    constexpr int Wc = WarpsPerCta;
    constexpr int RowCapacity = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles = (RowCapacity + 15) / 16;
    constexpr int Br = RowTiles * 16;
    constexpr int Bc = KeyBlock;
    constexpr int D = kGqaHeadDim;
    constexpr int DB16 = D / 2;
    constexpr int Threads = Wc * 32;
    constexpr int Groups = kGqaKvQuantGroups;
    constexpr int GroupKc = kGqaKvQuantGroup / 32;
    constexpr int QKKs = D / 32;
    constexpr int QKNt = Bc / 8;
    constexpr int ConsumerWarpsPerTile = Wc / RowTiles;
    constexpr int PVNtPerWarp = D / (ConsumerWarpsPerTile * 8);
    constexpr int PVKs = Bc / 16;
    constexpr int ProducerThreads = RowTiles * 32;
    constexpr int VLoaderThreads = Threads - ProducerThreads;
    constexpr float Log2E = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(TokenTile >= 1 && TokenTile <= 10);
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
    std::int8_t* r_s = DynamicArena ? dynamic_r_s : static_r_s;
    std::int8_t* q_i8 = q_s;
    std::int8_t* k_i8 = r_s;
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);
    __nv_bfloat16* k_b16 = reinterpret_cast<__nv_bfloat16*>(k_i8);
    std::int8_t* v_i8 = r_s + Bc * D;
    Narrow* v_bf16 = reinterpret_cast<Narrow*>(r_s + 2 * Bc * D);
    __shared__ __align__(16) Narrow p_s[Br * Bc];
    __shared__ __align__(16) __nv_bfloat16 p_lo[Prefill ? 8 : Br * Bc];
    __shared__ float q_scale_tmp[Br * Groups];
    __shared__ int key_absolute[Bc], key_physical[Bc];
    __shared__ float alpha_s[Br];
    __shared__ __align__(16) __half k_scale_s[Bc * Groups];
    __shared__ __align__(16) __half v_scale_s[Bc * Groups];

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
        for (int row = tid; row < RowCount; row += Threads) {
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
        for (int idx = tid; idx < RowCount * D; idx += Threads) {
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

    auto issue_kv_tile = [&](int tile_k0) {
        for (int j = tid; j < Bc; j += Threads) {
            const int ordinal = tile_k0 + j;
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
        for (int j = tid; j < Bc; j += Threads) {
            if (key_physical[j] >= 0) {
                const bool stage = key_physical[j] >= resident_pages;
                const auto source = stage ? staging : resident;
                const int page = key_physical[j] - (stage ? resident_pages : 0);
                const auto off =
                    gqa_kv_quant_scale_index<Geometry>(page, kv_head, 0, key_absolute[j] & 63);
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
            if (key_physical[j] >= 0) {
                const bool stage = key_physical[j] >= resident_pages;
                const auto source = stage ? staging : resident;
                const int page = key_physical[j] - (stage ? resident_pages : 0),
                          offset = key_absolute[j] & 63;
                if constexpr (Packed) {
                    const auto off = gqa_kv_i4_code_index<Geometry>(page, kv_head, d / 2, offset);
                    gqa_kv_unpack_i4x16(static_cast<const std::uint8_t*>(source.k) + off, kd);
                    gqa_kv_unpack_i4x16(static_cast<const std::uint8_t*>(source.v) + off, vd);
                } else {
                    const auto off = gqa_kv_quant_code_index<Geometry>(page, kv_head, d, offset);
                    cp_async<16>(kd, static_cast<const std::int8_t*>(source.k) + off);
                    cp_async<16>(vd, static_cast<const std::int8_t*>(source.v) + off);
                }
            } else {
                store_vec(kd, make_int4(0, 0, 0, 0));
                store_vec(vd, make_int4(0, 0, 0, 0));
            }
        }
        cp_commit();
    };
    issue_kv_tile(first_tile);

    // One warp owns every complete (row,group), including zero padding. No
    // cross-warp clear/write race; scale storage never aliases probabilities.
    for (int unit = warp; unit < Br * Groups; unit += Wc) {
        const int row = unit / Groups, grp = unit % Groups, d0 = grp * 64 + lane, d1 = d0 + 32;
        int q_head = 0, token = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        float x0 =
            row < RowCount ? __bfloat162float(q[gqa_q_index<Geometry>(q_head, d0, token)]) : 0.0f;
        float x1 =
            row < RowCount ? __bfloat162float(q[gqa_q_index<Geometry>(q_head, d1, token)]) : 0.0f;
        if constexpr (Packed)
            gqa_kv_hadamard64(x0, x1, FullMask);
        const float amax = warp_max(fmaxf(fabsf(x0), fabsf(x1)), FullMask);
        const float qs = amax > 0 ? amax / 127.0f : 0.0f, inv = qs > 0 ? 1.0f / qs : 0.0f;
        tiered_i8_store_swz(q_i8, row, d0, gqa_kv_quant_code(x0, inv));
        tiered_i8_store_swz(q_i8, row, d1, gqa_kv_quant_code(x1, inv));
        if (!lane)
            q_scale_tmp[row * Groups + grp] = qs;
    }
    ninfer::ops::cp_wait<0>();
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat = lane >> 3;
    const int a_rin = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;

    // Every lane of a four-lane quad wants the same q scale, and q_scale_tmp is
    // initialized over the whole Br x Groups tile, so each lane reads its own row
    // straight out of shared memory. The former "read on lid == 0, then
    // __shfl_sync(.., gid * 4)" broadcast cost a dynamic-index shuffle per group,
    // which ptxas has to guard with a divergence check and an out-of-line ABI
    // helper call under -rdc; a shared-memory broadcast is one conflict-free LDS.
    float q_scale_r0[Groups];
    float q_scale_r1[Groups];
    if (warp < RowTiles) {
        // warp < RowTiles and gid <= 7, so producer_row0 + 8 <= Br - 1.
        const int producer_row0 = warp * 16 + gid;
#pragma unroll
        for (int g = 0; g < Groups; ++g) {
            q_scale_r0[g] = q_scale_tmp[producer_row0 * Groups + g];
            q_scale_r1[g] = q_scale_tmp[(producer_row0 + 8) * Groups + g];
        }
    }

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[n][i] = 0.0f;
        }
    }

    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;

        // One warp per row tile produces P and alpha while the remaining warps
        // stream/dequant V.
        if (warp < RowTiles) {
            const int producer_row_base = warp * 16;
            Narrow* p_sw = &p_s[producer_row_base * Bc];
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
                unsigned af[GroupKc][4];
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int k = g * GroupKc + kk;
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
                        const int k = g * GroupKc + kk;
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
                    const int keya = nt * 8 + 2 * lid;
                    const int keyb = keya + 1;
                    const float ka = __half2float(k_scale_s[keya * Groups + g]);
                    const float kb2 = __half2float(k_scale_s[keyb * Groups + g]);
                    score[nt][0] += q_scale_r0[g] * ka * static_cast<float>(c0);
                    score[nt][1] += q_scale_r0[g] * kb2 * static_cast<float>(c1);
                    score[nt][2] += q_scale_r1[g] * ka * static_cast<float>(c2);
                    score[nt][3] += q_scale_r1[g] * kb2 * static_cast<float>(c3);
                }
            }

            const int row0 = producer_row_base + gid;
            const int row1 = row0 + 8;
            int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head0, token0);
            gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head1, token1);
            const int qabs0 = (row0 < RowCount) ? pos[token0] : -1;
            const int qabs1 = (row1 < RowCount) ? pos[token1] : -1;
            float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0 = nt * 8 + 2 * lid;
                const int col1 = col0 + 1;
                const int key0 = key_absolute[col0];
                const int key1 = key_absolute[col1];
                score[nt][0] = (row0 < RowCount && k0 + col0 >= split_start &&
                                k0 + col0 < split_end && key0 <= qabs0)
                                   ? score[nt][0] * scale
                                   : -CUDART_INF_F;
                score[nt][1] = (row0 < RowCount && k0 + col1 >= split_start &&
                                k0 + col1 < split_end && key1 <= qabs0)
                                   ? score[nt][1] * scale
                                   : -CUDART_INF_F;
                score[nt][2] = (row1 < RowCount && k0 + col0 >= split_start &&
                                k0 + col0 < split_end && key0 <= qabs1)
                                   ? score[nt][2] * scale
                                   : -CUDART_INF_F;
                score[nt][3] = (row1 < RowCount && k0 + col1 >= split_start &&
                                k0 + col1 < split_end && key1 <= qabs1)
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
                p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col0)] = tiered_probability<Prefill>(p00);
                p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col1)] = tiered_probability<Prefill>(p01);
                p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col0)] =
                    tiered_probability<Prefill>(p10);
                p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col1)] =
                    tiered_probability<Prefill>(p11);
                if constexpr (!Prefill) {
                    __nv_bfloat16* lo = &p_lo[producer_row_base * Bc];
                    const int indices[]{gid * Bc + gqa_small_t_tc_swz32(gid, col0),
                                        gid * Bc + gqa_small_t_tc_swz32(gid, col1),
                                        (gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col0),
                                        (gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col1)};
                    const float values[]{p00, p01, p10, p11};
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        lo[indices[i]] =
                            __float2bfloat16(values[i] - __bfloat162float(p_sw[indices[i]]));
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
                const int dc = chunk - key_l * (D / 8);
                const int d = dc * 8;
                const int key = k0 + key_l;
                Narrow* dst = &v_bf16[key_l * D + gqa_small_t_tc_swz(key_l, d)];
                if (key >= split_start && key < split_end) {
                    const int grp = d >> 6;
                    const float vs = __half2float(v_scale_s[key_l * Groups + grp]);
                    store_vec(dst, tiered_dequant_v<Prefill>(&v_i8[key_l * D + d], vs));
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) {
            const int next_k0 = k0 + Bc;
            issue_kv_tile(next_k0);
        }

        const int consumer_tile = warp % RowTiles;
        const int consumer_slice = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        Narrow* p_consumer = &p_s[consumer_row_base * Bc];
        const float alpha0 = alpha_s[consumer_row_base + gid];
        const float alpha1 = alpha_s[consumer_row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            const int global_n = consumer_slice * PVNtPerWarp + n;
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(
                    pf[0], pf[1], pf[2], pf[3],
                    smem_addr(&p_consumer[a_rowoff * Bc + gqa_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_bf16[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
                tiered_pv_mma<Prefill>(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1],
                                       pf[2], pf[3], vf[0], vf[1]);
                if constexpr (!Prefill) {
                    ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                                smem_addr(&p_lo[(consumer_row_base + a_rowoff) * Bc +
                                                gqa_small_t_tc_swz32(a_rowoff, pcol)]));
                    mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                             vf[0], vf[1]);
                }
            }
        }
        if (has_next) {
            ninfer::ops::cp_wait<0>();
        }
        __syncthreads();
    }

    if (warp < RowTiles && lid == 0) {
        const int row0 = warp * 16 + gid;
        const int row1 = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0;
            int token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, full_width)] = m0;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, full_width)] = l0;
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, full_width)] = m1;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, full_width)] = l1;
        }
    }

#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int consumer_tile = warp % RowTiles;
        const int consumer_slice = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        const int d0 = (consumer_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        const int row0 = consumer_row_base + gid;
        const int row1 = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0;
            int token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head, token);
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d0, token, split, full_width);
            *reinterpret_cast<float2*>(&partial_acc[dst]) = make_float2(acc[n][0], acc[n][1]);
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head, token);
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d0, token, split, full_width);
            *reinterpret_cast<float2*>(&partial_acc[dst]) = make_float2(acc[n][2], acc[n][3]);
        }
    }
}
} // namespace ninfer::ops

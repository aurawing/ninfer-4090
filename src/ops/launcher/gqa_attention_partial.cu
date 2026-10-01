#include "ops/launcher/gqa_attention_partial.h"
#include "ops/kernel/gqa_attention_partial_i8.cuh"
#include "ops/kernel/gqa_attention_partial_bf16.cuh"
#include "ops/kernel/attention_partial_lse_merge.cuh"
#include "core/device.h"

namespace ninfer::ops::detail {
namespace {
TieredKVPlanes planes(const PagedKVLayerView& c) {
    return {c.k_pages.data, c.v_pages.data, static_cast<const __half*>(c.k_scale_pages.data),
            static_cast<const __half*>(c.v_scale_pages.data)};
}

template <int Tile, bool Prefill, bool Packed>
void launch_i8(const Tensor& q, const Tensor& pos, float scale, const PagedKVLayerView& resident,
               const PagedKVLayerView& staging, const Tensor& pages, const Tensor& prefix,
               int frontier, int splits, AttentionPartial& out, cudaStream_t stream) {
    constexpr int rows         = ((Prefill ? Tile : Tile * 6) + 15) / 16;
    constexpr int warps        = rows * ((Prefill || Packed || Tile < 4) ? 4 : 2);
    constexpr int block        = Prefill ? 64 : 32;
    constexpr bool dynamic     = Prefill || Tile >= 8;
    constexpr std::size_t smem = dynamic ? 4 * block * 256 : 0;
    if constexpr (dynamic) {
        static const auto status =
            cudaFuncSetAttribute(gqa_attention_partial_i8_kernel<Gqa27Geometry, Tile, warps, block,
                                                                 dynamic, Packed, Prefill>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
        CUDA_CHECK(status);
    }
    // Query-tile-major rasterization matches dense prefill's K/V reuse.
    const dim3 grid = Prefill ? dim3((q.ne[2] + Tile - 1) / Tile, 24, splits)
                              : dim3(4, splits, (q.ne[2] + Tile - 1) / Tile);
    gqa_attention_partial_i8_kernel<Gqa27Geometry, Tile, warps, block, dynamic, Packed, Prefill>
        <<<grid, warps * 32, smem, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), static_cast<const int*>(pos.data),
            planes(resident), planes(staging), resident.k_pages.data ? resident.k_pages.ne[3] : 0,
            static_cast<const int2*>(pages.data), static_cast<const int*>(prefix.data),
            pages.data ? pages.ne[1] : 0, frontier, q.ne[2], scale, static_cast<float*>(out.o.data),
            static_cast<float*>(out.m.data), static_cast<float*>(out.l.data));
}

template <int Tile>
void launch_bf16(const Tensor& q, const Tensor& pos, float scale, const PagedKVLayerView& resident,
                 const PagedKVLayerView& staging, const Tensor& pages, const Tensor& prefix,
                 int frontier, int splits, AttentionPartial& out, cudaStream_t stream) {
    const dim3 grid(4, splits, (q.ne[2] + Tile - 1) / Tile);
    gqa_attention_partial_bf16_kernel<Gqa27Geometry, Tile, 4><<<grid, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const int*>(pos.data),
        planes(resident), planes(staging), resident.k_pages.data ? resident.k_pages.ne[3] : 0,
        static_cast<const int2*>(pages.data), static_cast<const int*>(prefix.data),
        pages.data ? pages.ne[1] : 0, frontier, q.ne[2], scale, static_cast<float*>(out.o.data),
        static_cast<float*>(out.m.data), static_cast<float*>(out.l.data));
}

template <int Tile, bool Prefill>
void launch(const Tensor& q, const Tensor& pos, float scale, const PagedKVLayerView& resident,
            const PagedKVLayerView& staging, const Tensor& pages, const Tensor& prefix,
            int frontier, int splits, AttentionPartial& out, cudaStream_t stream) {
    if (resident.dtype == DType::BF16)
        launch_bf16<Tile>(q, pos, scale, resident, staging, pages, prefix, frontier, splits, out,
                          stream);
    else if (resident.packed_k)
        launch_i8<Tile, Prefill, true>(q, pos, scale, resident, staging, pages, prefix, frontier,
                                       splits, out, stream);
    else
        launch_i8<Tile, Prefill, false>(q, pos, scale, resident, staging, pages, prefix, frontier,
                                        splits, out, stream);
}
} // namespace

void gqa_partial_launch(bool prefill, const Tensor& q, const Tensor& pos, float scale,
                        const PagedKVLayerView& resident, const PagedKVLayerView& staging,
                        const Tensor& pages, const Tensor& prefix, std::uint32_t frontier,
                        std::int32_t splits, AttentionPartial& out, cudaStream_t stream) {
    if (prefill) {
        if (resident.dtype == DType::BF16)
            launch_bf16<10>(q, pos, scale, resident, staging, pages, prefix, frontier, splits, out,
                            stream);
        else if (resident.packed_k)
            launch_i8<64, true, true>(q, pos, scale, resident, staging, pages, prefix, frontier,
                                      splits, out, stream);
        else
            launch_i8<64, true, false>(q, pos, scale, resident, staging, pages, prefix, frontier,
                                       splits, out, stream);
    } else if (q.ne[2] == 1)
        launch<1, false>(q, pos, scale, resident, staging, pages, prefix, frontier, splits, out,
                         stream);
    else if (q.ne[2] == 2)
        launch<2, false>(q, pos, scale, resident, staging, pages, prefix, frontier, splits, out,
                         stream);
    else if (q.ne[2] <= 4)
        launch<4, false>(q, pos, scale, resident, staging, pages, prefix, frontier, splits, out,
                         stream);
    else
        launch<8, false>(q, pos, scale, resident, staging, pages, prefix, frontier, splits, out,
                         stream);
    CUDA_CHECK(cudaGetLastError());
}

void partial_lse_merge_launch(const AttentionPartial& input, AttentionPartial& output,
                              cudaStream_t stream) {
    const int rows = 24 * input.o.ne[2], parts = input.o.ne[3];
    attention_partial_lse_merge_kernel<false><<<rows, 256, (parts + 3) * sizeof(float), stream>>>(
        static_cast<const float*>(input.o.data), static_cast<const float*>(input.m.data),
        static_cast<const float*>(input.l.data), rows, parts, static_cast<float*>(output.o.data),
        static_cast<float*>(output.m.data), static_cast<float*>(output.l.data));
    CUDA_CHECK(cudaGetLastError());
}

void partial_lse_accumulate_launch(const AttentionPartial& input, AttentionPartial& state,
                                   bool reset, cudaStream_t stream, Tensor* final_output) {
    const int rows = 24 * input.o.ne[2], parts = input.o.ne[3];
    auto launch_merge = [&]<bool Finalize>() {
        attention_partial_lse_merge_kernel<true, Finalize>
            <<<rows, 256, (parts + 3) * sizeof(float), stream>>>(
                static_cast<const float*>(input.o.data), static_cast<const float*>(input.m.data),
                static_cast<const float*>(input.l.data), rows, parts,
                static_cast<float*>(state.o.data), static_cast<float*>(state.m.data),
                static_cast<float*>(state.l.data), reset,
                final_output ? static_cast<__nv_bfloat16*>(final_output->data) : nullptr);
    };
    if (final_output)
        launch_merge.template operator()<true>();
    else
        launch_merge.template operator()<false>();
    CUDA_CHECK(cudaGetLastError());
}

void partial_finalize_launch(const AttentionPartial& input, Tensor& output, cudaStream_t stream) {
    const std::int64_t elements = std::int64_t(256) * 24 * input.o.ne[2];
    attention_partial_finalize_kernel<<<static_cast<unsigned>((elements + 255) / 256), 256, 0,
                                        stream>>>(static_cast<const float*>(input.o.data),
                                                  static_cast<const float*>(input.l.data), elements,
                                                  static_cast<__nv_bfloat16*>(output.data));
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail

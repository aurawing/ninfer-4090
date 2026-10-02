#include <ninfer/ops/gqa_kv_append_masked.h>
#include "core/device.h"
#include "ops/kernel/gqa_attention_prefill_bf16.cuh"
#include "ops/kernel/gqa_attention_prefill_i8.cuh"
#include <algorithm>
#include <initializer_list>
#include <stdexcept>
namespace ninfer::ops {
namespace {
void shape(const Tensor& t, DType dtype, std::initializer_list<int> dimensions) {
    int expected[4]{1, 1, 1, 1};
    std::copy(dimensions.begin(), dimensions.end(), expected);
    if (t.dtype != dtype || !t.data || !t.is_contiguous() ||
        !std::equal(std::begin(expected), std::end(expected), t.ne))
        throw std::invalid_argument("masked KV append shape/dtype/storage");
}
struct WindowWriteMetadata {
    const std::int32_t* table;
    const std::int32_t* valid;
    int column;
    __device__ int valid_tokens(int width) const {
        return valid ? min(width, max(0, valid[0] - column)) : width;
    }
    __device__ const std::int32_t* block_table() const { return table; }
};
} // namespace
void gqa_kv_append_masked(const Tensor& k, const Tensor& v, const Tensor& pos, const Tensor& valid,
                          int column, PagedKVLayerView cache, cudaStream_t stream) {
    const int width  = k.ne[2];
    const bool plain = !cache.packed_k && !cache.packed_v && !cache.rotate_k && !cache.rotate_v &&
                       !cache.e8_lattice && !cache.e8_root;
    const bool packed = cache.packed_k && cache.packed_v && cache.rotate_k && cache.rotate_v &&
                        cache.e8_lattice && !cache.e8_root;
    if (width < 1 || width > 16 || column < 0 || column > 15 || cache.head_dim != 256 ||
        cache.num_kv_heads != 4 || (cache.dtype != DType::BF16 && cache.dtype != DType::I8) ||
        (cache.dtype == DType::BF16 && (!plain || cache.quant_group != 0)) ||
        (cache.dtype == DType::I8 && ((!plain && !packed) || cache.quant_group != 64)))
        throw std::invalid_argument("masked KV append profile or extent");
    shape(k, DType::BF16, {256, 4, width});
    shape(v, DType::BF16, {256, 4, width});
    shape(pos, DType::I32, {width});
    if (valid.data) shape(valid, DType::I32, {1});
    const int pages     = cache.k_pages.ne[3];
    const int dimension = packed ? 128 : 256;
    const auto type = cache.dtype == DType::BF16 ? DType::BF16 : (packed ? DType::U8 : DType::I8);
    if (pages < 1 || cache.block_table.dtype != DType::I32 || !cache.block_table.data ||
        !cache.block_table.is_contiguous())
        throw std::invalid_argument("masked KV append pool or table");
    shape(cache.k_pages, type, {dimension, 64, 4, pages});
    shape(cache.v_pages, type, {dimension, 64, 4, pages});
    if (cache.dtype == DType::I8) {
        shape(cache.k_scale_pages, DType::FP16, {4, 64, 4, pages});
        shape(cache.v_scale_pages, DType::FP16, {4, 64, 4, pages});
    }
    WindowWriteMetadata metadata{static_cast<const int*>(cache.block_table.data),
                                 static_cast<const int*>(valid.data), column};
    if (cache.dtype == DType::BF16) {
        const int blocks = (width * 4 * 32 + 127) / 128;
        gqa_attention_prefill_fill_bf16_kernel<Gqa27Geometry, WindowWriteMetadata>
            <<<blocks, 128, 0, stream>>>(static_cast<const __nv_bfloat16*>(k.data),
                                         static_cast<const __nv_bfloat16*>(v.data),
                                         static_cast<const int*>(pos.data), metadata,
                                         static_cast<__nv_bfloat16*>(cache.k_pages.data),
                                         static_cast<__nv_bfloat16*>(cache.v_pages.data), width);
    } else {
        const int blocks = (width * 4 * 4 + 7) / 8;
        auto launch      = [&]<bool Packed>() {
            gqa_attention_prefill_fill_i8_kernel<Gqa27Geometry, Packed, Packed, Packed, Packed,
                                                 Packed, false, WindowWriteMetadata>
                <<<blocks, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(k.data),
                                             static_cast<const __nv_bfloat16*>(v.data),
                                             static_cast<const int*>(pos.data), metadata,
                                             static_cast<std::int8_t*>(cache.k_pages.data),
                                             static_cast<std::uint8_t*>(cache.v_pages.data),
                                             static_cast<__half*>(cache.k_scale_pages.data),
                                             static_cast<__half*>(cache.v_scale_pages.data), width);
        };
        if (cache.e8_lattice)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops

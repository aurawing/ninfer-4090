#pragma once
#include "ninfer/ops/gqa_attention_partial.h"

namespace ninfer::ops::detail {
void gqa_partial_launch(bool prefill, const Tensor& q, const Tensor& positions, float scale,
                        const PagedKVLayerView& resident, const PagedKVLayerView& staging,
                        const Tensor& pages, const Tensor& prefix, std::uint32_t frontier,
                        std::int32_t splits, AttentionPartial& partial, cudaStream_t stream);
void partial_lse_merge_launch(const AttentionPartial& input, AttentionPartial& output,
                              cudaStream_t stream);
void partial_finalize_launch(const AttentionPartial& input, Tensor& output, cudaStream_t stream);
} // namespace ninfer::ops::detail

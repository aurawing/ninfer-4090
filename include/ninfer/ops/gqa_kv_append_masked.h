#pragma once
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
namespace ninfer::ops {
// C=1 masked cache write, BF16 [256,4,W], sequential original positions [W].
// valid_columns is empty or device I32 [1]. Only the first
// clamp(valid_columns[0]-column_begin,0,W) columns may mutate any code/scale.
// W=1..16; BF16, ordinary INT8-G64 and rk4v4-e8 PageMajor planes are supported.
// Uses the same observable encoding as gqa_kv_append, owns no logical frontier.
void gqa_kv_append_masked(const Tensor& k, const Tensor& v, const Tensor& positions,
                          const Tensor& valid_columns, std::int32_t column_begin,
                          PagedKVLayerView cache, cudaStream_t stream);
} // namespace ninfer::ops

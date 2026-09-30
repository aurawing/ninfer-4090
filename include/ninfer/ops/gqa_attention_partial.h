#pragma once

#include "core/paged_kv_cache.h"
#include <span>
#include <vector>

namespace ninfer::ops {

struct AttentionPageAccess {
    std::int32_t logical_page;
    std::int32_t physical_page;
};
static_assert(sizeof(AttentionPageAccess) == 2 * sizeof(std::int32_t));

// Boundary-time validation; original logical IDs must be strictly increasing.
// Physical IDs below resident_pages address resident planes, larger IDs address
// staging at physical_page-resident_pages. Each page contributes at most 64 keys.
[[nodiscard]] std::vector<std::int32_t>
attention_access_prefix(std::span<const AttentionPageAccess> pages, std::uint32_t frontier,
                        std::uint32_t resident_pages, std::uint32_t staging_pages);

struct AttentionPartial {
    Tensor o; // FP32 [256,24,T,parts], unnormalized, original V coordinates
    Tensor m; // FP32 [24,T,parts], natural-log score maximum
    Tensor l; // FP32 [24,T,parts], sum exp(score-m)
};

// C=1, Qwen27B 24Q/4KV. Q is BF16 [256,24,T], positions I32 [T],
// monotonically increasing cache ordinals in [0,frontier). Pages are I32 [2,N],
// prefix I32 [N+1] from attention_access_prefix(), uploaded unchanged. Empty
// lists (pages = Tensor{}, prefix = I32 [1] containing zero) and splits are
// neutral: O=0,m=-inf,l=0. An empty physical pool has null code/scale tensors.
// All nonempty cache planes are PageMajor. Prefill T <= 655350, decode T <= 16;
// 96*T*parts <= INT32_MAX bounds the inverse-rotation CUDA grid.
// A pass may contain any sorted subset; covering each key exactly once across
// passes is the caller's responsibility. No cache mutation or ownership occurs.
// Output parts == splits (1..512). Splits depend on actual visible key prefix
// counts, never highest logical page or transfer arrival. Output storage is
// disjoint from inputs. Inputs must have completed their producer/H2D events.
// BF16, ordinary INT8-G64 and rk4v4-e8 are supported. Prefill uses native Q8 /
// FP16 probability-value MMA for quantized caches; small-T uses Q8 / BF16 PV.
// BF16 PV probabilities have a separate BF16 residual term to preserve
// consistency across streamed passes. Partial O is never rounded to BF16.
void gqa_attention_partial_prefill(const Tensor& q, const Tensor& positions, float scale,
                                   const PagedKVLayerView& resident,
                                   const PagedKVLayerView& staging, const Tensor& pages,
                                   const Tensor& prefix, std::uint32_t frontier,
                                   std::int32_t splits, AttentionPartial& partial,
                                   cudaStream_t stream);
void gqa_attention_partial_decode(const Tensor& q, const Tensor& positions, float scale,
                                  const PagedKVLayerView& resident, const PagedKVLayerView& staging,
                                  const Tensor& pages, const Tensor& prefix, std::uint32_t frontier,
                                  std::int32_t splits, AttentionPartial& partial,
                                  cudaStream_t stream);

// Fixed input-part order; output has exactly one part. Input/output must not
// overlap. Keeps FP32 (O,m,l), so multiple streamed passes can merge before any
// BF16 rounding. Neutral parts are ignored, including their O payload.
void attention_partial_lse_merge(const AttentionPartial& input, AttentionPartial& output,
                                 cudaStream_t stream);
void attention_partial_finalize(const AttentionPartial& input, Tensor& output,
                                cudaStream_t stream); // BF16 [256,24,T]

} // namespace ninfer::ops

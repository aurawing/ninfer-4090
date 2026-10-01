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
// The conservative launch-capacity limit 96*T*parts <= INT32_MAX is retained.
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
// Fold the current pass into one persistent FP32 state. A reset ignores prior
// state bytes; otherwise prior state is the first term, followed by ascending
// split ID. Scratch/state must be disjoint. Neutral passes preserve state bits.
// Optional final_output fuses the last normalization into the merge, leaving
// the persistent state in FP32. It must be BF16 [256,24,T] and disjoint.
void attention_partial_lse_accumulate(const AttentionPartial& input, AttentionPartial& state,
                                      bool reset, cudaStream_t stream,
                                      Tensor* final_output = nullptr);
// Loading-time budget: current split scratch plus one persistent state; no
// allocation here and no factor for the number of streamed passes.
[[nodiscard]] std::size_t attention_partial_workspace_bytes(int tokens, int splits);

struct AttentionPartialWorkspacePlan {
    std::int32_t splits;
    std::size_t bytes;
};

// Loading-time selection only: cap preferred_splits to scratch+state budget,
// fail if even one split cannot fit. Never depend on transfer arrival order.
[[nodiscard]] AttentionPartialWorkspacePlan
plan_attention_partial_workspace(int max_tokens, int preferred_splits, std::size_t byte_budget);

void attention_partial_finalize(const AttentionPartial& input, Tensor& output,
                                cudaStream_t stream); // BF16 [256,24,T]

// Optional rotated-V output epilogue for a single merged FP32 state. For each
// 64-coordinate group: normalize O/l, apply normalized Sylvester H64, round to
// BF16 (nearest-even), apply H64 again, and round the original-basis result to
// BF16. This matches the rotated-V dense output rounding boundary. O,m,l stay
// bitwise unchanged in original V coordinates; l<=0 ignores O and emits zeros.
// Same shape, storage and non-overlap contract as attention_partial_finalize.
void attention_partial_finalize_rotated(const AttentionPartial& input, Tensor& output,
                                        cudaStream_t stream);

} // namespace ninfer::ops

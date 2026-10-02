#pragma once
#include "core/tensor.h"
#include <cstdint>
#include <cuda_runtime_api.h>
namespace ninfer::ops {
struct ScoreWorkspace {
    Tensor logits;     // FP32 [pages, query_heads*tokens], fixed row logits
    Tensor row_stats;  // FP32 [2,query_heads*tokens]: max, stable denominator
    Tensor row_status; // I32 [query_heads*tokens]: bounded nonfinite indicator
    Tensor scores;     // FP32 [pages], accumulated layer mass
    Tensor status;     // I32 [1], sticky nonfinite indicator, reset on first_layer
};
// Q is normalized pre-RoPE BF16 [D,QH,M]; MeanK FP16 [D,KVH,P]. D contiguous.
// GQA kvh=qh/(QH/KVH); each row softmax spans exactly [domain_first,domain_end).
// Scores sum token probabilities, averaged over layers and query heads (mass M).
// Two GPU passes: global row max/denominator, then page-owned fixed qhead/token
// accumulation. No atomics or allocations. Caller submits layers in order on one
// compute stream, explicitly first_layer=true to reset stale scores/status.
// All five workspace ranges and both immutable inputs must be disjoint.
// Caller D2H-copies scores AND status only after all layers complete; status!=0
// invalidates the entire capture and must be consumed before any CPU selection.
// Nonfinite input anywhere, including outside denominator bands, zeros scores
// for this and subsequent layers. Inputs/scratch remain alive until completion.
void kvmem_score_layer(const Tensor& q, const Tensor& means, std::uint32_t domain_first,
                       std::uint32_t domain_end, std::uint32_t layers, bool first_layer,
                       ScoreWorkspace& workspace, cudaStream_t stream);
} // namespace ninfer::ops

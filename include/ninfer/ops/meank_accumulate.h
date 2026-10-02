#pragma once
#include "core/tensor.h"
#include <cstdint>
#include <cuda_runtime_api.h>

namespace ninfer::ops {
struct MeanKOutput {
    Tensor means; // FP16 [D,H,capacity>=max(1,touched_pages)], chronological patches
    Tensor sum;   // FP32 [D,H], final partial page; zero at a sealed boundary
    Tensor count; // I32 [1], final partial count (frontier % 64)
};

// BF16 normalized pre-RoPE K [D,H,T], dimension contiguous. Only valid_tokens
// prefix columns participate. Cache ordinal and RoPE positions are independent.
// seed_count == first_ordinal % 64; seed is an exact FP32 prefix sum. Each
// coordinate adds tokens chronologically in FP32, then divides by the page's
// actual count and converts to FP16 RNE. Zero valid_tokens regenerates a restored
// partial mean from its saved seed. All outputs and inputs must be disjoint:
// notably a last-page CTA may never overwrite the first-page read-only seed.
// Caller owns storage/lifetimes and stream ordering. No allocation or host raw-K copy.
// D<=1024, H<=64. Ordinal frontier must fit signed 32-bit indexing with 63
// columns of page-rounding headroom; target max-context admission is separate.
void meank_accumulate(const Tensor& k, std::uint32_t first_ordinal, std::uint32_t valid_tokens,
                      const Tensor& seed, std::uint32_t seed_count, MeanKOutput& output,
                      cudaStream_t stream);

// Capture actual ordinary/verify rows before an in-place RoPE overwrites K.
// Capacity is exactly 16 rows; accepted-prefix accumulation runs only once the
// application final accepted length is known. Rejecting all rows needs no launch.
void meank_stage(const Tensor& k, std::uint32_t valid_tokens, const Tensor& provisional,
                 cudaStream_t stream);

// Preserve only accepted final-page rows in a bounded [D,H,64] BF16 tail.
// Same-page appends keep the old prefix. A page crossing replaces the prefix
// with the new page's rows. Call after acceptance, never for rejected rows.
// To trim within this tail, replay its suffix chronologically using the exact
// saved base sum (zero at a page boundary) with meank_accumulate. After snapshot
// restore the saved frontier becomes the base; older raw tail rows are invalid.
void meank_retain_tail(const Tensor& k, std::uint32_t first_ordinal, std::uint32_t accepted_tokens,
                       const Tensor& tail, cudaStream_t stream);
} // namespace ninfer::ops

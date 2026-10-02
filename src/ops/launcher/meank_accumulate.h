#pragma once
#include "ninfer/ops/meank_accumulate.h"
namespace ninfer::ops::detail {
void meank_accumulate_launch(const Tensor&, std::uint32_t, std::uint32_t, const Tensor&,
                             MeanKOutput&, cudaStream_t);
void meank_retain_tail_launch(const Tensor&, std::uint32_t, std::uint32_t, const Tensor&,
                              cudaStream_t);
} // namespace ninfer::ops::detail

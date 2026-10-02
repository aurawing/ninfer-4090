#pragma once
#include "ninfer/ops/kvmem_score.h"
namespace ninfer::ops::detail {
void kvmem_score_launch(const Tensor&, const Tensor&, int, int, int, bool, ScoreWorkspace&,
                        cudaStream_t);
}

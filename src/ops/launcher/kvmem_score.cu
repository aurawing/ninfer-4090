#include "core/device.h"
#include "ops/kernel/kvmem_score.cuh"
#include "ops/launcher/kvmem_score.h"
namespace ninfer::ops::detail {
void kvmem_score_launch(const Tensor& q, const Tensor& k, int first, int end, int layers,
                        bool reset, ScoreWorkspace& w, cudaStream_t stream) {
    int h = q.ne[1], m = q.ne[2], p = k.ne[2];
    score_rows<<<h * m, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __half*>(k.data), q.ne[0], h,
        k.ne[1], m, p, first, end, reset, static_cast<float*>(w.logits.data),
        static_cast<float*>(w.row_stats.data), static_cast<int*>(w.row_status.data));
    CUDA_CHECK(cudaGetLastError());
    score_pages<<<(unsigned(p) + 127) / 128, 128, 0, stream>>>(
        h, m, p, first, end, layers, reset, static_cast<const float*>(w.logits.data),
        static_cast<const float*>(w.row_stats.data), static_cast<const int*>(w.row_status.data),
        static_cast<float*>(w.scores.data), static_cast<int*>(w.status.data));
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail

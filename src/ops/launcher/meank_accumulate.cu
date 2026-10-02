#include "core/device.h"
#include "ops/kernel/meank_accumulate.cuh"
#include "ops/launcher/meank_accumulate.h"
#include <algorithm>
namespace ninfer::ops::detail {
void meank_accumulate_launch(const Tensor& k, std::uint32_t first, std::uint32_t valid,
                             const Tensor& seed, MeanKOutput& out, cudaStream_t stream) {
    int rows = k.ne[0] * k.ne[1], pages = (first % 64 + valid + 63) / 64;
    meank_accumulate_kernel<<<dim3((rows + 255) / 256, std::max(1, pages)), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const float*>(seed.data), rows,
        first % 64, valid, pages, static_cast<__half*>(out.means.data),
        static_cast<float*>(out.sum.data), static_cast<int*>(out.count.data));
    CUDA_CHECK(cudaGetLastError());
}
void meank_retain_tail_launch(const Tensor& k, std::uint32_t first, std::uint32_t valid,
                              const Tensor& tail, cudaStream_t stream) {
    int rows = k.ne[0] * k.ne[1], n = std::min(valid, (first + valid) % 64);
    meank_retain_tail_kernel<<<(rows * n + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), rows, first, valid,
        static_cast<__nv_bfloat16*>(tail.data));
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail

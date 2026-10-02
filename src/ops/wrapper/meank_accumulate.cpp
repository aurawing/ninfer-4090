#include "ninfer/ops/meank_accumulate.h"
#include "core/device.h"
#include "ops/launcher/meank_accumulate.h"
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace ninfer::ops {
namespace {
void shape(const Tensor& t, DType type, std::initializer_list<int> dims) {
    int expected[4]{1, 1, 1, 1};
    std::copy(dims.begin(), dims.end(), expected);
    if (t.dtype != type || !std::equal(std::begin(expected), std::end(expected), t.ne) ||
        !t.is_contiguous() || (t.numel() > 0 && !t.data))
        throw std::invalid_argument("MeanK tensor shape/dtype/storage");
}
void disjoint(const Tensor& a, const Tensor& b) {
    if (!a.bytes() || !b.bytes())
        return;
    auto x = reinterpret_cast<std::uintptr_t>(a.data), y = reinterpret_cast<std::uintptr_t>(b.data);
    if (x <= y ? y - x < a.bytes() : x - y < b.bytes())
        throw std::invalid_argument("MeanK input/output storage must be disjoint");
}
void input(const Tensor& k, std::uint32_t first, std::uint32_t valid) {
    if (k.ne[0] < 1 || k.ne[0] > 1024 || k.ne[1] < 1 || k.ne[1] > 64 || k.ne[2] < 0 ||
        valid > std::uint32_t(k.ne[2]) || first > std::uint32_t(INT_MAX) - 63 ||
        valid > std::uint32_t(INT_MAX) - 63 - first)
        throw std::invalid_argument("MeanK dimension/token/ordinal domain");
    shape(k, DType::BF16, {k.ne[0], k.ne[1], k.ne[2]});
}
} // namespace
void meank_accumulate(const Tensor& k, std::uint32_t first, std::uint32_t valid, const Tensor& seed,
                      std::uint32_t seed_count, MeanKOutput& out, cudaStream_t stream) {
    input(k, first, valid);
    if (seed_count != first % 64)
        throw std::invalid_argument("MeanK seed count does not match ordinal");
    int pages = int((seed_count + valid + 63) / 64);
    if (pages > 65535)
        throw std::invalid_argument("MeanK touched pages exceed CUDA grid.y capacity");
    shape(seed, DType::FP32, {k.ne[0], k.ne[1]});
    disjoint(k, seed);
    shape(out.sum, DType::FP32, {k.ne[0], k.ne[1]});
    shape(out.count, DType::I32, {1});
    if (out.means.ne[2] < std::max(1, pages))
        throw std::invalid_argument("MeanK mean patch capacity");
    shape(out.means, DType::FP16, {k.ne[0], k.ne[1], out.means.ne[2]});
    for (auto* output : {&out.means, &out.sum, &out.count}) {
        disjoint(*output, k);
        disjoint(*output, seed);
    }
    disjoint(out.means, out.sum);
    disjoint(out.means, out.count);
    disjoint(out.sum, out.count);
    detail::meank_accumulate_launch(k, first, valid, seed, out, stream);
}
void meank_stage(const Tensor& k, std::uint32_t valid, const Tensor& provisional,
                 cudaStream_t stream) {
    input(k, 0, valid);
    if (valid > 16)
        throw std::invalid_argument("MeanK provisional capture exceeds 16 rows");
    shape(provisional, DType::BF16, {k.ne[0], k.ne[1], 16});
    disjoint(k, provisional);
    if (valid)
        CUDA_CHECK(cudaMemcpyAsync(provisional.data, k.data,
                                   std::size_t(k.ne[0]) * k.ne[1] * valid * 2,
                                   cudaMemcpyDeviceToDevice, stream));
}
void meank_retain_tail(const Tensor& k, std::uint32_t first, std::uint32_t accepted,
                       const Tensor& tail, cudaStream_t stream) {
    input(k, first, accepted);
    shape(tail, DType::BF16, {k.ne[0], k.ne[1], 64});
    disjoint(k, tail);
    if (accepted && (first + accepted) % 64)
        detail::meank_retain_tail_launch(k, first, accepted, tail, stream);
}
} // namespace ninfer::ops

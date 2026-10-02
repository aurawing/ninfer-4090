#include "ninfer/ops/kvmem_score.h"
#include "ops/launcher/kvmem_score.h"
#include <algorithm>
#include <climits>
#include <stdexcept>
namespace ninfer::ops {
namespace {
void shape(const Tensor& t, DType type, std::initializer_list<int> dims) {
    int expected[4]{1, 1, 1, 1};
    std::copy(dims.begin(), dims.end(), expected);
    if (t.dtype != type || !t.data || !t.is_contiguous() ||
        !std::equal(std::begin(expected), std::end(expected), t.ne))
        throw std::invalid_argument("KVMem scoring shape/dtype/storage");
}
void disjoint(const Tensor& a, const Tensor& b) {
    auto x = reinterpret_cast<std::uintptr_t>(a.data), y = reinterpret_cast<std::uintptr_t>(b.data);
    if (x <= y ? y - x < a.bytes() : x - y < b.bytes())
        throw std::invalid_argument("KVMem scoring storage overlaps");
}
} // namespace
void kvmem_score_layer(const Tensor& q, const Tensor& k, std::uint32_t first, std::uint32_t end,
                       std::uint32_t layers, bool reset, ScoreWorkspace& w, cudaStream_t stream) {
    int d = q.ne[0], h = q.ne[1], m = q.ne[2], kv = k.ne[1], p = k.ne[2];
    if (d < 1 || d > 1024 || h < 1 || h > 64 || m < 1 || m > 16 || kv < 1 || kv > h || h % kv ||
        p < 1 || std::int64_t(p) * h * m > INT_MAX || first >= end || end > std::uint32_t(p) ||
        !layers || layers > 65535)
        throw std::invalid_argument("KVMem scoring dimension/domain/capture");
    shape(q, DType::BF16, {d, h, m});
    shape(k, DType::FP16, {d, kv, p});
    shape(w.logits, DType::FP32, {p, h * m});
    shape(w.row_stats, DType::FP32, {2, h * m});
    shape(w.row_status, DType::I32, {h * m});
    shape(w.scores, DType::FP32, {p});
    shape(w.status, DType::I32, {1});
    const Tensor* tensors[]{&q, &k, &w.logits, &w.row_stats, &w.row_status, &w.scores, &w.status};
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < i; ++j)
            disjoint(*tensors[i], *tensors[j]);
    detail::kvmem_score_launch(q, k, int(first), int(end), int(layers), reset, w, stream);
}
} // namespace ninfer::ops

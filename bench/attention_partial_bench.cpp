#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/gqa_attention_partial.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

using namespace ninfer;

namespace {
template <typename T> DeviceBuffer upload(const std::vector<T>& values) {
    DeviceBuffer buffer(values.size() * sizeof(T));
    buffer.copy_from_host(values.data(), buffer.bytes);
    return buffer;
}
struct Events {
    cudaEvent_t begin{}, end{};
    Events() {
        CUDA_CHECK(cudaEventCreate(&begin));
        CUDA_CHECK(cudaEventCreate(&end));
    }
    ~Events() {
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }
};
template <typename F> float measure(F operation, Events& events) {
    for (int i = 0; i < 5; ++i)
        operation();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(events.begin));
    for (int i = 0; i < 20; ++i)
        operation();
    CUDA_CHECK(cudaEventRecord(events.end));
    CUDA_CHECK(cudaEventSynchronize(events.end));
    float elapsed = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, events.begin, events.end));
    return elapsed / 20;
}
} // namespace

int main() try {
    constexpr int keys = 131072, pages = keys / 64;
    constexpr std::size_t code_bytes = std::size_t(256) * 4 * keys;
    std::vector<std::int8_t> codes(code_bytes);
    std::uint32_t random = 719;
    for (auto& value : codes) {
        random = random * 1664525u + 1013904223u;
        value = std::int8_t((random >> 16) % 255 - 127);
    }
    auto k = upload(codes);
    std::reverse(codes.begin(), codes.end());
    auto v = upload(codes);
    std::vector<std::uint16_t> scales(std::size_t(4) * 4 * keys);
    for (std::size_t i = 0; i < scales.size(); ++i)
        scales[i] = std::uint16_t(0x2400 + int(i % 3) * 1024);
    auto ks = upload(scales);
    std::reverse(scales.begin(), scales.end());
    auto vs = upload(scales);
    std::vector<std::int32_t> table(pages);
    std::iota(table.begin(), table.end(), 0);
    auto dt = upload(table);
    PagedKVLayerView stage;
    stage.k_pages = Tensor(k.p, DType::I8, {256, 64, 4, pages});
    stage.v_pages = Tensor(v.p, DType::I8, {256, 64, 4, pages});
    stage.k_scale_pages = Tensor(ks.p, DType::FP16, {4, 64, 4, pages});
    stage.v_scale_pages = Tensor(vs.p, DType::FP16, {4, 64, 4, pages});
    stage.block_table = Tensor(dt.p, DType::I32, {pages});
    stage.dtype = DType::I8;
    stage.head_dim = 256;
    stage.num_kv_heads = 4;
    stage.quant_group = 64;
    PagedKVLayerView resident;
    resident.dtype = DType::I8;
    resident.head_dim = 256;
    resident.num_kv_heads = 4;
    resident.quant_group = 64;
    std::vector<ops::AttentionPageAccess> accesses;
    for (int p = 0; p < pages; ++p)
        accesses.push_back({p, p});
    auto prefix = ops::attention_access_prefix(accesses, keys, 0, pages);
    auto da = upload(accesses), df = upload(prefix);
    Tensor ta(da.p, DType::I32, {2, pages}), tf(df.p, DType::I32, {pages + 1});
    std::cout << "INT8 layer bytes=" << k.bytes + v.bytes + ks.bytes + vs.bytes
              << " (264 MiB), keys=" << keys
              << "; compute only, all tiered pages in staging; 5 warmups / 20 repeats\n";
    std::cout << "T,route,splits,dense_ms,partial_ms,merge_finalize_ms,total_ms,ratio\n";
    Events events;
    for (int tokens : {1, 4, 64, 1024}) {
        std::vector<std::uint16_t> q(std::size_t(256) * 24 * tokens);
        for (auto& bits : q) {
            random = random * 1664525u + 1013904223u;
            const float x = float(int((random >> 16) % 129) - 64) / 256;
            std::uint32_t raw;
            std::memcpy(&raw, &x, 4);
            bits = std::uint16_t((raw + 0x7fff + ((raw >> 16) & 1)) >> 16);
        }
        auto dq = upload(q);
        std::vector<std::int32_t> positions(tokens);
        std::iota(positions.begin(), positions.end(), keys - tokens);
        auto dp = upload(positions);
        Tensor tq(dq.p, DType::BF16, {256, 24, tokens}), tp(dp.p, DType::I32, {tokens});
        DeviceBuffer result(q.size() * 2);
        Tensor out(result.p, DType::BF16, {256, 24, tokens});
        const ops::GqaExecutionEnvelope envelope{keys, keys};
        const auto scratch_bytes =
            ops::gqa_attention_workspace_capacity_bytes(24, DType::I8, envelope, 1, tokens, tokens);
        DeviceBuffer scratch(std::max<std::size_t>(256, scratch_bytes));
        WorkspaceArena workspace({scratch.p, scratch.bytes});
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto dense = [&] {
            ops::gqa_attention_cached(tq, tp, 0.0625f, stage, envelope, workspace, out, nullptr);
        };
        const float dense_ms = measure(dense, events);
        for (int splits : {32, 64, 128, 256}) {
            DeviceBuffer po(q.size() * splits * 4), pm(std::size_t(24) * tokens * splits * 4),
                pl(pm.bytes);
            DeviceBuffer mo(q.size() * 4), mm(std::size_t(24) * tokens * 4), ml(mm.bytes);
            ops::AttentionPartial partial{Tensor(po.p, DType::FP32, {256, 24, tokens, splits}),
                                          Tensor(pm.p, DType::FP32, {24, tokens, splits}),
                                          Tensor(pl.p, DType::FP32, {24, tokens, splits})};
            ops::AttentionPartial merged{Tensor(mo.p, DType::FP32, {256, 24, tokens, 1}),
                                         Tensor(mm.p, DType::FP32, {24, tokens, 1}),
                                         Tensor(ml.p, DType::FP32, {24, tokens, 1})};
            const auto invoke = tokens <= 16 ? ops::gqa_attention_partial_decode
                                             : ops::gqa_attention_partial_prefill;
            const auto compute = [&] {
                invoke(tq, tp, 0.0625f, resident, stage, ta, tf, keys, splits, partial, nullptr);
            };
            const auto merge = [&] {
                ops::attention_partial_lse_merge(partial, merged, nullptr);
                ops::attention_partial_finalize(merged, out, nullptr);
            };
            const auto all = [&] {
                compute();
                merge();
            };
            const float partial_ms = measure(compute, events), merge_ms = measure(merge, events),
                        total_ms = measure(all, events);
            std::cout << tokens << ',' << (tokens <= 16 ? "decode" : "prefill") << ',' << splits
                      << ',' << dense_ms << ',' << partial_ms << ',' << merge_ms << ',' << total_ms
                      << ',' << total_ms / dense_ms << '\n';
        }
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}

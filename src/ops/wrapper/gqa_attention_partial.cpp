#include "ninfer/ops/gqa_attention_partial.h"
#include "ops/launcher/gqa_attention_partial.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ninfer::ops {
namespace {
constexpr std::uint32_t maximum_frontier = 1048576;

void shape(const Tensor& t, DType dtype, std::initializer_list<int> dimensions) {
    int expected[4]{1, 1, 1, 1};
    std::copy(dimensions.begin(), dimensions.end(), expected);
    if (t.dtype != dtype || !t.is_contiguous() ||
        !std::equal(std::begin(expected), std::end(expected), t.ne) || (t.numel() > 0 && !t.data))
        throw std::invalid_argument("partial attention tensor shape/dtype/storage");
}

bool overlap(const Tensor& a, const Tensor& b) {
    if (!a.bytes() || !b.bytes()) return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a.data),
               y = reinterpret_cast<std::uintptr_t>(b.data);
    return x <= y ? y - x < a.bytes() : x - y < b.bytes();
}

void disjoint(const Tensor& a, const Tensor& b) {
    if (overlap(a, b)) throw std::invalid_argument("partial attention storage must be disjoint");
}

int cache_pages(const PagedKVLayerView& c) {
    const bool bf16 = c.dtype == DType::BF16;
    const bool packed =
        c.packed_k && c.packed_v && c.rotate_k && c.rotate_v && c.e8_lattice && !c.e8_root;
    const bool plain =
        !c.packed_k && !c.packed_v && !c.rotate_k && !c.rotate_v && !c.e8_lattice && !c.e8_root;
    if ((!bf16 && c.dtype != DType::I8) || c.head_dim != 256 || c.num_kv_heads != 4 ||
        (bf16 && (!plain || c.quant_group)) ||
        (!bf16 && (!plain && !packed || c.quant_group != 64)))
        throw std::invalid_argument("partial attention supports BF16, INT8 and rk4v4-e8 only");
    if (!c.k_pages.data && !c.v_pages.data) {
        if (c.k_scale_pages.data || c.v_scale_pages.data)
            throw std::invalid_argument("empty cache has scale storage");
        return 0;
    }
    const int pages = c.k_pages.ne[3];
    if (pages < 0) throw std::invalid_argument("negative physical page capacity");
    const int dimension  = packed ? 128 : 256;
    const auto code_type = bf16 ? DType::BF16 : (packed ? DType::U8 : DType::I8);
    shape(c.k_pages, code_type, {dimension, 64, 4, pages});
    shape(c.v_pages, code_type, {dimension, 64, 4, pages});
    if (!bf16) {
        shape(c.k_scale_pages, DType::FP16, {4, 64, 4, pages});
        shape(c.v_scale_pages, DType::FP16, {4, 64, 4, pages});
    }
    return pages;
}

void parts_shape(const AttentionPartial& p, int tokens, int parts) {
    if (tokens <= 0 || tokens > 655350 || parts < 1 || parts > 512 ||
        std::int64_t(24) * tokens * parts * 4 > 2147483647)
        throw std::invalid_argument("partial attention tokens/part capacity");
    shape(p.o, DType::FP32, {256, 24, tokens, parts});
    shape(p.m, DType::FP32, {24, tokens, parts});
    shape(p.l, DType::FP32, {24, tokens, parts});
    disjoint(p.o, p.m);
    disjoint(p.o, p.l);
    disjoint(p.m, p.l);
}

void partial(bool prefill, const Tensor& q, const Tensor& pos, float scale,
             const PagedKVLayerView& resident, const PagedKVLayerView& staging, const Tensor& pages,
             const Tensor& prefix, std::uint32_t frontier, int splits, AttentionPartial& out,
             cudaStream_t stream) {
    const int tokens = q.ne[2];
    if (!frontier || frontier > maximum_frontier || tokens < 1 || (!prefill && tokens > 16) ||
        !std::isfinite(scale) || scale <= 0)
        throw std::invalid_argument("partial attention execution domain");
    shape(q, DType::BF16, {256, 24, tokens});
    shape(pos, DType::I32, {tokens});
    const int a = cache_pages(resident), b = cache_pages(staging);
    if ((!a && !b) || resident.dtype != staging.dtype || resident.packed_k != staging.packed_k)
        throw std::invalid_argument("partial attention resident/staging profile mismatch");
    const int page_count = pages.data ? pages.ne[1] : 0;
    if (page_count < 0 || page_count > int((frontier + 63) / 64))
        throw std::invalid_argument("partial attention access-list capacity");
    if (page_count) shape(pages, DType::I32, {2, page_count});
    shape(prefix, DType::I32, {page_count + 1});
    parts_shape(out, tokens, splits);
    for (const auto* output : {&out.o, &out.m, &out.l}) {
        for (const auto* input :
             {&q, &pos, &pages, &prefix, &resident.k_pages, &resident.v_pages,
              &resident.k_scale_pages, &resident.v_scale_pages, &staging.k_pages, &staging.v_pages,
              &staging.k_scale_pages, &staging.v_scale_pages})
            disjoint(*output, *input);
    }
    detail::gqa_partial_launch(prefill, q, pos, scale, resident, staging, pages, prefix, frontier,
                               splits, out, stream);
}
} // namespace

std::vector<std::int32_t> attention_access_prefix(std::span<const AttentionPageAccess> pages,
                                                  std::uint32_t frontier, std::uint32_t resident,
                                                  std::uint32_t staging) {
    if (frontier > maximum_frontier)
        throw std::invalid_argument("access frontier exceeds attention domain");
    std::vector<std::int32_t> prefix(1, 0);
    int previous = -1;
    for (auto page : pages) {
        if (page.logical_page <= previous || page.logical_page < 0 ||
            std::uint64_t(page.logical_page) * 64 >= frontier || page.physical_page < 0 ||
            std::uint64_t(page.physical_page) >= std::uint64_t(resident) + staging)
            throw std::invalid_argument("access page ordering, frontier or physical capacity");
        previous = page.logical_page;
        prefix.push_back(prefix.back() +
                         int(std::min<std::uint32_t>(64, frontier - page.logical_page * 64)));
    }
    return prefix;
}

void gqa_attention_partial_prefill(const Tensor& q, const Tensor& pos, float scale,
                                   const PagedKVLayerView& resident,
                                   const PagedKVLayerView& staging, const Tensor& pages,
                                   const Tensor& prefix, std::uint32_t frontier, int splits,
                                   AttentionPartial& out, cudaStream_t stream) {
    partial(true, q, pos, scale, resident, staging, pages, prefix, frontier, splits, out, stream);
}

void gqa_attention_partial_decode(const Tensor& q, const Tensor& pos, float scale,
                                  const PagedKVLayerView& resident, const PagedKVLayerView& staging,
                                  const Tensor& pages, const Tensor& prefix, std::uint32_t frontier,
                                  int splits, AttentionPartial& out, cudaStream_t stream) {
    partial(false, q, pos, scale, resident, staging, pages, prefix, frontier, splits, out, stream);
}

void attention_partial_lse_merge(const AttentionPartial& input, AttentionPartial& output,
                                 cudaStream_t stream) {
    parts_shape(input, input.o.ne[2], input.o.ne[3]);
    parts_shape(output, input.o.ne[2], 1);
    for (const auto* a : {&input.o, &input.m, &input.l})
        for (const auto* b : {&output.o, &output.m, &output.l}) disjoint(*a, *b);
    detail::partial_lse_merge_launch(input, output, stream);
}

void attention_partial_lse_accumulate(const AttentionPartial& input, AttentionPartial& state,
                                      bool reset, cudaStream_t stream, Tensor* final_output) {
    parts_shape(input, input.o.ne[2], input.o.ne[3]);
    parts_shape(state, input.o.ne[2], 1);
    for (const auto* a : {&input.o, &input.m, &input.l})
        for (const auto* b : {&state.o, &state.m, &state.l}) disjoint(*a, *b);
    if (final_output) {
        shape(*final_output, DType::BF16, {256, 24, input.o.ne[2]});
        for (const Tensor* p : std::initializer_list<const Tensor*>{&input.o, &input.m, &input.l,
                                                                    &state.o, &state.m, &state.l})
            disjoint(*p, *final_output);
    }
    detail::partial_lse_accumulate_launch(input, state, reset, stream, final_output);
}

std::size_t attention_partial_workspace_bytes(int tokens, int splits) {
    if (tokens < 1 || tokens > 655350 || splits < 1 || splits > 512 ||
        std::int64_t(96) * tokens * splits > 2147483647)
        throw std::invalid_argument("partial attention workspace domain");
    return std::size_t(258) * 24 * tokens * (splits + 1) * sizeof(float);
}

AttentionPartialWorkspacePlan plan_attention_partial_workspace(int max_tokens, int preferred_splits,
                                                               std::size_t byte_budget) {
    const auto one_split = attention_partial_workspace_bytes(max_tokens, 1);
    if (preferred_splits < 1 || preferred_splits > 512)
        throw std::invalid_argument("partial preferred split domain");
    if (byte_budget < one_split)
        throw std::invalid_argument("partial budget cannot fit one split plus state");
    const auto bytes_per_part = one_split / 2;
    const auto budget_splits  = byte_budget / bytes_per_part - 1;
    const auto launch_splits  = std::size_t(2147483647 / (std::int64_t(96) * max_tokens));
    const int splits = int(std::min({std::size_t(preferred_splits), budget_splits, launch_splits}));
    return {splits, attention_partial_workspace_bytes(max_tokens, splits)};
}

void attention_partial_finalize(const AttentionPartial& input, Tensor& output,
                                cudaStream_t stream) {
    parts_shape(input, input.o.ne[2], 1);
    shape(output, DType::BF16, {256, 24, input.o.ne[2]});
    for (const auto* p : {&input.o, &input.m, &input.l}) disjoint(*p, output);
    detail::partial_finalize_launch(input, output, stream);
}
} // namespace ninfer::ops

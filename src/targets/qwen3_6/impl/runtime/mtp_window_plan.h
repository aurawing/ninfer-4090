#pragma once
#include "core/layout.h"
#include "core/kvmem/planning_error.h"
#include <ninfer/types.h>
#include <algorithm>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail {
struct MtpWindowPlan {
    std::uint32_t capacity = 0, physical_pages = 0, sink_pages = 0, recent_pages = 0,
                  guard_pages = 0;
    TensorRegion block_table, access, prefix, page_tags;
    TensorRegion scratch_o, scratch_m, scratch_l, state_o, state_m, state_l;
    std::size_t bytes = 0;
};
inline MtpWindowPlan plan_mtp_window(std::uint32_t capacity, std::uint32_t chunk,
                                     const TieredKVOptions& options) {
    const auto pages = [](std::uint32_t n) { return n / 64 + (n % 64 != 0); };
    if (!capacity || !chunk || !options.mtp_window_tokens || options.mtp_window_tokens % 64)
        throw std::invalid_argument("MTP window must be a positive multiple of 64 tokens");
    MtpWindowPlan p;
    p.capacity       = capacity;
    p.physical_pages = std::min(pages(capacity), pages(options.mtp_window_tokens));
    p.sink_pages     = std::min(pages(options.sink_tokens), p.physical_pages);
    p.guard_pages    = pages(capacity) > p.physical_pages ? 1 : 0;
    if (p.guard_pages && p.sink_pages + p.guard_pages >= p.physical_pages)
        throw std::invalid_argument(
            "MTP window cannot fit sink, recent pages and provisional guard");
    p.recent_pages = p.physical_pages - p.sink_pages - p.guard_pages;
    if (p.guard_pages && pages(std::min(chunk, capacity)) + 1 > p.recent_pages)
        throw kvmem::TieredPrefillCapacityError(
            "MTP window cannot fit a prefill chunk and its partial boundary page");
    LayoutBuilder b;
    p.block_table =
        b.add_tensor(DType::I32, {int(pages(capacity))}, 256, "MTP original logical table");
    const int count = int(p.sink_pages + p.recent_pages);
    p.page_tags =
        b.add_tensor(DType::I32, {int(p.physical_pages)}, 256, "MTP physical original-page tags");
    p.access    = b.add_tensor(DType::I32, {2, count}, 256, "MTP sink/recent access");
    p.prefix    = b.add_tensor(DType::I32, {count + 1}, 256, "MTP visible prefix");
    p.scratch_o = b.add_tensor(DType::FP32, {256, 24, 1, 64}, 256, "MTP partial O");
    p.scratch_m = b.add_tensor(DType::FP32, {24, 1, 64}, 256, "MTP partial m");
    p.scratch_l = b.add_tensor(DType::FP32, {24, 1, 64}, 256, "MTP partial l");
    p.state_o   = b.add_tensor(DType::FP32, {256, 24, 1}, 256, "MTP merged O");
    p.state_m   = b.add_tensor(DType::FP32, {24, 1}, 256, "MTP merged m");
    p.state_l   = b.add_tensor(DType::FP32, {24, 1}, 256, "MTP merged l");
    p.bytes     = b.finish(256);
    return p;
}
} // namespace ninfer::targets::qwen3_6::detail

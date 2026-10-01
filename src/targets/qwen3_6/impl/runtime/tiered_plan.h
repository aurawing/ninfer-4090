#pragma once

#include "core/kvmem/host_kv_transfer.h"
#include <ninfer/ops/gqa_attention_partial.h>
#include <ninfer/types.h>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

struct TieredPageLimits {
    std::uint32_t minimum;
    std::uint32_t maximum;
};

inline TieredPageLimits tiered_page_limits(std::uint32_t logical_tokens,
                                           std::uint32_t prefill_chunk,
                                           const TieredKVOptions& options) {
    if (!logical_tokens || !prefill_chunk || !options.view_tokens) {
        throw std::invalid_argument("tiered context, prefill and view must be positive");
    }
    const auto pages     = [](std::uint32_t tokens) { return tokens / 64U + (tokens % 64U != 0); };
    const auto logical   = pages(logical_tokens);
    const auto minimum64 = std::uint64_t(pages(options.sink_tokens)) +
                           pages(std::min(prefill_chunk, logical_tokens)) + 1;
    const auto minimum   = static_cast<std::uint32_t>(std::min<std::uint64_t>(logical, minimum64));
    const auto maximum   = std::min(logical, pages(options.view_tokens));
    if (minimum > maximum) {
        throw std::invalid_argument(
            "tiered view cannot fit sink, prefill chunk and replacement page");
    }
    return {minimum, maximum};
}

struct TieredRuntimePlan {
    KvMode mode                         = KvMode::TieredExact;
    bool shadow_validate                = false;
    bool measure_transfer_waits         = false;
    std::uint32_t logical_tokens        = 0;
    std::uint32_t view_pages            = 0;
    std::uint32_t sink_pages            = 0;
    std::uint32_t max_query_tokens      = 0;
    kvmem::HostArchiveMode archive_mode = kvmem::HostArchiveMode::Auto;
    kvmem::HostKVArchiveLayout archive;
    kvmem::HostKVStagingPlan staging;
    ops::AttentionPartialWorkspacePlan partial{};
    LayoutRegion staging_region;
    TensorRegion block_table;
    std::array<TensorRegion, 3> access;
    std::array<TensorRegion, 3> prefix;
    TensorRegion scratch_o, scratch_m, scratch_l;
    TensorRegion state_o, state_m, state_l;
    TensorRegion shadow_output;
    std::vector<LayoutRegion> staging_plane_regions;
    std::uint32_t maximum_stream_pages = 0;
    std::size_t bytes                  = 0;
};

// The 262K RK4 shadow run only collects dense/tiered differences. Its verdict
// requires the separately captured full-block CPU FP64 reference, not exit 0.
inline bool tiered_shadow_fp64_required(std::uint32_t logical_tokens,
                                        const PagedKVLayerView& view) noexcept {
    return logical_tokens == 262144 && view.dtype == DType::I8 && view.packed_k && view.packed_v &&
           view.rotate_k && view.rotate_v && view.e8_lattice && !view.e8_root;
}

// A fixed staging allocation can cover fewer streamed pages than the automatic
// allocation at the sink/chunk floor. Raise the resident floor before building
// the capacity curve, using the actual largest layer's complete plane bytes.
inline TieredPageLimits tiered_staging_page_limits(const kvmem::HostKVArchiveLayout& archive,
                                                   TieredPageLimits limits,
                                                   std::size_t staging_capacity_bytes) {
    if (!staging_capacity_bytes) { return limits; }
    if (!archive.logical_pages || archive.layers.empty() || !limits.minimum ||
        limits.minimum > limits.maximum || limits.maximum > archive.logical_pages ||
        staging_capacity_bytes < kvmem::kHostKVTransferTileBytes) {
        throw std::invalid_argument(
            "fixed tiered staging cannot fit its transfer tile or page limits");
    }
    std::size_t maximum_layer_page_bytes = 0;
    for (const auto& layer : archive.layers) {
        std::size_t layer_page_bytes = 0;
        for (const auto& plane : layer) {
            if (plane.page_bytes > std::numeric_limits<std::size_t>::max() - layer_page_bytes) {
                throw std::overflow_error("tiered layer page byte size overflow");
            }
            layer_page_bytes += plane.page_bytes;
        }
        maximum_layer_page_bytes = std::max(maximum_layer_page_bytes, layer_page_bytes);
    }
    if (!maximum_layer_page_bytes) {
        throw std::invalid_argument("fixed tiered staging requires nonempty layer storage");
    }
    const auto stream_pages = std::min<std::size_t>(
        archive.logical_pages,
        (staging_capacity_bytes - kvmem::kHostKVTransferTileBytes) / maximum_layer_page_bytes);
    limits.minimum =
        std::max(limits.minimum, archive.logical_pages - static_cast<std::uint32_t>(stream_pages));
    if (limits.minimum > limits.maximum) {
        throw std::invalid_argument(
            "fixed tiered staging requires a resident view larger than view_tokens");
    }
    return limits;
}

// All regions address a dedicated backing starting at zero; no transient workspace aliasing.
inline TieredRuntimePlan plan_tiered_runtime(const PagedKVPoolLayout& main_pool,
                                             std::uint32_t logical_tokens, std::uint32_t view_pages,
                                             std::uint32_t max_query_tokens,
                                             const TieredKVOptions& options, bool shadow_validate,
                                             bool measure_transfer_waits) {
    if (!view_pages || !max_query_tokens ||
        (main_pool.planes.size() != 32 && main_pool.planes.size() != 64) ||
        main_pool.spec.plane_order != PagedKVPlaneOrder::PageMajor) {
        throw std::invalid_argument("tiered runtime requires the 16-layer Qwen27B Main pool");
    }
    const auto planes_per_layer = main_pool.planes.size() / 16;
    for (std::size_t i = 0; i < main_pool.planes.size(); ++i) {
        const auto& spec = main_pool.planes[i].spec;
        const bool codes = i % planes_per_layer < 2;
        const bool valid = planes_per_layer == 2
                               ? spec.dtype == DType::BF16 && spec.leading_extent == 256
                           : codes ? ((spec.dtype == DType::I8 && spec.leading_extent == 256) ||
                                      (spec.dtype == DType::U8 && spec.leading_extent == 128))
                                   : spec.dtype == DType::FP16 && spec.leading_extent == 4;
        if (!valid || spec.head_extent != 4) {
            throw std::invalid_argument(
                "tiered runtime requires BF16/INT8/rk4 24Q/4KV/256D planes");
        }
    }
    TieredRuntimePlan out;
    out.shadow_validate        = shadow_validate;
    out.measure_transfer_waits = measure_transfer_waits;
    out.logical_tokens         = logical_tokens;
    out.view_pages             = view_pages;
    out.max_query_tokens       = max_query_tokens;
    switch (options.host_archive) {
    case HostKVArchiveMode::Auto:
        out.archive_mode = kvmem::HostArchiveMode::Auto;
        break;
    case HostKVArchiveMode::Pinned:
        out.archive_mode = kvmem::HostArchiveMode::Pinned;
        break;
    case HostKVArchiveMode::Pageable:
        out.archive_mode = kvmem::HostArchiveMode::Pageable;
        break;
    default:
        throw std::invalid_argument("unknown host KV archive mode");
    }
    out.archive = kvmem::plan_host_kv_archive(main_pool, planes_per_layer, logical_tokens);
    if (view_pages > out.archive.logical_pages) {
        throw std::invalid_argument("tiered resident view exceeds logical capacity");
    }
    out.sink_pages =
        std::min(view_pages, options.sink_tokens / 64U + (options.sink_tokens % 64U != 0));
    out.maximum_stream_pages = out.archive.logical_pages - view_pages;
    out.staging =
        kvmem::plan_host_kv_staging(out.archive, view_pages, options.staging_capacity_bytes);
    out.partial = ops::plan_attention_partial_workspace(static_cast<int>(max_query_tokens), 1,
                                                        options.partial_budget_bytes);
    // Decode <=4 supports S64. Scratch and state planes are independently maximized.
    const auto scratch_tokens   = std::max<std::uint64_t>(max_query_tokens, 4ULL * 64);
    const auto state_tokens     = std::max<std::uint64_t>(max_query_tokens, 4);
    const auto required_partial = (scratch_tokens + state_tokens) * 258ULL * 24 * sizeof(float);
    if (required_partial > options.partial_budget_bytes) {
        throw std::invalid_argument(
            "tiered partial budget cannot fit prefill and four-query S64 decode");
    }
    out.partial.bytes    = required_partial;
    const auto dimension = [](std::uint64_t elements) {
        if (elements > std::numeric_limits<std::int32_t>::max()) {
            throw std::overflow_error("tiered tensor extent exceeds int32");
        }
        return static_cast<std::int32_t>(elements);
    };
    LayoutBuilder builder;
    // Put the variable-sized staging last so all fixed regions have stable offsets.
    out.block_table = builder.add_tensor(DType::I32, {dimension(out.archive.logical_pages)}, 256,
                                         "tiered block table");
    for (std::size_t i = 0; i < out.access.size(); ++i) {
        out.access[i] = builder.add_tensor(DType::I32, {2, dimension(out.archive.logical_pages)},
                                           256, "tiered access list");
        out.prefix[i] = builder.add_tensor(
            DType::I32, {dimension(out.archive.logical_pages + 1ULL)}, 256, "tiered access prefix");
    }
    const auto plane = [&](std::uint64_t tokens, std::uint64_t rows, const char* label) {
        return builder.add_tensor(DType::FP32, {dimension(tokens * rows)}, 256, label);
    };
    out.scratch_o = plane(scratch_tokens, 256 * 24, "tiered scratch O");
    out.scratch_m = plane(scratch_tokens, 24, "tiered scratch M");
    out.scratch_l = plane(scratch_tokens, 24, "tiered scratch L");
    out.state_o   = plane(state_tokens, 256 * 24, "tiered state O");
    out.state_m   = plane(state_tokens, 24, "tiered state M");
    out.state_l   = plane(state_tokens, 24, "tiered state L");
    if (shadow_validate) {
        out.shadow_output = builder.add_tensor(DType::BF16, {256, 24, dimension(max_query_tokens)},
                                               256, "tiered shadow output");
    }
    out.staging_region       = builder.add(out.staging.capacity_bytes, 256, "tiered staging");
    std::size_t plane_offset = out.staging_region.offset;
    for (const auto& spec : out.archive.layers.front()) {
        const auto bytes = spec.page_bytes * out.maximum_stream_pages;
        out.staging_plane_regions.push_back({plane_offset, bytes, 256});
        plane_offset += bytes;
    }
    out.bytes = builder.finish(256, "tiered runtime backing");
    return out;
}

} // namespace ninfer::targets::qwen3_6::detail

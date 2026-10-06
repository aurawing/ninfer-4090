#pragma once
#include "core/kvmem/host_meank_index.h"
#include "core/kvmem/query_capture.h"
namespace ninfer::targets::qwen3_6::detail {
struct SparseCaptureResources {
    kvmem::MeanKResources mean;
    kvmem::QueryCaptureResources query;
    std::size_t device_bytes{}, pinned_bytes{}, pageable_query_bytes{};
    bool operator==(const SparseCaptureResources&) const = default;
};
inline SparseCaptureResources plan_sparse_capture_resources(std::uint32_t context, std::uint32_t chunk,
                                                     std::uint32_t queries = 16) {
    SparseCaptureResources out;
    out.mean = kvmem::plan_meank_resources(context, 16, 256, 4, chunk);
    out.query = kvmem::plan_query_capture_resources(queries);
    out.device_bytes = out.mean.device_bytes + out.query.device_bytes;
    out.pinned_bytes = out.mean.host_patch_bytes + out.query.pinned_bytes;
    out.pageable_query_bytes = out.query.pageable_bytes;
    return out;
}
} // namespace ninfer::targets::qwen3_6::detail

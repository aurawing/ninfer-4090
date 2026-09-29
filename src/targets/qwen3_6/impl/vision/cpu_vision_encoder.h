#pragma once

#include "cpu_vision_bridge.h"
#include "runtime/contract/cancellation.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6 {

using CpuVisionCancelled = runtime::RequestCancelled;

inline constexpr std::size_t cpu_vision_cache_metadata_bytes = 256 * 1024;

struct CpuVisionEncodeResult {
    std::vector<std::uint16_t> embeddings;
    bool cache_hit = false;
};

class CpuVisionEncoder {
public:
    virtual ~CpuVisionEncoder() = default;
    // Validates geometry, numerical inputs, and total host-memory admission without executing.
    virtual void validate(const CpuVisionInput& input) const = 0;
    // Contiguous BF16 [merged_tokens,5120]. Failure/cancellation never returns partial embeddings.
    virtual std::vector<std::uint16_t> encode(const CpuVisionInput& input,
        const std::function<bool()>& cancelled = {}) = 0;
    virtual CpuVisionEncodeResult encode_with_status(const CpuVisionInput& input,
        const std::function<bool()>& cancelled = {}) {
        return {encode(input, cancelled), false};
    }
    virtual std::size_t cache_capacity_bytes() const noexcept { return 0; }
    virtual std::uint64_t identity_hash() const noexcept = 0;
    virtual std::size_t weight_bytes() const noexcept = 0;
};

bool cpu_vision_backend_available() noexcept;
std::shared_ptr<CpuVisionEncoder> make_gguf_cpu_vision_encoder(const std::string& path,
    std::uint32_t threads, std::size_t memory_budget_bytes);
// The caller reserves capacity plus cpu_vision_cache_metadata_bytes from host-memory admission.
std::shared_ptr<CpuVisionEncoder> make_cached_cpu_vision_encoder(
    std::shared_ptr<CpuVisionEncoder> inner, std::size_t cache_capacity_bytes);

} // namespace ninfer::targets::qwen3_6

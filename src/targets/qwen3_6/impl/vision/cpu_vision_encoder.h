#pragma once

#include "cpu_vision_bridge.h"
#include "runtime/contract/cancellation.h"

#include <functional>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
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
    // Validates input and admission again when called directly or on a cache miss. Returns
    // contiguous BF16 [merged_tokens,5120]; failure/cancellation never returns partial data.
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
std::shared_ptr<CpuVisionEncoder> make_gguf_cuda_vision_encoder(const std::string& path,
    std::size_t host_memory_budget_bytes);

enum class EmbeddedVisionDtype { F32, BF16 };
struct EmbeddedVisionTensor {
    std::string name;
    std::array<std::int64_t, 4> dimensions{1, 1, 1, 1};
    std::uint32_t rank = 0;
    EmbeddedVisionDtype dtype = EmbeddedVisionDtype::F32;
    // Called synchronously exactly once per tensor, after GGML allocates its final CPU buffer.
    std::function<void(std::span<std::byte>)> fill;
};

std::shared_ptr<CpuVisionEncoder> make_embedded_cpu_vision_encoder(
    std::span<const EmbeddedVisionTensor> tensors, std::uint32_t threads,
    std::size_t memory_budget_bytes, std::uint64_t content_hash);
// The caller reserves capacity plus cpu_vision_cache_metadata_bytes from host-memory admission.
std::shared_ptr<CpuVisionEncoder> make_cached_cpu_vision_encoder(
    std::shared_ptr<CpuVisionEncoder> inner, std::size_t cache_capacity_bytes);

} // namespace ninfer::targets::qwen3_6

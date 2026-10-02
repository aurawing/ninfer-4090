#pragma once

#include "targets/qwen3_6/impl/runtime/tiered_plan.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>

#include <memory>
#include <span>

namespace ninfer::targets::qwen3_6::detail {

// Serialized C=1 owner. Main cache, backing and compute stream outlive this object.
// The backing is the exact, fixed load-time TieredRuntimePlan allocation.
class TieredContext {
public:
    TieredContext(const TieredRuntimePlan& plan, PagedKVCache& main, DeviceSpan backing,
                  cudaStream_t compute_stream);
    ~TieredContext();
    TieredContext(const TieredContext&)            = delete;
    TieredContext& operator=(const TieredContext&) = delete;

    [[nodiscard]] bool shadow() const noexcept;
    [[nodiscard]] std::uint32_t view_pages() const noexcept;
    [[nodiscard]] std::uint32_t frontier() const noexcept;
    void bind_pages(std::span<const std::int32_t> lease_ids);
    enum class ExecutionPhase { Prefill, Decode };
    void begin_block(std::uint32_t base, std::uint32_t count, cudaStream_t stream,
                     ExecutionPhase phase = ExecutionPhase::Prefill);
    [[nodiscard]] PagedKVLayerView resident_layer(std::uint32_t layer) const;
    void attention(std::uint32_t layer, const Tensor& q, const Tensor& k, const Tensor& v,
                   const Tensor& positions, float scale, Tensor& out, cudaStream_t stream);
    // Dense A1 already appended Main; preserves dense_out and compares a separate output.
    void shadow_attention(std::uint32_t layer, const Tensor& q, const Tensor& positions,
                          float scale, const Tensor& dense_out, cudaStream_t stream);
    void trim(std::uint32_t frontier, cudaStream_t stream);
    void reset(cudaStream_t stream);
    void drain();
    void log_stats();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::targets::qwen3_6::detail

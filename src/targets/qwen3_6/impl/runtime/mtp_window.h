#pragma once
#include "targets/qwen3_6/impl/runtime/mtp_window_plan.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <vector>
#include <span>
#include <memory>
namespace ninfer::targets::qwen3_6::detail {
struct MtpSnapshotLifetime {
    std::uint32_t frontier = 0;
    bool valid = true;
};
struct MtpWindowSnapshot {
    std::uint64_t bundle_identity = 0, generation = 0;
    std::uint32_t frontier = 0;
    std::vector<std::int32_t> page_tags;
    std::shared_ptr<MtpSnapshotLifetime> lifetime;
};
// C=1 eager owner. Every region is fixed at loading, with no host KV archive.
class MtpWindow {
  public:
    MtpWindow(const MtpWindowPlan&, PagedKVCache&, DeviceSpan, cudaStream_t);
    void bind_pages(std::span<const std::int32_t>, cudaStream_t);
    void append(const Tensor& k, const Tensor& v, const Tensor& positions,
                const Tensor& valid_columns, cudaStream_t);
    void attention(const Tensor& q, const Tensor& positions, const Tensor& valid_columns,
                   float scale, Tensor& out, cudaStream_t, const Tensor* k = nullptr,
                   const Tensor* v = nullptr);
    // Only one page can be crossed by <=31 provisional tokens. Its old bytes
    // remain in a guard INSIDE the configured physical budget until trim.
    void begin_transaction(std::uint32_t base, std::uint32_t extent, cudaStream_t);
    void trim(std::uint32_t frontier, cudaStream_t);
    void reset(cudaStream_t);
    [[nodiscard]] MtpWindowSnapshot capture(std::uint32_t frontier, cudaStream_t);
    void restore(const MtpWindowSnapshot&, std::uint32_t frontier, cudaStream_t);
    [[nodiscard]] Tensor page_tags() const { return plan_.page_tags.bind(backing_); }
    [[nodiscard]] std::uint32_t physical_pages() const noexcept { return plan_.physical_pages; }
    [[nodiscard]] const MtpWindowPlan& plan() const noexcept { return plan_; }

  private:
    [[nodiscard]] PagedKVLayerView layer() const;
    void invalidate_captures(std::uint32_t frontier);
    std::uint64_t bundle_identity_ = 0, generation_ = 0;
    std::vector<std::weak_ptr<MtpSnapshotLifetime>> captures_;
    MtpWindowPlan plan_;
    PagedKVCache& cache_;
    DeviceSpan backing_;
    std::vector<std::int32_t> leases_, mapping_;
    std::uint32_t backup_new_page_ = 0;
    bool backup_live_              = false;
};
} // namespace ninfer::targets::qwen3_6::detail

#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::kvmem {
struct QueryCaptureResources {
    std::uint32_t layers = 16, dimension = 256, heads = 24, capacity = 16;
    std::size_t slot_bytes{}, pageable_bytes{}, device_bytes{}, pinned_bytes{};
    bool operator==(const QueryCaptureResources&) const = default;
};
[[nodiscard]] QueryCaptureResources plan_query_capture_resources(std::uint32_t capacity = 16);
struct QueryProvenance {
    std::uint64_t bundle_identity{}, lineage{}, capture_epoch{};
    std::uint32_t covered_frontier{};
    std::array<std::uint8_t, 32> source_prefix{};
    std::vector<std::uint32_t> ordinals;
};
class QueryCaptureStorage;
class QueryCapture {
  public:
    [[nodiscard]] std::uint64_t id() const noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const QueryProvenance& provenance() const noexcept;
    [[nodiscard]] std::span<const std::uint16_t> layer(std::uint32_t layer) const;

  private:
    friend class QueryCapturePool;
    QueryProvenance provenance_;
    std::shared_ptr<QueryCaptureStorage> storage_;
    std::uint64_t id_{}, epoch_{};
    std::uint32_t slot_{};
};
using QueryCaptureHandle = std::shared_ptr<const QueryCapture>;
class QueryCaptureBorrow {
  public:
    QueryCaptureBorrow() = default;
    ~QueryCaptureBorrow();
    QueryCaptureBorrow(const QueryCaptureBorrow&) = delete;
    QueryCaptureBorrow& operator=(const QueryCaptureBorrow&) = delete;
    QueryCaptureBorrow(QueryCaptureBorrow&&) noexcept;
    QueryCaptureBorrow& operator=(QueryCaptureBorrow&&) noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] QueryCaptureHandle capture() const;
    void finish();

  private:
    friend class QueryCapturePool;
    struct State;
    std::shared_ptr<State> state_;
};
// Loading allocates exactly three pageable BF16 slots. Published handles are immutable.
// Score/DMA consumers borrow one occupied slot. Their completion callback drains
// the consumer before replacement/reset; no fourth capture is allocated.
class QueryCapturePool {
  public:
    explicit QueryCapturePool(QueryCaptureResources);
    ~QueryCapturePool();
    void begin(QueryProvenance);
    void complete_layer(std::uint32_t layer, std::span<const std::uint16_t> rows);
    [[nodiscard]] QueryCaptureHandle publish(); // all 16 layers required
    void discard();
    [[nodiscard]] QueryCaptureBorrow borrow(const QueryCaptureHandle&, std::function<void()> drain);
    void drain_borrows();
    void reset();
    [[nodiscard]] bool owns(const QueryCaptureHandle&) const noexcept;
    [[nodiscard]] std::uint32_t live_slots() const noexcept;
    [[nodiscard]] std::uint32_t high_watermark() const noexcept;

  private:
    std::shared_ptr<QueryCaptureStorage> storage_;
    std::shared_ptr<QueryCapture> pending_;
    std::array<bool, 16> layers_{};
    std::vector<std::shared_ptr<QueryCaptureBorrow::State>> borrows_;
};
[[nodiscard]] bool query_capture_matches(const QueryCaptureHandle&, const QueryProvenance& expected,
                                         std::uint32_t restored_frontier) noexcept;
struct ExactCaptureContinuation {
    std::uint32_t frontier{};
    std::uint64_t bundle_identity{}, lineage{};
    bool exact_source_prefix = false;
};
enum class QueryRecaptureKind { Reuse, ExactContinuation, ColdStart };
struct QueryRecapturePlan {
    QueryRecaptureKind kind{};
    std::uint32_t frontier{};
};
[[nodiscard]] QueryRecapturePlan plan_query_recapture(const QueryProvenance&,
                                                      const QueryCaptureHandle&,
                                                      std::uint32_t restored_frontier,
                                                      std::span<const ExactCaptureContinuation>);
} // namespace ninfer::kvmem

#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace ninfer::kvmem {
// Pure loading-time layout. Offsets refer to one caller-owned device backing;
// no CUDA allocation occurs here. Host patches are separately bounded staging
// for completed D2H means/sums; pageable transfer uses the existing archive ring.
struct MeanKResources {
    std::uint32_t max_context{}, layers{}, dimension{}, heads{}, max_chunk{}, pages{},
        patch_pages{};
    std::size_t index_bytes{}, layer_stride_bytes{}, row_elements{};
    std::size_t active_sum_offset{}, base_sum_offset{}, seed_sum_offset{}, output_sum_offset{},
        tail_offset{}, provisional_offset{}, means_offset{}, counts_offset{}, device_bytes{};
    std::size_t tail_bytes{}, provisional_bytes{}, sum_bytes{}, means_bytes{}, snapshot_bytes{},
        host_continuation_bytes{}, host_patch_bytes{};
    bool operator==(const MeanKResources&) const = default;
};
[[nodiscard]] MeanKResources plan_meank_resources(std::uint32_t max_context, std::uint32_t layers,
                                                  std::uint32_t dimension, std::uint32_t heads,
                                                  std::uint32_t max_chunk);
// Combined archive + index admission belongs to loading. This helper charges
// every index, host continuation and bounded completed-patch byte plus 4 GiB.
[[nodiscard]] std::uint64_t meank_host_admission_bytes(const MeanKResources&);

enum class MeanKTransactionKind { ExactPrefill, OrdinaryMain, SpeculativeMain, Restore, Trim };
struct MeanKTicket {
    std::uint64_t owner{}, generation{}, serial{};
    std::uint32_t first{}, count{};
    MeanKTransactionKind kind{};
    bool operator==(const MeanKTicket&) const = default;
};
class MeanKSnapshot;
using MeanKSnapshotHandle = std::shared_ptr<const MeanKSnapshot>;

// Serialized C=1 owner of a fixed [layer,max_page,H,D] FP16 index. Constructor
// is loading-only: actual_archive_pinned selects cacheable PinnedHostBuffer;
// otherwise pageable storage. A completion is proof supplied by a coordinator
// after its GPU/DMA event has completed. It must not be called at enqueue time.
// Loading must admit archive + meank_host_admission_bytes() before construction,
// with the 4 GiB headroom charged once, and check available headroom after all
// pinned allocation. Auto fallback is a combined archive/index loading decision.
// No CPU raw-K accumulation, CUDA launches or scoring are owned by this class.
class HostMeanKIndex {
  public:
    explicit HostMeanKIndex(MeanKResources resources, bool actual_archive_pinned);
    ~HostMeanKIndex();
    HostMeanKIndex(const HostMeanKIndex&) = delete;
    HostMeanKIndex& operator=(const HostMeanKIndex&) = delete;
    [[nodiscard]] const MeanKResources& resources() const noexcept;
    [[nodiscard]] bool pinned() const noexcept;
    [[nodiscard]] bool published() const noexcept;
    [[nodiscard]] std::uint32_t frontier() const noexcept;
    [[nodiscard]] std::uint32_t base_frontier() const noexcept;
    [[nodiscard]] std::uint64_t generation() const noexcept;
    // Rejects access while any layer is pending. Fixed layer stride never depends on F.
    [[nodiscard]] std::span<const std::uint16_t> layer_data(std::uint32_t layer) const;
    // Exact active partial sum for append/restore regeneration. Not a trim seed:
    // trimming must replay bounded tail from an independently saved exact base.
    [[nodiscard]] std::span<const float> prefix_sum(std::uint32_t layer) const;
    // Independent exact sum at base_frontier, preserved across partial appends.
    // Restore seeds it from owning snapshot; a page seal resets it to zero.
    [[nodiscard]] std::span<const float> base_sum(std::uint32_t layer) const;
    [[nodiscard]] MeanKTicket begin(MeanKTransactionKind kind, std::uint32_t actual_count);
    // Must run BEFORE any speculative completion/accumulation. Final application
    // accepted length can be smaller than target acceptance. Zero discards.
    [[nodiscard]] MeanKTicket accept(const MeanKTicket&, std::uint32_t accepted_count);
    // Means are touched pages only. Sum is exact final partial FP32, zero at seal.
    // Returns false for old/duplicate completion without mutation. All-layer
    // publication copies staged patches atomically at this serialized boundary.
    bool complete_layer(const MeanKTicket&, std::uint32_t layer,
                        std::span<const std::uint16_t> means, std::span<const float> sum);
    void discard(const MeanKTicket&);
    [[nodiscard]] MeanKSnapshotHandle capture(); // maximum two distinct live owning handles
    // Coordinator first drains consumers and DMA. Snapshot saves only bounded
    // exact prefix sums, not KV or raw K. Partial FP16 mean is regenerated on GPU
    // with count=0 before all-layer republish. Older raw tail is invalid at new base.
    [[nodiscard]] MeanKTicket restore(const MeanKSnapshotHandle&);
    // Only current bounded tail, at/after base, may be replayed without snapshot.
    // Complete this ticket with GPU-recomputed sums/means at exact new frontier.
    [[nodiscard]] MeanKTicket begin_trim(std::uint32_t exact_frontier);
    void reset(); // caller drains first; invalidates snapshots and all old completions
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ninfer::kvmem

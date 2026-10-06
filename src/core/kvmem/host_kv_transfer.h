#pragma once

#include "core/kvmem/host_kv_archive.h"
#include "core/kvmem/cuda_status.h"
#include <atomic>

namespace ninfer::kvmem {

inline constexpr std::size_t kHostKVTransferTileBytes = 64ULL * 1024 * 1024;
struct HostKVStagingPlan {
    std::uint32_t resident_pages = 0;
    std::size_t maximum_layer_stream_bytes = 0;
    std::size_t capacity_bytes = 0;
    std::size_t ticket_capacity = 0;
};
[[nodiscard]] HostKVStagingPlan plan_host_kv_staging(const HostKVArchiveLayout& layout,
                                                    std::uint32_t resident_pages,
                                                    std::size_t capacity_override = 0);

struct HostKVTransferTicket {
    std::size_t slot = 0;
    std::uint64_t sequence = 0;
    std::uint64_t archive_generation = 0;
};

// Deterministic test-only faults, disabled by default. Throw after completed DMA,
// or ask CUDA to reject a safe invalid-kind/invalid-flag submission on valid
// fixed resources. Recovery disables the one-shot injections.
struct HostKVTransferFaultInjection {
    std::uint64_t h2d_copy_after = 0;
    std::uint64_t d2h_plane_after = 0;
    bool fail_consumed_flags_allocation = false;
    bool fail_producer_borrowed_flags_allocation = false;
    std::uint64_t second_worker_start_failure_attempt = 0;
    std::uint64_t pause_before_h2d_copy = 0;
    std::atomic<bool>* h2d_paused = nullptr;
    std::atomic<bool>* resume_h2d = nullptr;
    std::uint64_t reject_h2d_submission_after = 0;
    std::uint64_t reject_d2h_submission_after = 0;
    std::uint64_t pause_after_d2h_plane = 0;
    std::atomic<bool>* d2h_paused = nullptr;
    std::atomic<bool>* resume_d2h = nullptr;
    // Internal Tiered test seam releases a paused final writeback at selection,
    // after actual exact Q/Mean capture completed, with no scheduling callback.
    std::atomic<bool>* selection_after_exact = nullptr;
    bool reject_consumer_wait = false;
    bool reject_consumer_release = false;
    bool reject_producer_record = false;
};
struct HostKVTransferFailureState {
    bool failed = false;
    bool archive_bytes_uncertain = false;
    bool unrecoverable = false;
    // First typed CUDA failure and first failed drain remain sticky even when
    // a subsequent drain succeeds. Execution/context errors forbid recovery.
    cudaError_t cuda_status = cudaSuccess;
    const char* cuda_operation = nullptr;
    cudaError_t drain_status = cudaSuccess;
    const char* drain_operation = nullptr;
    std::uint64_t completed_h2d_copies = 0;
    std::uint64_t completed_d2h_planes = 0;
};

// Serialized C=1 scheduling; two copy workers, independent nonblocking H2D/D2H streams.
// Archive and caller-owned workspace/consumer streams must outlive this owner.
// Each prefetch covers <=64 MiB; callers tile larger planes in logical order.
class HostKVTransferEngine {
public:
    HostKVTransferEngine(HostKVArchive& archive, DeviceSpan staging, HostKVStagingPlan plan);
    HostKVTransferEngine(HostKVArchive& archive, DeviceSpan staging, HostKVStagingPlan plan,
                         HostKVTransferFaultInjection fault);
    ~HostKVTransferEngine();
    HostKVTransferEngine(const HostKVTransferEngine&) = delete;
    HostKVTransferEngine& operator=(const HostKVTransferEngine&) = delete;

    [[nodiscard]] HostKVTransferTicket prefetch(std::size_t layer, std::size_t plane,
                                                std::uint32_t first, std::uint32_t count,
                                                std::size_t device_offset);
    // No CPU wait for unrelated in-flight writes on this layer. The range
    // must already be archived and must not overlap any pending writeback.
    [[nodiscard]] HostKVTransferTicket prefetch_completed(
        std::size_t layer, std::size_t plane, std::uint32_t first, std::uint32_t count,
        std::size_t device_offset);
    // Sorted original logical IDs packed into one <=64 MiB fixed ticket.
    // Fragmentation changes CPU descriptors/copy count, never CUDA/ring/event capacity.
    [[nodiscard]] HostKVTransferTicket prefetch_gather_completed(
        std::size_t layer, std::size_t plane, std::span<const std::uint32_t> logical_pages,
        std::size_t device_offset);
    // Read-only Mean-K layer index. Uses the existing ring in pageable mode;
    // pinned mode requires cacheable CUDA-pinned source memory and copies directly.
    // Source bytes must remain immutable and alive until synchronize(), or until
    // the consuming stream has completed its wait and release. wait() alone only
    // submits a GPU dependency and does not establish host-source completion.
    [[nodiscard]] HostKVTransferTicket prefetch_index(
        std::span<const std::byte> source, std::size_t device_offset);
    // CPU waits only until the worker records ready, then GPU waits on the DMA event.
    void wait(HostKVTransferTicket ticket, cudaStream_t consumer_stream);
    [[nodiscard]] DeviceSpan staged(HostKVTransferTicket ticket);
    // Record after the consuming kernel(s); protects the range and event slot from reuse.
    void release(HostKVTransferTicket ticket, cudaStream_t consumer_stream);
    // The pool and producer stream must remain alive, and the source device
    // bytes must remain unchanged, until the returned future completes.
    [[nodiscard]] std::shared_future<void> writeback(
        const PagedKVPool& pool, std::size_t layer, std::uint32_t first,
        std::span<const std::int32_t> physical_pages, std::uint32_t new_frontier,
        cudaStream_t producer_stream);
    // Disabled unless a test explicitly configures a drained fixed-resource owner.
    void set_test_fault(HostKVTransferFaultInjection);
    void synchronize();
    // Always drain worker DMA, producer/consumer dependencies and ring borrowers,
    // even with a sticky job exception. GPU/context drain errors fail closed.
    [[nodiscard]] HostKVTransferFailureState quiesce();
    [[nodiscard]] HostKVTransferFailureState failure_state() const;
    // Pure sticky-cause guard; caller still drains every borrowed lane first.
    void throw_if_unrecoverable() const;
    // Quiesce, retire exceptional archive futures, invalidate uncertain D2H
    // archive bytes, and restart exactly two workers using existing resources.
    // Returns the preceding failure state; old tickets are invalid afterwards.
    [[nodiscard]] HostKVTransferFailureState recover();
    void reset();
    void trim(std::uint32_t frontier);
    // Commit metadata only after recover/quiesce and all correlated GPU clears.
    // The serialized owner must not enqueue work between preparation and commit.
    void commit_empty_reset() noexcept;
    [[nodiscard]] std::size_t pinned_ring_bytes() const noexcept;

private:
    friend struct HostKVTransferTestAccess;
    struct Impl;
    HostKVTransferTicket prefetch_impl(std::size_t layer, std::size_t plane,
        std::uint32_t first, std::uint32_t count, std::size_t offset, bool completed_only);
    HostKVTransferTicket enqueue_prefetch(std::span<const std::byte> source,
        std::size_t offset);
    HostKVTransferTicket enqueue_gather(std::vector<std::span<const std::byte>> sources,
        std::size_t bytes, std::size_t offset);
    HostKVArchive& archive_;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::kvmem

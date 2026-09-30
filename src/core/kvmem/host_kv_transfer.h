#pragma once

#include "core/kvmem/host_kv_archive.h"

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

// Serialized C=1 scheduling; two copy workers, independent nonblocking H2D/D2H streams.
// Archive and caller-owned workspace/consumer streams must outlive this owner.
// Each prefetch covers <=64 MiB; callers tile larger planes in logical order.
class HostKVTransferEngine {
public:
    HostKVTransferEngine(HostKVArchive& archive, DeviceSpan staging, HostKVStagingPlan plan);
    ~HostKVTransferEngine();
    HostKVTransferEngine(const HostKVTransferEngine&) = delete;
    HostKVTransferEngine& operator=(const HostKVTransferEngine&) = delete;

    [[nodiscard]] HostKVTransferTicket prefetch(std::size_t layer, std::size_t plane,
                                                std::uint32_t first, std::uint32_t count,
                                                std::size_t device_offset);
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
    void synchronize();
    void reset();
    void trim(std::uint32_t frontier);
    [[nodiscard]] std::size_t pinned_ring_bytes() const noexcept;

private:
    struct Impl;
    HostKVArchive& archive_;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::kvmem

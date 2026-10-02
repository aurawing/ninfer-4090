#pragma once

#include "core/paged_kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::kvmem {
class HostKVTransferEngine;

enum class HostArchiveMode { Auto, Pinned, Pageable };
inline constexpr std::uint64_t kHostArchivePhysicalHeadroom = std::uint64_t{4} << 30;

struct HostKVPlaneLayout {
    std::size_t offset = 0;
    std::size_t page_bytes = 0;
    std::size_t reserved_bytes = 0;
    std::size_t pool_plane = 0;
};
struct HostKVArchiveLayout {
    std::uint32_t max_context = 0;
    std::uint32_t logical_pages = 0;
    std::size_t os_page_bytes = 0;
    std::size_t bytes = 0;
    PagedKVPlaneOrder pool_order = PagedKVPlaneOrder::PageMajor;
    std::vector<std::vector<HostKVPlaneLayout>> layers;
};

// Bytes come from the existing pool layout, not duplicated model/KV formulas.
[[nodiscard]] HostKVArchiveLayout plan_host_kv_archive(const PagedKVPool& pool,
                                                      std::size_t planes_per_layer,
                                                      std::uint32_t max_context);
[[nodiscard]] HostKVArchiveLayout plan_host_kv_archive(const PagedKVPoolLayout& pool,
                                                      std::size_t planes_per_layer,
                                                      std::uint32_t max_context);
[[nodiscard]] std::uint64_t available_physical_memory_bytes();
void check_host_archive_admission(std::size_t archive_bytes, std::uint64_t available_bytes);
[[nodiscard]] bool keep_pinned_archive(bool allocation_succeeded,
                                        std::uint64_t available_after_pin) noexcept;

// C=1 owner. Callers serialize scheduling and drain consumers before trim/destruction.
// A writeback event covers every plane of a layer; only completed writes are readable.
class HostKVArchive {
public:
    explicit HostKVArchive(HostKVArchiveLayout layout, HostArchiveMode requested, bool lock_pageable = false);
    ~HostKVArchive();
    HostKVArchive(const HostKVArchive&) = delete;
    HostKVArchive& operator=(const HostKVArchive&) = delete;

    [[nodiscard]] const HostKVArchiveLayout& layout() const noexcept;
    [[nodiscard]] HostArchiveMode mode() const noexcept;
    [[nodiscard]] std::size_t committed_bytes() const noexcept;
    [[nodiscard]] bool os_locked() const noexcept;
    [[nodiscard]] std::uint64_t generation() const noexcept;
    [[nodiscard]] std::uint32_t frontier(std::size_t layer);

    void writeback(const PagedKVPool& pool, std::size_t layer, std::uint32_t first_logical_page,
                   std::span<const std::int32_t> physical_pages, std::uint32_t new_frontier,
                   cudaStream_t stream);
    void restore_to_pool(PagedKVPool& pool, std::size_t layer, std::uint32_t first_logical_page,
                          std::span<const std::int32_t> physical_pages, cudaStream_t stream);
    // Synchronizes pending writeback before handing CPU-readable bytes to a transfer engine.
    [[nodiscard]] std::span<const std::byte> pages(std::size_t layer, std::size_t plane,
                                                   std::uint32_t first, std::uint32_t count);
    void synchronize_layer(std::size_t layer);
    void synchronize();
    // Cannot grow any layer frontier. Preserves a partial final page; its tail is invalid.
    void trim(std::uint32_t exact_frontier);

private:
    friend class HostKVTransferEngine;
    void attach_transfer_owner(void* owner);
    void detach_transfer_owner(void* owner) noexcept;
    void trim_owned(std::uint32_t frontier);
    // Read completed history without waiting for a disjoint pending tail write.
    [[nodiscard]] std::span<const std::byte> completed_pages(
        std::size_t layer, std::size_t plane, std::uint32_t first, std::uint32_t count);
    std::vector<std::span<std::byte>> prepare_async_writeback(
        std::size_t layer, std::uint32_t first, std::uint32_t count, std::uint32_t frontier,
        std::shared_future<void> completion);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::kvmem

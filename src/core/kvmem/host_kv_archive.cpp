#include "core/kvmem/host_kv_archive.h"
#include "core/device.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace ninfer::kvmem {
namespace {
std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error("host KV archive size overflow");
    }
    return a + b;
}
std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::overflow_error("host KV archive size overflow");
    }
    return a * b;
}
std::size_t rounded(std::size_t bytes, std::size_t alignment) {
    return checked_add(bytes, alignment - 1) / alignment * alignment;
}
std::size_t os_page_bytes() {
#if defined(_WIN32)
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    return info.dwPageSize;
#else
    const auto bytes = sysconf(_SC_PAGESIZE);
    if (bytes <= 0) { throw std::runtime_error("cannot query host page size"); }
    return static_cast<std::size_t>(bytes);
#endif
}
std::uint32_t page_count(std::uint32_t tokens) {
    return tokens / kPagedKVPageSize + (tokens % kPagedKVPageSize != 0);
}
}

std::uint64_t available_physical_memory_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) { throw std::runtime_error("GlobalMemoryStatusEx failed"); }
    return status.ullAvailPhys;
#else
    const auto pages = sysconf(_SC_AVPHYS_PAGES);
    if (pages < 0) { throw std::runtime_error("cannot query available host memory"); }
    return checked_mul(static_cast<std::size_t>(pages), os_page_bytes());
#endif
}
void check_host_archive_admission(std::size_t bytes, std::uint64_t available) {
    if (available < kHostArchivePhysicalHeadroom ||
        bytes > available - kHostArchivePhysicalHeadroom) {
        throw std::runtime_error("host KV archive needs its full capacity plus 4 GiB physical "
                                 "headroom; shrink --max-context or use a smaller --kv-dtype "
                                 "(BF16 -> int8 -> rk4v4-e8). --kvmem-host-archive pageable "
                                 "avoids CUDA pin limits but does not waive physical admission");
    }
}
bool keep_pinned_archive(bool succeeded, std::uint64_t after) noexcept {
    return succeeded && after >= kHostArchivePhysicalHeadroom;
}

HostKVArchiveLayout plan_host_kv_archive(const PagedKVPool& pool, std::size_t per_layer,
                                         std::uint32_t max_context) {
    if (max_context == 0 || per_layer == 0 || pool.plane_count() % per_layer != 0 ||
        page_count(max_context) > pool.logical_page_capacity()) {
        throw std::invalid_argument("invalid host KV archive pool/context layout");
    }
    HostKVArchiveLayout result;
    result.max_context = max_context;
    result.logical_pages = page_count(max_context);
    result.os_page_bytes = os_page_bytes();
    result.pool_order = pool.plane_order();
    result.layers.resize(pool.plane_count() / per_layer);
    for (std::size_t layer = 0; layer < result.layers.size(); ++layer) {
        for (std::size_t plane = 0; plane < per_layer; ++plane) {
            const auto index = layer * per_layer + plane;
            const auto bytes = pool.page_bytes(index);
            const auto capacity = rounded(checked_mul(bytes, result.logical_pages),
                                            result.os_page_bytes);
            result.layers[layer].push_back({result.bytes, bytes, capacity, index});
            result.bytes = checked_add(result.bytes, capacity);
        }
    }
    return result;
}

HostKVArchiveLayout plan_host_kv_archive(const PagedKVPoolLayout& pool, std::size_t per_layer,
                                         std::uint32_t max_context) {
    if (max_context == 0 || per_layer == 0 || pool.planes.empty() ||
        pool.planes.size() % per_layer != 0 ||
        page_count(max_context) > pool.spec.logical_page_capacity ||
        pool.spec.page_group_count == 0) {
        throw std::invalid_argument("invalid host KV archive pool/context layout");
    }
    HostKVArchiveLayout result;
    result.max_context = max_context;
    result.logical_pages = page_count(max_context);
    result.os_page_bytes = os_page_bytes();
    result.pool_order = pool.spec.plane_order;
    result.layers.resize(pool.planes.size() / per_layer);
    for (std::size_t layer = 0; layer < result.layers.size(); ++layer) {
        for (std::size_t plane = 0; plane < per_layer; ++plane) {
            const auto index = layer * per_layer + plane;
            const auto plane_bytes = pool.planes[index].storage.region.bytes;
            if (plane_bytes % pool.spec.page_group_count != 0) {
                throw std::invalid_argument("host KV pool plane is not an integral page layout");
            }
            const auto bytes = plane_bytes / pool.spec.page_group_count;
            const auto capacity = rounded(checked_mul(bytes, result.logical_pages),
                                          result.os_page_bytes);
            result.layers[layer].push_back({result.bytes, bytes, capacity, index});
            result.bytes = checked_add(result.bytes, capacity);
        }
    }
    return result;
}

struct HostKVArchive::Impl {
    HostKVArchiveLayout layout;
    HostArchiveMode mode = HostArchiveMode::Pageable;
    void* data = nullptr;
    std::uint64_t generation = 0;
    std::vector<std::vector<std::size_t>> committed;
    std::vector<std::uint32_t> frontiers, pending_frontiers;
    std::vector<std::uint32_t> pending_first, pending_end;
    std::vector<cudaEvent_t> done, read_done;
    std::vector<bool> pending, reading;
    std::vector<std::shared_future<void>> external_done;
    void* transfer_owner = nullptr;
    bool os_locked = false;
#if defined(_WIN32)
    SIZE_T previous_min = 0, previous_max = 0;
    DWORD previous_flags = 0;
    bool working_set_changed = false;
#endif

    ~Impl() {
        for (auto& completion : external_done) {
            if (completion.valid()) { completion.wait(); }
        }
        for (std::size_t i = 0; i < done.size(); ++i) {
            if (done[i]) {
                if (pending[i]) { (void)cudaEventSynchronize(done[i]); }
                (void)cudaEventDestroy(done[i]);
            }
            if (read_done[i]) {
                if (reading[i]) { (void)cudaEventSynchronize(read_done[i]); }
                (void)cudaEventDestroy(read_done[i]);
            }
        }
        if (!data) { return; }
        if (mode == HostArchiveMode::Pinned) { (void)cudaFreeHost(data); }
        else {
#if defined(_WIN32)
            if (os_locked) (void)VirtualUnlock(data, layout.bytes);
            (void)VirtualFree(data, 0, MEM_RELEASE);
            if (working_set_changed)
                (void)SetProcessWorkingSetSizeEx(GetCurrentProcess(), previous_min,
                                                 previous_max, previous_flags);
#else
            if (os_locked) (void)munlock(data, layout.bytes);
            (void)munmap(data, layout.bytes);
#endif
        }
    }
    std::byte* address(const HostKVPlaneLayout& plane, std::uint32_t first) const {
        return static_cast<std::byte*>(data) + plane.offset + plane.page_bytes * first;
    }
    void ensure_committed(std::size_t layer, std::size_t plane, std::uint32_t end_page) {
        if (mode == HostArchiveMode::Pinned || os_locked) { return; }
        const auto& spec = layout.layers.at(layer).at(plane);
        auto& old = committed.at(layer).at(plane);
        const auto end = rounded(spec.page_bytes * end_page, layout.os_page_bytes);
        if (end <= old) { return; }
        void* begin = static_cast<std::byte*>(data) + spec.offset + old;
#if defined(_WIN32)
        if (VirtualAlloc(begin, end - old, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
            throw std::runtime_error("host KV archive page commit failed");
        }
#else
        if (mprotect(begin, end - old, PROT_READ | PROT_WRITE) != 0) {
            throw std::runtime_error("host KV archive page commit failed");
        }
#endif
        old = end;
    }
    void validate_pool(const PagedKVPool& pool, std::size_t layer,
                        std::span<const std::int32_t> ids) const {
        if (pool.plane_order() != layout.pool_order) {
            throw std::invalid_argument("host KV archive pool order mismatch");
        }
        for (const auto& plane : layout.layers.at(layer)) {
            if (plane.pool_plane >= pool.plane_count() ||
                pool.page_bytes(plane.pool_plane) != plane.page_bytes) {
                throw std::invalid_argument("host KV archive plane shape mismatch");
            }
        }
        for (auto id : ids) {
            if (id < 0 || static_cast<std::uint32_t>(id) >= pool.page_group_count()) {
                throw std::invalid_argument("host KV archive physical page outside pool");
            }
        }
    }
};

HostKVArchive::HostKVArchive(HostKVArchiveLayout layout, HostArchiveMode requested, bool lock_pageable)
    : impl_(std::make_unique<Impl>()) {
    if (requested != HostArchiveMode::Auto && requested != HostArchiveMode::Pinned &&
        requested != HostArchiveMode::Pageable) {
        throw std::invalid_argument("invalid host KV archive mode");
    }
    if (lock_pageable && requested == HostArchiveMode::Pinned)
        throw std::invalid_argument("VirtualLock conflicts with CUDA pinned archive; use pageable");
    if (lock_pageable) requested = HostArchiveMode::Pageable;
    if (layout.layers.empty() || layout.max_context == 0 ||
        layout.logical_pages != page_count(layout.max_context) ||
        layout.os_page_bytes != os_page_bytes()) {
        throw std::invalid_argument("invalid host KV archive layout");
    }
    std::size_t offset = 0;
    for (const auto& layer : layout.layers) {
        if (layer.empty()) { throw std::invalid_argument("empty host KV archive layer"); }
        for (const auto& plane : layer) {
            if (plane.offset != offset || plane.page_bytes == 0 ||
                plane.reserved_bytes != rounded(checked_mul(plane.page_bytes, layout.logical_pages),
                                                  layout.os_page_bytes)) {
                throw std::invalid_argument("invalid host KV archive plane offsets");
            }
            offset = checked_add(offset, plane.reserved_bytes);
        }
    }
    if (layout.bytes != offset) { throw std::invalid_argument("invalid host KV archive byte size"); }
    check_host_archive_admission(layout.bytes, available_physical_memory_bytes());
    auto& state = *impl_;
    state.layout = std::move(layout);
    const auto layers = state.layout.layers.size();
    state.committed.resize(layers);
    state.frontiers.resize(layers);
    state.pending_frontiers.resize(layers);
    state.pending_first.resize(layers);
    state.pending_end.resize(layers);
    state.external_done.resize(layers);
    state.pending.resize(layers);
    state.reading.resize(layers);
    state.done.resize(layers);
    state.read_done.resize(layers);
    for (std::size_t i = 0; i < layers; ++i) {
        state.committed[i].resize(state.layout.layers[i].size());
        CUDA_CHECK(cudaEventCreateWithFlags(&state.done[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&state.read_done[i], cudaEventDisableTiming));
    }
    const char* reason = "explicit pageable";
    if (requested != HostArchiveMode::Pageable) {
        void* pinned = nullptr;
        const auto error = cudaMallocHost(&pinned, state.layout.bytes);
        // Own the successful allocation before the potentially throwing memory query.
        if (error == cudaSuccess) {
            state.data = pinned;
            state.mode = HostArchiveMode::Pinned;
        }
        const auto after = error == cudaSuccess ? available_physical_memory_bytes() : 0;
        if (keep_pinned_archive(error == cudaSuccess, after)) {
            state.data = pinned;
            state.mode = HostArchiveMode::Pinned;
            reason = "whole archive pinned, physical headroom >= 4 GiB";
        } else {
            if (pinned) { CUDA_CHECK(cudaFreeHost(pinned)); }
            state.data = nullptr;
            state.mode = HostArchiveMode::Pageable;
            // A failed admission probe must not leak its CUDA last-error status.
            if (error != cudaSuccess) { (void)cudaGetLastError(); }
            if (requested == HostArchiveMode::Pinned) {
                throw std::runtime_error("explicit pinned KV archive could not pin full capacity "
                                         "with 4 GiB physical headroom; use --kvmem-host-archive pageable, "
                                         "shrink --max-context or use a smaller --kv-dtype");
            }
            reason = error == cudaSuccess ? "auto fallback: physical headroom < 4 GiB"
                                           : "auto fallback: whole archive pin failed";
        }
    }
    if (!state.data) {
#if defined(_WIN32)
        state.data = VirtualAlloc(nullptr, state.layout.bytes, MEM_RESERVE, PAGE_NOACCESS);
        if (!state.data) { throw std::runtime_error("host KV archive virtual reserve failed"); }
#else
        state.data = mmap(nullptr, state.layout.bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (state.data == MAP_FAILED) {
            state.data = nullptr;
            throw std::runtime_error("host KV archive virtual reserve failed");
        }
#endif
    }
    if (lock_pageable) {
        // Explicit residency protection commits/locks the full archive at loading.
        // Ordinary pageable archives continue to commit on demand.
        for (std::size_t layer = 0; layer < layers; ++layer)
            for (std::size_t plane = 0; plane < state.layout.layers[layer].size(); ++plane)
                state.ensure_committed(layer, plane, state.layout.logical_pages);
#if defined(_WIN32)
        const auto process = GetCurrentProcess();
        if (!GetProcessWorkingSetSizeEx(process, &state.previous_min, &state.previous_max,
                                       &state.previous_flags))
            throw std::runtime_error("cannot query working set for --kvmem-lock-archive");
        const auto minimum = checked_add(state.previous_min, state.layout.bytes);
        const auto maximum = std::max<std::size_t>(state.previous_max, checked_add(minimum, 64ULL << 20));
        if (!SetProcessWorkingSetSizeEx(process, minimum, maximum, state.previous_flags))
            throw std::runtime_error("cannot raise working set for --kvmem-lock-archive; disable it, use pageable, shrink --max-context or use smaller KV");
        state.working_set_changed = true;
        if (!VirtualLock(state.data, state.layout.bytes))
            throw std::runtime_error("VirtualLock failed for --kvmem-lock-archive; disable it, use pageable, shrink --max-context or use smaller KV");
#else
        if (mlock(state.data, state.layout.bytes) != 0)
            throw std::runtime_error("OS archive lock failed; disable --kvmem-lock-archive or shrink context/KV");
#endif
        state.os_locked = true;
        if (available_physical_memory_bytes() < kHostArchivePhysicalHeadroom)
            throw std::runtime_error("OS archive lock left less than 4 GiB; shrink --max-context or use smaller KV");
        reason = "explicit OS lock, pageable transfers through fixed ring";
    }
    std::clog << "[kvmem] host_archive="
              << (state.mode == HostArchiveMode::Pinned ? "pinned" : "pageable")
              << " capacity_bytes=" << state.layout.bytes << " os_locked=" << state.os_locked
              << " available_physical_after_bytes=" << available_physical_memory_bytes()
              << " required_headroom_bytes=" << kHostArchivePhysicalHeadroom
              << " reason=" << reason << '\n';
}
HostKVArchive::~HostKVArchive() = default;
const HostKVArchiveLayout& HostKVArchive::layout() const noexcept { return impl_->layout; }
HostArchiveMode HostKVArchive::mode() const noexcept { return impl_->mode; }
bool HostKVArchive::os_locked() const noexcept { return impl_->os_locked; }
std::uint64_t HostKVArchive::generation() const noexcept { return impl_->generation; }
std::size_t HostKVArchive::committed_bytes() const noexcept {
    if (impl_->mode == HostArchiveMode::Pinned) { return impl_->layout.bytes; }
    std::size_t result = 0;
    for (const auto& layer : impl_->committed) { for (auto bytes : layer) { result += bytes; } }
    return result;
}
void HostKVArchive::synchronize_layer(std::size_t layer) {
    auto& state = *impl_;
    auto& external = state.external_done.at(layer);
    if (external.valid()) {
        external.get();
        state.frontiers[layer] = state.pending_frontiers[layer];
        external = {};
    }
    if (state.pending.at(layer)) {
        CUDA_CHECK(cudaEventSynchronize(state.done[layer]));
        state.frontiers[layer] = state.pending_frontiers[layer];
        state.pending[layer] = false;
    }
    if (state.reading.at(layer)) {
        CUDA_CHECK(cudaEventSynchronize(state.read_done[layer]));
        state.reading[layer] = false;
    }
}
void HostKVArchive::synchronize() {
    for (std::size_t i = 0; i < impl_->layout.layers.size(); ++i) { synchronize_layer(i); }
}
std::uint32_t HostKVArchive::frontier(std::size_t layer) {
    synchronize_layer(layer);
    return impl_->frontiers.at(layer);
}

void HostKVArchive::writeback(const PagedKVPool& pool, std::size_t layer, std::uint32_t first,
                              std::span<const std::int32_t> ids, std::uint32_t frontier,
                              cudaStream_t stream) {
    auto& state = *impl_;
    if (state.transfer_owner) {
        throw std::logic_error("attached archive writeback must use its transfer owner");
    }
    state.validate_pool(pool, layer, ids);
    synchronize_layer(layer);
    const auto old = state.frontiers.at(layer);
    if (ids.empty() || frontier < old || frontier > state.layout.max_context ||
        first > page_count(old) || ids.size() > state.layout.logical_pages - first ||
        first + ids.size() > page_count(frontier) ||
        (frontier > old && (first > old / kPagedKVPageSize ||
                           first + ids.size() < page_count(frontier)))) {
        throw std::invalid_argument("host KV writeback invalid range, frontier or logical hole");
    }
    const auto end = first + static_cast<std::uint32_t>(ids.size());
    for (std::size_t p = 0; p < state.layout.layers[layer].size(); ++p) {
        state.ensure_committed(layer, p, end);
    }
    try {
        for (const auto& plane : state.layout.layers[layer]) {
            auto* destination = state.address(plane, first);
            if (state.layout.pool_order == PagedKVPlaneOrder::PageMajor) {
                pool.copy_pages_to_host(plane.pool_plane, ids, destination, stream);
            } else {
                for (std::size_t i = 0; i < ids.size(); ++i) {
                    pool.copy_page_to_host(plane.pool_plane, ids[i],
                                            destination + i * plane.page_bytes, stream);
                }
            }
        }
        CUDA_CHECK(cudaEventRecord(state.done[layer], stream));
    } catch (...) {
        // Earlier planes may already be in flight even if a later enqueue fails.
        (void)cudaStreamSynchronize(stream);
        throw;
    }
    state.pending_frontiers[layer] = frontier;
    state.pending_first[layer] = first;
    state.pending_end[layer] = end;
    state.pending[layer] = true;
}

std::span<const std::byte> HostKVArchive::pages(std::size_t layer, std::size_t plane,
                                               std::uint32_t first, std::uint32_t count) {
    synchronize_layer(layer);
    const auto& spec = impl_->layout.layers.at(layer).at(plane);
    const auto pages = page_count(impl_->frontiers.at(layer));
    if (first > pages || count > pages - first) {
        throw std::out_of_range("host KV archive range beyond valid frontier");
    }
    return {impl_->address(spec, first), spec.page_bytes * count};
}
std::span<const std::byte> HostKVArchive::completed_pages(
    std::size_t layer, std::size_t plane, std::uint32_t first, std::uint32_t count) {
    const auto& state = *impl_;
    const auto& spec = state.layout.layers.at(layer).at(plane);
    const auto valid_pages = page_count(state.frontiers.at(layer));
    if (first > valid_pages || count > valid_pages - first) {
        throw std::out_of_range("completed KV history beyond committed frontier");
    }
    if ((state.pending.at(layer) || state.external_done.at(layer).valid()) && count &&
        first < state.pending_end.at(layer) && first + count > state.pending_first.at(layer)) {
        throw std::logic_error("completed KV history overlaps pending writeback");
    }
    return {state.address(spec, first), spec.page_bytes * count};
}
void HostKVArchive::restore_to_pool(PagedKVPool& pool, std::size_t layer, std::uint32_t first,
                                    std::span<const std::int32_t> ids, cudaStream_t stream) {
    if (impl_->transfer_owner) {
        throw std::logic_error("attached archive restore must use its transfer owner");
    }
    impl_->validate_pool(pool, layer, ids);
    synchronize_layer(layer);
    if (ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("host KV restore range too large");
    }
    try {
        for (std::size_t p = 0; p < impl_->layout.layers[layer].size(); ++p) {
            const auto bytes = pages(layer, p, first, static_cast<std::uint32_t>(ids.size()));
            const auto& spec = impl_->layout.layers[layer][p];
            if (impl_->layout.pool_order == PagedKVPlaneOrder::PageMajor) {
                pool.copy_pages_from_host(spec.pool_plane, ids, bytes.data(), stream);
            } else {
                for (std::size_t i = 0; i < ids.size(); ++i) {
                    pool.copy_page_from_host(spec.pool_plane, ids[i],
                                              bytes.data() + i * spec.page_bytes, stream);
                }
            }
        }
        CUDA_CHECK(cudaEventRecord(impl_->read_done[layer], stream));
    } catch (...) {
        (void)cudaStreamSynchronize(stream);
        throw;
    }
    impl_->reading[layer] = true;
}
void HostKVArchive::trim(std::uint32_t frontier) {
    if (impl_->transfer_owner) {
        throw std::logic_error("attached archive trim must use its transfer owner");
    }
    trim_owned(frontier);
}
void HostKVArchive::trim_owned(std::uint32_t frontier) {
    auto& state = *impl_;
    synchronize();
    for (auto old : state.frontiers) {
        if (frontier > old) { throw std::invalid_argument("host KV archive trim cannot grow"); }
    }
    if (state.generation == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("host KV archive generation exhausted");
    }
    for (std::size_t layer = 0; layer < state.layout.layers.size(); ++layer) {
        for (std::size_t p = 0; p < state.layout.layers[layer].size(); ++p) {
            auto& old = state.committed[layer][p];
            const auto& plane = state.layout.layers[layer][p];
            const auto kept = rounded(plane.page_bytes * page_count(frontier),
                                        state.layout.os_page_bytes);
            if (state.mode == HostArchiveMode::Pageable && !state.os_locked && old > kept) {
                auto* begin = static_cast<std::byte*>(state.data) + plane.offset + kept;
#if defined(_WIN32)
                if (!VirtualFree(begin, old - kept, MEM_DECOMMIT)) {
                    throw std::runtime_error("host KV archive decommit failed");
                }
#else
                if (mprotect(begin, old - kept, PROT_NONE) != 0 ||
                    madvise(begin, old - kept, MADV_DONTNEED) != 0) {
                    throw std::runtime_error("host KV archive decommit failed");
                }
#endif
                old = kept;
            }
        }
        state.frontiers[layer] = frontier;
        state.pending_frontiers[layer] = frontier;
    }
    ++state.generation;
}

void HostKVArchive::attach_transfer_owner(void* owner) {
    if (!owner || impl_->transfer_owner) {
        throw std::logic_error("host KV archive already has a transfer owner");
    }
    synchronize();
    impl_->transfer_owner = owner;
}
void HostKVArchive::detach_transfer_owner(void* owner) noexcept {
    if (impl_->transfer_owner == owner) { impl_->transfer_owner = nullptr; }
}
std::vector<std::span<std::byte>> HostKVArchive::prepare_async_writeback(
    std::size_t layer, std::uint32_t first, std::uint32_t count, std::uint32_t frontier,
    std::shared_future<void> completion) {
    auto& state = *impl_;
    synchronize_layer(layer);
    const auto old = state.frontiers.at(layer);
    if (count == 0 || frontier < old || frontier > state.layout.max_context ||
        first > page_count(old) || count > state.layout.logical_pages - first ||
        first + count > page_count(frontier) ||
        (frontier > old && (first > old / kPagedKVPageSize || first + count < page_count(frontier)))) {
        throw std::invalid_argument("async host KV writeback invalid range or frontier");
    }
    std::vector<std::span<std::byte>> result;
    result.reserve(state.layout.layers[layer].size());
    for (std::size_t p = 0; p < state.layout.layers[layer].size(); ++p) {
        state.ensure_committed(layer, p, first + count);
        const auto& spec = state.layout.layers[layer][p];
        result.emplace_back(state.address(spec, first), spec.page_bytes * count);
    }
    state.pending_frontiers[layer] = frontier;
    state.pending_first[layer] = first;
    state.pending_end[layer] = first + count;
    state.external_done[layer] = std::move(completion);
    return result;
}

} // namespace ninfer::kvmem

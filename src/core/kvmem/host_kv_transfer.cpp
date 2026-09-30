#include "core/kvmem/host_kv_transfer.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace ninfer::kvmem {
namespace {
std::size_t add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error("host KV staging budget overflow");
    }
    return a + b;
}
std::size_t multiply(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::overflow_error("host KV staging budget overflow");
    }
    return a * b;
}
bool overlaps(std::size_t a, std::size_t an, std::size_t b, std::size_t bn) {
    return a < b + bn && b < a + an;
}
}
HostKVStagingPlan plan_host_kv_staging(const HostKVArchiveLayout& layout,
                                       std::uint32_t resident, std::size_t override_bytes) {
    if (resident > layout.logical_pages || layout.layers.empty()) {
        throw std::invalid_argument("invalid host KV resident page budget");
    }
    std::size_t largest = 0, planes = 0;
    for (const auto& layer : layout.layers) {
        std::size_t bytes = 0;
        planes = std::max(planes, layer.size());
        for (const auto& plane : layer) {
            bytes = add(bytes, multiply(plane.page_bytes, layout.logical_pages - resident));
        }
        largest = std::max(largest, bytes);
    }
    const auto minimum = add(largest, kHostKVTransferTileBytes);
    if (override_bytes && override_bytes < minimum) {
        throw std::invalid_argument("staging override is smaller than streamed layer plus one tile");
    }
    const auto capacity = override_bytes ? override_bytes : minimum;
    const auto tiles = capacity / kHostKVTransferTileBytes + (capacity % kHostKVTransferTileBytes != 0);
    return {resident, largest, capacity, add(add(tiles, multiply(planes, 2)), 4)};
}

struct HostKVTransferEngine::Impl {
    struct Slot {
        std::uint64_t sequence = 0;
        std::uint64_t generation = 0;
        std::size_t offset = 0, bytes = 0, references = 0;
        bool enqueued = false, waited = false, released = false;
        cudaStream_t consumer = nullptr;
    };
    struct RingSlot {
        std::unique_ptr<PinnedHostBuffer> memory;
        cudaEvent_t done = nullptr;
        bool used = false;
    };
    struct Job {
        bool writeback = false;
        std::size_t slot = 0, layer = 0;
        const std::byte* source = nullptr;
        const PagedKVPool* pool = nullptr;
        std::vector<std::size_t> dependencies;
        std::vector<std::int32_t> pages;
        std::vector<std::span<std::byte>> destinations;
        std::shared_ptr<std::promise<void>> completion;
    };
    HostKVArchive& archive;
    DeviceSpan staging;
    HostKVStagingPlan plan;
    int device = 0;
    cudaStream_t stream = nullptr;
    std::vector<Slot> slots;
    std::vector<cudaEvent_t> ready, consumed, producer_ready, writeback_done;
    std::array<RingSlot, 4> ring;
    std::size_t ring_cursor = 0;
    std::uint64_t next_sequence = 1;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<Job> jobs;
    std::thread worker;
    bool stopping = false, active = false;
    std::exception_ptr error;

    Impl(HostKVArchive& a, DeviceSpan s, HostKVStagingPlan p) : archive(a), staging(s), plan(p) {}
    ~Impl() {
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        changed.notify_all();
        if (worker.joinable()) { worker.join(); }
        if (stream) { (void)cudaStreamSynchronize(stream); }
        for (const auto& slot : slots) {
            if (slot.waited && !slot.released) { (void)cudaStreamSynchronize(slot.consumer); }
        }
        for (std::size_t i = 0; i < consumed.size(); ++i) {
            if (slots[i].released) { (void)cudaEventSynchronize(consumed[i]); }
        }
        for (auto& buffers : {&ready, &consumed, &producer_ready, &writeback_done}) {
            for (auto event : *buffers) { if (event) { (void)cudaEventDestroy(event); } }
        }
        for (auto& slot : ring) { if (slot.done) { (void)cudaEventDestroy(slot.done); } }
        if (stream) { (void)cudaStreamDestroy(stream); }
    }
    void initialize() {
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        slots.resize(plan.ticket_capacity);
        ready.resize(slots.size());
        consumed.resize(slots.size());
        producer_ready.resize(archive.layout().layers.size());
        writeback_done.resize(producer_ready.size());
        for (auto& buffers : {&ready, &consumed, &producer_ready, &writeback_done}) {
            for (auto& event : *buffers) {
                CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            }
        }
        if (archive.mode() == HostArchiveMode::Pageable) {
            for (auto& slot : ring) {
                slot.memory = std::make_unique<PinnedHostBuffer>(kHostKVTransferTileBytes);
                CUDA_CHECK(cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming));
            }
        }
        auto started = std::make_shared<std::promise<void>>();
        auto startup = started->get_future();
        worker = std::thread([this, started] { run(started); });
        startup.get();
    }
    void check_error() const { if (error) { std::rethrow_exception(error); } }
    Slot& validate(HostKVTransferTicket ticket) {
        check_error();
        if (ticket.slot >= slots.size() || !ticket.sequence ||
            slots[ticket.slot].sequence != ticket.sequence ||
            slots[ticket.slot].generation != ticket.archive_generation ||
            ticket.archive_generation != archive.generation()) {
            throw std::logic_error("stale host KV transfer ticket");
        }
        return slots[ticket.slot];
    }
    RingSlot& acquire_ring() {
        auto& slot = ring[ring_cursor++ % ring.size()];
        if (slot.used) { CUDA_CHECK(cudaEventSynchronize(slot.done)); }
        return slot;
    }
    void process_prefetch(const Job& job) {
        for (auto dependency : job.dependencies) {
            // Reference counts prevent another ticket from re-recording this event
            // until this stream has captured its precise prior-consumer generation.
            CUDA_CHECK(cudaStreamWaitEvent(stream, consumed[dependency], 0));
            std::lock_guard lock(mutex);
            --slots[dependency].references;
        }
        Slot target;
        { std::lock_guard lock(mutex); target = slots[job.slot]; }
        const void* source = job.source;
        RingSlot* host_slot = nullptr;
        if (archive.mode() == HostArchiveMode::Pageable) {
            host_slot = &acquire_ring();
            std::memcpy(host_slot->memory->data(), source, target.bytes);
            source = host_slot->memory->data();
        }
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(staging.data) + target.offset, source,
                                    target.bytes, cudaMemcpyHostToDevice, stream));
        if (host_slot) {
            CUDA_CHECK(cudaEventRecord(host_slot->done, stream));
            host_slot->used = true;
        }
        CUDA_CHECK(cudaEventRecord(ready[job.slot], stream));
        { std::lock_guard lock(mutex); slots[job.slot].enqueued = true; }
        changed.notify_all();
    }
    void pool_copy(const Job& job, std::size_t plane, std::size_t begin, std::size_t count,
                    void* destination) {
        const auto& layout = archive.layout().layers[job.layer][plane];
        const auto ids = std::span<const std::int32_t>(job.pages).subspan(begin, count);
        if (archive.layout().pool_order == PagedKVPlaneOrder::PageMajor) {
            job.pool->copy_pages_to_host(layout.pool_plane, ids, destination, stream);
        } else {
            for (std::size_t i = 0; i < count; ++i) {
                job.pool->copy_page_to_host(layout.pool_plane, ids[i],
                    static_cast<std::byte*>(destination) + i * layout.page_bytes, stream);
            }
        }
    }
    void process_writeback(const Job& job) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, producer_ready[job.layer], 0));
        for (std::size_t plane = 0; plane < job.destinations.size(); ++plane) {
            const auto page_bytes = archive.layout().layers[job.layer][plane].page_bytes;
            if (archive.mode() == HostArchiveMode::Pinned) {
                pool_copy(job, plane, 0, job.pages.size(), job.destinations[plane].data());
            } else {
                const auto per_tile = kHostKVTransferTileBytes / page_bytes;
                if (!per_tile) { throw std::invalid_argument("KV page exceeds fixed transfer tile"); }
                for (std::size_t begin = 0; begin < job.pages.size(); begin += per_tile) {
                    const auto count = std::min(per_tile, job.pages.size() - begin);
                    auto& slot = acquire_ring();
                    pool_copy(job, plane, begin, count, slot.memory->data());
                    CUDA_CHECK(cudaEventRecord(slot.done, stream));
                    slot.used = true;
                    CUDA_CHECK(cudaEventSynchronize(slot.done));
                    std::memcpy(job.destinations[plane].data() + begin * page_bytes,
                                slot.memory->data(), count * page_bytes);
                }
            }
        }
        CUDA_CHECK(cudaEventRecord(writeback_done[job.layer], stream));
        CUDA_CHECK(cudaEventSynchronize(writeback_done[job.layer]));
        job.completion->set_value();
    }
    void run(const std::shared_ptr<std::promise<void>>& started) noexcept {
        try { CUDA_CHECK(cudaSetDevice(device)); started->set_value(); }
        catch (...) {
            { std::lock_guard lock(mutex); error = std::current_exception(); }
            started->set_exception(error);
            changed.notify_all();
            return;
        }
        for (;;) {
            Job job;
            {
                std::unique_lock lock(mutex);
                changed.wait(lock, [&] { return stopping || !jobs.empty(); });
                if (jobs.empty()) { if (stopping) { return; } continue; }
                job = std::move(jobs.front());
                jobs.pop_front();
                active = true;
            }
            try {
                if (job.writeback) { process_writeback(job); }
                else { process_prefetch(job); }
            } catch (...) {
                const auto failure = std::current_exception();
                (void)cudaStreamSynchronize(stream);
                std::lock_guard lock(mutex);
                error = failure;
                if (job.completion) { job.completion->set_exception(failure); }
                for (auto& queued : jobs) {
                    if (queued.completion) { queued.completion->set_exception(failure); }
                }
                jobs.clear();
                active = false;
                changed.notify_all();
                return;
            }
            { std::lock_guard lock(mutex); active = false; }
            changed.notify_all();
        }
    }
    void synchronize() {
        {
            std::unique_lock lock(mutex);
            changed.wait(lock, [&] { return error || (!active && jobs.empty()); });
            check_error();
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (std::size_t i = 0; i < slots.size(); ++i) {
            if (slots[i].released) { CUDA_CHECK(cudaEventSynchronize(consumed[i])); }
            else if (slots[i].waited) { CUDA_CHECK(cudaStreamSynchronize(slots[i].consumer)); }
        }
    }
};

HostKVTransferEngine::HostKVTransferEngine(HostKVArchive& archive, DeviceSpan staging,
                                           HostKVStagingPlan plan)
    : archive_(archive), impl_(std::make_unique<Impl>(archive, staging, plan)) {
    const auto expected = plan_host_kv_staging(archive.layout(), plan.resident_pages,
                                              plan.capacity_bytes);
    if (!staging.data || staging.bytes < plan.capacity_bytes ||
        expected.maximum_layer_stream_bytes != plan.maximum_layer_stream_bytes ||
        expected.ticket_capacity != plan.ticket_capacity) {
        throw std::invalid_argument("invalid caller-owned KV staging workspace plan");
    }
    archive_.attach_transfer_owner(this);
    try { impl_->initialize(); }
    catch (...) { archive_.detach_transfer_owner(this); throw; }
    std::clog << "[kvmem] stage_bytes=" << plan.capacity_bytes
              << " copy_workers=1 pinned_ring_bytes=" << pinned_ring_bytes() << '\n';
}
HostKVTransferEngine::~HostKVTransferEngine() {
    // Impl joins queued jobs and drains DMA/consumer events before freeing its ring.
    impl_.reset();
    archive_.detach_transfer_owner(this);
}
std::size_t HostKVTransferEngine::pinned_ring_bytes() const noexcept {
    return archive_.mode() == HostArchiveMode::Pageable ? 4 * kHostKVTransferTileBytes : 0;
}
HostKVTransferTicket HostKVTransferEngine::prefetch(std::size_t layer, std::size_t plane,
                                                   std::uint32_t first, std::uint32_t count,
                                                   std::size_t offset) {
    auto source = archive_.pages(layer, plane, first, count);
    if (source.empty() || source.size() > kHostKVTransferTileBytes ||
        offset > impl_->plan.capacity_bytes || source.size() > impl_->plan.capacity_bytes - offset) {
        throw std::invalid_argument("KV prefetch tile or device offset exceeds startup capacity");
    }
    auto& state = *impl_;
    std::lock_guard lock(state.mutex);
    state.check_error();
    if (state.next_sequence == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("KV transfer sequence exhausted");
    }
    std::size_t selected = state.slots.size();
    for (std::size_t i = 0; i < state.slots.size(); ++i) {
        const auto& slot = state.slots[i];
        if (slot.sequence && !slot.released && overlaps(offset, source.size(), slot.offset, slot.bytes)) {
            throw std::logic_error("prefetch would overwrite a live staging segment");
        }
        if (!slot.sequence && selected == state.slots.size()) { selected = i; }
    }
    if (selected == state.slots.size()) {
        for (std::size_t i = 0; i < state.slots.size(); ++i) {
            if (state.slots[i].released && state.slots[i].references == 0) { selected = i; break; }
        }
    }
    if (selected == state.slots.size()) {
        throw std::runtime_error("startup KV transfer ticket capacity exhausted");
    }
    Impl::Job job;
    job.slot = selected;
    job.source = source.data();
    for (std::size_t i = 0; i < state.slots.size(); ++i) {
        const auto& slot = state.slots[i];
        if (slot.sequence && slot.released &&
            (i == selected || overlaps(offset, source.size(), slot.offset, slot.bytes))) {
            job.dependencies.push_back(i);
        }
    }
    // Allocate the queue node before changing slot ownership/reference counts.
    state.jobs.push_back(std::move(job));
    for (auto dependency : state.jobs.back().dependencies) { ++state.slots[dependency].references; }
    auto& slot = state.slots[selected];
    const auto references = slot.references;
    slot = {};
    slot.references = references;
    slot.sequence = state.next_sequence++;
    slot.generation = archive_.generation();
    slot.offset = offset;
    slot.bytes = source.size();
    state.changed.notify_all();
    return {selected, slot.sequence, slot.generation};
}
void HostKVTransferEngine::wait(HostKVTransferTicket ticket, cudaStream_t stream) {
    auto& state = *impl_;
    std::unique_lock lock(state.mutex);
    (void)state.validate(ticket);
    state.changed.wait(lock, [&] { return state.error || state.slots[ticket.slot].enqueued; });
    auto& slot = state.validate(ticket);
    if (slot.released || (slot.waited && slot.consumer != stream)) {
        throw std::logic_error("KV ticket must have one consumer stream before release");
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, state.ready[ticket.slot], 0));
    slot.waited = true;
    slot.consumer = stream;
}
DeviceSpan HostKVTransferEngine::staged(HostKVTransferTicket ticket) {
    std::lock_guard lock(impl_->mutex);
    const auto& slot = impl_->validate(ticket);
    if (slot.released) { throw std::logic_error("released KV staging range is no longer readable"); }
    return {static_cast<std::byte*>(impl_->staging.data) + slot.offset, slot.bytes};
}
void HostKVTransferEngine::release(HostKVTransferTicket ticket, cudaStream_t stream) {
    std::lock_guard lock(impl_->mutex);
    auto& slot = impl_->validate(ticket);
    if (!slot.waited || slot.released || slot.consumer != stream) {
        throw std::logic_error("KV release must follow its consumer wait and kernel submission");
    }
    CUDA_CHECK(cudaEventRecord(impl_->consumed[ticket.slot], stream));
    slot.released = true;
}
std::shared_future<void> HostKVTransferEngine::writeback(
    const PagedKVPool& pool, std::size_t layer, std::uint32_t first,
    std::span<const std::int32_t> pages, std::uint32_t frontier, cudaStream_t producer) {
    const auto& planes = archive_.layout().layers.at(layer);
    if (pool.plane_order() != archive_.layout().pool_order || pages.empty() ||
        pages.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid KV writeback pool/order/pages");
    }
    for (const auto& plane : planes) {
        if (pool.page_bytes(plane.pool_plane) != plane.page_bytes ||
            plane.page_bytes > kHostKVTransferTileBytes) {
            throw std::invalid_argument("KV writeback plane bytes mismatch");
        }
    }
    for (auto page : pages) {
        if (page < 0 || static_cast<std::uint32_t>(page) >= pool.page_group_count()) {
            throw std::invalid_argument("KV writeback physical page outside pool");
        }
    }
    // FIFO worker + one staging stream place this write after earlier archive reads.
    // The producer event orders device KV writes without a whole-layer CPU wait.
    Impl::Job job;
    job.writeback = true;
    job.layer = layer;
    job.pool = &pool;
    job.pages.assign(pages.begin(), pages.end());
    job.completion = std::make_shared<std::promise<void>>();
    auto completed = job.completion->get_future().share();
    job.destinations = archive_.prepare_async_writeback(layer, first,
        static_cast<std::uint32_t>(pages.size()), frontier, completed);
    try {
        CUDA_CHECK(cudaEventRecord(impl_->producer_ready[layer], producer));
        std::lock_guard lock(impl_->mutex);
        impl_->check_error();
        impl_->jobs.push_back(std::move(job));
        impl_->changed.notify_all();
    } catch (...) {
        if (job.completion) { job.completion->set_exception(std::current_exception()); }
        throw;
    }
    return completed;
}
void HostKVTransferEngine::synchronize() { impl_->synchronize(); }
void HostKVTransferEngine::reset() {
    synchronize();
    std::lock_guard lock(impl_->mutex);
    for (auto& slot : impl_->slots) { slot = {}; }
}
void HostKVTransferEngine::trim(std::uint32_t frontier) {
    reset();
    archive_.trim_owned(frontier);
}

} // namespace ninfer::kvmem

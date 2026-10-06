#include "core/kvmem/host_kv_transfer.h"
#include "core/kvmem/host_kv_transfer_test_access.h"
#include "core/device.h"
#include "core/kvmem/checked_buffers.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iostream>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#define KV_TRANSFER_CHECK(call) check_transfer_cuda((call), #call)
#define KV_TRANSFER_DRAIN_CHECK(call) check_transfer_cuda((call), #call, true)

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
    // Keep the established startup ticket budget for the active layer and
    // overlapping prefetch handoffs. Archive read fences use fixed layer events.
    return {resident, largest, capacity,
            add(add(add(tiles, multiply(planes, 2)), 4), layout.layers.size())};
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
        std::unique_ptr<CheckedPinnedHostBuffer> memory;
        cudaEvent_t done = nullptr;
        bool used = false, claimed = false;
    };
    struct Job {
        bool writeback = false;
        bool archive_read_fence = false;
        bool destination_touched = false;
        std::size_t slot = 0, layer = 0;
        std::vector<std::span<const std::byte>> sources;
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
    cudaStream_t stream = nullptr; // H2D: only the prefetch worker submits here.
    cudaStream_t writeback_stream = nullptr;
    std::vector<Slot> slots;
    std::vector<cudaEvent_t> ready, consumed, producer_ready, writeback_done;
    // Ticket metadata can be replaced while its old consumed event is still a
    // queued dependency. Failed/canceled replacement jobs may never capture it.
    // Keep every recorded borrower event independently until quiesce drains it.
    std::vector<bool> consumed_recorded;
    // A CUDA wait on an unrecorded event is already complete. This acknowledgment
    // belongs to the current per-layer writeback, whose future serializes reuse.
    std::vector<bool> writeback_fence_recorded;
    std::vector<cudaStream_t> producer_streams;
    std::vector<bool> producer_borrowed;
    std::array<RingSlot, 4> ring;
    std::size_t ring_cursor = 0;
    std::uint64_t next_sequence = 1;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<Job> jobs, writeback_jobs;
    std::thread worker, writeback_worker;
    bool stopping = false, starting = false, active = false, writeback_active = false;
    std::exception_ptr error;
    HostKVTransferFaultInjection fault;
    HostKVTransferFailureState failure;
    std::uint64_t h2d_copies = 0, d2h_planes = 0;
    std::uint64_t worker_start_attempt = 0;

    Impl(HostKVArchive& a, DeviceSpan s, HostKVStagingPlan p, HostKVTransferFaultInjection f)
        : archive(a), staging(s), plan(p), fault(f) {}
    ~Impl() {
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        changed.notify_all();
        if (worker.joinable()) { worker.join(); }
        if (writeback_worker.joinable()) { writeback_worker.join(); }
        if (writeback_stream) { (void)cudaStreamSynchronize(writeback_stream); }
        if (stream) { (void)cudaStreamSynchronize(stream); }
        for (std::size_t i = 0; i < producer_streams.size(); ++i) {
            if (producer_borrowed[i]) (void)cudaStreamSynchronize(producer_streams[i]);
        }
        for (const auto& slot : slots) {
            if (slot.waited && !slot.released) { (void)cudaStreamSynchronize(slot.consumer); }
        }
        for (std::size_t i = 0; i < consumed.size(); ++i) {
            if (consumed_recorded[i]) { (void)cudaEventSynchronize(consumed[i]); }
        }
        for (auto& buffers : {&ready, &consumed, &producer_ready, &writeback_done}) {
            for (auto event : *buffers) { if (event) { (void)cudaEventDestroy(event); } }
        }
        for (auto& slot : ring) { if (slot.done) { (void)cudaEventDestroy(slot.done); } }
        if (stream) { (void)cudaStreamDestroy(stream); }
        if (writeback_stream) { (void)cudaStreamDestroy(writeback_stream); }
    }
    void initialize() {
        KV_TRANSFER_CHECK(cudaGetDevice(&device));
        KV_TRANSFER_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        KV_TRANSFER_CHECK(cudaStreamCreateWithFlags(&writeback_stream, cudaStreamNonBlocking));
        slots.resize(plan.ticket_capacity);
        ready.resize(slots.size());
        if (fault.fail_consumed_flags_allocation) throw std::bad_alloc();
        consumed_recorded.resize(slots.size());
        // Destruction indexes flags for every consumed event, including when a
        // later allocation fails. Allocate the index domain first.
        consumed.resize(slots.size());
        producer_ready.resize(archive.layout().layers.size());
        writeback_done.resize(producer_ready.size());
        writeback_fence_recorded.resize(producer_ready.size());
        if (fault.fail_producer_borrowed_flags_allocation) throw std::bad_alloc();
        producer_borrowed.resize(producer_ready.size());
        // Destruction indexes borrower flags for every producer stream. Keep
        // that index domain valid even when the stream allocation fails.
        producer_streams.resize(producer_ready.size());
        for (auto& buffers : {&ready, &consumed, &producer_ready, &writeback_done}) {
            for (auto& event : *buffers) {
                KV_TRANSFER_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            }
        }
        if (archive.mode() == HostArchiveMode::Pageable) {
            for (auto& slot : ring) {
                slot.memory = std::make_unique<CheckedPinnedHostBuffer>(kHostKVTransferTileBytes);
                KV_TRANSFER_CHECK(cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming));
            }
        }
        start_workers();
    }
    void start_workers() {
        {
            std::lock_guard lock(mutex);
            starting = true;
            stopping = true;
        }
        try {
            ++worker_start_attempt;
            auto started = std::make_shared<std::promise<void>>();
            auto startup = started->get_future();
            worker = std::thread([this, started] { run(started, false); });
            startup.get();
            auto wb_started = std::make_shared<std::promise<void>>();
            auto wb_startup = wb_started->get_future();
            if (fault.second_worker_start_failure_attempt == worker_start_attempt) {
                fault.second_worker_start_failure_attempt = 0;
                throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again),
                                        "injected second KV transfer worker creation failure");
            }
            writeback_worker = std::thread([this, wb_started] { run(wb_started, true); });
            wb_startup.get();
        } catch (...) {
            const auto startup_error = std::current_exception();
            {
                std::lock_guard lock(mutex);
                poison_locked(startup_error);
                stopping = true;
                starting = false;
            }
            changed.notify_all();
            if (worker.joinable()) worker.join();
            if (writeback_worker.joinable()) writeback_worker.join();
            std::rethrow_exception(startup_error);
        }
        {
            std::lock_guard lock(mutex);
            error = {};
            failure = {};
            stopping = false;
            starting = false;
        }
        changed.notify_all();
    }
    void check_error() const {
        if (error) { std::rethrow_exception(error); }
        if (failure.failed) throw std::runtime_error("host KV transfer failed without a recorded cause");
        if (stopping) throw std::logic_error("host KV transfer owner is quiesced");
    }
    // Caller holds mutex. Poison cancels scheduling immediately. Canceled jobs
    // retire after draining producer/source lifetimes; popped jobs retire their
    // own promises after their exception drains.
    void poison_locked(std::exception_ptr cause, bool uncertain = false) {
        prefer_unrecoverable_exception(error, cause);
        failure.failed = true;
        failure.archive_bytes_uncertain |= uncertain;
        try { std::rethrow_exception(cause); }
        catch (const CudaTransferError& cuda) {
            if (failure.cuda_status == cudaSuccess) {
                failure.cuda_status = cuda.status();
                failure.cuda_operation = cuda.operation();
            }
            failure.unrecoverable |= cuda_transfer_failure_is_unrecoverable(cuda.status(), cuda.drain_failed());
            if (cuda.drain_failed()) {
                failure.unrecoverable = true;
                if (failure.drain_status == cudaSuccess) {
                    failure.drain_status = cuda.status();
                    failure.drain_operation = cuda.operation();
                }
            }
        } catch (...) {}
        changed.notify_all();
    }
    void cancel_queued_locked() {
        for (auto* queue : {&jobs, &writeback_jobs}) {
            for (auto& job : *queue) if (job.completion) job.completion->set_exception(error);
            queue->clear();
        }
    }
    void poison(std::exception_ptr cause, bool uncertain = false) {
        std::lock_guard lock(mutex);
        poison_locked(cause, uncertain);
    }
    bool record_drain(cudaError_t status, const char* operation) noexcept {
        if (status == cudaSuccess) return true;
        std::lock_guard lock(mutex);
        failure.failed = true;
        failure.unrecoverable = true; // A failed drain never establishes lifetime safety.
        if (failure.drain_status == cudaSuccess) {
            failure.drain_status = status;
            failure.drain_operation = operation;
        }
        try { poison_locked(std::make_exception_ptr(CudaTransferError(status, operation, true))); }
        catch (...) { stopping = true; changed.notify_all(); }
        return false;
    }
    void drain_producers() {
        for (std::size_t i = 0; i < producer_streams.size(); ++i) {
            cudaStream_t producer = nullptr;
            bool borrowed = false;
            {
                std::lock_guard lock(mutex);
                borrowed = producer_borrowed[i];
                producer = producer_streams[i];
            }
            if (borrowed && record_drain(cudaStreamSynchronize(producer), "canceled producer stream drain")) {
                std::lock_guard lock(mutex); producer_borrowed[i] = false;
            }
        }
    }
    void retire_canceled() {
        drain_producers();
        std::lock_guard lock(mutex);
        if (error) cancel_queued_locked();
    }
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
    // Both workers share the same four startup slots. A D2H worker waits for
    // its producer before claiming a slot; it never holds up H2D for that wait.
    std::size_t acquire_ring() {
        std::size_t chosen = 0;
        {
            std::unique_lock lock(mutex);
            changed.wait(lock, [&] {
                return error || std::any_of(ring.begin(), ring.end(),
                    [](const RingSlot& slot) { return !slot.claimed; });
            });
            check_error();
            for (std::size_t n = 0; n < ring.size(); ++n) {
                chosen = ring_cursor++ % ring.size();
                if (!ring[chosen].claimed) { ring[chosen].claimed = true; break; }
            }
        }
        try {
            if (ring[chosen].used) { KV_TRANSFER_DRAIN_CHECK(cudaEventSynchronize(ring[chosen].done)); }
        } catch (...) {
            poison(std::current_exception());
            // The preceding DMA completion could not be established. Keep this
            // slot claimed and fail closed even for a nonfatal event API error.
            (void)record_drain(cudaStreamSynchronize(stream), "acquire ring H2D drain");
            (void)record_drain(cudaStreamSynchronize(writeback_stream), "acquire ring D2H drain");
            { std::lock_guard lock(mutex); failure.unrecoverable = true; }
            throw;
        }
        return chosen;
    }
    void release_ring(std::size_t slot) noexcept {
        { std::lock_guard lock(mutex); ring[slot].claimed = false; }
        changed.notify_all();
    }
    struct RingLease {
        Impl& owner;
        std::size_t index;
        cudaStream_t dma_stream;
        int exceptions = std::uncaught_exceptions();
        RingLease(Impl& o, cudaStream_t s) : owner(o), index(o.acquire_ring()), dma_stream(s) {}
        RingLease(const RingLease&) = delete;
        RingLease& operator=(const RingLease&) = delete;
        ~RingLease() {
            // A failed event record must not publish a slot while its DMA still
            // uses host bytes. Drain before another worker can claim it.
            if (std::uncaught_exceptions() > exceptions &&
                !owner.record_drain(cudaStreamSynchronize(dma_stream), "ring lease unwind drain")) return;
            owner.release_ring(index);
        }
        RingSlot& slot() { return owner.ring[index]; }
    };
    void process_prefetch(const Job& job) {
        if (fault.pause_before_h2d_copy && h2d_copies + 1 == fault.pause_before_h2d_copy &&
            fault.h2d_paused && fault.resume_h2d) {
            fault.h2d_paused->store(true);
            while (!fault.resume_h2d->load()) std::this_thread::yield();
        }
        for (auto dependency : job.dependencies) {
            // Reference counts prevent another ticket from re-recording this event
            // until this stream has captured its precise prior-consumer generation.
            KV_TRANSFER_CHECK(cudaStreamWaitEvent(stream, consumed[dependency], 0));
            std::lock_guard lock(mutex);
            --slots[dependency].references;
            changed.notify_all();
        }
        Slot target;
        { std::lock_guard lock(mutex); target = slots[job.slot]; }
        const void* source = job.sources.front().data();
        std::optional<RingLease> lease;
        RingSlot* host_slot = nullptr;
        if (archive.mode() == HostArchiveMode::Pageable) {
            lease.emplace(*this, stream);
            host_slot = &lease->slot();
            std::size_t packed = 0;
            for (auto run : job.sources) {
                std::memcpy(static_cast<std::byte*>(host_slot->memory->data()) + packed,
                            run.data(), run.size());
                packed += run.size();
            }
            source = host_slot->memory->data();
        }
        if (host_slot) {
            KV_TRANSFER_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(staging.data) + target.offset,
                source, target.bytes, cudaMemcpyHostToDevice, stream));
        } else {
            std::size_t packed = 0;
            for (auto run : job.sources) {
                KV_TRANSFER_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(staging.data) + target.offset + packed,
                    run.data(), run.size(), cudaMemcpyHostToDevice, stream));
                packed += run.size();
            }
        }
        if (host_slot) {
            KV_TRANSFER_CHECK(cudaEventRecord(host_slot->done, stream));
            host_slot->used = true;
        }
        ++h2d_copies;
        if (fault.reject_h2d_submission_after == h2d_copies) {
            KV_TRANSFER_DRAIN_CHECK(cudaStreamSynchronize(stream));
            { std::lock_guard lock(mutex); failure.completed_h2d_copies = h2d_copies; }
            KV_TRANSFER_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(staging.data) + target.offset,
                source, job.sources.front().size(), static_cast<cudaMemcpyKind>(99), stream));
        }
        if (fault.h2d_copy_after && h2d_copies == fault.h2d_copy_after) {
            KV_TRANSFER_DRAIN_CHECK(cudaStreamSynchronize(stream));
            { std::lock_guard lock(mutex); failure.completed_h2d_copies = h2d_copies; }
            throw std::runtime_error("injected host KV failure after completed H2D copy");
        }
        KV_TRANSFER_CHECK(cudaEventRecord(ready[job.slot], stream));
        { std::lock_guard lock(mutex); slots[job.slot].enqueued = true; }
        changed.notify_all();
    }
    void pool_copy(const Job& job, std::size_t plane, std::size_t begin, std::size_t count,
                    void* destination) {
        const auto& layout = archive.layout().layers[job.layer][plane];
        const auto ids = std::span<const std::int32_t>(job.pages).subspan(begin, count);
        KV_TRANSFER_CHECK(job.pool->copy_pages_to_host_status(layout.pool_plane, ids,
            destination, writeback_stream, PagedKVPool::HostCopyLayout::IndividualPages));
    }
    void process_writeback(Job& job) {
        // Waiting here blocks only the second worker, never the prefetch worker.
        KV_TRANSFER_DRAIN_CHECK(cudaEventSynchronize(producer_ready[job.layer]));
        {
            std::unique_lock lock(mutex);
            producer_borrowed[job.layer] = false; // Future now permits producer lifetime release.
            changed.wait(lock, [&] { return error || writeback_fence_recorded[job.layer]; });
            check_error();
        }
        // The H2D FIFO records this generation after every earlier archive read.
        // The layer's pending future prevents another fence from re-recording it
        // until the final D2H event below has completed.
        KV_TRANSFER_CHECK(cudaStreamWaitEvent(writeback_stream, writeback_done[job.layer], 0));
        for (std::size_t plane = 0; plane < job.destinations.size(); ++plane) {
            { std::lock_guard lock(mutex); check_error(); }
            const auto page_bytes = archive.layout().layers[job.layer][plane].page_bytes;
            if (archive.mode() == HostArchiveMode::Pinned) {
                job.destination_touched = true;
                pool_copy(job, plane, 0, job.pages.size(), job.destinations[plane].data());
            } else {
                const auto per_tile = kHostKVTransferTileBytes / page_bytes;
                if (!per_tile) { throw std::invalid_argument("KV page exceeds fixed transfer tile"); }
                for (std::size_t begin = 0; begin < job.pages.size(); begin += per_tile) {
                    const auto count = std::min(per_tile, job.pages.size() - begin);
                    RingLease lease(*this, writeback_stream);
                    auto& slot = lease.slot();
                    pool_copy(job, plane, begin, count, slot.memory->data());
                    KV_TRANSFER_CHECK(cudaEventRecord(slot.done, writeback_stream));
                    slot.used = true;
                    KV_TRANSFER_DRAIN_CHECK(cudaEventSynchronize(slot.done));
                    job.destination_touched = true;
                    std::memcpy(job.destinations[plane].data() + begin * page_bytes,
                                slot.memory->data(), count * page_bytes);
                }
            }
            ++d2h_planes;
            if (fault.pause_after_d2h_plane == d2h_planes && fault.d2h_paused && fault.resume_d2h) {
                fault.d2h_paused->store(true);
                while (!fault.resume_d2h->load()) std::this_thread::yield();
            }
            if (fault.reject_d2h_submission_after == d2h_planes) {
                KV_TRANSFER_DRAIN_CHECK(cudaStreamSynchronize(writeback_stream));
                { std::lock_guard lock(mutex); failure.completed_d2h_planes = d2h_planes; }
                KV_TRANSFER_CHECK(cudaMemcpyAsync(job.destinations[plane].data(),
                    job.pool->plane(archive.layout().layers[job.layer][plane].pool_plane).data,
                    page_bytes, static_cast<cudaMemcpyKind>(99), writeback_stream));
            }
            if (fault.d2h_plane_after && d2h_planes == fault.d2h_plane_after) {
                KV_TRANSFER_DRAIN_CHECK(cudaStreamSynchronize(writeback_stream));
                { std::lock_guard lock(mutex); failure.completed_d2h_planes = d2h_planes; }
                throw std::runtime_error("injected host KV failure after completed D2H plane");
            }
        }
        KV_TRANSFER_CHECK(cudaEventRecord(writeback_done[job.layer], writeback_stream));
        KV_TRANSFER_DRAIN_CHECK(cudaEventSynchronize(writeback_done[job.layer]));
        { std::lock_guard lock(mutex); check_error(); job.completion->set_value(); }
    }
    void run(const std::shared_ptr<std::promise<void>>& started, bool writebacks) noexcept {
        auto& queue = writebacks ? writeback_jobs : jobs;
        auto& busy = writebacks ? writeback_active : active;
        try { KV_TRANSFER_CHECK(cudaSetDevice(device)); started->set_value(); }
        catch (...) {
            const auto cause = std::current_exception();
            poison(cause);
            started->set_exception(cause);
            changed.notify_all();
            return;
        }
        for (;;) {
            Job job;
            {
                std::unique_lock lock(mutex);
                changed.wait(lock, [&] { return !starting && (error || stopping || !queue.empty()); });
                if (error) { return; }
                if (queue.empty()) { if (stopping) { return; } continue; }
                job = std::move(queue.front());
                queue.pop_front();
                busy = true;
            }
            try {
                if (job.archive_read_fence) {
                    KV_TRANSFER_CHECK(cudaEventRecord(writeback_done[job.layer], stream));
                    { std::lock_guard lock(mutex); writeback_fence_recorded[job.layer] = true; }
                    changed.notify_all();
                } else if (job.writeback) { process_writeback(job); }
                else { process_prefetch(job); }
            } catch (...) {
                const auto cause = std::current_exception();
                poison(cause, job.writeback && job.destination_touched);
                (void)record_drain(cudaStreamSynchronize(writebacks ? writeback_stream : stream),
                                   "worker exception DMA drain");
                retire_canceled();
                std::lock_guard lock(mutex);
                if (job.completion) job.completion->set_exception(error);
                busy = false;
                changed.notify_all();
                return;
            }
            { std::lock_guard lock(mutex); busy = false; }
            changed.notify_all();
        }
    }
    HostKVTransferFailureState quiesce() {
        bool failed = false;
        {
            std::unique_lock lock(mutex);
            changed.wait(lock, [&] { return error || (!active && !writeback_active && jobs.empty() && writeback_jobs.empty()); });
            failed = bool(error);
            if (failed) stopping = true;
        }
        changed.notify_all();
        if (failed) {
            if (worker.joinable()) worker.join();
            if (writeback_worker.joinable()) writeback_worker.join();
        }
        // Drain every submitted stream even if one of the drains fails. A fatal
        // CUDA context cannot be repaired with the existing allocations.
        (void)record_drain(cudaStreamSynchronize(stream), "quiesce H2D stream");
        (void)record_drain(cudaStreamSynchronize(writeback_stream), "quiesce D2H stream");
        for (std::size_t i = 0; i < slots.size(); ++i) {
            if (consumed_recorded[i])
                (void)record_drain(cudaEventSynchronize(consumed[i]), "quiesce released consumer");
            if (slots[i].waited && !slots[i].released)
                (void)record_drain(cudaStreamSynchronize(slots[i].consumer), "quiesce consumer stream");
        }
        // A rejected producer-event record does not cover its current stream
        // generation. Retain the caller stream independently before recording.
        drain_producers();
        for (std::size_t i = 0; i < producer_ready.size(); ++i) {
            (void)record_drain(cudaEventSynchronize(producer_ready[i]), "quiesce producer event");
        }
        for (auto& slot : ring) if (slot.used)
            (void)record_drain(cudaEventSynchronize(slot.done), "quiesce ring event");
        // A drain can be the first observed error. Stop and join workers after
        // poisoning too; no live worker survives a failed quiesce.
        bool failed_after_drain = false;
        { std::lock_guard lock(mutex); failed_after_drain = bool(error); if (error) stopping = true; }
        changed.notify_all();
        if (failed_after_drain) {
            if (worker.joinable()) worker.join();
            if (writeback_worker.joinable()) writeback_worker.join();
        }
        std::lock_guard lock(mutex);
        if (error) cancel_queued_locked();
        return failure;
    }
    void synchronize() {
        (void)quiesce();
        std::lock_guard lock(mutex);
        check_error();
    }
};

HostKVTransferEngine::HostKVTransferEngine(HostKVArchive& archive, DeviceSpan staging,
                                           HostKVStagingPlan plan)
    : HostKVTransferEngine(archive, staging, plan, {}) {}

HostKVTransferEngine::HostKVTransferEngine(HostKVArchive& archive, DeviceSpan staging,
                                           HostKVStagingPlan plan, HostKVTransferFaultInjection fault)
    : archive_(archive), impl_(std::make_unique<Impl>(archive, staging, plan, fault)) {
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
              << " copy_workers=2 pinned_ring_bytes=" << pinned_ring_bytes() << '\n';
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
    return prefetch_impl(layer, plane, first, count, offset, false);
}
HostKVTransferTicket HostKVTransferEngine::prefetch_completed(
    std::size_t layer, std::size_t plane, std::uint32_t first, std::uint32_t count,
    std::size_t offset) {
    return prefetch_impl(layer, plane, first, count, offset, true);
}
HostKVTransferTicket HostKVTransferEngine::prefetch_impl(
    std::size_t layer, std::size_t plane, std::uint32_t first, std::uint32_t count,
    std::size_t offset, bool completed_only) {
    try {
        auto source = completed_only ? archive_.completed_pages(layer, plane, first, count)
                                     : archive_.pages(layer, plane, first, count);
        return enqueue_prefetch(source, offset);
    } catch (const CudaTransferError& cuda) {
        impl_->poison(std::current_exception());
        if (cuda.drain_failed()) (void)impl_->record_drain(cuda.status(), cuda.operation());
        impl_->retire_canceled();
        throw;
    }
}
HostKVTransferTicket HostKVTransferEngine::prefetch_gather_completed(
    std::size_t layer, std::size_t plane, std::span<const std::uint32_t> pages,
    std::size_t offset) {
    const auto page_bytes = archive_.layout().layers.at(layer).at(plane).page_bytes;
    if (pages.empty() || pages.size() > kHostKVTransferTileBytes / page_bytes)
        throw std::invalid_argument("gather exceeds fixed transfer tile");
    std::vector<std::span<const std::byte>> runs;
    for (std::size_t first = 0; first < pages.size();) {
        if (first && pages[first-1] >= pages[first])
            throw std::invalid_argument("gather requires sorted unique original logical IDs");
        std::size_t count = 1;
        while (first + count < pages.size() && pages[first + count] == pages[first] + count) ++count;
        runs.push_back(archive_.completed_pages(layer, plane, pages[first], std::uint32_t(count)));
        first += count;
    }
    return enqueue_gather(std::move(runs), pages.size() * page_bytes, offset);
}
HostKVTransferTicket HostKVTransferEngine::prefetch_index(
    std::span<const std::byte> source, std::size_t offset) {
    if (source.empty() || !source.data()) {
        throw std::invalid_argument("Mean-K index source is empty");
    }
    if (archive_.mode() == HostArchiveMode::Pinned) {
        unsigned flags = 0;
        auto result = cudaHostGetFlags(&flags, const_cast<std::byte*>(source.data()));
        if (result != cudaSuccess) {
            if (result != cudaErrorInvalidValue) {
                try { check_transfer_cuda(result, "Mean-K pinned source query"); }
                catch (...) { impl_->poison(std::current_exception()); impl_->retire_canceled(); throw; }
            }
            (void)cudaGetLastError();
            throw std::invalid_argument("direct Mean-K transfer requires CUDA-pinned memory");
        }
        if (flags & cudaHostAllocWriteCombined) {
            throw std::invalid_argument("Mean-K index must use cacheable pinned memory");
        }
    }
    return enqueue_prefetch(source, offset);
}
HostKVTransferTicket HostKVTransferEngine::enqueue_prefetch(
    std::span<const std::byte> source, std::size_t offset) {
    return enqueue_gather({source}, source.size(), offset);
}
HostKVTransferTicket HostKVTransferEngine::enqueue_gather(
    std::vector<std::span<const std::byte>> sources, std::size_t bytes, std::size_t offset) {
    if (!bytes || bytes > kHostKVTransferTileBytes ||
        offset > impl_->plan.capacity_bytes || bytes > impl_->plan.capacity_bytes - offset) {
        throw std::invalid_argument("KV prefetch tile or device offset exceeds startup capacity");
    }
    auto& state = *impl_;
    std::unique_lock lock(state.mutex);
    state.check_error();
    if (std::none_of(state.slots.begin(), state.slots.end(), [](const auto& slot) {
            return !slot.sequence || (slot.released && !slot.references);
        }) && std::any_of(state.slots.begin(), state.slots.end(), [](const auto& slot) {
            return slot.released;
        })) {
        state.changed.wait(lock, [&] {
            return state.error || std::any_of(state.slots.begin(), state.slots.end(), [](const auto& slot) {
                return slot.released && !slot.references;
            });
        });
        state.check_error();
    }
    if (state.next_sequence == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("KV transfer sequence exhausted");
    }
    std::size_t selected = state.slots.size();
    std::size_t unused = state.slots.size();
    for (std::size_t i = 0; i < state.slots.size(); ++i) {
        const auto& slot = state.slots[i];
        if (slot.sequence && !slot.released && overlaps(offset, bytes, slot.offset, slot.bytes)) {
            throw std::logic_error("prefetch would overwrite a live staging segment");
        }
        if (!slot.sequence && unused == state.slots.size()) { unused = i; }
        if (slot.released && slot.references == 0 && selected == state.slots.size()) {
            selected = i;
        }
    }
    if (selected == state.slots.size()) {
        selected = unused;
    }
    if (selected == state.slots.size()) {
        std::size_t references = 0;
        for (const auto& slot : state.slots) { references += slot.references; }
        throw std::runtime_error("startup KV transfer ticket capacity exhausted: capacity=" +
            std::to_string(state.slots.size()) +
            " references=" + std::to_string(references));
    }
    Impl::Job job;
    job.slot = selected;
    job.sources = std::move(sources);
    for (std::size_t i = 0; i < state.slots.size(); ++i) {
        const auto& slot = state.slots[i];
        if (slot.sequence && slot.released &&
            (i == selected || overlaps(offset, bytes, slot.offset, slot.bytes))) {
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
    slot.bytes = bytes;
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
    slot.waited = true;
    slot.consumer = stream;
    try {
        KV_TRANSFER_CHECK(cudaStreamWaitEvent(stream, state.ready[ticket.slot],
            state.fault.reject_consumer_wait ? 0x80000000U : 0));
    } catch (...) {
        state.poison_locked(std::current_exception());
        lock.unlock();
        state.retire_canceled();
        throw;
    }
}
DeviceSpan HostKVTransferEngine::staged(HostKVTransferTicket ticket) {
    std::lock_guard lock(impl_->mutex);
    const auto& slot = impl_->validate(ticket);
    if (slot.released) { throw std::logic_error("released KV staging range is no longer readable"); }
    return {static_cast<std::byte*>(impl_->staging.data) + slot.offset, slot.bytes};
}
void HostKVTransferEngine::release(HostKVTransferTicket ticket, cudaStream_t stream) {
    std::unique_lock lock(impl_->mutex);
    auto& slot = impl_->validate(ticket);
    if (!slot.waited || slot.released || slot.consumer != stream) {
        throw std::logic_error("KV release must follow its consumer wait and kernel submission");
    }
    try {
        KV_TRANSFER_CHECK(cudaEventRecordWithFlags(impl_->consumed[ticket.slot], stream,
            impl_->fault.reject_consumer_release ? 0x80000000U : 0));
    } catch (...) {
        impl_->poison_locked(std::current_exception());
        lock.unlock();
        impl_->retire_canceled();
        throw;
    }
    impl_->consumed_recorded[ticket.slot] = true;
    slot.released = true;
}
std::shared_future<void> HostKVTransferEngine::writeback(
    const PagedKVPool& pool, std::size_t layer, std::uint32_t first,
    std::span<const std::int32_t> pages, std::uint32_t frontier, cudaStream_t producer) {
    { std::lock_guard lock(impl_->mutex); impl_->check_error(); }
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
    // Fence earlier H2D archive readers before the separate D2H stream overwrites
    // their source bytes. Unrelated later prefetches remain independent.
    Impl::Job job;
    job.writeback = true;
    job.layer = layer;
    job.pool = &pool;
    job.pages.assign(pages.begin(), pages.end());
    job.completion = std::make_shared<std::promise<void>>();
    const auto completion = job.completion;
    auto completed = job.completion->get_future().share();
    bool archive_prepared = false;
    bool producer_registered = false;
    try {
        job.destinations = archive_.prepare_async_writeback(layer, first,
            static_cast<std::uint32_t>(pages.size()), frontier, completed);
        archive_prepared = true;
        std::lock_guard lock(impl_->mutex);
        impl_->check_error();
        impl_->producer_streams[layer] = producer;
        impl_->producer_borrowed[layer] = true;
        producer_registered = true;
        KV_TRANSFER_CHECK(cudaEventRecordWithFlags(impl_->producer_ready[layer], producer,
            impl_->fault.reject_producer_record ? 0x80000000U : 0));
        Impl::Job fence;
        fence.archive_read_fence = true;
        fence.layer = layer;
        // Queue allocation is transactional while the workers cannot pop either
        // node. This fence owns no reusable ticket/event reference.
        impl_->jobs.push_back(std::move(fence));
        try { impl_->writeback_jobs.push_back(std::move(job)); }
        catch (...) { impl_->jobs.pop_back(); throw; }
        impl_->writeback_fence_recorded[layer] = false;
        impl_->changed.notify_all();
    } catch (...) {
        const auto cause = std::current_exception();
        // Includes producer-event rejection and indirect archive synchronization.
        // Ordinary range/ownership validation does not poison a healthy owner.
        if (archive_prepared) impl_->poison(cause);
        try { std::rethrow_exception(cause); }
        catch (const CudaTransferError& cuda) {
            impl_->poison(cause);
            if (!archive_prepared && cuda.drain_failed())
                (void)impl_->record_drain(cuda.status(), cuda.operation());
        }
        catch (...) {}
        if (producer_registered || archive_prepared) impl_->retire_canceled();
        completion->set_exception(cause);
        throw;
    }
    return completed;
}
void HostKVTransferEngine::set_test_fault(HostKVTransferFaultInjection fault) {
    synchronize();
    std::lock_guard lock(impl_->mutex);
    impl_->fault = fault; impl_->h2d_copies = impl_->d2h_planes = 0;
}
void HostKVTransferEngine::synchronize() { impl_->synchronize(); }
HostKVTransferFailureState HostKVTransferEngine::quiesce() { return impl_->quiesce(); }
HostKVTransferFailureState HostKVTransferEngine::failure_state() const {
    std::lock_guard lock(impl_->mutex);
    auto result = impl_->failure;
    result.failed |= bool(impl_->error);
    return result;
}
HostKVTransferFailureState HostKVTransferEngine::recover() {
    auto& state = *impl_;
    const auto previous = state.quiesce();
    throw_if_unrecoverable();
    try { archive_.retire_transfer_completions(previous.archive_bytes_uncertain); }
    catch (const CudaTransferError& cuda) {
        impl_->poison(std::current_exception());
        (void)impl_->record_drain(cuda.status(), cuda.operation());
        throw;
    }
    {
        std::lock_guard lock(state.mutex);
        for (auto& slot : state.slots) slot = {};
        std::fill(state.consumed_recorded.begin(), state.consumed_recorded.end(), false);
        std::fill(state.writeback_fence_recorded.begin(), state.writeback_fence_recorded.end(), false);
        for (auto& slot : state.ring) { slot.used = false; slot.claimed = false; }
        state.fault.h2d_copy_after = state.fault.d2h_plane_after = 0;
        state.fault.reject_h2d_submission_after = state.fault.reject_d2h_submission_after = 0;
        state.fault.pause_after_d2h_plane = 0;
        state.fault.d2h_paused = state.fault.resume_d2h = state.fault.selection_after_exact = nullptr;
        state.fault.reject_consumer_wait = state.fault.reject_consumer_release = false;
        state.fault.reject_producer_record = false;
        std::fill(state.producer_borrowed.begin(), state.producer_borrowed.end(), false);
        state.active = state.writeback_active = false;
    }
    if (previous.failed) state.start_workers();
    return previous;
}
void HostKVTransferEngine::reset() { (void)recover(); }
void HostKVTransferEngine::trim(std::uint32_t frontier) {
    reset();
    archive_.trim_owned(frontier);
}
void HostKVTransferEngine::throw_if_unrecoverable() const {
    std::lock_guard lock(impl_->mutex);
    if (impl_->failure.unrecoverable) {
        if (impl_->error) std::rethrow_exception(impl_->error);
        throw std::runtime_error("unrecoverable CUDA KV transfer failure without a recorded cause");
    }
}
void HostKVTransferTestAccess::poison(HostKVTransferEngine& owner, std::exception_ptr cause) {
    owner.impl_->poison(std::move(cause));
}
void HostKVTransferTestAccess::record_drain(HostKVTransferEngine& owner, cudaError_t status, const char* operation) {
    (void)owner.impl_->record_drain(status, operation);
}
void HostKVTransferEngine::commit_empty_reset() noexcept { archive_.commit_empty_frontiers(); }

} // namespace ninfer::kvmem

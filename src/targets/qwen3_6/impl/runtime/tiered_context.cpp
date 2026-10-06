#include "targets/qwen3_6/impl/runtime/tiered_context.h"
#include "targets/qwen3_6/impl/runtime/tiered_context_test_access.h"
#include "core/kvmem/host_kv_transfer_test_access.h"
#include "core/device.h"
#include "core/kvmem/checked_buffers.h"
#include <ninfer/ops/kvmem_score.h>
#include <cstdlib>
#include <optional>
#define OWNER_CHECK(call) checked((call), #call)
#include "core/kvmem/kv_view_table.h"
#include <ninfer/ops/gqa_attention.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <thread>

namespace ninfer::targets::qwen3_6::detail {
namespace {
std::uint32_t page_count(std::uint32_t tokens) { return tokens / 64 + (tokens % 64 != 0); }

std::atomic<std::uint64_t> next_bundle_identity{1};

struct SparseLoading {
    std::unique_ptr<kvmem::HostKVArchive> archive;
    std::unique_ptr<SparseCaptureOwner> capture;
    std::unique_ptr<kvmem::CheckedPinnedHostBuffer> score;
};
SparseLoading load_owner_resources(const TieredRuntimePlan& p, cudaStream_t compute) {
    if (!p.sparse_capture)
        return {std::make_unique<kvmem::HostKVArchive>(p.archive, p.archive_mode, p.lock_archive), {}, {}};
    auto attempt = [&](kvmem::HostArchiveMode requested) {
        kvmem::check_host_archive_admission(sparse_host_payload_bytes(p, requested == kvmem::HostArchiveMode::Pageable),
                                          kvmem::available_physical_memory_bytes());
        SparseLoading r;
        r.archive = std::make_unique<kvmem::HostKVArchive>(p.archive, requested, p.lock_archive);
        r.capture = std::make_unique<SparseCaptureOwner>(*p.sparse_capture,
            r.archive->mode() == kvmem::HostArchiveMode::Pinned, compute);
        r.score = std::make_unique<kvmem::CheckedPinnedHostBuffer>(p.host_score_bytes);
        if (!kvmem::keep_pinned_archive(true, kvmem::available_physical_memory_bytes()))
            throw kvmem::HostMemoryAdmissionError("combined KVMem pinned resources lost physical headroom");
        return r;
    };
    const auto first = sparse_first_loading_mode(p.archive_mode, p.lock_archive);
    if (first == kvmem::HostArchiveMode::Pageable) return attempt(first);
    try { return attempt(first); }
    catch (const kvmem::CudaTransferError& error) {
        if (p.archive_mode != kvmem::HostArchiveMode::Auto ||
            error.status() != cudaErrorMemoryAllocation || error.drain_failed()) throw;
        (void)cudaGetLastError();
    } catch (const kvmem::HostMemoryAdmissionError&) {
        if (p.archive_mode != kvmem::HostArchiveMode::Auto) throw;
    } catch (const std::bad_alloc&) {
        if (p.archive_mode != kvmem::HostArchiveMode::Auto) throw;
    }
    // Every attempted resource is unwound before constructing the pageable owner/ring.
    return attempt(kvmem::HostArchiveMode::Pageable);
}
double wall_ms(std::chrono::steady_clock::time_point first) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - first).count();
}
float bf16_float(std::uint16_t bits) { return std::bit_cast<float>(std::uint32_t(bits) << 16); }
} // namespace

struct TieredContext::Impl {
    std::uint64_t bundle_identity = next_bundle_identity.fetch_add(1);
    std::vector<std::weak_ptr<TieredSnapshotLifetime>> captures;
    TieredRuntimePlan plan;
    PagedKVCache& main;
    DeviceSpan backing;
    cudaStream_t compute;
    kvmem::KVViewTable table;
    SparseLoading loading;
    kvmem::HostKVArchive& archive;
    kvmem::HostKVTransferEngine transfer;
    std::unique_ptr<SparseCaptureOwner>& sparse;
    SparseExecutionPhase sparse_phase = SparseExecutionPhase::ExactPrefill;
    std::vector<std::uint32_t> exact_hard;
    std::optional<kvmem::MeanKTransactionKind> transaction_kind;
    std::uint32_t transaction_first{}, transaction_count{};
    std::uint64_t caller_bundle{};
    double capture_wait_ms{};
    kvmem::HostKVTransferFaultInjection test_fault;
    bool poisoned = false, unrecoverable = false;
    std::exception_ptr unrecoverable_cause;
    std::array<cudaEvent_t, 2> score_events{};
    std::vector<std::int32_t> lease_ids, block_table, write_ids;
    std::vector<kvmem::KVViewWrite> writes;
    kvmem::KVViewSnapshot snapshot;
    std::vector<ops::AttentionPageAccess> access;
    std::vector<std::int32_t> prefix;
    std::array<std::shared_future<void>, 16> pending;
    std::vector<kvmem::HostKVTransferTicket> tickets;
    std::array<cudaEvent_t, 16> wait_start{}, wait_end{};
    std::array<bool, 16> measured{};
    std::array<std::array<double, 16>, 2> wait_ms{}, enqueue_wait_ms{};
    std::size_t execution_phase = 0;
    std::unique_ptr<PinnedHostBuffer> shadow_dense, shadow_tiered;
    std::uint32_t block_tokens = 0, current_frontier = 0, next_layer = 16;
    std::uint64_t blocks = 0, streamed_bytes = 0, partial_passes = 0;
    double shadow_max_relative = 0, shadow_max_abs = 0;
    std::array<double, 16> shadow_relative{}, shadow_absolute{};
    bool shadow_fp64_reference = false;

    Impl(const TieredRuntimePlan& p, PagedKVCache& m, DeviceSpan b, cudaStream_t stream,
         kvmem::HostKVTransferFaultInjection fault)
        : plan(p), main(m), backing(b), compute(stream),
          table(p.logical_tokens, p.view_pages, p.sink_pages, 16),
          loading(load_owner_resources(p, stream)), archive(*loading.archive),
          transfer(archive, p.staging_region.bind(b), p.staging, fault), sparse(loading.capture), test_fault(fault) {
        if (m.layers() != 16 || b.bytes < p.bytes ||
            p.staging_plane_regions.size() != p.archive.layers.front().size() ||
            (!p.shadow_validate && m.pool().page_group_count() < p.view_pages) ||
            (p.shadow_validate && m.pool().page_group_count() < p.archive.logical_pages)) {
            throw std::invalid_argument(
                "tiered owner backing or Main pool does not match its plan");
        }
        block_table.resize(p.archive.logical_pages, -1);
        lease_ids.reserve(p.shadow_validate ? p.archive.logical_pages : p.view_pages);
        tickets.reserve(p.staging.ticket_capacity);
        if (p.shadow_validate) {
            shadow_fp64_reference = tiered_shadow_fp64_required(p.logical_tokens, resident(0));
            shadow_dense  = std::make_unique<PinnedHostBuffer>(p.shadow_output.region.bytes);
            shadow_tiered = std::make_unique<PinnedHostBuffer>(p.shadow_output.region.bytes);
            if (shadow_fp64_reference)
                std::clog << "[kvmem-shadow] criterion=offline_fp64 status=not_evaluated; "
                             "capture selected late-prefill blocks and require tiered/FP64 <= "
                             "dense/FP64 per layer; process exit 0 is not a gate verdict\n";
        }
        try {
            if (sparse) {
                for (auto& event : score_events) OWNER_CHECK(cudaEventCreate(&event));
                if (kvmem::available_physical_memory_bytes() < kvmem::kHostArchivePhysicalHeadroom)
                    throw std::runtime_error("combined KVMem resources lost physical headroom");
            }
            if (p.measure_transfer_waits) {
                for (std::size_t layer = 0; layer < 16; ++layer) {
                    OWNER_CHECK(cudaEventCreate(&wait_start[layer]));
                    OWNER_CHECK(cudaEventCreate(&wait_end[layer]));
                }
            }
            // Main pool construction/touch-fill may use Stream 0. Publish only after draining it.
            OWNER_CHECK(cudaDeviceSynchronize());
        } catch (...) {
            destroy_events();
            throw;
        }
    }

    ~Impl() {
        // Source buffers and stream remain borrowed; complete both workers before destruction.
        try {
            drain();
        } catch (...) { (void)cudaStreamSynchronize(compute); }
        destroy_events();
    }

    void destroy_events() noexcept {
        for (auto event : score_events) if (event) (void)cudaEventDestroy(event);
        for (auto& events : {&wait_start, &wait_end})
            for (auto event : *events)
                if (event) (void)cudaEventDestroy(event);
    }

    void checked(cudaError_t status, const char* operation, bool drain_failed = false) {
        if (status != cudaSuccess) {
            poisoned = true;
            unrecoverable |= kvmem::cuda_transfer_failure_is_unrecoverable(status, drain_failed);
        }
        try { kvmem::check_transfer_cuda(status, operation, drain_failed); }
        catch (...) {
            if (unrecoverable) kvmem::prefer_unrecoverable_exception(unrecoverable_cause, std::current_exception());
            if (kvmem::exception_is_typed_unrecoverable(unrecoverable_cause))
                std::rethrow_exception(unrecoverable_cause);
            throw;
        }
    }
    void throw_if_unrecoverable() const {
        auto cause = unrecoverable_cause;
        try { transfer.throw_if_unrecoverable(); }
        catch (...) { kvmem::prefer_unrecoverable_exception(cause, std::current_exception()); }
        if (sparse) {
            try { sparse->throw_if_unrecoverable(); }
            catch (...) { kvmem::prefer_unrecoverable_exception(cause, std::current_exception()); }
        }
        if (cause) std::rethrow_exception(cause);
        if (unrecoverable) throw std::runtime_error("Main fatal/failed drain cannot reset CUDA resources");
    }
    void seed_historical_failure() {
        // Read existing causes before a fresh stream/worker drain can latch a
        // different typed failure. This is pure metadata inspection; it must
        // not skip the subsequent drain attempts or authorize any recovery.
        try { throw_if_unrecoverable(); }
        catch (...) { kvmem::prefer_unrecoverable_exception(unrecoverable_cause, std::current_exception()); }
    }
    void healthy() {
        if (poisoned || (sparse && sparse->poisoned()))
            throw std::logic_error("Main KVMem owner poisoned; attention is blocked");
        if (transfer.failure_state().failed) {
            // Observe worker rejection BEFORE capturing or appending any more Main bytes.
            try { transfer.synchronize(); }
            catch (...) { fail_incomplete_append(std::current_exception()); }
            fail_incomplete_append(std::make_exception_ptr(
                std::runtime_error("Main asynchronous transfer owner failed")));
        }
    }
    void check_stream(cudaStream_t stream) const {
        if (stream != compute)
            throw std::invalid_argument("tiered owner requires its bound compute stream");
    }

    std::int32_t physical(const kvmem::KVViewWrite& write) const {
        return lease_ids.at(plan.shadow_validate ? write.logical_page
                                                 : std::uint32_t(write.physical_slot));
    }

    void complete_writebacks() {
        try {
        for (std::uint32_t layer = 0; layer < 16; ++layer) {
            if (!pending[layer].valid()) continue;
            pending[layer].get();
            // Future completion guarantees the archive synchronization cannot wait on DMA.
            archive.synchronize_layer(layer);
            for (const auto& write : writes)
                if (!table.complete_writeback(layer, write.logical_page, write.epoch))
                    throw std::logic_error("tiered writeback epoch changed before completion");
            pending[layer] = {};
        }
        } catch (...) {
            const auto failure = transfer.failure_state();
            if (plan.mode == KvMode::KVMem && failure.failed && failure.archive_bytes_uncertain)
                fail_incomplete_append(std::current_exception());
            throw;
        }
    }

    void collect_times() {
        if (!plan.measure_transfer_waits) return;
        for (std::uint32_t layer = 0; layer < 16; ++layer) {
            if (!measured[layer]) continue;
            checked(cudaEventSynchronize(wait_end[layer]), "transfer timing event drain", plan.mode == KvMode::KVMem);
            float elapsed = 0;
            OWNER_CHECK(cudaEventElapsedTime(&elapsed, wait_start[layer], wait_end[layer]));
            wait_ms[execution_phase][layer] += elapsed;
            measured[layer] = false;
        }
    }

    void drain_borrowers(const char* operation) {
        seed_historical_failure();
        drain_borrowers_status(unrecoverable_cause, cudaStreamSynchronize(compute), operation);
    }
    void drain_borrowers_status(std::exception_ptr first, cudaError_t status, const char* operation) {
        bool failed = false;
        try { checked(status, operation, true); }
        catch (...) { failed = true; kvmem::prefer_unrecoverable_exception(first, std::current_exception()); }
        if (sparse) {
            try { sparse->drain(); }
            catch (...) { failed = poisoned = true; kvmem::prefer_unrecoverable_exception(first, std::current_exception()); }
            unrecoverable |= sparse->unrecoverable();
        }
        if (failed) {
            unrecoverable = true;
            kvmem::prefer_unrecoverable_exception(unrecoverable_cause, first);
            try { throw_if_unrecoverable(); }
            catch (...) { kvmem::prefer_unrecoverable_exception(first, std::current_exception()); }
            std::rethrow_exception(first);
        }
    }

    void drain() {
        seed_historical_failure();
        const bool append_pending = std::any_of(pending.begin(), pending.end(),
            [](const auto& future) { return future.valid(); });
        auto first = unrecoverable_cause;
        bool failed = false;
        const auto attempt = [&](auto&& operation) {
            try { operation(); } catch (...) { failed = poisoned = true; kvmem::prefer_unrecoverable_exception(first, std::current_exception()); }
        };
        attempt([&] { drain_borrowers("Main compute borrower drain"); });
        attempt([&] { transfer.synchronize(); });
        attempt([&] { complete_writebacks(); });
        attempt([&] { collect_times(); });
        if (failed) {
            try { throw_if_unrecoverable(); }
            catch (...) { kvmem::prefer_unrecoverable_exception(first, std::current_exception()); }
            const auto failure = transfer.failure_state();
            if (plan.mode == KvMode::KVMem && append_pending && failure.failed && failure.archive_bytes_uncertain)
                fail_incomplete_append(first);
            std::rethrow_exception(first);
        }
    }
    kvmem::HostKVTransferFailureState drain_failed_operation(
        const std::exception_ptr& cause, const char* operation) {
        seed_historical_failure();
        if (kvmem::exception_is_typed_unrecoverable(cause))
            kvmem::prefer_unrecoverable_exception(unrecoverable_cause, cause);
        auto strongest = unrecoverable_cause;
        kvmem::prefer_unrecoverable_exception(strongest, cause);
        // Both identities are seeded BEFORE any fresh worker/compute drain.
        const auto failure = transfer.quiesce();
        prepare_after_quiesce(strongest);
        return finish_failed_operation(strongest, failure, cudaStreamSynchronize(compute), operation);
    }
    std::function<void()> score_query_drain() const {
        // A failed completion stays in the subordinate Q pool through teardown.
        // Its callback must not borrow Main fields whose lifetime ends first.
        // Live Sparse/Main drain collectors latch the standalone typed error.
        return [stream = compute] {
            kvmem::check_transfer_cuda(cudaStreamSynchronize(stream), "score Q borrower drain", true);
        };
    }
    void finish_score_query_borrow(kvmem::QueryCaptureBorrow& borrower) {
        // Sparse/transfer may already own an original failure that Main has
        // not observed. Import it before this fresh completion can latch B.
        seed_historical_failure();
        try { borrower.finish(); }
        catch (const kvmem::CudaTransferError& error) {
            // Explicit score release happens while Main is alive. Retained
            // callbacks themselves stay independent of Main during teardown.
            checked(error.status(), error.operation(), error.drain_failed());
            throw;
        }
    }
    void prepare_after_quiesce(std::exception_ptr& strongest) {
        // Quiesce itself can first establish a typed worker/event failed-drain
        // cause. Ingest it before compute/Q cleanup can report a later cause.
        seed_historical_failure();
        kvmem::prefer_unrecoverable_exception(strongest, unrecoverable_cause);
    }
    kvmem::HostKVTransferFailureState finish_failed_operation(std::exception_ptr strongest,
        const kvmem::HostKVTransferFailureState& failure, cudaError_t status, const char* operation) {
        bool failed = false;
        try { drain_borrowers_status(strongest, status, operation); }
        catch (...) { failed = true; kvmem::prefer_unrecoverable_exception(strongest, std::current_exception()); }
        try { transfer.throw_if_unrecoverable(); }
        catch (...) { failed = true; kvmem::prefer_unrecoverable_exception(strongest, std::current_exception()); }
        if (failed) {
            unrecoverable = true;
            kvmem::prefer_unrecoverable_exception(unrecoverable_cause, strongest);
            std::rethrow_exception(strongest);
        }
        throw_if_unrecoverable();
        return failure;
    }

    [[noreturn]] void fail_incomplete_append(const std::exception_ptr& cause) {
        poisoned = true;
        (void)drain_failed_operation(cause, "failed Main block-start/worker compute drain");
        // The table/write epoch may already describe provisional new columns,
        // while the archive and Mean-K remain at the accepted frontier. There
        // is no accepted complete block to restore automatically at this boundary.
        throw KVMemColdResetRequired("Main incomplete append/worker failure: whole bundle cold exact rebuild required");
    }

    PagedKVLayerView resident(std::uint32_t layer) const {
        const auto v = main.batch_layer_view(layer);
        return {v.k_pages,
                v.v_pages,
                v.k_scale_pages,
                v.v_scale_pages,
                plan.block_table.bind(backing),
                v.head_dim,
                v.num_kv_heads,
                v.dtype,
                v.quant_group,
                v.packed_v,
                v.rotate_k,
                v.rotate_v,
                v.packed_k,
                v.e8_lattice,
                v.e8_root};
    }

    PagedKVLayerView staged(std::uint32_t layer) const {
        auto v           = resident(layer);
        const auto pages = sparse_phase == SparseExecutionPhase::SparseDecode ? 0 :
            static_cast<std::int32_t>(snapshot.host_only.size());
        std::array<Tensor*, 4> planes{&v.k_pages, &v.v_pages, &v.k_scale_pages, &v.v_scale_pages};
        for (std::size_t plane = 0; plane < plan.staging_plane_regions.size(); ++plane) {
            auto& t = *planes[plane];
            if (!pages) {
                t = {};
                continue;
            }
            t = Tensor(plan.staging_plane_regions[plane].bind(backing).data, t.dtype,
                       {t.ne[0], 64, t.ne[2], pages});
        }
        return v;
    }

    void restore_prefix(const kvmem::KVViewWrite& write) {
        if (!write.restore_from_host) return;
        // The extra tile does not alias the full streamed layer. Transfers own every host copy.
        const auto offset = plan.staging.maximum_layer_stream_bytes;
        for (std::uint32_t layer = 0; layer < 16; ++layer) {
            for (std::size_t plane = 0; plane < plan.archive.layers[layer].size(); ++plane) {
                const auto& spec = plan.archive.layers[layer][plane];
                auto ticket =
                    transfer.prefetch_completed(layer, plane, write.logical_page, 1, offset);
                transfer.wait(ticket, compute);
                const auto& tensor = main.pool().plane(spec.pool_plane);
                auto* destination =
                    static_cast<std::byte*>(tensor.data) + physical(write) * tensor.nb[3];
                OWNER_CHECK(cudaMemcpyAsync(destination, transfer.staged(ticket).data,
                                           spec.page_bytes, cudaMemcpyDeviceToDevice, compute));
                transfer.release(ticket, compute);
                checked(cudaStreamSynchronize(compute), "restore prefix consumer drain", true);
                transfer.synchronize();
            }
        }
    }

    void prefetch(std::uint32_t layer) {
        if (!tickets.empty()) throw std::logic_error("tiered prefetch has unreleased consumers");
        if (sparse_phase == SparseExecutionPhase::SparseDecode) return;
        for (std::size_t plane = 0; plane < plan.archive.layers[layer].size(); ++plane) {
            const auto page_bytes = plan.archive.layers[layer][plane].page_bytes;
            const auto tile_pages = kvmem::kHostKVTransferTileBytes / page_bytes;
            if (!tile_pages)
                throw std::logic_error("tiered archive page exceeds one transfer tile");
            for (std::size_t first = 0; first < snapshot.host_only.size();) {
                const auto count = std::min<std::size_t>(tile_pages, snapshot.host_only.size() - first);
                const auto offset = plan.staging_plane_regions[plane].offset -
                                    plan.staging_region.offset + first * page_bytes;
                tickets.push_back(
                    transfer.prefetch_gather_completed(layer, plane,
                        std::span<const std::uint32_t>(snapshot.host_only).subspan(first, count), offset));
                streamed_bytes += count * page_bytes;
                first += count;
            }
        }
    }

    void invalidate_captures(std::uint32_t frontier) {
        std::erase_if(captures, [frontier](const auto& weak) {
            const auto capture = weak.lock();
            if (!capture) return true;
            if (capture->frontier > frontier) capture->valid = false;
            return !capture->valid;
        });
    }

    void publish() {
        std::fill(block_table.begin(), block_table.end(), -1);
        access.clear();
        std::size_t host = 0;
        for (std::uint32_t logical = 0; logical < snapshot.blocktable.size(); ++logical) {
            const auto slot = snapshot.blocktable[logical];
            const auto actual =
                slot < 0 ? -1 : lease_ids.at(plan.shadow_validate ? logical : std::uint32_t(slot));
            block_table[logical] = actual;
            if (actual < 0 && sparse_phase == SparseExecutionPhase::SparseDecode) continue;
            const auto mixed =
                actual < 0 ? main.pool().page_group_count() + host++ : std::uint32_t(actual);
            if (mixed > std::uint32_t(std::numeric_limits<std::int32_t>::max()))
                throw std::overflow_error("tiered physical access ID exceeds int32");
            access.push_back(
                {static_cast<std::int32_t>(logical), static_cast<std::int32_t>(mixed)});
        }
        prefix =
            ops::attention_access_prefix(access, current_frontier, main.pool().page_group_count(),
                                         sparse_phase == SparseExecutionPhase::SparseDecode ? 0U :
                                             static_cast<std::uint32_t>(snapshot.host_only.size()));
        OWNER_CHECK(cudaMemcpyAsync(plan.block_table.bind(backing).data, block_table.data(),
                                   block_table.size() * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, compute));
        OWNER_CHECK(cudaMemcpyAsync(plan.access[0].bind(backing).data, access.data(),
                                   access.size() * sizeof(ops::AttentionPageAccess),
                                   cudaMemcpyHostToDevice, compute));
        OWNER_CHECK(cudaMemcpyAsync(plan.prefix[0].bind(backing).data, prefix.data(),
                                   prefix.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   compute));
    }

    std::size_t hydrate(const kvmem::KVViewSnapshot& restored) {
        if (restored.hydration_pages.empty()) return 0;
        std::size_t scheduled_bytes = 0;
        const auto& missing = restored.hydration_pages;
        const auto physical_id_for = [&](std::uint32_t logical) {
            return lease_ids.at(plan.shadow_validate ? logical :
                std::uint32_t(restored.blocktable.at(logical)));
        };
        const auto offset = plan.staging.maximum_layer_stream_bytes;
        for (std::uint32_t layer = 0; layer < 16; ++layer)
            for (std::size_t plane = 0; plane < plan.archive.layers[layer].size(); ++plane) {
                const auto& spec = plan.archive.layers[layer][plane];
                const auto tile_pages = kvmem::kHostKVTransferTileBytes / spec.page_bytes;
                const auto& tensor = main.pool().plane(spec.pool_plane);
                for (std::size_t first = 0; first < missing.size();) {
                    const auto count = std::min<std::size_t>(tile_pages, missing.size() - first);
                    auto ticket = transfer.prefetch_gather_completed(layer, plane,
                        std::span<const std::uint32_t>(missing).subspan(first, count), offset);
                    scheduled_bytes += count * spec.page_bytes;
                    transfer.wait(ticket, compute);
                    const auto staged = transfer.staged(ticket);
                    // Logical host ranges are coalesced. Lease IDs may be noncontiguous.
                    for (std::size_t j = 0; j < count;) {
                        const auto physical_id = physical_id_for(missing[first + j]);
                        std::size_t run = 1;
                        if (tensor.nb[3] == spec.page_bytes)
                            while (j + run < count) {
                                if (physical_id_for(missing[first + j + run]) != physical_id + run) break;
                                ++run;
                            }
                        auto* target = static_cast<std::byte*>(tensor.data) + physical_id * tensor.nb[3];
                        OWNER_CHECK(cudaMemcpyAsync(target, static_cast<const std::byte*>(staged.data) +
                            j * spec.page_bytes, run * spec.page_bytes, cudaMemcpyDeviceToDevice, compute));
                        j += run;
                    }
                    transfer.release(ticket, compute);
                    // Backpressure is explicit: no reused tile can accumulate ticket references.
                    checked(cudaStreamSynchronize(compute), "hydrate tile consumer drain", true);
                    transfer.synchronize();
                    first += count;
                }
            }
        checked(cudaStreamSynchronize(compute), "hydrate final consumer drain", plan.mode == KvMode::KVMem);
        if (sparse) sparse->drain();
        transfer.synchronize();
        return scheduled_bytes;
    }

    std::vector<std::uint32_t> logical_resident() const {
        std::vector<std::uint32_t> result;
        for (const auto& p : table.snapshot().resident) result.push_back(p.logical_page);
        return result;
    }
    void finish_exact_block() {
        healthy();
        if (!sparse || !transaction_kind) return;
        if (next_layer != 16 || *transaction_kind != kvmem::MeanKTransactionKind::ExactPrefill)
            throw std::logic_error("finish exact requires all Main layers");
        const auto started = std::chrono::steady_clock::now();
        try { sparse->finish_exact_chunk(); }
        catch (...) { poisoned = true; unrecoverable |= sparse->unrecoverable(); throw; }
        capture_wait_ms += wall_ms(started);
        transaction_kind.reset();
    }
    void recover_prior_view(const std::vector<std::uint32_t>& prior,
                            const std::vector<std::uint32_t>& prior_hard,
                            SparseExecutionPhase phase, const std::exception_ptr& cause) {
        poisoned = true; // no stale physical mapping can be consumed once DMA has started
        const auto failure = drain_failed_operation(cause, "poison compute drain");
        try { std::rethrow_exception(cause); }
        catch (const kvmem::CudaTransferError& error) {
            if (kvmem::cuda_transfer_failure_is_unrecoverable(error.status(), error.drain_failed())) {
                unrecoverable = true;
                throw;
            }
        } catch (...) {
            // Only a classified CUDA API rejection establishes recoverable DMA.
            // Generic source failures keep the owner poisoned until an explicit
            // whole-bundle cold reset; never republish uncertain authority.
            throw;
        }
        if (failure.unrecoverable) transfer.throw_if_unrecoverable();
        if (sparse) {
            unrecoverable |= sparse->unrecoverable();
            sparse->throw_if_unrecoverable();
        }
        if (sparse && sparse->poisoned())
            throw KVMemColdResetRequired("Main capture resource rejected CUDA work: cold exact rebuild required");
        if (failure.archive_bytes_uncertain)
            throw KVMemColdResetRequired("Main archive uncertain: whole bundle cold reset/exact rebuild required");
        (void)transfer.recover();
        table.set_generation_protection({}, 0);
        table.invalidate_device_ownership();
        auto recovered = table.plan_selection(current_frontier, prior);
        hydrate(recovered);
        table.install_selection(recovered);
        exact_hard = prior_hard;
        table.set_generation_protection(exact_hard,
            phase == SparseExecutionPhase::SparseDecode ? plan.recent_tokens : 0);
        snapshot = table.snapshot(); sparse_phase = phase;
        publish();
        checked(cudaStreamSynchronize(compute), "recovered Main publication drain", true);
        poisoned = false;
    }
    std::uint64_t install_selected(const kvmem::KVViewSnapshot& selected,
                                   const std::vector<std::uint32_t>& hard,
                                   SparseExecutionPhase phase, double* hydrate_ms = nullptr,
                                   double* publish_ms = nullptr) {
        const auto prior = logical_resident();
        const auto prior_hard = exact_hard;
        const auto prior_phase = sparse_phase;
        std::uint64_t bytes = 0;
        try {
            const auto hydrate_started = std::chrono::steady_clock::now();
            bytes = hydrate(selected);
            if (hydrate_ms) *hydrate_ms = wall_ms(hydrate_started);
            const auto publish_started = std::chrono::steady_clock::now();
            table.install_selection(selected);
            exact_hard = hard;
            sparse_phase = phase;
            table.set_generation_protection(exact_hard,
                phase == SparseExecutionPhase::SparseDecode ? plan.recent_tokens : 0);
            snapshot = table.snapshot(); publish();
            checked(cudaStreamSynchronize(compute), "selected Main publication drain", true);
            if (publish_ms) *publish_ms = wall_ms(publish_started);
        } catch (...) {
            const auto cause = std::current_exception();
            recover_prior_view(prior, prior_hard, prior_phase, cause);
            std::rethrow_exception(cause);
        }
        return bytes;
    }
    std::uint64_t prepare_exact_prefill() {
        healthy();
        if (!sparse || sparse_phase == SparseExecutionPhase::ExactPrefill) return 0;
        if (next_layer != 16 || transaction_kind)
            throw std::logic_error("exact transition requires drained completed transaction");
        drain();
        // Preserve the CURRENT arbitrary owners. Fill free slots with missing newer pages.
        auto ids = logical_resident();
        const auto target = std::min(plan.view_pages, page_count(current_frontier));
        for (auto page = page_count(current_frontier); ids.size() < target && page;) {
            --page;
            if (!std::binary_search(ids.begin(), ids.end(), page)) {
                ids.insert(std::lower_bound(ids.begin(), ids.end(), page), page);
            }
        }
        table.set_generation_protection({}, 0);
        kvmem::KVViewSnapshot selected;
        try { selected = table.plan_selection(current_frontier, ids); }
        catch (...) {
            table.set_generation_protection(exact_hard, plan.recent_tokens);
            throw;
        }
        const auto bytes = install_selected(selected, {}, SparseExecutionPhase::ExactPrefill);
        std::clog << "[kvmem-prefill-refill] frontier=" << current_frontier << " added_pages="
                  << selected.hydration_pages.size() << " hydrate_bytes=" << bytes << '\n';
        return bytes;
    }

    void begin(std::uint32_t base, std::uint32_t count, cudaStream_t stream, ExecutionPhase phase) {
        healthy();
        check_stream(stream);
        if (next_layer != 16) throw std::logic_error("tiered previous block has incomplete layers");
        if (!count || count > plan.max_query_tokens || base != current_frontier ||
            std::uint64_t(base) + count > plan.logical_tokens)
            throw std::invalid_argument(
                "tiered block range does not match its frontier or capacity");
        const auto needed = plan.shadow_validate ? page_count(base + count) : plan.view_pages;
        if (lease_ids.size() < needed)
            throw std::logic_error("tiered block has insufficient materialized lease IDs");
        // This is the physical-overwrite boundary: old source bytes must no longer belong to D2H.
        complete_writebacks();
        collect_times();
        if (plan.measure_transfer_waits) execution_phase=phase==ExecutionPhase::Prefill ? 0 : 1;
        if (sparse && plan.mode == KvMode::KVMem && phase == ExecutionPhase::Prefill) prepare_exact_prefill();
        try { table.preflight_append(base, count); }
        catch (const std::length_error&) {
            if (sparse && transaction_kind) {
                sparse->discard_transaction(); transaction_kind.reset();
            }
            std::clog << "[kvmem-capacity] frontier=" << base << " hard_pages=" << exact_hard.size()
                      << " resident_pages=" << table.snapshot().resident.size()
                      << " recent_tokens=" << plan.recent_tokens << " reserve_pages=" << plan.reserve_pages
                      << " guard_pages=" << plan.guard_pages << " status=no_safe_victim\n";
            throw;
        }
        if (sparse && plan.mode == KvMode::KVMem) {
            if (phase == ExecutionPhase::Prefill) {
                if (transaction_kind) throw std::logic_error("Main exact chunk overlaps transaction");
                sparse->prepare_transaction(kvmem::MeanKTransactionKind::ExactPrefill, base, count);
                transaction_kind = kvmem::MeanKTransactionKind::ExactPrefill;
                transaction_first = base; transaction_count = count;
            } else if (!transaction_kind || transaction_first != base || transaction_count != count)
                throw std::logic_error("Main decode must prepare its actual valid transaction first");
        }
        try { writes = table.begin_append(base, count); }
        catch (const std::length_error&) {
            if (sparse && transaction_kind) {
                if (*transaction_kind != kvmem::MeanKTransactionKind::ExactPrefill) sparse->discard_transaction();
                else poisoned = true;
                transaction_kind.reset();
            }
            std::clog << "[kvmem-capacity] frontier=" << base << " hard_pages=" << exact_hard.size()
                      << " resident_pages=" << table.snapshot().resident.size()
                      << " recent_tokens=" << plan.recent_tokens << " reserve_pages=" << plan.reserve_pages
                      << " guard_pages=" << plan.guard_pages << " status=no_safe_victim\n";
            throw;
        }
        catch (...) { fail_incomplete_append(std::current_exception()); }
        try {
            // Writeback archives entire page planes, including an unused suffix.
            // Initialize newly assigned logical pages on the producer stream;
            // retained tail pages keep their prefix (or restore it from archive).
            const auto previous_pages = page_count(base);
            for (const auto& write : writes) {
                if (write.logical_page < previous_pages) continue;
                for (const auto& layer : plan.archive.layers)
                    for (const auto& spec : layer) {
                        const auto& tensor = main.pool().plane(spec.pool_plane);
                        auto* page =
                            static_cast<std::byte*>(tensor.data) + physical(write) * tensor.nb[3];
                        OWNER_CHECK(cudaMemsetAsync(page, 0, spec.page_bytes, compute));
                    }
            }
            for (const auto& write : writes) restore_prefix(write);
            snapshot         = table.snapshot();
            current_frontier = base + count;
            block_tokens     = count;
            write_ids.clear();
            for (const auto& write : writes) write_ids.push_back(physical(write));
            publish();
            next_layer = 0;
            ++blocks;
            // Queued before the model's initial GDN layers to overlap their compute.
            prefetch(0);
        } catch (...) { fail_incomplete_append(std::current_exception()); }
    }

    void writeback(std::uint32_t layer) {
        pending[layer] = transfer.writeback(main.pool(), layer, writes.front().logical_page,
                                            write_ids, current_frontier, compute);
    }

    void partial(std::uint32_t layer, const Tensor& q, const Tensor& positions, float scale,
                 Tensor& out) {
        if (layer != next_layer || layer >= 16)
            throw std::logic_error("tiered FA layers must execute in order");
        if (q.ne[2] < static_cast<std::int32_t>(block_tokens) ||
            positions.ne[0] < static_cast<std::int32_t>(block_tokens))
            throw std::invalid_argument("tiered attention inputs are shorter than the valid block");
        if (plan.measure_transfer_waits) OWNER_CHECK(cudaEventRecord(wait_start[layer], compute));
        const auto start = plan.measure_transfer_waits ? std::chrono::steady_clock::now()
                                                       : std::chrono::steady_clock::time_point{};
        for (auto ticket : tickets) transfer.wait(ticket, compute);
        if (plan.measure_transfer_waits) {
            enqueue_wait_ms[execution_phase][layer] +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
            OWNER_CHECK(cudaEventRecord(wait_end[layer], compute));
            measured[layer] = true;
        }
        const auto tokens        = static_cast<std::int32_t>(block_tokens);
        const auto resident_view = resident(layer), staging_view = staged(layer);
        const auto preferred = tokens > 4 ? 1 : (resident_view.packed_k ? 64 : 32);
        // Bound the launch by both actual key count and the independent fixed scratch planes.
        const auto visible_splits = std::max(1, static_cast<int>((std::uint64_t(prefix.back()) + 63) / 64));
        const auto scratch_capacity =
            plan.scratch_m.region.bytes / (24ULL * tokens * sizeof(float));
        const auto splits = static_cast<std::int32_t>(
            std::min<std::size_t>(std::min(preferred, visible_splits), scratch_capacity));
        ops::AttentionPartial scratch{
            Tensor(plan.scratch_o.bind(backing).data, DType::FP32, {256, 24, tokens, splits}),
            Tensor(plan.scratch_m.bind(backing).data, DType::FP32, {24, tokens, splits}),
            Tensor(plan.scratch_l.bind(backing).data, DType::FP32, {24, tokens, splits})};
        ops::AttentionPartial state{
            Tensor(plan.state_o.bind(backing).data, DType::FP32, {256, 24, tokens, 1}),
            Tensor(plan.state_m.bind(backing).data, DType::FP32, {24, tokens, 1}),
            Tensor(plan.state_l.bind(backing).data, DType::FP32, {24, tokens, 1})};
        Tensor pages(plan.access[0].bind(backing).data, DType::I32,
                     {2, static_cast<int>(access.size())});
        Tensor sums(plan.prefix[0].bind(backing).data, DType::I32,
                    {static_cast<int>(prefix.size())});
        const auto query = q.slice(2, 0, tokens), positions_view = positions.slice(0, 0, tokens);
        auto output = out.slice(2, 0, tokens);
        if (tokens > 4)
            ops::gqa_attention_partial_prefill(query, positions_view, scale, resident_view,
                                               staging_view, pages, sums, current_frontier, splits,
                                               scratch, compute);
        else
            ops::gqa_attention_partial_decode(query, positions_view, scale, resident_view,
                                              staging_view, pages, sums, current_frontier, splits,
                                              scratch, compute);
        if (resident_view.rotate_v) {
            // The FP32 state remains in original V coordinates. Match dense's
            // BF16 output boundary in the rotated basis before the final inverse.
            ops::attention_partial_lse_accumulate(scratch, state, true, compute);
            ops::attention_partial_finalize_rotated(state, output, compute);
        } else {
            ops::attention_partial_lse_accumulate(scratch, state, true, compute, &output);
        }

        ++partial_passes;
        for (auto ticket : tickets) transfer.release(ticket, compute);
        tickets.clear();
        if (++next_layer < 16) prefetch(next_layer);
        else if (sparse && transaction_kind == kvmem::MeanKTransactionKind::ExactPrefill)
            finish_exact_block();
    }
};

TieredContext::TieredContext(const TieredRuntimePlan& plan, PagedKVCache& main, DeviceSpan backing,
                             cudaStream_t stream, kvmem::HostKVTransferFaultInjection test_fault)
    : impl_(std::make_unique<Impl>(plan, main, backing, stream, test_fault)) {}

TieredContext::~TieredContext() = default;

bool TieredContext::shadow() const noexcept { return impl_->plan.shadow_validate; }

std::uint32_t TieredContext::view_pages() const noexcept { return impl_->plan.view_pages; }

std::uint32_t TieredContext::frontier() const noexcept { return impl_->current_frontier; }

void TieredContext::bind_pages(std::span<const std::int32_t> ids) {
    if (impl_->next_layer != 16)
        throw std::logic_error("tiered lease cannot change during a block");
    const auto required = shadow() ? page_count(impl_->current_frontier) : view_pages();
    if (ids.size() < required)
        throw std::invalid_argument("tiered lease does not cover its resident view");
    for (auto id : ids)
        if (id < 0 || std::uint32_t(id) >= impl_->main.pool().page_group_count())
            throw std::invalid_argument("tiered lease ID outside Main physical pool");
    auto sorted = std::vector<std::int32_t>(ids.begin(), ids.end());
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
        throw std::invalid_argument("tiered lease contains duplicate physical pages");
    // Resident conceptual slots cannot move while their archived/device identity is live.
    const auto retained = shadow() ? page_count(impl_->current_frontier)
                                   : (impl_->current_frontier ? view_pages() : 0U);
    for (std::uint32_t slot = 0; slot < retained; ++slot)
        if (impl_->lease_ids.at(slot) != ids[slot])
            throw std::invalid_argument("tiered live lease IDs changed before reset");
    impl_->lease_ids.assign(ids.begin(), ids.end());
}

void TieredContext::attach_sparse_capture(std::unique_ptr<SparseCaptureOwner> owner) {
    auto& i = *impl_;
    if (!owner || i.sparse || i.current_frontier || i.next_layer != 16 ||
        owner->resources().mean.max_context != i.plan.logical_tokens ||
        owner->resources().mean.max_chunk < i.plan.max_query_tokens || owner->index().frontier())
        throw std::invalid_argument("Main sparse capture must attach at loading before first block");
    i.sparse = std::move(owner);
}
SparseCaptureOwner* TieredContext::sparse_capture() noexcept { return impl_->sparse.get(); }
std::uint64_t TieredContext::bundle_identity() const noexcept { return impl_->bundle_identity; }
void TieredContext::capture_pre_rope(std::uint32_t layer, const Tensor& qn, const Tensor& kn,
                                    cudaStream_t stream) {
    auto& i = *impl_;
    if (i.sparse) {
        i.healthy();
        try { i.sparse->capture_pre_rope(layer, qn, kn, stream); }
        catch (...) { i.poisoned = true; i.unrecoverable |= i.sparse->unrecoverable(); throw; }
    }
}

SparseLoadingInfo TieredContext::loading_info() const {
    const auto& i = *impl_;
    const bool pageable = i.archive.mode() == kvmem::HostArchiveMode::Pageable;
    const auto pinned = (pageable ? i.transfer.pinned_ring_bytes() : i.plan.archive.bytes) +
        (i.plan.sparse_capture ? i.plan.sparse_capture->pinned_bytes + i.plan.host_score_bytes +
            (pageable ? 0 : i.plan.sparse_capture->mean.index_bytes) : 0);
    return {i.archive.mode(), sparse_host_payload_bytes(i.plan, pageable), pinned,
            i.plan.bytes, i.plan.owned_device_bytes, kvmem::available_physical_memory_bytes()};
}
SparseExecutionPhase TieredContext::sparse_phase() const noexcept { return impl_->sparse_phase; }
bool TieredContext::poisoned() const noexcept {
    return impl_->poisoned || (impl_->sparse && impl_->sparse->poisoned()) ||
           impl_->transfer.failure_state().failed;
}
void TieredContext::set_transfer_test_fault(kvmem::HostKVTransferFaultInjection fault) {
    impl_->healthy(); impl_->drain(); impl_->transfer.set_test_fault(fault);
    impl_->test_fault = fault;
}
void TieredContext::begin_query(kvmem::QueryProvenance p, std::uint64_t caller_bundle) {
    auto& i = *impl_;
    i.healthy();
    if (!i.sparse || !caller_bundle || p.bundle_identity != caller_bundle ||
        (i.caller_bundle && i.caller_bundle != caller_bundle))
        throw std::invalid_argument("Main query caller bundle binding");
    i.sparse->begin_query(std::move(p));
    i.caller_bundle = caller_bundle;
}
void TieredContext::use_query(const kvmem::QueryCaptureHandle& q, const kvmem::QueryProvenance& p,
                             std::uint64_t caller_bundle) {
    auto& i = *impl_;
    i.healthy();
    if (!i.sparse || !caller_bundle || p.bundle_identity != caller_bundle ||
        (i.caller_bundle && i.caller_bundle != caller_bundle))
        throw std::invalid_argument("Main restored query caller bundle binding");
    i.sparse->use_query(q, p, i.current_frontier);
    i.caller_bundle = caller_bundle;
}
void TieredContext::prepare_main_transaction(kvmem::MeanKTransactionKind kind,
                                            std::uint32_t count) {
    auto& i = *impl_;
    i.healthy();
    if (!i.sparse || i.next_layer != 16 || i.transaction_kind || !count ||
        count > std::min(16U, i.plan.max_query_tokens) ||
        (kind != kvmem::MeanKTransactionKind::OrdinaryMain &&
         kind != kvmem::MeanKTransactionKind::SpeculativeMain) ||
        (kind == kvmem::MeanKTransactionKind::OrdinaryMain && count != 1))
        throw std::invalid_argument("Main decode transaction actual extent/kind");
    i.sparse->prepare_transaction(kind, i.current_frontier, count);
    i.transaction_kind = kind;
    i.transaction_first = i.current_frontier; i.transaction_count = count;
}
void TieredContext::finish_exact_block() { impl_->finish_exact_block(); }
void TieredContext::flush_accepted_frontier(std::uint32_t accepted) {
    auto& i = *impl_;
    i.healthy();
    if (!i.sparse || !i.transaction_kind || i.next_layer != 16 || accepted > i.transaction_count ||
        *i.transaction_kind == kvmem::MeanKTransactionKind::ExactPrefill)
        throw std::invalid_argument("Main acceptance requires completed actual transaction");
    const auto first = i.transaction_first;
    try { i.sparse->flush_accepted_frontier(accepted); }
    catch (...) { i.poisoned = true; i.unrecoverable |= i.sparse->unrecoverable(); throw; }
    i.transaction_kind.reset();
    if (i.current_frontier != first + accepted) trim(first + accepted, i.compute);
    else i.drain(); // full acceptance still flushes and establishes checkpoint authority
    if (i.sparse->index().frontier() != i.current_frontier)
        throw std::logic_error("Main accepted KV/index frontiers disagree");
}
std::uint64_t TieredContext::prepare_exact_prefill() { return impl_->prepare_exact_prefill(); }
SparseTurnMetrics TieredContext::select_and_publish(const SparseTurnInput& input) {
    auto& i = *impl_;
    i.healthy();
    if (!i.sparse || i.plan.mode != KvMode::KVMem || i.next_layer != 16 || i.transaction_kind ||
        i.sparse_phase != SparseExecutionPhase::ExactPrefill || !input.bundle_identity ||
        input.bundle_identity != input.query.bundle_identity || input.bundle_identity != i.caller_bundle ||
        !std::isfinite(input.prefill_ms) || input.prefill_ms < 0)
        throw std::invalid_argument("Main sparse selection phase/query/prefill timing binding");
    if (i.test_fault.selection_after_exact) {
        if (!i.test_fault.d2h_paused || !i.test_fault.resume_d2h ||
            i.sparse->index().frontier() != i.current_frontier || i.sparse->transaction_pending())
            throw std::logic_error("late writeback test requires completed exact capture");
        while (!i.test_fault.d2h_paused->load()) std::this_thread::yield();
        i.test_fault.selection_after_exact->store(true);
        i.test_fault.resume_d2h->store(true);
    }
    i.drain();
    const auto binding = i.sparse->bind_query(input.query, i.current_frontier);
    kvmem::SelectionInput selection;
    selection.frontier = i.current_frontier; selection.physical_pages = i.plan.view_pages;
    selection.future_reserve = i.plan.reserve_pages; selection.provisional_guards = i.plan.guard_pages;
    selection.sink_pages = i.plan.sink_pages; selection.recent_tokens = i.plan.recent_tokens;
    selection.binding_generation = selection.score_generation = binding.index_generation;
    for (auto ordinal : input.query.ordinals) selection.query_spans.push_back({ordinal, ordinal + 1});
    selection.current_input_spans = input.current_input_spans; selection.image_groups = input.image_groups;
    selection.current_resident = i.logical_resident();
    const auto policy = kvmem::preflight_selection(selection);
    const auto* all = std::getenv("NINFER_KVMEM_SCORE_ALL_PAGES");
    const bool all_pages = all && std::string_view(all) == "1";
    const auto domain = kvmem::make_scoring_domain(i.current_frontier, policy.capacity,
        i.plan.sink_pages, i.plan.recent_tokens, all_pages);
    SparseTurnMetrics result;
    result.binding = binding;
    result.frontier = i.current_frontier; result.query_tokens = std::uint32_t(input.query.ordinals.size());
    result.reserve_pages = i.plan.reserve_pages; result.guard_pages = i.plan.guard_pages;
    result.archive_mode = i.archive.mode(); result.prefill_ms = input.prefill_ms;
    result.capture_wait_ms = i.capture_wait_ms;
    result.domain_first_page = domain.first; result.domain_end_page = domain.end;
    result.denominator_pages = domain.end - domain.first;
    result.all_pages_denominator = all_pages;
    const auto& r = i.plan.scoring;
    const auto stage = i.plan.staging_region.bind(i.backing);
    const auto logits = i.plan.scratch_o.bind(i.backing);
    const auto aux = i.plan.state_o.bind(i.backing);
    auto* aux_bytes = static_cast<std::byte*>(aux.data);
    const auto rows = 24 * result.query_tokens;
    ops::ScoreWorkspace workspace{
        Tensor(logits.data, DType::FP32, {int(domain.pages), int(rows)}),
        Tensor(aux_bytes + r.stats_offset, DType::FP32, {2, int(rows)}),
        Tensor(aux_bytes + r.row_status_offset, DType::I32, {int(rows)}),
        Tensor(aux_bytes + r.scores_offset, DType::FP32, {int(domain.pages)}),
        Tensor(aux_bytes + r.status_offset, DType::I32, {1})};
    try {
    auto started = std::chrono::steady_clock::now();
    const auto query = i.sparse->upload_query();
    i.checked(cudaStreamSynchronize(i.compute), "score immutable Q upload completion", true);
    result.scoring_h2d_ms += wall_ms(started); // wall interval includes host bounce/worker wait
    result.scoring_h2d_bytes += query.bytes();
    auto borrower = i.sparse->borrow_query(i.score_query_drain());
    for (std::uint32_t layer = 0; layer < 16; ++layer) {
        const auto source = std::as_bytes(i.sparse->index().layer_data(layer));
        started = std::chrono::steady_clock::now();
        const auto ticket = i.transfer.prefetch_index(source, r.index_offset);
        i.transfer.wait(ticket, i.compute);
        i.checked(cudaStreamSynchronize(i.compute), "score Mean-K layer upload completion", true);
        result.scoring_h2d_ms += wall_ms(started);
        result.scoring_h2d_bytes += source.size();
        i.OWNER_CHECK(cudaEventRecord(i.score_events[0], i.compute));
        auto q = query.slice(3, int(layer), 1);
        Tensor means(static_cast<std::byte*>(stage.data) + r.index_offset, DType::FP16,
                     {256, 4, int(domain.pages)});
        ops::kvmem_score_layer(q, means, domain.first, domain.end, 16, layer == 0, workspace, i.compute);
        i.OWNER_CHECK(cudaEventRecord(i.score_events[1], i.compute));
        i.transfer.release(ticket, i.compute);
        i.checked(cudaEventSynchronize(i.score_events[1]), "score layer completion", true);
        float elapsed = 0;
        i.OWNER_CHECK(cudaEventElapsedTime(&elapsed, i.score_events[0], i.score_events[1]));
        result.scoring_compute_ms += elapsed;
    }
    started = std::chrono::steady_clock::now();
    auto* score = static_cast<float*>(i.loading.score->data());
    auto* status = reinterpret_cast<std::int32_t*>(score + i.plan.archive.logical_pages);
    i.OWNER_CHECK(cudaMemcpyAsync(score, workspace.scores.data, domain.pages * sizeof(float),
                                  cudaMemcpyDeviceToHost, i.compute));
    i.OWNER_CHECK(cudaMemcpyAsync(status, workspace.status.data, sizeof(*status),
                                  cudaMemcpyDeviceToHost, i.compute));
    i.checked(cudaStreamSynchronize(i.compute), "score/status D2H completion", true);
    result.score_d2h_ms = wall_ms(started);
    const bool valid = !*status && i.sparse->binding_valid(binding, input.query, i.current_frontier) &&
        std::none_of(score, score + domain.pages, [](float value) { return !std::isfinite(value); });
    i.finish_score_query_borrow(borrower); // fallible live release after binding/status validation
    i.transfer.synchronize();
    if (!valid)
        throw std::runtime_error("Main score capture nonfinite or stale after D2H; old view preserved");
    result.page_scores.assign(score, score + domain.pages);
    } catch (...) {
        const auto cause = std::current_exception();
        if (i.poisoned || i.transfer.failure_state().failed)
            i.recover_prior_view(selection.current_resident, i.exact_hard, i.sparse_phase, cause);
        std::rethrow_exception(cause);
    }
    selection.scores = result.page_scores;
    auto started = std::chrono::steady_clock::now();
    result.selection = kvmem::select_pages(selection);
    const auto selected = i.table.plan_selection(i.current_frontier, result.selection.selected);
    result.selection_ms = wall_ms(started);
    result.hydrate_bytes = i.install_selected(selected, result.selection.hard_logical_ids,
        SparseExecutionPhase::SparseDecode, &result.hydrate_ms, &result.publish_ms);
    result.actual_view_tokens = i.plan.view_pages * 64U;
    result.selected_valid_tokens = std::uint32_t(i.prefix.back());
    i.capture_wait_ms = 0;
    return result;
}

void TieredContext::begin_block(std::uint32_t base, std::uint32_t count, cudaStream_t stream, ExecutionPhase phase) {
    impl_->begin(base, count, stream, phase);
}

PagedKVLayerView TieredContext::resident_layer(std::uint32_t layer) const {
    impl_->healthy();
    return impl_->resident(layer);
}

void TieredContext::attention(std::uint32_t layer, const Tensor& q, const Tensor& k,
                              const Tensor& v, const Tensor& positions, float scale, Tensor& out,
                              cudaStream_t stream) {
    auto& i = *impl_;
    i.healthy();
    i.check_stream(stream);
    if (shadow() || layer != i.next_layer || layer >= 16)
        throw std::logic_error("tiered append attention route or layer is invalid");
    if (out.ne[2] < static_cast<std::int32_t>(i.block_tokens))
        throw std::invalid_argument("tiered output is shorter than its block");
    const auto count = static_cast<std::int32_t>(i.block_tokens);
    if (out.ne[2] > count) {
        auto tail = out.slice(2, count, out.ne[2] - count);
        i.OWNER_CHECK(cudaMemsetAsync(tail.data, 0, tail.bytes(), stream));
    }
    try {
        ops::gqa_kv_append(k.slice(2, 0, count), v.slice(2, 0, count), positions.slice(0, 0, count),
                           i.resident(layer), stream);
        i.writeback(layer);
        i.partial(layer, q, positions, scale, out);
    } catch (...) {
        i.poisoned = true;
        // DMA/writeback may have altered Main or archive. Drain ALL borrowers even
        // when the first failure is already known; no uncertain authority is published.
        const auto failure = i.drain_failed_operation(std::current_exception(), "failed Main attention compute drain");
        if (failure.unrecoverable || i.unrecoverable) { i.unrecoverable = true; throw; }
        if (i.sparse && i.sparse->poisoned())
            throw KVMemColdResetRequired("Main capture resource rejected CUDA work: cold exact rebuild required");
        if (failure.archive_bytes_uncertain)
            throw KVMemColdResetRequired("Main D2H authority uncertain: cold exact rebuild required");
        throw;
    }
}

void TieredContext::shadow_attention(std::uint32_t layer, const Tensor& q, const Tensor& positions,
                                     float scale, const Tensor& dense_out, cudaStream_t stream) {
    auto& i = *impl_;
    i.check_stream(stream);
    if (!shadow() || layer != i.next_layer || layer >= 16)
        throw std::logic_error("tiered shadow route or layer is invalid");
    i.writeback(layer);
    auto out =
        i.plan.shadow_output.bind(i.backing).slice(2, 0, static_cast<std::int32_t>(i.block_tokens));
    i.partial(layer, q, positions, scale, out);
    const auto dense = dense_out.slice(2, 0, static_cast<std::int32_t>(i.block_tokens));

    i.OWNER_CHECK(cudaMemcpyAsync(i.shadow_dense->data(), dense.data, dense.bytes(),
                               cudaMemcpyDeviceToHost, stream));
    i.OWNER_CHECK(cudaMemcpyAsync(i.shadow_tiered->data(), out.data, out.bytes(),
                               cudaMemcpyDeviceToHost, stream));
    i.OWNER_CHECK(cudaStreamSynchronize(stream));
    auto* a           = static_cast<const std::uint16_t*>(i.shadow_dense->data());
    auto* b           = static_cast<const std::uint16_t*>(i.shadow_tiered->data());
    double difference = 0, norm = 0, max_abs = 0;
    for (std::int64_t n = 0; n < out.numel(); ++n) {
        const double x = bf16_float(a[n]), y = bf16_float(b[n]);
        if (!std::isfinite(x) || !std::isfinite(y))
            throw std::runtime_error("tiered shadow output is nonfinite");
        difference += (x - y) * (x - y);
        norm += x * x;
        max_abs = std::max(max_abs, std::abs(x - y));
    }
    const double relative    = std::sqrt(difference / std::max(norm, 1e-24));
    i.shadow_max_relative    = std::max(i.shadow_max_relative, relative);
    i.shadow_max_abs         = std::max(i.shadow_max_abs, max_abs);
    i.shadow_relative[layer] = std::max(i.shadow_relative[layer], relative);
    i.shadow_absolute[layer] = std::max(i.shadow_absolute[layer], max_abs);
    if (relative > 1e-3) {
        std::clog << "[kvmem-shadow] layer=" << layer << " frontier=" << i.current_frontier
                  << " relative_l2=" << relative << " max_abs=" << max_abs << '\n';
        if (!i.shadow_fp64_reference)
            throw std::runtime_error("tiered shadow attention relative L2 exceeds 1e-3");
    }
}

TieredSnapshot TieredContext::capture() {
    auto& i = *impl_;
    if (i.next_layer != 16) throw std::logic_error("snapshot requires a complete Main block");
    i.drain();
    auto view = i.table.snapshot();
    for (std::uint32_t layer = 0; layer < 16; ++layer)
        if (i.archive.frontier(layer) != view.frontier)
            throw std::logic_error("snapshot archive/view frontiers disagree");
    std::shared_ptr<const SparseDerivedSnapshot> derived;
    if (i.sparse) {
        if (i.sparse->index().frontier() != view.frontier)
            throw std::logic_error("Main snapshot accepted index/KV frontiers disagree");
        derived = std::make_shared<const SparseDerivedSnapshot>(i.sparse->capture_snapshot());
    }
    i.invalidate_captures(i.current_frontier);
    auto lifetime = std::make_shared<TieredSnapshotLifetime>();
    lifetime->frontier = view.frontier;
    i.captures.push_back(lifetime);
    TieredSnapshot result{i.bundle_identity, i.archive.generation(), view.frontier, view.frontier,
                          view.generation, std::move(lifetime), std::move(derived)};
    if (i.plan.mode == KvMode::KVMem) {
        result.selected_logical_ids = i.logical_resident(); result.exact_hard_logical_ids = i.exact_hard;
        result.recent_tokens = i.plan.recent_tokens; result.reserve_pages = i.plan.reserve_pages;
        result.guard_pages = i.plan.guard_pages; result.phase = i.sparse_phase;
    }
    return result;
}
kvmem::KVViewSnapshot TieredContext::current_view() const { return impl_->table.snapshot(); }

void TieredContext::restore(const TieredSnapshot& saved, cudaStream_t stream) {
    auto& i = *impl_;
    i.check_stream(stream);
    if (saved.bundle_identity != i.bundle_identity || !saved.lifetime || !saved.lifetime->valid ||
        saved.lifetime->frontier != saved.frontier || saved.archive_frontier != saved.frontier ||
        saved.frontier > i.current_frontier || saved.view_generation > i.table.snapshot().generation ||
        saved.archive_generation > i.archive.generation())
        throw std::logic_error("Main snapshot is stale or belongs to a different bundle");
    if (i.sparse) {
        if (!saved.sparse || saved.sparse->frontier != saved.frontier ||
            !i.sparse->snapshot_valid(*saved.sparse))
            throw std::invalid_argument("Main snapshot missing/foreign/stale derived authority");
    } else if (saved.sparse) throw std::invalid_argument("Main snapshot derived state without owner");
    const bool logical_restore = i.plan.mode == KvMode::KVMem;
    if (logical_restore) {
        const auto sorted_unique = [](const auto& pages) {
            return std::is_sorted(pages.begin(), pages.end()) &&
                std::adjacent_find(pages.begin(), pages.end()) == pages.end();
        };
        if (!sorted_unique(saved.selected_logical_ids) || !sorted_unique(saved.exact_hard_logical_ids) ||
            saved.selected_logical_ids.size() > i.plan.view_pages ||
            (!saved.selected_logical_ids.empty() && saved.selected_logical_ids.back() >= page_count(saved.frontier)) ||
            !std::includes(saved.selected_logical_ids.begin(), saved.selected_logical_ids.end(),
                           saved.exact_hard_logical_ids.begin(), saved.exact_hard_logical_ids.end()) ||
            saved.recent_tokens != i.plan.recent_tokens || saved.reserve_pages != i.plan.reserve_pages ||
            saved.guard_pages != i.plan.guard_pages ||
            (saved.phase != SparseExecutionPhase::ExactPrefill && saved.phase != SparseExecutionPhase::SparseDecode))
            throw std::invalid_argument("Main sparse snapshot invalid logical/hard/policy state");
        for (std::uint32_t p = 0; p < std::min(i.plan.sink_pages, page_count(saved.frontier)); ++p)
            if (!std::binary_search(saved.selected_logical_ids.begin(), saved.selected_logical_ids.end(), p))
                throw std::invalid_argument("Main sparse snapshot omits sink obligation");
        if (saved.phase == SparseExecutionPhase::SparseDecode && saved.frontier && saved.recent_tokens)
            for (std::uint32_t p = (saved.frontier > saved.recent_tokens ? saved.frontier - saved.recent_tokens : 0) / 64;
                 p < page_count(saved.frontier); ++p)
                if (!std::binary_search(saved.selected_logical_ids.begin(), saved.selected_logical_ids.end(), p))
                    throw std::invalid_argument("Main sparse snapshot omits recent obligation");
    }
    const auto started = std::chrono::steady_clock::now();
    // Borrowed host/device pointers stay live until both workers and consumers are drained.
    i.drain();
    if (logical_restore) i.table.set_generation_protection({}, 0);
    kvmem::KVViewSnapshot restored;
    try { restored = logical_restore ?
        i.table.plan_selection(saved.frontier, saved.selected_logical_ids) : i.table.plan_restore(saved.frontier); }
    catch (...) {
        if (logical_restore) i.table.set_generation_protection(i.exact_hard,
            i.sparse_phase == SparseExecutionPhase::SparseDecode ? i.plan.recent_tokens : 0);
        throw;
    }
    // Hydration runs while the prior archive/index authority is still available for recovery.
    std::size_t scheduled_bytes = 0;
    if (logical_restore) {
        const auto prior = i.logical_resident(); const auto prior_hard = i.exact_hard;
        const auto prior_phase = i.sparse_phase;
        try { scheduled_bytes = i.hydrate(restored); }
        catch (...) {
            const auto cause = std::current_exception();
            i.recover_prior_view(prior, prior_hard, prior_phase, cause);
            std::rethrow_exception(cause);
        }
    }
    try {
        if (i.sparse) {
            if (!saved.sparse || saved.sparse->frontier != saved.frontier)
                throw std::logic_error("Main snapshot missing exact derived index/Q state");
            i.sparse->restore_snapshot(*saved.sparse);
        } else if (saved.sparse) throw std::logic_error("Main derived snapshot has no capture owner");
        i.transfer.trim(saved.frontier);
        i.invalidate_captures(saved.frontier);
        i.writes.clear();
        i.write_ids.clear();
        i.tickets.clear();
        if (!logical_restore) scheduled_bytes = i.hydrate(restored);
        if (logical_restore) {
            i.table.install_selection(restored);
            i.exact_hard = saved.exact_hard_logical_ids;
            i.sparse_phase = saved.phase;
            i.table.set_generation_protection(i.exact_hard,
                saved.phase == SparseExecutionPhase::SparseDecode ? saved.recent_tokens : 0);
        } else i.table.install_restore(restored);
        i.snapshot = i.table.snapshot();
        i.current_frontier = restored.frontier;
        i.block_tokens = 0;
        i.next_layer = 16;
        i.publish();
        i.checked(cudaStreamSynchronize(stream), "restored Main publication drain", i.plan.mode == KvMode::KVMem);
    } catch (...) {
        i.poisoned = true;
        const auto failure = i.drain_failed_operation(std::current_exception(), "failed restore compute drain");
        if (failure.unrecoverable || i.unrecoverable) { i.unrecoverable = true; throw; }
        throw KVMemColdResetRequired("late Main restore failure after victim DMA: cold exact rebuild required");
    }
    std::clog << "[kvmem-restore] frontier=" << restored.frontier
              << " retained_pages=" << restored.resident.size() - restored.hydration_pages.size()
              << " missing_pages=" << restored.hydration_pages.size()
              << " scheduled_h2d_bytes=" << scheduled_bytes
              << " elapsed_ms=" << std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - started).count() << '\n';
}

void TieredContext::trim(std::uint32_t frontier, cudaStream_t stream) {
    auto& i = *impl_;
    i.check_stream(stream);
    if (frontier > i.current_frontier)
        throw std::out_of_range("tiered trim cannot grow its frontier");
    i.drain();
    if (i.sparse) i.sparse->trim(frontier);
    i.transfer.trim(frontier);
    i.table.trim(frontier);
    i.invalidate_captures(frontier);
    i.snapshot = i.table.snapshot();
    i.current_frontier = frontier;
    i.writes.clear();
    i.write_ids.clear();
    i.tickets.clear();
    i.next_layer = 16;
    i.publish();
    i.checked(cudaStreamSynchronize(stream), "trimmed Main publication drain", i.plan.mode == KvMode::KVMem);
}

void TieredContextTestAccess::poison_transfer(TieredContext& owner, std::exception_ptr cause) {
    kvmem::HostKVTransferTestAccess::poison(owner.impl_->transfer, std::move(cause));
}
void TieredContextTestAccess::drain_borrowers(TieredContext& owner, cudaError_t status, const char* operation) {
    auto& i = *owner.impl_;
    i.seed_historical_failure();
    auto first = i.unrecoverable_cause;
    kvmem::check_transfer_cuda(cudaStreamSynchronize(i.compute), "test actual normal Main drain", true);
    i.drain_borrowers_status(first, status, operation);
}
std::function<void()> TieredContextTestAccess::score_query_drain(TieredContext& owner) {
    return owner.impl_->score_query_drain();
}
void TieredContextTestAccess::finish_score_query_borrow(TieredContext& owner, kvmem::QueryCaptureBorrow& borrower) {
    owner.impl_->finish_score_query_borrow(borrower);
}
void TieredContextTestAccess::drain_after_quiesce(TieredContext& owner, cudaError_t quiesce_status,
    const char* quiesce_operation, cudaError_t compute_status, const char* compute_operation) {
    auto& i = *owner.impl_;
    i.seed_historical_failure();
    auto strongest = i.unrecoverable_cause;
    (void)i.transfer.quiesce(); // real healthy worker/event drains before boundary status feed
    kvmem::HostKVTransferTestAccess::record_drain(i.transfer, quiesce_status, quiesce_operation);
    const auto failure = i.transfer.failure_state();
    i.prepare_after_quiesce(strongest);
    kvmem::check_transfer_cuda(cudaStreamSynchronize(i.compute), "test actual post-quiesce compute drain", true);
    (void)i.finish_failed_operation(strongest, failure, compute_status, compute_operation);
}
void TieredContext::check_compute_drain(cudaError_t status, const char* operation) {
    impl_->throw_if_unrecoverable();
    impl_->checked(status, operation, true);
}
void TieredContext::synchronize_prefill() {
    check_compute_drain(cudaSuccess, "KVMem inner prefill compute drain");
    check_compute_drain(cudaStreamSynchronize(impl_->compute), "KVMem inner prefill compute drain");
}
void TieredContext::reset(cudaStream_t stream) { reset_impl(stream, nullptr, nullptr); }
void TieredContext::reset_impl(cudaStream_t stream, void (*cleanup)(void*), void* value) {
    auto& i = *impl_;
    i.check_stream(stream);
    if (i.plan.mode == KvMode::KVMem) {
        const auto failure = i.drain_failed_operation({}, "cold Main reset compute drain");
        if (i.unrecoverable || failure.unrecoverable)
            throw std::runtime_error("failed Main drain/context: cold reset cannot repair existing CUDA resources");
        // Construct all potentially allocating host replacements before issuing
        // clears. Their publication waits for the complete correlated cleanup.
        auto reset_table = i.table;
        reset_table.reset();
        auto reset_snapshot = reset_table.snapshot();
        std::vector<std::int32_t> reset_prefix{0};
        i.poisoned = true;
        try {
            (void)i.transfer.recover();
            if (i.sparse) i.sparse->prepare_reset();
            i.checked(cudaMemsetAsync(i.plan.block_table.bind(i.backing).data, 0xff,
                i.block_table.size() * sizeof(std::int32_t), i.compute), "cold Main reset block table");
            i.checked(cudaMemsetAsync(i.plan.access[0].bind(i.backing).data, 0,
                i.plan.access[0].region.bytes, i.compute), "cold Main reset access");
            i.checked(cudaMemsetAsync(i.plan.prefix[0].bind(i.backing).data, 0,
                i.plan.prefix[0].region.bytes, i.compute), "cold Main reset prefix");
            i.checked(cudaStreamSynchronize(i.compute), "cold Main reset publication drain", true);
            if (cleanup) cleanup(value);
            if (i.sparse) i.sparse->commit_reset();
            // transfer completions were retired before the GPU work. Trimming
            // the archive here publishes its new frontier together with Main.
            i.transfer.commit_empty_reset();
            i.table = std::move(reset_table); i.snapshot = std::move(reset_snapshot);
            std::fill(i.block_table.begin(), i.block_table.end(), -1);
            i.access.clear(); i.prefix = std::move(reset_prefix);
            i.pending.fill({}); i.writes.clear(); i.write_ids.clear(); i.tickets.clear();
            i.current_frontier = 0; i.block_tokens = 0; i.next_layer = 16;
            i.exact_hard.clear(); i.sparse_phase = SparseExecutionPhase::ExactPrefill;
            i.poisoned = false;
            i.test_fault = {};
        } catch (...) {
            i.poisoned = true;
            try { throw; }
            catch (const kvmem::CudaTransferError& error) {
                if (kvmem::cuda_transfer_failure_is_unrecoverable(error.status(), error.drain_failed())) {
                    i.unrecoverable = true;
                    kvmem::prefer_unrecoverable_exception(i.unrecoverable_cause, std::current_exception());
                }
            } catch (...) {}
            throw;
        }
    } else {
        if (i.sparse) i.sparse->reset();
        trim(0, stream);
        if (cleanup) cleanup(value);
    }
    for (const auto& weak : i.captures)
        if (auto capture = weak.lock()) capture->valid = false;
    i.captures.clear();
    i.bundle_identity = next_bundle_identity.fetch_add(1);
    i.caller_bundle = 0; i.exact_hard.clear();
    i.sparse_phase = SparseExecutionPhase::ExactPrefill;
    i.transaction_kind.reset(); i.capture_wait_ms = 0;
}

void TieredContext::drain() { impl_->drain(); }

void TieredContext::log_stats() {
    auto& i = *impl_;
    i.collect_times();
    std::clog << "[kvmem] runtime_blocks=" << i.blocks << " partial_passes=" << i.partial_passes
              << " streamed_bytes=" << i.streamed_bytes << " frontier=" << i.current_frontier
              << " view_pages=" << i.plan.view_pages << " shadow=" << shadow() << '\n';
    if (shadow())
        std::clog << "[kvmem-shadow] max_relative_l2=" << i.shadow_max_relative
                  << " max_abs=" << i.shadow_max_abs << '\n';
    if (shadow() && i.shadow_fp64_reference)
        std::clog << "[kvmem-shadow] criterion=offline_fp64 status=not_evaluated; "
                     "independent reference results required\n";
    if (shadow())
        for (std::size_t layer = 0; layer < 16; ++layer)
            std::clog << "[kvmem-shadow] layer=" << layer
                      << " max_relative_l2=" << i.shadow_relative[layer]
                      << " max_abs=" << i.shadow_absolute[layer] << '\n';
    if (i.plan.measure_transfer_waits)
        for (std::size_t phase = 0; phase < 2; ++phase)
            for (std::size_t layer = 0; layer < 16; ++layer)
                std::clog << "[kvmem-transfer] phase=" << (phase==0 ? "prefill" : "decode")
                          << " layer=" << layer << " gpu_wait_ms=" << i.wait_ms[phase][layer]
                          << " cpu_enqueue_wait_ms=" << i.enqueue_wait_ms[phase][layer] << '\n';
}

} // namespace ninfer::targets::qwen3_6::detail

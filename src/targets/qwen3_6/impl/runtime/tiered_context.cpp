#include "targets/qwen3_6/impl/runtime/tiered_context.h"
#include "core/device.h"
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

namespace ninfer::targets::qwen3_6::detail {
namespace {
std::uint32_t page_count(std::uint32_t tokens) { return tokens / 64 + (tokens % 64 != 0); }

std::atomic<std::uint64_t> next_bundle_identity{1};

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
    kvmem::HostKVArchive archive;
    kvmem::HostKVTransferEngine transfer;
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

    Impl(const TieredRuntimePlan& p, PagedKVCache& m, DeviceSpan b, cudaStream_t stream)
        : plan(p), main(m), backing(b), compute(stream),
          table(p.logical_tokens, p.view_pages, p.sink_pages, 16),
          archive(p.archive, p.archive_mode, p.lock_archive),
          transfer(archive, p.staging_region.bind(b), p.staging) {
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
            if (p.measure_transfer_waits) {
                for (std::size_t layer = 0; layer < 16; ++layer) {
                    CUDA_CHECK(cudaEventCreate(&wait_start[layer]));
                    CUDA_CHECK(cudaEventCreate(&wait_end[layer]));
                }
            }
            // Main pool construction/touch-fill may use Stream 0. Publish only after draining it.
            CUDA_CHECK(cudaDeviceSynchronize());
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
        for (auto& events : {&wait_start, &wait_end})
            for (auto event : *events)
                if (event) (void)cudaEventDestroy(event);
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
    }

    void collect_times() {
        if (!plan.measure_transfer_waits) return;
        for (std::uint32_t layer = 0; layer < 16; ++layer) {
            if (!measured[layer]) continue;
            CUDA_CHECK(cudaEventSynchronize(wait_end[layer]));
            float elapsed = 0;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, wait_start[layer], wait_end[layer]));
            wait_ms[execution_phase][layer] += elapsed;
            measured[layer] = false;
        }
    }

    void drain() {
        CUDA_CHECK(cudaStreamSynchronize(compute));
        transfer.synchronize();
        complete_writebacks();
        collect_times();
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
        const auto pages = static_cast<std::int32_t>(snapshot.host_only.size());
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
                CUDA_CHECK(cudaMemcpyAsync(destination, transfer.staged(ticket).data,
                                           spec.page_bytes, cudaMemcpyDeviceToDevice, compute));
                transfer.release(ticket, compute);
            }
        }
    }

    void prefetch(std::uint32_t layer) {
        if (!tickets.empty()) throw std::logic_error("tiered prefetch has unreleased consumers");
        for (std::size_t plane = 0; plane < plan.archive.layers[layer].size(); ++plane) {
            const auto page_bytes = plan.archive.layers[layer][plane].page_bytes;
            const auto tile_pages = kvmem::kHostKVTransferTileBytes / page_bytes;
            if (!tile_pages)
                throw std::logic_error("tiered archive page exceeds one transfer tile");
            for (std::size_t first = 0; first < snapshot.host_only.size();) {
                std::size_t count = 1;
                while (first + count < snapshot.host_only.size() && count < tile_pages &&
                       snapshot.host_only[first + count] == snapshot.host_only[first] + count)
                    ++count;
                const auto offset = plan.staging_plane_regions[plane].offset -
                                    plan.staging_region.offset + first * page_bytes;
                tickets.push_back(
                    transfer.prefetch_completed(layer, plane, snapshot.host_only[first],
                                                static_cast<std::uint32_t>(count), offset));
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
            const auto mixed =
                actual < 0 ? main.pool().page_group_count() + host++ : std::uint32_t(actual);
            if (mixed > std::uint32_t(std::numeric_limits<std::int32_t>::max()))
                throw std::overflow_error("tiered physical access ID exceeds int32");
            access.push_back(
                {static_cast<std::int32_t>(logical), static_cast<std::int32_t>(mixed)});
        }
        prefix =
            ops::attention_access_prefix(access, current_frontier, main.pool().page_group_count(),
                                         static_cast<std::uint32_t>(snapshot.host_only.size()));
        CUDA_CHECK(cudaMemcpyAsync(plan.block_table.bind(backing).data, block_table.data(),
                                   block_table.size() * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, compute));
        CUDA_CHECK(cudaMemcpyAsync(plan.access[0].bind(backing).data, access.data(),
                                   access.size() * sizeof(ops::AttentionPageAccess),
                                   cudaMemcpyHostToDevice, compute));
        CUDA_CHECK(cudaMemcpyAsync(plan.prefix[0].bind(backing).data, prefix.data(),
                                   prefix.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   compute));
    }

    void hydrate(const kvmem::KVViewSnapshot& restored) {
        const auto offset = plan.staging.maximum_layer_stream_bytes;
        for (std::uint32_t layer = 0; layer < 16; ++layer)
            for (std::size_t plane = 0; plane < plan.archive.layers[layer].size(); ++plane) {
                const auto& spec = plan.archive.layers[layer][plane];
                const auto tile_pages = kvmem::kHostKVTransferTileBytes / spec.page_bytes;
                const auto& tensor = main.pool().plane(spec.pool_plane);
                for (std::size_t first = 0; first < restored.resident.size();) {
                    std::size_t count = 1;
                    while (first + count < restored.resident.size() && count < tile_pages &&
                           restored.resident[first + count].logical_page ==
                               restored.resident[first].logical_page + count) ++count;
                    auto ticket = transfer.prefetch_completed(layer, plane,
                        restored.resident[first].logical_page, static_cast<std::uint32_t>(count), offset);
                    transfer.wait(ticket, compute);
                    const auto staged = transfer.staged(ticket);
                    // Logical host ranges are coalesced. Lease IDs may be noncontiguous.
                    for (std::size_t j = 0; j < count;) {
                        const auto& page = restored.resident[first + j];
                        const auto physical_id = lease_ids.at(plan.shadow_validate ? page.logical_page :
                                                              std::uint32_t(page.physical_slot));
                        std::size_t run = 1;
                        if (tensor.nb[3] == spec.page_bytes)
                            while (j + run < count) {
                                const auto& next = restored.resident[first + j + run];
                                if (lease_ids.at(plan.shadow_validate ? next.logical_page :
                                                 std::uint32_t(next.physical_slot)) != physical_id + run) break;
                                ++run;
                            }
                        auto* target = static_cast<std::byte*>(tensor.data) + physical_id * tensor.nb[3];
                        CUDA_CHECK(cudaMemcpyAsync(target, static_cast<const std::byte*>(staged.data) +
                            j * spec.page_bytes, run * spec.page_bytes, cudaMemcpyDeviceToDevice, compute));
                        j += run;
                    }
                    transfer.release(ticket, compute);
                    first += count;
                }
            }
        CUDA_CHECK(cudaStreamSynchronize(compute));
        transfer.synchronize();
    }

    void begin(std::uint32_t base, std::uint32_t count, cudaStream_t stream, ExecutionPhase phase) {
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
        writes = table.begin_append(base, count);
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
                    CUDA_CHECK(cudaMemsetAsync(page, 0, spec.page_bytes, compute));
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
        if (plan.measure_transfer_waits) CUDA_CHECK(cudaEventRecord(wait_start[layer], compute));
        const auto start = plan.measure_transfer_waits ? std::chrono::steady_clock::now()
                                                       : std::chrono::steady_clock::time_point{};
        for (auto ticket : tickets) transfer.wait(ticket, compute);
        if (plan.measure_transfer_waits) {
            enqueue_wait_ms[execution_phase][layer] +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
            CUDA_CHECK(cudaEventRecord(wait_end[layer], compute));
            measured[layer] = true;
        }
        const auto tokens        = static_cast<std::int32_t>(block_tokens);
        const auto resident_view = resident(layer), staging_view = staged(layer);
        const auto preferred = tokens > 4 ? 1 : (resident_view.packed_k ? 64 : 32);
        // Bound the launch by both actual key count and the independent fixed scratch planes.
        const auto visible_splits = std::max(1, static_cast<int>((current_frontier + 63) / 64));
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
    }
};

TieredContext::TieredContext(const TieredRuntimePlan& plan, PagedKVCache& main, DeviceSpan backing,
                             cudaStream_t stream)
    : impl_(std::make_unique<Impl>(plan, main, backing, stream)) {}

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

void TieredContext::begin_block(std::uint32_t base, std::uint32_t count, cudaStream_t stream, ExecutionPhase phase) {
    impl_->begin(base, count, stream, phase);
}

PagedKVLayerView TieredContext::resident_layer(std::uint32_t layer) const {
    return impl_->resident(layer);
}

void TieredContext::attention(std::uint32_t layer, const Tensor& q, const Tensor& k,
                              const Tensor& v, const Tensor& positions, float scale, Tensor& out,
                              cudaStream_t stream) {
    auto& i = *impl_;
    i.check_stream(stream);
    if (shadow() || layer != i.next_layer || layer >= 16)
        throw std::logic_error("tiered append attention route or layer is invalid");
    if (out.ne[2] < static_cast<std::int32_t>(i.block_tokens))
        throw std::invalid_argument("tiered output is shorter than its block");
    const auto count = static_cast<std::int32_t>(i.block_tokens);
    if (out.ne[2] > count) {
        auto tail = out.slice(2, count, out.ne[2] - count);
        CUDA_CHECK(cudaMemsetAsync(tail.data, 0, tail.bytes(), stream));
    }
    ops::gqa_kv_append(k.slice(2, 0, count), v.slice(2, 0, count), positions.slice(0, 0, count),
                       i.resident(layer), stream);
    i.writeback(layer);
    i.partial(layer, q, positions, scale, out);
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

    CUDA_CHECK(cudaMemcpyAsync(i.shadow_dense->data(), dense.data, dense.bytes(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(i.shadow_tiered->data(), out.data, out.bytes(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
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
    i.invalidate_captures(i.current_frontier);
    auto lifetime = std::make_shared<TieredSnapshotLifetime>();
    lifetime->frontier = view.frontier;
    i.captures.push_back(lifetime);
    return {i.bundle_identity, i.archive.generation(), view.frontier, view.frontier, view.generation, std::move(lifetime)};
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
    // Borrowed host/device pointers stay live until both workers and consumers are drained.
    i.drain();
    const auto restored = i.table.plan_restore(saved.frontier);
    i.transfer.trim(saved.frontier);
    i.invalidate_captures(saved.frontier);
    i.writes.clear();
    i.write_ids.clear();
    i.tickets.clear();
    i.hydrate(restored);
    i.table.install_restore(restored);
    i.snapshot = i.table.snapshot();
    i.current_frontier = restored.frontier;
    i.block_tokens = 0;
    i.next_layer = 16;
    i.publish();
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void TieredContext::trim(std::uint32_t frontier, cudaStream_t stream) {
    auto& i = *impl_;
    i.check_stream(stream);
    if (frontier > i.current_frontier)
        throw std::out_of_range("tiered trim cannot grow its frontier");
    i.drain();
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
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void TieredContext::reset(cudaStream_t stream) {
    trim(0, stream);
    for (const auto& weak : impl_->captures)
        if (auto capture = weak.lock()) capture->valid = false;
    impl_->captures.clear();
    impl_->bundle_identity = next_bundle_identity.fetch_add(1);
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

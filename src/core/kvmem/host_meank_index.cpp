#include "core/kvmem/host_meank_index.h"
#include "core/arena.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::kvmem {
namespace {
std::size_t add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("MeanK size addition");
    return a + b;
}
std::size_t mul(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error("MeanK size multiplication");
    return a * b;
}
std::size_t region(std::size_t& end, std::size_t bytes) {
    end = add(end, 255) & ~std::size_t(255);
    auto offset = end;
    end = add(end, bytes);
    return offset;
}
std::atomic<std::uint64_t> next_owner{1};
} // namespace
MeanKResources plan_meank_resources(std::uint32_t max_context, std::uint32_t layers,
                                    std::uint32_t dimension, std::uint32_t heads,
                                    std::uint32_t max_chunk) {
    if (!max_context || max_context > 1048576 || !layers || layers > 65535 || !dimension ||
        dimension > 1024 || !heads || heads > 64 || !max_chunk || max_chunk > max_context)
        throw std::invalid_argument("MeanK resource domain");
    MeanKResources p;
    p.max_context = max_context;
    p.layers = layers;
    p.dimension = dimension;
    p.heads = heads;
    p.max_chunk = max_chunk;
    p.pages = (max_context + 63) / 64;
    p.patch_pages = (std::max(max_chunk, std::uint32_t(16)) + 126) / 64;
    p.row_elements = mul(dimension, heads);
    p.layer_stride_bytes = mul(mul(p.pages, p.row_elements), 2);
    p.index_bytes = mul(p.layer_stride_bytes, layers);
    p.sum_bytes = mul(mul(p.row_elements, layers), 4);
    p.tail_bytes = mul(mul(mul(p.row_elements, layers), 64), 2);
    p.provisional_bytes = mul(mul(mul(p.row_elements, layers), 16), 2);
    p.means_bytes = mul(mul(mul(p.row_elements, layers), p.patch_pages), 2);
    p.snapshot_bytes = mul(p.sum_bytes, 2);
    p.host_continuation_bytes = mul(p.sum_bytes, 2);
    p.host_patch_bytes = add(p.means_bytes, p.sum_bytes);
    std::size_t end = 0;
    p.active_sum_offset = region(end, p.sum_bytes);
    p.base_sum_offset = region(end, p.sum_bytes);
    p.seed_sum_offset = region(end, p.sum_bytes);
    p.output_sum_offset = region(end, p.sum_bytes);
    p.tail_offset = region(end, p.tail_bytes);
    p.provisional_offset = region(end, p.provisional_bytes);
    p.means_offset = region(end, p.means_bytes);
    p.counts_offset = region(end, mul(layers, 4));
    p.device_bytes = end;
    return p;
}
std::uint64_t meank_host_admission_bytes(const MeanKResources& p) {
    return add(add(add(add(p.index_bytes, p.snapshot_bytes), p.host_continuation_bytes),
                   p.host_patch_bytes),
               std::uint64_t(4) << 30);
}
class MeanKSnapshot {
    friend class HostMeanKIndex;
    std::uint64_t owner{}, epoch{}, generation{};
    std::uint32_t frontier{};
    mutable bool valid = true;
    std::vector<float> sums;
};
struct HostMeanKIndex::Impl {
    MeanKResources plan;
    std::unique_ptr<PinnedHostBuffer> pin;
    std::vector<std::uint16_t> storage, patches;
    std::vector<float> sums, base_sums, patch_sums;
    std::vector<bool> completed;
    std::array<std::weak_ptr<const MeanKSnapshot>, 2> snapshots;
    std::uint64_t owner = next_owner.fetch_add(1), epoch = 1, generation = 1, serial = 0;
    std::uint32_t frontier = 0, base = 0;
    bool published = true, pending = false, accepted = false;
    MeanKTicket ticket;
    std::uint16_t* data() {
        return pin ? static_cast<std::uint16_t*>(pin->data()) : storage.data();
    }
    void layer(std::uint32_t layer) const {
        if (layer >= plan.layers)
            throw std::out_of_range("MeanK layer");
    }
    bool matches(const MeanKTicket& candidate) const {
        return pending && this->ticket == candidate;
    }
    MeanKTicket prepare(MeanKTransactionKind kind, std::uint32_t first, std::uint32_t count) {
        if (pending)
            throw std::logic_error("MeanK transaction already pending");
        ticket = {owner, generation, ++serial, first, count, kind};
        pending = true;
        published = false;
        accepted = kind != MeanKTransactionKind::SpeculativeMain;
        std::fill(completed.begin(), completed.end(), false);
        return ticket;
    }
};
HostMeanKIndex::HostMeanKIndex(MeanKResources p, bool pinned) : impl_(std::make_unique<Impl>()) {
    // Recompute to reject forged offsets/capacities before any allocation.
    auto verified =
        plan_meank_resources(p.max_context, p.layers, p.dimension, p.heads, p.max_chunk);
    if (p != verified)
        throw std::invalid_argument("MeanK resource layout is not canonical");
    impl_->plan = verified;
    auto& state = *impl_;
    if (pinned)
        state.pin = std::make_unique<PinnedHostBuffer>(verified.index_bytes);
    else
        state.storage.resize(verified.index_bytes / 2);
    state.patches.resize(verified.means_bytes / 2);
    state.sums.resize(verified.sum_bytes / 4);
    state.base_sums.resize(verified.sum_bytes / 4);
    state.patch_sums.resize(verified.sum_bytes / 4);
    state.completed.resize(verified.layers);
    std::memset(state.data(), 0, verified.index_bytes);
}
HostMeanKIndex::~HostMeanKIndex() = default;
const MeanKResources& HostMeanKIndex::resources() const noexcept {
    return impl_->plan;
}
bool HostMeanKIndex::pinned() const noexcept {
    return bool(impl_->pin);
}
bool HostMeanKIndex::published() const noexcept {
    return impl_->published;
}
std::uint32_t HostMeanKIndex::frontier() const noexcept {
    return impl_->frontier;
}
std::uint32_t HostMeanKIndex::base_frontier() const noexcept {
    return impl_->base;
}
std::uint64_t HostMeanKIndex::generation() const noexcept {
    return impl_->generation;
}
std::span<const std::uint16_t> HostMeanKIndex::layer_data(std::uint32_t layer) const {
    auto& state = *impl_;
    state.layer(layer);
    if (!state.published)
        throw std::logic_error("MeanK index not published");
    return {state.data() + std::size_t(layer) * (state.plan.layer_stride_bytes / 2),
            std::size_t((state.frontier + 63) / 64) * state.plan.row_elements};
}
std::span<const float> HostMeanKIndex::prefix_sum(std::uint32_t layer) const {
    auto& state = *impl_;
    state.layer(layer);
    if (state.pending && state.ticket.kind == MeanKTransactionKind::Trim)
        throw std::logic_error("MeanK trim requires exact base replay, not active sum");
    return {state.sums.data() + std::size_t(layer) * state.plan.row_elements,
            state.plan.row_elements};
}
std::span<const float> HostMeanKIndex::base_sum(std::uint32_t layer) const {
    auto& state = *impl_;
    state.layer(layer);
    return {state.base_sums.data() + std::size_t(layer) * state.plan.row_elements,
            state.plan.row_elements};
}
MeanKTicket HostMeanKIndex::begin(MeanKTransactionKind kind, std::uint32_t count) {
    auto& state = *impl_;
    if (!count || count > state.plan.max_context - state.frontier ||
        (kind == MeanKTransactionKind::ExactPrefill      ? count > state.plan.max_chunk
         : kind == MeanKTransactionKind::OrdinaryMain    ? count != 1
         : kind == MeanKTransactionKind::SpeculativeMain ? count > 16
                                                         : true))
        throw std::invalid_argument("MeanK capture transaction/count");
    return state.prepare(kind, state.frontier, count);
}
MeanKTicket HostMeanKIndex::accept(const MeanKTicket& ticket, std::uint32_t n) {
    auto& state = *impl_;
    if (!state.matches(ticket) || ticket.kind != MeanKTransactionKind::SpeculativeMain ||
        state.accepted || n > ticket.count)
        throw std::invalid_argument("MeanK accepted-prefix transaction");
    if (!n) {
        discard(ticket);
        return ticket;
    }
    state.ticket.count = n;
    state.accepted = true;
    return state.ticket;
}
bool HostMeanKIndex::complete_layer(const MeanKTicket& ticket, std::uint32_t layer,
                                    std::span<const std::uint16_t> mean,
                                    std::span<const float> sum) {
    auto& state = *impl_;
    if (!state.matches(ticket))
        return false;
    state.layer(layer);
    if (!state.accepted)
        throw std::logic_error("MeanK speculative acceptance not finalized");
    if (state.completed[layer])
        return false;
    auto rows = state.plan.row_elements;
    auto pages = (ticket.first % 64 + ticket.count + 63) / 64;
    if (mean.size() != std::size_t(pages) * rows || sum.size() != rows ||
        !std::all_of(sum.begin(), sum.end(), [](float v) { return std::isfinite(v); }) ||
        !std::all_of(mean.begin(), mean.end(),
                     [](std::uint16_t v) { return (v & 0x7c00) != 0x7c00; }) ||
        ((ticket.first + ticket.count) % 64 == 0 &&
         !std::all_of(sum.begin(), sum.end(), [](float v) { return v == 0.f; })))
        throw std::invalid_argument("MeanK completed patch size/value");
    auto patch_stride = std::size_t(state.plan.patch_pages) * rows;
    std::copy(mean.begin(), mean.end(), state.patches.begin() + layer * patch_stride);
    std::copy(sum.begin(), sum.end(), state.patch_sums.begin() + layer * rows);
    state.completed[layer] = true;
    if (!std::all_of(state.completed.begin(), state.completed.end(), [](bool b) { return b; }))
        return true;
    for (std::uint32_t layer = 0; layer < state.plan.layers; ++layer)
        std::copy_n(state.patches.data() + layer * patch_stride, mean.size(),
                    state.data() + layer * (state.plan.layer_stride_bytes / 2) +
                        std::size_t(ticket.first / 64) * rows);
    state.sums.swap(state.patch_sums);
    auto old = state.frontier;
    state.frontier = ticket.first + ticket.count;
    if (ticket.kind == MeanKTransactionKind::Trim) {
        for (auto& slot : state.snapshots)
            if (auto snapshot = slot.lock(); snapshot && snapshot->frontier > state.frontier)
                snapshot->valid = false;
    }
    if (ticket.kind == MeanKTransactionKind::Restore)
        state.base = state.frontier;
    else if (state.frontier / 64 != old / 64) {
        state.base = state.frontier / 64 * 64;
        std::fill(state.base_sums.begin(), state.base_sums.end(), 0.f);
    }
    state.pending = false;
    state.published = true;
    ++state.generation;
    return true;
}
void HostMeanKIndex::discard(const MeanKTicket& ticket) {
    auto& state = *impl_;
    if (!state.matches(ticket))
        throw std::invalid_argument("MeanK discard stale transaction");
    if (ticket.kind == MeanKTransactionKind::Restore)
        throw std::logic_error("MeanK restored partial must be republished before discard");
    state.pending = false;
    state.published = true;
    ++state.generation;
}
MeanKSnapshotHandle HostMeanKIndex::capture() {
    auto& state = *impl_;
    if (!state.published || state.pending)
        throw std::logic_error("MeanK snapshot requires published index");
    auto slot = std::find_if(state.snapshots.begin(), state.snapshots.end(),
                             [](const auto& p) { return p.expired(); });
    if (slot == state.snapshots.end())
        throw std::logic_error("MeanK maximum two live prefix snapshots");
    auto snapshot = std::make_shared<MeanKSnapshot>();
    snapshot->owner = state.owner;
    snapshot->epoch = state.epoch;
    snapshot->generation = state.generation;
    snapshot->frontier = state.frontier;
    snapshot->sums = state.sums;
    *slot = snapshot;
    return snapshot;
}
MeanKTicket HostMeanKIndex::restore(const MeanKSnapshotHandle& snapshot) {
    auto& state = *impl_;
    if (!snapshot || !snapshot->valid || snapshot->owner != state.owner ||
        snapshot->epoch != state.epoch || snapshot->frontier > state.frontier)
        throw std::invalid_argument("MeanK snapshot owner/epoch/frontier");
    // Caller has drained the in-flight transaction. Its completion identity is
    // invalidated before the saved exact prefix becomes the new continuation.
    state.pending = false;
    ++state.generation;
    state.frontier = snapshot->frontier;
    state.base = snapshot->frontier;
    state.sums = snapshot->sums;
    state.base_sums = snapshot->sums;
    for (auto& w : state.snapshots)
        if (auto p = w.lock(); p && p->frontier > snapshot->frontier)
            p->valid = false;
    return state.prepare(MeanKTransactionKind::Restore, state.frontier, 0);
}
MeanKTicket HostMeanKIndex::begin_trim(std::uint32_t f) {
    auto& state = *impl_;
    if (f > state.frontier || f < state.base || f / 64 != state.frontier / 64)
        throw std::invalid_argument("MeanK trim before bounded exact base requires snapshot");
    return state.prepare(MeanKTransactionKind::Trim, f, 0);
}
void HostMeanKIndex::reset() {
    auto& state = *impl_;
    state.pending = false;
    state.frontier = 0;
    state.base = 0;
    state.published = true;
    ++state.generation;
    ++state.epoch;
    std::fill(state.sums.begin(), state.sums.end(), 0.f);
    std::fill(state.base_sums.begin(), state.base_sums.end(), 0.f);
    for (auto& w : state.snapshots)
        if (auto p = w.lock())
            p->valid = false;
}
} // namespace ninfer::kvmem

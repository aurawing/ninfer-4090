#include "core/kvmem/kv_view_table.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::kvmem {
namespace {
std::uint32_t page_count(std::uint32_t tokens) noexcept {
    return tokens / kKVViewPageTokens + (tokens % kKVViewPageTokens != 0);
}
} // namespace

KVViewTable::KVViewTable(std::uint32_t max_context, std::uint32_t view_physical_pages,
                         std::uint32_t sink_pages, std::uint32_t layers)
    : max_context_(max_context), sink_pages_(sink_pages), layers_(layers) {
    if (max_context == 0 || view_physical_pages == 0 || layers == 0 ||
        view_physical_pages >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        sink_pages > view_physical_pages || sink_pages > page_count(max_context)) {
        throw std::invalid_argument("invalid KV view table dimensions");
    }
    const auto logical_pages = page_count(max_context);
    if (logical_pages > std::numeric_limits<std::size_t>::max() / layers) {
        throw std::length_error("KV view layer completion table is too large");
    }
    pages_.resize(logical_pages);
    slot_owners_.assign(view_physical_pages, -1);
    layer_completion_.resize(static_cast<std::size_t>(logical_pages) * layers);
}

std::vector<KVViewWrite> KVViewTable::begin_append(std::uint32_t first_token, std::uint32_t count) {
    if (first_token != frontier_) {
        throw std::invalid_argument("KV view append must start at the current frontier");
    }
    if (count > max_context_ - frontier_) {
        throw std::out_of_range("KV view append exceeds max context");
    }
    if (count == 0) { return {}; }

    const auto new_frontier  = frontier_ + count;
    const auto first_page    = first_token / kKVViewPageTokens;
    const auto end_page      = page_count(new_frontier);
    const auto old_end       = page_count(frontier_);
    const auto touched_count = end_page - first_page;
    if (touched_count > slot_owners_.size()) {
        throw std::length_error("KV view cannot hold every page of this append");
    }

    std::uint32_t missing_slots = 0;
    for (auto logical = first_page; logical < end_page; ++logical) {
        if (pages_[logical].revision == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("KV view page revision exhausted");
        }
        if (pages_[logical].physical_slot < 0) { ++missing_slots; }
    }

    // All allocating work and admission checks finish before mutating the table.
    std::vector<std::int32_t> available_slots;
    std::vector<std::uint32_t> victims;
    available_slots.reserve(missing_slots);
    victims.reserve(missing_slots);
    for (std::size_t slot = 0; slot < slot_owners_.size() && available_slots.size() < missing_slots;
         ++slot) {
        if (slot_owners_[slot] < 0) { available_slots.push_back(static_cast<std::int32_t>(slot)); }
    }
    // Scan original logical order: the oldest completed non-sink page goes first.
    for (auto logical = sink_pages_; logical < old_end && available_slots.size() < missing_slots;
         ++logical) {
        const auto& descriptor = pages_[logical];
        if (logical >= first_page && logical < end_page) { continue; }
        if (descriptor.state == KVViewState::Both) {
            victims.push_back(logical);
            available_slots.push_back(descriptor.physical_slot);
        }
    }
    if (available_slots.size() != missing_slots) {
        throw std::length_error("KV view has no completed non-sink page available for eviction");
    }

    std::vector<KVViewWrite> writes;
    writes.reserve(touched_count);
    std::size_t next_slot = 0;
    for (auto logical = first_page; logical < end_page; ++logical) {
        const auto& descriptor = pages_[logical];
        const auto slot =
            descriptor.physical_slot >= 0 ? descriptor.physical_slot : available_slots[next_slot++];
        writes.push_back({logical,
                          slot,
                          {generation_, descriptor.revision + 1},
                          logical < old_end && descriptor.state == KVViewState::HostOnly});
    }

    for (auto logical : victims) {
        auto& descriptor                       = pages_[logical];
        slot_owners_[descriptor.physical_slot] = -1;
        descriptor.physical_slot               = -1;
        descriptor.state                       = KVViewState::HostOnly;
    }
    for (const auto& write : writes) {
        auto& descriptor                  = pages_[write.logical_page];
        descriptor.physical_slot          = write.physical_slot;
        descriptor.state                  = KVViewState::DeviceOnly;
        descriptor.revision               = write.epoch.revision;
        descriptor.completed_layers       = 0;
        slot_owners_[write.physical_slot] = static_cast<std::int32_t>(write.logical_page);
        const auto offset                 = static_cast<std::size_t>(write.logical_page) * layers_;
        std::fill_n(layer_completion_.begin() + offset, layers_, std::uint8_t{0});
    }
    frontier_ = new_frontier;
    return writes;
}

bool KVViewTable::complete_writeback(std::uint32_t layer, std::uint32_t logical_page,
                                     KVViewEpoch epoch) noexcept {
    if (layer >= layers_ || logical_page >= page_count(frontier_) ||
        epoch.generation != generation_) {
        return false;
    }
    auto& descriptor = pages_[logical_page];
    if (descriptor.state != KVViewState::DeviceOnly || epoch.revision != descriptor.revision) {
        return false;
    }
    auto& completed = layer_completion_[static_cast<std::size_t>(logical_page) * layers_ + layer];
    if (completed != 0) { return false; }
    completed = 1;
    if (++descriptor.completed_layers == layers_) { descriptor.state = KVViewState::Both; }
    return true;
}

void KVViewTable::trim(std::uint32_t frontier) {
    if (frontier > frontier_) { throw std::out_of_range("KV view trim cannot grow the frontier"); }
    if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("KV view generation exhausted");
    }
    const auto retained_end = page_count(frontier);
    const auto old_end      = page_count(frontier_);
    for (auto logical = retained_end; logical < old_end; ++logical) {
        auto& descriptor = pages_[logical];
        if (descriptor.physical_slot >= 0) { slot_owners_[descriptor.physical_slot] = -1; }
        descriptor.physical_slot    = -1;
        descriptor.state            = KVViewState::HostOnly;
        descriptor.completed_layers = 0;
    }
    for (std::uint32_t logical = 0; logical < retained_end; ++logical) {
        auto& descriptor = pages_[logical];
        if (descriptor.state == KVViewState::DeviceOnly) {
            descriptor.completed_layers = 0;
            const auto offset           = static_cast<std::size_t>(logical) * layers_;
            std::fill_n(layer_completion_.begin() + offset, layers_, std::uint8_t{0});
        }
    }
    frontier_ = frontier;
    ++generation_;
}

void KVViewTable::reset() { trim(0); }

KVViewSnapshot KVViewTable::plan_restore(std::uint32_t frontier) const {
    if (frontier > frontier_) throw std::out_of_range("KV restore cannot grow frontier");
    if (generation_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("KV restore generation exhausted");
    const auto end = page_count(frontier);
    for (std::uint32_t logical = 0; logical < end; ++logical)
        if (pages_[logical].state == KVViewState::DeviceOnly)
            throw std::logic_error("KV restore requires completed archive writebacks");
    KVViewSnapshot result;
    result.frontier = frontier;
    result.generation = generation_ + 1;
    result.blocktable.assign(end, -1);
    const auto sinks = std::min(sink_pages_, end);
    const auto recent = static_cast<std::uint32_t>(slot_owners_.size()) - sinks;
    const auto first_recent = std::max(sinks, end > recent ? end - recent : 0U);
    for (std::uint32_t logical = 0; logical < end; ++logical) {
        if (logical < sinks || logical >= first_recent) {
            const auto slot = static_cast<std::int32_t>(result.resident.size());
            result.blocktable[logical] = slot;
            result.resident.push_back({logical, slot, KVViewState::Both,
                                       {result.generation, pages_[logical].revision}});
        } else result.host_only.push_back(logical);
    }
    return result;
}

void KVViewTable::install_restore(const KVViewSnapshot& restored) {
    const auto expected = plan_restore(restored.frontier);
    if (restored.generation != expected.generation ||
        restored.blocktable != expected.blocktable || restored.resident != expected.resident ||
        restored.host_only != expected.host_only)
        throw std::logic_error("stale or invalid KV restoration plan");
    std::fill(slot_owners_.begin(), slot_owners_.end(), -1);
    std::fill(layer_completion_.begin(), layer_completion_.end(), std::uint8_t{0});
    for (auto& descriptor : pages_) {
        descriptor.physical_slot = -1;
        descriptor.state = KVViewState::HostOnly;
        descriptor.completed_layers = 0;
    }
    for (const auto& page : restored.resident) {
        auto& descriptor = pages_[page.logical_page];
        descriptor.physical_slot = page.physical_slot;
        descriptor.state = KVViewState::Both;
        descriptor.completed_layers = layers_;
        slot_owners_[page.physical_slot] = static_cast<std::int32_t>(page.logical_page);
    }
    frontier_ = restored.frontier;
    generation_ = restored.generation;
}

std::optional<KVViewPage> KVViewTable::page(std::uint32_t logical_page) const noexcept {
    if (logical_page >= page_count(frontier_)) { return {}; }
    const auto& descriptor = pages_[logical_page];
    return KVViewPage{logical_page,
                      descriptor.physical_slot,
                      descriptor.state,
                      {generation_, descriptor.revision}};
}

KVViewSnapshot KVViewTable::snapshot() const {
    KVViewSnapshot result;
    result.frontier       = frontier_;
    result.generation     = generation_;
    const auto live_pages = page_count(frontier_);
    result.blocktable.reserve(live_pages);
    result.resident.reserve(std::min<std::size_t>(live_pages, slot_owners_.size()));
    for (std::uint32_t logical = 0; logical < live_pages; ++logical) {
        const auto descriptor = *page(logical);
        result.blocktable.push_back(descriptor.physical_slot);
        if (descriptor.state == KVViewState::HostOnly) {
            result.host_only.push_back(logical);
        } else {
            result.resident.push_back(descriptor);
        }
    }
    return result;
}
} // namespace ninfer::kvmem

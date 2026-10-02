#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::kvmem {

inline constexpr std::uint32_t kKVViewPageTokens = 64;
enum class KVViewState { DeviceOnly, Both, HostOnly };

struct KVViewEpoch {
    std::uint64_t generation                  = 0;
    std::uint64_t revision                    = 0;
    bool operator==(const KVViewEpoch&) const = default;
};

struct KVViewPage {
    std::uint32_t logical_page = 0;
    std::int32_t physical_slot = -1;
    KVViewState state          = KVViewState::HostOnly;
    KVViewEpoch epoch;
    bool operator==(const KVViewPage&) const = default;
};

struct KVViewWrite {
    std::uint32_t logical_page = 0;
    std::int32_t physical_slot = -1;
    KVViewEpoch epoch;
    // Restore the retained prefix before writing when a partial HostOnly page is touched.
    bool restore_from_host = false;
};

struct KVViewSnapshot {
    std::uint32_t frontier   = 0;
    std::uint64_t generation = 0;
    // Original logical page indexing through ceil(frontier / 64); -1 means HostOnly.
    std::vector<std::int32_t> blocktable;
    std::vector<KVViewPage> resident;
    std::vector<std::uint32_t> host_only;
};

// CPU-only metadata for one sequence. Callers serialize mutations and drain consumers
// before trim/reset. A slot denotes one shared physical page group across every layer;
// runtime translates slots into pool page IDs and owns transfer polling/synchronization.
class KVViewTable {
public:
    KVViewTable(std::uint32_t max_context, std::uint32_t view_physical_pages,
                std::uint32_t sink_pages, std::uint32_t layers = 16);

    // Requires first_token == frontier. Protects every touched page, invalidates its
    // old host copy, and returns fresh writeback epochs. Admission failure is atomic.
    [[nodiscard]] std::vector<KVViewWrite> begin_append(std::uint32_t first_token,
                                                        std::uint32_t count);
    // One notification covers every plane of this layer. Stale, duplicate, invalid,
    // or nonresident notifications return false. Only all layers can publish Both.
    [[nodiscard]] bool complete_writeback(std::uint32_t layer, std::uint32_t logical_page,
                                          KVViewEpoch epoch) noexcept;
    // Cannot grow the frontier. Preserves the valid prefix of a partial final page.
    // Cancels old-generation notifications and pending partial layer completions.
    void trim(std::uint32_t frontier);
    void reset();
    // Plan fresh sink/recent slots after all archive writes have completed.
    // Caller hydrates every resident plane before installation; captured slots are never reused.
    [[nodiscard]] KVViewSnapshot plan_restore(std::uint32_t frontier) const;
    void install_restore(const KVViewSnapshot& restored);

    [[nodiscard]] std::optional<KVViewPage> page(std::uint32_t logical_page) const noexcept;
    // Lists are in ascending original logical page order and share one exact boundary.
    [[nodiscard]] KVViewSnapshot snapshot() const;

private:
    struct Descriptor {
        std::int32_t physical_slot     = -1;
        KVViewState state              = KVViewState::HostOnly;
        std::uint64_t revision         = 0;
        std::uint32_t completed_layers = 0;
    };

    std::uint32_t max_context_;
    std::uint32_t sink_pages_;
    std::uint32_t layers_;
    std::uint32_t frontier_   = 0;
    std::uint64_t generation_ = 0;
    std::vector<Descriptor> pages_;
    std::vector<std::int32_t> slot_owners_;
    std::vector<std::uint8_t> layer_completion_;
};

} // namespace ninfer::kvmem

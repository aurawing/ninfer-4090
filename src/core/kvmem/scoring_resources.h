#pragma once
#include <cstddef>
#include <cstdint>
namespace ninfer::kvmem {
struct ScoringDomain {
    std::uint32_t pages{}, first{}, end{};
};
// Reference denominator mask: config R/64 bands, independent of hard recent coverage.
[[nodiscard]] ScoringDomain make_scoring_domain(std::uint32_t frontier,
                                                std::uint32_t selection_budget,
                                                std::uint32_t sink_pages,
                                                std::uint32_t recent_tokens, bool all_pages);
struct ScoringResources {
    std::size_t index_offset{}, index_bytes{}, query_offset{}, query_bytes{}, staging_bytes{};
    std::size_t logits_offset{}, logits_bytes{}, partial_bytes{};
    std::size_t stats_offset{}, stats_bytes{}, row_status_offset{}, row_status_bytes{},
        scores_offset{}, scores_bytes{}, status_offset{}, auxiliary_bytes{};
};
// Three disjoint caller-owned backings: idle prefill staging, partial O, other partials.
// Drain prefill borrowers before borrowing; ordered scoring and score/status D2H must
// complete before releasing the borrow and reusing it for hydration/prefill. No allocation.
[[nodiscard]] ScoringResources
plan_scoring_resources(std::uint32_t pages, std::uint32_t dimension, std::uint32_t query_heads,
                       std::uint32_t kv_heads, std::uint32_t tokens, std::size_t staging_capacity,
                       std::size_t partial_capacity, std::size_t auxiliary_capacity);
} // namespace ninfer::kvmem

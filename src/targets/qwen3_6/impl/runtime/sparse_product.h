#pragma once

#include "targets/qwen3_6/impl/runtime/tiered_context.h"
#include <cmath>
#include <iomanip>
#include <ostream>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail {

inline std::vector<kvmem::TokenSpan> sparse_spans(const PreparedPromptData& prompt,
                                                std::span<const TokenSpan> source) {
    std::vector<kvmem::TokenSpan> result;
    for (const auto& span : source) {
        if (!span.count || span.begin > prompt.token_ids.size() ||
            span.count > prompt.token_ids.size() - span.begin)
            throw std::invalid_argument("KVMem input span outside prepared prompt");
        result.push_back({static_cast<std::uint32_t>(span.begin),
                          static_cast<std::uint32_t>(span.begin + span.count)});
    }
    return result;
}

inline SparseTurnInput sparse_turn_input(const PreparedPromptData& prompt,
                                         kvmem::QueryProvenance query, double prefill_ms = 0) {
    SparseTurnInput input;
    input.bundle_identity = query.bundle_identity;
    input.query = std::move(query);
    input.current_input_spans = sparse_spans(prompt, prompt.input_spans.current);
    for (const auto& image : prompt.vision_items)
        input.image_groups.push_back({sparse_spans(prompt, image.token_spans)});
    input.prefill_ms = prefill_ms;
    return input;
}

inline void preflight_sparse_turn(const SparseTurnInput& input, const TieredRuntimePlan& plan,
                                  std::uint32_t frontier) {
    kvmem::SelectionInput selection;
    selection.frontier = frontier;
    selection.physical_pages = plan.view_pages;
    selection.future_reserve = plan.reserve_pages;
    selection.provisional_guards = plan.guard_pages;
    selection.sink_pages = plan.sink_pages;
    selection.recent_tokens = plan.recent_tokens;
    selection.binding_generation = selection.score_generation = 1;
    for (const auto ordinal : input.query.ordinals)
        selection.query_spans.push_back({ordinal, ordinal + 1});
    selection.current_input_spans = input.current_input_spans;
    selection.image_groups = input.image_groups;
    (void)kvmem::preflight_selection(selection);
}

// One complete JSON line. Owner-returned counts/domain/timings are authoritative;
// capture_wait_ms is a subset of prefill_ms, never an additional TTFT stage.
inline void log_sparse_turn(std::ostream& out, const SparseTurnMetrics& m,
                            const SparseTurnInput& input) {
    const auto& query = input.query;
    const double times[] = {m.prefill_ms, m.capture_wait_ms, m.scoring_h2d_ms,
        m.scoring_compute_ms, m.score_d2h_ms, m.selection_ms, m.hydrate_ms, m.publish_ms};
    for (const auto time : times)
        if (!std::isfinite(time) || time < 0) throw std::logic_error("nonfinite KVMem timing");
    const auto& s = m.selection;
    out << "[kvmem-turn] {\"frontier\":" << m.frontier
        << ",\"selection_generation\":" << s.generation
        << ",\"index_generation\":" << m.binding.index_generation
        << ",\"capture_id\":" << m.binding.capture_id
        << ",\"owner_epoch\":" << m.binding.epoch << ",\"owner_id\":" << m.binding.owner
        << ",\"query_tokens\":" << m.query_tokens
        << ",\"actual_view_tokens\":" << m.actual_view_tokens
        << ",\"selected_valid_tokens\":" << m.selected_valid_tokens
        << ",\"denominator\":\"" << (m.all_pages_denominator ? "all-committed" : "kept-band")
        << "\",\"denominator_pages\":" << m.denominator_pages
        << ",\"domain_first_page\":" << m.domain_first_page
        << ",\"domain_end_page\":" << m.domain_end_page
        << ",\"sink_pages\":" << s.sink_pages << ",\"recent_pages\":" << s.recent_pages
        << ",\"query_pages\":" << s.query_pages << ",\"image_pages\":" << s.image_pages
        << ",\"image_closure_pages\":" << s.image_closure_pages
        << ",\"hard_pages\":" << s.hard_pages
        << ",\"current_input_pages\":" << s.current_input_pages
        << ",\"current_input_softened_pages\":" << s.current_input_softened_pages
        << ",\"reserve_pages\":" << m.reserve_pages << ",\"guard_pages\":" << m.guard_pages
        << ",\"selected_pages\":" << s.selected.size() << ",\"retained_pages\":" << s.retained.size()
        << ",\"added_pages\":" << s.added.size() << ",\"removed_pages\":" << s.removed.size()
        << ",\"archive_mode\":\"" << (m.archive_mode == kvmem::HostArchiveMode::Pinned ? "pinned" : "pageable")
        << "\",\"hydrate_bytes\":" << m.hydrate_bytes << ",\"scoring_h2d_bytes\":" << m.scoring_h2d_bytes
        << std::setprecision(17) << ",\"prefill_ms\":" << m.prefill_ms
        << ",\"capture_wait_ms\":" << m.capture_wait_ms << ",\"scoring_h2d_ms\":" << m.scoring_h2d_ms
        << ",\"scoring_compute_ms\":" << m.scoring_compute_ms << ",\"score_d2h_ms\":" << m.score_d2h_ms
        << ",\"selection_ms\":" << m.selection_ms << ",\"hydrate_ms\":" << m.hydrate_ms
        << ",\"publish_ms\":" << m.publish_ms << ",\"capture_wait_is_prefill_subset\":true"
        << ",\"bundle_identity\":" << query.bundle_identity << ",\"lineage\":" << query.lineage
        << ",\"capture_epoch\":" << query.capture_epoch << ",\"query_covered_frontier\":" << query.covered_frontier
        << ",\"query_ordinals\":[";
    for (std::size_t i = 0; i < query.ordinals.size(); ++i) {
        if (i) out << ',';
        out << query.ordinals[i];
    }
    out << "],\"current_input_spans\":[";
    for (std::size_t i = 0; i < input.current_input_spans.size(); ++i) {
        if (i) out << ',';
        out << "{\"begin\":" << input.current_input_spans[i].first
            << ",\"end\":" << input.current_input_spans[i].end << '}';
    }
    out << "],\"source_prefix\":\"";
    constexpr char hex[] = "0123456789abcdef";
    for (const auto byte : query.source_prefix) out << hex[byte >> 4] << hex[byte & 15];
    out << "\"}\n";
}

} // namespace ninfer::targets::qwen3_6::detail

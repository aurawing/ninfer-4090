#pragma once
// Policy/diff adapted from https://github.com/kvmem/kvmem-qw3, Apache-2.0,
// commit 1cf3b2f83bfc071ada9c57491a7d121723051ac0, src/kvmem_store.cpp:
// pick_topk_blocks, pick_semantic_groups, set_selection. Modifications: pure plan,
// actual physical budget minus separate reserves, full recent token coverage,
// query/current hard policy and transitive atomic image page closure. No remapping.
#include <cstdint>
#include <span>
#include <vector>
namespace ninfer::kvmem {
struct TokenSpan {
    std::uint32_t first{}, end{};
};
struct ImageGroup {
    // One semantic image may have disjoint spans (e.g. multiple frames).
    std::vector<TokenSpan> spans;
};
struct SelectionInput {
    std::uint32_t frontier{}, physical_pages{}, future_reserve{}, provisional_guards{},
        sink_pages{}, recent_tokens{};
    std::uint64_t binding_generation{}, score_generation{};
    // Mandatory scoring capture metadata. Spans are committed ordinal [first,end).
    std::vector<TokenSpan> query_spans, current_input_spans;
    std::vector<ImageGroup> image_groups;
    std::span<const float> scores;
    std::vector<std::uint32_t> current_resident;
};
struct SelectionPlan {
    std::uint64_t generation{};
    std::uint32_t capacity{}, sink_pages{}, recent_pages{}, query_pages{}, image_closure_pages{},
        hard_pages{}, current_input_softened_pages{}, skipped_image_groups{}, current_input_pages{},
        image_pages{};
    std::vector<std::uint32_t> selected, retained, added, removed;
};
// Rejects stale/missing capture metadata and hard overflow before any owner mutation.
// Logical IDs remain original; sorted unique diff contains no compact positions/RoPE.
[[nodiscard]] SelectionPlan select_pages(const SelectionInput&);
} // namespace ninfer::kvmem

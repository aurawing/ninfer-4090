#include "core/kvmem/selection_plan.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
namespace ninfer::kvmem {
SelectionPlan select_pages(const SelectionInput& in) {
    if (!in.frontier || !in.binding_generation || in.binding_generation != in.score_generation ||
        in.query_spans.empty())
        throw std::invalid_argument("selection requires committed frontier, current binding "
                                    "generation and scoring query capture");
    auto n = std::uint32_t((std::uint64_t(in.frontier) + 63) / 64);
    if (in.scores.size() != n ||
        std::any_of(in.scores.begin(), in.scores.end(), [](float x) { return !std::isfinite(x); }))
        throw std::invalid_argument("selection scores missing or nonfinite");
    if (std::uint64_t(in.future_reserve) + in.provisional_guards > in.physical_pages)
        throw std::invalid_argument("selection reserves exceed physical pages");
    SelectionPlan out;
    out.generation = in.binding_generation;
    out.capacity = in.physical_pages - in.future_reserve - in.provisional_guards;
    std::vector<unsigned char> hard(n), current(n), image(n), old(n);
    std::vector<std::uint32_t> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    auto root = [&](std::uint32_t p) {
        while (parent[p] != p) {
            parent[p] = parent[parent[p]];
            p = parent[p];
        }
        return p;
    };
    auto span = [&](TokenSpan s, auto f) {
        if (s.first >= s.end || s.end > in.frontier)
            throw std::invalid_argument("selection span outside committed frontier");
        for (auto p = s.first / 64; p <= (s.end - 1) / 64; ++p)
            f(p);
    };
    auto count = [](const auto& v) { return std::uint32_t(std::count(v.begin(), v.end(), 1)); };
    for (const auto& group : in.image_groups) {
        if (group.spans.empty())
            throw std::invalid_argument("empty semantic image group");
        auto first = group.spans.front().first / 64;
        for (auto s : group.spans) {
            span(s, [&](auto p) {
                image[p] = 1;
                parent[root(p)] = root(first);
            });
        }
    }
    out.sink_pages = std::min(in.sink_pages, n);
    for (std::uint32_t p = 0; p < out.sink_pages; ++p)
        hard[p] = 1;
    if (in.recent_tokens) {
        std::vector<unsigned char> recent(n);
        span({in.frontier > in.recent_tokens ? in.frontier - in.recent_tokens : 0, in.frontier},
             [&](auto p) { hard[p] = recent[p] = 1; });
        out.recent_pages = count(recent);
    }
    std::vector<unsigned char> query(n);
    for (auto s : in.query_spans)
        span(s, [&](auto p) { hard[p] = query[p] = 1; });
    out.query_pages = count(query);
    for (auto s : in.current_input_spans)
        span(s, [&](auto p) { current[p] = 1; });
    out.current_input_pages = count(current);
    out.image_pages = count(image);
    auto closure = [&](auto& set) {
        std::vector<unsigned char> hit(n);
        for (std::uint32_t p = 0; p < n; ++p)
            if (image[p] && set[p])
                hit[root(p)] = 1;
        for (std::uint32_t p = 0; p < n; ++p)
            if (image[p] && hit[root(p)])
                set[p] = 1;
    };
    auto before = count(hard);
    closure(hard);
    out.image_closure_pages = count(hard) - before;
    auto all = hard;
    for (std::uint32_t p = 0; p < n; ++p)
        all[p] |= current[p];
    closure(all);
    if (count(all) <= out.capacity)
        hard = std::move(all);
    else
        for (std::uint32_t p = 0; p < n; ++p)
            out.current_input_softened_pages += current[p] && !hard[p];
    out.hard_pages = count(hard);
    if (out.hard_pages > out.capacity)
        throw std::runtime_error(
            "selection hard overflow: hard=" + std::to_string(out.hard_pages) + " sink=" +
            std::to_string(out.sink_pages) + " recent=" + std::to_string(out.recent_pages) +
            " query=" + std::to_string(out.query_pages) + " image_closure=" +
            std::to_string(out.image_closure_pages) + " G=" + std::to_string(in.future_reserve) +
            " H=" + std::to_string(in.provisional_guards) + " C=" + std::to_string(out.capacity) +
            " current=" + std::to_string(out.current_input_pages) +
            " image=" + std::to_string(out.image_pages));
    struct Candidate {
        std::vector<std::uint32_t> pages;
        float score{};
        std::uint32_t newest{};
        bool image{};
    };
    std::vector<Candidate> candidates;
    std::vector<std::size_t> group(n, n);
    for (std::uint32_t p = 0; p < n; ++p)
        if (!hard[p]) {
            auto r = root(p);
            if (image[p] && group[r] != n) {
                auto& c = candidates[group[r]];
                c.pages.push_back(p);
                c.score = std::max(c.score, in.scores[p]);
                c.newest = p;
            } else {
                if (image[p])
                    group[r] = candidates.size();
                candidates.push_back({{p}, in.scores[p], p, bool(image[p])});
            }
        }
    std::sort(candidates.begin(), candidates.end(), [](auto& a, auto& b) {
        return a.score != b.score ? a.score > b.score : a.newest > b.newest;
    });
    auto kept = out.hard_pages;
    for (auto& c : candidates) {
        if (c.pages.size() > out.capacity - kept) {
            out.skipped_image_groups += c.image;
            continue;
        }
        for (auto p : c.pages)
            hard[p] = 1;
        kept += std::uint32_t(c.pages.size());
    }
    for (auto p : in.current_resident) {
        if (p >= n || old[p])
            throw std::invalid_argument("selection current resident duplicate/out of range");
        old[p] = 1;
    }
    for (std::uint32_t p = 0; p < n; ++p) {
        if (hard[p]) {
            out.selected.push_back(p);
            (old[p] ? out.retained : out.added).push_back(p);
        } else if (old[p])
            out.removed.push_back(p);
    }
    return out;
}
} // namespace ninfer::kvmem

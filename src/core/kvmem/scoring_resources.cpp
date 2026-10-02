#include "core/kvmem/scoring_resources.h"
#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>
namespace ninfer::kvmem {
namespace {
std::size_t mul(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error("scoring scratch overflow");
    return a * b;
}
std::size_t add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("scoring scratch overflow");
    return a + b;
}
std::size_t align(std::size_t a) {
    return add(a, 255) & ~std::size_t(255);
}
} // namespace
ScoringDomain make_scoring_domain(std::uint32_t f, std::uint32_t budget, std::uint32_t sink,
                                  std::uint32_t r, bool all) {
    if (!f)
        throw std::invalid_argument("scoring capture has empty committed page domain");
    const auto n = std::uint32_t((std::uint64_t(f) + 63) / 64), recent = std::min(r / 64, n);
    sink = std::min(sink, n);
    if (!all && budget && n > budget && std::uint64_t(sink) + recent < n)
        return {n, sink, n - recent};
    return {n, 0, n};
}
ScoringResources plan_scoring_resources(std::uint32_t p, std::uint32_t d, std::uint32_t h,
                                        std::uint32_t k, std::uint32_t m, std::size_t stage,
                                        std::size_t part, std::size_t aux) {
    if (!p || !d || d > 1024 || !h || h > 64 || !k || h % k || !m || m > 16)
        throw std::invalid_argument("scoring scratch geometry");
    ScoringResources s;
    auto rows = mul(h, m);
    if (mul(p, rows) > INT_MAX)
        throw std::invalid_argument("scoring scratch exceeds tensor indexing domain");
    s.index_bytes = mul(mul(mul(p, d), k), 2);
    s.query_offset = align(s.index_bytes);
    s.query_bytes = mul(mul(d, rows), 2);
    s.staging_bytes = add(s.query_offset, s.query_bytes);
    s.logits_bytes = mul(mul(p, rows), 4);
    s.partial_bytes = s.logits_bytes;
    s.stats_bytes = mul(rows, 8);
    s.row_status_offset = align(s.stats_bytes);
    s.row_status_bytes = mul(rows, 4);
    s.scores_offset = align(add(s.row_status_offset, s.row_status_bytes));
    s.scores_bytes = mul(p, 4);
    s.status_offset = align(add(s.scores_offset, s.scores_bytes));
    s.auxiliary_bytes = add(s.status_offset, 4);
    if (s.staging_bytes > stage || s.partial_bytes > part || s.auxiliary_bytes > aux)
        throw std::invalid_argument("scoring borrow exceeds caller-owned scratch capacity");
    return s;
}
} // namespace ninfer::kvmem

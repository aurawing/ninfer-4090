#pragma once

#include "runtime/contract/types.h"
#include "core/kvmem/planning_error.h"

#include <cstddef>
#include <string>
#include <utility>

namespace ninfer::runtime {

[[nodiscard]] inline KvCapacityPolicy execution_kv_capacity_policy(KvMode mode,
                                                                  const KvCapacityPolicy& requested,
                                                                  bool shadow_validate) {
    if (mode != KvMode::TieredExact || shadow_validate) return requested;
    // An ignored dense explicit capacity carries no automatic reserve. Use
    // the normal automatic default rather than interpreting that zero as an
    // intentional reserve override. Preserve explicit automatic API tuning.
    return requested.mode == KvCapacityMode::Automatic ? requested : KvCapacityPolicy::automatic();
}

[[nodiscard]] KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve,
                                                       std::size_t available_runtime_bytes);

template <class Planner>
struct PrefillPlanSelection {
    std::uint32_t prefill_chunk;
    Planner planner;
    KvCapacityResolution capacity;
    std::string fallback_reason;
};

// Plan the complete candidate before testing its budget. The factory includes
// Main/MTP pools, workspace, staging, partial state and graph allowances.
template <class Factory>
[[nodiscard]] auto select_prefill_plan(const EngineOptions& requested, bool shadow_validate,
                                       std::size_t available_runtime_bytes, Factory&& make_planner) {
    using Planner = decltype(make_planner(requested));
    const bool normal_tiered = requested.kv_mode == KvMode::TieredExact && !shadow_validate;
    const auto policy = execution_kv_capacity_policy(requested.kv_mode, requested.kv_capacity,
                                                     shadow_validate);
    const auto build = [&](std::uint32_t chunk, bool automatic_trial) {
        auto options = requested;
        options.prefill_chunk = chunk;
        auto planner = make_planner(options);
        const auto& curve = planner.capacity_curve();
        // Validate the curve independently of feasibility. Only a known budget
        // shortage gets the retry marker; malformed curves retain their error.
        const auto minimum = curve.reservation_bytes(curve.minimum_main_page_groups);
        if (automatic_trial &&
            (available_runtime_bytes < policy.automatic_headroom_bytes ||
             minimum > available_runtime_bytes - policy.automatic_headroom_bytes))
            throw kvmem::TieredPrefillCapacityError(
                "2048 complete runtime minimum=" + std::to_string(minimum) +
                " automatic_headroom=" + std::to_string(policy.automatic_headroom_bytes) +
                " available_after_weights=" + std::to_string(available_runtime_bytes));
        const auto capacity = resolve_kv_capacity(policy, curve, available_runtime_bytes);
        return PrefillPlanSelection<Planner>{chunk, std::move(planner), capacity, {}};
    };
    if (!normal_tiered || requested.prefill_chunk_explicit || requested.prefill_chunk != 1024)
        return build(requested.prefill_chunk, false);
    std::string fallback_reason;
    try { return build(2048, true); }
    catch (const kvmem::TieredPrefillCapacityError& error) { fallback_reason = error.what(); }
    auto selected = build(1024, false);
    selected.fallback_reason = std::move(fallback_reason);
    return selected;
}

} // namespace ninfer::runtime

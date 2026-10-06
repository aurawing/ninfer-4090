#pragma once

#include <ninfer/targets/qwen3_6/runtime.h>
#include "targets/qwen3_6/impl/runtime/tiered_context.h"

namespace ninfer::targets::qwen3_6::detail {

// Repository-only access to the actual fixed-lane Program, with no runtime
// branch, configuration flag, or fault state added to production execution.
struct ProgramTestState {
    std::uint32_t execution_frontier{}, ledger_frontier{}, text_valid{}, mtp_valid{};
    bool owns_kv{}, ledger_empty{}, hidden_valid{}, retained{}, checkpoint_valid{},
        resume_valid{}, original_query{}, cleanup_failure{};
    bool gdn_zero{}, positions_zero{}, mtp_tags_empty{};
};
struct ProgramTestAccess {
    template<class Variant>
    static TieredContext& owner(Program<Variant>&);
    template<class Variant>
    static ProgramTestState inspect(Program<Variant>&);
    template<class Variant>
    static void check_compute_drain(Program<Variant>&, cudaError_t);
    template<class Variant>
    static void fail_reset_after_hardware_prepare(Program<Variant>&, cudaError_t);
};

} // namespace ninfer::targets::qwen3_6::detail

#pragma once

#include "runtime/contract/types.h"

#include <cstddef>

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

} // namespace ninfer::runtime

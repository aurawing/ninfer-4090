#include "runtime/engine/kv_capacity.h"

#include <iostream>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const auto explicit_policy = ninfer::KvCapacityPolicy::explicit_capacity(262144);
    const auto tiered_policy = ninfer::runtime::execution_kv_capacity_policy(
        ninfer::KvMode::TieredExact, explicit_policy, false);
    failures += check(tiered_policy.mode == ninfer::KvCapacityMode::Automatic &&
                          tiered_policy.automatic_headroom_bytes == ninfer::kDefaultKvCapacityHeadroomBytes,
                      "normal tiered must retain default automatic GPU headroom");
    for (const auto mode : {ninfer::KvMode::Dense, ninfer::KvMode::TieredExact}) {
        const auto policy = ninfer::runtime::execution_kv_capacity_policy(mode, explicit_policy, true);
        failures += check(policy.mode == explicit_policy.mode &&
                              policy.explicit_tokens == explicit_policy.explicit_tokens &&
                              policy.automatic_headroom_bytes == explicit_policy.automatic_headroom_bytes,
                          "dense/shadow explicit policy changed");
    }
    const auto dense_policy = ninfer::runtime::execution_kv_capacity_policy(
        ninfer::KvMode::Dense, explicit_policy, false);
    failures += check(dense_policy.mode == ninfer::KvCapacityMode::Explicit &&
                          dense_policy.explicit_tokens == 262144,
                      "normal dense physical policy changed");
    const auto zero_auto = ninfer::runtime::execution_kv_capacity_policy(
        ninfer::KvMode::TieredExact, ninfer::KvCapacityPolicy::automatic(0), false);
    failures += check(zero_auto.automatic_headroom_bytes == 0,
                      "explicit API automatic headroom override changed");
    const ninfer::runtime::SequenceCapacityCurve curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 6,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 128,
    };

    const auto automatic =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 1360);
    failures +=
        check(automatic.main_page_groups == 4 && automatic.resolved_tokens == 256 &&
                  automatic.runtime_reservation_bytes == 1256 &&
                  automatic.automatic_headroom_bytes == 50 && automatic.planned_slack_bytes == 104,
              "automatic KV capacity did not select the largest fitting page count");

    const auto capped =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 10000);
    failures += check(capped.main_page_groups == 6 && capped.resolved_tokens == 384,
                      "automatic KV capacity exceeded or missed the target maximum");

    const auto explicit_capacity = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), curve, 1200);
    failures +=
        check(explicit_capacity.main_page_groups == 3 && explicit_capacity.resolved_tokens == 192 &&
                  explicit_capacity.runtime_reservation_bytes == 1128,
              "explicit KV capacity did not use page-aligned token semantics");

    bool insufficient_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve,
                                                   1049);
    } catch (const std::invalid_argument&) { insufficient_rejected = true; }
    failures += check(insufficient_rejected,
                      "automatic KV capacity accepted less than the minimum reservation");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}

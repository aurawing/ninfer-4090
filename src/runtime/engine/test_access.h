#pragma once
#include "runtime/engine/concurrent_executor.h"

namespace ninfer::runtime {
// Repository-only access to serialize fault setup with the actual worker's
// GPU-unit lock. Production execution contains no additional callback or flag.
struct ConcurrentExecutorTestAccess {
    // Caller holds with_execution_lock. Run the actual executor admission unit
    // before installing an owner fault, since cold/restore preparation retires
    // all old transfer faults by design.
    template<class Instance>
    static void start_one_pending(ConcurrentExecutor<Instance>& executor) {
        if (executor.prefill_lane_ || executor.try_admit_one() !=
            ConcurrentExecutor<Instance>::AdmissionProgress::RanGpuUnit || !executor.prefill_lane_)
            throw std::logic_error("executor fault fixture requires a partial first prefill unit");
    }
    template<class Instance>
    static void advance_one_prefill(ConcurrentExecutor<Instance>& executor) {
        if (!executor.prefill_lane_) throw std::logic_error("executor fixture has no staged prefill");
        executor.run_prefill_step();
    }
    template<class Instance>
    static void finish_prefill(ConcurrentExecutor<Instance>& executor) {
        while (executor.prefill_lane_) executor.run_prefill_step();
        if (!executor.slots_[0] || !executor.slots_[0]->decode_ready)
            throw std::logic_error("executor fault fixture requires an active decode request");
    }
    template<class Instance>
    static void cancel_active(ConcurrentExecutor<Instance>& executor) {
        if (!executor.slots_[0]) throw std::logic_error("executor fault fixture has no active request");
        executor.slots_[0]->cancelled.store(true, std::memory_order_release);
    }
    template<class Instance, class Function>
    static decltype(auto) with_execution_lock(ConcurrentExecutor<Instance>& executor, Function&& function) {
        std::scoped_lock lock(executor.execution_mutex_);
        return std::forward<Function>(function)();
    }
};
}

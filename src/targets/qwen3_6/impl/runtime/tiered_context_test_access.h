#pragma once
#include "tiered_context.h"
namespace ninfer::targets::qwen3_6::detail {
struct TieredContextTestAccess {
    // The same production completion, detached from Main ownership for the
    // permanent-failure teardown regression. The external stream outlives it.
    static std::function<void()> score_query_drain(TieredContext&);
    static void finish_score_query_borrow(TieredContext&, kvmem::QueryCaptureBorrow&);
    static void poison_transfer(TieredContext&, std::exception_ptr);
    // Real synchronization first; supplied returned status models its failed
    // drain placement in the common cleanup collector without hurting CUDA.
    static void drain_borrowers(TieredContext&, cudaError_t, const char*);
    static void drain_after_quiesce(TieredContext&, cudaError_t, const char*, cudaError_t, const char*);
};
}

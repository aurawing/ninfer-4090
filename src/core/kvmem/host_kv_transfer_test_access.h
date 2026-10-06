#pragma once
#include "host_kv_transfer.h"
#include <exception>

namespace ninfer::kvmem {
// Repository-only classification seam: drain the actual owner before feeding
// returned API status into a simulated failed-drain placement. No GPU work,
// scheduling hook or application fault interface is added.
struct HostKVTransferTestAccess {
    static void poison(HostKVTransferEngine&, std::exception_ptr);
    static void record_drain(HostKVTransferEngine&, cudaError_t, const char*);
};
}

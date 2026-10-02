#pragma once

#include <stdexcept>

namespace ninfer::kvmem {

// Only chunk-dependent planning shortages are eligible for the automatic
// 2048 -> 1024 retry. Configuration, allocation and CUDA errors propagate.
class TieredPrefillCapacityError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

} // namespace ninfer::kvmem

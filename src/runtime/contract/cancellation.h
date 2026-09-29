#pragma once

#include <stdexcept>

namespace ninfer::runtime {

class RequestCancelled : public std::runtime_error {
public:
    RequestCancelled() : std::runtime_error("inference request cancelled") {}
};

} // namespace ninfer::runtime

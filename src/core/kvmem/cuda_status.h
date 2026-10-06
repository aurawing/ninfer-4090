#pragma once

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <exception>
#include <string>

namespace ninfer::kvmem {

// Only known API rejections are eligible for recovery, after every borrowed
// resource drains successfully. Unknown/execution/context errors fail closed.
[[nodiscard]] inline bool cuda_failure_is_fatal(cudaError_t status) noexcept {
    switch (status) {
    case cudaSuccess:
    case cudaErrorInvalidValue:
    case cudaErrorMemoryAllocation:
    case cudaErrorInvalidDevice:
    case cudaErrorInvalidResourceHandle:
    case cudaErrorInvalidMemcpyDirection:
    case cudaErrorNotSupported:
    case cudaErrorLaunchOutOfResources:
        return false;
    case cudaErrorIllegalAddress:
    case cudaErrorAssert:
    case cudaErrorLaunchTimeout:
    case cudaErrorLaunchFailure:
    case cudaErrorContextIsDestroyed:
    default:
        return true;
    }
}

[[nodiscard]] inline bool cuda_transfer_failure_is_unrecoverable(
    cudaError_t status, bool failed_drain) noexcept {
    return failed_drain || cuda_failure_is_fatal(status);
}

class CudaTransferError : public std::runtime_error {
public:
    CudaTransferError(cudaError_t status, const char* operation, bool drain_failed = false)
        : std::runtime_error(std::string(operation) + ": " + cudaGetErrorName(status) +
                             ": " + cudaGetErrorString(status)),
          status_(status), operation_(operation), drain_failed_(drain_failed) {}
    [[nodiscard]] cudaError_t status() const noexcept { return status_; }
    [[nodiscard]] const char* operation() const noexcept { return operation_; }
    [[nodiscard]] bool drain_failed() const noexcept { return drain_failed_; }
private:
    cudaError_t status_;
    const char* operation_; // All callers supply static operation labels.
    bool drain_failed_;
};

inline void check_transfer_cuda(cudaError_t status, const char* operation, bool drain_failed = false) {
    if (status != cudaSuccess) throw CudaTransferError(status, operation, drain_failed);
}

[[nodiscard]] inline bool exception_is_typed_unrecoverable(const std::exception_ptr& cause) noexcept {
    if (!cause) return false;
    try { std::rethrow_exception(cause); }
    catch (const CudaTransferError& error) {
        return cuda_transfer_failure_is_unrecoverable(error.status(), error.drain_failed());
    } catch (...) { return false; }
}

// Preserve the first unrecoverable typed cause, including its static operation
// and failed-drain flag. Earlier generic/known API errors remain fail-closed,
// but cannot hide a later error that proves CUDA/resource lifetime is unsafe.
inline void prefer_unrecoverable_exception(std::exception_ptr& first,
                                           const std::exception_ptr& candidate) noexcept {
    if (!first || (!exception_is_typed_unrecoverable(first) && exception_is_typed_unrecoverable(candidate)))
        first = candidate;
}

} // namespace ninfer::kvmem

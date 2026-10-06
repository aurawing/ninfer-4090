#pragma once
#include "core/kvmem/cuda_status.h"
#include <cstddef>
namespace ninfer::kvmem {
// Local KVMem loading allocations retain CUDA status for admission/fatal classification.
// Legacy arena/device allocation behavior remains unchanged.
class CheckedPinnedHostBuffer {
public:
    explicit CheckedPinnedHostBuffer(std::size_t bytes) : bytes_(bytes) {
        check_transfer_cuda(cudaMallocHost(&data_, bytes), "KVMem cacheable pinned allocation");
    }
    ~CheckedPinnedHostBuffer() { if (data_) (void)cudaFreeHost(data_); }
    CheckedPinnedHostBuffer(const CheckedPinnedHostBuffer&) = delete;
    CheckedPinnedHostBuffer& operator=(const CheckedPinnedHostBuffer&) = delete;
    void* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return bytes_; }
private:
    void* data_{};
    std::size_t bytes_{};
};
class CheckedDeviceBuffer {
public:
    explicit CheckedDeviceBuffer(std::size_t count) : bytes(count) {
        check_transfer_cuda(cudaMalloc(&p, count), "KVMem capture device allocation");
    }
    ~CheckedDeviceBuffer() { if (p) (void)cudaFree(p); }
    CheckedDeviceBuffer(const CheckedDeviceBuffer&) = delete;
    CheckedDeviceBuffer& operator=(const CheckedDeviceBuffer&) = delete;
    void* p{};
    std::size_t bytes{};
};
} // namespace ninfer::kvmem

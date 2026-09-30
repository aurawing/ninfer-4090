#include "cpu_vision_encoder.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6 {
bool cpu_vision_backend_available() noexcept { return false; }

std::shared_ptr<CpuVisionEncoder> make_gguf_cpu_vision_encoder(const std::string&,
    std::uint32_t, std::size_t) {
    throw std::runtime_error("CPU vision backend was not built; configure NINFER_BUILD_CPU_VISION=ON");
}

std::shared_ptr<CpuVisionEncoder> make_gguf_cuda_vision_encoder(const std::string&,
    std::size_t) {
    throw std::runtime_error("GGML CUDA vision backend was not built");
}

std::shared_ptr<CpuVisionEncoder> make_embedded_cpu_vision_encoder(
    std::span<const EmbeddedVisionTensor>, std::uint32_t, std::size_t, std::uint64_t) {
    throw std::runtime_error("CPU vision backend was not built; configure NINFER_BUILD_CPU_VISION=ON");
}
} // namespace ninfer::targets::qwen3_6

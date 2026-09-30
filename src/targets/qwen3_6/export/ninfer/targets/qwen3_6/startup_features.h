#pragma once

#include "ninfer/types.h"
#include <stdexcept>

namespace ninfer::targets::qwen3_6 {

struct StartupFeatures {
    bool vision                    = false;
    std::uint32_t vision_max_tokens = 8192;
    VisionDevice vision_device      = VisionDevice::Cuda;
    std::filesystem::path vision_mmproj_path;
    std::uint32_t vision_cpu_threads = 6;
    std::uint32_t vision_cpu_memory_mib = 4096;
    std::uint32_t vision_cpu_cache_mib = 128;
    SpeculativeBackend speculative = SpeculativeBackend::None;
    ProposalHead proposal_head     = ProposalHead::Full;

    bool operator==(const StartupFeatures&) const = default;

    [[nodiscard]] bool cuda_vision() const noexcept {
        return vision && vision_device == VisionDevice::Cuda && vision_mmproj_path.empty();
    }
    [[nodiscard]] bool cpu_vision() const noexcept {
        return vision && vision_device == VisionDevice::Cpu;
    }
    [[nodiscard]] bool ggml_cuda_vision() const noexcept {
        return vision && vision_device == VisionDevice::Cuda && !vision_mmproj_path.empty();
    }
    [[nodiscard]] bool ggml_vision() const noexcept {
        return cpu_vision() || ggml_cuda_vision();
    }

    [[nodiscard]] bool speculative_enabled() const noexcept {
        return speculative != SpeculativeBackend::None;
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool optimized_proposal() const noexcept {
        return speculative_enabled() && proposal_head == ProposalHead::Optimized;
    }
};

[[nodiscard]] inline StartupFeatures startup_features(const EngineOptions& options) {
    switch (options.vision_device) {
    case VisionDevice::Cuda:
        if (options.vision_cpu_threads != 6 || options.vision_cpu_memory_mib != 4096 ||
            options.vision_cpu_cache_mib != 128) {
            throw std::invalid_argument("CPU vision tuning requires vision_device=cpu");
        }
        break;
    case VisionDevice::Cpu:
        if (!options.enable_vision) {
            throw std::invalid_argument("CPU vision requires enable_vision");
        }
        if (options.vision_cpu_threads == 0 ||
            options.vision_cpu_threads > kMaximumVisionCpuThreads || options.vision_cpu_memory_mib == 0) {
            throw std::invalid_argument("CPU vision requires threads in [1,512] and a positive memory budget");
        }
        if (options.vision_cpu_cache_mib >= options.vision_cpu_memory_mib) {
            throw std::invalid_argument("CPU vision cache must be smaller than the total host-memory budget");
        }
        break;
    default:
        throw std::invalid_argument("unknown vision device");
    }
    return StartupFeatures{
        .vision            = options.enable_vision,
        .vision_max_tokens = options.vision_max_tokens > 0 ? options.vision_max_tokens : 8192,
        .vision_device = options.vision_device,
        .vision_mmproj_path = options.vision_mmproj_path,
        .vision_cpu_threads = options.vision_cpu_threads,
        .vision_cpu_memory_mib = options.vision_cpu_memory_mib,
        .vision_cpu_cache_mib = options.vision_cpu_cache_mib,
        .speculative       = options.speculative.backend,
        .proposal_head     = options.speculative.proposal_head,
    };
}

} // namespace ninfer::targets::qwen3_6

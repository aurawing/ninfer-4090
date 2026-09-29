#include <ninfer/targets/qwen3_6/startup_features.h>

#include <iostream>
#include <stdexcept>

using namespace ninfer;
using namespace ninfer::targets::qwen3_6;

int main() {
    try {
        EngineOptions options;
        auto gpu = startup_features(options);
        if (gpu.vision || gpu.cuda_vision() || gpu.cpu_vision()) {
            throw std::runtime_error("vision must remain disabled by default");
        }
        options.enable_vision = true;
        gpu = startup_features(options);
        if (!gpu.cuda_vision() || gpu.cpu_vision()) {
            throw std::runtime_error("existing vision flag must select CUDA");
        }
        options.vision_device = VisionDevice::Cpu;
        options.vision_mmproj_path = "models/mmproj-BF16.gguf";
        options.vision_cpu_threads = 4;
        options.vision_cpu_memory_mib = 2048;
        const auto cpu = startup_features(options);
        if (cpu.vision_cpu_cache_mib != 128) { throw std::runtime_error("cache default not frozen"); }
        if (!cpu.cpu_vision() || cpu.cuda_vision() || cpu.vision_mmproj_path != options.vision_mmproj_path ||
            cpu.vision_cpu_threads != 4 || cpu.vision_cpu_memory_mib != 2048 || cpu == gpu) {
            throw std::runtime_error("CPU settings must be frozen with startup features");
        }
        auto changed = cpu;
        changed.vision_mmproj_path = "different.gguf";
        if (changed == cpu) { throw std::runtime_error("weight source must participate in equality"); }
        const auto rejects = [](const EngineOptions& invalid) {
            try { (void)startup_features(invalid); }
            catch (const std::invalid_argument&) { return; }
            throw std::runtime_error("invalid Engine CPU settings accepted");
        };
        auto invalid = options;
        invalid.vision_cpu_cache_mib = invalid.vision_cpu_memory_mib;
        rejects(invalid);
        invalid = options;
        invalid.vision_cpu_cache_mib = invalid.vision_cpu_memory_mib + 1;
        rejects(invalid);
        auto uncached = options;
        uncached.vision_cpu_cache_mib = 0;
        if (startup_features(uncached).vision_cpu_cache_mib != 0 || startup_features(uncached) == cpu) {
            throw std::runtime_error("disabled cache must be frozen and participate in equality");
        }
        auto gpu_cache = EngineOptions{};
        gpu_cache.vision_cpu_cache_mib = 0;
        rejects(gpu_cache);
        invalid = options;
        invalid.vision_mmproj_path.clear();
        rejects(invalid);
        invalid = options;
        invalid.vision_cpu_threads = kMaximumVisionCpuThreads + 1;
        rejects(invalid);
        invalid = options;
        invalid.vision_cpu_memory_mib = 0;
        rejects(invalid);
        invalid = options;
        invalid.enable_vision = false;
        rejects(invalid);
        invalid = options;
        invalid.vision_device = VisionDevice::Cuda;
        rejects(invalid);
        invalid.vision_mmproj_path.clear();
        rejects(invalid);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

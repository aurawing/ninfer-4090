#include "targets/qwen3_6/impl/vision/cpu_vision_encoder.h"

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

int main(int argc, char** argv) {
    using namespace ninfer::targets::qwen3_6;
    try {
        if (argc != 5) {
            throw std::invalid_argument("Usage: bench_cpu_vision MMPROJ THREADS HEIGHT WIDTH (pixels, multiples of 32)");
        }
        const int threads = std::stoi(argv[2]), height = std::stoi(argv[3]), width = std::stoi(argv[4]);
        if (height <= 0 || width <= 0 || height % 32 || width % 32 || height > 1024 || width > 1024) {
            throw std::invalid_argument("Unsupported benchmark geometry");
        }
        const auto encoder = make_gguf_cpu_vision_encoder(argv[1], threads, std::size_t(4) << 30);
        std::vector<float> patches(std::size_t(height / 16) * (width / 16) * 1536);
        for (std::size_t i = 0; i < patches.size(); ++i) { patches[i] = float(std::sin(double(i) * 0.017)); }
        // Still-image temporal planes represent the same frame.
        for (std::size_t p = 0; p < patches.size() / 1536; ++p) {
            for (int c = 0; c < 3; ++c) {
                for (int i = 0; i < 256; ++i) {
                    patches[p * 1536 + (c * 2 + 1) * 256 + i] = patches[p * 1536 + c * 512 + i];
                }
            }
        }
        const CpuVisionInput input{patches, 1, height / 16, width / 16, false};
        encoder->validate(input);
        for (int run = 0; run < 2; ++run) {
            const auto start = std::chrono::steady_clock::now();
            const auto output = encoder->encode(input);
            const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            for (auto value : output) {
                if (!std::isfinite(std::bit_cast<float>(std::uint32_t(value) << 16))) {
                    throw std::runtime_error("Nonfinite benchmark embedding");
                }
            }
            std::size_t rss = 0, private_bytes = 0;
#ifdef _WIN32
            PROCESS_MEMORY_COUNTERS_EX memory{};
            memory.cb = sizeof(memory);
            if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory), sizeof(memory))) {
                throw std::runtime_error("Cannot read benchmark process memory");
            }
            rss = memory.WorkingSetSize;
            private_bytes = memory.PrivateUsage;
#endif
            std::cout << "MEASURE," << threads << ',' << height << ',' << width << ',' << run << ','
                      << output.size() / 5120 << ',' << ms << ',' << rss << ',' << private_bytes << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_6/impl/vision/cpu_vision_encoder.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/vision/embedded_vision.h"

#include <ninfer/targets/qwen3_6_27b/package.h>

#include <bit>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        const char* path = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
        if (!path || !*path || !std::filesystem::is_regular_file(path)) {
            std::cout << "SKIP: NINFER_QWEN3_8_27B_WEIGHTS is not set\n";
            return 77;
        }
        namespace artifact = ninfer::artifact;
        namespace model = ninfer::targets::qwen3_6_27b;
        artifact::Reader reader(path);
        {
            artifact::Binder binder(reader);
            const auto vision = model::detail::bind_optional_vision(
                binder, artifact::TensorPlacement::ValidateOnly);
            if (!vision) { throw std::runtime_error("real .ninfer is missing embedded Vision weights"); }
            auto encoder = model::detail::make_embedded_vision_encoder(
                binder, *vision, 2, std::size_t(4) << 30);
            if (encoder->weight_bytes() < 931126208) {
                throw std::runtime_error("embedded Vision weights were not materialized on CPU");
            }
            constexpr int height = 2, width = 4;
            std::vector<float> patches(height * width * 1536);
            for (int patch = 0; patch < height * width; ++patch) {
                for (int channel = 0; channel < 3; ++channel) {
                    for (int pixel = 0; pixel < 256; ++pixel) {
                        const float value = static_cast<float>(
                            std::sin(double(patch * 768 + channel * 256 + pixel) * 0.005));
                        for (int time = 0; time < 2; ++time) {
                            patches[patch * 1536 + (channel * 2 + time) * 256 + pixel] = value;
                        }
                    }
                }
            }
            const ninfer::targets::qwen3_6::CpuVisionInput input{patches, 1, height, width, false};
            encoder->validate(input);
            const auto output = encoder->encode(input);
            if (output.size() != height * width / 4 * 5120 ||
                output != encoder->encode(input)) {
                throw std::runtime_error("embedded CPU Vision output shape or repeatability differs");
            }
            for (const auto value : output) {
                if ((value & 0x7f80U) == 0x7f80U) {
                    throw std::runtime_error("embedded CPU Vision produced nonfinite BF16");
                }
            }
            if (const char* gguf = std::getenv("NINFER_TEST_VISION_GGUF"); gguf && *gguf) {
                auto reference = ninfer::targets::qwen3_6::make_gguf_cpu_vision_encoder(
                    gguf, 2, std::size_t(4) << 30);
                const auto reference_output = reference->encode(input);
                if (reference_output.size() != output.size()) {
                    throw std::runtime_error("embedded and external Vision output shapes differ");
                }
                double squared_error = 0, squared_reference = 0, dot = 0, squared_output = 0;
                for (std::size_t i = 0; i < output.size(); ++i) {
                    const double a = std::bit_cast<float>(std::uint32_t(output[i]) << 16U);
                    const double b = std::bit_cast<float>(std::uint32_t(reference_output[i]) << 16U);
                    squared_error += (a - b) * (a - b);
                    squared_reference += b * b;
                    squared_output += a * a;
                    dot += a * b;
                }
                const double relative_l2 = std::sqrt(squared_error / squared_reference);
                const double cosine = dot / std::sqrt(squared_output * squared_reference);
                std::cout << "Embedded versus BF16 GGUF: relative L2=" << relative_l2
                          << " cosine=" << cosine << '\n';
                if (!std::isfinite(relative_l2) || !std::isfinite(cosine) ||
                    cosine < 0.95 || relative_l2 > 0.40) {
                    throw std::runtime_error("embedded/external Vision embedding agreement differs");
                }
            }
        }
        artifact::Binder startup_binder(reader);
        ninfer::EngineOptions options;
        options.enable_vision = true;
        options.vision_device = ninfer::VisionDevice::Cpu;
        (void)model::Package::plan_load(startup_binder, options,
                                        model::Package::resolve_weights(reader.identity()));
        std::cout << "Embedded .ninfer CPU Vision loaded and encoded without external GGUF\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

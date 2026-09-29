#include "cpu_vision_bridge.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6 {
namespace {

std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::invalid_argument("CPU vision input size overflow");
    }
    return a * b;
}

} // namespace

CpuVisionShape validate_cpu_vision_input(const CpuVisionInput& input) {
    if (input.temporal <= 0 || input.height <= 0 || input.width <= 0 ||
        input.height % 2 || input.width % 2 || (!input.video && input.temporal != 1)) {
        throw std::invalid_argument("CPU vision requires positive temporal groups and even spatial grids");
    }
    // clip's geometry and CHW uploader use signed int products. Keep those products safe too.
    const auto width = multiply(static_cast<std::size_t>(input.width), 16);
    const auto height = multiply(static_cast<std::size_t>(input.height), 16);
    const auto pixels = multiply(width, height);
    if (width > 46000 || height > 46000 || multiply(pixels, input.video ? 6 : 3) >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("CPU vision frame exceeds backend geometry limits");
    }
    const auto tokens = multiply(multiply(static_cast<std::size_t>(input.temporal),
        static_cast<std::size_t>(input.height)), static_cast<std::size_t>(input.width));
    const auto elements = multiply(tokens, 1536);
    if (tokens > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        input.patches.size() != elements) {
        throw std::invalid_argument("CPU vision patch count disagrees with the grid");
    }
    for (std::size_t patch = 0; patch < tokens; ++patch) {
        for (std::size_t c = 0; c < 3; ++c) {
            for (std::size_t t = 0; t < 2; ++t) {
                for (std::size_t p = 0; p < 256; ++p) {
                    const float value = input.patches[patch * 1536 + (c * 2 + t) * 256 + p];
                    if (!std::isfinite(value)) {
                        throw std::invalid_argument("CPU vision patch contains a nonfinite value");
                    }
                    if (!input.video && t && value != input.patches[patch * 1536 + c * 512 + p]) {
                        throw std::invalid_argument("CPU vision still-image temporal replicas disagree");
                    }
                }
            }
        }
    }
    return {tokens, tokens / 4, elements, static_cast<std::int32_t>(width),
            static_cast<std::int32_t>(height), input.video ? 2U : 1U};
}

std::vector<CpuVisionRgbFrame> unpack_cpu_vision_group(const CpuVisionInput& input,
                                                      std::uint32_t group) {
    const auto shape = validate_cpu_vision_input(input);
    if (group >= static_cast<std::uint32_t>(input.temporal)) {
        throw std::invalid_argument("CPU vision temporal group is out of range");
    }
    const auto frame_elements = multiply(multiply(static_cast<std::size_t>(shape.frame_width),
        static_cast<std::size_t>(shape.frame_height)), 3);
    std::vector<CpuVisionRgbFrame> frames;
    frames.reserve(shape.frames_per_group);
    for (std::uint32_t t = 0; t < shape.frames_per_group; ++t) {
        frames.push_back({shape.frame_width, shape.frame_height, std::vector<float>(frame_elements)});
    }
    const auto group_patches = multiply(static_cast<std::size_t>(input.height), input.width);
    std::size_t patch = multiply(group, group_patches);
    for (std::int32_t by = 0; by < input.height; by += 2) {
        for (std::int32_t bx = 0; bx < input.width; bx += 2) {
            for (std::int32_t dy = 0; dy < 2; ++dy) {
                for (std::int32_t dx = 0; dx < 2; ++dx, ++patch) {
                    for (std::size_t c = 0; c < 3; ++c) {
                        for (std::size_t t = 0; t < shape.frames_per_group; ++t) {
                            for (std::size_t y = 0; y < 16; ++y) {
                                for (std::size_t x = 0; x < 16; ++x) {
                                    const auto rgb_index = ((static_cast<std::size_t>(by + dy) * 16 + y) *
                                        shape.frame_width + static_cast<std::size_t>(bx + dx) * 16 + x) * 3 + c;
                                    frames[t].rgb[rgb_index] = input.patches[patch * 1536 +
                                        (c * 2 + t) * 256 + y * 16 + x];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return frames;
}

} // namespace ninfer::targets::qwen3_6

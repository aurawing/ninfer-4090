#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6 {

// Already resized and normalized frontend data; temporal/height/width are patch grids.
struct CpuVisionInput {
    std::span<const float> patches;
    std::int32_t temporal;
    std::int32_t height;
    std::int32_t width;
    bool video;
    // Hash of the exact processed patch bytes, supplied by the runtime when caching is enabled.
    std::optional<std::array<std::uint8_t, 32>> processed_fingerprint;
};

struct CpuVisionShape {
    std::size_t patch_tokens;
    std::size_t merged_tokens;
    std::size_t patch_elements;
    std::int32_t frame_width;
    std::int32_t frame_height;
    std::uint32_t frames_per_group;
};

struct CpuVisionRgbFrame {
    std::int32_t width;
    std::int32_t height;
    std::vector<float> rgb;
};

// Geometry-only check for callers that already rely on an encoder to validate patch values.
CpuVisionShape cpu_vision_shape(const CpuVisionInput& input);
CpuVisionShape validate_cpu_vision_input(const CpuVisionInput& input);
std::vector<CpuVisionRgbFrame> unpack_cpu_vision_group(const CpuVisionInput& input,
                                                      std::uint32_t group);
// Call only with a shape returned by validate_cpu_vision_input for this input.
std::vector<CpuVisionRgbFrame> unpack_validated_cpu_vision_group(
    const CpuVisionInput& input, const CpuVisionShape& shape, std::uint32_t group);

} // namespace ninfer::targets::qwen3_6

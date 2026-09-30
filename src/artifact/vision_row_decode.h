#pragma once

#include "artifact/reader.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::artifact {

// Decode one logical [K] row from the row-split payload. The destination has
// exactly K elements; padded columns are never exposed to GGML.
void decode_vision_row_f32(std::span<const std::byte> payload, NumericFormat format,
                           std::uint64_t rows, std::uint64_t columns, std::uint64_t row,
                           std::span<float> destination);

// IEEE round-to-nearest-even at the BF16 boundary. Nonfinite values are rejected.
std::uint16_t vision_f32_to_bf16(float value);

} // namespace ninfer::artifact

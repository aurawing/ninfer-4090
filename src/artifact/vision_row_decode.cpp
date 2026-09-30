#include "artifact/vision_row_decode.h"

#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::artifact {
namespace {

unsigned byte_at(std::span<const std::byte> bytes, std::uint64_t offset) {
    return std::to_integer<unsigned>(bytes[static_cast<std::size_t>(offset)]);
}

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
    const std::uint32_t exponent = (bits >> 10U) & 0x1fU;
    std::uint32_t mantissa = bits & 0x03ffU;
    if (exponent == 0x1fU) {
        return std::bit_cast<float>(sign | 0x7f800000U | (mantissa << 13U));
    }
    if (exponent == 0) {
        if (mantissa == 0) { return std::bit_cast<float>(sign); }
        int shift = 0;
        while ((mantissa & 0x0400U) == 0) { mantissa <<= 1U; ++shift; }
        mantissa &= 0x03ffU;
        return std::bit_cast<float>(sign | (static_cast<std::uint32_t>(113 - shift) << 23U) |
                                    (mantissa << 13U));
    }
    return std::bit_cast<float>(sign | ((exponent + 112U) << 23U) | (mantissa << 13U));
}

int signed_code(std::span<const std::byte> bytes, const RowSplitGeometry& g,
                NumericFormat format, std::uint64_t group, std::uint64_t column) {
    if (format == NumericFormat::W8G32_F16S) {
        const auto unsigned_code = byte_at(bytes, group * g.low_bytes_per_group + column);
        return unsigned_code < 128U ? static_cast<int>(unsigned_code)
                                    : static_cast<int>(unsigned_code) - 256;
    }
    unsigned code = (byte_at(bytes, group * g.low_bytes_per_group + column / 2U) >>
                     ((column & 1U) * 4U)) & 0x0fU;
    unsigned bits = 4;
    if (format == NumericFormat::Q5G64_F16S) {
        code |= ((byte_at(bytes, g.high_plane_offset + group * g.high_bytes_per_group +
                               column / 8U) >> (column % 8U)) & 1U) << 4U;
        bits = 5;
    } else if (format == NumericFormat::Q6G64_F16S) {
        code |= ((byte_at(bytes, g.high_plane_offset + group * g.high_bytes_per_group +
                               column / 4U) >> ((column % 4U) * 2U)) & 3U) << 4U;
        bits = 6;
    }
    const unsigned sign = 1U << (bits - 1U);
    return (code & sign) ? static_cast<int>(code) - static_cast<int>(1U << bits)
                         : static_cast<int>(code);
}

} // namespace

void decode_vision_row_f32(std::span<const std::byte> payload, NumericFormat format,
                           std::uint64_t rows, std::uint64_t columns, std::uint64_t row,
                           std::span<float> destination) {
    const std::array<std::uint64_t, 2> shape{rows, columns};
    const auto g = row_split_geometry(format, shape);
    if (row >= rows || columns > std::numeric_limits<std::size_t>::max() ||
        destination.size() != columns || payload.size() != g.encoded_bytes) {
        throw std::invalid_argument("invalid vision row-split payload, row, or destination");
    }
    for (std::uint64_t column = 0; column < columns; ++column) {
        const auto group = row * g.groups_per_row + column / g.group_size;
        const auto within = column % g.group_size;
        const std::uint16_t scale_bits = static_cast<std::uint16_t>(
            byte_at(payload, g.scale_plane_offset + group * 2U) |
            (byte_at(payload, g.scale_plane_offset + group * 2U + 1U) << 8U));
        const float scale = half_to_float(scale_bits);
        if (!std::isfinite(scale)) { throw std::invalid_argument("vision quantization scale is nonfinite"); }
        destination[static_cast<std::size_t>(column)] =
            static_cast<float>(signed_code(payload, g, format, group, within)) * scale;
    }
}

std::uint16_t vision_f32_to_bf16(float value) {
    if (!std::isfinite(value)) { throw std::invalid_argument("vision BF16 source is nonfinite"); }
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const auto rounded = static_cast<std::uint16_t>(
        (bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
    if ((rounded & 0x7f80U) == 0x7f80U) {
        throw std::invalid_argument("vision BF16 source overflows after rounding");
    }
    return rounded;
}

} // namespace ninfer::artifact

#include "artifact/vision_row_decode.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using ninfer::artifact::NumericFormat;
using ninfer::artifact::decode_vision_row_f32;
using ninfer::artifact::row_split_geometry;
using ninfer::artifact::vision_f32_to_bf16;

namespace {

void check(bool ok, const char* message) {
    if (!ok) { throw std::runtime_error(message); }
}

std::vector<std::byte> payload_for(NumericFormat format, std::size_t rows, std::size_t columns) {
    const std::array<std::uint64_t, 2> shape{rows, columns};
    const auto g = row_split_geometry(format, shape);
    std::vector<std::byte> data(g.encoded_bytes);
    // Both rows use 1.5 in the first group and 0.25 in the second. This
    // supplies physical FP16 scale bytes independent of the decoder.
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t first = g.scale_plane_offset + (row * g.groups_per_row) * 2;
        data.at(first) = std::byte{0x00};
        data.at(first + 1) = std::byte{0x3e};
        data.at(first + 2) = std::byte{0x00};
        data.at(first + 3) = std::byte{0x34};
    }
    return data;
}

void check_prefix(NumericFormat format, const std::vector<unsigned>& low,
                  const std::vector<unsigned>& high, const std::vector<int>& expected,
                  std::size_t columns) {
    const std::array<std::uint64_t, 2> shape{2, columns};
    const auto g = row_split_geometry(format, shape);
    auto data = payload_for(format, 2, columns);
    const std::size_t row_low = g.groups_per_row * g.low_bytes_per_group;
    const std::size_t row_high = g.groups_per_row * g.high_bytes_per_group;
    for (std::size_t i = 0; i < low.size(); ++i) {
        data.at(row_low + i) = std::byte(low[i]);
    }
    for (std::size_t i = 0; i < high.size(); ++i) {
        data.at(g.high_plane_offset + row_high + i) = std::byte(high[i]);
    }
    std::vector<float> out(columns, -999.0F);
    decode_vision_row_f32(data, format, 2, columns, 1, out);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        check(out[i] == static_cast<float>(expected[i]) * 1.5F, "signed code or scale differs");
    }
    for (std::size_t i = expected.size(); i < columns; ++i) {
        check(out[i] == 0.0F, "zero/padded logical column differs");
    }
    std::vector<float> first(columns, -999.0F);
    decode_vision_row_f32(data, format, 2, columns, 0, first);
    check(first[0] == 0.0F, "row indexing differs");
    bool rejected = false;
    try { decode_vision_row_f32(data, format, 2, columns, 2, out); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "out-of-range row was accepted");
    data.pop_back();
    rejected = false;
    try { decode_vision_row_f32(data, format, 2, columns, 1, out); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "truncated payload was accepted");
}

void scale_edges() {
    const std::array<std::uint64_t, 2> shape{1, 33};
    const auto g = row_split_geometry(NumericFormat::W8G32_F16S, shape);
    std::vector<std::byte> data(g.encoded_bytes);
    data[0] = std::byte{1};
    data[g.scale_plane_offset] = std::byte{1}; // smallest positive FP16 subnormal
    std::vector<float> out(33);
    decode_vision_row_f32(data, NumericFormat::W8G32_F16S, 1, 33, 0, out);
    check(out[0] == std::ldexp(1.0F, -24), "FP16 subnormal scale differs");
    data[g.scale_plane_offset] = std::byte{0};
    data[g.scale_plane_offset + 1] = std::byte{0x7c}; // +infinity
    bool rejected = false;
    try { decode_vision_row_f32(data, NumericFormat::W8G32_F16S, 1, 33, 0, out); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "nonfinite FP16 scale was accepted");
}

} // namespace

int main() {
    try {
        check_prefix(NumericFormat::Q4G64_F16S, {0x98, 0x0f, 0x71}, {},
                     {-8, -7, -1, 0, 1, 7}, 65);
        check_prefix(NumericFormat::Q5G64_F16S, {0x10, 0x0f, 0xf1}, {0x07},
                     {-16, -15, -1, 0, 1, 15}, 65);
        check_prefix(NumericFormat::Q6G64_F16S, {0x10, 0x0f, 0x0f, 0xff}, {0xea, 0x43},
                     {-32, -31, -17, -16, -1, 0, 15, 31}, 65);
        check_prefix(NumericFormat::W8G32_F16S, {0x81, 0xff, 0x00, 0x01, 0x7f}, {},
                     {-127, -1, 0, 1, 127}, 33);
        scale_edges();
        check(vision_f32_to_bf16(std::bit_cast<float>(0x3f808000U)) == 0x3f80,
              "BF16 halfway-even down differs");
        check(vision_f32_to_bf16(std::bit_cast<float>(0x3f818000U)) == 0x3f82,
              "BF16 halfway-even up differs");
        bool rejected = false;
        try { (void)vision_f32_to_bf16(INFINITY); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "nonfinite BF16 source was accepted");
        rejected = false;
        try { (void)vision_f32_to_bf16(std::bit_cast<float>(0x7f7fffffU)); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "BF16 overflow after rounding was accepted");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

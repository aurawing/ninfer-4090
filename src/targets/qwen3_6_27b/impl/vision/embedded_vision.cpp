#include "embedded_vision.h"

#include "artifact/vision_row_decode.h"
#include "targets/qwen3_6/impl/vision/cpu_vision_encoder.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

using qwen3_6::EmbeddedVisionDtype;
using qwen3_6::EmbeddedVisionTensor;

std::size_t element_count(const EmbeddedVisionTensor& tensor) {
    std::size_t count = 1;
    if (tensor.rank == 0 || tensor.rank > tensor.dimensions.size()) {
        throw std::invalid_argument("embedded Vision tensor rank is invalid");
    }
    for (std::size_t i = 0; i < tensor.rank; ++i) {
        const auto axis = tensor.dimensions[i];
        if (axis <= 0 || static_cast<std::uint64_t>(axis) >
                std::numeric_limits<std::size_t>::max() / count) {
            throw std::invalid_argument("embedded Vision tensor element count overflows");
        }
        count *= static_cast<std::size_t>(axis);
    }
    return count;
}

void put_bf16(std::span<std::byte> output, std::size_t index, std::uint16_t bits) {
    output[index * 2] = std::byte(bits & 0xffU);
    output[index * 2 + 1] = std::byte(bits >> 8U);
}

void put_f32(std::span<std::byte> output, std::size_t index, float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    for (unsigned i = 0; i < 4; ++i) {
        output[index * 4 + i] = std::byte((bits >> (i * 8U)) & 0xffU);
    }
}

std::uint16_t get_bf16(std::span<const std::byte> input, std::size_t index) {
    return static_cast<std::uint16_t>(
        std::to_integer<unsigned>(input[index * 2]) |
        (std::to_integer<unsigned>(input[index * 2 + 1]) << 8U));
}

void fill_tensor(const artifact::TensorDescriptor& source,
                 std::span<const std::byte> payload, const EmbeddedVisionTensor& target,
                 int patch_time, std::span<std::byte> output) {
    const auto target_elements = element_count(target);
    const std::size_t word_bytes = target.dtype == EmbeddedVisionDtype::BF16 ? 2 : 4;
    if (target_elements > std::numeric_limits<std::size_t>::max() / word_bytes ||
        output.size() != target_elements * word_bytes) {
        throw std::invalid_argument("embedded Vision GGML destination size differs");
    }
    if (source.format == artifact::NumericFormat::BF16) {
        if (patch_time >= 0 || target.dtype != EmbeddedVisionDtype::F32 ||
            payload.size() != target_elements * 2) {
            throw std::invalid_argument("embedded Vision BF16 tensor mapping differs");
        }
        for (std::size_t index = 0; index < target_elements; ++index) {
            const auto bits = static_cast<std::uint32_t>(get_bf16(payload, index)) << 16U;
            put_f32(output, index, std::bit_cast<float>(bits));
        }
        return;
    }
    if (source.shape.size() != 2 ||
        source.layout != artifact::StorageLayout::RowSplitK128V1 ||
        source.shape[0] > std::numeric_limits<std::size_t>::max() ||
        source.shape[1] > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("embedded Vision quantized tensor mapping differs");
    }
    const auto rows = static_cast<std::size_t>(source.shape[0]);
    const auto columns = static_cast<std::size_t>(source.shape[1]);
    const bool patch = patch_time >= 0;
    if ((!patch && (target.dtype != EmbeddedVisionDtype::BF16 ||
                    target_elements != rows * columns)) ||
        (patch && (patch_time > 1 || target.dtype != EmbeddedVisionDtype::F32 ||
                   columns != 1536 || target_elements != rows * 768))) {
        throw std::invalid_argument("embedded Vision quantized tensor shape differs");
    }
    std::vector<float> decoded(columns);
    for (std::size_t row = 0; row < rows; ++row) {
        artifact::decode_vision_row_f32(payload, source.format, rows, columns, row, decoded);
        if (patch) {
            for (std::size_t channel = 0; channel < 3; ++channel) {
                for (std::size_t pixel = 0; pixel < 256; ++pixel) {
                    const auto source_index = (channel * 2 + patch_time) * 256 + pixel;
                    const auto bf16 = artifact::vision_f32_to_bf16(decoded[source_index]);
                    put_f32(output, row * 768 + channel * 256 + pixel,
                            std::bit_cast<float>(static_cast<std::uint32_t>(bf16) << 16U));
                }
            }
        } else {
            for (std::size_t column = 0; column < columns; ++column) {
                put_bf16(output, row * columns + column,
                         artifact::vision_f32_to_bf16(decoded[column]));
            }
        }
    }
}

} // namespace

std::shared_ptr<qwen3_6::CpuVisionEncoder> make_embedded_vision_encoder(
    artifact::Binder& binder, const VisionBindingPlan& vision,
    std::uint32_t threads, std::size_t memory_budget_bytes) {
    std::vector<EmbeddedVisionTensor> tensors;
    tensors.reserve(334);
    std::uint64_t identity = 14695981039346656037ULL;
    const auto add = [&](std::string name, artifact::ObjectHandle handle,
                         std::initializer_list<std::int64_t> shape,
                         EmbeddedVisionDtype dtype, int patch_time = -1) {
        const auto& source = std::get<artifact::TensorDescriptor>(binder.descriptor(handle));
        const auto payload = binder.payload(handle).data;
        if (payload.size() != source.bytes) {
            throw std::invalid_argument("embedded Vision source payload size differs");
        }
        EmbeddedVisionTensor target;
        target.name = std::move(name);
        target.rank = static_cast<std::uint32_t>(shape.size());
        if (target.rank == 0 || target.rank > 4) {
            throw std::invalid_argument("embedded Vision GGML rank differs");
        }
        std::copy(shape.begin(), shape.end(), target.dimensions.begin());
        target.dtype = dtype;
        const auto captured = target;
        target.fill = [source, payload, captured, patch_time](std::span<std::byte> output) {
            fill_tensor(source, payload, captured, patch_time, output);
        };
        for (const char ch : target.name) {
            identity = (identity ^ static_cast<unsigned char>(ch)) * 1099511628211ULL;
        }
        for (const auto value : payload) {
            identity = (identity ^ std::to_integer<unsigned char>(value)) * 1099511628211ULL;
        }
        tensors.push_back(std::move(target));
    };

    using D = EmbeddedVisionDtype;
    add("v.patch_embd.weight", vision.backbone.patch_embedding, {16, 16, 3, 1152}, D::F32, 0);
    add("v.patch_embd.weight.1", vision.backbone.patch_embedding, {16, 16, 3, 1152}, D::F32, 1);
    add("v.patch_embd.bias", vision.backbone.patch_embedding_bias, {1152}, D::F32);
    add("v.position_embd.weight", vision.backbone.position_embedding, {1152, 2304}, D::F32);
    for (std::size_t layer = 0; layer < vision.backbone.layers.size(); ++layer) {
        const auto& item = vision.backbone.layers[layer];
        const std::string prefix = "v.blk." + std::to_string(layer) + ".";
        add(prefix + "attn_qkv.weight", item.qkv, {1152, 3456}, D::BF16);
        add(prefix + "attn_qkv.bias", item.qkv_bias, {3456}, D::F32);
        add(prefix + "attn_out.weight", item.output, {1152, 1152}, D::BF16);
        add(prefix + "attn_out.bias", item.output_bias, {1152}, D::F32);
        add(prefix + "ffn_up.weight", item.fc1, {1152, 4304}, D::BF16);
        add(prefix + "ffn_up.bias", item.fc1_bias, {4304}, D::F32);
        add(prefix + "ffn_down.weight", item.fc2, {4304, 1152}, D::BF16);
        add(prefix + "ffn_down.bias", item.fc2_bias, {1152}, D::F32);
        add(prefix + "ln1.weight", item.norm1_weight, {1152}, D::F32);
        add(prefix + "ln1.bias", item.norm1_bias, {1152}, D::F32);
        add(prefix + "ln2.weight", item.norm2_weight, {1152}, D::F32);
        add(prefix + "ln2.bias", item.norm2_bias, {1152}, D::F32);
    }
    add("mm.0.weight", vision.merger_input.fc1, {4608, 4608}, D::BF16);
    add("mm.0.bias", vision.merger_input.fc1_bias, {4608}, D::F32);
    add("mm.2.weight", vision.merger_fc2, {4608, 5120}, D::BF16);
    add("mm.2.bias", vision.merger_fc2_bias, {5120}, D::F32);
    add("v.post_ln.weight", vision.merger_norm.weight, {1152}, D::F32);
    add("v.post_ln.bias", vision.merger_norm.bias, {1152}, D::F32);
    if (tensors.size() != 334) { throw std::logic_error("embedded Vision GGML inventory differs"); }
    return qwen3_6::make_embedded_cpu_vision_encoder(tensors, threads, memory_budget_bytes, identity);
}

} // namespace ninfer::targets::qwen3_6_27b::detail

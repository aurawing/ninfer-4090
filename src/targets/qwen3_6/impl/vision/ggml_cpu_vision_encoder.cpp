#include "cpu_vision_encoder.h"

#include "clip.h"
#include "clip-impl.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <array>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <exception>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace ninfer::targets::qwen3_6 {
namespace {

constexpr std::size_t host_metadata_allowance = 32 * 1024 * 1024;
// Windows worker stacks reserve 1 MiB by default. Include their complete reservation
// in admission in addition to graph_plan's numerical scratch and metadata allowance.
constexpr std::size_t worker_stack_allowance = 1024 * 1024;
constexpr std::string_view numerical_identity =
    "ninfer-cpu-vision-v2:b81c99b479d4c24e5eeca10de99032ebd343ef8f:"
    "avx2:llamafile:openmp:fa-f32:patch-f32:bf16-boundaries:block-tanh-f32:merger-erf";

std::size_t add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b) {
        throw std::invalid_argument("CPU vision memory size overflow");
    }
    return a + b;
}

std::size_t mul(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::invalid_argument("CPU vision memory size overflow");
    }
    return a * b;
}

void admit(std::size_t requested, std::size_t budget) {
    if (requested > budget) {
        throw std::runtime_error("CPU vision host-memory budget exceeded: requires " +
            std::to_string(requested) + " bytes, budget " + std::to_string(budget));
    }
}

struct GgufDelete { void operator()(gguf_context* p) const { gguf_free(p); } };
struct GgmlDelete { void operator()(ggml_context* p) const { ggml_free(p); } };
struct ClipDelete { void operator()(clip_ctx* p) const { clip_free(p); } };
using Gguf = std::unique_ptr<gguf_context, GgufDelete>;
using Ggml = std::unique_ptr<ggml_context, GgmlDelete>;
using Clip = std::unique_ptr<clip_ctx, ClipDelete>;

std::int64_t key(gguf_context* meta, const char* name, gguf_type type) {
    const auto id = gguf_find_key(meta, name);
    if (id < 0 || gguf_get_kv_type(meta, id) != type) {
        throw std::invalid_argument(std::string("CPU vision GGUF metadata missing or mistyped: ") + name);
    }
    return id;
}

void integer(gguf_context* meta, const char* name, std::uint32_t expected) {
    if (gguf_get_val_u32(meta, key(meta, name, GGUF_TYPE_UINT32)) != expected) {
        throw std::invalid_argument(std::string("CPU vision GGUF incompatible metadata: ") + name);
    }
}

void text_value(gguf_context* meta, const char* name, std::string_view expected) {
    if (gguf_get_val_str(meta, key(meta, name, GGUF_TYPE_STRING)) != expected) {
        throw std::invalid_argument(std::string("CPU vision GGUF incompatible metadata: ") + name);
    }
}

void tensor(gguf_context* meta, const std::string& name, std::initializer_list<std::int64_t> dims,
            ggml_type expected) {
    const auto id = gguf_find_tensor(meta, name.c_str());
    if (id < 0 || gguf_get_tensor_type(meta, id) != expected) {
        throw std::invalid_argument("CPU vision GGUF tensor missing or wrong dtype: " + name);
    }
    const auto* actual = gguf_get_tensor_ne(meta, id);
    std::size_t axis = 0;
    for (const auto dim : dims) {
        if (actual[axis++] != dim) {
            throw std::invalid_argument("CPU vision GGUF tensor has incompatible shape: " + name);
        }
    }
    for (; axis < GGML_MAX_DIMS; ++axis) {
        if (actual[axis] != 1) {
            throw std::invalid_argument("CPU vision GGUF tensor has incompatible rank: " + name);
        }
    }
}

// FNV-1a is a cache identity, not an authentication mechanism. Hash every file byte
// plus the pinned dependency and numerical policy so replacement weights invalidate caches.
std::uint64_t content_hash(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("Cannot read CPU vision GGUF: " + path); }
    std::uint64_t hash = 14695981039346656037ULL;
    const auto consume = [&](const char* data, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) {
            hash ^= static_cast<unsigned char>(data[i]);
            hash *= 1099511628211ULL;
        }
    };
    consume(numerical_identity.data(), numerical_identity.size());
    std::array<char, 64 * 1024> buffer;
    while (file) {
        file.read(buffer.data(), buffer.size());
        consume(buffer.data(), static_cast<std::size_t>(file.gcount()));
    }
    if (!file.eof()) { throw std::runtime_error("Cannot hash CPU vision GGUF: " + path); }
    return hash;
}

std::size_t preflight(const std::string& path, std::size_t budget) {
    admit(host_metadata_allowance, budget);
    // gguf no_alloc really creates metadata only. clip's no_alloc still allocates
    // its complete weight buffer and is deliberately not used for this preflight.
    std::ifstream header(path, std::ios::binary);
    std::array<unsigned char, 24> prefix{};
    header.read(reinterpret_cast<char*>(prefix.data()), prefix.size());
    const auto little_endian = [&](std::size_t start, std::size_t count) {
        std::uint64_t value = 0;
        for (std::size_t i = 0; i < count; ++i) { value |= std::uint64_t(prefix[start + i]) << (8 * i); }
        return value;
    };
    if (!header || prefix[0] != 'G' || prefix[1] != 'G' || prefix[2] != 'U' || prefix[3] != 'F' ||
        little_endian(4, 4) != 3 || little_endian(8, 8) != 334 || little_endian(16, 8) > 128) {
        throw std::invalid_argument("CPU vision requires the regular BF16 Qwen 27B vision GGUF header");
    }
    const auto reader = +[](void* userdata, void* output, std::uint64_t offset, std::size_t count) -> std::size_t {
        auto& file = *static_cast<std::ifstream*>(userdata);
        file.clear();
        file.seekg(static_cast<std::streamoff>(offset));
        file.read(static_cast<char*>(output), static_cast<std::streamsize>(count));
        return static_cast<std::size_t>(file.gcount());
    };
    ggml_context* raw = nullptr;
    // Bound the parser's remaining-byte count to 1 MiB so malformed metadata cannot
    // allocate a payload-sized string/array before the shape checks run.
    Gguf meta(gguf_init_from_callback(reader, &header, 64 * 1024,
        std::min<std::uint64_t>(std::filesystem::file_size(path), 1024 * 1024), {true, &raw}));
    Ggml tensors(raw);
    if (!meta || !tensors) { throw std::invalid_argument("Invalid CPU vision GGUF header: " + path); }
    if (gguf_get_version(meta.get()) != 3 || gguf_get_n_tensors(meta.get()) != 334) {
        throw std::invalid_argument("CPU vision requires the regular BF16 Qwen 27B vision GGUF");
    }
    text_value(meta.get(), "general.architecture", "clip");
    text_value(meta.get(), "clip.projector_type", "qwen3vl_merger");
    if (!gguf_get_val_bool(meta.get(), key(meta.get(), "clip.has_vision_encoder", GGUF_TYPE_BOOL)) ||
        !gguf_get_val_bool(meta.get(), key(meta.get(), "clip.use_gelu", GGUF_TYPE_BOOL))) {
        throw std::invalid_argument("CPU vision GGUF must contain the GELU vision encoder");
    }
    for (const char* name : {"clip.has_audio_encoder", "clip.has_audio_decoder"}) {
        const auto id = gguf_find_key(meta.get(), name);
        if (id >= 0 && (gguf_get_kv_type(meta.get(), id) != GGUF_TYPE_BOOL ||
            gguf_get_val_bool(meta.get(), id))) {
            throw std::invalid_argument("CPU vision GGUF must not contain audio weights");
        }
    }
    integer(meta.get(), "clip.vision.block_count", 27);
    integer(meta.get(), "clip.vision.embedding_length", 1152);
    integer(meta.get(), "clip.vision.feed_forward_length", 4304);
    integer(meta.get(), "clip.vision.attention.head_count", 16);
    integer(meta.get(), "clip.vision.projection_dim", 5120);
    integer(meta.get(), "clip.vision.patch_size", 16);
    integer(meta.get(), "clip.vision.spatial_merge_size", 2);
    integer(meta.get(), "clip.vision.image_size", 768);
    const float eps = gguf_get_val_f32(meta.get(), key(meta.get(),
        "clip.vision.attention.layer_norm_epsilon", GGUF_TYPE_FLOAT32));
    if (!std::isfinite(eps) || std::abs(eps - 1e-6F) > 1e-12F) {
        throw std::invalid_argument("CPU vision GGUF LayerNorm epsilon differs from NInfer");
    }
    const auto deepstack = key(meta.get(), "clip.vision.is_deepstack_layers", GGUF_TYPE_ARRAY);
    if (gguf_get_arr_type(meta.get(), deepstack) != GGUF_TYPE_BOOL ||
        gguf_get_arr_n(meta.get(), deepstack) != 27) {
        throw std::invalid_argument("CPU vision GGUF DeepStack metadata is incompatible");
    }
    const auto* deep = static_cast<const std::int8_t*>(gguf_get_arr_data(meta.get(), deepstack));
    for (std::size_t i = 0; i < 27; ++i) {
        if (deep[i]) { throw std::invalid_argument("CPU vision does not accept DeepStack weights"); }
    }
    tensor(meta.get(), "v.patch_embd.weight", {16, 16, 3, 1152}, GGML_TYPE_F32);
    tensor(meta.get(), "v.patch_embd.weight.1", {16, 16, 3, 1152}, GGML_TYPE_F32);
    tensor(meta.get(), "v.patch_embd.bias", {1152}, GGML_TYPE_F32);
    tensor(meta.get(), "v.position_embd.weight", {1152, 2304}, GGML_TYPE_F32);
    tensor(meta.get(), "v.post_ln.weight", {1152}, GGML_TYPE_F32);
    tensor(meta.get(), "v.post_ln.bias", {1152}, GGML_TYPE_F32);
    tensor(meta.get(), "mm.0.weight", {4608, 4608}, GGML_TYPE_BF16);
    tensor(meta.get(), "mm.0.bias", {4608}, GGML_TYPE_F32);
    tensor(meta.get(), "mm.2.weight", {4608, 5120}, GGML_TYPE_BF16);
    tensor(meta.get(), "mm.2.bias", {5120}, GGML_TYPE_F32);
    for (int layer = 0; layer < 27; ++layer) {
        const auto prefix = "v.blk." + std::to_string(layer) + ".";
        tensor(meta.get(), prefix + "attn_qkv.weight", {1152, 3456}, GGML_TYPE_BF16);
        tensor(meta.get(), prefix + "attn_qkv.bias", {3456}, GGML_TYPE_F32);
        tensor(meta.get(), prefix + "attn_out.weight", {1152, 1152}, GGML_TYPE_BF16);
        tensor(meta.get(), prefix + "attn_out.bias", {1152}, GGML_TYPE_F32);
        tensor(meta.get(), prefix + "ffn_up.weight", {1152, 4304}, GGML_TYPE_BF16);
        tensor(meta.get(), prefix + "ffn_up.bias", {4304}, GGML_TYPE_F32);
        tensor(meta.get(), prefix + "ffn_down.weight", {4304, 1152}, GGML_TYPE_BF16);
        tensor(meta.get(), prefix + "ffn_down.bias", {1152}, GGML_TYPE_F32);
        for (const char* name : {"ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"}) {
            tensor(meta.get(), prefix + name, {1152}, GGML_TYPE_F32);
        }
    }
    const auto file_size = std::filesystem::file_size(path);
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    for (std::int64_t i = 0; i < gguf_get_n_tensors(meta.get()); ++i) {
        const auto start = add(gguf_get_data_offset(meta.get()), gguf_get_tensor_offset(meta.get(), i));
        const auto end = add(start, gguf_get_tensor_size(meta.get(), i));
        if (end > file_size) { throw std::invalid_argument("Truncated CPU vision GGUF tensor payload"); }
        ranges.emplace_back(start, end);
    }
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].first < ranges[i - 1].second) {
            throw std::invalid_argument("CPU vision GGUF tensor payloads overlap");
        }
    }
    const auto bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(tensors.get(),
        ggml_backend_cpu_buffer_type());
    admit(add(bytes, host_metadata_allowance), budget);
    return bytes;
}

class GgufVisionEncoder final : public CpuVisionEncoder {
public:
    GgufVisionEncoder(const std::string& path, std::uint32_t threads, std::size_t budget,
                      bool cuda)
        : threads_(threads), budget_(budget), cuda_(cuda) {
        if (threads < 1 || threads > 512) {
            throw std::invalid_argument("CPU vision thread count must be in [1,512]");
        }
        const auto expected_weights = preflight(path, cuda_ ? std::numeric_limits<std::size_t>::max() : budget);
        if (!cuda_) {
            admit(add(add(expected_weights, host_metadata_allowance),
                      mul(threads_, worker_stack_allowance)), budget_);
        }
        const auto before = content_hash(path);
        clip_context_params params{};
        params.use_gpu = cuda_;
        params.ninfer_cpu_math = !cuda_;
        params.device = nullptr;
        if (cuda_) {
#ifdef NINFER_GGML_CUDA_VISION
            device_ = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            if (!device_) { throw std::runtime_error("GGML CUDA vision has no GPU device"); }
            std::size_t free_bytes = 0, total_bytes = 0;
            ggml_backend_dev_memory(device_, &free_bytes, &total_bytes);
            constexpr std::size_t minimum_compute_reserve = 512ULL * 1024 * 1024;
            if (free_bytes < add(expected_weights, minimum_compute_reserve)) {
                throw std::runtime_error("GGML CUDA vision has insufficient free VRAM for weights and compute reserve");
            }
            params.device = device_;
#else
            throw std::runtime_error("GGML CUDA vision backend was not built");
#endif
        }
        params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_ENABLED;
        params.warmup = false;
        params.no_alloc = false;
        // These limits do not preprocess input. Geometry is supplied by NInfer's frontend.
        params.image_min_tokens = 1;
        params.image_max_tokens = std::numeric_limits<int>::max() / 1024;
        const auto loaded = clip_init(path.c_str(), params);
        Clip audio(loaded.ctx_a), audio_gen(loaded.ctx_gen_a);
        context_.reset(loaded.ctx_v);
        if (!context_ || audio || audio_gen || clip_n_mmproj_embd(context_.get()) != 5120 ||
            clip_model_n_temporal_merge(context_.get()) != 2) {
            throw std::runtime_error("Failed to load compatible CPU vision GGUF");
        }
        weights_ = clip_ninfer_cpu_weight_bytes(context_.get());
        if (!cuda_ && weights_ > add(expected_weights, 4096)) {
            throw std::runtime_error("CPU vision loaded weight allocation exceeds preflight");
        }
        if (!cuda_) {
            admit(add(add(weights_, host_metadata_allowance),
                      mul(threads_, worker_stack_allowance)), budget_);
        } else {
            const auto usage = clip_get_mem_usage(context_.get());
            if (!usage.contains(device_) || usage.at(device_) < expected_weights) {
                throw std::runtime_error("GGML CUDA vision weights were not allocated on the GPU");
            }
        }
        const auto after = content_hash(path);
        if (after != before) {
            throw std::runtime_error("CPU vision GGUF changed while its weights were loading");
        }
        identity_ = after ^ (cuda_ ? 0xa19c601f3df58e03ULL : 0ULL);
    }

    GgufVisionEncoder(Clip vision, Clip audio, Clip audio_gen,
                      std::uint32_t threads, std::size_t budget,
                      std::size_t expected_weights, std::uint64_t identity)
        : threads_(threads), budget_(budget), identity_(identity) {
        context_ = std::move(vision);
        if (!context_ || audio || audio_gen || clip_n_mmproj_embd(context_.get()) != 5120 ||
            clip_model_n_temporal_merge(context_.get()) != 2) {
            throw std::runtime_error("Failed to load compatible embedded CPU Vision weights");
        }
        weights_ = clip_ninfer_cpu_weight_bytes(context_.get());
        if (weights_ > add(expected_weights, 4096)) {
            throw std::runtime_error("embedded CPU Vision weight allocation exceeds preflight");
        }
        admit(add(add(weights_, host_metadata_allowance),
                  mul(threads_, worker_stack_allowance)), budget_);
    }

    void validate(const CpuVisionInput& input) const override {
        const auto shape = validate_cpu_vision_input(input);
        std::lock_guard lock(mutex_);
        measure_and_admit(input, shape);
    }

    std::vector<std::uint16_t> encode(const CpuVisionInput& input,
            const std::function<bool()>& cancelled) override {
        const auto shape = validate_cpu_vision_input(input);
        std::lock_guard lock(mutex_);
        const auto check_cancelled = [&] { if (cancelled && cancelled()) { throw CpuVisionCancelled(); } };
        check_cancelled();
        measure_and_admit(input, shape);
        check_cancelled();
        if (!cuda_) {
            clip_ninfer_cpu_reserve(context_.get(), shape.frame_width, shape.frame_height,
                static_cast<int>(shape.frames_per_group));
        }
        struct AbortState {
            const std::function<bool()>& callback;
            std::atomic<bool> requested{false};
            std::atomic<bool> callback_failed{false};
        } abort{cancelled};
        const auto abort_fn = +[](void* raw) noexcept -> bool {
            auto& state = *static_cast<AbortState*>(raw);
            try {
                if (state.callback && state.callback()) { state.requested.store(true); }
            } catch (...) {
                state.callback_failed.store(true);
                state.requested.store(true);
            }
            return state.requested.load();
        };
        if (!cuda_) { clip_ninfer_cpu_set_abort(context_.get(), abort_fn, &abort); }
        struct ResetAbort {
            clip_ctx* context;
            bool enabled;
            ~ResetAbort() { if (enabled) { clip_ninfer_cpu_set_abort(context, nullptr, nullptr); } }
        } reset{context_.get(), !cuda_};
        std::vector<std::uint16_t> result(mul(shape.merged_tokens, 5120));
        const auto per_group = mul(static_cast<std::size_t>(input.height) * input.width / 4, 5120);
        const auto group_patch_elements = mul(mul(static_cast<std::size_t>(input.height), input.width), 1536);
        for (std::uint32_t group = 0; group < static_cast<std::uint32_t>(input.temporal); ++group) {
            check_cancelled();
            const CpuVisionInput group_input{input.patches.subspan(
                mul(group, group_patch_elements), group_patch_elements), 1, input.height, input.width, input.video};
            auto frames = unpack_validated_cpu_vision_group(group_input, shape, 0);
            clip_image_f32_batch batch;
            batch.entries.reserve(frames.size());
            for (auto& frame : frames) {
                clip_image_f32 image;
                image.set_size({frame.width, frame.height}, true, false);
                image.cpy_buf(frame.rgb);
                batch.entries.push_back(std::move(image));
            }
            frames.clear();
            std::vector<float> output(per_group);
            const bool success = clip_image_batch_encode(context_.get(), static_cast<int>(threads_), &batch, output);
            if (abort.callback_failed.load()) {
                throw std::runtime_error("CPU vision cancellation callback failed");
            }
            if (abort.requested.load()) { throw CpuVisionCancelled(); }
            check_cancelled();
            if (!success || output.size() != per_group) {
                throw std::runtime_error("CPU vision GGML graph execution failed");
            }
            const auto begin = static_cast<std::size_t>(group) * per_group;
            for (std::size_t i = 0; i < output.size(); ++i) {
                if (!std::isfinite(output[i])) {
                    throw std::runtime_error("CPU vision produced nonfinite output");
                }
                const auto value = ggml_fp32_to_bf16(output[i]);
                if ((value.bits & 0x7f80U) == 0x7f80U) {
                    throw std::runtime_error("CPU vision output overflows BF16");
                }
                result[begin + i] = value.bits;
            }
        }
        check_cancelled();
        return result;
    }

    std::uint64_t identity_hash() const noexcept override { return identity_; }
    std::size_t weight_bytes() const noexcept override { return weights_; }

private:
    void measure_and_admit(const CpuVisionInput& input, const CpuVisionShape& shape) const {
        // Bound external allocations before constructing a graph or inverse-layout frames.
        const auto rgb = mul(mul(mul(static_cast<std::size_t>(shape.frame_width), shape.frame_height), 3),
            mul(shape.frames_per_group, sizeof(float)));
        const auto output_all = mul(mul(shape.merged_tokens, 5120), sizeof(std::uint16_t));
        const auto output_group = mul(mul(static_cast<std::size_t>(input.height) * input.width / 4, 5120), sizeof(float));
        // Caller patches + bridge/clip RGB copy overlap + clip CHW + F32 embeddings + BF16
        // result and future pinned H2D staging. Keep both BF16 copies inside this admission.
        auto external = add(mul(input.patches.size(), sizeof(float)), mul(rgb, 3));
        external = add(external, output_group);
        external = add(external, mul(output_all, 2));
        // clip also creates 4-axis MRoPE position indices; CHW and index construction do not
        // overlap but conservatively count both, with graph/scheduler/thread metadata allowance.
        external = add(external, mul(static_cast<std::size_t>(input.height) * input.width, 4 * sizeof(std::int32_t)));
        admit(add(add(cuda_ ? 0 : weights_, host_metadata_allowance), external), budget_);
        if (cuda_) { return; }
        const auto plan = clip_ninfer_cpu_measure(context_.get(), shape.frame_width, shape.frame_height,
            static_cast<int>(shape.frames_per_group), static_cast<int>(threads_));
        auto total = add(weights_, external);
        total = add(total, host_metadata_allowance);
        total = add(total, mul(threads_, worker_stack_allowance));
        total = add(total, plan.metadata_bytes);
        // The pinned graph allocator and CPU scratch allocator both release an old
        // buffer before growing it. Retained capacity is reused, so it does not
        // overlap with the requested replacement and must not be charged twice.
        total = add(total, std::max(plan.graph_bytes, plan.retained_graph_bytes));
        total = add(total, std::max(plan.scratch_bytes, plan.retained_scratch_bytes));
        admit(total, budget_);
    }

    Clip context_;
    std::uint32_t threads_;
    std::size_t budget_;
    std::size_t weights_ = 0;
    std::uint64_t identity_ = 0;
    bool cuda_ = false;
    ggml_backend_dev_t device_ = nullptr;
    mutable std::mutex mutex_;
};

} // namespace

bool cpu_vision_backend_available() noexcept { return true; }

std::shared_ptr<CpuVisionEncoder> make_gguf_cpu_vision_encoder(const std::string& path,
    std::uint32_t threads, std::size_t memory_budget_bytes) {
    return std::make_shared<GgufVisionEncoder>(path, threads, memory_budget_bytes, false);
}

std::shared_ptr<CpuVisionEncoder> make_gguf_cuda_vision_encoder(const std::string& path,
    std::size_t host_memory_budget_bytes) {
    return std::make_shared<GgufVisionEncoder>(path, 6, host_memory_budget_bytes, true);
}

std::shared_ptr<CpuVisionEncoder> make_embedded_cpu_vision_encoder(
    std::span<const EmbeddedVisionTensor> tensors, std::uint32_t threads,
    std::size_t memory_budget_bytes, std::uint64_t content_hash_value) {
    if (threads < 1 || threads > 512 || tensors.size() != 334) {
        throw std::invalid_argument("embedded CPU Vision requires 334 tensors and threads in [1,512]");
    }
    Gguf metadata(gguf_init_empty());
    Ggml descriptions(ggml_init({(tensors.size() + 1) * ggml_tensor_overhead(), nullptr, true}));
    if (!metadata || !descriptions) { throw std::runtime_error("cannot allocate embedded Vision GGUF metadata"); }
    gguf_set_val_str(metadata.get(), "general.architecture", "clip");
    gguf_set_val_str(metadata.get(), "general.name", "NInfer embedded Qwen3 Vision");
    gguf_set_val_str(metadata.get(), "clip.projector_type", "qwen3vl_merger");
    gguf_set_val_bool(metadata.get(), "clip.has_vision_encoder", true);
    gguf_set_val_bool(metadata.get(), "clip.use_gelu", true);
    for (const auto& [key_name, value] : std::array<std::pair<const char*, std::uint32_t>, 8>{
             {{"clip.vision.block_count", 27}, {"clip.vision.embedding_length", 1152},
              {"clip.vision.feed_forward_length", 4304},
              {"clip.vision.attention.head_count", 16},
              {"clip.vision.projection_dim", 5120}, {"clip.vision.patch_size", 16},
              {"clip.vision.spatial_merge_size", 2}, {"clip.vision.image_size", 768}}}) {
        gguf_set_val_u32(metadata.get(), key_name, value);
    }
    gguf_set_val_f32(metadata.get(), "clip.vision.attention.layer_norm_epsilon", 1e-6F);
    const std::array<float, 3> normalization{0.5F, 0.5F, 0.5F};
    gguf_set_arr_data(metadata.get(), "clip.vision.image_mean", GGUF_TYPE_FLOAT32,
                      normalization.data(), normalization.size());
    gguf_set_arr_data(metadata.get(), "clip.vision.image_std", GGUF_TYPE_FLOAT32,
                      normalization.data(), normalization.size());
    const std::array<std::int8_t, 27> no_deepstack{};
    gguf_set_arr_data(metadata.get(), "clip.vision.is_deepstack_layers", GGUF_TYPE_BOOL,
                      no_deepstack.data(), no_deepstack.size());

    std::unordered_map<std::string_view, std::size_t> names;
    names.reserve(tensors.size());
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        const auto& source = tensors[index];
        if (source.name.empty() || !names.emplace(source.name, index).second ||
            source.rank < 1 || source.rank > 4 || !source.fill) {
            throw std::invalid_argument("embedded Vision GGUF has duplicate or invalid tensor descriptors");
        }
        for (std::size_t axis = 0; axis < source.rank; ++axis) {
            if (source.dimensions[axis] <= 0) {
                throw std::invalid_argument("embedded Vision GGUF tensor axis is invalid");
            }
        }
        const auto type = source.dtype == EmbeddedVisionDtype::BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F32;
        auto* tensor = ggml_new_tensor(descriptions.get(), type,
                                       static_cast<int>(source.rank), source.dimensions.data());
        if (!tensor) { throw std::runtime_error("embedded Vision GGUF tensor metadata allocation failed"); }
        ggml_set_name(tensor, source.name.c_str());
        gguf_add_tensor(metadata.get(), tensor);
    }
    const auto expected_weights = ggml_backend_alloc_ctx_tensors_from_buft_size(
        descriptions.get(), ggml_backend_cpu_buffer_type());
    admit(add(add(expected_weights, host_metadata_allowance),
              mul(threads, worker_stack_allowance)), memory_budget_bytes);
    std::vector<std::uint8_t> bytes(gguf_get_meta_size(metadata.get()));
    if (bytes.size() > 1024 * 1024) {
        throw std::invalid_argument("embedded Vision GGUF metadata exceeds 1 MiB");
    }
    gguf_get_meta_data(metadata.get(), bytes.data());

    struct FillState {
        std::span<const EmbeddedVisionTensor> tensors;
        const std::unordered_map<std::string_view, std::size_t>& names;
        std::vector<bool> filled;
        std::exception_ptr error;
    } state{tensors, names, std::vector<bool>(tensors.size()), nullptr};
    const auto fill = +[](void* raw, const char* name, ggml_type type,
                          const std::int64_t* dimensions, int rank, void* destination,
                          std::size_t count) noexcept -> bool {
        auto& state = *static_cast<FillState*>(raw);
        try {
            const auto it = state.names.find(name);
            if (it == state.names.end() || state.filled[it->second]) {
                throw std::invalid_argument("embedded Vision GGML requested an unknown or duplicate tensor");
            }
            const auto& spec = state.tensors[it->second];
            const auto expected_type = spec.dtype == EmbeddedVisionDtype::BF16
                ? GGML_TYPE_BF16 : GGML_TYPE_F32;
            if (type != expected_type || rank != static_cast<int>(spec.rank)) {
                throw std::invalid_argument("embedded Vision GGML requested a mismatched tensor dtype or rank");
            }
            for (int axis = 0; axis < rank; ++axis) {
                if (dimensions[axis] != spec.dimensions[axis]) {
                    throw std::invalid_argument("embedded Vision GGML requested a mismatched tensor shape");
                }
            }
            spec.fill({static_cast<std::byte*>(destination), count});
            state.filled[it->second] = true;
            return true;
        } catch (...) {
            state.error = std::current_exception();
            return false;
        }
    };
    clip_context_params params{};
    params.use_gpu = false;
    params.ninfer_cpu_math = true;
    params.device = nullptr;
    params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_ENABLED;
    params.warmup = false;
    params.no_alloc = false;
    params.image_min_tokens = 1;
    params.image_max_tokens = std::numeric_limits<int>::max() / 1024;
    const auto loaded = clip_ninfer_init_from_metadata(bytes.data(), bytes.size(), fill, &state, params);
    Clip vision(loaded.ctx_v), audio(loaded.ctx_a), audio_gen(loaded.ctx_gen_a);
    if (state.error) { std::rethrow_exception(state.error); }
    if (std::find(state.filled.begin(), state.filled.end(), false) != state.filled.end()) {
        throw std::runtime_error("embedded Vision GGML did not consume every weight tensor");
    }
    std::uint64_t identity = content_hash_value;
    for (const char ch : numerical_identity) {
        identity = (identity ^ static_cast<unsigned char>(ch)) * 1099511628211ULL;
    }
    return std::make_shared<GgufVisionEncoder>(
        std::move(vision), std::move(audio), std::move(audio_gen),
        threads, memory_budget_bytes,
                                                expected_weights, identity);
}

} // namespace ninfer::targets::qwen3_6

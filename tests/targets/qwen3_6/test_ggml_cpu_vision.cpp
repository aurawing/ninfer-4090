#include "targets/qwen3_6/impl/vision/cpu_vision_encoder.h"

#include <bit>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#ifdef NINFER_TEST_GGML_CPU_VISION
#include "clip.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#endif

namespace {
using namespace ninfer::targets::qwen3_6;
void require(bool good, const char* message) { if (!good) { throw std::runtime_error(message); } }

#ifdef NINFER_TEST_GGML_CPU_VISION
// Every numerical expectation below comes from scalar FP64 mathematics rather than
// another implementation of the production graph.
struct Graph {
    ggml_context* context = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    Graph() {
        context = ggml_init({8 * 1024 * 1024, nullptr, true});
        backend = ggml_backend_cpu_init();
        require(context && backend, "cannot create numerical graph");
        ggml_backend_cpu_set_n_threads(backend, 2);
    }
    ~Graph() {
        if (buffer) { ggml_backend_buffer_free(buffer); }
        if (backend) { ggml_backend_free(backend); }
        if (context) { ggml_free(context); }
    }
    Graph(const Graph&) = delete;
    std::vector<float> run(ggml_tensor* output,
        std::initializer_list<std::pair<ggml_tensor*, const std::vector<float>*>> inputs) {
        auto* graph = ggml_new_graph_custom(context, 1024, false);
        ggml_build_forward_expand(graph, output);
        buffer = ggml_backend_alloc_ctx_tensors(context, backend);
        require(buffer != nullptr, "numerical graph allocation failed");
        for (const auto& [tensor, values] : inputs) {
            require(ggml_nbytes(tensor) == values->size() * sizeof(float), "bad numerical input shape");
            ggml_backend_tensor_set(tensor, values->data(), 0, ggml_nbytes(tensor));
        }
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "numerical graph failed");
        std::vector<float> result(static_cast<std::size_t>(ggml_nelements(output)));
        ggml_backend_tensor_get(output, result.data(), 0, ggml_nbytes(output));
        return result;
    }
};

void near(double actual, double expected, double tolerance, const char* message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::cerr << message << ": actual=" << actual << ", expected=" << expected << '\n';
        throw std::runtime_error(message);
    }
}

void gelu() {
    std::vector<float> values;
    for (int i = -80; i <= 80; ++i) { values.push_back(static_cast<float>(i) / 10); }
    for (bool exact : {false, true}) {
        Graph graph;
        auto* input = ggml_new_tensor_1d(graph.context, GGML_TYPE_F32, values.size());
        auto* output = exact ? clip_ninfer_cpu_merger_gelu(graph.context, input) : ggml_gelu(graph.context, input);
        const auto result = graph.run(output, {{input, &values}});
        for (std::size_t i = 0; i < values.size(); ++i) {
            const double x = values[i];
            const double expected = exact ? 0.5 * x * (1 + std::erf(x / std::sqrt(2.0))) :
                0.5 * x * (1 + std::tanh(std::sqrt(2.0 / std::acos(-1.0)) * (x + 0.044715 * x * x * x)));
            near(result[i], expected, 3e-6, exact ? "merger erf GELU" : "block tanh GELU");
        }
    }
}

void attention_case(int tokens) {
    constexpr int d = 72, heads = 2;
    std::vector<float> q(d * heads * tokens), k(q.size()), v(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) {
        q[i] = static_cast<float>(std::sin(double(i) * 0.13) * 1e-5);
        k[i] = static_cast<float>(std::cos(double(i) * 0.19) * 90000);
        v[i] = static_cast<float>(70000 + std::sin(double(i) * 0.07) * 30000);
    }
    Graph graph;
    auto* qt = ggml_new_tensor_4d(graph.context, GGML_TYPE_F32, d, tokens, heads, 1);
    auto* kt = ggml_new_tensor_4d(graph.context, GGML_TYPE_F32, d, tokens, heads, 1);
    auto* vt = ggml_new_tensor_4d(graph.context, GGML_TYPE_F32, d, tokens, heads, 1);
    const float scale = 1.0F / std::sqrt(static_cast<float>(d));
    auto* output = clip_ninfer_cpu_attention(graph.context, qt, kt, vt, scale);
    require(output->src[1]->type == GGML_TYPE_F32 && output->src[2]->type == GGML_TYPE_F32,
        "CPU attention recast K/V to a narrow exponent range");
    const auto result = graph.run(output, {{qt, &q}, {kt, &k}, {vt, &v}});
    for (int head = 0; head < heads; ++head) {
        for (int query = 0; query < tokens; ++query) {
            std::vector<double> scores(tokens);
            double maximum = -std::numeric_limits<double>::infinity();
            for (int key = 0; key < tokens; ++key) {
                double dot = 0;
                for (int x = 0; x < d; ++x) {
                    dot += double(q[(head * tokens + query) * d + x]) * k[(head * tokens + key) * d + x];
                }
                scores[key] = dot * scale;
                maximum = std::max(maximum, scores[key]);
            }
            double sum = 0;
            for (auto& score : scores) { score = std::exp(score - maximum); sum += score; }
            for (int x = 0; x < d; ++x) {
                double expected = 0;
                for (int key = 0; key < tokens; ++key) {
                    expected += scores[key] / sum * v[(head * tokens + key) * d + x];
                }
                near(result[(query * heads + head) * d + x], expected, 0.08,
                    tokens >= 64 ? "72-dimension tiled F32 Flash Attention FP64 oracle" :
                                   "72-dimension untiled F32 Flash Attention FP64 oracle");
            }
        }
    }
}

void attention() {
    attention_case(5);
    // Pinned GGML_FA_TILE_Q is 64: 65 queries exercise a full F32 query tile
    // and a one-query tail, with every output checked against scalar FP64.
    attention_case(65);
}

void bf16_rounding() {
    // Adjacent BF16 midpoints with even/odd lower mantissas, both signs.
    for (const auto& [bits, expected] : std::vector<std::pair<std::uint32_t, std::uint16_t>>{
        {0x3f808000, 0x3f80}, {0x3f818000, 0x3f82}, {0xbf808000, 0xbf80},
        {0xbf818000, 0xbf82}, {0x00008000, 0x0000}, {0x00018000, 0x0002}}) {
        require(ggml_fp32_to_bf16(std::bit_cast<float>(bits)).bits == expected, "BF16 conversion is not ties-to-even");
    }
    std::vector<float> values{1.00390625F, 1.01171875F, -1.00390625F, -1.01171875F, 131072.0F};
    Graph graph;
    auto* input = ggml_new_tensor_1d(graph.context, GGML_TYPE_F32, values.size());
    const auto result = graph.run(clip_ninfer_cpu_bf16(graph.context, input), {{input, &values}});
    const std::vector<double> expected{1.0, 1.015625, -1.0, -1.015625, 131072.0};
    for (std::size_t i = 0; i < result.size(); ++i) { near(result[i], expected[i], 0, "BF16 graph boundary"); }
}

double round_bf16(double value) {
    // Independent positive/negative quantization oracle using the binary exponent
    // and nearest-even rounding of the seven stored fraction bits.
    if (!value) { return value; }
    int exponent = 0;
    const double fraction = std::frexp(value, &exponent);
    return std::ldexp(std::nearbyint(fraction * 256), exponent - 8);
}

void matrix_product() {
    // Vision-style BF16 weights/activations, large enough for the optimized GEMM
    // path, plus an irregular tail. Expectations use scalar FP64 accumulation.
    for (const int columns : {32, 35}) {
        constexpr int inner = 256, rows = 64;
        std::vector<float> weights(inner * rows), values(inner * columns);
        for (std::size_t i = 0; i < weights.size(); ++i) { weights[i] = float(int(i * 17 % 41) - 20) / 17; }
        for (std::size_t i = 0; i < values.size(); ++i) { values[i] = float(int(i * 13 % 37) - 18) / 19; }
        Graph graph;
        auto* wt = ggml_new_tensor_2d(graph.context, GGML_TYPE_F32, inner, rows);
        auto* xt = ggml_new_tensor_2d(graph.context, GGML_TYPE_F32, inner, columns);
        auto* output = ggml_mul_mat(graph.context, ggml_cast(graph.context, wt, GGML_TYPE_BF16),
                                    ggml_cast(graph.context, xt, GGML_TYPE_BF16));
        const auto result = graph.run(output, {{wt, &weights}, {xt, &values}});
        for (int column = 0; column < columns; ++column) {
            for (int row = 0; row < rows; ++row) {
                double expected = 0;
                for (int k = 0; k < inner; ++k) {
                    expected += round_bf16(weights[row * inner + k]) * round_bf16(values[column * inner + k]);
                }
                near(result[column * rows + row], expected, 0.0002, "BF16 matrix FP64 oracle");
            }
        }
    }
}

void patch_projection() {
    constexpr int patch = 2, width = 4, height = 2, channels = 3, outputs = 2;
    std::vector<float> x(width * height * channels), weights(patch * patch * channels * outputs);
    for (std::size_t i = 0; i < x.size(); ++i) { x[i] = static_cast<float>(70000 + i * 128); }
    for (std::size_t i = 0; i < weights.size(); ++i) { weights[i] = static_cast<float>(int(i % 7) - 3) / 7; }
    Graph graph;
    auto* xt = ggml_new_tensor_4d(graph.context, GGML_TYPE_F32, width, height, channels, 1);
    auto* wt = ggml_new_tensor_4d(graph.context, GGML_TYPE_F32, patch, patch, channels, outputs);
    auto* output = clip_ninfer_cpu_patch_projection(graph.context, wt, clip_ninfer_cpu_bf16(graph.context, xt), patch);
    const auto result = graph.run(output, {{xt, &x}, {wt, &weights}});
    for (int oc = 0; oc < outputs; ++oc) {
        for (int px = 0; px < width / patch; ++px) {
            double expected = 0;
            for (int c = 0; c < channels; ++c) {
                for (int y = 0; y < patch; ++y) {
                    for (int xx = 0; xx < patch; ++xx) {
                        expected += round_bf16(x[(c * height + y) * width + px * patch + xx]) *
                            round_bf16(weights[((oc * channels + c) * patch + y) * patch + xx]);
                    }
                }
            }
            near(result[oc * (width / patch) + px], expected, 0.03, "temporal patch projection FP64 oracle");
        }
    }
}

void positions_and_merge_order() {
    constexpr int hidden = 3, side = 48, width = 6, height = 4;
    const auto coordinate = [](int c, int y, int x) { return c * 100.0 + y * y * 0.25 + x * x * 0.125; };
    std::vector<float> table(hidden * side * side);
    for (int y = 0; y < side; ++y) {
        for (int x = 0; x < side; ++x) {
            for (int c = 0; c < hidden; ++c) { table[(y * side + x) * hidden + c] = static_cast<float>(coordinate(c, y, x)); }
        }
    }
    Graph graph;
    auto* input = ggml_new_tensor_2d(graph.context, GGML_TYPE_F32, hidden, side * side);
    auto* interpolated = clip_ninfer_cpu_position(graph.context, input, hidden, width, height, side);
    auto* output = clip_ninfer_cpu_merge_order(graph.context, interpolated, hidden, width, height);
    const auto result = graph.run(output, {{input, &table}});
    int token = 0;
    for (int by = 0; by < height; by += 2) {
        for (int bx = 0; bx < width; bx += 2) {
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx, ++token) {
                    const double y = double(by + dy) * (side - 1) / (height - 1);
                    const double x = double(bx + dx) * (side - 1) / (width - 1);
                    const int y0 = static_cast<int>(std::floor(y)), x0 = static_cast<int>(std::floor(x));
                    const int y1 = std::min(y0 + 1, side - 1), x1 = std::min(x0 + 1, side - 1);
                    const double ay = y - y0, ax = x - x0;
                    for (int c = 0; c < hidden; ++c) {
                        const double expected = (1 - ay) * ((1 - ax) * coordinate(c, y0, x0) + ax * coordinate(c, y0, x1)) +
                            ay * ((1 - ax) * coordinate(c, y1, x0) + ax * coordinate(c, y1, x1));
                        near(result[token * hidden + c], expected, 0.001, "non-square 48x48 position interpolation/2x2 merger order");
                    }
                }
            }
        }
    }
}

void invalid_files() {
    const auto path = std::filesystem::temp_directory_path() / "ninfer-cpu-vision-invalid.gguf";
    { std::ofstream file(path, std::ios::binary); file << "GGUFbad header"; }
    bool rejected = false;
    try { make_gguf_cpu_vision_encoder(path.string(), 1, std::size_t(2) << 30); }
    catch (const std::invalid_argument&) { rejected = true; }
    std::filesystem::remove(path);
    require(rejected, "malformed GGUF was accepted");
    for (std::uint32_t threads : {0U, 513U}) {
        rejected = false;
        try { make_gguf_cpu_vision_encoder("absent.gguf", threads, std::size_t(2) << 30); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid CPU thread count was accepted");
    }
}

int real_gguf() {
    const char* path = std::getenv("NINFER_TEST_VISION_GGUF");
    if (!path || !*path) {
        std::cout << "SKIP: NINFER_TEST_VISION_GGUF is not set\n";
        return 77;
    }
    bool rejected = false;
    try { make_gguf_cpu_vision_encoder(path, 1, 32 * 1024 * 1024); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "weight allocation was admitted with insufficient budget");
    std::cerr << "real GGUF: loading CPU weights\n";
    auto encoder = make_gguf_cpu_vision_encoder(path, 2, std::size_t(4) << 30);
    std::cerr << "real GGUF: weights loaded\n";
    require(encoder->weight_bytes() >= 931126208 && encoder->identity_hash() != 0, "invalid loaded-weight accounting");
    constexpr int gh = 2, gw = 4;
    std::vector<float> patches(gh * gw * 1536);
    for (int p = 0; p < gh * gw; ++p) {
        for (int c = 0; c < 3; ++c) {
            for (int t = 0; t < 2; ++t) {
                for (int i = 0; i < 256; ++i) {
                    patches[p * 1536 + (c * 2 + t) * 256 + i] = static_cast<float>(std::sin((p * 31 + c * 7 + i) * 0.1));
                }
            }
        }
    }
    const CpuVisionInput input{patches, 1, gh, gw, false};
    std::cerr << "real GGUF: measuring rectangular graph\n";
    encoder->validate(input);
    std::cerr << "real GGUF: graph measured\n";
    rejected = false;
    std::cerr << "real GGUF: pre-cancellation\n";
    try { encoder->encode(input, [] { return true; }); }
    catch (const CpuVisionCancelled&) { rejected = true; }
    require(rejected, "CPU vision cancellation was not typed");
    int cancellation_checks = 0;
    rejected = false;
    std::cerr << "real GGUF: graph cancellation\n";
    try { encoder->encode(input, [&] { return ++cancellation_checks >= 4; }); }
    catch (const CpuVisionCancelled&) { rejected = true; }
    require(rejected && cancellation_checks >= 4, "GGML graph cancellation did not discard partial output");
    std::cerr << "real GGUF: full graph\n";
    const auto output = encoder->encode(input);
    require(output.size() == gh * gw / 4 * 5120, "real GGUF output shape mismatch");
    for (auto value : output) { require((value & 0x7f80U) != 0x7f80U, "nonfinite BF16 embedding"); }
    require(output == encoder->encode(input), "CPU encoding changed on repeated requests");
    // A pair of identical video frames is mathematically the same temporal patch
    // projection as the validated still-image replica. This also exercises 2-frame CHW.
    const auto video = encoder->encode({patches, 1, gh, gw, true});
    require(output == video, "temporal frame packing changed repeated-frame image output");
    auto second_group = patches;
    for (auto& value : second_group) { value = -value; }
    const auto second_output = encoder->encode({second_group, 1, gh, gw, true});
    auto sequence = patches;
    sequence.insert(sequence.end(), second_group.begin(), second_group.end());
    auto sequence_expected = video;
    sequence_expected.insert(sequence_expected.end(), second_output.begin(), second_output.end());
    require(encoder->encode({sequence, 2, gh, gw, true}) == sequence_expected,
        "video temporal groups were reordered or encoded with another group's pixels");
    // Discover the advertised admission boundary from a rejected request, then run
    // twice with exactly that allowance. Retaining a reusable graph must not make
    // an otherwise identical second request consume the allowance twice.
    const auto probe_budget = encoder->weight_bytes() + 34 * 1024 * 1024;
    auto probe = make_gguf_cpu_vision_encoder(path, 2, probe_budget);
    std::size_t tight_budget = 0;
    try { probe->validate(input); }
    catch (const std::runtime_error& error) {
        const std::string message = error.what();
        const auto start = message.find("requires ");
        require(start != std::string::npos, "budget probe failed for an unrelated reason");
        tight_budget = std::stoull(message.substr(start + 9));
    }
    require(tight_budget > probe_budget, "budget probe did not reject the graph before allocation");
    probe.reset();
    auto tight = make_gguf_cpu_vision_encoder(path, 2, tight_budget);
    require(tight->encode(input) == output, "tight-budget first encoding changed output");
    require(tight->encode(input) == output, "tight-budget repeated encoding changed output");
    std::cout << "Real BF16 GGUF rectangular-image encoding passed; weights=" << encoder->weight_bytes() << " bytes\n";
    return 0;
}

int real_gguf_cuda() {
#ifndef NINFER_GGML_CUDA_VISION
    std::cout << "SKIP: GGML CUDA vision was not built\n";
    return 77;
#else
    const char* path = std::getenv("NINFER_TEST_VISION_GGUF");
    if (!path || !*path) {
        std::cout << "SKIP: NINFER_TEST_VISION_GGUF is not set\n";
        return 77;
    }
    auto encoder = make_gguf_cuda_vision_encoder(path, std::size_t(4) << 30);
    require(encoder->weight_bytes() >= 931126208, "GGML CUDA weights were not loaded");
    constexpr int gh = 2, gw = 4;
    std::vector<float> patches(gh * gw * 1536);
    for (int patch = 0; patch < gh * gw; ++patch) {
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
    const CpuVisionInput input{patches, 1, gh, gw, false};
    encoder->validate(input);
    const auto output = encoder->encode(input);
    require(output.size() == gh * gw / 4 * 5120, "GGML CUDA output shape differs");
    for (const auto value : output) {
        require((value & 0x7f80U) != 0x7f80U, "GGML CUDA output contains nonfinite BF16");
    }
    require(output == encoder->encode(input), "GGML CUDA repeated encode differs");
    std::cout << "GGML CUDA GGUF image encoding passed; weights=" << encoder->weight_bytes() << " bytes\n";
    return 0;
#endif
}
#endif
} // namespace

int main(int argc, char** argv) {
    try {
        const std::string_view mode = argc > 1 ? argv[1] : "--numerics";
#ifdef NINFER_TEST_GGML_CPU_VISION
        require(cpu_vision_backend_available(), "CPU backend build is not available");
        if (mode == "--gguf") { return real_gguf(); }
        if (mode == "--gguf-cuda") { return real_gguf_cuda(); }
        require(mode == "--numerics", "unknown test mode");
        require(ggml_cpu_has_llamafile(), "CPU vision build omitted llamafile kernels");
        auto* registry = ggml_backend_cpu_reg();
        const auto get_features = reinterpret_cast<ggml_backend_get_features_t>(
            ggml_backend_reg_get_proc_address(registry, "ggml_backend_get_features"));
        require(get_features != nullptr, "CPU backend omitted feature inspection");
        bool openmp = false;
        for (auto* feature = get_features(registry); feature->name; ++feature) {
            if (std::string_view(feature->name) == "OPENMP" && std::string_view(feature->value) == "1") { openmp = true; }
        }
        require(openmp, "CPU vision build omitted OpenMP support");
        gelu(); attention(); bf16_rounding(); matrix_product(); patch_projection(); positions_and_merge_order(); invalid_files();
        std::cout << "GGML CPU vision FP64 numerical and admission checks passed\n";
#else
        require(mode == "--stub", "CPU vision test was built without the GGML test definition");
        require(!cpu_vision_backend_available(), "stub backend claims availability");
        bool rejected = false;
        try { make_gguf_cpu_vision_encoder("absent.gguf", 1, 1); }
        catch (const std::runtime_error& e) { rejected = std::string_view(e.what()).find("not built") != std::string_view::npos; }
        require(rejected, "stub did not explain build requirement");
#endif
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

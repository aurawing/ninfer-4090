#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/gqa_attention_partial.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace ninfer;

namespace {
template <typename T>
DeviceBuffer upload(const std::vector<T>& values) {
    DeviceBuffer buffer(values.size() * sizeof(T));
    if (!values.empty()) buffer.copy_from_host(values.data(), buffer.bytes);
    return buffer;
}

struct Events {
    cudaEvent_t begin{}, end{};

    Events() {
        CUDA_CHECK(cudaEventCreate(&begin));
        CUDA_CHECK(cudaEventCreate(&end));
    }

    ~Events() {
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }
};

template <typename F>
float measure(F operation, Events& events, int repeats) {
    for (int i = 0; i < 5; ++i) operation();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(events.begin));
    for (int i = 0; i < repeats; ++i) operation();
    CUDA_CHECK(cudaEventRecord(events.end));
    CUDA_CHECK(cudaEventSynchronize(events.end));
    float elapsed = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, events.begin, events.end));
    return elapsed / repeats;
}

std::uint16_t bf16(float x) {
    std::uint32_t raw;
    std::memcpy(&raw, &x, 4);
    return std::uint16_t((raw + 0x7fff + ((raw >> 16) & 1)) >> 16);
}

struct Parts {
    DeviceBuffer o, m, l;
    ops::AttentionPartial value;

    Parts(int t, int s)
        : o(std::size_t(256) * 24 * t * s * 4), m(std::size_t(24) * t * s * 4), l(m.bytes),
          value{Tensor(o.p, DType::FP32, {256, 24, t, s}), Tensor(m.p, DType::FP32, {24, t, s}),
                Tensor(l.p, DType::FP32, {24, t, s})} {}

    std::size_t bytes() const { return o.bytes + m.bytes + l.bytes; }
};

struct Options {
    std::string format = "all", profile;
    int tokens = 0, splits = 0, repeats = 20, passes = 1;
};

Options options(int argc, char** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 == argc) throw std::invalid_argument("benchmark option needs value");
        const std::string key = argv[i], value = argv[++i];
        if (key == "--format")
            result.format = value;
        else if (key == "--tokens")
            result.tokens = std::stoi(value);
        else if (key == "--splits")
            result.splits = std::stoi(value);
        else if (key == "--repeats")
            result.repeats = std::stoi(value);
        else if (key == "--passes")
            result.passes = std::stoi(value);
        else if (key == "--profile")
            result.profile = value;
        else
            throw std::invalid_argument("unknown benchmark option: " + key);
    }
    if ((result.format != "all" && result.format != "int8" && result.format != "rk4v4-e8" &&
         result.format != "bf16") ||
        (!result.profile.empty() && result.profile != "dense" && result.profile != "partial") ||
        result.tokens < 0 || result.tokens > 2048 || result.splits < 0 || result.splits > 512 ||
        result.repeats < 1 || result.repeats > 1000 || result.passes < 1 || result.passes > 64 ||
        (!result.profile.empty() && (result.format == "all" || !result.tokens || !result.splits)))
        throw std::invalid_argument("benchmark option outside domain");
    return result;
}

void run(const Options& opt, const std::string& format) {
    constexpr int keys = 131072, pages = keys / 64;
    const bool packed = format == "rk4v4-e8", plain = format == "bf16";
    const int dimension        = packed ? 128 : 256;
    const auto dtype           = plain ? DType::BF16 : (packed ? DType::U8 : DType::I8);
    const std::size_t elements = std::size_t(dimension) * 4 * keys;
    std::uint32_t random       = 719;
    auto next                  = [&] {
        random = random * 1664525u + 1013904223u;
        return random >> 16;
    };
    std::vector<std::uint8_t> codes(elements * (plain ? 2 : 1));
    if (plain) {
        for (std::size_t i = 0; i < elements; ++i) {
            const auto bits = bf16(float(int(next() % 255) - 127) / 128);
            std::memcpy(codes.data() + 2 * i, &bits, 2);
        }
    } else {
        for (auto& value : codes)
            value =
                packed ? std::uint8_t(next()) : std::uint8_t(std::int8_t(int(next() % 255) - 127));
    }
    auto k = upload(codes);
    if (plain) {
        for (std::size_t i = 0; i < elements; ++i) {
            const auto bits = bf16(float(int(next() % 255) - 127) / 128);
            std::memcpy(codes.data() + 2 * i, &bits, 2);
        }
    } else
        std::reverse(codes.begin(), codes.end());
    auto v = upload(codes);
    std::vector<std::uint16_t> scales(plain ? 0 : std::size_t(4) * 4 * keys);
    for (std::size_t i = 0; i < scales.size(); ++i)
        scales[i] = std::uint16_t((packed ? 0x2800 : 0x2400) + int(i % 3) * 1024);
    auto ks = upload(scales);
    std::reverse(scales.begin(), scales.end());
    auto vs = upload(scales);
    std::vector<std::int32_t> table(pages);
    std::iota(table.begin(), table.end(), 0);
    auto dt = upload(table);
    PagedKVLayerView stage;
    stage.k_pages = Tensor(k.p, dtype, {dimension, 64, 4, pages});
    stage.v_pages = Tensor(v.p, dtype, {dimension, 64, 4, pages});
    if (!plain) {
        stage.k_scale_pages = Tensor(ks.p, DType::FP16, {4, 64, 4, pages});
        stage.v_scale_pages = Tensor(vs.p, DType::FP16, {4, 64, 4, pages});
    }
    stage.block_table  = Tensor(dt.p, DType::I32, {pages});
    stage.dtype        = plain ? DType::BF16 : DType::I8;
    stage.head_dim     = 256;
    stage.num_kv_heads = 4;
    stage.quant_group  = plain ? 0 : 64;
    stage.packed_k = stage.packed_v = stage.rotate_k = stage.rotate_v = stage.e8_lattice = packed;
    auto resident                                                                        = stage;
    resident.k_pages                                                                     = {};
    resident.v_pages                                                                     = {};
    resident.k_scale_pages                                                               = {};
    resident.v_scale_pages                                                               = {};
    resident.block_table                                                                 = {};
    std::vector<ops::AttentionPageAccess> accesses;
    for (int p = 0; p < pages; ++p) accesses.push_back({p, p});
    std::vector<DeviceBuffer> pass_access, pass_prefix;
    std::vector<Tensor> access_tensor, prefix_tensor;
    for (int pass = 0; pass < opt.passes; ++pass) {
        const int begin = pages * pass / opt.passes, end = pages * (pass + 1) / opt.passes;
        std::vector<ops::AttentionPageAccess> subset(accesses.begin() + begin,
                                                     accesses.begin() + end);
        auto count = ops::attention_access_prefix(subset, keys, 0, pages);
        pass_access.push_back(upload(subset));
        pass_prefix.push_back(upload(count));
        access_tensor.emplace_back(pass_access.back().p, DType::I32,
                                   std::initializer_list<int>{2, end - begin});
        prefix_tensor.emplace_back(pass_prefix.back().p, DType::I32,
                                   std::initializer_list<int>{end - begin + 1});
    }
    std::cout << "# " << format << " layer_bytes=" << k.bytes + v.bytes + ks.bytes + vs.bytes
              << " keys=" << keys << " all pages in staging; compute only; 5 warmups / "
              << opt.repeats << " repeats; passes=" << opt.passes << "\n";
    Events events;
    const auto widths =
        opt.tokens ? std::vector<int>{opt.tokens} : std::vector<int>{1, 4, 64, 1024, 2048};
    const auto partitions =
        opt.splits ? std::vector<int>{opt.splits} : std::vector<int>{1, 2, 4, 8, 16, 32};
    for (int t : widths) {
        std::vector<std::uint16_t> q(std::size_t(256) * 24 * t);
        for (auto& bits : q) bits = bf16(float(int(next() % 129) - 64) / 256);
        auto dq = upload(q);
        std::vector<std::int32_t> positions(t);
        std::iota(positions.begin(), positions.end(), keys - t);
        auto dp = upload(positions);
        Tensor tq(dq.p, DType::BF16, {256, 24, t}), tp(dp.p, DType::I32, {t});
        DeviceBuffer result(q.size() * 2);
        Tensor out(result.p, DType::BF16, {256, 24, t});
        const ops::GqaExecutionEnvelope envelope{keys, keys};
        const auto scratch_bytes =
            ops::gqa_attention_workspace_capacity_bytes(24, stage.dtype, envelope, 1, t, t);
        DeviceBuffer scratch(std::max<std::size_t>(256, scratch_bytes));
        WorkspaceArena workspace({scratch.p, scratch.bytes});
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto dense = [&] {
            ops::gqa_attention_cached(tq, tp, 0.0625f, stage, envelope, workspace, out, nullptr);
        };
        if (opt.profile == "dense") {
            dense();
            CUDA_CHECK(cudaDeviceSynchronize());
            return;
        }
        for (int s : partitions) {
            Parts partial(t, s), merged(t, 1);
            const auto invoke =
                t <= 16 ? ops::gqa_attention_partial_decode : ops::gqa_attention_partial_prefill;
            const auto compute = [&] {
                for (int pass = 0; pass < opt.passes; ++pass) {
                    invoke(tq, tp, 0.0625f, resident, stage, access_tensor[pass],
                           prefix_tensor[pass], keys, s, partial.value, nullptr);
                    if (opt.passes > 1)
                        ops::attention_partial_lse_accumulate(
                            partial.value, merged.value, pass == 0, nullptr,
                            pass + 1 == opt.passes ? &out : nullptr);
                }
            };
            const auto merge = [&] {
                if (opt.passes == 1)
                    ops::attention_partial_lse_accumulate(partial.value, merged.value, true,
                                                          nullptr, &out);
            };
            const auto all = [&] {
                compute();
                merge();
            };
            if (opt.profile == "partial") {
                all();
                CUDA_CHECK(cudaDeviceSynchronize());
                return;
            }
            const float dense_before = measure(dense, events, opt.repeats);
            // With multiple passes compute includes all fixed-order folds and
            // finalization; merge is empty. total_ms always times the whole path.
            const float partial_ms  = measure(compute, events, opt.repeats);
            const float merge_ms    = measure(merge, events, opt.repeats);
            const float total_ms    = measure(all, events, opt.repeats);
            const float dense_after = measure(dense, events, opt.repeats);
            const float dense_ms    = (dense_before + dense_after) * 0.5f;
            std::cout << format << ',' << t << ',' << (t <= 16 ? "decode" : "prefill") << ',' << s
                      << ',' << dense_ms << ',' << partial_ms << ',' << merge_ms << ',' << total_ms
                      << ',' << total_ms / dense_ms << ',' << partial.bytes() + merged.bytes()
                      << ',' << dense_before << ',' << dense_after << ',' << opt.passes << '\n'
                      << std::flush;
        }
    }
}
} // namespace
int main(int argc, char** argv) try {
    const auto opt = options(argc, argv);
    std::cout
        << "format,T,route,splits,dense_ms,compute_or_fold_ms,terminal_merge_ms,total_ms,ratio,"
           "workspace_bytes,dense_before_ms,dense_after_ms,passes\n";
    for (const std::string format : {"int8", "rk4v4-e8", "bf16"})
        if (opt.format == "all" || opt.format == format) run(opt, format);
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}

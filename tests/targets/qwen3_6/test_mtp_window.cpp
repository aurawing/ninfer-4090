#include "targets/qwen3_6/impl/runtime/mtp_window.h"
#include "core/device.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <memory>
#include <cuda_fp16.h>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace ninfer;
using namespace ninfer::targets::qwen3_6;
using namespace ninfer::targets::qwen3_6::detail;
namespace {
void require(bool v, const char* message) {
    if (!v) throw std::runtime_error(message);
}
std::uint16_t bf16(float v) {
    auto n = std::bit_cast<std::uint32_t>(v);
    return std::uint16_t((n + 0x7fff + ((n >> 16) & 1)) >> 16);
}
float unpack(std::uint16_t v) { return std::bit_cast<float>(std::uint32_t(v) << 16); }
float token_value(int position) { return float((position / 64) % 8 + 1) * 0.125f; }
struct Fixture {
    DeviceContext device;
    TieredKVOptions options;
    MtpWindowPlan plan;
    std::unique_ptr<DeviceBuffer> cache_memory, runtime_memory;
    std::unique_ptr<PagedKVCache> cache;
    PagedKVAllocation lease;
    std::unique_ptr<MtpWindow> owner;
    DeviceBuffer q{256 * 24 * 4 * 2}, k{256 * 4 * 128 * 2}, v{k.bytes}, pos{128 * 4}, out{q.bytes},
        valid{4};
    DType dtype;
    bool packed;
    explicit Fixture(DType type, bool packed_profile = false)
        : dtype(type), packed(packed_profile) {
        options.sink_tokens       = 64;
        options.mtp_window_tokens = 512;
        plan                      = plan_mtp_window(131072, 128, options);
        LayoutBuilder builder;
        DecoderStateSpec spec;
        spec.capacity              = plan.capacity;
        spec.full_attention_layers = 1;
        spec.mtp_layers            = 1;
        spec.kv_heads              = 4;
        spec.attention_head_dim    = 256;
        spec.kv_dtype              = type;
        spec.kv_quant_group        = type == DType::I8 ? 64 : 0;
        spec.kv_packed_k = spec.kv_packed_v = spec.kv_rotate_k = spec.kv_rotate_v =
            spec.kv_e8_lattice                                 = packed;
        spec.text_physical_page_groups                         = 1;
        spec.mtp_physical_page_groups                          = plan.physical_pages;
        spec.allow_tiered_text_pages                           = true;
        spec.allow_tiered_mtp_pages                            = true;
        spec.enable_mtp                                        = true;
        spec.linear_attention                                  = {1, 1, 1, 1, 1, 1};
        const auto layout                                      = plan_decoder_state(builder, spec);
        cache_memory = std::make_unique<DeviceBuffer>(builder.finish(256));
        cache = std::make_unique<PagedKVCache>(DeviceSpan{cache_memory->p, cache_memory->bytes},
                                               *layout.mtp_kv);
        CUDA_CHECK(cudaDeviceSynchronize());
        // Whole-plane byte comparisons include unoccupied slots and page tails.
        // Give those test bytes a defined initial value before any D2H readback.
        CUDA_CHECK(cudaMemsetAsync(cache_memory->p, 0, cache_memory->bytes, device.stream));
        lease = cache->pool().reserve(plan.physical_pages);
        lease.materialize_pages(plan.physical_pages, device.stream);
        runtime_memory = std::make_unique<DeviceBuffer>(plan.bytes);
        owner          = std::make_unique<MtpWindow>(
            plan, *cache, DeviceSpan{runtime_memory->p, runtime_memory->bytes}, device.stream);
        owner->bind_pages(lease.page_ids(), device.stream);
        CUDA_CHECK(cudaMemsetAsync(q.p, 0, q.bytes, device.stream));
        CUDA_CHECK(cudaMemsetAsync(k.p, 0, k.bytes, device.stream));
        device.synchronize();
    }
    void inputs(int base, int count) {
        std::vector<int> positions(count);
        std::vector<std::uint16_t> values(256 * 4 * count);
        for (int j = 0; j < count; ++j) {
            positions[j] = base + j;
            std::fill_n(values.begin() + j * 256 * 4, 256 * 4, bf16(token_value(base + j)));
        }
        CUDA_CHECK(cudaMemcpyAsync(pos.p, positions.data(), count * 4, cudaMemcpyHostToDevice,
                                   device.stream));
        CUDA_CHECK(cudaMemcpyAsync(v.p, values.data(), values.size() * 2, cudaMemcpyHostToDevice,
                                   device.stream));
        device.synchronize();
    }
    void append(int base, int count, int live = -1) {
        inputs(base, count);
        Tensor kt{k.p, DType::BF16, {256, 4, count}}, vt{v.p, DType::BF16, {256, 4, count}},
            pt{pos.p, DType::I32, {count}};
        Tensor mask;
        if (live >= 0) {
            CUDA_CHECK(cudaMemcpyAsync(valid.p, &live, 4, cudaMemcpyHostToDevice, device.stream));
            mask = Tensor{valid.p, DType::I32, {1}};
        }
        owner->append(kt, vt, pt, mask, device.stream);
        device.synchronize();
    }
    std::vector<std::byte> bytes() {
        auto plane = cache->batch_layer_view(0).v_pages;
        std::vector<std::byte> result(plane.bytes());
        CUDA_CHECK(cudaMemcpyAsync(result.data(), plane.data, plane.bytes(), cudaMemcpyDeviceToHost,
                                   device.stream));
        device.synchronize();
        return result;
    }
    std::vector<std::vector<std::byte>> all_bytes() {
        std::vector<std::vector<std::byte>> result;
        for (std::size_t i = 0; i < cache->pool().plane_count(); ++i) {
            const auto& plane = cache->pool().plane(i);
            result.emplace_back(plane.bytes());
            CUDA_CHECK(cudaMemcpyAsync(result.back().data(), plane.data, plane.bytes(),
                                       cudaMemcpyDeviceToHost, device.stream));
        }
        device.synchronize();
        return result;
    }
    std::vector<int> tags() {
        std::vector<int> result(plan.physical_pages);
        CUDA_CHECK(cudaMemcpyAsync(result.data(), owner->page_tags().data,
                                   owner->page_tags().bytes(), cudaMemcpyDeviceToHost,
                                   device.stream));
        device.synchronize();
        return result;
    }
    double oracle(int position) {
        // Independent FP64 uniform attention over tagged original logical pages.
        // Decode the observable cache storage, without reusing an attention implementation.
        const auto layer = cache->batch_layer_view(0);
        auto codes       = bytes();
        std::vector<std::uint16_t> scales(layer.v_scale_pages.data ? layer.v_scale_pages.bytes() / 2
                                                                   : 0);
        std::vector<int> tags(plan.physical_pages);
        if (!scales.empty())
            CUDA_CHECK(cudaMemcpyAsync(scales.data(), layer.v_scale_pages.data,
                                       layer.v_scale_pages.bytes(), cudaMemcpyDeviceToHost,
                                       device.stream));
        CUDA_CHECK(cudaMemcpyAsync(tags.data(), owner->page_tags().data, owner->page_tags().bytes(),
                                   cudaMemcpyDeviceToHost, device.stream));
        device.synchronize();
        const int end   = position / 64 + 1,
                  first = std::max(int(plan.sink_pages), end - int(plan.recent_pages));
        double sum      = 0;
        int count       = 0;
        for (int logical = 0; logical < end; ++logical) {
            if (logical >= int(plan.sink_pages) && logical < first) continue;
            const int slot =
                logical < int(plan.sink_pages)
                    ? logical
                    : plan.sink_pages + (logical - plan.sink_pages) % plan.recent_pages;
            const int physical = lease.page_ids()[slot];
            if (tags[physical] != logical) continue;
            for (int offset = 0; offset < 64 && logical * 64 + offset <= position; ++offset) {
                const auto address =
                    std::size_t(physical) * layer.v_pages.nb[3] + offset * layer.v_pages.nb[1];
                if (dtype == DType::BF16) {
                    std::uint16_t bits;
                    std::memcpy(&bits, codes.data() + address, 2);
                    sum += unpack(bits);
                } else {
                    const auto scale_byte = std::size_t(physical) * layer.v_scale_pages.nb[3] +
                                            offset * layer.v_scale_pages.nb[1];
                    const auto scale = __half2float(std::bit_cast<__half>(scales[scale_byte / 2]));
                    if (packed) {
                        double group_sum = 0;
                        for (int d = 0; d < 64; ++d) {
                            const auto byte  = std::uint8_t(codes[address + d / 2]);
                            const int nibble = (byte >> (4 * (d % 2))) & 15;
                            group_sum +=
                                (nibble >= 8 ? nibble - 16 : nibble) * double(scale) * 0.125;
                        }
                        sum += group_sum;
                    } else
                        sum += double(std::int8_t(codes[address])) * scale;
                }
                ++count;
            }
        }
        return count ? sum / count : 0;
    }
    void query(int base, int width, int live = -1, bool writes = false) {
        inputs(base, width);
        Tensor qt{q.p, DType::BF16, {256, 24, width}}, pt{pos.p, DType::I32, {width}},
            ot{out.p, DType::BF16, {256, 24, width}};
        Tensor kt{k.p, DType::BF16, {256, 4, width}}, vt{v.p, DType::BF16, {256, 4, width}}, mask;
        if (live >= 0) {
            CUDA_CHECK(cudaMemcpyAsync(valid.p, &live, 4, cudaMemcpyHostToDevice, device.stream));
            mask = Tensor{valid.p, DType::I32, {1}};
        }
        owner->attention(qt, pt, mask, 0.0625f, ot, device.stream, writes ? &kt : nullptr,
                         writes ? &vt : nullptr);
        std::vector<std::uint16_t> actual(256 * 24 * width);
        CUDA_CHECK(cudaMemcpyAsync(actual.data(), out.p, actual.size() * 2, cudaMemcpyDeviceToHost,
                                   device.stream));
        device.synchronize();
        for (int j = 0; j < width; ++j) {
            const auto reference = live >= 0 && j >= live ? 0 : oracle(base + j);
            const auto rounded   = unpack(bf16(float(reference)));
            for (int i = 0; i < 256 * 24; ++i)
                require(std::abs(unpack(actual[j * 256 * 24 + i]) - reference) <=
                            std::max(0.000001, std::abs(double(rounded) - reference) + 0.00001),
                        "MTP window output violates FP64 uniform-mean BF16 rounding oracle");
        }
    }
};
void masked_append_preserves_bytes(DType type, bool packed = false) {
    Fixture f(type, packed);
    f.append(64, 64);
    const auto before      = f.all_bytes();
    const auto tags_before = f.tags();
    f.inputs(66, 4);
    std::vector<std::uint16_t> changed(256 * 4 * 4, bf16(0.75f));
    CUDA_CHECK(cudaMemcpyAsync(f.v.p, changed.data(), changed.size() * 2, cudaMemcpyHostToDevice,
                               f.device.stream));
    int count = 2;
    CUDA_CHECK(cudaMemcpyAsync(f.valid.p, &count, 4, cudaMemcpyHostToDevice, f.device.stream));
    Tensor kt{f.k.p, DType::BF16, {256, 4, 4}}, vt{f.v.p, DType::BF16, {256, 4, 4}},
        pt{f.pos.p, DType::I32, {4}}, mask{f.valid.p, DType::I32, {1}};
    f.owner->append(kt, vt, pt, mask, f.device.stream);
    const auto after    = f.all_bytes();
    const auto plane    = f.cache->batch_layer_view(0).v_pages;
    const auto physical = f.lease.page_ids()[f.plan.sink_pages];
    for (std::size_t p = 0; p < before.size(); ++p) {
        const auto& plane = f.cache->pool().plane(p);
        for (int token = 4; token < 6; ++token)
            for (int head = 0; head < 4; ++head) {
                const auto offset =
                    physical * plane.nb[3] + head * plane.nb[2] + token * plane.nb[1];
                require(std::equal(before[p].begin() + offset,
                                   before[p].begin() + offset + plane.nb[1],
                                   after[p].begin() + offset),
                        "masked MTP append changed invalid-column KV code or scale bytes");
            }
    }
    require(tags_before == f.tags(), "masked append changed unrelated physical-page tags");
    f.owner->reset(f.device.stream);
    f.append(0, 128);
    const auto crossing_tags = f.tags();
    f.append(127, 4, 1);
    require(crossing_tags == f.tags(),
            "masked append published invalid columns' next logical page");
}
void ring_and_provisional(DType type, bool packed = false) {
    Fixture f(type, packed);
    f.append(0, 64);
    f.query(0, 1);
    f.query(3, 2);
    f.query(7, 4, 2);
    for (int base = 64; base < 448; base += 64)
        f.append(base, 64);
    const auto before      = f.all_bytes();
    const auto tags_before = f.tags();
    f.owner->begin_transaction(447, 3, f.device.stream);
    f.query(447, 1, 1, true);
    f.query(448, 2, 2, true);
    f.owner->trim(448, f.device.stream);
    f.device.synchronize();
    const auto after  = f.all_bytes();
    const auto plane  = f.cache->batch_layer_view(0).v_pages;
    const auto oldest = f.lease.page_ids()[f.plan.sink_pages];
    for (std::size_t p = 0; p < before.size(); ++p) {
        const auto& plane = f.cache->pool().plane(p);
        require(std::equal(before[p].begin() + oldest * plane.nb[3],
                           before[p].begin() + (oldest + 1) * plane.nb[3],
                           after[p].begin() + oldest * plane.nb[3]),
                "trim failed to restore every wrapped provisional code and scale plane");
    }
    require(f.tags()[oldest] == tags_before[oldest],
            "trim failed to restore original logical-page tag");
    f.query(447, 1);
    // Far original positions and an intentional missing history page must use
    // only live tags, never read aliased old bytes or recompute missing KV.
    const int first = 1000 * 64;
    for (int base = first; base < first + 6 * 64; base += 64)
        f.append(base, 64);
    f.query(first + 6 * 64 - 1, 1);
    f.owner->reset(f.device.stream);
    f.append(0, 64);
    f.query(20, 4, 2, true);
}
} // namespace
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        for (auto type : {DType::BF16, DType::I8}) {
            masked_append_preserves_bytes(type);
            ring_and_provisional(type);
        }
        masked_append_preserves_bytes(DType::I8, true);
        ring_and_provisional(DType::I8, true);
        std::cout << "MTP sink/recent/guard, far positions, masks and FP64 oracle passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

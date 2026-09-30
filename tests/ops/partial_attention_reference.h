#pragma once
#include "attention_reference.h"
#include "ninfer/ops/gqa_attention_partial.h"

namespace ninfer::test::partial_attention_reference {
using namespace ninfer::test::attention_reference;
struct Parts {
    DeviceBuffer o, m, l;
    ops::AttentionPartial tensors;
    Parts(int tokens, int parts)
        : o(std::size_t(256) * 24 * tokens * parts * 4), m(std::size_t(24) * tokens * parts * 4),
          l(m.bytes), tensors{Tensor(o.p, DType::FP32, {256, 24, tokens, parts}),
                              Tensor(m.p, DType::FP32, {24, tokens, parts}),
                              Tensor(l.p, DType::FP32, {24, tokens, parts})} {}
};
inline int run(Format format, int tokens, int splits, bool prefill, int scenario = 0,
               bool multipass = false, int pool_mode = 0) {
    EncodedCache host(format, 512);
    // Distinguish all physical pages, heads, groups, offsets and K/V planes.
    // Exact powers of two commute with H64 within each 64-element group.
    if (format != Format::Bf16)
        for (int key = 0; key < host.keys; ++key)
            for (int h = 0; h < 4; ++h)
                for (int which = 0; which < 2; ++which)
                    for (int g = 0; g < 4; ++g) {
                        const int shift = (key / 64 + h + g + key % 3 + which) % 3 - 1;
                        auto& scales = which ? host.vs : host.ks;
                        auto& values = which ? host.decoded_v : host.decoded_k;
                        scales[page_index(4, key / 64, h, key % 64, g)] = std::uint16_t(
                            (format == Format::Int8 ? 0x2400 : 0x2800) + shift * 1024);
                        for (int d = g * 64; d < (g + 1) * 64; ++d)
                            values[d + 256 * (h + 4 * key)] *= std::ldexp(1.0, shift);
                    }
    auto dk = to_device(host.k), dv = to_device(host.v), dks = to_device(host.ks),
         dvs = to_device(host.vs);
    auto view = [&](int first, int count) {
        const bool bf16 = format == Format::Bf16, packed = format == Format::Rk4v4E8;
        const auto bytes = std::size_t(host.code_dim) * 64 * 4 * (bf16 ? 2 : 1);
        PagedKVLayerView v;
        if (count) {
            v.k_pages = Tensor(static_cast<std::byte*>(dk.p) + first * bytes,
                               bf16 ? DType::BF16 : (packed ? DType::U8 : DType::I8),
                               {host.code_dim, 64, 4, count});
            v.v_pages = Tensor(static_cast<std::byte*>(dv.p) + first * bytes, v.k_pages.dtype,
                               {host.code_dim, 64, 4, count});
            v.k_scale_pages = Tensor(static_cast<std::uint16_t*>(dks.p) + first * 64 * 4 * 4,
                                     DType::FP16, {4, 64, 4, count});
            v.v_scale_pages = Tensor(static_cast<std::uint16_t*>(dvs.p) + first * 64 * 4 * 4,
                                     DType::FP16, {4, 64, 4, count});
        }
        v.head_dim = 256;
        v.num_kv_heads = 4;
        v.dtype = bf16 ? DType::BF16 : DType::I8;
        v.quant_group = bf16 ? 0 : 64;
        v.packed_k = v.packed_v = v.rotate_k = v.rotate_v = v.e8_lattice = packed;
        return v;
    };
    const int resident_count = pool_mode == 1 ? 8 : (pool_mode == 2 ? 0 : 4);
    auto resident = view(0, resident_count), staging = view(resident_count, 8 - resident_count);
    std::vector<ops::AttentionPageAccess> pages{{0, 3}, {2, 7}, {100, 0}, {2998, 6}, {3000, 4}};
    if (scenario == 2)
        pages.clear();
    if (scenario == 3)
        pages = {{3000, 4}};
    constexpr int frontier = 192013;
    auto prefix = ops::attention_access_prefix(pages, frontier, resident_count, 8 - resident_count);
    auto da = to_device(pages), df = to_device(prefix);
    const auto q = queries(tokens);
    auto dq = to_device(q);
    std::vector<std::int32_t> positions(tokens);
    std::iota(positions.begin(), positions.end(), frontier - tokens);
    if (scenario == 1) {
        const int boundaries[]{0,   1,   31,   62,   63,     64,     65,     127,
                               128, 129, 6399, 6400, 191872, 192000, 192004, 192012};
        if (tokens != 16)
            throw std::invalid_argument("boundary fixture needs T=16");
        std::copy(std::begin(boundaries), std::end(boundaries), positions.begin());
    }
    if (scenario == 3)
        std::iota(positions.begin(), positions.end(), 0);
    auto dp = to_device(positions);
    Tensor tq(dq.p, DType::BF16, {256, 24, tokens}), tp(dp.p, DType::I32, {tokens});
    Tensor ta = pages.empty() ? Tensor{} : Tensor(da.p, DType::I32, {2, int(pages.size())});
    Tensor tf(df.p, DType::I32, {int(prefix.size())});
    std::vector<int> logical_positions(512, std::numeric_limits<int>::max());
    for (auto page : pages)
        for (int j = 0; j < 64; ++j) {
            const int pos = page.logical_page * 64 + j;
            if (pos < frontier)
                logical_positions[page.physical_page * 64 + j] = pos;
        }
    const auto expected = oracle(host, q, positions, {}, logical_positions);
    Parts partial(tokens, splits), merged(tokens, 1);
    DeviceBuffer output(q.size() * 2);
    Tensor out(output.p, DType::BF16, {256, 24, tokens});
    auto invoke = prefill ? ops::gqa_attention_partial_prefill : ops::gqa_attention_partial_decode;
    std::vector<std::uint16_t> first, actual(q.size());
    std::vector<float> first_o, first_m, first_l;
    for (int repeat = 0; repeat < 2; ++repeat) {
        partial.o.fill(255);
        partial.m.fill(255);
        partial.l.fill(255);
        invoke(tq, tp, 0.0625f, resident, staging, ta, tf, frontier, splits, partial.tensors,
               nullptr);
        ops::attention_partial_lse_merge(partial.tensors, merged.tensors, nullptr);
        ops::attention_partial_finalize(merged.tensors, out, nullptr);
        cuda_synchronize();
        output.copy_to_host(actual.data(), output.bytes);
        std::vector<float> a(partial.o.bytes / 4), b(partial.m.bytes / 4), c(partial.l.bytes / 4);
        partial.o.copy_to_host(a.data(), partial.o.bytes);
        partial.m.copy_to_host(b.data(), partial.m.bytes);
        partial.l.copy_to_host(c.data(), partial.l.bytes);
        if (std::any_of(a.begin(), a.end(), [](float x) { return !std::isfinite(x); }) ||
            std::any_of(b.begin(), b.end(),
                        [](float x) {
                            return !std::isfinite(x) &&
                                   x != -std::numeric_limits<float>::infinity();
                        }) ||
            std::any_of(c.begin(), c.end(), [](float x) { return !std::isfinite(x) || x < 0; })) {
            std::cerr << "FAIL non-finite/unwritten partial state\n";
            return 1;
        }
        if (!repeat) {
            first = actual;
            first_o = a;
            first_m = b;
            first_l = c;
        } else if (first != actual || std::memcmp(first_o.data(), a.data(), partial.o.bytes) ||
                   std::memcmp(first_m.data(), b.data(), partial.m.bytes) ||
                   std::memcmp(first_l.data(), c.data(), partial.l.bytes)) {
            std::cerr << "FAIL partial repeat mismatch\n";
            return 1;
        }
    }
    if (from_device<std::uint8_t>(dk, host.k.size()) != host.k ||
        from_device<std::uint8_t>(dv, host.v.size()) != host.v ||
        from_device<std::uint16_t>(dks, host.ks.size()) != host.ks ||
        from_device<std::uint16_t>(dvs, host.vs.size()) != host.vs) {
        std::cerr << "FAIL partial attention changed cache bytes\n";
        return 1;
    }
    std::vector<double> got(actual.size());
    std::transform(actual.begin(), actual.end(), got.begin(),
                   [](auto x) { return double(bf16_to_f32(x)); });
    const auto label = std::string("tiered ") + std::to_string(int(format)) +
                       " T=" + std::to_string(tokens) + " splits=" + std::to_string(splits) +
                       (prefill ? " prefill" : " decode");
    int failures = verify_reduction((label + " scenario=" + std::to_string(scenario)).c_str(), got,
                                    expected, {1.0 / 256, 1.1e-3, 3.9e-3});
    if (multipass) {
        Parts pieces(tokens, splits * 3), streamed(tokens, 1);
        std::vector<float> combined_first;
        for (int repeat = 0; repeat < 2; ++repeat) {
            for (int pass = 0; pass < 3; ++pass) {
                std::vector<ops::AttentionPageAccess> subset;
                for (std::size_t i = pass; i < pages.size(); i += 3)
                    subset.push_back(pages[i]);
                auto count = ops::attention_access_prefix(subset, frontier, resident_count,
                                                          8 - resident_count);
                auto ds = to_device(subset), dc = to_device(count);
                Tensor ts =
                    subset.empty() ? Tensor{} : Tensor(ds.p, DType::I32, {2, int(subset.size())});
                Tensor tc(dc.p, DType::I32, {int(count.size())});
                invoke(tq, tp, 0.0625f, resident, staging, ts, tc, frontier, splits,
                       partial.tensors, nullptr);
                for (auto buffers :
                     {std::pair{&partial.o, &pieces.o}, std::pair{&partial.m, &pieces.m},
                      std::pair{&partial.l, &pieces.l}})
                    cuda_check(cudaMemcpy(static_cast<std::byte*>(buffers.second->p) +
                                              pass * buffers.first->bytes,
                                          buffers.first->p, buffers.first->bytes,
                                          cudaMemcpyDeviceToDevice),
                               "collect partial pass");
            }
            ops::attention_partial_lse_merge(pieces.tensors, streamed.tensors, nullptr);
            ops::attention_partial_finalize(streamed.tensors, out, nullptr);
            cuda_synchronize();
            const auto so = from_device<float>(streamed.o, streamed.o.bytes / 4);
            const auto sl = from_device<float>(streamed.l, streamed.l.bytes / 4);
            std::vector<float> normalized(so.size());
            for (std::size_t i = 0; i < so.size(); ++i)
                normalized[i] = sl[i / 256] > 0 ? so[i] / sl[i / 256] : 0;
            if (!repeat)
                combined_first = normalized;
            else if (std::memcmp(combined_first.data(), normalized.data(), normalized.size() * 4)) {
                std::cerr << "FAIL multipass repeat mismatch\n";
                ++failures;
            }
        }
        const auto one_o = from_device<float>(merged.o, merged.o.bytes / 4),
                   one_l = from_device<float>(merged.l, merged.l.bytes / 4);
        std::vector<double> streamed_fp32(combined_first.begin(), combined_first.end()),
            one_fp32(one_o.size());
        for (std::size_t i = 0; i < one_o.size(); ++i)
            one_fp32[i] = one_l[i / 256] > 0 ? one_o[i] / one_l[i / 256] : 0;
        failures += verify_reduction((label + " multipass-vs-single FP32").c_str(), streamed_fp32,
                                     one_fp32, {1e-3, 1e-4, 1e-3});
        output.copy_to_host(actual.data(), output.bytes);
        std::transform(actual.begin(), actual.end(), got.begin(),
                       [](auto x) { return double(bf16_to_f32(x)); });
        failures += verify_reduction((label + " multipass oracle").c_str(), got, expected,
                                     {1.0 / 256, 1.1e-3, 3.9e-3});
    }
    return failures;
}
} // namespace ninfer::test::partial_attention_reference

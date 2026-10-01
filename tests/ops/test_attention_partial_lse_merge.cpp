#include "partial_attention_reference.h"
using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::partial_attention_reference;

namespace {
int test_large_common_logits(bool exhaustive_bytes = false) {
    int failures = 0;
    for (auto format : {Format::Int8, Format::Rk4v4E8}) {
        if (exhaustive_bytes && format != Format::Rk4v4E8) continue;
        EncodedCache host(format, 128);
        const bool packed = format == Format::Rk4v4E8;
        std::fill(host.k.begin(), host.k.end(), packed ? 0x11 : 1);
        std::fill(host.ks.begin(), host.ks.end(), 0x2800);
        std::fill(host.vs.begin(), host.vs.end(), 0x2800);
        for (int key = 0; key < 128; ++key)
            for (int h = 0; h < 4; ++h) {
                const int code        = key / 64 ? 3 : 1;
                const std::size_t row = std::size_t(256) * (h + 4 * key);
                for (int d = 0; d < 256; ++d) {
                    host.decoded_k[row + d] = 1.0 / 32;
                    const unsigned byte     = exhaustive_bytes
                                                  ? unsigned((d / 2 + 17 * key + 13 * h) & 255)
                                                  : unsigned(code * 17);
                    const int nibble        = (byte >> (4 * (d & 1))) & 15;
                    host.decoded_v[row + d] =
                        packed ? (nibble < 8 ? nibble : nibble - 16) / 32.0 : code / 32.0;
                    if (packed)
                        host.v[page_index(128, key / 64, h, key % 64, d / 2)] = byte;
                    else
                        host.v[page_index(256, key / 64, h, key % 64, d)] = code;
                }
                if (packed)
                    for (int g = 0; g < 4; ++g) {
                        hadamard(host.decoded_k.data() + row + g * 64);
                        hadamard(host.decoded_v.data() + row + g * 64);
                    }
            }
        auto dk = to_device(host.k), dv = to_device(host.v), dks = to_device(host.ks),
             dvs = to_device(host.vs);
        PagedKVLayerView resident;
        resident.k_pages = Tensor(dk.p, packed ? DType::U8 : DType::I8, {host.code_dim, 64, 4, 2});
        resident.v_pages = Tensor(dv.p, resident.k_pages.dtype, {host.code_dim, 64, 4, 2});
        resident.k_scale_pages = Tensor(dks.p, DType::FP16, {4, 64, 4, 2});
        resident.v_scale_pages = Tensor(dvs.p, DType::FP16, {4, 64, 4, 2});
        resident.dtype         = DType::I8;
        resident.head_dim      = 256;
        resident.num_kv_heads  = 4;
        resident.quant_group   = 64;
        resident.packed_k = resident.packed_v = resident.rotate_k = resident.rotate_v =
            resident.e8_lattice                                   = packed;
        auto staging                                              = resident;
        staging.k_pages                                           = {};
        staging.v_pages                                           = {};
        staging.k_scale_pages                                     = {};
        staging.v_scale_pages                                     = {};
        const std::vector<ops::AttentionPageAccess> pages{{0, 0}, {1, 1}};
        const auto prefix = ops::attention_access_prefix(pages, 128, 2, 0);
        auto da = to_device(pages), df = to_device(prefix);
        Tensor ta(da.p, DType::I32, {2, 2}), tf(df.p, DType::I32, {3});
        for (int t : {4, 64})
            for (float common : {1e9f, 1e10f, 1.2e10f}) {
                std::vector<std::uint16_t> q(std::size_t(256) * 24 * t, f32_to_bf16(common));
                std::vector<std::int32_t> pos(t);
                std::iota(pos.begin(), pos.end(), 128 - t);
                const auto expected = oracle(host, q, pos);
                auto dq = to_device(q), dp = to_device(pos);
                Tensor tq(dq.p, DType::BF16, {256, 24, t}), tp(dp.p, DType::I32, {t});
                Parts partial(t, 1), state(t, 1);
                DeviceBuffer output(q.size() * 2);
                Tensor out(output.p, DType::BF16, {256, 24, t});
                std::vector<std::uint16_t> first;
                std::vector<float> first_o, first_m, first_l;
                auto invoke = t <= 16 ? ops::gqa_attention_partial_decode
                                      : ops::gqa_attention_partial_prefill;
                for (int repeat = 0; repeat < 2; ++repeat) {
                    invoke(tq, tp, .0625f, resident, staging, ta, tf, 128, 1, partial.tensors,
                           nullptr);
                    ops::attention_partial_lse_accumulate(partial.tensors, state.tensors, true,
                                                          nullptr, &out);
                    cuda_synchronize();
                    const auto bits = from_device<std::uint16_t>(output, q.size());
                    const auto po   = from_device<float>(partial.o, partial.o.bytes / 4),
                               pm   = from_device<float>(partial.m, partial.m.bytes / 4),
                               pl   = from_device<float>(partial.l, partial.l.bytes / 4);
                    if (repeat &&
                        (first != bits || std::memcmp(first_o.data(), po.data(), partial.o.bytes) ||
                         std::memcmp(first_m.data(), pm.data(), partial.m.bytes) ||
                         std::memcmp(first_l.data(), pl.data(), partial.l.bytes)))
                        ++failures;
                    first   = bits;
                    first_o = po;
                    first_m = pm;
                    first_l = pl;
                    std::vector<double> got(bits.size());
                    std::transform(bits.begin(), bits.end(), got.begin(),
                                   [](auto b) { return double(bf16_to_f32(b)); });
                    const auto name =
                        std::string(exhaustive_bytes
                                        ? "exhaustive packed bytes / large common logits "
                                        : "large common logits ") +
                        std::to_string(int(format)) + " T=" + std::to_string(t) +
                        " common=" + std::to_string(common);
                    failures +=
                        verify_reduction(name.c_str(), got, expected, {1.0 / 256, 1.1e-3, 3.9e-3});
                }
            }
    }
    return failures;
}

int test_merge_oracle() {
    constexpr int rows = 48, parts = 7;
    Parts input(2, parts), merged(2, 1), hierarchy(2, 2), recursive(2, 1);
    std::vector<float> o(rows * parts * 256), m(rows * parts), l(m.size());
    for (int p = 0; p < parts; ++p)
        for (int r = 0; r < rows; ++r) {
            const int index    = r + rows * p;
            const bool neutral = p == 2 || r == 23;
            m[index] =
                neutral ? -std::numeric_limits<float>::infinity() : float((p - 3) * 25 + r % 7);
            l[index] = neutral ? 0.0f : float(1 + p) * 0.25f;
            for (int d = 0; d < 256; ++d)
                o[d + 256 * index] = neutral
                                         ? std::numeric_limits<float>::quiet_NaN()
                                         : l[index] * float(((d + 3 * r + p) % 17) - 8) * 0.125f;
        }
    input.o.copy_from_host(o.data(), input.o.bytes);
    input.m.copy_from_host(m.data(), input.m.bytes);
    input.l.copy_from_host(l.data(), input.l.bytes);
    std::vector<double> expected_o(rows * 256), expected_m(rows), expected_l(rows);
    for (int r = 0; r < rows; ++r) {
        double maximum = -std::numeric_limits<double>::infinity();
        for (int p = 0; p < parts; ++p)
            if (l[r + rows * p] > 0) maximum = std::max(maximum, double(m[r + rows * p]));
        expected_m[r] = maximum;
        for (int p = 0; p < parts; ++p) {
            const int index = r + rows * p;
            if (l[index] <= 0) continue;
            const double weight = std::exp(double(m[index]) - maximum);
            expected_l[r] += l[index] * weight;
            for (int d = 0; d < 256; ++d) expected_o[d + 256 * r] += o[d + 256 * index] * weight;
        }
    }
    std::vector<float> previous_o, previous_m, previous_l;
    for (int repeat = 0; repeat < 2; ++repeat) {
        ops::attention_partial_lse_merge(input.tensors, merged.tensors, nullptr);
        cuda_synchronize();
        auto a = from_device<float>(merged.o, merged.o.bytes / 4),
             b = from_device<float>(merged.m, rows), c = from_device<float>(merged.l, rows);
        if (repeat && (std::memcmp(a.data(), previous_o.data(), merged.o.bytes) ||
                       std::memcmp(b.data(), previous_m.data(), merged.m.bytes) ||
                       std::memcmp(c.data(), previous_l.data(), merged.l.bytes)))
            return 1;
        previous_o = a;
        previous_m = b;
        previous_l = c;
        for (int r = 0; r < rows; ++r) {
            if (b[r] != expected_m[r] ||
                std::abs(double(c[r]) - expected_l[r]) > 2e-6 * std::max(1.0, expected_l[r]))
                return 1;
            for (int d = 0; d < 256; ++d)
                if (!std::isfinite(a[d + 256 * r]) ||
                    std::abs(double(a[d + 256 * r]) - expected_o[d + 256 * r]) >
                        2e-6 * std::max(1.0, std::abs(expected_o[d + 256 * r])))
                    return 1;
        }
    }
    // Fused finalization matches the separate path, including the neutral row.
    Parts persistent(2, 1);
    DeviceBuffer fused(rows * 256 * 2), separate(rows * 256 * 2);
    Tensor fused_out(fused.p, DType::BF16, {256, 24, 2}),
        separate_out(separate.p, DType::BF16, {256, 24, 2});
    ops::attention_partial_finalize(merged.tensors, separate_out, nullptr);
    for (int repeat = 0; repeat < 2; ++repeat) {
        persistent.o.fill(255);
        persistent.m.fill(255);
        persistent.l.fill(255);
        fused.fill(255);
        ops::attention_partial_lse_accumulate(input.tensors, persistent.tensors, true, nullptr,
                                              &fused_out);
        cuda_synchronize();
        if (from_device<std::uint16_t>(fused, fused.bytes / 2) !=
                from_device<std::uint16_t>(separate, separate.bytes / 2) ||
            from_device<float>(persistent.o, persistent.o.bytes / 4) != previous_o ||
            from_device<float>(persistent.m, persistent.m.bytes / 4) != previous_m ||
            from_device<float>(persistent.l, persistent.l.bytes / 4) != previous_l)
            return 1;
    }
    std::vector<float> first_carry_o, first_carry_m, first_carry_l;
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (int pass = 0; pass < 2; ++pass) {
            const int begin = pass ? 3 : 0, count = pass ? 4 : 3;
            ops::AttentionPartial slice{
                Tensor(static_cast<float*>(input.o.p) + 256 * rows * begin, DType::FP32,
                       {256, 24, 2, count}),
                Tensor(static_cast<float*>(input.m.p) + rows * begin, DType::FP32, {24, 2, count}),
                Tensor(static_cast<float*>(input.l.p) + rows * begin, DType::FP32, {24, 2, count})};
            ops::attention_partial_lse_accumulate(slice, persistent.tensors, pass == 0, nullptr,
                                                  pass == 1 ? &fused_out : nullptr);
        }
        cuda_synchronize();
        const auto a = from_device<float>(persistent.o, persistent.o.bytes / 4),
                   b = from_device<float>(persistent.m, rows),
                   c = from_device<float>(persistent.l, rows);
        if (repeat && (std::memcmp(first_carry_o.data(), a.data(), persistent.o.bytes) ||
                       std::memcmp(first_carry_m.data(), b.data(), persistent.m.bytes) ||
                       std::memcmp(first_carry_l.data(), c.data(), persistent.l.bytes)))
            return 1;
        first_carry_o = a;
        first_carry_m = b;
        first_carry_l = c;
        for (int r = 0; r < rows; ++r) {
            if (b[r] != expected_m[r] ||
                std::abs(double(c[r]) - expected_l[r]) > 2e-6 * std::max(1.0, expected_l[r]))
                return 1;
            for (int d = 0; d < 256; ++d)
                if (!std::isfinite(a[d + 256 * r]) ||
                    std::abs(double(a[d + 256 * r]) - expected_o[d + 256 * r]) >
                        2e-6 * std::max(1.0, std::abs(expected_o[d + 256 * r])))
                    return 1;
        }
    }
    // Merge two contiguous groups, then merge the resulting FP32 tuples again.
    for (int pass = 0; pass < 2; ++pass) {
        const int begin = pass ? 3 : 0, count = pass ? 4 : 3;
        ops::AttentionPartial slice{
            Tensor(static_cast<float*>(input.o.p) + 256 * rows * begin, DType::FP32,
                   {256, 24, 2, count}),
            Tensor(static_cast<float*>(input.m.p) + rows * begin, DType::FP32, {24, 2, count}),
            Tensor(static_cast<float*>(input.l.p) + rows * begin, DType::FP32, {24, 2, count})};
        ops::AttentionPartial destination{
            Tensor(static_cast<float*>(hierarchy.o.p) + 256 * rows * pass, DType::FP32,
                   {256, 24, 2, 1}),
            Tensor(static_cast<float*>(hierarchy.m.p) + rows * pass, DType::FP32, {24, 2, 1}),
            Tensor(static_cast<float*>(hierarchy.l.p) + rows * pass, DType::FP32, {24, 2, 1})};
        ops::attention_partial_lse_merge(slice, destination, nullptr);
    }
    ops::attention_partial_lse_merge(hierarchy.tensors, recursive.tensors, nullptr);
    cuda_synchronize();
    const auto result = from_device<float>(recursive.o, recursive.o.bytes / 4);
    int failures =
        verify_reduction("LSE hierarchical FP64", std::vector<double>(result.begin(), result.end()),
                         expected_o, {2e-6, 2e-6, 2e-6});
    try {
        ops::attention_partial_lse_merge(merged.tensors, merged.tensors, nullptr);
        ++failures;
    } catch (const std::invalid_argument&) {}
    try {
        ops::attention_partial_lse_accumulate(merged.tensors, merged.tensors, false, nullptr);
        ++failures;
    } catch (const std::invalid_argument&) {}
    for (auto bad : {std::pair{0, 1}, std::pair{1024, 0}, std::pair{1024, 513}}) {
        try {
            (void)ops::attention_partial_workspace_bytes(bad.first, bad.second);
            ++failures;
        } catch (const std::invalid_argument&) {}
    }
    for (const auto bad : {std::vector<ops::AttentionPageAccess>{{1, 0}, {0, 1}},
                           std::vector<ops::AttentionPageAccess>{{0, 0}, {0, 1}},
                           std::vector<ops::AttentionPageAccess>{{3000, 0}},
                           std::vector<ops::AttentionPageAccess>{{0, 8}}}) {
        try {
            (void)ops::attention_access_prefix(bad, 512, 4, 4);
            ++failures;
        } catch (const std::invalid_argument&) {}
    }
    std::cout << (failures ? "FAIL" : "PASS")
              << " fixed-order/hierarchical LSE oracle and input validation\n";
    return failures;
}
} // namespace
int main() try {
    if (cuda_unavailable()) return 77;
    int failures =
        test_large_common_logits() + test_large_common_logits(true) + test_merge_oracle();
    if (ops::attention_partial_workspace_bytes(1024, 1) != 50724864 ||
        ops::attention_partial_workspace_bytes(1024, 2) != 76087296 ||
        ops::attention_partial_workspace_bytes(2048, 1) != 101449728)
        ++failures;
    const auto budget1024 = ops::plan_attention_partial_workspace(1024, 32, 100 * 1024 * 1024),
               budget2048 = ops::plan_attention_partial_workspace(2048, 32, 100 * 1024 * 1024);
    if (budget1024.splits != 3 || budget1024.bytes != 101449728 || budget2048.splits != 1 ||
        budget2048.bytes != 101449728)
        ++failures;
    const auto capped_large =
        ops::plan_attention_partial_workspace(50000, 512, std::size_t(3) << 30);
    if (capped_large.splits != 1 || capped_large.bytes != 2476800000ull) ++failures;
    const auto launch_cap = ops::plan_attention_partial_workspace(50000, 512, ~std::size_t(0));
    if (launch_cap.splits != 447 || launch_cap.bytes != 554803200000ull) ++failures;
    for (const int invalid_preferred : {0, 513}) {
        try {
            (void)ops::plan_attention_partial_workspace(1024, invalid_preferred, 100 * 1024 * 1024);
            ++failures;
        } catch (const std::invalid_argument&) {}
    }
    try {
        (void)ops::plan_attention_partial_workspace(1024, 32, 50724863);
        ++failures;
    } catch (const std::invalid_argument&) {}
    for (auto f : {Format::Bf16, Format::Int8, Format::Rk4v4E8}) {
        for (int t : {1, 2, 4}) { failures += run(f, t, 7, false); }
        failures += run(f, 4, 300, false);        // more splits than visible keys
        failures += run(f, 64, 2, true);          // all producer row tiles
        failures += run(f, 65, 2, true, 0, true); // second query tile / carry
        failures += run(f, 17, 7, true);          // partial final query tile
        failures += run(f, 9, 7, false);          // two small-T tiles, partial second tile
        failures += run(f, 16, 7, true, 1, true); // causal boundaries and three streamed passes
        failures += run(f, 4, 7, false, 0, true);
        failures += run(f, 17, 7, true, 2);           // empty access list
        failures += run(f, 4, 7, false, 3);           // all pages causally masked
        failures += run(f, 1, 7, false, 0, false, 1); // resident only
        failures += run(f, 17, 7, true, 0, false, 2); // staging only
    }
    std::cout << (failures ? "FAIL" : "PASS")
              << " partial attention / LSE FP64 and repeat checks\n";
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
}

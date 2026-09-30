#include "partial_attention_reference.h"
using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::partial_attention_reference;

namespace {
int test_merge_oracle() {
    constexpr int rows = 48, parts = 7;
    Parts input(2, parts), merged(2, 1), hierarchy(2, 2), recursive(2, 1);
    std::vector<float> o(rows * parts * 256), m(rows * parts), l(m.size());
    for (int p = 0; p < parts; ++p)
        for (int r = 0; r < rows; ++r) {
            const int index = r + rows * p;
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
            if (l[r + rows * p] > 0)
                maximum = std::max(maximum, double(m[r + rows * p]));
        expected_m[r] = maximum;
        for (int p = 0; p < parts; ++p) {
            const int index = r + rows * p;
            if (l[index] <= 0)
                continue;
            const double weight = std::exp(double(m[index]) - maximum);
            expected_l[r] += l[index] * weight;
            for (int d = 0; d < 256; ++d)
                expected_o[d + 256 * r] += o[d + 256 * index] * weight;
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
    } catch (const std::invalid_argument&) {
    }
    for (const auto bad : {std::vector<ops::AttentionPageAccess>{{1, 0}, {0, 1}},
                           std::vector<ops::AttentionPageAccess>{{0, 0}, {0, 1}},
                           std::vector<ops::AttentionPageAccess>{{3000, 0}},
                           std::vector<ops::AttentionPageAccess>{{0, 8}}}) {
        try {
            (void)ops::attention_access_prefix(bad, 512, 4, 4);
            ++failures;
        } catch (const std::invalid_argument&) {
        }
    }
    std::cout << (failures ? "FAIL" : "PASS")
              << " fixed-order/hierarchical LSE oracle and input validation\n";
    return failures;
}
} // namespace
int main() try {
    if (cuda_unavailable())
        return 77;
    int failures = test_merge_oracle();
    for (auto f : {Format::Bf16, Format::Int8, Format::Rk4v4E8}) {
        for (int t : {1, 2, 4}) {
            failures += run(f, t, 7, false);
        }
        failures += run(f, 4, 300, false);        // more splits than visible keys
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

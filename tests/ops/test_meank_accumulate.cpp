#include "core/arena.h"
#include "core/kvmem/host_meank_index.h"
#include "ninfer/ops/meank_accumulate.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace ninfer;
using namespace ninfer::kvmem;
namespace {
void require(bool b, const char* m) {
    if (!b)
        throw std::runtime_error(m);
}
template <class F> void rejects(F f) {
    bool b = false;
    try {
        f();
    } catch (const std::exception&) {
        b = true;
    }
    require(b, "invalid operation accepted");
}
void check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
// Direct FP64->binary16 nearest-even oracle, independent of CUDA conversion.
std::uint16_t half(double x) {
    auto sign = std::uint16_t(std::signbit(x) ? 0x8000 : 0);
    x = std::abs(x);
    if (!x)
        return sign;
    int e;
    double f = std::frexp(x, &e);
    bool sub = e < -13;
    double s = sub ? std::ldexp(x, 24) : f * 2048;
    auto r = std::uint32_t(std::floor(s));
    double rem = s - r;
    if (rem > 0.5 || (rem == 0.5 && (r & 1)))
        ++r;
    if (e > 16)
        return std::uint16_t(sign | 0x7c00);
    return std::uint16_t(sign | std::min(std::uint32_t(0x7c00),
                                         sub ? r : std::uint32_t((e + 14) * 1024) + r - 1024));
}
double bf(std::uint16_t x) {
    return double(std::bit_cast<float>(std::uint32_t(x) << 16));
}
constexpr int R = 256 * 4;
struct GPU {
    DeviceBuffer k{R * 160 * 2}, seed{R * 4}, sum{R * 4}, means{R * 4 * 2}, count{4},
        tail{R * 64 * 2}, provisional{R * 16 * 2};
    cudaStream_t stream{};
    GPU() {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~GPU() {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    std::vector<std::uint16_t> mean;
    std::vector<float> sums;
    int n{};
    void run(std::span<const std::uint16_t> input, int first, int valid,
             std::span<const float> prefix = {}) {
        std::vector<float> zero(R);
        seed.copy_from_host(prefix.empty() ? zero.data() : prefix.data(), R * 4);
        if (!input.empty())
            k.copy_from_host(input.data(), input.size() * 2);
        check(
            cudaDeviceSynchronize()); // drain default-stream fixture uploads before nonblocking op
        int pages = (first % 64 + valid + 63) / 64;
        ops::MeanKOutput out{Tensor(means.p, DType::FP16, {256, 4, std::max(1, pages)}),
                             Tensor(sum.p, DType::FP32, {256, 4}),
                             Tensor(count.p, DType::I32, {1})};
        ops::meank_accumulate(
            Tensor(k.p, DType::BF16, {256, 4, std::max(1, int(input.size() / R))}), first, valid,
            Tensor(seed.p, DType::FP32, {256, 4}), first % 64, out, stream);
        check(cudaStreamSynchronize(stream));
        mean.resize(pages * R);
        sums.resize(R);
        means.copy_to_host(mean.data(), mean.size() * 2);
        sum.copy_to_host(sums.data(), R * 4);
        count.copy_to_host(&n, 4);
    }
};
std::vector<std::uint16_t> data(int tokens, int offset = 0) {
    std::vector<std::uint16_t> v(tokens * R);
    for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < R; ++r)
            v[t * R + r] =
                std::uint16_t(0x3d80 + ((t + offset) * 13 + r * 7) % 80); // exact small dyadics
    return v;
}
void oracle(GPU& g, const std::vector<std::uint16_t>& v, int first, int count,
            const std::vector<double>& prefix = {}) {
    int pages = (first % 64 + count + 63) / 64;
    for (int p = 0; p < pages; ++p)
        for (int r = 0; r < R; ++r) {
            double s = (p == 0 && !prefix.empty()) ? prefix[r] : 0;
            int lo = std::max(0, p * 64 - first % 64),
                hi = std::min(count, (p + 1) * 64 - first % 64);
            for (int t = lo; t < hi; ++t)
                s += bf(v[t * R + r]);
            int c = hi - lo + (p == 0 ? first % 64 : 0);
            if (g.mean[p * R + r] != half(s / c)) {
                std::cerr << "oracle first=" << first << " count=" << count << " page=" << p
                          << " coordinate=" << r << " sum=" << s << " divisor=" << c
                          << " actual=" << g.mean[p * R + r] << " expected=" << half(s / c) << '\n';
                throw std::runtime_error("GPU mean differs from independent FP64 oracle");
            }
            if (p == pages - 1)
                require(g.sums[r] == (g.n ? float(s) : 0.f), "FP32 tail sum lost exact prefix");
        }
    require(g.n == (first + count) % 64, "tail count wrong");
}
void numerical() {
    GPU g;
    for (int t : {1, 63, 64, 65, 129}) {
        auto v = data(t);
        g.run(v, 0, t);
        oracle(g, v, 0, t);
    }
    auto repeated = data(129);
    g.run(repeated, 0, 129);
    auto fixed_means = g.mean;
    auto fixed_sums = g.sums;
    for (int run = 0; run < 8; ++run) {
        g.run(repeated, 0, 129);
        require(g.mean == fixed_means && g.sums == fixed_sums,
                "chronological GPU accumulation not bitwise repeatable");
    }
    auto v = data(65);
    g.run(std::span(v).first(13 * R), 0, 13);
    auto saved = g.sums;
    std::vector<double> seed(saved.begin(), saved.end());
    std::vector<std::uint16_t> rest(v.begin() + 13 * R, v.end());
    g.run(rest, 13, 52, saved);
    oracle(g, rest, 13, 52, seed);
    // BF16 mixtures produce FP16 halfway values with both even and odd low bits.
    std::vector<std::uint16_t> ties(16 * R, 0x3f80);
    for (int r = 0; r < R; ++r) {
        ties[r] = 0x3f81;
        if (r & 1)
            ties[R + r] = 0x3f81;
        if (r & 1)
            ties[2 * R + r] = 0x3f81;
    }
    g.run(ties, 0, 16);
    oracle(g, ties, 0, 16);
    ops::MeanKOutput out{Tensor(g.means.p, DType::FP16, {256, 4, 2}),
                         Tensor(g.seed.p, DType::FP32, {256, 4}),
                         Tensor(g.count.p, DType::I32, {1})};
    rejects([&] {
        ops::meank_accumulate(Tensor(g.k.p, DType::BF16, {256, 4, 65}), 0, 65,
                              Tensor(g.seed.p, DType::FP32, {256, 4}), 0, out, g.stream);
    });
    out.sum = Tensor(g.sum.p, DType::FP32, {256, 4});
    rejects([&] {
        ops::meank_accumulate(Tensor(g.k.p, DType::BF16, {256, 4, 65}), 0, 66,
                              Tensor(g.seed.p, DType::FP32, {256, 4}), 0, out, g.stream);
    });
    rejects([&] {
        ops::meank_accumulate(Tensor(g.k.p, DType::BF16, {256, 4, 65}), 1, 65,
                              Tensor(g.seed.p, DType::FP32, {256, 4}), 0, out, g.stream);
    });
    rejects([&] {
        ops::meank_accumulate(Tensor(g.k.p, DType::BF16, {256, 4, 65}), 0, 65,
                              Tensor(g.k.p, DType::FP32, {256, 4}), 0, out, g.stream);
    });
}
void complete(HostMeanKIndex& index, const MeanKTicket& ticket, GPU& g, bool reverse = false) {
    auto v = data(ticket.count, ticket.first);
    for (int i = 0; i < 16; ++i) {
        int layer = reverse ? 15 - i : i;
        std::vector<float> zero(R);
        auto prefix = ticket.kind == MeanKTransactionKind::Trim ? std::span<const float>(zero)
                                                                : index.prefix_sum(layer);
        g.run(v, ticket.first, ticket.count, prefix);
        require(index.complete_layer(ticket, layer, g.mean, g.sums), "fresh completion rejected");
        require(!index.complete_layer(ticket, layer, g.mean, g.sums),
                "duplicate completion accepted");
        require(index.published() == (i == 15), "partial layer completion published index");
    }
}
void owner() {
    auto layout = plan_meank_resources(256, 16, 256, 4, 129);
    require(layout.index_bytes == 16 * 4 * R * 2, "fixed maximum index stride");
    require(layout.tail_bytes == 2 * 1024 * 1024 && layout.provisional_bytes == 512 * 1024,
            "bounded GPU raw K");
    require(layout.snapshot_bytes == 128 * 1024, "two bounded snapshot sums");
    require(layout.device_bytes >= layout.output_sum_offset + 16 * R * 4,
            "device highwater includes outputs");
    require(layout.base_sum_offset != layout.active_sum_offset &&
                layout.base_sum_offset != layout.seed_sum_offset &&
                layout.base_sum_offset != layout.output_sum_offset,
            "independent base/seed/output sums alias");
    require(meank_host_admission_bytes(layout) ==
                (std::uint64_t(4) << 30) + layout.index_bytes + layout.snapshot_bytes +
                    layout.host_continuation_bytes + layout.host_patch_bytes,
            "host admission omitted bytes/headroom");
    auto forged = layout;
    forged.device_bytes = 1;
    rejects([&] { HostMeanKIndex bad(forged, false); });
    rejects([&] {
        (void)plan_meank_resources(std::numeric_limits<std::uint32_t>::max(),
                                   std::numeric_limits<std::uint32_t>::max(), 256, 4, 129);
    });
    HostMeanKIndex index(layout, false);
    GPU g;
    auto t = index.begin(MeanKTransactionKind::ExactPrefill, 63);
    complete(index, t, g, true);
    auto fixed = index.layer_data(1).data();
    auto snapshot = index.capture();
    auto other = index.capture();
    rejects([&] { (void)index.capture(); });
    other.reset();
    t = index.begin(MeanKTransactionKind::ExactPrefill, 66);
    complete(index, t, g);
    require(index.frontier() == 129 && index.layer_data(1).data() == fixed,
            "growth reinterpreted layer stride");
    auto newer = index.capture();
    auto stale = index.begin(MeanKTransactionKind::OrdinaryMain, 1);
    index.discard(stale);
    auto restore = index.restore(snapshot);
    require(!index.published() && index.frontier() == 63 && index.base_frontier() == 63,
            "restore must block scoring and establish exact base");
    require(!index.complete_layer(stale, 0, g.mean, g.sums), "stale completion accepted");
    auto foreign_ticket = restore;
    foreign_ticket.owner++;
    require(!index.complete_layer(foreign_ticket, 0, g.mean, g.sums),
            "foreign ticket matched a pending current transaction");
    // Saved prefix survives seal and overwrite, without raw K copy or rounded mean reconstruction.
    complete(index, restore, g);
    require(index.published(), "restore republish failed");
    auto original = data(63);
    for (int r = 0; r < R; ++r) {
        double sum = 0;
        for (int token = 0; token < 63; ++token)
            sum += bf(original[token * R + r]);
        require(index.layer_data(0)[r] == half(sum / 63),
                "snapshot prefix corrupted after original tail sealed/overwritten");
    }
    rejects([&] { (void)index.restore(newer); });
    rejects([&] { (void)index.begin_trim(62); });
    t = index.begin(MeanKTransactionKind::SpeculativeMain, 16);
    require(!index.published(), "inflight transaction index exposed");
    rejects([&] { (void)index.layer_data(0); });
    rejects([&] { (void)index.capture(); });
    rejects([&] { index.complete_layer(t, 0, g.mean, g.sums); });
    t = index.accept(t, 2);
    complete(index, t, g);
    require(index.frontier() == 65, "accepted crossing failed");
    t = index.begin(MeanKTransactionKind::SpeculativeMain, 16);
    t = index.accept(t, 0);
    require(index.published() && index.frontier() == 65,
            "rejected verify changed accepted frontier");
    rejects([&] { (void)index.accept(t, 1); });
    t = index.begin(MeanKTransactionKind::SpeculativeMain, 16);
    t = index.accept(t, 16);
    complete(index, t, g);
    require(index.frontier() == 81, "full accepted width commit depended on generic trim");
    auto trim = index.begin_trim(64);
    complete(index, trim, g);
    require(index.frontier() == 64, "bounded trim failed");
    index.reset();
    rejects([&] { (void)index.restore(snapshot); });
    HostMeanKIndex foreign(layout, false);
    rejects([&] { (void)foreign.restore(snapshot); });
    HostMeanKIndex pinned(plan_meank_resources(64, 1, 256, 4, 16), true);
    require(pinned.pinned(), "actual pinned archive did not select pinned MeanK");
    unsigned flags = ~0u;
    auto pinned_ticket = pinned.begin(MeanKTransactionKind::ExactPrefill, 1);
    auto one = data(1);
    g.run(one, 0, 1);
    pinned.complete_layer(pinned_ticket, 0, g.mean, g.sums);
    check(cudaHostGetFlags(&flags, const_cast<std::uint16_t*>(pinned.layer_data(0).data())));
    require((flags & cudaHostAllocWriteCombined) == 0, "host MeanK must be cacheable pinned");
}
void trim_replay() {
    HostMeanKIndex index(plan_meank_resources(256, 16, 256, 4, 129), false);
    GPU g;
    complete(index, index.begin(MeanKTransactionKind::ExactPrefill, 45), g);
    auto snapshot = index.capture();
    complete(index, index.begin(MeanKTransactionKind::ExactPrefill, 84), g);
    complete(index, index.restore(snapshot), g);
    auto suffix = data(3, 45);
    g.k.copy_from_host(suffix.data(), suffix.size() * 2);
    check(cudaDeviceSynchronize());
    ops::meank_retain_tail(Tensor(g.k.p, DType::BF16, {256, 4, 3}), 45, 3,
                           Tensor(g.tail.p, DType::BF16, {256, 4, 64}), g.stream);
    check(cudaStreamSynchronize(g.stream));
    complete(index, index.begin(MeanKTransactionKind::ExactPrefill, 3), g);
    require(index.frontier() == 48 && index.base_frontier() == 45,
            "base lost across restored partial append");
    auto abandoned_future = index.capture();
    auto trim = index.begin_trim(46);
    for (int layer = 0; layer < 16; ++layer) {
        auto base = index.base_sum(layer);
        g.seed.copy_from_host(base.data(), R * 4);
        check(cudaDeviceSynchronize());
        ops::MeanKOutput out{Tensor(g.means.p, DType::FP16, {256, 4, 1}),
                             Tensor(g.sum.p, DType::FP32, {256, 4}),
                             Tensor(g.count.p, DType::I32, {1})};
        auto row = Tensor(g.tail.p, DType::BF16, {256, 4, 64}).slice(2, 45, 1);
        ops::meank_accumulate(row, 45, 1, Tensor(g.seed.p, DType::FP32, {256, 4}), 45, out,
                              g.stream);
        check(cudaStreamSynchronize(g.stream));
        std::vector<std::uint16_t> means(R);
        std::vector<float> sums(R);
        g.means.copy_to_host(means.data(), R * 2);
        g.sum.copy_to_host(sums.data(), R * 4);
        require(index.complete_layer(trim, layer, means, sums), "bounded prefix replay rejected");
    }
    auto expected = data(46);
    for (int r = 0; r < R; ++r) {
        double s = 0;
        for (int t = 0; t < 46; ++t)
            s += bf(expected[t * R + r]);
        require(index.layer_data(15)[r] == half(s / 46),
                "trim used evolving48 sum or rounded45 mean instead of exact saved45 sum");
    }
    rejects([&] { (void)index.begin_trim(44); });
    auto follow = index.begin(MeanKTransactionKind::OrdinaryMain, 1);
    complete(index, follow, g);
    require(index.frontier() == 47 && index.base_frontier() == 45,
            "ordinary accepted anchor advanced wrong frontier");
    complete(index, index.begin(MeanKTransactionKind::ExactPrefill, 3), g);
    rejects([&] { (void)index.restore(abandoned_future); });
}
void provisional() {
    GPU g;
    auto v = data(16, 63);
    g.k.copy_from_host(v.data(), v.size() * 2);
    check(cudaDeviceSynchronize());
    ops::meank_stage(Tensor(g.k.p, DType::BF16, {256, 4, 16}), 16,
                     Tensor(g.provisional.p, DType::BF16, {256, 4, 16}), g.stream);
    check(cudaStreamSynchronize(g.stream));
    // Poison original projected K: accepted computation must consume captured pre-RoPE rows.
    g.k.fill(0);
    ops::MeanKOutput out{Tensor(g.means.p, DType::FP16, {256, 4, 2}),
                         Tensor(g.sum.p, DType::FP32, {256, 4}),
                         Tensor(g.count.p, DType::I32, {1})};
    std::vector<float> seed(R, 1.f);
    g.seed.copy_from_host(seed.data(), R * 4);
    check(cudaDeviceSynchronize());
    ops::meank_accumulate(Tensor(g.provisional.p, DType::BF16, {256, 4, 16}), 63, 2,
                          Tensor(g.seed.p, DType::FP32, {256, 4}), 63, out, g.stream);
    ops::meank_retain_tail(Tensor(g.provisional.p, DType::BF16, {256, 4, 16}), 63, 2,
                           Tensor(g.tail.p, DType::BF16, {256, 4, 64}), g.stream);
    check(cudaStreamSynchronize(g.stream));
    std::vector<std::uint16_t> mean(2 * R), tail(64 * R);
    g.means.copy_to_host(mean.data(), mean.size() * 2);
    // Frontier 65 has one valid row in its final page; the other tail slots
    // remain uninitialized until accepted rows or the fixture populate them.
    g.tail.copy_to_host(tail.data(), R * sizeof(std::uint16_t));
    for (int r = 0; r < R; ++r) {
        require(mean[r] == half((1 + bf(v[r])) / 64), "rejected suffix contaminated mean");
        require(tail[r] == v[R + r], "cross-page accepted tail retained incorrectly");
    }
    auto prior = data(64);
    auto samepage = data(3, 900);
    g.tail.copy_from_host(prior.data(), prior.size() * 2);
    g.k.copy_from_host(samepage.data(), samepage.size() * 2);
    check(cudaDeviceSynchronize());
    ops::meank_retain_tail(Tensor(g.k.p, DType::BF16, {256, 4, 3}), 45, 2,
                           Tensor(g.tail.p, DType::BF16, {256, 4, 64}), g.stream);
    check(cudaStreamSynchronize(g.stream));
    g.tail.copy_to_host(tail.data(), tail.size() * 2);
    for (int token = 0; token < 64; ++token)
        for (int r = 0; r < R; ++r)
            require(tail[token * R + r] == (token >= 45 && token < 47
                                                ? samepage[(token - 45) * R + r]
                                                : prior[token * R + r]),
                    "same-page accepted tail overwritten prefix/rejected suffix");
    rejects([&] {
        ops::meank_stage(Tensor(g.k.p, DType::BF16, {256, 4, 16}), 17,
                         Tensor(g.provisional.p, DType::BF16, {256, 4, 16}), g.stream);
    });
}
void full_accept_cross_page() {
    HostMeanKIndex index(plan_meank_resources(256, 16, 256, 4, 129), false);
    GPU g;
    complete(index, index.begin(MeanKTransactionKind::ExactPrefill, 63), g);

    auto prefix = data(63);
    std::vector<double> expected_seed(R, 0.0);
    for (int token = 0; token < 63; ++token)
        for (int r = 0; r < R; ++r)
            expected_seed[r] += bf(prefix[token * R + r]);

    auto verify = data(16, 63);
    g.k.copy_from_host(verify.data(), verify.size() * 2);
    g.tail.fill(0);
    check(cudaDeviceSynchronize());
    ops::meank_stage(Tensor(g.k.p, DType::BF16, {256, 4, 16}), 16,
                     Tensor(g.provisional.p, DType::BF16, {256, 4, 16}), g.stream);
    check(cudaStreamSynchronize(g.stream));
    g.k.fill(0); // Captured pre-RoPE K must survive overwrite of the projection tensor.

    auto ticket = index.begin(MeanKTransactionKind::SpeculativeMain, 16);
    ticket = index.accept(ticket, 16);
    g.mean.resize(2 * R);
    g.sums.resize(R);
    std::vector<std::uint16_t> tail(64 * R);
    for (int layer = 0; layer < 16; ++layer) {
        auto seed = index.prefix_sum(layer);
        g.seed.copy_from_host(seed.data(), seed.size_bytes());
        check(cudaDeviceSynchronize());
        ops::MeanKOutput out{Tensor(g.means.p, DType::FP16, {256, 4, 2}),
                             Tensor(g.sum.p, DType::FP32, {256, 4}),
                             Tensor(g.count.p, DType::I32, {1})};
        ops::meank_accumulate(Tensor(g.provisional.p, DType::BF16, {256, 4, 16}), 63, 16,
                              Tensor(g.seed.p, DType::FP32, {256, 4}), 63, out, g.stream);
        ops::meank_retain_tail(Tensor(g.provisional.p, DType::BF16, {256, 4, 16}), 63, 16,
                               Tensor(g.tail.p, DType::BF16, {256, 4, 64}), g.stream);
        check(cudaStreamSynchronize(g.stream));
        g.means.copy_to_host(g.mean.data(), g.mean.size() * 2);
        g.sum.copy_to_host(g.sums.data(), g.sums.size() * 4);
        g.count.copy_to_host(&g.n, 4);
        g.tail.copy_to_host(tail.data(), tail.size() * 2);

        // The first accepted row seals page 0; the remaining 15 rows form page 1.
        // Truncating this full acceptance to two rows produces count=1 and a
        // one-row page mean, so the independent count/sum/mean/tail checks fail.
        oracle(g, verify, 63, 16, expected_seed);
        for (int token = 0; token < 64; ++token)
            for (int r = 0; r < R; ++r)
                require(tail[token * R + r] == (token < 15 ? verify[(token + 1) * R + r] : 0),
                        "full accepted cross-page tail omitted/misplaced valid rows");
        require(index.complete_layer(ticket, layer, g.mean, g.sums),
                "full accepted cross-page completion rejected");
        require(index.published() == (layer == 15),
                "full accepted cross-page index published before all layers completed");
    }
    // Commit and publication must work without any generic trim call.
    require(index.frontier() == 79 && index.base_frontier() == 64,
            "full accepted cross-page Main frontier incorrect");
    for (int layer = 0; layer < 16; ++layer)
        require(std::equal(index.layer_data(layer).begin(), index.layer_data(layer).end(),
                           g.mean.begin(), g.mean.end()),
                "full accepted cross-page published means differ from completed oracle patches");
}
} // namespace
int main() {
    try {
        int n = 0;
        auto status = cudaGetDeviceCount(&n);
        if (status != cudaSuccess || !n)
            return 77;
        std::cout << "numerical\n";
        numerical();
        std::cout << "owner\n";
        owner();
        std::cout << "trim_replay\n";
        trim_replay();
        std::cout << "provisional\n";
        provisional();
        std::cout << "full_accept_cross_page\n";
        full_accept_cross_page();
        std::cout << "MeanK FP64 oracle, snapshots, generation and accepted-only tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}

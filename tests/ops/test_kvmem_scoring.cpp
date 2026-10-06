#include "core/arena.h"
#include "core/device.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "ninfer/ops/kvmem_score.h"
#include "core/kvmem/selection_plan.h"
#include "core/kvmem/scoring_resources.h"
using namespace ninfer;
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
    require(b, "invalid accepted");
}
using namespace ninfer::kvmem;
double bf(std::uint16_t x) {
    return std::bit_cast<float>(std::uint32_t(x) << 16);
}
double hf(std::uint16_t x) {
    int e = (x >> 10) & 31, m = x & 1023;
    return (x & 32768 ? -1 : 1) *
           (e ? std::ldexp(1. + m / 1024., e - 15) : std::ldexp(double(m), -24));
}
template <int P = 37, int D = 256, int H = 24, int K = 4, int M = 16, int L = 16> void gpu() {
    std::cout << "[kernel-case] score P=" << P << " D=" << D << " H=" << H
              << " K=" << K << " M=" << M << " L=" << L << '\n' << std::flush;
    auto plan =
        plan_scoring_resources(4096, 256, 24, 4, 16, 9 * 1024 * 1024, 7 * 1024 * 1024, 64 * 1024);
    require(plan.index_bytes == 8 * 1024 * 1024 && plan.query_bytes == 192 * 1024 &&
                plan.logits_bytes == 6 * 1024 * 1024,
            "262K resource proof");
    rejects([&] {
        (void)plan_scoring_resources(4096, 256, 24, 4, 16, 9 * 1024 * 1024, 6 * 1024 * 1024 - 1,
                                     64 * 1024);
    });
    rejects([&] {
        (void)plan_scoring_resources(UINT32_MAX, 1, 64, 1, 16, SIZE_MAX, SIZE_MAX, SIZE_MAX);
    });
    DeviceBuffer q(D * H * M * 2), k(D * K * P * 2), logs(P * H * M * 4), stats(H * M * 2 * 4),
        rows(H * M * 4), scores(P * 4), status(4);
    cudaStream_t stream;
    require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess, "stream");
    std::vector<std::uint16_t> a(D * H * M), b(D * K * P);
    std::vector<float> actual(P), last(P);
    std::vector<double> oracle(P);
    ops::ScoreWorkspace w{Tensor(logs.p, DType::FP32, {P, H * M}),
                          Tensor(stats.p, DType::FP32, {2, H * M}),
                          Tensor(rows.p, DType::I32, {H * M}), Tensor(scores.p, DType::FP32, {P}),
                          Tensor(status.p, DType::I32, {1})};
    double worst = 0;
    for (int mode = 0; mode < 3; ++mode) {
        auto domain = make_scoring_domain(P * 64 - 7, 18, 2, 130, mode == 1);
        require(domain.first == (mode == 1 ? 0 : 2) && domain.end == (mode == 1 ? P : P - 2),
                "reference domain bands");
        std::fill(oracle.begin(), oracle.end(), 0);
        for (int repeat = 0; repeat < 2; ++repeat) {
            for (int l = 0; l < L; ++l) {
                for (std::size_t i = 0; i < a.size(); ++i)
                    a[i] = std::uint16_t((mode == 2 ? 0x4280 : 0x3d00) + ((i * 7 + l * 11) % 127));
                for (std::size_t i = 0; i < b.size(); ++i)
                    b[i] = std::uint16_t(0x2800 + ((i * 17 + l * 29) % 2048));
                q.copy_from_host(a.data(), a.size() * 2);
                k.copy_from_host(b.data(), b.size() * 2);
                CUDA_CHECK(cudaDeviceSynchronize());
                ops::kvmem_score_layer(Tensor(q.p, DType::BF16, {D, H, M}),
                                       Tensor(k.p, DType::FP16, {D, K, P}), domain.first,
                                       domain.end, L, l == 0, w, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                if (!repeat)
                    for (int h = 0; h < H; ++h)
                        for (int t = 0; t < M; ++t) {
                            std::vector<double> logits(P);
                            double mx = -std::numeric_limits<double>::infinity(), sum = 0;
                            for (int p = domain.first; p < int(domain.end); ++p) {
                                double s = 0;
                                for (int d = 0; d < D; ++d)
                                    s += bf(a[d + D * (h + H * t)]) *
                                         hf(b[d + D * (h / (H / K) + K * p)]);
                                logits[p] = s / std::sqrt(double(D));
                                mx = std::max(mx, logits[p]);
                            }
                            for (int p = domain.first; p < int(domain.end); ++p)
                                sum += std::exp(logits[p] - mx);
                            for (int p = domain.first; p < int(domain.end); ++p)
                                oracle[p] += std::exp(logits[p] - mx) / sum / (L * H);
                        }
            }
            scores.copy_to_host(actual.data(), P * 4);
            int error;
            status.copy_to_host(&error, 4);
            require(error == 0, "finite score status");
            if (repeat)
                require(std::memcmp(actual.data(), last.data(), P * 4) == 0, "bitwise repeat");
            else
                last = actual;
        }
        double err = 0, norm = 0, mass = 0;
        for (int p = 0; p < P; ++p) {
            err += std::pow(actual[p] - oracle[p], 2);
            norm += oracle[p] * oracle[p];
            mass += actual[p];
        }
        double rel = std::sqrt(err / norm);
        worst = std::max(worst, rel);
        require(rel <= 1e-4 && std::abs(mass - M) < 1e-3, "FP64 whole-vector/mass");
        SelectionInput in;
        in.frontier = P * 64 - 7;
        in.physical_pages = 18;
        in.binding_generation = in.score_generation = 1;
        in.query_spans = {{64, 65}};
        in.scores = actual;
        auto selected = select_pages(in).selected;
        // Independent FP64 selection: page 1 is hard; the remaining 17 places
        // use score descending, then logical ID descending, without FP32 casts.
        std::vector<std::uint32_t> ranking;
        for (int p = 0; p < P; ++p)
            if (p != 1)
                ranking.push_back(p);
        std::sort(ranking.begin(), ranking.end(), [&](auto a, auto b) {
            return oracle[a] != oracle[b] ? oracle[a] > oracle[b] : a > b;
        });
        double threshold = oracle[ranking[16]];
        double eps = 1e-6 * std::max(1., std::abs(threshold));
        std::vector<std::uint32_t> expected{1};
        expected.insert(expected.end(), ranking.begin(), ranking.begin() + 17);
        std::sort(expected.begin(), expected.end());
        std::cout << "domain=" << mode << " relL2=" << rel << " epsilon=" << eps << " selected=";
        for (auto p : selected)
            std::cout << p << ',';
        std::cout << " oracle=";
        for (auto p : expected)
            std::cout << p << ',';
        std::cout << '\n';
        std::vector<std::uint32_t> difference;
        std::set_symmetric_difference(selected.begin(), selected.end(), expected.begin(),
                                      expected.end(), std::back_inserter(difference));
        for (auto p : difference)
            require(p != 1 && std::abs(oracle[p] - threshold) <= eps,
                    "FP64 selected set differs outside fixed threshold epsilon");
        std::cout << "threshold=" << threshold << " boundary_differences=" << difference.size()
                  << '\n';
    }
    a[0] = 0x7fc0;
    q.copy_from_host(a.data(), a.size() * 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    ops::kvmem_score_layer(Tensor(q.p, DType::BF16, {D, H, M}), Tensor(k.p, DType::FP16, {D, K, P}),
                           2, P - 2, L, true, w, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int error;
    status.copy_to_host(&error, 4);
    scores.copy_to_host(actual.data(), P * 4);
    require(error == 1 && std::all_of(actual.begin(), actual.end(), [](float f) { return f == 0; }),
            "NaN must erase stale scores");
    a[0] = 0x3d00;
    q.copy_from_host(a.data(), a.size() * 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto run_layer = [&](bool reset) {
        ops::kvmem_score_layer(Tensor(q.p, DType::BF16, {D, H, M}),
                               Tensor(k.p, DType::FP16, {D, K, P}), 2, P - 2, L, reset, w, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        status.copy_to_host(&error, 4);
        scores.copy_to_host(actual.data(), P * 4);
    };
    run_layer(false);
    require(error == 1 && std::all_of(actual.begin(), actual.end(), [](float f) { return f == 0; }),
            "nonfinite status must remain sticky on a finite later layer");
    run_layer(true);
    require(error == 0 && std::any_of(actual.begin(), actual.end(), [](float f) { return f > 0; }),
            "new capture resets invalid prior capture");
    // Excluded pages still belong to the capture and must contain valid Mean-K.
    b[0] = 0x7e00;
    k.copy_from_host(b.data(), b.size() * 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_layer(true);
    require(error == 1 && std::all_of(actual.begin(), actual.end(), [](float f) { return f == 0; }),
            "excluded page NaN must invalidate capture");
    rejects([&] {
        ops::kvmem_score_layer(Tensor(q.p, DType::BF16, {D, H, 0}),
                               Tensor(k.p, DType::FP16, {D, K, P}), 0, P, L, true, w, stream);
    });
    auto overlap = w;
    overlap.scores = Tensor(q.p, DType::FP32, {P});
    rejects([&] {
        ops::kvmem_score_layer(Tensor(q.p, DType::BF16, {D, H, M}),
                               Tensor(k.p, DType::FP16, {D, K, P}), 0, P, L, true, overlap, stream);
    });
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "max_relL2=" << worst << '\n';
}
void cpu() {
    require(make_scoring_domain(65, 1, 0, 64, false).first == 0, "no empty masked domain");
    require(make_scoring_domain(65, 2, 1, 64, false).first == 0, "everything fits no mask");
    rejects([] { (void)make_scoring_domain(0, 1, 0, 0, false); });
    SelectionInput i;
    i.frontier = 640;
    i.physical_pages = 5;
    i.future_reserve = 1;
    i.provisional_guards = 1;
    i.binding_generation = i.score_generation = 3;
    i.query_spans = {{128, 129}};
    std::vector<float> scores(10, 1);
    i.scores = scores;
    i.current_resident = {0, 2, 4};
    auto p = select_pages(i);
    require(p.hard_logical_ids == std::vector<std::uint32_t>({2}), "exact hard excludes scored candidates");
    require(preflight_selection(i).hard_logical_ids == p.hard_logical_ids, "preflight shares hard policy");
    require(p.selected == std::vector<std::uint32_t>({2, 8, 9}) &&
                p.retained == std::vector<std::uint32_t>({2}) &&
                p.added == std::vector<std::uint32_t>({8, 9}) &&
                p.removed == std::vector<std::uint32_t>({0, 4}),
            "newer ties/diff/exact reserves");
    i.current_input_spans = {{192, 640}};
    p = select_pages(i);
    require(p.hard_logical_ids == std::vector<std::uint32_t>({2}) &&
                preflight_selection(i).hard_logical_ids == p.hard_logical_ids, "softened current is not hard");
    require(p.current_input_softened_pages == 7, "long current input softens");
    i.physical_pages = 12;
    p = select_pages(i);
    require(p.selected.size() == 10 && p.current_input_softened_pages == 0, "current fits hard");
    // The shared boundary closes three pages; capacity two must reject it.
    i.physical_pages = 4;
    i.current_input_spans = {};
    i.image_groups = {{{{64, 130}}}, {{{129, 193}}}};
    rejects([&] { select_pages(i); });
    i.physical_pages = 6;
    p = select_pages(i);
    require(p.selected == std::vector<std::uint32_t>({1, 2, 3, 9}),
            "transitive boundary image closure/query hard");
    i.image_groups = {{{{192, 448}}}};
    i.physical_pages = 5;
    scores[3] = 100;
    p = select_pages(i);
    require(p.selected == std::vector<std::uint32_t>({2, 8, 9}) && p.skipped_image_groups == 1,
            "oversized historic image skipped whole");
    i.frontier = 641;
    i.physical_pages = 6;
    i.image_groups = {};
    scores.resize(11, 1);
    i.scores = scores;
    i.recent_tokens = 128;
    p = select_pages(i);
    require(p.recent_pages == 3, "partial page recent full coverage");
    i.sink_pages = 2;
    rejects([&] { select_pages(i); });
    i.sink_pages = 0;
    i.recent_tokens = 0;
    i.score_generation = 2;
    rejects([&] { select_pages(i); });
    i.score_generation = 3;
    i.query_spans = {};
    rejects([&] { select_pages(i); });
    i.query_spans = {{0, 642}};
    rejects([&] { select_pages(i); });
    i.query_spans = {{128, 129}};
    scores[0] = NAN;
    rejects([&] { select_pages(i); });
    SelectionInput atomic;
    atomic.frontier = 640;
    atomic.physical_pages = 2;
    atomic.binding_generation = atomic.score_generation = 1;
    atomic.query_spans = {{64, 65}};
    std::vector<float> atomic_scores(10, 1);
    atomic_scores[9] = 100;
    atomic.scores = atomic_scores;
    atomic.image_groups = {{{{64, 65}, {256, 257}}}};
    require(select_pages(atomic).selected == std::vector<std::uint32_t>({1, 4}),
            "disjoint spans of one image must close hard pages atomically");
    atomic.physical_pages = 1;
    rejects([&] { (void)select_pages(atomic); });
    atomic.query_spans = {{576, 577}};
    atomic.physical_pages = 3;
    atomic_scores[1] = 200;
    require(select_pages(atomic).selected == std::vector<std::uint32_t>({1, 4, 9}),
            "disjoint historic image spans selected as one candidate");
    atomic.physical_pages = 2;
    auto skipped = select_pages(atomic);
    require(skipped.selected == std::vector<std::uint32_t>({8, 9}) &&
                skipped.skipped_image_groups == 1,
            "disjoint candidate cannot be selected partially");
    atomic.image_groups.push_back({{{256, 257}, {448, 449}}});
    atomic.query_spans = {{64, 65}};
    atomic.physical_pages = 3;
    require(select_pages(atomic).selected == std::vector<std::uint32_t>({1, 4, 7}),
            "semantic groups sharing a page close transitively");
    atomic.image_groups.push_back({});
    rejects([&] { (void)select_pages(atomic); });
}
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        return 77;
    try {
        cpu();
        gpu<>();
        // Exact owner row grid (24 heads x two immutable query rows) at both
        // owner logical-index capacities, including kept-band/all-page masks.
        gpu<64, 256, 24, 4, 2, 16>();
        gpu<128, 256, 24, 4, 2, 16>();
        gpu<513, 16, 6, 2, 2, 2>();
        std::cout << "PASS kvmem scoring/selection\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

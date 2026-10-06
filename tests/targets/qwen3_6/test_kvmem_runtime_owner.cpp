#include "targets/qwen3_6/impl/runtime/tiered_context.h"
#include "targets/qwen3_6/impl/runtime/tiered_context_test_access.h"
#include "core/device.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>
using namespace ninfer;
using namespace ninfer::targets::qwen3_6;
using namespace ninfer::targets::qwen3_6::detail;
namespace {
void require(bool value, const char* text) { if (!value) throw std::runtime_error(text); }
template<class F> void rejects(F&& operation, const char* text) {
    bool caught = false;
    try { operation(); } catch (const std::exception&) { caught = true; }
    require(caught, text);
}
std::uint16_t bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    return std::uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
float decode(std::uint16_t value) { return std::bit_cast<float>(std::uint32_t(value) << 16); }
float v_value(std::uint32_t ordinal, std::uint32_t layer) {
    return float(ordinal / 64) * .125f + float(layer + 1) * .015625f;
}
float k_value(std::uint32_t ordinal, std::uint32_t layer) {
    return float(ordinal / 64 + layer + 1) * .015625f;
}
float score_q(std::uint32_t layer, std::uint32_t head, std::uint32_t ordinal) {
    return float(int((layer * 7 + head * 3 + ordinal * 5) % 29) - 14) * .25f;
}
float score_k(std::uint32_t layer, std::uint32_t head, std::uint32_t page) {
    return float(int((page * 19 + layer * 11 + head * 7) % 37) - 18) * .5f;
}
struct Fixture {
    static constexpr std::uint32_t chunk = 64;
    std::uint32_t context = 4096, resident = 16;
    DeviceContext device;
    LayoutBuilder builder;
    PagedKVCacheLayout layout;
    std::unique_ptr<DeviceBuffer> pool_memory, backing;
    std::unique_ptr<PagedKVCache> main;
    PagedKVAllocation lease;
    TieredRuntimePlan plan;
    std::unique_ptr<TieredContext> owner;
    DeviceBuffer q{256 * 24 * chunk * 2}, k{256 * 4 * chunk * 2}, v{k.bytes},
        out{q.bytes}, positions{chunk * sizeof(std::int32_t)};
    std::vector<std::int32_t> ids;
    kvmem::QueryProvenance provenance;
    bool mtp{}, nonuniform_score{}, rk4{}, int8{};
    std::vector<std::array<std::vector<std::byte>,4>> encoded_history;
    double prefill_elapsed_ms{};
    Fixture(HostKVArchiveMode mode, bool speculative, std::uint32_t recent = 128,
            std::uint32_t max_context = 4096, std::uint32_t view_pages = 16, bool nonuniform = false,
            bool rotated = false, bool compressed_int8 = false)
        : context(max_context), resident(view_pages), mtp(speculative), nonuniform_score(nonuniform),
          rk4(rotated), int8(compressed_int8) {
        DecoderStateSpec spec;
        spec.capacity = context; spec.full_attention_layers = 16; spec.kv_heads = 4;
        spec.attention_head_dim = 256; spec.text_physical_page_groups = resident;
        if (rk4 || int8) {
            spec.kv_dtype = DType::I8; spec.kv_quant_group = 64;
            spec.kv_packed_k = spec.kv_packed_v = spec.kv_rotate_k = spec.kv_rotate_v = spec.kv_e8_lattice = rk4;
        }
        spec.allow_tiered_text_pages = true; spec.linear_attention = {1,1,1,1,1,1};
        layout = plan_decoder_state(builder, spec).text_kv;
        pool_memory = std::make_unique<DeviceBuffer>(builder.finish(256));
        main = std::make_unique<PagedKVCache>(DeviceSpan{pool_memory->p, pool_memory->bytes}, layout);
        lease = main->pool().reserve(resident);
        lease.materialize_pages(resident, device.stream); lease.bind_row(0, device.stream);
        ids.assign(lease.page_ids().begin(), lease.page_ids().end());
        std::reverse(ids.begin(), ids.end());
        TieredKVOptions options;
        options.view_tokens = resident * 64; options.sink_tokens = 64;
        options.recent_tokens = recent; options.gen_reserve_tokens = 128;
        options.query_tokens = 2; options.host_archive = mode;
        plan = plan_tiered_runtime(layout.pool, context, resident, chunk, options, false, true, KvMode::KVMem);
        if (rk4 || int8) {
            encoded_history.resize(16);
            for (unsigned layer = 0; layer < 16; ++layer)
                for (unsigned plane = 0; plane < 4; ++plane)
                    encoded_history[layer][plane].resize(plan.archive.logical_pages * plan.archive.layers[layer][plane].page_bytes);
        }
        backing = std::make_unique<DeviceBuffer>(plan.bytes);
        owner = std::make_unique<TieredContext>(plan, *main, DeviceSpan{backing->p, backing->bytes}, device.stream);
        owner->bind_pages(ids);
        CUDA_CHECK(cudaMemsetAsync(q.p, 0, q.bytes, device.stream));
        CUDA_CHECK(cudaMemsetAsync(k.p, 0, k.bytes, device.stream));
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        provenance.bundle_identity = 111; provenance.lineage = 222; provenance.capture_epoch = 1;
        provenance.covered_frontier = 66; provenance.ordinals = {64,65}; provenance.source_prefix[0] = 77;
        const auto loading = owner->loading_info();
        require(loading.archive_mode == (mode == HostKVArchiveMode::Pinned ? kvmem::HostArchiveMode::Pinned :
                kvmem::HostArchiveMode::Pageable), "actual combined pin mode");
        require(owner->sparse_capture()->index().pinned() == (mode == HostKVArchiveMode::Pinned),
                "archive and index actual pin mode must agree");
        require(loading.owned_device_bytes == plan.owned_device_bytes && loading.device_backing_bytes == plan.bytes,
                "loading accurately exposes own capture and arena device bytes");
        owner->begin_query(provenance, 111);
    }
    std::vector<std::uint32_t> resident_ids() {
        std::vector<std::uint32_t> result;
        for (const auto& page : owner->current_view().resident) result.push_back(page.logical_page);
        return result;
    }
    // Independent FP64 full or subset causal softmax: Q=K=0, all valid original
    // logical ordinals have equal probability. No compact positions are used.
    double oracle(std::uint32_t pos, std::uint32_t layer, const std::vector<std::uint32_t>& pages, bool sparse) {
        double total = 0; std::uint32_t count = 0;
        for (std::uint32_t ordinal = 0; ordinal <= pos; ++ordinal)
            if (!sparse || std::binary_search(pages.begin(), pages.end(), ordinal / 64)) {
                total += decode(bf16(v_value(ordinal, layer))); ++count;
            }
        require(count != 0, "oracle visible keys");
        return total / count;
    }
    static double half_value(std::uint16_t h) {
        const auto e = (h >> 10) & 31, m = h & 1023;
        require(e != 31, "encoded FP16 scale finite");
        return std::ldexp(double(e ? 1024 + m : m), e ? int(e) - 25 : -24) * (h & 0x8000 ? -1 : 1);
    }
    void retain_encoded(std::uint32_t layer, std::uint32_t first, std::uint32_t count) {
        const auto view = owner->current_view();
        for (unsigned plane = 0; plane < 4; ++plane) {
            const auto& spec = plan.archive.layers[layer][plane];
            const auto& tensor = main->pool().plane(spec.pool_plane);
            for (auto page = first / 64; page < (first + count + 63) / 64; ++page)
                CUDA_CHECK(cudaMemcpy(encoded_history[layer][plane].data() + page * spec.page_bytes,
                    static_cast<const std::byte*>(tensor.data) + ids.at(view.blocktable[page]) * tensor.nb[3],
                    spec.page_bytes, cudaMemcpyDeviceToHost));
        }
    }
    void stored_code_oracle(const std::vector<std::uint16_t>& observed, std::uint32_t layer,
                        std::uint32_t first, std::uint32_t count,
                        const std::vector<std::uint32_t>& pages, bool sparse) {
        std::array<double,1024> sum{};
        unsigned keys = 0;
        auto add = [&](unsigned ordinal) {
            if (sparse && !std::binary_search(pages.begin(), pages.end(), ordinal / 64)) return;
            ++keys;
            const auto& codes = encoded_history[layer][1];
            const auto& scales = encoded_history[layer][3];
            for (unsigned head = 0; head < 4; ++head)
                for (unsigned d = 0; d < 256; ++d) {
                    const auto row = (ordinal / 64 * 4 + head) * 64 + ordinal % 64;
                    int code;
                    if (rk4) {
                        const auto byte = std::to_integer<unsigned>(codes[row * 128 + d / 2]);
                        const int nibble = (byte >> (4 * (d % 2))) & 15;
                        code = nibble < 8 ? nibble : nibble - 16;
                    } else {
                        std::int8_t stored;
                        std::memcpy(&stored, codes.data() + row * 256 + d, 1);
                        code = stored;
                    }
                    std::uint16_t scale;
                    std::memcpy(&scale, scales.data() + (row * 4 + d / 64) * 2, 2);
                    sum[head * 256 + d] += code * half_value(scale);
                }
        };
        for (unsigned ordinal = 0; ordinal < first; ++ordinal) add(ordinal);
        for (unsigned token = 0; token < count; ++token) {
            add(first + token);
            if (!sparse && token + 1 != count) continue;
            auto expected = sum;
            // Independent normalized Sylvester inverse H64, in FP64. Uniform
            // softmax permits averaging stored quantized V before this linear transform.
            if (rk4) for (unsigned head = 0; head < 4; ++head)
                for (unsigned group = 0; group < 4; ++group) {
                    auto* values = expected.data() + head * 256 + group * 64;
                    for (unsigned step = 1; step < 64; step *= 2)
                        for (unsigned base = 0; base < 64; base += 2 * step)
                            for (unsigned i = 0; i < step; ++i) {
                                const auto a = values[base + i], b = values[base + step + i];
                                values[base + i] = a + b; values[base + step + i] = a - b;
                            }
                    for (unsigned d = 0; d < 64; ++d) values[d] /= 8. * keys;
                }
            else for (auto& value : expected) value /= keys;
            for (unsigned head = 0; head < 24; ++head)
                for (unsigned d = 0; d < 256; ++d) {
                    const auto reference = expected[head / 6 * 256 + d];
                    require(std::abs(decode(observed[token * 6144 + head * 256 + d]) - reference) <=
                        .0041 * std::max(1., std::abs(reference)),
                        rk4 ? "owner RK4 stored-code FP64 subset softmax/inverse-H64 oracle" :
                              "owner INT8 signed-code/FP16-scale FP64 causal-subset softmax oracle");
                }
        }
    }
    void run(std::uint32_t count, bool sparse = false, std::uint32_t padded = 0) {
        const auto first = owner->frontier(), width = std::max(count, padded);
        require(width <= chunk, "fixture width");
        if (sparse) owner->prepare_main_transaction(mtp ? kvmem::MeanKTransactionKind::SpeculativeMain :
                kvmem::MeanKTransactionKind::OrdinaryMain, count);
        std::vector<std::int32_t> pos(width);
        std::iota(pos.begin(), pos.begin() + count, int(first));
        CUDA_CHECK(cudaMemcpyAsync(positions.p, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice, device.stream));
        owner->begin_block(first, count, device.stream, sparse ? TieredContext::ExecutionPhase::Decode :
                                                                  TieredContext::ExecutionPhase::Prefill);
        const auto pages = resident_ids();
        Tensor query(q.p, DType::BF16, {256,24,int(width)}), keys(k.p, DType::BF16, {256,4,int(width)}),
            values(v.p, DType::BF16, {256,4,int(width)}), output(out.p, DType::BF16, {256,24,int(width)}),
            position(positions.p, DType::I32, {int(width)});
        for (std::uint32_t layer = 0; layer < 16; ++layer) {
            std::vector<std::uint16_t> input(width * 1024, 0x7fc0); // poison invalid padded columns
            for (std::uint32_t token = 0; token < count; ++token)
                std::fill_n(input.begin() + token * 1024, 1024, bf16(v_value(first + token, layer)));
            std::vector<std::uint16_t> input_k(width * 1024, 0x7fc0);
            for (std::uint32_t token = 0; token < count; ++token)
                std::fill_n(input_k.begin() + token * 1024, 1024, bf16(k_value(first + token, layer)));
            std::vector<std::uint16_t> input_q;
            if (nonuniform_score) {
                input_q.resize(width * 6144);
                std::fill(input_k.begin(), input_k.end(), std::uint16_t(0));
                for (std::uint32_t token = 0; token < count; ++token) {
                    for (std::uint32_t head = 0; head < 4; ++head)
                        input_k[token * 1024 + head * 256] = bf16(score_k(layer, head, (first + token) / 64));
                    for (std::uint32_t head = 0; head < 24; ++head)
                        input_q[token * 6144 + head * 256] = bf16(score_q(layer, head, first + token));
                }
                CUDA_CHECK(cudaMemcpyAsync(q.p, input_q.data(), input_q.size() * 2, cudaMemcpyHostToDevice, device.stream));
            }
            CUDA_CHECK(cudaMemcpyAsync(k.p, input_k.data(), input_k.size() * 2, cudaMemcpyHostToDevice, device.stream));
            CUDA_CHECK(cudaMemcpyAsync(v.p, input.data(), input.size() * 2, cudaMemcpyHostToDevice, device.stream));
            owner->capture_pre_rope(layer, query, keys, device.stream);
            owner->attention(layer, query, keys, values, position, .0625f, output, device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            std::vector<std::uint16_t> observed(width * 6144);
            out.copy_to_host(observed.data(), observed.size() * 2);
            if (rk4 || int8) {
                retain_encoded(layer, first, count);
                stored_code_oracle(observed, layer, first, count, pages, sparse);
            }
            if (!rk4 && !int8 && !nonuniform_score) for (std::uint32_t token = 0; token < count; ++token) {
                const auto reference = oracle(first + token, layer, pages, sparse);
                for (int element = 0; element < 6144; ++element)
                    require(std::abs(decode(observed[token * 6144 + element]) - reference) <=
                        .0041 * std::max(1., std::abs(reference)), "owner attention independent FP64 original-ordinal oracle");
            }
            require(std::all_of(observed.begin() + count * 6144, observed.end(), [](auto x) { return x == 0; }),
                    "invalid padded output columns must be BF16 zero");
        }
        if (!sparse) {
            owner->finish_exact_block();
            require(owner->sparse_capture()->index().frontier() == owner->frontier(), "exact capture precedes snapshot boundary");
        }
    }
    void prefill(std::uint32_t target) {
        const auto start = std::chrono::steady_clock::now();
        while (owner->frontier() < target) run(std::min(chunk, target - owner->frontier()));
        owner->drain();
        prefill_elapsed_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    SparseTurnInput turn(bool all_hard = false) {
        SparseTurnInput input;
        input.query = provenance; input.bundle_identity = 111; input.prefill_ms = prefill_elapsed_ms;
        input.current_input_spans = {{128,owner->frontier()}}; // long event softens
        input.image_groups = all_hard ? std::vector<kvmem::ImageGroup>{{{{64,65},{128,129},{192,193}}}} :
            std::vector<kvmem::ImageGroup>{{{{64,65},{256,257},{640,641},{1024,1025},{1408,1409}}}};
        return input;
    }
    void byte_oracle() {
        const auto view = owner->current_view();
        if (rk4 || int8) {
            for (unsigned layer = 0; layer < 16; ++layer)
                for (unsigned plane = 0; plane < 4; ++plane)
                    for (const auto& page : view.resident) {
                        const auto& spec = plan.archive.layers[layer][plane];
                        const auto& tensor = main->pool().plane(spec.pool_plane);
                        std::vector<std::byte> actual(spec.page_bytes);
                        CUDA_CHECK(cudaMemcpy(actual.data(), static_cast<const std::byte*>(tensor.data) +
                            ids.at(page.physical_slot) * tensor.nb[3], spec.page_bytes, cudaMemcpyDeviceToHost));
                        require(std::memcmp(actual.data(), encoded_history[layer][plane].data() +
                            page.logical_page * spec.page_bytes, spec.page_bytes) == 0,
                            rk4 ? "all64 RK4 code/scale planes preserve original logical page bytes" :
                                  "all64 INT8 code/scale planes preserve original logical page bytes");
                    }
            return;
        }
        for (const auto& layer : plan.archive.layers)
            for (const auto& plane : layer)
                for (const auto& page : view.resident) {
                    const auto& tensor = main->pool().plane(plane.pool_plane);
                    std::vector<std::uint16_t> bytes(plane.page_bytes / 2);
                    CUDA_CHECK(cudaMemcpy(bytes.data(), static_cast<const std::byte*>(tensor.data) +
                        ids.at(page.physical_slot) * tensor.nb[3], plane.page_bytes, cudaMemcpyDeviceToHost));
                    const auto valid = std::min(64U, view.frontier - page.logical_page * 64U);
                    for (int head = 0; head < 4; ++head)
                        for (std::uint32_t token = 0; token < valid; ++token)
                            for (int d = 0; d < 256; ++d) {
                                const auto observed = bytes[(head * tensor.nb[2] + token * tensor.nb[1]) / 2 + d];
                                const auto expected = plane.pool_plane % 2 ? bf16(v_value(page.logical_page * 64 + token,
                                        std::uint32_t(plane.pool_plane / 2))) :
                                    bf16(k_value(page.logical_page * 64 + token, std::uint32_t(plane.pool_plane / 2)));
                                require(observed == expected, "all16-layer restored original KV plane byte oracle");
                            }
                }
    }
};
void exercise(HostKVArchiveMode mode, bool mtp) {
    Fixture f(mode, mtp);
    f.prefill(3905); // near max frontier, with partial recent/query/image pages and far holes
    const auto old = f.owner->current_view();
    rejects([&] { auto wrong = f.turn(); wrong.bundle_identity = 112; (void)f.owner->select_and_publish(wrong); },
            "foreign caller query binding must fail before view mutation");
    require(f.owner->current_view().blocktable == old.blocktable, "pure rejection leaves prior view valid");
    const auto result = f.owner->select_and_publish(f.turn());
    require(result.selection.current_input_softened_pages > 0 && result.selection.hard_logical_ids.size() == 9,
            "D9 image closure and long-current soften exact hard policy");
    require(result.selection.selected.size() == 12 && result.denominator_pages == 59 && !result.all_pages_denominator &&
            result.domain_first_page == 1 && result.domain_end_page == 60,
            "default reference denominator excludes configured kept bands only");
    for (std::uint32_t page = 0; page < result.page_scores.size(); ++page)
        require(std::abs(result.page_scores[page] - (page >= 1 && page < 60 ? 2.f / 59 : 0.f)) < 2e-6f,
                "owner 16-layer GPU scoring independent uniform oracle");
    require(result.scoring_h2d_bytes == 16ULL * (62 * 256 * 4 * 2 + 2 * 256 * 24 * 2),
            "truthful exact scoring index+Q H2D bytes");
    require(result.scoring_h2d_ms > 0 && result.scoring_compute_ms > 0 && result.score_d2h_ms > 0 &&
            result.selection_ms > 0 && result.publish_ms > 0, "measured stage durations");
    f.byte_oracle();
    auto saved = f.owner->capture();
    const auto selected = f.owner->current_view();
    const auto index_generation = f.owner->sparse_capture()->index().generation();
    for (int corrupt = 0; corrupt < 4; ++corrupt) {
        auto bad = saved;
        if (!corrupt) bad.sparse.reset();
        else if (corrupt == 1) {
            auto foreign = std::make_shared<SparseDerivedSnapshot>(*saved.sparse);
            ++foreign->owner; bad.sparse = foreign;
        } else if (corrupt == 2) bad.exact_hard_logical_ids.push_back(2);
        else bad.selected_logical_ids.erase(bad.selected_logical_ids.begin());
        rejects([&] { f.owner->restore(bad, f.device.stream); }, "invalid restore metadata rejected BEFORE victim DMA");
        require(!f.owner->poisoned() && f.owner->current_view().generation == selected.generation &&
            f.owner->current_view().blocktable == selected.blocktable &&
            f.owner->sparse_capture()->index().generation() == index_generation,
            "pure invalid restore leaves old view, index generation and protection valid");
        f.byte_oracle();
    }

    const auto streamed_before = result.hydrate_bytes;
    (void)streamed_before;
    f.run(mtp ? 3U : 1U, true, mtp ? 8U : 4U);
    f.owner->flush_accepted_frontier(mtp ? 2 : 1);
    require(f.owner->frontier() == 3905 + (mtp ? 2 : 1) &&
            f.owner->sparse_capture()->index().frontier() == f.owner->frontier(), "final application acceptance frontier");
    f.owner->restore(saved, f.device.stream);
    require(f.resident_ids() == saved.selected_logical_ids && f.owner->sparse_phase() == SparseExecutionPhase::SparseDecode,
            "restore selected original IDs and sparse phase from current owners");
    f.byte_oracle();
    auto restored_query = saved.sparse->query;
    f.owner->use_query(restored_query, f.provenance, 111);
    const auto refill_bytes = f.owner->prepare_exact_prefill();
    const auto refilled = f.owner->current_view();
    require(refilled.resident.size() == 16 && refill_bytes == 4ULL * 16 * 2 * 256 * 4 * 64 * 2,
            "arbitrary sparse set must refill full V before next exact prefill");
    for (auto page : saved.selected_logical_ids)
        require(refilled.blocktable[page] == selected.blocktable[page], "next exact prefill preserves physical intersection");
    f.run(64); // full-history oracle with >ticket-capacity fragmented logical holes
    f.owner->use_query(restored_query, f.provenance, 111);
    f.provenance.capture_epoch = 99; // capture epoch may legally differ from restored index epoch
#if defined(_WIN32)
    _putenv_s("NINFER_KVMEM_SCORE_ALL_PAGES", "1");
#else
    setenv("NINFER_KVMEM_SCORE_ALL_PAGES", "1", 1);
#endif
    const auto all = f.owner->select_and_publish(f.turn());
    require(all.domain_first_page == 0 && all.domain_end_page == 63 && all.denominator_pages == 63 && all.all_pages_denominator,
            "hidden all-pages denominator bound to owner");
#if defined(_WIN32)
    _putenv_s("NINFER_KVMEM_SCORE_ALL_PAGES", "");
#else
    unsetenv("NINFER_KVMEM_SCORE_ALL_PAGES");
#endif
    f.run(mtp ? 4U : 1U, true, mtp ? 8U : 4U);
    const auto full = f.owner->frontier();
    f.owner->flush_accepted_frontier(mtp ? 4 : 1); // full accept, no trim path
    require(f.owner->sparse_capture()->index().frontier() == full, "full acceptance must flush without trim");
    f.run(mtp ? 3U : 1U, true, mtp ? 8U : 4U);
    f.owner->flush_accepted_frontier(0);
    require(f.owner->frontier() == full && f.owner->sparse_capture()->index().frontier() == full, "zero/cancel discards provisional Main");
    f.owner->drain(); f.byte_oracle();
    saved = {}; restored_query.reset();
    f.owner->reset(f.device.stream);
    rejects([&] { (void)f.owner->select_and_publish(f.turn()); }, "reset invalidates old Q/list/binding");
}
void nonuniform_owner_score() {
    Fixture f(HostKVArchiveMode::Pinned, false, 128, 4096, 16, true);
    f.prefill(1089);
    SparseTurnInput input;
    input.query = f.provenance; input.bundle_identity = 111; input.prefill_ms = f.prefill_elapsed_ms;
    const auto result = f.owner->select_and_publish(input);
    std::vector<double> expected(18);
    // Independent scalar FP64 global softmax. Only dim-0 is nonzero. Values
    // are exact BF16/FP16 binary fractions, so this also detects layer/head/
    // ordinal offsets, captured Q restoration and fixed-stride index mistakes.
    for (std::uint32_t layer = 0; layer < 16; ++layer)
        for (std::uint32_t head = 0; head < 24; ++head)
            for (const auto ordinal : f.provenance.ordinals) {
                std::vector<double> logits(18);
                double maximum = -INFINITY, denominator = 0;
                for (std::uint32_t page = 1; page < 16; ++page) {
                    logits[page] = double(decode(bf16(score_q(layer, head, ordinal)))) *
                        double(score_k(layer, head / 6, page)) / 16.;
                    maximum = std::max(maximum, logits[page]);
                }
                for (std::uint32_t page = 1; page < 16; ++page)
                    denominator += std::exp(logits[page] - maximum);
                for (std::uint32_t page = 1; page < 16; ++page)
                    expected[page] += std::exp(logits[page] - maximum) / denominator / (16 * 24);
            }
    double error = 0, norm = 0, mass = 0;
    for (std::uint32_t page = 0; page < 18; ++page) {
        error += std::pow(result.page_scores[page] - expected[page], 2);
        norm += expected[page] * expected[page]; mass += result.page_scores[page];
    }
    require(std::sqrt(error / norm) <= 1e-4 && std::abs(mass - 2) < 1e-3,
            "real owner nonuniform capture/H2D/scoring independent FP64 vector/mass");
    const std::vector<std::uint32_t> hard{0,1,15,16,17};
    require(result.selection.hard_logical_ids == hard, "nonuniform owner exact hard set");
    std::vector<std::uint32_t> candidates;
    for (std::uint32_t page = 0; page < 18; ++page)
        if (!std::binary_search(hard.begin(), hard.end(), page)) candidates.push_back(page);
    std::sort(candidates.begin(), candidates.end(), [&](auto a, auto b) {
        return expected[a] != expected[b] ? expected[a] > expected[b] : a > b;
    });
    auto chosen = hard;
    chosen.insert(chosen.end(), candidates.begin(), candidates.begin() + 7);
    std::sort(chosen.begin(), chosen.end());
    require(result.selection.selected == chosen, "owner nonuniform selection agrees with independent FP64 ranking");
    require(result.actual_view_tokens == 1024 && result.selected_valid_tokens == 705,
            "physical V distinct from selected valid key count");
}

void quantized_owner(HostKVArchiveMode mode, bool mtp, bool int8 = false) {
    Fixture f(mode, mtp, 128, 4096, 16, false, !int8, int8);
    f.prefill(1087); // partial final page followed by an actual cross-page transaction
    auto input = f.turn(); input.image_groups.clear();
    input.image_groups = {{{{64,65},{256,257},{640,641}}}};
    const auto selected = f.owner->select_and_publish(input);
    require(selected.selected_valid_tokens == 767, "quantized sparse prefix uses actual valid keys");
    f.byte_oracle();
    auto saved = f.owner->capture();
    f.run(mtp ? 3 : 1, true, 8);
    f.owner->flush_accepted_frontier(mtp ? 2 : 1);
    f.owner->restore(saved, f.device.stream); f.byte_oracle();
    f.run(mtp ? 4 : 1, true, 8);
    f.owner->flush_accepted_frontier(mtp ? 4 : 1);
    f.owner->drain(); f.byte_oracle();
    (void)f.owner->prepare_exact_prefill();
    f.run(32); f.owner->drain(); f.byte_oracle();
}
void fragmented_next_exact() {
    Fixture f(HostKVArchiveMode::Pageable, true, 128, 8192, 40);
    f.prefill(8129);
    auto input = f.turn(); input.image_groups.clear();
    kvmem::ImageGroup image;
    for (std::uint32_t page = 1; page <= 37; page += 2)
        image.spans.push_back({page * 64, page * 64 + 1});
    input.image_groups.push_back(std::move(image));
    const auto result = f.owner->select_and_publish(input);
    require(result.selection.selected.size() == 36 && result.selection.hard_pages == 23,
            "interleaved far retrieval/image closure fixture");
    const auto prior = f.owner->current_view();
    require(prior.host_only.size() > f.plan.maximum_stream_pages,
            "partly empty sparse view would exceed original fixed staging plane capacity");
    (void)f.owner->prepare_exact_prefill();
    const auto filled = f.owner->current_view();
    for (const auto& page : prior.resident)
        require(filled.blocktable[page.logical_page] == page.physical_slot,
                "far interleaved exact refill preserves current physical intersection");
    std::size_t runs = 0;
    for (std::size_t page = 0; page < filled.host_only.size(); ++page)
        runs += !page || filled.host_only[page] != filled.host_only[page-1] + 1;
    require(runs * 2 > f.plan.staging.ticket_capacity,
            "fragmented exact history exceeds one-ticket-per-run startup capacity");
    f.run(32); // complete independent FP64 FULL-history causal attention near max F
    f.owner->drain(); f.byte_oracle();
}

void poison_recovery(HostKVArchiveMode mode) {
    Fixture f(mode, true);
    f.prefill(3905);
    const auto before_score = f.owner->current_view();
    kvmem::HostKVTransferFaultInjection fault;
    fault.reject_h2d_submission_after = 3; // real rejection after two completed index uploads
    f.owner->set_transfer_test_fault(fault);
    bool score_rejected = false;
    try { (void)f.owner->select_and_publish(f.turn()); }
    catch (const kvmem::CudaTransferError&) { score_rejected = true; }
    require(score_rejected && !f.owner->poisoned() &&
        f.owner->current_view().generation > before_score.generation,
        "score DMA rejection drains source borrower and restores prior logical view");
    f.byte_oracle();
    const auto old = f.owner->current_view();
    fault = {}; fault.reject_h2d_submission_after = 18; // 16 scoring layers, first hydrate plane completed, next rejected
    f.owner->set_transfer_test_fault(fault);
    bool typed = false;
    try { (void)f.owner->select_and_publish(f.turn()); }
    catch (const kvmem::CudaTransferError& error) { typed = error.status() == cudaErrorInvalidMemcpyDirection ||
        error.status() == cudaErrorInvalidValue; }
    require(typed && !f.owner->poisoned(), "real nonfatal CUDA hydrate rejection restores healthy prior logical view");
    const auto restored = f.owner->current_view();
    require(restored.generation > old.generation && f.resident_ids().size() == old.resident.size(),
            "poison recovery republishes prior logical view with fresh generation");
    for (const auto& p : old.resident)
        require(restored.blocktable[p.logical_page] >= 0, "every prior logical owner rehydrated");
    f.byte_oracle();
    (void)f.owner->select_and_publish(f.turn());
    fault = {}; fault.reject_d2h_submission_after = 1;
    f.owner->set_transfer_test_fault(fault);
    bool uncertain = false;
    try { f.run(3, true, 8); f.owner->flush_accepted_frontier(3); }
    catch (const KVMemColdResetRequired&) { uncertain = true; }
    catch (const kvmem::CudaTransferError&) { // asynchronous failure may be observed at explicit drain
        uncertain = true;
    }
    require(uncertain, "real D2H uncertainty cannot republish stale authority");
    require(f.owner->poisoned(), "uncertain archive blocks attention immediately");
    rejects([&] { (void)f.owner->resident_layer(0); }, "poison blocks physical attention view access");
    f.owner->reset(f.device.stream);
    require(!f.owner->poisoned() && f.owner->frontier() == 0 && f.owner->sparse_capture()->index().frontier() == 0,
            "nonfatal uncertain authority resets whole Main bundle for caller exact rebuild");
}
void reserve_failure() {
    Fixture f(HostKVArchiveMode::Pageable, true, 512);
    f.prefill(2048);
    const auto result = f.owner->select_and_publish(f.turn(true));
    require(result.selection.hard_pages == 12 && result.selection.selected.size() == 12,
            "all-hard/R>G request admitted when exact hard fits C");
    bool rejected = false;
    for (int block = 0; block < 24 && !rejected; ++block) {
        const auto before = f.owner->current_view();
        try { f.run(16, true); f.owner->flush_accepted_frontier(16); }
        catch (const std::length_error&) {
            rejected = true;
            require(f.owner->current_view().blocktable == before.blocktable && f.owner->frontier() == before.frontier,
                    "no-safe-victim capacity failure is atomic");
        }
    }
    require(rejected && !f.owner->poisoned(), "all-hard reserve exhaustion explicit capacity failure keeps old execution valid");
    f.byte_oracle();
}
void generic_source_failure(HostKVArchiveMode mode) {
    Fixture f(mode, false);
    f.prefill(3905);
    const auto prior = f.owner->current_view();
    kvmem::HostKVTransferFaultInjection fault;
    fault.h2d_copy_after = 2; // generic source failure after two real completed uploads
    f.owner->set_transfer_test_fault(fault);
    bool rejected = false;
    try { (void)f.owner->select_and_publish(f.turn()); }
    catch (const std::runtime_error& error) {
        rejected = std::string_view(error.what()).find("injected host KV failure after completed H2D copy") !=
                   std::string_view::npos;
    }
    require(rejected && f.owner->poisoned(), "generic score source failure remains fail-closed");
    require(f.owner->current_view().generation == prior.generation,
            "generic source failure cannot republish automatic recovery authority");
    rejects([&] { (void)f.owner->resident_layer(0); }, "generic failure blocks Main attention access");
    f.owner->reset(f.device.stream);
    require(!f.owner->poisoned() && f.owner->frontier() == 0,
            "explicit drained cold reset permits full exact rebuild after generic source failure");
}
void restore_before_query() {
    Fixture f(HostKVArchiveMode::Pageable, false);
    f.prefill(64); auto before = f.owner->capture();
    f.run(1); auto inside = f.owner->capture();
    require(!before.sparse->query && !inside.sparse->query, "checkpoint cannot own partially collected Q");
    f.owner->restore(inside, f.device.stream);
    rejects([&] { f.owner->begin_query(f.provenance, 111); }, "missing Q behind first required ordinal explicitly demands earlier exact recapture");
    f.owner->restore(before, f.device.stream);
    f.owner->begin_query(f.provenance, 111); f.run(2);
    require(bool(f.owner->sparse_capture()->query()), "legal exact continuation before Q captures all ordinals");
}
void sticky_capture_failure(cudaError_t status, bool failed_drain) {
    Fixture f(HostKVArchiveMode::Pageable, false);
    f.prefill(66); // publish a real Q source, without corrupting the actual CUDA context
    const auto prior = f.owner->current_view();
    unsigned drains = 0;
    auto borrower = f.owner->sparse_capture()->borrow_query([&] {
        if (++drains == 1)
            throw kvmem::CudaTransferError(status, "injected capture source classification", failed_drain);
    });
    bool classified = false;
    try { f.owner->drain(); }
    catch (const kvmem::CudaTransferError& error) {
        classified = error.status() == status && error.drain_failed() == failed_drain;
    }
    require(classified && f.owner->poisoned(), "capture failure preserves original typed status");
    f.owner->sparse_capture()->drain(); // later successful completion must not erase fatal classification
    require(drains >= 2 && !borrower.valid(), "later real borrower drain succeeds and releases source");
    bool main_reset_rejected = false;
    try { f.owner->reset(f.device.stream); }
    catch (const kvmem::CudaTransferError& error) {
        main_reset_rejected = error.status() == status && error.drain_failed() == failed_drain;
    }
    require(main_reset_rejected && f.owner->poisoned() &&
            f.owner->current_view().generation == prior.generation && f.owner->frontier() == prior.frontier,
            "prior fatal/unknown/failed-drain capture cannot Main-reset after later successful drain");
    bool capture_reset_rejected = false;
    try { f.owner->sparse_capture()->reset(); }
    catch (const kvmem::CudaTransferError& error) {
        capture_reset_rejected = error.status() == status && error.drain_failed() == failed_drain;
    }
    require(capture_reset_rejected && f.owner->sparse_capture()->index().frontier() == prior.frontier,
            "capture reset preserves fatal classification and accepted frontier");
}
void permanent_score_completion_teardown() {
    Fixture f(HostKVArchiveMode::Pageable, false);
    f.prefill(66); f.owner->drain();
    const auto prior = f.owner->current_view();
    auto completion = TieredContextTestAccess::score_query_drain(*f.owner);
    int source = 1, destination = 0;
    const auto returned = cudaMemcpy(&destination, &source, sizeof(source), static_cast<cudaMemcpyKind>(99));
    require(returned == cudaErrorInvalidMemcpyDirection && cudaGetLastError() == returned,
        "permanent completion consume only actual safe kind99 status");
    unsigned attempts = 0;
    auto borrower = f.owner->sparse_capture()->borrow_query([completion, returned, &attempts] {
        completion(); // actual production factory, actual healthy CUDA drain
        ++attempts;
        kvmem::check_transfer_cuda(returned, "permanent score completion test drain", true);
    });
    const auto original = [&](const kvmem::CudaTransferError& error) {
        return error.status() == returned && error.drain_failed() &&
            std::string_view(error.operation()) == "permanent score completion test drain";
    };
    bool caught = false;
    try { TieredContextTestAccess::finish_score_query_borrow(*f.owner, borrower); }
    catch (const kvmem::CudaTransferError& error) { caught = original(error); }
    require(caught && borrower.valid() && attempts == 1 && f.owner->poisoned(),
        "live collector lost permanent typed completion cause or prematurely retired Q");
    caught = false;
    try { f.owner->reset(f.device.stream); } catch (const kvmem::CudaTransferError& error) { caught = original(error); }
    require(caught && borrower.valid() && attempts == 2 && f.owner->frontier() == prior.frontier &&
        f.owner->current_view().generation == prior.generation,
        "permanent completion reset erased original failure or accepted metadata");
    f.owner.reset(); // Main, Sparse and Q pool retry the still-failed completion.
    require(attempts >= 4, "permanent completion did not survive subordinate teardown retries");
    completion(); // Main and its subordinate capture pool are gone; stream is externally owned.
    const auto retired_attempts = attempts;
    // The unresolved borrow is quarantined; teardown clears its callback rather
    // than claiming completion or releasing the possibly borrowed source.
    bool quarantined = false;
    try { borrower.finish(); } catch (const std::bad_function_call&) { quarantined = true; }
    require(quarantined && borrower.valid() && attempts == retired_attempts,
        "detached unresolved Q handle invoked a callback or falsely released its source");
    std::cout << "permanent score completion teardown PASS attempts=" << attempts
        << " returned_status=" << int(returned) << " simulated_failed_completion=1 detached_production_drain=1\n";
}
bool historical_score_release_before_borrower(bool transfer) {
    Fixture f(HostKVArchiveMode::Pageable, false);
    f.prefill(66); f.owner->drain();
    const auto prior = f.owner->current_view();
    auto completion = TieredContextTestAccess::score_query_drain(*f.owner);
    const char* operation = transfer ? "historical score transfer A" : "historical score Sparse A";
    int source = 1, destination = 0;
    const auto returned = cudaMemcpy(&destination, &source, sizeof(source), static_cast<cudaMemcpyKind>(99));
    require(returned == cudaErrorInvalidMemcpyDirection && cudaGetLastError() == returned,
        "historical score consume only actual safe kind99 status");
    bool fresh = transfer, retire = false;
    unsigned attempts = 0;
    std::optional<kvmem::QueryCaptureBorrow> older;
    if (!transfer) older.emplace(f.owner->sparse_capture()->borrow_query([&] {
        if (!retire) throw kvmem::CudaTransferError(cudaErrorIllegalAddress, operation);
    }));
    auto borrower = f.owner->sparse_capture()->borrow_query([completion, returned, &fresh, &retire, &attempts] {
        completion(); ++attempts;
        if (retire) return;
        if (!fresh) throw std::runtime_error("historical score pending fresh completion");
        kvmem::check_transfer_cuda(returned, "fresh live score borrower B", true);
    });
    if (transfer) TieredContextTestAccess::poison_transfer(*f.owner,
        std::make_exception_ptr(kvmem::CudaTransferError(cudaErrorIllegalAddress, operation)));
    else {
        // Both callbacks are attempted. A is established in Sparse before the
        // next live Main release; a generic pending B cannot replace typed A.
        bool stored = false;
        try { f.owner->sparse_capture()->drain(); }
        catch (const kvmem::CudaTransferError& error) {
            stored = error.status() == cudaErrorIllegalAddress && !error.drain_failed() &&
                std::string_view(error.operation()) == operation;
        }
        if (!stored) { retire = true; f.owner->sparse_capture()->drain(); f.owner.reset(); }
        require(stored, "historical score Sparse did not establish original A");
        fresh = true;
    }
    bool original = false;
    try { TieredContextTestAccess::finish_score_query_borrow(*f.owner, borrower); }
    catch (const kvmem::CudaTransferError& error) {
        std::cout << "historical score release observer transfer=" << transfer << " status=" << int(error.status())
            << " operation=" << error.operation() << " drain_failed=" << error.drain_failed() << '\n';
        original = error.status() == cudaErrorIllegalAddress && !error.drain_failed() &&
            std::string_view(error.operation()) == operation;
    }
    const bool accepted = f.owner->frontier() == prior.frontier &&
        f.owner->current_view().generation == prior.generation;
    retire = true;
    f.owner->sparse_capture()->drain(); // real successful retry releases both source borrows
    const bool drained = !borrower.valid() && (!older || !older->valid());
    bool sticky = false;
    try { f.owner->reset(f.device.stream); }
    catch (const kvmem::CudaTransferError& error) {
        sticky = error.status() == cudaErrorIllegalAddress && !error.drain_failed() &&
            std::string_view(error.operation()) == operation;
    }
    f.owner.reset(); // no callback keeps references to fixture locals through their destruction
    std::cout << "historical score release result transfer=" << transfer << " original=" << original
        << " sticky=" << sticky << " attempts=" << attempts << " returned_status=" << int(returned)
        << " simulated_fresh_failed_drain=1\n";
    return original && sticky && accepted && drained && attempts == (transfer ? 1u : 2u) + 1u;
}
void historical_main_cause_before_borrower() {
    Fixture f(HostKVArchiveMode::Pageable, false);
    f.prefill(66);
    f.owner->drain(); // real healthy worker/compute drains before status feeding
    const auto prior = f.owner->current_view();
    int source = 1, destination = 0;
    const auto returned = cudaMemcpy(&destination, &source, sizeof(source), static_cast<cudaMemcpyKind>(99));
    require(returned == cudaErrorInvalidMemcpyDirection, "historical Main safe returned status");
    require(cudaGetLastError() == returned, "consume only the expected safe kind99 test rejection");
    try { f.owner->check_compute_drain(returned, "historical Main original A"); }
    catch (const kvmem::CudaTransferError&) {}
    unsigned drains = 0;
    auto borrower = f.owner->sparse_capture()->borrow_query([&] {
        if (++drains == 1) throw kvmem::CudaTransferError(cudaErrorIllegalAddress, "later borrower B");
    });
    bool original = false;
    try { f.owner->reset(f.device.stream); }
    catch (const kvmem::CudaTransferError& error) {
        std::cout << "historical Main observer status=" << int(error.status()) << " operation="
            << error.operation() << " drain_failed=" << error.drain_failed() << '\n';
        original = error.status() == returned && error.drain_failed() &&
            std::string_view(error.operation()) == "historical Main original A";
    }
    require(original && drains == 1, "later Q fatal masked already-latched Main original failed-drain cause");
    f.owner->sparse_capture()->drain();
    require(drains == 2 && !borrower.valid() && f.owner->frontier() == prior.frontier &&
        f.owner->current_view().generation == prior.generation, "historical cause lost accepted metadata or borrower drain");
    bool sticky = false;
    try { f.owner->reset(f.device.stream); }
    catch (const kvmem::CudaTransferError& error) {
        sticky = error.status() == returned && error.drain_failed() &&
            std::string_view(error.operation()) == "historical Main original A";
    }
    require(sticky, "later successful drains cleared historical Main original A");
    std::cout << "historical Main original cause PASS returned_status=" << int(returned)
        << " simulated_failed_drain_placement=1\n";
}
void historical_cross_owner_before_compute(bool transfer) {
    Fixture f(HostKVArchiveMode::Pageable, false);
    f.prefill(66);
    f.owner->drain();
    const auto prior = f.owner->current_view();
    const char* operation = transfer ? "historical transfer A" : "historical capture A";
    unsigned drains = 0;
    std::optional<kvmem::QueryCaptureBorrow> borrower;
    if (transfer) {
        TieredContextTestAccess::poison_transfer(*f.owner,
            std::make_exception_ptr(kvmem::CudaTransferError(cudaErrorIllegalAddress, operation)));
    } else {
        borrower.emplace(f.owner->sparse_capture()->borrow_query([&] {
            if (++drains == 1) throw kvmem::CudaTransferError(cudaErrorIllegalAddress, operation);
        }));
        try { f.owner->sparse_capture()->drain(); }
        catch (const kvmem::CudaTransferError&) {}
    }
    int source = 1, destination = 0;
    const auto returned = cudaMemcpy(&destination, &source, sizeof(source), static_cast<cudaMemcpyKind>(99));
    require(returned == cudaErrorInvalidMemcpyDirection, "cross-owner actual returned status");
    require(cudaGetLastError() == returned, "consume only the expected safe kind99 test rejection");
    bool original = false;
    try { TieredContextTestAccess::drain_borrowers(*f.owner, returned, "later compute drain B"); }
    catch (const kvmem::CudaTransferError& error) {
        original = error.status() == cudaErrorIllegalAddress && !error.drain_failed() &&
            std::string_view(error.operation()) == operation;
    }
    require(original && (transfer || (drains == 2 && !borrower->valid())),
        "fresh checked compute failed-drain masked historical other-owner typed cause or skipped Q drain");
    bool sticky = false;
    try { f.owner->reset(f.device.stream); }
    catch (const kvmem::CudaTransferError& error) {
        sticky = error.status() == cudaErrorIllegalAddress && !error.drain_failed() &&
            std::string_view(error.operation()) == operation;
    }
    require(sticky && f.owner->frontier() == prior.frontier && f.owner->current_view().generation == prior.generation,
        "cross-owner original cause or accepted metadata changed after later healthy CUDA drains");
    std::cout << "historical cross-owner cause PASS transfer=" << transfer << " returned_status=" << int(returned)
        << " simulated_compute_drain_placement=1\n";
}
void fresh_quiesce_before_compute() {
    Fixture f(HostKVArchiveMode::Pageable, false);
    f.prefill(66); f.owner->drain();
    const auto prior = f.owner->current_view();
    unsigned callbacks = 0;
    auto first = f.owner->sparse_capture()->borrow_query([&] { ++callbacks; });
    auto second = f.owner->sparse_capture()->borrow_query([&] { ++callbacks; });
    int source = 1, destination = 0;
    const auto returned = cudaMemcpy(&destination, &source, sizeof(source), static_cast<cudaMemcpyKind>(99));
    require(returned == cudaErrorInvalidMemcpyDirection && cudaGetLastError() == returned,
        "fresh quiesce consume only expected real kind99 status");
    bool original = false;
    try {
        TieredContextTestAccess::drain_after_quiesce(*f.owner, returned, "fresh quiesce original A",
            returned, "later checked compute B");
    } catch (const kvmem::CudaTransferError& error) {
        std::cout << "fresh quiesce observer status=" << int(error.status()) << " operation="
            << error.operation() << " drain_failed=" << error.drain_failed() << '\n';
        original = error.status() == returned && error.drain_failed() &&
            std::string_view(error.operation()) == "fresh quiesce original A";
    }
    require(original && callbacks == 2 && !first.valid() && !second.valid(),
        "fresh compute cause masked newly recorded quiesce original or skipped callbacks");
    bool sticky = false;
    try { f.owner->reset(f.device.stream); }
    catch (const kvmem::CudaTransferError& error) {
        sticky = error.status() == returned && error.drain_failed() &&
            std::string_view(error.operation()) == "fresh quiesce original A";
    }
    require(sticky && f.owner->frontier() == prior.frontier && f.owner->current_view().generation == prior.generation,
        "new quiesce original cause/accepted metadata changed after later successful drains");
    std::cout << "fresh quiesce original cause PASS returned_status=" << int(returned)
        << " simulated_quiesce_and_compute_drain_placement=1\n";
}
std::vector<std::vector<std::byte>> physical_bytes(Fixture& f) {
    CUDA_CHECK(cudaStreamSynchronize(f.device.stream));
    std::vector<std::vector<std::byte>> result;
    for (const auto& layer : f.plan.archive.layers) for (const auto& plane : layer) {
        const auto& tensor = f.main->pool().plane(plane.pool_plane);
        result.emplace_back(tensor.bytes());
        CUDA_CHECK(cudaMemcpy(result.back().data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
    }
    return result;
}
void require_blocked_main(Fixture& f) {
    Tensor query(f.q.p, DType::BF16, {256,24,1}), key(f.k.p, DType::BF16, {256,4,1}),
        value(f.v.p, DType::BF16, {256,4,1}), output(f.out.p, DType::BF16, {256,24,1}),
        positions(f.positions.p, DType::I32, {1});
    rejects([&] { f.owner->capture_pre_rope(0, query, key, f.device.stream); },
            "failed begin/worker blocks pre-RoPE capture before GPU mutation");
    rejects([&] { f.owner->attention(0, query, key, value, positions, .0625f, output, f.device.stream); },
            "failed begin/worker blocks append attention before GPU mutation");
    rejects([&] { (void)f.owner->resident_layer(0); }, "failed begin/worker blocks physical view access");
}
void begin_prefix_failure() {
    Fixture f(HostKVArchiveMode::Pageable, true, 0, 4096, 16, true);
    f.prefill(3905);
    auto input = f.turn(); input.image_groups.clear();
    (void)f.owner->select_and_publish(input);
    const auto prior = f.owner->current_view();
    require(prior.blocktable[61] < 0, "R0 nonuniform selection must omit HostOnly partial frontier page");
    const auto bytes = physical_bytes(f);
    const auto loading = f.owner->loading_info();
    const auto index_generation = f.owner->sparse_capture()->index().generation();
    kvmem::HostKVTransferFaultInjection fault; fault.h2d_copy_after = 1;
    f.owner->set_transfer_test_fault(fault);
    f.owner->prepare_main_transaction(kvmem::MeanKTransactionKind::SpeculativeMain, 1);
    bool rejected = false;
    try { f.owner->begin_block(prior.frontier, 1, f.device.stream, TieredContext::ExecutionPhase::Decode); }
    catch (const std::exception&) { rejected = true; }
    require(rejected && f.owner->poisoned(), "failed partial-page prefix restore must fail-closed after begin mutation");
    const auto damaged = f.owner->current_view();
    require(f.owner->frontier() == prior.frontier && damaged.frontier == prior.frontier + 1 &&
            f.owner->sparse_capture()->index().frontier() == prior.frontier &&
            f.owner->sparse_capture()->index().generation() == index_generation,
            "failed prefix exposes only poisoned incomplete append diagnostics; accepted index remains unchanged");
    require_blocked_main(f);
    require(physical_bytes(f) == bytes, "rejected prefix and blocked operations preserve every physical Main plane byte");
    const auto after = f.owner->loading_info();
    require(after.pinned_bytes == loading.pinned_bytes && after.owned_device_bytes == loading.owned_device_bytes &&
            after.device_backing_bytes == loading.device_backing_bytes, "failed begin adds no fixed GPU/pinned resources");
    f.owner->reset(f.device.stream);
    require(!f.owner->poisoned() && f.owner->frontier() == 0 && f.owner->sparse_capture()->index().frontier() == 0,
            "failed begin needs explicit whole Main cold reset before exact rebuild");
}
void async_prefill_failure() {
    std::atomic<bool> paused{false}, resume{false};
    Fixture f(HostKVArchiveMode::Pageable, false);
    struct ResumeOnExit { std::atomic<bool>& flag; ~ResumeOnExit() { flag.store(true); } } guard{resume};
    f.prefill(1088);
    kvmem::HostKVTransferFaultInjection fault;
    fault.h2d_copy_after = fault.pause_before_h2d_copy = 1;
    fault.h2d_paused = &paused; fault.resume_h2d = &resume;
    f.owner->set_transfer_test_fault(fault);
    f.owner->begin_block(1088, 1, f.device.stream, TieredContext::ExecutionPhase::Prefill);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!paused.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    require(paused.load(), "real asynchronous prefill H2D worker reached controlled copy boundary");
    const auto begun = f.owner->current_view();
    const auto index_frontier = f.owner->sparse_capture()->index().frontier();
    const auto index_generation = f.owner->sparse_capture()->index().generation();
    const auto bytes = physical_bytes(f);
    resume.store(true);
    while (!f.owner->poisoned() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    require(f.owner->poisoned(), "asynchronous prefill worker failure must poison before any further capture/append");
    require_blocked_main(f);
    require(f.owner->current_view().blocktable == begun.blocktable &&
            f.owner->current_view().generation == begun.generation && f.owner->frontier() == begun.frontier &&
            f.owner->sparse_capture()->index().frontier() == index_frontier &&
            f.owner->sparse_capture()->index().generation() == index_generation && physical_bytes(f) == bytes,
            "asynchronous rejection preserves snapshot mappings, accepted index and all physical bytes");
    f.owner->reset(f.device.stream);
    require(!f.owner->poisoned() && f.owner->frontier() == 0, "async failure needs explicit drained whole Main reset");
}
} // namespace
int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--only-score-teardown") {
            permanent_score_completion_teardown(); return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--only-score-history") {
            const bool transfer = historical_score_release_before_borrower(true);
            const bool sparse = historical_score_release_before_borrower(false);
            require(transfer && sparse, "fresh live score failed-drain masked historical transfer/Sparse original cause");
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--only-fresh-quiesce") {
            fresh_quiesce_before_compute(); return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--only-historical-cause") {
            historical_main_cause_before_borrower();
            historical_cross_owner_before_compute(false);
            historical_cross_owner_before_compute(true);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--only-begin-prefix") {
            begin_prefix_failure(); return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--only-async-prefill") {
            async_prefill_failure(); return 0;
        }
        for (auto mode : {HostKVArchiveMode::Pinned, HostKVArchiveMode::Pageable})
            for (bool mtp : {false,true}) exercise(mode, mtp);
        for (auto mode : {HostKVArchiveMode::Pinned, HostKVArchiveMode::Pageable}) poison_recovery(mode);
        for (auto mode : {HostKVArchiveMode::Pinned, HostKVArchiveMode::Pageable}) generic_source_failure(mode);
        quantized_owner(HostKVArchiveMode::Pinned, true);
        quantized_owner(HostKVArchiveMode::Pageable, false);
        quantized_owner(HostKVArchiveMode::Pinned, false, true);
        quantized_owner(HostKVArchiveMode::Pageable, true, true);
        nonuniform_owner_score(); fragmented_next_exact(); reserve_failure(); restore_before_query();
        sticky_capture_failure(cudaErrorInvalidValue, true);
        sticky_capture_failure(cudaErrorIllegalAddress, false);
        sticky_capture_failure(cudaErrorUnknown, false);
        historical_main_cause_before_borrower();
        historical_cross_owner_before_compute(false);
        historical_cross_owner_before_compute(true);
        fresh_quiesce_before_compute();
        permanent_score_completion_teardown();
        const bool score_transfer = historical_score_release_before_borrower(true);
        const bool score_sparse = historical_score_release_before_borrower(false);
        require(score_transfer && score_sparse, "fresh live score failed-drain masked historical transfer/Sparse original cause");
        begin_prefix_failure(); async_prefill_failure();
        std::cout << "PASS KVMem runtime owner\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

#include "targets/qwen3_6/impl/runtime/sparse_capture_owner.h"
#include "core/device.h"
#include "targets/qwen3_6/impl/runtime/tiered_context.h"
#include <ninfer/ops/rope.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace ninfer;
using namespace ninfer::kvmem;
using namespace ninfer::targets::qwen3_6;
using namespace ninfer::targets::qwen3_6::detail;
namespace {
void require(bool v, const char* m) {
    if (!v)
        throw std::runtime_error(m);
}
template <class F> void rejects(F f) {
    bool caught = false;
    try {
        f();
    } catch (const std::exception&) {
        caught = true;
    }
    require(caught, "invalid capture action accepted");
}
QueryProvenance provenance(std::uint64_t bundle = 17, std::uint64_t lineage = 19) {
    QueryProvenance p;
    p.bundle_identity = bundle;
    p.lineage = lineage;
    p.capture_epoch = 3;
    p.ordinals = {60, 61, 65, 67};
    p.covered_frontier = 68;
    p.source_prefix[0] = 42;
    return p;
}
void cpu_contracts() {
    auto resource = plan_query_capture_resources();
    require(resource.slot_bytes == 3 * 1024 * 1024 && resource.pageable_bytes == 9 * 1024 * 1024 &&
                resource.device_bytes == resource.slot_bytes &&
                resource.pinned_bytes == resource.slot_bytes,
            "three-slot resource ledger");
    QueryCapturePool pool(resource);
    const auto publish = [&](QueryProvenance p) {
        pool.begin(std::move(p));
        std::vector<std::uint16_t> rows(256 * 24 * 4, 0x3f80);
        for (std::uint32_t layer = 0; layer < 16; ++layer)
            pool.complete_layer(layer, rows);
        return pool.publish();
    };
    pool.begin(provenance());
    std::vector<std::uint16_t> rows(256 * 24 * 4, 0x3f80);
    for (std::uint32_t layer = 0; layer < 15; ++layer)
        pool.complete_layer(layer, rows);
    rejects([&] { (void)pool.publish(); });
    pool.complete_layer(15, rows);
    auto a = pool.publish();
    auto shared = a;
    require(pool.live_slots() == 1, "shared handle allocated another slot");
    auto b = publish(provenance());
    auto c = publish(provenance());
    require(pool.live_slots() == 3 && pool.high_watermark() == 3, "slot high-watermark");
    rejects([&] { pool.begin(provenance()); });
    bool drained = false;
    auto borrow = pool.borrow(a, [&] { drained = true; });
    a.reset();
    shared.reset();
    rejects([&] { pool.begin(provenance()); });
    pool.drain_borrows();
    require(drained && !borrow.valid() && pool.live_slots() == 2, "borrow not drained/released");
    auto d = publish(provenance());
    require(b->layer(0).front() == 0x3f80, "published immutable slot overwritten");
    require(query_capture_matches(d, provenance(), 100),
            "capture epoch incorrectly bound to index epoch");
    auto changed = provenance();
    changed.source_prefix[0]++;
    require(!query_capture_matches(d, changed, 100), "rewritten prefix accepted");
    require(!query_capture_matches(d, provenance(), 67), "restore before last query accepted");
    changed = provenance();
    changed.bundle_identity++;
    require(!query_capture_matches(d, changed, 100), "foreign bundle accepted");
    changed = provenance();
    changed.lineage++;
    require(!query_capture_matches(d, changed, 100), "foreign lineage accepted");
    changed = provenance();
    changed.ordinals.back()++;
    require(!query_capture_matches(d, changed, 100), "changed source query ordinals accepted");
    pool.reset();
    require(!d->valid(), "reset left old capture valid");
}
void continuation_contracts() {
    const auto p = provenance();
    const std::vector<ExactCaptureContinuation> states{
        {50, 17, 19, true}, {60, 17, 19, true}, {61, 17, 19, true}, {0, 99, 19, true}};
    auto plan = plan_query_recapture(p, {}, 100, states);
    require(plan.kind == QueryRecaptureKind::ExactContinuation && plan.frontier == 60,
            "missing capture did not choose legal exact boundary before first Q");
    plan = plan_query_recapture(p, {}, 100,
                                std::span<const ExactCaptureContinuation>(states).subspan(2));
    require(plan.kind == QueryRecaptureKind::ColdStart,
            "missing legal continuation did not cold start");
    QueryCapturePool pool(plan_query_capture_resources());
    pool.begin(p);
    std::vector<std::uint16_t> q(256 * 24 * 4, 0x3f80);
    for (int l = 0; l < 16; ++l)
        pool.complete_layer(l, q);
    auto saved = pool.publish();
    require(plan_query_recapture(p, saved, 100, states).kind == QueryRecaptureKind::Reuse,
            "matching original user capture not reused for tool input");
    require(plan_query_recapture(p, saved, 60, states).kind ==
                QueryRecaptureKind::ExactContinuation,
            "query frontier truncation reused stale Q");
}
void provenance_identity_contracts() {
    PreparedPromptData prompt;
    prompt.token_ids.resize(90);
    prompt.token_types.resize(90);
    prompt.positions.resize(270);
    for (int axis = 0; axis < 3; ++axis)
        for (int i = 0; i < 90; ++i) {
            prompt.token_ids[i] = i;
            prompt.positions[axis * 90 + i] = i;
        }
    prompt.input_spans.available = true;
    prompt.input_spans.current = {{0, 90}};
    prompt.input_spans.query = {{10, 20}, {40, 10}};
    prompt.input_spans.source_query = prompt.input_spans.query;
    auto source = query_provenance(prompt, 17, 19, 1);
    require(source.ordinals.size() == 16 && source.ordinals.front() == 24 &&
                source.ordinals.back() == 49 && source.covered_frontier == 50,
            "lastN selected physical tail instead of text query spans");
    prompt.token_ids[80]++;
    auto append = query_provenance(prompt, 17, 19, 2);
    require(source.source_prefix == append.source_prefix,
            "appended tool suffix rewrote query source digest");
    prompt.positions[1]++;
    require(source.source_prefix != query_provenance(prompt, 17, 19, 2).source_prefix,
            "position rewrite was not bound to query source prefix");
    prompt.positions[1]--;
    prompt.token_types[2] = 1;
    require(source.source_prefix != query_provenance(prompt, 17, 19, 2).source_prefix,
            "modality rewrite was not bound to query source prefix");
    prompt.token_types[2] = 0;
    prompt.vision_items.push_back({});
    prompt.vision_items.back().token_spans = {{0, 2}, {5, 2}};
    auto image_source = query_provenance(prompt, 17, 19, 2);
    prompt.vision_items.back().content_digest[0]++;
    require(image_source.source_prefix != query_provenance(prompt, 17, 19, 3).source_prefix,
            "image content digest absent from prefix identity");
    image_source = query_provenance(prompt, 17, 19, 2);
    prompt.vision_items.back().timestamps = {0.01};
    require(image_source.source_prefix != query_provenance(prompt, 17, 19, 3).source_prefix,
            "video timestamp identity absent from query prefix");
    const auto multi_source = query_provenance(prompt, 17, 19, 3);
    prompt.input_spans.query.clear();
    require(query_provenance(prompt, 17, 19, 3, 16, true).ordinals == source.ordinals,
            "tool continuation lost original user source query");
    // A prior explicit event contained two users. A later tool-only prompt has
    // a latest-user fallback with fewer rows, but the saved capture stays valid.
    prompt.input_spans.all_user_text = {{10, 20}, {40, 10}};
    prompt.input_spans.source_query = {{40, 10}};
    prompt.input_spans.current = {{70, 20}};
    const auto fallback = query_provenance(prompt, 17, 19, 7, 16, true);
    require(fallback.ordinals.size() == 10, "tool fallback included earlier historical user");
    const auto known = query_provenance_for_ordinals(prompt, 17, 19, 7, multi_source.ordinals);
    require(known.ordinals == multi_source.ordinals &&
                known.source_prefix == multi_source.source_prefix,
            "known multi-user capture was replaced by latest-user fallback");
    QueryCapturePool capture_pool(plan_query_capture_resources());
    capture_pool.begin(known);
    std::vector<std::uint16_t> q(256 * 24 * known.ordinals.size(), 0x3f80);
    for (int l = 0; l < 16; ++l)
        capture_pool.complete_layer(l, q);
    auto capture = capture_pool.publish();
    prompt.token_ids[80]++;
    require(query_capture_matches(
                capture, query_provenance_for_ordinals(prompt, 17, 19, 99, known.ordinals), 90),
            "multi-user capture failed revalidation in tool prompt");
    prompt.token_ids[20]++;
    require(!query_capture_matches(
                capture, query_provenance_for_ordinals(prompt, 17, 19, 99, known.ordinals), 90),
            "known ordinals hid rewritten source prefix");
    prompt.token_ids[20]--;
    const std::vector<std::uint32_t> header{9}, assistant{70}, duplicate{40, 40}, reversed{41, 40};
    for (const auto& invalid : {header, assistant, duplicate, reversed})
        rejects([&] { (void)query_provenance_for_ordinals(prompt, 17, 19, 7, invalid); });
    prompt.token_types[40] = 1;
    rejects([&] { (void)query_provenance_for_ordinals(prompt, 17, 19, 7, known.ordinals); });
    prompt.token_types[40] = 0;
    auto too_many = known.ordinals;
    too_many.insert(too_many.begin(), 23);
    rejects([&] { (void)query_provenance_for_ordinals(prompt, 17, 19, 7, too_many); });
    rejects([&] { (void)query_provenance_for_ordinals(prompt, 17, 19, 7, {}); });
    rejects([&] { (void)query_provenance(prompt, 17, 19, 3); });
    prompt.input_spans.available = false;
    rejects([&] { (void)query_provenance(prompt, 17, 19, 3, 16, true); });
    rejects([&] { (void)query_provenance_for_ordinals(prompt, 17, 19, 7, known.ordinals); });
}
constexpr std::size_t Q = 256 * 24, K = 256 * 4;
std::vector<std::uint16_t> values(std::uint32_t first, std::uint32_t n, std::size_t rows,
                                  int layer = 0) {
    std::vector<std::uint16_t> result(rows * n);
    for (std::uint32_t t = 0; t < n; ++t)
        for (std::size_t r = 0; r < rows; ++r)
            result[t * rows + r] = std::uint16_t(
                std::bit_cast<std::uint32_t>(float(1 + (first + t) % 7) + float(r % 3) / 4.f +
                                             float(layer) / 8.f) >>
                16);
    return result;
}
void gpu_contracts(bool pinned) {
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
        auto resources = plan_sparse_capture_resources(256, 48);
        SparseCaptureOwner owner(resources, pinned, stream);
        require(owner.index().pinned() == pinned, "actual archive/index pin policy mismatch");
        auto p = provenance();
        p.ordinals = {40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55};
        p.covered_frontier = 56;
        owner.begin_query(p);
        DeviceBuffer queries(Q * 48 * 2), keys(K * 48 * 2);
        const auto exact = [&](std::uint32_t first, std::uint32_t count) {
            auto q = values(first, count, Q), k = values(first, count, K);
            queries.copy_from_host(q.data(), q.size() * 2);
            keys.copy_from_host(k.data(), k.size() * 2);
            CUDA_CHECK(cudaDeviceSynchronize());
            owner.prepare_transaction(MeanKTransactionKind::ExactPrefill, first, count);
            for (std::uint32_t layer = 0; layer < 16; ++layer) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                q = values(first, count, Q, layer);
                k = values(first, count, K, layer);
                queries.copy_from_host(q.data(), q.size() * 2);
                keys.copy_from_host(k.data(), k.size() * 2);
                CUDA_CHECK(cudaDeviceSynchronize());
                owner.capture_pre_rope(layer, Tensor(queries.p, DType::BF16, {256, 24, int(count)}),
                                       Tensor(keys.p, DType::BF16, {256, 4, int(count)}), stream);
            }
            owner.finish_exact_chunk();
        };
        exact(0, 48);
        require(!owner.query(), "incomplete cross-chunk Q published");
        auto checkpoint = owner.capture_snapshot();
        exact(48, 32);
        auto captured = owner.query();
        require(captured && captured->provenance().ordinals == p.ordinals,
                "lastN query ordinal capture missing");
        for (int layer = 0; layer < 16; ++layer) {
            const auto q = captured->layer(layer);
            const auto expected = values(40, 16, Q, layer);
            require(std::equal(q.begin(), q.end(), expected.begin()),
                    "Q rows from physical chunk tail");
        }
        auto index_generation = owner.index().generation();
        auto binding = owner.bind_query(p, 80);
        require(owner.binding_valid(binding, p, 80), "valid current index/capture pair rejected");
        auto retained = owner.capture_snapshot();
        auto forged_snapshot = retained;
        --forged_snapshot.frontier;
        rejects([&] { owner.restore_snapshot(forged_snapshot); });
        require(owner.index().frontier() == 80 && owner.index().published(),
                "bad snapshot mutated Main index before validation");
        auto device_q = owner.upload_query();
        bool drained = false;
        auto borrower = owner.borrow_query([&] {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            drained = true;
        });
        exact(80, 32);
        require(drained && !borrower.valid(), "next capture did not drain score/upload borrow");
        require(!owner.binding_valid(binding, p, 112), "old index score ticket accepted");
        owner.restore_snapshot(retained);
        require(owner.index().frontier() == 80 && owner.index().base_frontier() == 80 &&
                    owner.index().generation() > index_generation && owner.query() == captured,
                "restore failed exact index frontier/Q ownership");
        require(!owner.binding_valid(binding, p, 80), "old binding survived restore generation");
        auto rebound = owner.bind_query(p, 80);
        require(owner.binding_valid(rebound, p, 80),
                "old capture epoch cannot bind restored index");
        for (int layer = 1; layer < 16; ++layer)
            for (std::size_t r = 0; r < K; ++r) {
                double expected = 0;
                for (std::uint32_t t = 64; t < 80; ++t)
                    expected += double(float(1 + t % 7) + float(r % 3) / 4.f + float(layer) / 8.f);
                require(double(owner.index().prefix_sum(layer)[r]) == expected,
                        "layer identity/stride lost exact Mean-K snapshot sum");
            }
        const auto sum = owner.index().prefix_sum(0);
        for (std::size_t r = 0; r < K; ++r) {
            double expected = 0;
            for (std::uint32_t t = 64; t < 80; ++t)
                expected += double(float(1 + t % 7) + float(r % 3) / 4.f);
            require(double(sum[r]) == expected, "snapshot sum recovered from rounded mean");
        }
        // Same-page append/trim must replay from independent exact restored base.
        exact(80, 3);
        owner.trim(81);
        require(owner.index().frontier() == 81, "bounded tail trim frontier");
        for (std::size_t r = 0; r < K; ++r) {
            double expected = 0;
            for (std::uint32_t t = 64; t < 81; ++t)
                expected += double(float(1 + t % 7) + float(r % 3) / 4.f);
            require(double(owner.index().prefix_sum(0)[r]) == expected,
                    "trim used mutated active base sum");
        }
        rejects([&] { owner.trim(79); });
        owner.restore_snapshot(checkpoint);
        require(owner.index().frontier() == 48 && !owner.query(),
                "restore before query retained stale capture");
        require(!owner.binding_valid(rebound, p, 48), "restored-before-Q binding valid");
        // Application accepted prefix only. Rejected speculative rows must never enter the index.
        auto q = values(48, 16, Q), k = values(48, 16, K);
        queries.copy_from_host(q.data(), q.size() * 2);
        keys.copy_from_host(k.data(), k.size() * 2);
        CUDA_CHECK(cudaDeviceSynchronize());
        owner.prepare_transaction(MeanKTransactionKind::SpeculativeMain, 48, 16);
        for (int l = 0; l < 16; ++l)
            owner.capture_pre_rope(l, Tensor(queries.p, DType::BF16, {256, 24, 16}),
                                   Tensor(keys.p, DType::BF16, {256, 4, 16}), stream);
        owner.flush_accepted_frontier(2);
        require(owner.index().frontier() == 50,
                "provisional rejection polluted committed frontier");
        for (std::size_t r = 0; r < K; ++r) {
            double expected = 0;
            for (std::uint32_t t = 0; t < 50; ++t)
                expected += double(float(1 + t % 7) + float(r % 3) / 4.f);
            require(double(owner.index().prefix_sum(0)[r]) == expected,
                    "rejected speculative K committed");
        }
        owner.prepare_transaction(MeanKTransactionKind::OrdinaryMain, 50, 1);
        for (int l = 0; l < 16; ++l)
            owner.capture_pre_rope(l, Tensor(queries.p, DType::BF16, {256, 24, 16}),
                                   Tensor(keys.p, DType::BF16, {256, 4, 16}), stream);
        owner.flush_accepted_frontier(1);
        require(owner.index().frontier() == 51, "ordinary Main anchor not accepted exactly once");
        owner.prepare_transaction(MeanKTransactionKind::SpeculativeMain, 51, 16);
        for (int l = 0; l < 16; ++l)
            owner.capture_pre_rope(l, Tensor(queries.p, DType::BF16, {256, 24, 16}),
                                   Tensor(keys.p, DType::BF16, {256, 4, 16}), stream);
        owner.flush_accepted_frontier(0);
        require(owner.index().frontier() == 51 && owner.index().published(),
                "zero application acceptance mutated committed index");
        owner.reset();
        rejects([&] { owner.restore_snapshot(checkpoint); });
        require(owner.index().frontier() == 0, "reset index frontier");
    } catch (...) {
        cudaStreamDestroy(stream);
        throw;
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}
void integrated_main_contracts() {
    DeviceContext device;
    LayoutBuilder builder;
    DecoderStateSpec spec;
    spec.capacity = 192;
    spec.full_attention_layers = 16;
    spec.kv_heads = 4;
    spec.attention_head_dim = 256;
    spec.kv_dtype = DType::BF16;
    spec.text_physical_page_groups = 3;
    spec.allow_tiered_text_pages = true;
    spec.linear_attention = {1, 1, 1, 1, 1, 1};
    const auto layout = plan_decoder_state(builder, spec).text_kv;
    DeviceBuffer pool_memory(builder.finish(256));
    PagedKVCache cache({pool_memory.p, pool_memory.bytes}, layout);
    auto lease = cache.pool().reserve(3);
    lease.materialize_pages(3, device.stream);
    lease.bind_row(0, device.stream);
    TieredKVOptions options;
    options.sink_tokens = 0;
    options.view_tokens = 192;
    options.host_archive = HostKVArchiveMode::Pageable;
    const auto plan = plan_tiered_runtime(layout.pool, 192, 3, 64, options, false, false);
    DeviceBuffer backing(plan.bytes);
    TieredContext main(plan, cache, {backing.p, backing.bytes}, device.stream);
    main.bind_pages(lease.page_ids());
    main.attach_sparse_capture(std::make_unique<SparseCaptureOwner>(
        plan_sparse_capture_resources(192, 64), false, device.stream));
    auto* sparse = main.sparse_capture();
    auto source = provenance(main.bundle_identity(), 71);
    source.ordinals = {5, 6, 70};
    source.covered_frontier = 71;
    sparse->begin_query(source);
    DeviceBuffer q_memory(Q * 64 * 2), k_memory(K * 64 * 2), v_memory(K * 64 * 2),
        o_memory(Q * 64 * 2), pos_memory(64 * 4);
    const auto run = [&](std::uint32_t first, std::uint32_t count) {
        auto q = values(first, count, Q), k = values(first, count, K), v = values(first, count, K);
        std::vector<std::int32_t> pos(count);
        for (std::uint32_t t = 0; t < count; ++t)
            pos[t] = first + t;
        pos_memory.copy_from_host(pos.data(), pos.size() * 4);
        sparse->prepare_transaction(MeanKTransactionKind::ExactPrefill, first, count);
        main.begin_block(first, count, device.stream);
        for (int layer = 0; layer < 16; ++layer) {
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            q = values(first, count, Q, layer);
            k = values(first, count, K, layer);
            q_memory.copy_from_host(q.data(), q.size() * 2);
            k_memory.copy_from_host(k.data(), k.size() * 2);
            v_memory.copy_from_host(v.data(), v.size() * 2);
            CUDA_CHECK(cudaDeviceSynchronize());
            Tensor query(q_memory.p, DType::BF16, {256, 24, int(count)}),
                key(k_memory.p, DType::BF16, {256, 4, int(count)}),
                value(v_memory.p, DType::BF16, {256, 4, int(count)}),
                output(o_memory.p, DType::BF16, {256, 24, int(count)}),
                positions(pos_memory.p, DType::I32, {int(count)});
            main.capture_pre_rope(layer, query, key, device.stream);
            ops::rope(positions, 64, 1000000.f, query, key, device.stream);
            main.attention(layer, query, key, value, positions, 0.0625f, output, device.stream);
        }
        sparse->finish_exact_chunk();
    };
    run(0, 64);
    auto checkpoint = main.capture();
    require(checkpoint.sparse && checkpoint.sparse->frontier == 64 && !checkpoint.sparse->query,
            "real Main snapshot omitted derived index/Q state");
    run(64, 32);
    auto retained = main.capture();
    const auto q = sparse->query();
    require(retained.sparse && retained.sparse->query == q,
            "Main snapshot copied scratch pointer instead of owning Q");
    for (int layer = 0; layer < 16; ++layer)
        for (std::size_t row = 0; row < source.ordinals.size(); ++row) {
            const auto expected = values(source.ordinals[row], 1, Q, layer);
            require(std::equal(expected.begin(), expected.end(), q->layer(layer).begin() + row * Q),
                    "real Main capture observed post-RoPE values");
        }
    run(96, 32);
    main.restore(retained, device.stream);
    require(main.frontier() == 96 && sparse->index().frontier() == 96 && sparse->query() == q,
            "real Main restore did not restore exact derived state");
    main.restore(retained, device.stream);
    main.restore(checkpoint, device.stream);
    require(main.frontier() == 64 && sparse->index().frontier() == 64 && !sparse->query(),
            "Main checkpoint before query left a stale capture");
    main.reset(device.stream);
    require(main.frontier() == 0 && sparse->index().frontier() == 0 && !q->valid(),
            "Main bundle reset did not invalidate derived state");
}

} // namespace
int main() {
    try {
        cpu_contracts();
        continuation_contracts();
        provenance_identity_contracts();
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || !count)
            return 77;
        gpu_contracts(false);
        gpu_contracts(true);
        integrated_main_contracts();
        std::cout << "query capture: spans/chunks/three slots/borrow/continuation/snapshot/stale "
                     "epoch passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

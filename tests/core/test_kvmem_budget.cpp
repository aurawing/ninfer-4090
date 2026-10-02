#include "targets/qwen3_6/impl/runtime/tiered_plan.h"
#include "targets/qwen3_6/impl/runtime/mtp_window_plan.h"
#include "runtime/engine/kv_capacity.h"

#include <iostream>
#include <stdexcept>

using namespace ninfer;
using namespace ninfer::targets::qwen3_6::detail;

namespace {
void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

void shadow_reference_policy() {
    PagedKVLayerView rk4;
    rk4.dtype    = DType::I8;
    rk4.packed_k = rk4.packed_v = rk4.rotate_k = rk4.rotate_v = rk4.e8_lattice = true;
    require(!tiered_shadow_fp64_required(32768, rk4), "32K keeps the dense shadow bound");
    require(!tiered_shadow_fp64_required(131072, rk4), "128K keeps the dense shadow bound");
    require(tiered_shadow_fp64_required(262144, rk4), "262K RK4 requires independent FP64");
    for (auto flag :
         {&PagedKVLayerView::packed_k, &PagedKVLayerView::packed_v, &PagedKVLayerView::rotate_k,
          &PagedKVLayerView::rotate_v, &PagedKVLayerView::e8_lattice}) {
        auto other  = rk4;
        other.*flag = false;
        require(!tiered_shadow_fp64_required(262144, other), "other KV formats retain dense bound");
    }
    auto root    = rk4;
    root.e8_root = true;
    require(!tiered_shadow_fp64_required(262144, root), "E8 root is outside RK4 revision");
    auto bf16  = rk4;
    bf16.dtype = DType::BF16;
    require(!tiered_shadow_fp64_required(262144, bf16), "BF16 is outside RK4 revision");
}

PagedKVPoolLayout pool(std::uint32_t context, std::uint32_t resident) {
    LayoutBuilder builder;
    PagedKVPoolSpec spec{.page_group_count      = resident,
                         .logical_page_capacity = (context + 63U) / 64U,
                         .table_rows            = 1};
    for (int i = 0; i < 32; ++i) { spec.planes.push_back({DType::BF16, 256, 4, 256}); }
    return plan_paged_kv_pool(builder, spec);
}

void budgets() {
    TieredKVOptions options;
    options.view_tokens = 8192;
    const auto limits   = tiered_page_limits(32768, 1024, options);
    require(limits.minimum == 21 && limits.maximum == 128, "sink/chunk/replacement page floor");
    const auto a = plan_tiered_runtime(pool(32768, 128), 32768, 128, 1024, options, false, false);
    const auto b = plan_tiered_runtime(pool(32768, 129), 32768, 129, 1024, options, false, false);
    require(a.partial.splits == 1, "prefill uses one split");
    require(a.partial.bytes == std::size_t(258) * 24 * 1024 * 2 * sizeof(float),
            "prefill scratch plus state bytes");
    require(a.staging.maximum_layer_stream_bytes == std::size_t(512 - 128) * 2 * 256 * 64 * 4 * 2,
            "staging covers all nonresident pages of one layer");
    require(a.staging.capacity_bytes == a.staging.maximum_layer_stream_bytes + (64ULL << 20),
            "one additional transfer tile");
    require(a.bytes - b.bytes == 2 * 256 * 64 * 4 * 2, "tiered backing has affine staging stride");
    require(a.scratch_o.region.bytes >= std::size_t(256) * 24 * 4 * 64 * sizeof(float),
            "scratch also supports four-query 64-split decode");
    require(a.state_o.region.offset >= a.scratch_l.region.offset + a.scratch_l.region.bytes,
            "partial state does not alias scratch");
    require(a.shadow_output.region.bytes == 0, "normal tiered has no shadow output");
    const auto short_block =
        plan_tiered_runtime(pool(32768, 128), 32768, 128, 64, options, false, false);
    require(short_block.partial.bytes ==
                short_block.scratch_o.region.bytes + short_block.scratch_m.region.bytes +
                    short_block.scratch_l.region.bytes + short_block.state_o.region.bytes +
                    short_block.state_m.region.bytes + short_block.state_l.region.bytes,
            "short prefill reports decode scratch in partial bytes, not metadata");
    const auto shadow =
        plan_tiered_runtime(pool(32768, 512), 32768, 128, 2048, options, true, true);
    require(shadow.view_pages == 128 && shadow.archive.logical_pages == 512,
            "shadow keeps full logical pool and conceptual view");
    require(shadow.partial.bytes == std::size_t(258) * 24 * 2048 * 2 * sizeof(float),
            "2048 prefill exact partial bytes");
    require(shadow.shadow_output.region.bytes == std::size_t(256) * 24 * 2048 * 2,
            "shadow validation output reservation");
    require(shadow.measure_transfer_waits, "transfer timing propagated");
    options.view_tokens = 64;
    bool rejected       = false;
    try {
        (void)tiered_page_limits(32768, 1024, options);
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "view ceiling must admit sink/chunk/replacement floor");
}

void quantized_budgets() {
    for (bool packed : {false, true}) {
        LayoutBuilder builder;
        PagedKVPoolSpec spec{
            .page_group_count = 2048, .logical_page_capacity = 4096, .table_rows = 1};
        for (int layer = 0; layer < 16; ++layer) {
            spec.planes.insert(spec.planes.end(),
                               {{packed ? DType::U8 : DType::I8, packed ? 128 : 256, 4, 256},
                                {packed ? DType::U8 : DType::I8, packed ? 128 : 256, 4, 256},
                                {DType::FP16, 4, 4, 256},
                                {DType::FP16, 4, 4, 256}});
        }
        const auto planned = plan_tiered_runtime(plan_paged_kv_pool(builder, spec), 262144, 2048,
                                                 2048, TieredKVOptions{}, false, false);
        require(planned.archive.layers.size() == 16 && planned.archive.layers.front().size() == 4,
                "quantized archive keeps four planes per layer");
        require(planned.staging.maximum_layer_stream_bytes ==
                    (packed ? 136ULL : 264ULL) * (1ULL << 20),
                "quantized staging uses actual packed plane bytes");
        require(planned.staging_plane_regions.size() == 4,
                "all quantized planes get staging regions");
        require(planned.partial.bytes == 101449728, "2048 partial scratch/state is 96.75 MiB");
    }
}

void fixed_staging_floor() {
    for (bool packed : {false, true}) {
        LayoutBuilder builder;
        PagedKVPoolSpec spec{
            .page_group_count = 37, .logical_page_capacity = 4096, .table_rows = 1};
        for (int layer = 0; layer < 16; ++layer) {
            spec.planes.insert(spec.planes.end(),
                               {{packed ? DType::U8 : DType::I8, packed ? 128 : 256, 4, 256},
                                {packed ? DType::U8 : DType::I8, packed ? 128 : 256, 4, 256},
                                {DType::FP16, 4, 4, 256},
                                {DType::FP16, 4, 4, 256}});
        }
        const auto archive =
            kvmem::plan_host_kv_archive(plan_paged_kv_pool(builder, spec), 4, 262144);
        TieredKVOptions options;
        const auto base = tiered_page_limits(262144, 2048, options);
        require(base.minimum == 37 && base.maximum == 2048,
                "original floor before staging override");
        const auto capacity = (packed ? 200ULL : 328ULL) << 20;
        const auto raised   = tiered_staging_page_limits(archive, base, capacity);
        require(raised.minimum == 2048 && raised.maximum == 2048,
                "feasible fixed staging raises resident floor to final view");
        require(kvmem::plan_host_kv_staging(archive, raised.minimum, capacity).capacity_bytes ==
                    capacity,
                "raised floor admits fixed staging at startup");
        const auto layer_page_bytes = (packed ? 69632ULL : 135168ULL);
        const auto extra_page =
            tiered_staging_page_limits(archive, base, capacity + layer_page_bytes);
        require(extra_page.minimum == 2047, "another streamed page lowers minimum by exactly one");
        const auto fractional_page =
            tiered_staging_page_limits(archive, base, capacity + layer_page_bytes - 1);
        require(fractional_page.minimum == 2048,
                "partial streamed page cannot lower resident floor");
        require(tiered_staging_page_limits(archive, base, 0).minimum == base.minimum,
                "automatic staging leaves page floor unchanged");
        bool rejected = false;
        try {
            (void)tiered_staging_page_limits(archive, base, capacity - 1);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "staging requiring more than view maximum is rejected");
        rejected = false;
        try {
            (void)tiered_staging_page_limits(archive, base, (64ULL << 20) - 1);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "staging must retain its entire additional transfer tile");
    }
}

struct CpuPlanner {
    runtime::SequenceCapacityCurve curve;
    const runtime::SequenceCapacityCurve& capacity_curve() const { return curve; }
};

// Real tiered pool/staging/partial allocations, with a fixed CPU stand-in for
// the target's remaining workspace. No CUDA allocator or free-memory query.
CpuPlanner candidate(const EngineOptions& options) {
    const auto limits = tiered_page_limits(options.max_context, options.prefill_chunk, options.kvmem);
    const auto reservation = [&](std::uint32_t pages) {
        const auto layout = pool(options.max_context, pages);
        auto bytes = layout.payload_bytes() + layout.metadata_bytes() + plan_tiered_runtime(layout, options.max_context, pages,
            std::min(options.prefill_chunk, options.max_context), options.kvmem, false, false).bytes;
        bytes += 8ULL * 1024 * options.prefill_chunk + (32ULL << 20);
        if (options.speculative.backend == SpeculativeBackend::Mtp)
            bytes += plan_mtp_window(options.max_context, options.prefill_chunk, options.kvmem).bytes;
        return bytes;
    };
    const auto minimum = reservation(limits.minimum);
    return {{64, limits.minimum, limits.maximum, minimum,
             limits.minimum < limits.maximum ? reservation(limits.minimum + 1) - minimum : 0}};
}

void automatic_prefill_complete_budget() {
    EngineOptions options;
    options.max_context = 32768;
    options.kv_mode = KvMode::TieredExact;
    options.kvmem.view_tokens = 8192;
    options.kv_capacity = KvCapacityPolicy::automatic(16ULL << 20);
    auto large = options;
    large.prefill_chunk = 2048;
    const auto large_minimum = candidate(large).curve.minimum_device_reservation_bytes;
    const auto headroom = options.kv_capacity.automatic_headroom_bytes;
    const auto ample = runtime::select_prefill_plan(options, false, large_minimum + headroom, candidate);
    require(ample.prefill_chunk == 2048 && ample.fallback_reason.empty(),
            "complete 2048 candidate must be selected exactly at its budget boundary");
    const auto constrained = runtime::select_prefill_plan(options, false, large_minimum + headroom - 1, candidate);
    require(constrained.prefill_chunk == 1024 && !constrained.fallback_reason.empty(),
            "one byte below complete 2048 budget must fall back with reason");
    bool rejected = false;
    try { (void)runtime::select_prefill_plan(options, false,
            candidate(options).curve.minimum_device_reservation_bytes + headroom - 1, candidate); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "both infeasible complete candidates must reject loading");
    options.prefill_chunk_explicit = true;
    const auto explicit_small = runtime::select_prefill_plan(options, false, large_minimum + headroom, candidate);
    require(explicit_small.prefill_chunk == 1024 && explicit_small.fallback_reason.empty(),
            "explicit 1024 must be honored even when 2048 fits");
    options.prefill_chunk = 2048;
    rejected = false;
    try { (void)runtime::select_prefill_plan(options, false, large_minimum + headroom - 1, candidate); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "explicit 2048 must reject instead of silently falling back");
    options.prefill_chunk_explicit = false;
    options.prefill_chunk = 128;
    require(runtime::select_prefill_plan(options, false, large_minimum + headroom, candidate).prefill_chunk == 128,
            "nondefault API chunk retains backward compatible explicit semantics");
    options.prefill_chunk = 1024;
    require(runtime::select_prefill_plan(options, true, large_minimum + headroom, candidate).prefill_chunk == 1024,
            "shadow default chunk must stay unchanged");
    options.kv_mode = KvMode::Dense;
    require(runtime::select_prefill_plan(options, false, large_minimum + headroom, candidate).prefill_chunk == 1024,
            "dense default chunk must stay unchanged");
    options.kv_mode = KvMode::TieredExact;
    options.kvmem.view_tokens = 2048;
    require(runtime::select_prefill_plan(options, false, large_minimum + headroom, candidate).prefill_chunk == 1024,
            "2048 view floor failure must try the feasible 1024 candidate");
    options.kvmem.view_tokens = 8192;
    options.kvmem.partial_budget_bytes = 64ULL << 20;
    require(runtime::select_prefill_plan(options, false, large_minimum + headroom, candidate).prefill_chunk == 1024,
            "2048 partial budget failure must try the feasible 1024 candidate");
    options.kvmem.partial_budget_bytes = 128ULL << 20;
    options.speculative.backend = SpeculativeBackend::Mtp;
    options.kvmem.mtp_window_tokens = 2048;
    require(runtime::select_prefill_plan(options, false, large_minimum + headroom, candidate).prefill_chunk == 1024,
            "2048 MTP window floor failure must try the feasible 1024 candidate");
    bool unrelated = false;
    try { (void)runtime::select_prefill_plan(options, false, large_minimum + headroom,
            [](const EngineOptions&) -> CpuPlanner { throw std::invalid_argument("unrelated configuration"); }); }
    catch (const std::invalid_argument& e) { unrelated = std::string(e.what()) == "unrelated configuration"; }
    require(unrelated, "fallback must not mask unrelated configuration errors");
}
} // namespace

int main() {
    try {
        budgets();
        quantized_budgets();
        fixed_staging_floor();
        shadow_reference_policy();
        automatic_prefill_complete_budget();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

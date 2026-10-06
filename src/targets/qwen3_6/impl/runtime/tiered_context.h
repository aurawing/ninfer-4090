#pragma once

#include "targets/qwen3_6/impl/runtime/tiered_plan.h"
#include "core/kvmem/kv_view_table.h"
#include "core/kvmem/selection_plan.h"
#include "targets/qwen3_6/impl/runtime/sparse_capture_owner.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>

#include <memory>
#include <span>
#include <type_traits>

namespace ninfer::targets::qwen3_6::detail {

struct TieredSnapshotLifetime {
    std::uint32_t frontier = 0;
    bool valid = true;
};
enum class SparseExecutionPhase { ExactPrefill, SparseDecode };
struct TieredSnapshot {
    std::uint64_t bundle_identity = 0;
    std::uint64_t archive_generation = 0;
    std::uint32_t archive_frontier = 0;
    std::uint32_t frontier = 0;
    std::uint64_t view_generation = 0;
    std::shared_ptr<TieredSnapshotLifetime> lifetime;
    std::shared_ptr<const SparseDerivedSnapshot> sparse;
    std::vector<std::uint32_t> selected_logical_ids, exact_hard_logical_ids;
    std::uint32_t recent_tokens{}, reserve_pages{}, guard_pages{};
    SparseExecutionPhase phase = SparseExecutionPhase::ExactPrefill;
};

struct SparseTurnInput {
    kvmem::QueryProvenance query;
    std::uint64_t bundle_identity{}; // caller bundle, distinct from internal owner lifetime
    std::vector<kvmem::TokenSpan> current_input_spans;
    std::vector<kvmem::ImageGroup> image_groups;
    double prefill_ms{}; // caller measured exact prefill wall duration
};
struct SparseTurnMetrics {
    bool all_pages_denominator{}; // requested variant, captured before GPU scoring
    kvmem::SelectionPlan selection;
    QueryIndexBinding binding; // actual immutable Q/index binding used by this score and selection
    std::vector<float> page_scores; // independent CPU copy of finite GPU result, useful for diagnostics/oracles
    kvmem::HostArchiveMode archive_mode{};
    std::uint32_t frontier{}, query_tokens{}, actual_view_tokens{}, selected_valid_tokens{}, denominator_pages{},
        domain_first_page{}, domain_end_page{}, reserve_pages{}, guard_pages{};
    std::uint64_t hydrate_bytes{}, scoring_h2d_bytes{};
    double prefill_ms{}, capture_wait_ms{}, scoring_h2d_ms{}, scoring_compute_ms{},
        score_d2h_ms{}, selection_ms{}, hydrate_ms{}, publish_ms{};
    // H2D/D2H are serial CPU wall intervals including host/worker readiness.
    // Only scoring_compute_ms is a CUDA-event interval. Stages are not forced to sum.
};
struct SparseLoadingInfo {
    kvmem::HostArchiveMode archive_mode{};
    std::size_t host_payload_bytes{}, pinned_bytes{}, device_backing_bytes{}, owned_device_bytes{};
    std::uint64_t physical_remaining_bytes{};
};
class KVMemColdResetRequired : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Serialized C=1 owner. Main cache, backing and compute stream outlive this object.
// The backing is the exact, fixed load-time TieredRuntimePlan allocation.
class TieredContext {
public:
    TieredContext(const TieredRuntimePlan& plan, PagedKVCache& main, DeviceSpan backing,
                  cudaStream_t compute_stream,
                  kvmem::HostKVTransferFaultInjection test_fault = {});
    ~TieredContext();
    TieredContext(const TieredContext&)            = delete;
    TieredContext& operator=(const TieredContext&) = delete;

    [[nodiscard]] bool shadow() const noexcept;
    [[nodiscard]] std::uint32_t view_pages() const noexcept;
    [[nodiscard]] std::uint32_t frontier() const noexcept;
    void bind_pages(std::span<const std::int32_t> lease_ids);
    // Loading attaches this subordinate owner only for the KVMem Main pool.
    // Scheduling remains explicit until the stage4.4 eager route is enabled.
    void attach_sparse_capture(std::unique_ptr<SparseCaptureOwner>);
    [[nodiscard]] SparseCaptureOwner* sparse_capture() noexcept;
    [[nodiscard]] std::uint64_t bundle_identity() const noexcept;
    [[nodiscard]] SparseLoadingInfo loading_info() const;
    [[nodiscard]] SparseExecutionPhase sparse_phase() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept;
    void set_transfer_test_fault(kvmem::HostKVTransferFaultInjection);
    // Caller has validated token/position/source-prefix provenance. Bundle binding is explicit.
    void begin_query(kvmem::QueryProvenance, std::uint64_t caller_bundle_identity);
    void use_query(const kvmem::QueryCaptureHandle&, const kvmem::QueryProvenance&,
                   std::uint64_t caller_bundle_identity);
    // Decode count is actual Main valid columns, never padded draft width. Exact chunks
    // prepare inside begin_block using its actual count before the pre-RoPE hook.
    void prepare_main_transaction(kvmem::MeanKTransactionKind, std::uint32_t actual_count);
    void finish_exact_block();
    // Always called for final application acceptance, including full acceptance and zero/cancel.
    void flush_accepted_frontier(std::uint32_t accepted_count);
    [[nodiscard]] SparseTurnMetrics select_and_publish(const SparseTurnInput&);
    // Drained transition extends CURRENT arbitrary owners to min(V, committed pages).
    [[nodiscard]] std::uint64_t prepare_exact_prefill();
    void capture_pre_rope(std::uint32_t layer, const Tensor& qn, const Tensor& kn, cudaStream_t);
    enum class ExecutionPhase { Prefill, Decode };
    void begin_block(std::uint32_t base, std::uint32_t count, cudaStream_t stream,
                     ExecutionPhase phase = ExecutionPhase::Prefill);
    [[nodiscard]] PagedKVLayerView resident_layer(std::uint32_t layer) const;
    void attention(std::uint32_t layer, const Tensor& q, const Tensor& k, const Tensor& v,
                   const Tensor& positions, float scale, Tensor& out, cudaStream_t stream);
    // Dense A1 already appended Main; preserves dense_out and compares a separate output.
    void shadow_attention(std::uint32_t layer, const Tensor& q, const Tensor& positions,
                          float scale, const Tensor& dense_out, cudaStream_t stream);
    [[nodiscard]] TieredSnapshot capture();
    [[nodiscard]] kvmem::KVViewSnapshot current_view() const;
    void restore(const TieredSnapshot&, cudaStream_t stream);
    void trim(std::uint32_t frontier, cudaStream_t stream);
    void reset(cudaStream_t stream);
    // Sparse whole-bundle reset: hardware preparation and the caller's checked
    // correlated cleanup finish before Main/capture host metadata is committed.
    template<class Cleanup>
    void reset_with_cleanup(cudaStream_t stream, Cleanup&& cleanup) {
        reset_impl(stream, [](void* value) { (*static_cast<std::remove_reference_t<Cleanup>*>(value))(); },
                   std::addressof(cleanup));
    }
    void check_compute_drain(cudaError_t status, const char* operation);
    void synchronize_prefill();
    void drain();
    void log_stats();

private:
    friend struct TieredContextTestAccess;
    void reset_impl(cudaStream_t, void (*cleanup)(void*), void*);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::targets::qwen3_6::detail

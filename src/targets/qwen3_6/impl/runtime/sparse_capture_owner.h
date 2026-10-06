#pragma once
#include "core/arena.h"
#include "targets/qwen3_6/impl/runtime/sparse_capture_resources.h"
#include "core/kvmem/host_meank_index.h"
#include "core/kvmem/query_capture.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <array>
#include <optional>
#include <cuda_runtime_api.h>
namespace ninfer::targets::qwen3_6::detail {
// Source prefix includes IDs, all position axes, modality, image identity/groups.
// It ends at last Q+1, and therefore remains stable as tool/assistant suffixes grow.
[[nodiscard]] kvmem::QueryProvenance
query_provenance(const PreparedPromptData&, std::uint64_t bundle_identity, std::uint64_t lineage,
                 std::uint64_t capture_epoch, std::uint32_t last_n = 16,
                 bool original_user_source = false);
// Revalidate a saved capture using its actual bounded ordinals in a later
// prompt. all_user_text proves the original roles; latest-user source_query
// remains only the fallback for recapture when no valid saved capture exists.
[[nodiscard]] kvmem::QueryProvenance query_provenance_for_ordinals(const PreparedPromptData&,
                                                                   std::uint64_t bundle_identity,
                                                                   std::uint64_t lineage,
                                                                   std::uint64_t capture_epoch,
                                                                   std::span<const std::uint32_t>);
struct SparseDerivedSnapshot {
    kvmem::MeanKSnapshotHandle mean;
    kvmem::QueryCaptureHandle query;
    std::uint32_t frontier{};
    std::uint64_t owner{}, epoch{};
};
struct QueryIndexBinding {
    std::uint64_t owner{}, epoch{}, index_generation{}, capture_id{};
    std::uint32_t index_frontier{};
};
// Load-time subordinate Main owner. All CUDA and cacheable pinned backing is
// fixed here; persistent Q lives in three pageable slots. No model math or KV
// bytes are modified. TieredContext owns this optional component, invokes the
// pre-RoPE hook and includes derived state in capture/restore/reset/drain.
// 4.4 must explicitly prepare Exact/Ordinary/Speculative transactions at Main
// accepted ordinals and flush final application acceptance before scoring.
class SparseCaptureOwner {
  public:
    SparseCaptureOwner(SparseCaptureResources, bool actual_archive_pinned, cudaStream_t compute);
    ~SparseCaptureOwner();
    SparseCaptureOwner(const SparseCaptureOwner&) = delete;
    SparseCaptureOwner& operator=(const SparseCaptureOwner&) = delete;
    [[nodiscard]] const SparseCaptureResources& resources() const noexcept;
    [[nodiscard]] const kvmem::HostMeanKIndex& index() const noexcept;
    [[nodiscard]] const kvmem::QueryCaptureHandle& query() const noexcept;
    [[nodiscard]] bool transaction_pending() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept;
    [[nodiscard]] bool unrecoverable() const noexcept;
    // Pure validation retains the first fatal/failed-drain exception across later drains.
    void throw_if_unrecoverable() const;
    void begin_query(kvmem::QueryProvenance); // drains borrowers, releases active before replacing
    void use_query(const kvmem::QueryCaptureHandle&, const kvmem::QueryProvenance&,
                   std::uint32_t restored_frontier);
    void prepare_transaction(kvmem::MeanKTransactionKind, std::uint32_t first, std::uint32_t count);
    void capture_pre_rope(std::uint32_t layer, const Tensor& qn, const Tensor& kn, cudaStream_t);
    void finish_exact_chunk();
    void flush_accepted_frontier(std::uint32_t accepted_count);
    void discard_transaction();
    [[nodiscard]] SparseDerivedSnapshot capture_snapshot();
    [[nodiscard]] bool snapshot_valid(const SparseDerivedSnapshot&) const noexcept;
    void restore_snapshot(const SparseDerivedSnapshot&);
    void trim(std::uint32_t frontier);
    void reset();
    void prepare_reset(); // checked hardware clear; accepted host metadata remains intact on failure
    void commit_reset(); // only after whole-bundle hardware cleanup has completed
    void drain();
    [[nodiscard]] QueryIndexBinding bind_query(const kvmem::QueryProvenance&,
                                               std::uint32_t frontier) const;
    [[nodiscard]] bool binding_valid(const QueryIndexBinding&, const kvmem::QueryProvenance&,
                                     std::uint32_t frontier) const noexcept;
    [[nodiscard]] kvmem::QueryCaptureBorrow borrow_query(std::function<void()> completion);
    // Loading-time device buffer + pinned bounce are borrowed until completion.
    // Drain registered consumers before another upload or capture can overwrite it.
    [[nodiscard]] Tensor upload_query();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ninfer::targets::qwen3_6::detail

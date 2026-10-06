#include "targets/qwen3_6/impl/runtime/sparse_capture_owner.h"
#include "targets/qwen3_6/impl/frontend/digest.h"
#include "core/device.h"
#include "core/kvmem/checked_buffers.h"
#define CAPTURE_CHECK(call) checked((call), #call)
#include <ninfer/ops/meank_accumulate.h>
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace ninfer::targets::qwen3_6::detail {
namespace {
std::atomic<std::uint64_t> next_owner{1};
std::uint32_t patches(std::uint32_t first, std::uint32_t count) {
    return (first % 64 + count + 63) / 64;
}
} // namespace
static kvmem::QueryProvenance query_provenance_impl(const PreparedPromptData& prompt,
                                                    std::uint64_t bundle, std::uint64_t lineage,
                                                    std::uint64_t epoch, std::uint32_t last_n,
                                                    bool original,
                                                    std::span<const std::uint32_t> known) {
    if (!prompt.input_spans.available || !last_n || last_n > 16 ||
        prompt.token_types.size() != prompt.token_ids.size() ||
        prompt.positions.size() != 3 * prompt.token_ids.size())
        throw std::invalid_argument(
            "query capture requires explicit frontend token/position provenance");
    kvmem::QueryProvenance result;
    result.bundle_identity = bundle;
    result.lineage = lineage;
    result.capture_epoch = epoch;
    const auto& spans = original ? prompt.input_spans.source_query : prompt.input_spans.query;
    if (known.empty() && !original)
        for (const auto& query : spans) {
            if (!std::any_of(prompt.input_spans.current.begin(), prompt.input_spans.current.end(),
                             [&](const auto& current) {
                                 return query.begin >= current.begin &&
                                        query.begin + query.count <= current.begin + current.count;
                             }))
                throw std::invalid_argument("new user query lies outside current input event");
        }
    if (!known.empty()) {
        for (const auto t : known) {
            if (t >= prompt.token_ids.size() || prompt.token_types[t] != 0 ||
                !std::any_of(prompt.input_spans.all_user_text.begin(),
                             prompt.input_spans.all_user_text.end(), [&](const auto& user) {
                                 return user.count && user.begin <= prompt.token_ids.size() &&
                                        user.count <= prompt.token_ids.size() - user.begin &&
                                        t >= user.begin && t - user.begin < user.count;
                             }))
                throw std::invalid_argument("saved query ordinal is not proven user text");
            result.ordinals.push_back(t);
        }
    } else
        for (const auto& span : spans) {
            if (!span.count || span.begin > prompt.token_ids.size() ||
                span.count > prompt.token_ids.size() - span.begin)
                throw std::invalid_argument("query span outside prepared prompt");
            for (auto t = span.begin; t < span.begin + span.count; ++t) {
                if (prompt.token_types[t] != 0)
                    throw std::invalid_argument("query contains media token");
                if (t > std::numeric_limits<std::uint32_t>::max())
                    throw std::overflow_error("query ordinal");
                result.ordinals.push_back(static_cast<std::uint32_t>(t));
            }
        }
    if (result.ordinals.empty())
        throw std::invalid_argument("no valid user text query capture source");
    if (!std::is_sorted(result.ordinals.begin(), result.ordinals.end()) ||
        std::adjacent_find(result.ordinals.begin(), result.ordinals.end()) != result.ordinals.end())
        throw std::invalid_argument("query spans overlap or are unordered");
    if (result.ordinals.size() > last_n)
        result.ordinals.erase(result.ordinals.begin(), result.ordinals.end() - last_n);
    result.covered_frontier = result.ordinals.back() + 1;
    std::vector<std::uint8_t> identity;
    identity.reserve(std::size_t(result.covered_frontier) * 17);
    const auto append = [&](std::uint64_t value) {
        for (int b = 0; b < 8; ++b)
            identity.push_back(static_cast<std::uint8_t>(value >> (8 * b)));
    };
    append(result.covered_frontier);
    for (std::size_t t = 0; t < result.covered_frontier; ++t) {
        append(static_cast<std::uint32_t>(prompt.token_ids[t]));
        identity.push_back(prompt.token_types[t]);
        for (int axis = 0; axis < 3; ++axis)
            append(static_cast<std::uint32_t>(prompt.position_axis(axis)[t]));
    }
    for (const auto& item : prompt.vision_items) {
        if (item.token_spans.empty())
            throw std::invalid_argument("image identity missing token spans");
        if (item.token_spans.front().begin >= result.covered_frontier)
            continue;
        append(item.patch_begin);
        append(item.patch_count);
        append(item.timestamps.size());
        for (double timestamp : item.timestamps)
            append(std::bit_cast<std::uint64_t>(timestamp));
        append(item.occurrence_id);
        append(item.source_message);
        append(static_cast<std::uint8_t>(item.modality));
        append(item.grid.temporal);
        append(item.grid.height);
        append(item.grid.width);
        identity.insert(identity.end(), item.content_digest.begin(), item.content_digest.end());
        append(item.token_spans.size());
        for (const auto& span : item.token_spans) {
            if (span.begin + span.count > result.covered_frontier)
                throw std::invalid_argument("query prefix divides image atomic group");
            append(span.begin);
            append(span.count);
        }
    }
    result.source_prefix = frontend_internal::sha256(identity);
    return result;
}
kvmem::QueryProvenance query_provenance(const PreparedPromptData& prompt, std::uint64_t bundle,
                                        std::uint64_t lineage, std::uint64_t epoch,
                                        std::uint32_t last_n, bool original) {
    return query_provenance_impl(prompt, bundle, lineage, epoch, last_n, original, {});
}
kvmem::QueryProvenance query_provenance_for_ordinals(const PreparedPromptData& prompt,
                                                     std::uint64_t bundle, std::uint64_t lineage,
                                                     std::uint64_t epoch,
                                                     std::span<const std::uint32_t> known) {
    if (known.empty() || known.size() > 16)
        throw std::invalid_argument("saved query requires one to sixteen actual ordinals");
    return query_provenance_impl(prompt, bundle, lineage, epoch, 16, true, known);
}
struct SparseCaptureOwner::Impl {
    SparseCaptureResources plan;
    cudaStream_t compute;
    kvmem::CheckedDeviceBuffer device;
    kvmem::CheckedPinnedHostBuffer bounce;
    kvmem::HostMeanKIndex index;
    kvmem::QueryCapturePool captures;
    kvmem::QueryCaptureHandle active;
    std::optional<kvmem::QueryProvenance> collecting;
    std::optional<kvmem::MeanKTicket> transaction;
    std::array<bool, 16> layers{};
    std::array<std::uint16_t, 16> query_rows{};
    std::uint64_t owner = next_owner.fetch_add(1), epoch = 1;
    cudaEvent_t completion{};
    bool poisoned = false, unrecoverable = false;
    bool reset_prepared = false;
    std::exception_ptr unrecoverable_cause;
    Impl(SparseCaptureResources p, bool pinned, cudaStream_t stream)
        : plan(p), compute(stream), device(p.device_bytes), bounce(p.pinned_bytes),
          index(p.mean, pinned), captures(p.query) {
        if (p != plan_sparse_capture_resources(p.mean.max_context, p.mean.max_chunk,
                                               p.query.capacity) ||
            !stream)
            throw std::invalid_argument("Main capture loading plan or stream");
        CAPTURE_CHECK(cudaEventCreateWithFlags(&completion, cudaEventDisableTiming));
        try {
            // DeviceBuffer/other loading resources may be initialized on default stream.
            CAPTURE_CHECK(cudaDeviceSynchronize());
            CAPTURE_CHECK(cudaMemsetAsync(device.p, 0, device.bytes, compute));
            CAPTURE_CHECK(cudaStreamSynchronize(compute));
        } catch (...) {
            cudaEventDestroy(completion);
            throw;
        }
    }
    ~Impl() {
        try {
            drain();
        } catch (...) {
            cudaStreamSynchronize(compute);
        }
        cudaEventDestroy(completion);
    }
    void checked(cudaError_t status, const char* operation, bool drain_failed = false) {
        if (status == cudaSuccess) return;
        poisoned = true;
        const bool fatal = kvmem::cuda_transfer_failure_is_unrecoverable(status, drain_failed);
        unrecoverable |= fatal;
        try { kvmem::check_transfer_cuda(status, operation, drain_failed); }
        catch (...) {
            if (fatal) kvmem::prefer_unrecoverable_exception(unrecoverable_cause, std::current_exception());
            throw;
        }
    }
    void throw_if_unrecoverable() const {
        if (unrecoverable_cause) std::rethrow_exception(unrecoverable_cause);
        if (unrecoverable) throw std::runtime_error("Main capture fatal/failed drain cannot reset CUDA resources");
    }
    void healthy() const {
        if (poisoned)
            throw std::logic_error("Main capture owner poisoned");
    }
    void drain() {
        // Even failure must not skip draining borrowed DMA/device addresses.
        std::exception_ptr first;
        try { captures.drain_borrows(); }
        catch (...) {
            poisoned = unrecoverable = true;
            first = std::current_exception();
            kvmem::prefer_unrecoverable_exception(unrecoverable_cause, first);
        }
        try { checked(cudaStreamSynchronize(compute), "capture borrower stream drain", true); }
        catch (...) { kvmem::prefer_unrecoverable_exception(first, std::current_exception()); }
        if (first) {
            kvmem::prefer_unrecoverable_exception(unrecoverable_cause, first);
            first = unrecoverable_cause;
        }
        if (first) std::rethrow_exception(first);
    }
    void* ptr(std::size_t offset) { return static_cast<std::byte*>(device.p) + offset; }
    void* host(std::size_t offset) { return static_cast<std::byte*>(bounce.data()) + offset; }
    Tensor matrix(std::size_t offset, DType dtype, int heads, int columns = 1) {
        return Tensor(ptr(offset), dtype, {256, heads, columns});
    }
    std::size_t row() const { return plan.mean.row_elements; }
    Tensor k_tail(std::uint32_t layer) {
        return matrix(plan.mean.tail_offset + layer * row() * 64 * 2, DType::BF16, 4, 64);
    }
    Tensor provisional(std::uint32_t layer) {
        return matrix(plan.mean.provisional_offset + layer * row() * 16 * 2, DType::BF16, 4, 16);
    }
    void seeds(bool base = false) {
        auto* saved = static_cast<float*>(host(plan.mean.means_bytes));
        for (std::uint32_t l = 0; l < 16; ++l) {
            const auto sum = base ? index.base_sum(l) : index.prefix_sum(l);
            std::copy(sum.begin(), sum.end(), saved + l * row());
        }
        CAPTURE_CHECK(cudaMemcpyAsync(ptr(plan.mean.seed_sum_offset), saved, plan.mean.sum_bytes,
                                   cudaMemcpyHostToDevice, compute));
    }
    void accumulate(std::uint32_t layer, const Tensor& k, std::uint32_t first, std::uint32_t count,
                    bool retain_tail = true) {
        const auto& r = plan.mean;
        ops::MeanKOutput output{
            matrix(r.means_offset + layer * row() * r.patch_pages * 2, DType::FP16, 4,
                   int(r.patch_pages)),
            Tensor(ptr(r.output_sum_offset + layer * row() * 4), DType::FP32, {256, 4}),
            Tensor(ptr(r.counts_offset + layer * 4), DType::I32, {1})};
        const Tensor seed(ptr(r.seed_sum_offset + layer * row() * 4), DType::FP32, {256, 4});
        ops::meank_accumulate(k, first, count, seed, first % 64, output, compute);
        if (count && retain_tail)
            ops::meank_retain_tail(k, first, count, k_tail(layer), compute);
        const auto bytes = std::size_t(patches(first, count)) * row() * 2;
        if (bytes)
            CAPTURE_CHECK(cudaMemcpyAsync(host(layer * row() * r.patch_pages * 2), output.means.data,
                                       bytes, cudaMemcpyDeviceToHost, compute));
        CAPTURE_CHECK(cudaMemcpyAsync(host(r.means_bytes + layer * row() * 4), output.sum.data,
                                   row() * 4, cudaMemcpyDeviceToHost, compute));
    }
    void publish_index() {
        CAPTURE_CHECK(cudaEventRecord(completion, compute));
        checked(cudaEventSynchronize(completion), "Main exact capture publication drain", true);
        const auto ticket = *transaction;
        const auto pages = patches(ticket.first, ticket.count);
        for (std::uint32_t layer = 0; layer < 16; ++layer) {
            const auto means = std::span<const std::uint16_t>(
                static_cast<const std::uint16_t*>(host(layer * row() * plan.mean.patch_pages * 2)),
                pages * row());
            const auto sums = std::span<const float>(
                static_cast<const float*>(host(plan.mean.means_bytes + layer * row() * 4)), row());
            if (!index.complete_layer(ticket, layer, means, sums))
                throw std::logic_error("Main index rejected current capture completion");
        }
        transaction.reset();
    }
    void publish_query_if_ready() {
        if (!collecting)
            return;
        const auto count = collecting->ordinals.size();
        const std::uint16_t expected = static_cast<std::uint16_t>((1U << count) - 1);
        if (!std::all_of(query_rows.begin(), query_rows.end(),
                         [&](auto rows) { return rows == expected; }))
            return;
        const auto row_count = count * 256 * 24;
        CAPTURE_CHECK(cudaMemcpyAsync(host(plan.mean.host_patch_bytes), ptr(plan.mean.device_bytes),
                                   row_count * 16 * 2, cudaMemcpyDeviceToHost, compute));
        CAPTURE_CHECK(cudaEventRecord(completion, compute));
        checked(cudaEventSynchronize(completion), "Main query capture publication drain", true);
        const auto* q = static_cast<const std::uint16_t*>(host(plan.mean.host_patch_bytes));
        for (std::uint32_t l = 0; l < 16; ++l)
            captures.complete_layer(l, {q + l * row_count, row_count});
        active = captures.publish();
        collecting.reset();
    }
};
SparseCaptureOwner::SparseCaptureOwner(SparseCaptureResources r, bool pinned,
                                       cudaStream_t compute) {
    // Verify before allocating any user-forged sizes.
    if (r != plan_sparse_capture_resources(r.mean.max_context, r.mean.max_chunk, r.query.capacity))
        throw std::invalid_argument("noncanonical Main capture plan");
    impl_ = std::make_unique<Impl>(r, pinned, compute);
}
SparseCaptureOwner::~SparseCaptureOwner() = default;
const SparseCaptureResources& SparseCaptureOwner::resources() const noexcept { return impl_->plan; }
const kvmem::HostMeanKIndex& SparseCaptureOwner::index() const noexcept { return impl_->index; }
const kvmem::QueryCaptureHandle& SparseCaptureOwner::query() const noexcept {
    return impl_->active;
}
bool SparseCaptureOwner::poisoned() const noexcept { return impl_->poisoned; }
bool SparseCaptureOwner::unrecoverable() const noexcept { return impl_->unrecoverable; }
void SparseCaptureOwner::throw_if_unrecoverable() const { impl_->throw_if_unrecoverable(); }
bool SparseCaptureOwner::transaction_pending() const noexcept {
    return impl_->transaction.has_value();
}
void SparseCaptureOwner::drain() { impl_->drain(); }
void SparseCaptureOwner::begin_query(kvmem::QueryProvenance p) {
    auto& i = *impl_;
    i.healthy();
    i.drain();
    if (i.transaction)
        throw std::logic_error("replace Q during Main transaction");
    if (p.ordinals.empty() || p.ordinals.front() < i.index.frontier() ||
        p.covered_frontier > i.plan.mean.max_context)
        throw std::invalid_argument("query capture beyond admitted Main context");
    i.captures.discard();
    i.collecting.reset();
    i.active.reset();
    i.captures.begin(p);
    i.collecting = std::move(p);
    i.query_rows.fill(0);
}
void SparseCaptureOwner::use_query(const kvmem::QueryCaptureHandle& q,
                                   const kvmem::QueryProvenance& p, std::uint32_t frontier) {
    auto& i = *impl_;
    i.healthy();
    i.drain();
    if (i.transaction || !i.captures.owns(q) || !kvmem::query_capture_matches(q, p, frontier))
        throw std::invalid_argument("tool continuation Q prefix/bundle/lineage/frontier mismatch");
    i.captures.discard();
    i.collecting.reset();
    i.active = q;
}
void SparseCaptureOwner::prepare_transaction(kvmem::MeanKTransactionKind kind, std::uint32_t first,
                                             std::uint32_t count) {
    auto& i = *impl_;
    i.healthy();
    if (i.transaction || first != i.index.frontier())
        throw std::invalid_argument("Main capture accepted frontier");
    i.drain();
    i.transaction = i.index.begin(kind, count);
    i.layers.fill(false);
    i.seeds();
}
void SparseCaptureOwner::capture_pre_rope(std::uint32_t layer, const Tensor& qn, const Tensor& kn,
                                          cudaStream_t stream) {
    auto& i = *impl_;
    i.healthy();
    if (stream != i.compute || !i.transaction || layer >= 16 || i.layers[layer])
        throw std::invalid_argument("pre-RoPE capture stream/transaction/layer");
    const auto& ticket = *i.transaction;
    if (qn.dtype != DType::BF16 || qn.ne[0] != 256 || qn.ne[1] != 24 || kn.dtype != DType::BF16 ||
        kn.ne[0] != 256 || kn.ne[1] != 4 || qn.ne[2] < int(ticket.count) ||
        kn.ne[2] < int(ticket.count) || qn.nb[0] != 2 || qn.nb[1] != 512 || qn.nb[2] != 12288 ||
        kn.nb[0] != 2 || kn.nb[1] != 512 || kn.nb[2] != 2048)
        throw std::invalid_argument("pre-RoPE capture normalized tensor shape/strides");
    if (ticket.kind == kvmem::MeanKTransactionKind::ExactPrefill) {
        i.accumulate(layer, kn, ticket.first, ticket.count);
        if (i.collecting) {
            const auto& ordinals = i.collecting->ordinals;
            for (std::size_t q = 0; q < ordinals.size(); ++q) {
                const auto ordinal = ordinals[q];
                if (ordinal < ticket.first || ordinal >= ticket.first + ticket.count)
                    continue;
                if (i.query_rows[layer] & (1U << q))
                    throw std::logic_error("Q ordinal captured twice");
                const auto bytes = 256ULL * 24 * 2;
                i.CAPTURE_CHECK(cudaMemcpyAsync(
                    i.ptr(i.plan.mean.device_bytes + (layer * ordinals.size() + q) * bytes),
                    static_cast<const std::byte*>(qn.data) + (ordinal - ticket.first) * bytes,
                    bytes, cudaMemcpyDeviceToDevice, stream));
                i.query_rows[layer] |= std::uint16_t(1U << q);
            }
        }
    } else {
        ops::meank_stage(kn, ticket.count, i.provisional(layer), stream);
    }
    i.layers[layer] = true;
}
void SparseCaptureOwner::finish_exact_chunk() {
    auto& i = *impl_;
    i.healthy();
    if (!i.transaction || i.transaction->kind != kvmem::MeanKTransactionKind::ExactPrefill ||
        !std::all_of(i.layers.begin(), i.layers.end(), [](bool b) { return b; }))
        throw std::logic_error("exact capture chunk requires all Main layers");
    i.publish_index();
    i.publish_query_if_ready();
}
void SparseCaptureOwner::flush_accepted_frontier(std::uint32_t n) {
    auto& i = *impl_;
    i.healthy();
    if (!i.transaction || i.transaction->kind == kvmem::MeanKTransactionKind::ExactPrefill ||
        !std::all_of(i.layers.begin(), i.layers.end(), [](bool b) { return b; }))
        throw std::logic_error("accepted Main capture requires complete ordinary/spec transaction");
    auto ticket = *i.transaction;
    if (n > ticket.count || (ticket.kind == kvmem::MeanKTransactionKind::OrdinaryMain && n > 1))
        throw std::invalid_argument("accepted Main capture count");
    if (!n) {
        discard_transaction();
        return;
    }
    if (ticket.kind == kvmem::MeanKTransactionKind::SpeculativeMain)
        ticket = i.index.accept(ticket, n);
    i.transaction = ticket;
    for (std::uint32_t layer = 0; layer < 16; ++layer)
        i.accumulate(layer, i.provisional(layer), ticket.first, n);
    i.publish_index();
}
void SparseCaptureOwner::discard_transaction() {
    auto& i = *impl_;
    i.drain();
    if (i.transaction) {
        if (i.transaction->kind == kvmem::MeanKTransactionKind::ExactPrefill)
            throw std::logic_error(
                "discarding incomplete exact prefill requires Main bundle reset");
        i.index.discard(*i.transaction);
        i.transaction.reset();
    }
    i.captures.discard();
    i.collecting.reset();
    i.query_rows.fill(0);
}
SparseDerivedSnapshot SparseCaptureOwner::capture_snapshot() {
    auto& i = *impl_;
    i.healthy();
    i.drain();
    if (i.transaction)
        throw std::logic_error("snapshot before accepted Main publication");
    return {i.index.capture(), i.active, i.index.frontier(), i.owner, i.epoch};
}
bool SparseCaptureOwner::snapshot_valid(const SparseDerivedSnapshot& snapshot) const noexcept {
    const auto& i = *impl_;
    return !i.poisoned && snapshot.owner == i.owner && snapshot.epoch == i.epoch && snapshot.mean &&
        i.index.snapshot_matches(snapshot.mean, snapshot.frontier) &&
        (!snapshot.query || (i.captures.owns(snapshot.query) &&
                            snapshot.query->provenance().covered_frontier <= snapshot.frontier));
}
void SparseCaptureOwner::restore_snapshot(const SparseDerivedSnapshot& snapshot) {
    auto& i = *impl_;
    i.healthy();
    if (snapshot.owner != i.owner || snapshot.epoch != i.epoch || !snapshot.mean ||
        !i.index.snapshot_matches(snapshot.mean, snapshot.frontier))
        throw std::invalid_argument("foreign/stale Main derived snapshot");
    if (snapshot.query && (!i.captures.owns(snapshot.query) ||
                           snapshot.query->provenance().covered_frontier > snapshot.frontier))
        throw std::invalid_argument("Main snapshot Q beyond restored prefix");
    i.drain();
    i.transaction = i.index.restore(snapshot.mean);
    if (i.transaction->first != snapshot.frontier)
        throw std::invalid_argument("Main snapshot frontier identity");
    i.captures.discard();
    i.collecting.reset();
    i.query_rows.fill(0);
    i.active = snapshot.query;
    i.seeds();
    i.CAPTURE_CHECK(
        cudaMemsetAsync(i.ptr(i.plan.mean.tail_offset), 0, i.plan.mean.tail_bytes, i.compute));
    for (std::uint32_t l = 0; l < 16; ++l)
        i.accumulate(l, i.provisional(l), snapshot.frontier, 0);
    i.publish_index();
}
void SparseCaptureOwner::trim(std::uint32_t frontier) {
    auto& i = *impl_;
    i.healthy();
    i.drain();
    if (i.transaction)
        throw std::logic_error("flush accepted Main before trim");
    if (frontier == i.index.frontier())
        return;
    const auto base = i.index.base_frontier();
    i.transaction = i.index.begin_trim(frontier);
    i.seeds(true);
    for (std::uint32_t l = 0; l < 16; ++l) {
        auto tail = i.k_tail(l).slice(2, int(base % 64), int(std::max(1U, frontier - base)));
        i.accumulate(l, tail, base, frontier - base, false);
    }
    i.publish_index();
    if (i.active && i.active->provenance().covered_frontier > frontier)
        i.active.reset();
    if (i.collecting) {
        i.captures.discard();
        i.collecting.reset();
        i.query_rows.fill(0);
    }
}
void SparseCaptureOwner::reset() {
    prepare_reset();
    commit_reset();
}
void SparseCaptureOwner::prepare_reset() {
    auto& i = *impl_;
    i.reset_prepared = false;
    i.drain();
    i.throw_if_unrecoverable();
    i.poisoned = true;
    i.CAPTURE_CHECK(cudaMemsetAsync(i.device.p, 0, i.device.bytes, i.compute));
    i.checked(cudaStreamSynchronize(i.compute), "Main capture reset drain", true);
    i.reset_prepared = true;
}
void SparseCaptureOwner::commit_reset() {
    auto& i = *impl_;
    i.throw_if_unrecoverable();
    if (!i.reset_prepared) throw std::logic_error("Main capture reset was not prepared");
    // prepare_reset already drained every borrower before issuing the clears.
    i.captures.reset();
    i.index.reset();
    i.transaction.reset();
    i.active.reset();
    i.collecting.reset();
    ++i.epoch;
    i.query_rows.fill(0);
    i.reset_prepared = false;
    i.poisoned = false;
}
QueryIndexBinding SparseCaptureOwner::bind_query(const kvmem::QueryProvenance& p,
                                                 std::uint32_t frontier) const {
    auto& i = *impl_;
    i.healthy();
    if (!i.index.published() || i.index.frontier() != frontier ||
        !kvmem::query_capture_matches(i.active, p, frontier))
        throw std::invalid_argument("score binding index/Q provenance");
    return {i.owner, i.epoch, i.index.generation(), i.active->id(), frontier};
}
bool SparseCaptureOwner::binding_valid(const QueryIndexBinding& b, const kvmem::QueryProvenance& p,
                                       std::uint32_t frontier) const noexcept {
    const auto& i = *impl_;
    return !i.poisoned && i.index.published() && b.owner == i.owner && b.epoch == i.epoch &&
           b.index_generation == i.index.generation() && b.index_frontier == i.index.frontier() &&
           frontier == b.index_frontier && i.active && b.capture_id == i.active->id() &&
           kvmem::query_capture_matches(i.active, p, frontier);
}
kvmem::QueryCaptureBorrow SparseCaptureOwner::borrow_query(std::function<void()> completion) {
    impl_->healthy();
    return impl_->captures.borrow(impl_->active, std::move(completion));
}
Tensor SparseCaptureOwner::upload_query() {
    auto& i = *impl_;
    i.healthy();
    i.drain();
    if (i.collecting || i.transaction || !i.active || !i.active->valid())
        throw std::logic_error("query upload before immutable capture publication");
    const auto count = i.active->provenance().ordinals.size();
    const auto rows = count * 256 * 24;
    auto* pinned = static_cast<std::uint16_t*>(i.host(i.plan.mean.host_patch_bytes));
    for (std::uint32_t l = 0; l < 16; ++l) {
        const auto q = i.active->layer(l);
        std::copy(q.begin(), q.end(), pinned + l * rows);
    }
    i.CAPTURE_CHECK(cudaMemcpyAsync(i.ptr(i.plan.mean.device_bytes), pinned, rows * 16 * 2,
                               cudaMemcpyHostToDevice, i.compute));
    return Tensor(i.ptr(i.plan.mean.device_bytes), DType::BF16, {256, 24, int(count), 16});
}
} // namespace ninfer::targets::qwen3_6::detail

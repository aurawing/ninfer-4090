#include "core/kvmem/query_capture.h"
#include "core/kvmem/cuda_status.h"
#include <algorithm>
#include <atomic>
#include <exception>
#include <stdexcept>
#include <utility>
namespace ninfer::kvmem {
namespace {
std::atomic<std::uint64_t> next_capture{1};
}
QueryCaptureResources plan_query_capture_resources(std::uint32_t capacity) {
    if (!capacity || capacity > 16)
        throw std::invalid_argument("query capacity must be in [1,16]");
    QueryCaptureResources p;
    p.capacity = capacity;
    p.slot_bytes = std::size_t(p.layers) * p.dimension * p.heads * p.capacity * 2;
    p.pageable_bytes = 3 * p.slot_bytes;
    p.device_bytes = p.pinned_bytes = p.slot_bytes;
    return p;
}
class QueryCaptureStorage {
  public:
    QueryCaptureResources resource;
    std::vector<std::uint16_t> data;
    std::array<std::weak_ptr<const QueryCapture>, 3> slots;
    std::uint64_t epoch = 1;
    std::uint32_t high_watermark = 0;
};
std::uint64_t QueryCapture::id() const noexcept { return id_; }
bool QueryCapture::valid() const noexcept { return storage_ && epoch_ == storage_->epoch; }
const QueryProvenance& QueryCapture::provenance() const noexcept { return provenance_; }
std::span<const std::uint16_t> QueryCapture::layer(std::uint32_t layer) const {
    if (!valid() || layer >= storage_->resource.layers)
        throw std::invalid_argument("stale Q capture/layer");
    const auto& r = storage_->resource;
    const auto size = provenance_.ordinals.size() * r.dimension * r.heads;
    return {storage_->data.data() + slot_ * (r.slot_bytes / 2) + layer * size, size};
}
struct QueryCaptureBorrow::State : std::enable_shared_from_this<QueryCaptureBorrow::State> {
    QueryCaptureHandle handle;
    std::function<void()> drain;
    // A failed completion retains its source even if the public borrower/pool dies.
    // Successful explicit retry releases this exceptional quarantine lease.
    std::shared_ptr<State> failed_lease;
    void finish() {
        if (!handle)
            return;
        try { drain(); }
        catch (...) { failed_lease = shared_from_this(); throw; }
        handle.reset();
        failed_lease.reset();
    }
};
QueryCaptureBorrow::~QueryCaptureBorrow() {
    if (state_) {
        try {
            state_->finish();
        } catch (...) {
            // The pool and failed_lease retain the source until a proven drain.
        }
    }
}
QueryCaptureBorrow::QueryCaptureBorrow(QueryCaptureBorrow&& other) noexcept
    : state_(std::move(other.state_)) {}
QueryCaptureBorrow& QueryCaptureBorrow::operator=(QueryCaptureBorrow&& other) noexcept {
    if (this != &other) {
        if (state_) {
            try {
                state_->finish();
            } catch (...) {
                // The pool and failed_lease retain the source until a proven drain.
            }
        }
        state_ = std::move(other.state_);
    }
    return *this;
}
bool QueryCaptureBorrow::valid() const noexcept {
    return state_ && state_->handle && state_->handle->valid();
}
QueryCaptureHandle QueryCaptureBorrow::capture() const {
    if (!valid())
        throw std::invalid_argument("drained Q borrow");
    return state_->handle;
}
void QueryCaptureBorrow::finish() {
    if (state_)
        state_->finish();
}
QueryCapturePool::QueryCapturePool(QueryCaptureResources r)
    : storage_(std::make_shared<QueryCaptureStorage>()) {
    if (r != plan_query_capture_resources(r.capacity))
        throw std::invalid_argument("query resource layout");
    storage_->resource = r;
    storage_->data.resize(r.pageable_bytes / 2);
}
QueryCapturePool::~QueryCapturePool() {
    try {
        drain_borrows();
    } catch (...) {
        // An unknown completion must retain source storage. Never invoke the
        // callbacks after the owning pool/runtime has been destroyed.
        for (auto& task : borrows_) if (task->handle) task->drain = {};
    }
}
void QueryCapturePool::begin(QueryProvenance p) {
    if (pending_)
        throw std::logic_error("Q capture already collecting");
    if (!p.bundle_identity || !p.lineage || p.ordinals.empty() ||
        p.ordinals.size() > storage_->resource.capacity ||
        !std::is_sorted(p.ordinals.begin(), p.ordinals.end()) ||
        std::adjacent_find(p.ordinals.begin(), p.ordinals.end()) != p.ordinals.end() ||
        p.ordinals.back() >= p.covered_frontier)
        throw std::invalid_argument("Q provenance ordinals/frontier/lineage");
    const auto slot = std::find_if(storage_->slots.begin(), storage_->slots.end(),
                                   [](const auto& s) { return s.expired(); });
    if (slot == storage_->slots.end())
        throw std::logic_error("maximum three distinct live Q captures");
    pending_ = std::make_shared<QueryCapture>();
    pending_->storage_ = storage_;
    pending_->provenance_ = std::move(p);
    pending_->epoch_ = storage_->epoch;
    pending_->id_ = next_capture.fetch_add(1);
    pending_->slot_ = static_cast<std::uint32_t>(slot - storage_->slots.begin());
    *slot = pending_;
    layers_.fill(false);
    storage_->high_watermark = std::max(storage_->high_watermark, live_slots());
}
void QueryCapturePool::complete_layer(std::uint32_t layer, std::span<const std::uint16_t> rows) {
    if (!pending_ || layer >= layers_.size() || layers_[layer])
        throw std::logic_error("duplicate Q layer");
    const auto& r = storage_->resource;
    const auto count = pending_->provenance_.ordinals.size() * r.dimension * r.heads;
    if (rows.size() != count ||
        !std::all_of(rows.begin(), rows.end(), [](auto v) { return (v & 0x7f80) != 0x7f80; }))
        throw std::invalid_argument("Q capture row count/nonfinite");
    std::copy(rows.begin(), rows.end(),
              storage_->data.begin() + pending_->slot_ * (r.slot_bytes / 2) + layer * count);
    layers_[layer] = true;
}
QueryCaptureHandle QueryCapturePool::publish() {
    if (!pending_ || !std::all_of(layers_.begin(), layers_.end(), [](bool ready) { return ready; }))
        throw std::logic_error("Q publication requires all 16 layers");
    QueryCaptureHandle result = std::move(pending_);
    return result;
}
void QueryCapturePool::discard() {
    pending_.reset();
    layers_.fill(false);
}
QueryCaptureBorrow QueryCapturePool::borrow(const QueryCaptureHandle& handle,
                                            std::function<void()> drain) {
    if (!handle || !handle->valid() || handle->storage_ != storage_ || !drain)
        throw std::invalid_argument("Q borrow owner/lifetime/completion");
    QueryCaptureBorrow result;
    result.state_ = std::make_shared<QueryCaptureBorrow::State>();
    result.state_->handle = handle;
    result.state_->drain = std::move(drain);
    borrows_.erase(
        std::remove_if(borrows_.begin(), borrows_.end(), [](const auto& task) { return !task->handle; }),
        borrows_.end());
    borrows_.push_back(result.state_);
    return result;
}
void QueryCapturePool::drain_borrows() {
    std::exception_ptr first;
    for (auto& task : borrows_) {
        try { task->finish(); }
        catch (...) { prefer_unrecoverable_exception(first, std::current_exception()); }
    }
    std::erase_if(borrows_, [](const auto& task) { return !task->handle; });
    if (first) std::rethrow_exception(first);
}
void QueryCapturePool::reset() {
    drain_borrows();
    discard();
    ++storage_->epoch;
}
bool QueryCapturePool::owns(const QueryCaptureHandle& capture) const noexcept {
    return capture && capture->valid() && capture->storage_ == storage_;
}
std::uint32_t QueryCapturePool::live_slots() const noexcept {
    return static_cast<std::uint32_t>(std::count_if(storage_->slots.begin(), storage_->slots.end(),
                                                    [](const auto& s) { return !s.expired(); }));
}
std::uint32_t QueryCapturePool::high_watermark() const noexcept { return storage_->high_watermark; }
bool query_capture_matches(const QueryCaptureHandle& capture, const QueryProvenance& p,
                           std::uint32_t frontier) noexcept {
    if (!capture || !capture->valid() || p.ordinals.empty())
        return false;
    const auto& old = capture->provenance();
    return old.bundle_identity == p.bundle_identity && old.lineage == p.lineage &&
           old.source_prefix == p.source_prefix && old.ordinals == p.ordinals &&
           old.covered_frontier >= old.ordinals.back() + 1 && frontier >= old.covered_frontier &&
           p.covered_frontier == old.covered_frontier;
}
QueryRecapturePlan plan_query_recapture(const QueryProvenance& p, const QueryCaptureHandle& capture,
                                        std::uint32_t frontier,
                                        std::span<const ExactCaptureContinuation> states) {
    if (p.ordinals.empty())
        throw std::invalid_argument("no original user query source for capture");
    if (query_capture_matches(capture, p, frontier))
        return {QueryRecaptureKind::Reuse, frontier};
    QueryRecapturePlan result{QueryRecaptureKind::ColdStart, 0};
    for (const auto& state : states)
        if (state.exact_source_prefix && state.bundle_identity == p.bundle_identity &&
            state.lineage == p.lineage && state.frontier <= p.ordinals.front() &&
            state.frontier <= frontier &&
            (result.kind == QueryRecaptureKind::ColdStart || state.frontier > result.frontier))
            result = {QueryRecaptureKind::ExactContinuation, state.frontier};
    return result;
}
} // namespace ninfer::kvmem

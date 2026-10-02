#include "targets/qwen3_6/impl/runtime/mtp_window.h"
#include "core/device.h"
#include <ninfer/ops/gqa_kv_append_masked.h>
#include <ninfer/ops/gqa_attention.h>
#include <ninfer/ops/gqa_attention_partial.h>
#include <algorithm>
#include <atomic>
#include <iostream>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail {
namespace {
std::atomic<std::uint64_t> next_mtp_bundle_identity{1};

// Padding entries are future logical pages with zero prefix contribution. The
// partial operator clips them away through the same original-position mask.
__global__ void window_tag_writes(const int* positions, const int* valid, int width, int column,
                                  const int* table, int* tags) {
    if (threadIdx.x) return;
    const int count = valid ? min(width, max(0, valid[0] - column)) : width;
    for (int j = 0; j < count; ++j) {
        const int logical    = (positions[0] + j) / 64;
        tags[table[logical]] = logical;
    }
}
__global__ void window_access(const int* positions, const int* valid, int column, int sink,
                              int recent, int capacity, const int* table, const int* tags,
                              int2* access, int* prefix) {
    if (threadIdx.x) return;
    const int n     = sink + recent;
    const int live  = (!valid || column < valid[0]) ? positions[column] + 1 : 0;
    const int end   = (live + 63) / 64;
    const int first = max(sink, end - recent);
    int count = 0, keys = 0;
    prefix[0]            = 0;
    const int sinks      = min(sink, end);
    const int candidates = sinks + max(0, end - first);
    for (int i = 0; i < candidates; ++i) {
        const int logical  = i < sinks ? i : first + i - sinks;
        const int physical = table[logical];
        if (tags[physical] != logical) continue;
        access[count] = make_int2(logical, physical);
        keys += min(64, live - logical * 64);
        prefix[++count] = keys;
    }
    for (int i = count; i < n; ++i) {
        access[i]     = make_int2((capacity + 63) / 64 + i, 0);
        prefix[i + 1] = keys;
    }
}
} // namespace

MtpWindow::MtpWindow(const MtpWindowPlan& p, PagedKVCache& cache, DeviceSpan backing,
                     cudaStream_t stream)
    : plan_(p), cache_(cache), backing_(backing) {
    if (backing.bytes < p.bytes || cache.layers() != 1 ||
        cache.pool().page_group_count() != p.physical_pages)
        throw std::invalid_argument("MTP ring owner/cache/backing mismatch");
    mapping_.resize((p.capacity + 63) / 64, -1);
    reset(stream);
}
void MtpWindow::bind_pages(std::span<const std::int32_t> ids, cudaStream_t stream) {
    if (ids.size() != plan_.physical_pages) throw std::invalid_argument("MTP ring lease capacity");
    if (std::equal(ids.begin(), ids.end(), leases_.begin(), leases_.end())) return;
    if (!leases_.empty()) reset(stream);
    leases_.assign(ids.begin(), ids.end());
    for (std::size_t logical = 0; logical < mapping_.size(); ++logical)
        mapping_[logical] =
            leases_[logical < plan_.sink_pages
                        ? logical
                        : plan_.sink_pages + (logical - plan_.sink_pages) % plan_.recent_pages];
    auto table = plan_.block_table.bind(backing_);
    CUDA_CHECK(cudaMemcpyAsync(table.data, mapping_.data(), table.bytes(), cudaMemcpyHostToDevice,
                               stream));
    backup_live_ = false;
}
PagedKVLayerView MtpWindow::layer() const {
    const auto b = cache_.batch_layer_view(0);
    return {b.k_pages,
            b.v_pages,
            b.k_scale_pages,
            b.v_scale_pages,
            plan_.block_table.bind(backing_),
            b.head_dim,
            b.num_kv_heads,
            b.dtype,
            b.quant_group,
            b.packed_v,
            b.rotate_k,
            b.rotate_v,
            b.packed_k,
            b.e8_lattice,
            b.e8_root};
}
void MtpWindow::append(const Tensor& k, const Tensor& v, const Tensor& positions,
                       const Tensor& valid, cudaStream_t stream) {
    if (leases_.empty()) throw std::logic_error("MTP ring is not bound");
    window_tag_writes<<<1, 32, 0, stream>>>(
        static_cast<const int*>(positions.data), static_cast<const int*>(valid.data), k.ne[2], 0,
        static_cast<const int*>(plan_.block_table.bind(backing_).data),
        static_cast<int*>(page_tags().data));
    CUDA_CHECK(cudaGetLastError());
    if (valid.data)
        ops::gqa_kv_append_masked(k, v, positions, valid, 0, layer(), stream);
    else
        ops::gqa_kv_append(k, v, positions, layer(), stream);
}
void MtpWindow::attention(const Tensor& q, const Tensor& positions, const Tensor& valid,
                          float scale, Tensor& out, cudaStream_t stream, const Tensor* k,
                          const Tensor* v) {
    const int tokens = q.ne[2];
    if (tokens < 1 || tokens > 16 || out.ne[2] != tokens)
        throw std::invalid_argument("MTP ring attention supports 1..16 query columns");
    const auto resident = layer();
    auto empty          = resident;
    empty.k_pages       = {};
    empty.v_pages       = {};
    empty.k_scale_pages = {};
    empty.v_scale_pages = {};
    auto access = plan_.access.bind(backing_), prefix = plan_.prefix.bind(backing_);
    ops::AttentionPartial scratch{plan_.scratch_o.bind(backing_), plan_.scratch_m.bind(backing_),
                                  plan_.scratch_l.bind(backing_)};
    ops::AttentionPartial state{plan_.state_o.bind(backing_), plan_.state_m.bind(backing_),
                                plan_.state_l.bind(backing_)};
    for (int column = 0; column < tokens; ++column) {
        if (k && v) {
            window_tag_writes<<<1, 32, 0, stream>>>(
                static_cast<const int*>(positions.slice(0, column, 1).data),
                static_cast<const int*>(valid.data), 1, column,
                static_cast<const int*>(resident.block_table.data),
                static_cast<int*>(page_tags().data));
            CUDA_CHECK(cudaGetLastError());
            ops::gqa_kv_append_masked(k->slice(2, column, 1), v->slice(2, column, 1),
                                      positions.slice(0, column, 1), valid, column, resident,
                                      stream);
        }
        window_access<<<1, 256, 0, stream>>>(
            static_cast<const int*>(positions.data), static_cast<const int*>(valid.data), column,
            plan_.sink_pages, plan_.recent_pages, plan_.capacity,
            static_cast<const int*>(resident.block_table.data),
            static_cast<const int*>(page_tags().data), static_cast<int2*>(access.data),
            static_cast<int*>(prefix.data));
        CUDA_CHECK(cudaGetLastError());
        const auto query = q.slice(2, column, 1), position = positions.slice(0, column, 1);
        auto output = out.slice(2, column, 1);
        ops::gqa_attention_partial_decode(query, position, scale, resident, empty, access, prefix,
                                          plan_.capacity, 64, scratch, stream);
        ops::attention_partial_lse_merge(scratch, state, stream);
        if (resident.rotate_v)
            ops::attention_partial_finalize_rotated(state, output, stream);
        else
            ops::attention_partial_finalize(state, output, stream);
    }
}
void MtpWindow::begin_transaction(std::uint32_t base, std::uint32_t extent, cudaStream_t stream) {
    trim(base, stream);
    if (!plan_.guard_pages || !extent) return;
    if (extent > 63 || std::uint64_t(base) + extent > plan_.capacity)
        throw std::invalid_argument("MTP provisional extent cannot cross more than one page");
    const auto page = (base + extent - 1) / 64;
    if (page < plan_.sink_pages + plan_.recent_pages || page * 64 < base) return;
    const auto old    = page - plan_.recent_pages;
    const auto source = leases_[plan_.sink_pages + (old - plan_.sink_pages) % plan_.recent_pages];
    const auto guard  = leases_.back();
    for (std::size_t plane = 0; plane < cache_.pool().plane_count(); ++plane) {
        const auto& t = cache_.pool().plane(plane);
        auto* bytes   = static_cast<std::byte*>(t.data);
        CUDA_CHECK(cudaMemcpyAsync(bytes + guard * t.nb[3], bytes + source * t.nb[3],
                                   cache_.pool().page_bytes(plane), cudaMemcpyDeviceToDevice,
                                   stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(static_cast<int*>(page_tags().data) + guard,
                               static_cast<const int*>(page_tags().data) + source, sizeof(int),
                               cudaMemcpyDeviceToDevice, stream));
    backup_new_page_ = page;
    backup_live_     = true;
}
void MtpWindow::trim(std::uint32_t frontier, cudaStream_t stream) {
    if (frontier > plan_.capacity)
        throw std::invalid_argument("MTP ring trim exceeds logical capacity");
    ++generation_;
    invalidate_captures(frontier);
    if (!backup_live_) return;
    if (frontier <= backup_new_page_ * 64) {
        const auto old = backup_new_page_ - plan_.recent_pages;
        const auto target =
            leases_[plan_.sink_pages + (old - plan_.sink_pages) % plan_.recent_pages];
        const auto guard = leases_.back();
        for (std::size_t plane = 0; plane < cache_.pool().plane_count(); ++plane) {
            const auto& t = cache_.pool().plane(plane);
            auto* bytes   = static_cast<std::byte*>(t.data);
            CUDA_CHECK(cudaMemcpyAsync(bytes + target * t.nb[3], bytes + guard * t.nb[3],
                                       cache_.pool().page_bytes(plane), cudaMemcpyDeviceToDevice,
                                       stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(static_cast<int*>(page_tags().data) + target,
                                   static_cast<const int*>(page_tags().data) + guard, sizeof(int),
                                   cudaMemcpyDeviceToDevice, stream));
    }
    backup_live_ = false;
}
void MtpWindow::invalidate_captures(std::uint32_t frontier) {
    std::erase_if(captures_, [frontier](const auto& weak) {
        const auto capture = weak.lock();
        if (!capture) return true;
        if (capture->frontier > frontier) capture->valid = false;
        return !capture->valid;
    });
}
MtpWindowSnapshot MtpWindow::capture(std::uint32_t frontier, cudaStream_t stream) {
    if (frontier > plan_.capacity || leases_.empty())
        throw std::invalid_argument("MTP snapshot frontier or bundle unavailable");
    MtpWindowSnapshot saved;
    saved.bundle_identity = bundle_identity_;
    saved.generation = generation_;
    saved.frontier = frontier;
    saved.page_tags.resize(plan_.physical_pages);
    CUDA_CHECK(cudaMemcpyAsync(saved.page_tags.data(), page_tags().data, page_tags().bytes(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::size_t slot = 0; slot < saved.page_tags.size(); ++slot) {
        const auto logical = saved.page_tags[slot];
        if (logical < 0 || std::uint64_t(logical) * 64 >= frontier ||
            logical >= mapping_.size() || mapping_[logical] != slot) saved.page_tags[slot] = -1;
    }
    invalidate_captures(plan_.capacity);
    saved.lifetime = std::make_shared<MtpSnapshotLifetime>();
    saved.lifetime->frontier = frontier;
    captures_.push_back(saved.lifetime);
    return saved;
}
void MtpWindow::restore(const MtpWindowSnapshot& saved, std::uint32_t frontier, cudaStream_t stream) {
    if (saved.bundle_identity != bundle_identity_ || saved.generation > generation_ ||
        !saved.lifetime || !saved.lifetime->valid || saved.lifetime->frontier != saved.frontier ||
        frontier > saved.frontier || saved.page_tags.size() != plan_.physical_pages)
        throw std::logic_error("MTP snapshot is stale or belongs to a different bundle");
    // Resolve a provisional guard before inspecting the current ring tags.
    trim(frontier, stream);
    std::vector<std::int32_t> tags(plan_.physical_pages);
    CUDA_CHECK(cudaMemcpyAsync(tags.data(), page_tags().data, page_tags().bytes(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::size_t missing = 0, retained = 0;
    for (std::size_t slot = 0; slot < tags.size(); ++slot) {
        const auto logical = saved.page_tags[slot];
        const bool desired = logical >= 0 && std::uint64_t(logical) * 64 < frontier &&
                             logical < mapping_.size() && mapping_[logical] == slot;
        if (desired && tags[slot] == logical) ++retained;
        else {
            missing += desired;
            tags[slot] = -1;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(page_tags().data, tags.data(), page_tags().bytes(),
                               cudaMemcpyHostToDevice, stream));
    // Metadata only. Missing historical KV is never replayed or reconstructed.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (missing)
        std::clog << "[kvmem-mtp] restore_frontier=" << frontier << " surviving_pages=" << retained
                  << " missing_snapshot_pages=" << missing
                  << " draft_history=reduced acceptance_may_decline=1 replay=0\n";
}
void MtpWindow::reset(cudaStream_t stream) {
    for (const auto& weak : captures_)
        if (auto capture = weak.lock()) capture->valid = false;
    captures_.clear();
    bundle_identity_ = next_mtp_bundle_identity.fetch_add(1);
    ++generation_;
    CUDA_CHECK(cudaMemsetAsync(page_tags().data, 0xff, page_tags().bytes(), stream));
    backup_live_ = false;
}
} // namespace ninfer::targets::qwen3_6::detail

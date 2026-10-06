#include "core/kvmem/host_kv_transfer.h"
#include "core/kvmem/host_kv_transfer_test_access.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::kvmem;
namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}
template<class F> void rejects(F&& f, const char* message) {
    bool caught = false;
    try { f(); } catch (const std::exception&) { caught = true; }
    require(caught, message);
}
std::vector<std::byte> bytes(std::size_t count, unsigned plane, unsigned page, unsigned version = 0) {
    std::vector<std::byte> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        result[i] = std::byte((i * 37 + plane * 53 + page * 17 + version * 97) & 255);
    }
    return result;
}
void staging_plan_contract() {
    LayoutBuilder builder;
    const auto layout = plan_paged_kv_pool(builder, {1, 4096, 1, PagedKVPlaneOrder::PageMajor,
        {{DType::I8, 256, 4}, {DType::I8, 256, 4},
         {DType::FP16, 4, 4}, {DType::FP16, 4, 4}}});
    DeviceBuffer storage(builder.finish(256));
    PagedKVPool pool({storage.p, storage.bytes}, layout);
    auto host = plan_host_kv_archive(pool, 4, 262144);
    auto plan = plan_host_kv_staging(host, 2048);
    require(plan.maximum_layer_stream_bytes == 264ULL * 1024 * 1024, "real INT8 stream budget");
    require(plan.capacity_bytes == 328ULL * 1024 * 1024, "stage must add one 64 MiB tile");
    require(plan.ticket_capacity >= host.layers.size() + 18,
            "tickets cover current layer transfers plus queued layer writeback fences");
    rejects([&] { (void)plan_host_kv_staging(host, 2048, plan.capacity_bytes - 1); },
            "undersized stage override must fail");
}
void stronger_cleanup_cause(unsigned scenario) {
    LayoutBuilder builder;
    auto layout = plan_paged_kv_pool(builder, {1, 1, 1, PagedKVPlaneOrder::PageMajor, {{DType::I8, 16, 2}}});
    DeviceBuffer backing(builder.finish(256));
    PagedKVPool pool({backing.p, backing.bytes}, layout);
    HostKVArchive archive(plan_host_kv_archive(pool, 1, 64), HostArchiveMode::Pinned);
    auto plan = plan_host_kv_staging(archive.layout(), 1);
    DeviceBuffer staging(plan.capacity_bytes);
    HostKVTransferEngine engine(archive, {staging.p, staging.bytes}, plan);
    engine.synchronize(); // exercise and establish real healthy stream/worker drains
    int source = 1, destination = 0;
    const auto returned = cudaMemcpy(&destination, &source, sizeof(source), static_cast<cudaMemcpyKind>(99));
    require(returned == cudaErrorInvalidMemcpyDirection, "safe real returned status fixture");
    if (scenario == 0) {
        HostKVTransferTestAccess::poison(engine, std::make_exception_ptr(std::runtime_error("earlier generic")));
        HostKVTransferTestAccess::record_drain(engine, returned, "test returned failed drain");
    } else if (scenario == 1) {
        HostKVTransferTestAccess::poison(engine, std::make_exception_ptr(CudaTransferError(returned, "earlier nonfatal")));
        HostKVTransferTestAccess::poison(engine, std::make_exception_ptr(CudaTransferError(cudaErrorIllegalAddress, "first typed fatal")));
    } else {
        HostKVTransferTestAccess::poison(engine, std::make_exception_ptr(CudaTransferError(cudaErrorIllegalAddress, "first typed fatal")));
        HostKVTransferTestAccess::poison(engine, std::make_exception_ptr(std::runtime_error("later generic")));
        HostKVTransferTestAccess::record_drain(engine, returned, "later failed drain");
    }
    const auto verify = [&](auto&& operation) {
        bool typed = false;
        try { operation(); }
        catch (const CudaTransferError& error) {
            typed = error.status() == (scenario == 0 ? returned : cudaErrorIllegalAddress) &&
                error.drain_failed() == (scenario == 0) && std::string_view(error.operation()) ==
                    (scenario == 0 ? "test returned failed drain" : "first typed fatal");
        }
        require(typed, "first typed unrecoverable cause masked by generic/nonfatal/later failure");
    };
    verify([&] { engine.synchronize(); });
    (void)engine.quiesce(); // later actual successful CUDA drains cannot clear the sticky cause
    verify([&] { engine.synchronize(); });
    verify([&] { (void)engine.recover(); });
    std::cout << "stronger transfer cause PASS scenario=" << scenario << " returned_status=" << int(returned)
        << " simulated_drain_placement=1\n";
}
void maximum_fragmentation(HostArchiveMode mode) {
    LayoutBuilder builder;
    auto layout = plan_paged_kv_pool(builder, {1024, 1024, 1, PagedKVPlaneOrder::PageMajor,
                                              {{DType::I8, 16, 2}}});
    DeviceBuffer backing(builder.finish(256));
    PagedKVPool pool({backing.p, backing.bytes}, layout);
    DeviceContext device;
    CUDA_CHECK(cudaDeviceSynchronize());
    HostKVArchive archive(plan_host_kv_archive(pool, 1, 65536), mode);
    std::vector<std::int32_t> physical(1024);
    for (unsigned page = 0; page < 1024; ++page) {
        physical[page] = page * 17 % 1024;
        auto value = bytes(pool.page_bytes(0), 0, page);
        pool.copy_page_from_host(0, physical[page], value.data(), device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
    }
    archive.writeback(pool, 0, 0, physical, 65536, device.stream);
    archive.synchronize();
    auto plan = plan_host_kv_staging(archive.layout(), 512);
    DeviceBuffer staging(plan.capacity_bytes);
    HostKVTransferEngine engine(archive, {staging.p, staging.bytes}, plan);
    std::vector<std::uint32_t> ids;
    for (unsigned page = 1; page < 1024; page += 2) ids.push_back(page);
    require(ids.size() > plan.ticket_capacity * 20, "512 original fragments exceed fixed ticket capacity");
    PinnedHostBuffer actual(ids.size() * pool.page_bytes(0));
    HostKVTransferTicket stale;
    // No worker catch-up assumption: 128 packed gathers reuse the fixed stage
    // with consumer events and bounded ticket-reference backpressure.
    for (unsigned repeat = 0; repeat < 128; ++repeat) {
        auto ticket = engine.prefetch_gather_completed(0, 0, ids, 0);
        engine.wait(ticket, device.stream);
        CUDA_CHECK(cudaMemcpyAsync(actual.data(), engine.staged(ticket).data, actual.size(),
                                  cudaMemcpyDeviceToHost, device.stream));
        engine.release(ticket, device.stream);
        stale = ticket;
    }
    engine.synchronize();
    for (unsigned i = 0; i < ids.size(); ++i) {
        auto expected = bytes(pool.page_bytes(0), 0, ids[i]);
        require(std::memcmp(static_cast<std::byte*>(actual.data()) + i * expected.size(),
                            expected.data(), expected.size()) == 0,
                "512-fragment packed transfer preserves original page bytes");
    }
    rejects([&] { engine.wait(stale, device.stream); }, "packed ticket stale after bounded reuse");
}
void exercise(HostArchiveMode mode, PagedKVPlaneOrder order) {
    LayoutBuilder builder;
    PagedKVPoolSpec spec{48, 64, 1, order, {}};
    for (int layer = 0; layer < 2; ++layer) {
        spec.planes.insert(spec.planes.end(), {{DType::I8, 16, 2}, {DType::I8, 16, 2},
                                             {DType::FP16, 1, 2}, {DType::FP16, 1, 2}});
    }
    auto pool_layout = plan_paged_kv_pool(builder, spec);
    DeviceBuffer backing(builder.finish(256));
    PagedKVPool pool({backing.p, backing.bytes}, pool_layout);
    DeviceContext device;
    CUDA_CHECK(cudaDeviceSynchronize());
    HostKVArchive archive(plan_host_kv_archive(pool, 4, 4096), mode);
    std::vector<std::int32_t> source(40);
    for (std::size_t page = 0; page < 40; ++page) { source[page] = static_cast<int>(page * 7 % 48); }
    for (std::size_t p = 0; p < 8; ++p) {
        for (std::size_t page = 0; page < source.size(); ++page) {
            auto value = bytes(pool.page_bytes(p), static_cast<unsigned>(p), static_cast<unsigned>(page));
            pool.copy_page_from_host(p, source[page], value.data(), device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
        }
    }
    archive.writeback(pool, 0, 0, source, 2560, device.stream);
    archive.writeback(pool, 1, 0, source, 2560, device.stream);
    archive.synchronize();
    const auto plan = plan_host_kv_staging(archive.layout(), 44);
    DeviceBuffer staging(plan.capacity_bytes);
    PinnedHostBuffer check_a(pool.page_bytes(0)), check_b(pool.page_bytes(1));
    HostKVTransferEngine engine(archive, {staging.p, staging.bytes}, plan);
    require(engine.pinned_ring_bytes() == (mode == HostArchiveMode::Pageable ?
            4 * kHostKVTransferTileBytes : 0), "mode must allocate exactly the required ring");
    // Forty isolated logical fragments use one packed ticket per plane, independent
    // of source fragmentation. Repetition stresses the fixed consumed-event budget.
    std::vector<std::uint32_t> fragmented;
    for (unsigned p = 0; p < 40; p += 2) fragmented.push_back(p);
    PinnedHostBuffer gathered(fragmented.size() * pool.page_bytes(0));
    for (unsigned repeat = 0; repeat < 64; ++repeat) {
        auto ticket = engine.prefetch_gather_completed(0, 0, fragmented, 0);
        engine.wait(ticket, device.stream);
        CUDA_CHECK(cudaMemcpyAsync(gathered.data(), engine.staged(ticket).data, gathered.size(),
                                    cudaMemcpyDeviceToHost, device.stream));
        engine.release(ticket, device.stream);
        engine.synchronize();
        for (std::size_t p = 0; p < fragmented.size(); ++p) {
            auto expected = bytes(pool.page_bytes(0), 0, fragmented[p]);
            require(std::memcmp(static_cast<std::byte*>(gathered.data()) + p * expected.size(),
                                expected.data(), expected.size()) == 0, "fragmented gather byte oracle");
        }
        rejects([&] { engine.wait(ticket, device.stream); }, "released gather ticket is stale");
    }
    // All planes and a different layer are queued before the first consumer waits.
    std::vector<HostKVTransferTicket> first;
    std::size_t offset = 0;
    for (std::size_t p = 0; p < 4; ++p) {
        first.push_back(engine.prefetch(0, p, 20, 20, offset));
        offset += 20 * pool.page_bytes(p);
    }
    auto next_layer = engine.prefetch(1, 0, 0, 1, offset);
    rejects([&] { (void)engine.prefetch(1, 0, 0, 1, 0); },
            "cannot overwrite a live staging segment");
    for (std::size_t p = 0; p < first.size(); ++p) {
        engine.wait(first[p], device.stream);
        auto staged = engine.staged(first[p]);
        PinnedHostBuffer actual(staged.bytes);
        CUDA_CHECK(cudaMemcpyAsync(actual.data(), staged.data, staged.bytes,
                                    cudaMemcpyDeviceToHost, device.stream));
        engine.release(first[p], device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        for (std::size_t i = 0; i < 20; ++i) {
            auto expected = bytes(pool.page_bytes(p), static_cast<unsigned>(p), static_cast<unsigned>(20 + i));
            require(std::memcmp(static_cast<std::byte*>(actual.data()) + i * expected.size(),
                                expected.data(), expected.size()) == 0, "multi-plane staged bytes");
        }
    }
    engine.wait(next_layer, device.stream);
    CUDA_CHECK(cudaMemcpyAsync(check_b.data(), engine.staged(next_layer).data, check_b.size(),
                                cudaMemcpyDeviceToHost, device.stream));
    engine.release(next_layer, device.stream);
    engine.synchronize();
    const auto next_expected = bytes(check_b.size(), 4, 0);
    require(std::memcmp(check_b.data(), next_expected.data(), next_expected.size()) == 0,
            "cross-layer prefetch must preserve the next layer's own bytes");
    // Delay consumption, enqueue a conflicting transfer, and prove the old bytes survived.
    auto a = engine.prefetch(0, 0, 0, 1, 0);
    engine.wait(a, device.stream);
    CUDA_CHECK(cudaLaunchHostFunc(device.stream, [](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }, nullptr));
    CUDA_CHECK(cudaMemcpyAsync(check_a.data(), engine.staged(a).data, check_a.size(),
                                cudaMemcpyDeviceToHost, device.stream));
    engine.release(a, device.stream);
    auto b = engine.prefetch(0, 1, 3, 1, 0);
    engine.wait(b, device.load_stream);
    CUDA_CHECK(cudaMemcpyAsync(check_b.data(), engine.staged(b).data, check_b.size(),
                                cudaMemcpyDeviceToHost, device.load_stream));
    engine.release(b, device.load_stream);
    engine.synchronize();
    auto expected_a = bytes(check_a.size(), 0, 0), expected_b = bytes(check_b.size(), 1, 3);
    require(std::memcmp(check_a.data(), expected_a.data(), expected_a.size()) == 0,
            "release must protect old consumption before overwrite");
    require(std::memcmp(check_b.data(), expected_b.data(), expected_b.size()) == 0,
            "new segment must contain new plane bytes");
    // Enough consecutive copies to reuse every ring slot and event ticket repeatedly.
    for (unsigned repeat = 0; repeat < 48; ++repeat) {
        auto ticket = engine.prefetch(repeat % 2, repeat % 4, repeat % 40, 1, 0);
        engine.wait(ticket, device.stream);
        CUDA_CHECK(cudaMemcpyAsync(check_a.data(), engine.staged(ticket).data,
                                    pool.page_bytes(repeat % 4), cudaMemcpyDeviceToHost, device.stream));
        engine.release(ticket, device.stream);
        engine.synchronize();
        auto expected = bytes(pool.page_bytes(repeat % 4), (repeat % 2) * 4 + repeat % 4, repeat % 40);
        require(std::memcmp(check_a.data(), expected.data(), expected.size()) == 0, "ring reuse bytes");
    }
    const std::array<std::int32_t, 1> changed{0};
    for (std::size_t p = 0; p < 4; ++p) {
        auto value = bytes(pool.page_bytes(p), static_cast<unsigned>(p), 2, 1);
        pool.copy_page_from_host(p, 0, value.data(), device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
    }
    auto writeback = engine.writeback(pool, 0, 2, changed, 2560, device.stream);
    writeback.get();
    for (std::size_t p = 0; p < 4; ++p) {
        auto expected = bytes(pool.page_bytes(p), static_cast<unsigned>(p), 2, 1);
        auto host = archive.pages(0, p, 2, 1);
        require(std::equal(host.begin(), host.end(), expected.begin()), "ring/direct writeback bytes");
    }
    // Capture old archive bytes, then enqueue a delayed GPU producer and overwrite
    // the same host page. FIFO DMA must finish the old read before that writeback.
    auto old_read = engine.prefetch(0, 0, 2, 1, 0);
    engine.wait(old_read, device.load_stream);
    CUDA_CHECK(cudaMemcpyAsync(check_a.data(), engine.staged(old_read).data, check_a.size(),
                                cudaMemcpyDeviceToHost, device.load_stream));
    engine.release(old_read, device.load_stream);
    PinnedHostBuffer produced(pool.page_bytes(0));
    const auto replacement = bytes(produced.size(), 0, 2, 2);
    std::memcpy(produced.data(), replacement.data(), replacement.size());
    CUDA_CHECK(cudaLaunchHostFunc(device.stream, [](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }, nullptr));
    pool.copy_page_from_host(0, 0, produced.data(), device.stream);
    auto delayed_writeback = engine.writeback(pool, 0, 2, changed, 2560, device.stream);
    delayed_writeback.get();
    engine.synchronize();
    const auto old_value = bytes(check_a.size(), 0, 2, 1);
    require(std::memcmp(check_a.data(), old_value.data(), old_value.size()) == 0,
            "queued archive read must survive a later overwrite");
    const auto new_value = archive.pages(0, 0, 2, 1);
    require(std::equal(new_value.begin(), new_value.end(), replacement.begin()),
            "writeback must wait for the producing CUDA stream");
    // A delayed producer on layer 0 cannot block a subsequent layer-1 H2D.
    // Completion timestamps come from separate observers, not assumed ordering.
    engine.synchronize();
    const auto start = std::chrono::steady_clock::now();
    auto milliseconds = [&] {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    };
    CUDA_CHECK(cudaLaunchHostFunc(device.stream, [](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }, nullptr));
    auto independent_wb = engine.writeback(pool, 0, 2, changed, 2560, device.stream);
    std::atomic<double> writeback_ms{0};
    std::jthread observer([&] { independent_wb.wait(); writeback_ms.store(milliseconds()); });
    auto independent_pf = engine.prefetch(1, 0, 0, 1, 0);
    engine.wait(independent_pf, device.load_stream);
    const double prefetch_issued_ms = milliseconds();
    CUDA_CHECK(cudaMemcpyAsync(check_a.data(), engine.staged(independent_pf).data,
                                check_a.size(), cudaMemcpyDeviceToHost, device.load_stream));
    engine.release(independent_pf, device.load_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.load_stream));
    const double prefetch_ms = milliseconds();
    const bool writeback_was_pending = independent_wb.wait_for(std::chrono::milliseconds(0)) ==
                                      std::future_status::timeout;
    independent_wb.get(); observer.join();
    std::cout << "delayed-writeback mode=" << (mode == HostArchiveMode::Pinned ? "pinned" : "pageable")
              << " order=" << (order == PagedKVPlaneOrder::PageMajor ? "page" : "head")
              << " prefetch_issued_ms=" << prefetch_issued_ms
              << " prefetch_complete_ms=" << prefetch_ms
              << " writeback_complete_ms=" << writeback_ms.load() << '\n';
    require(writeback_was_pending && prefetch_ms < writeback_ms.load(),
            "delayed writeback must not hold up later unrelated prefetch");
    const auto independent_value = bytes(check_a.size(), 4, 0);
    require(std::memcmp(check_a.data(), independent_value.data(), independent_value.size()) == 0,
            "independent prefetch bytes during delayed writeback");
    engine.synchronize();

    // A tail writeback must not hold up completed history on the SAME layer.
    const auto same_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaLaunchHostFunc(device.stream, [](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }, nullptr));
    auto same_wb = engine.writeback(pool, 0, 2, changed, 2560, device.stream);
    rejects([&] { (void)engine.prefetch_completed(0, 0, 2, 1, 0); },
            "completed history must reject an overlapping pending writeback");
    rejects([&] { (void)engine.prefetch_completed(0, 0, 40, 1, 0); },
            "completed history must reject unarchived pages");
    auto same_pf = engine.prefetch_completed(0, 0, 0, 1, 0);
    engine.wait(same_pf, device.load_stream);
    CUDA_CHECK(cudaMemcpyAsync(check_a.data(), engine.staged(same_pf).data,
                                check_a.size(), cudaMemcpyDeviceToHost, device.load_stream));
    engine.release(same_pf, device.load_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.load_stream));
    const double same_pf_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - same_start).count();
    const bool same_pending = same_wb.wait_for(std::chrono::milliseconds(0)) ==
                              std::future_status::timeout;
    same_wb.get();
    const double same_wb_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - same_start).count();
    std::cout << "same-layer-prefetch mode=" << (mode == HostArchiveMode::Pinned ? "pinned" : "pageable")
              << " prefetch_complete_ms=" << same_pf_ms << " writeback_complete_ms=" << same_wb_ms << '\n';
    require(same_pending && same_pf_ms < same_wb_ms,
            "completed same-layer history prefetch must not wait for tail writeback");
    const auto same_value = bytes(check_a.size(), 0, 0);
    require(std::memcmp(check_a.data(), same_value.data(), same_value.size()) == 0,
            "same-layer completed history bytes");
    engine.synchronize();

    // Force an earlier archive H2D reader to remain queued behind a consumer.
    // The D2H worker must capture its ready event before overwriting that archive.
    auto held = engine.prefetch(1, 0, 0, 1, 0);
    engine.wait(held, device.load_stream);
    CUDA_CHECK(cudaLaunchHostFunc(device.load_stream, [](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }, nullptr));
    engine.release(held, device.load_stream);
    auto queued_old = engine.prefetch(0, 0, 2, 1, 0);
    engine.wait(queued_old, device.load_stream);
    CUDA_CHECK(cudaMemcpyAsync(check_a.data(), engine.staged(queued_old).data,
                                check_a.size(), cudaMemcpyDeviceToHost, device.load_stream));
    engine.release(queued_old, device.load_stream);
    // All startup tickets have been used by the reuse loop above. The next
    // disjoint segment therefore reuses a released ticket's event/metadata.
    auto reused_slot = engine.prefetch(1, 1, 0, 1, 4 * check_a.size());
    engine.wait(reused_slot, device.load_stream);
    engine.release(reused_slot, device.load_stream);
    const auto version3 = bytes(produced.size(), 0, 2, 3);
    std::memcpy(produced.data(), version3.data(), version3.size());
    pool.copy_page_from_host(0, 0, produced.data(), device.stream);
    engine.writeback(pool, 0, 2, changed, 2560, device.stream).get();
    engine.synchronize();
    require(std::memcmp(check_a.data(), replacement.data(), replacement.size()) == 0,
            "separate D2H must wait for prior archive H2D readers");
    const auto version3_host = archive.pages(0, 0, 2, 1);
    require(std::equal(version3_host.begin(), version3_host.end(), version3.begin()),
            "separate D2H must publish new archive bytes");
    auto stale = engine.prefetch(0, 0, 0, 1, 0);
    rejects([&] { archive.trim(128); }, "attached archive mutations must go through transfer owner");
    engine.trim(130);
    rejects([&] { engine.wait(stale, device.stream); }, "trim invalidates prior transfer tickets");
    rejects([&] { (void)engine.staged(stale); }, "trim invalidates staged access");
    rejects([&] { engine.release(stale, device.stream); }, "trim invalidates consumed-event release");
    require(archive.frontier(0) == 130 && archive.frontier(1) == 130, "engine trim frontier");
}

void actual_copy_failure_and_restart(HostArchiveMode mode, bool d2h, bool rejected_api = false,
                                     PagedKVPlaneOrder order = PagedKVPlaneOrder::PageMajor) {
    LayoutBuilder builder;
    auto layout = plan_paged_kv_pool(builder, {4, 16, 1, order,
        {{DType::I8,16,2}, {DType::I8,16,2}, {DType::I8,16,2}, {DType::I8,16,2}}});
    DeviceBuffer backing(builder.finish(256));
    PagedKVPool pool({backing.p, backing.bytes}, layout);
    DeviceContext device;
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::array<std::int32_t, 2> source{0,1};
    for (unsigned p = 0; p < 4; ++p) for (unsigned id = 0; id < 2; ++id) {
        auto value = bytes(pool.page_bytes(p), p, id);
        pool.copy_page_from_host(p, id, value.data(), device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
    }
    HostKVArchive archive(plan_host_kv_archive(pool,4,1024), mode);
    archive.writeback(pool,0,0,source,128,device.stream);
    archive.synchronize();
    auto plan = plan_host_kv_staging(archive.layout(),4);
    DeviceBuffer staging(plan.capacity_bytes);
    const auto stage_pointer = staging.p;
    HostKVTransferFaultInjection fault;
    if (rejected_api) {
        if (d2h) fault.reject_d2h_submission_after = 1;
        else fault.reject_h2d_submission_after = 2;
    } else if (d2h) fault.d2h_plane_after = 1;
    else fault.h2d_copy_after = 2;
    HostKVTransferEngine engine(archive,{staging.p,staging.bytes},plan,fault);
    const auto ring_bytes = engine.pinned_ring_bytes();
    auto old = engine.prefetch(0,0,0,1,0);
    engine.wait(old,device.load_stream);
    std::atomic<bool> consumer_done{false};
    CUDA_CHECK(cudaLaunchHostFunc(device.load_stream, [](void* pointer) {
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        static_cast<std::atomic<bool>*>(pointer)->store(true);
    }, &consumer_done));
    engine.release(old,device.load_stream);
    // The injected H2D copy is a real overwrite queued behind the old consumer.
    auto overwrite = engine.prefetch(0,1,1,1,0);
    auto queued = engine.prefetch(0,2,0,1,2*pool.page_bytes(0));
    CUDA_CHECK(cudaLaunchHostFunc(device.stream, [](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
    }, nullptr));
    // Change all planes: D2H failure must not trust an earlier committed prefix.
    PinnedHostBuffer produced(4*pool.page_bytes(0));
    for (unsigned p=0;p<4;++p) {
        auto value = bytes(pool.page_bytes(p),p,0,1);
        std::memcpy(static_cast<std::byte*>(produced.data())+p*value.size(),value.data(),value.size());
        pool.copy_page_from_host(p,0,static_cast<std::byte*>(produced.data())+p*value.size(),device.stream);
    }
    const std::array<std::int32_t,1> changed{0};
    auto future = engine.writeback(pool,0,0,changed,128,device.stream);
    if (rejected_api) {
        bool typed = false;
        try { engine.synchronize(); }
        catch (const CudaTransferError& error) {
            typed = error.status() == cudaErrorInvalidMemcpyDirection && error.operation();
        }
        require(typed, "real CUDA rejection must propagate its typed status and operation");
    } else rejects([&] { engine.synchronize(); }, "actual copied plane must poison transfer engine");
    const auto failed = engine.quiesce();
    require(!failed.unrecoverable, "safe rejected API submission must retain recoverable resources");
    if (rejected_api) {
        require(failed.cuda_status == cudaErrorInvalidMemcpyDirection && failed.cuda_operation &&
                failed.drain_status == cudaSuccess,
                "sticky CUDA failure retains rejection and successful drain evidence");
    }
    require(failed.failed && consumer_done.load(), "failure quiesce drains real consumer work");
    require(d2h ? (failed.archive_bytes_uncertain && failed.completed_d2h_planes >= 1)
                : (!failed.archive_bytes_uncertain && failed.completed_h2d_copies >= 2),
            "fault must occur after an actual completed DMA plane");
    require(future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready,
            "quiesce must retire queued writeback futures");
    rejects([&] { future.get(); }, "failed or canceled writeback remains explicitly exceptional");
    if (!d2h) {
        const auto overwritten = bytes(pool.page_bytes(1),1,1);
        CUDA_CHECK(cudaMemcpyAsync(produced.data(),staging.p,overwritten.size(),
                                   cudaMemcpyDeviceToHost,device.load_stream));
        CUDA_CHECK(cudaStreamSynchronize(device.load_stream));
        require(std::memcmp(produced.data(),overwritten.data(),overwritten.size())==0,
                "H2D fault must leave independently verified real overwritten bytes");
    }
    const auto repaired = engine.recover();
    require(repaired.archive_bytes_uncertain == d2h && !engine.failure_state().failed,
            "restart reports whether archive was invalidated and clears transfer poison");
    require(staging.p == stage_pointer && engine.pinned_ring_bytes() == ring_bytes,
            "recovery reuses loading-time workspace and ring budget");
    rejects([&] { engine.wait(old,device.load_stream); }, "old ready ticket invalid after recovery");
    rejects([&] { (void)engine.staged(overwrite); }, "old overwrite ticket invalid after recovery");
    rejects([&] { engine.release(queued,device.load_stream); }, "queued ticket invalid after recovery");
    if (d2h) {
        require(archive.frontier(0)==0, "uncertain D2H archive cannot retain committed frontier");
        rejects([&] { (void)archive.pages(0,0,0,1); }, "uncertain host bytes must be unreadable");
        engine.writeback(pool,0,0,source,128,device.stream).get();
    } else require(archive.frontier(0)==128, "H2D failure preserves authoritative archive frontier");
    for (unsigned p=0;p<4;++p) {
        auto expected = bytes(pool.page_bytes(p),p,0,d2h?1:0);
        auto host=archive.pages(0,p,0,1);
        require(std::equal(host.begin(),host.end(),expected.begin()), "repaired archive independent byte oracle");
        auto ticket=engine.prefetch(0,p,0,1,0);
        engine.wait(ticket,device.load_stream);
        CUDA_CHECK(cudaMemcpyAsync(produced.data(),engine.staged(ticket).data,expected.size(),
                                   cudaMemcpyDeviceToHost,device.load_stream));
        engine.release(ticket,device.load_stream);
        engine.synchronize();
        require(std::memcmp(produced.data(),expected.data(),expected.size())==0,
                "restarted workers must transfer correct plane bytes");
    }
    engine.reset();
}

void cuda_failure_classification() {
    for (auto status : {cudaErrorIllegalAddress, cudaErrorAssert, cudaErrorLaunchTimeout,
                       cudaErrorLaunchFailure, cudaErrorContextIsDestroyed,
                       cudaErrorIllegalInstruction, cudaErrorMisalignedAddress,
                       cudaErrorInvalidAddressSpace, cudaErrorInvalidPc, cudaErrorUnknown})
        require(cuda_failure_is_fatal(status), "execution/context/unknown CUDA status must fail closed");
    for (auto status : {cudaErrorInvalidValue, cudaErrorInvalidMemcpyDirection,
                       cudaErrorInvalidResourceHandle, cudaErrorLaunchOutOfResources}) {
        require(!cuda_failure_is_fatal(status), "known API rejection requires successful drains to recover");
        require(!cuda_transfer_failure_is_unrecoverable(status,false) &&
                cuda_transfer_failure_is_unrecoverable(status,true),
                "any failed drain permanently forbids recovery, including nonfatal status");
    }
    require(cuda_transfer_failure_is_unrecoverable(cudaErrorIllegalAddress,false),
            "fatal execution status forbids recovery even when subsequent drains succeed");
}

void public_cuda_rejection_drains_and_recovers(HostArchiveMode mode, unsigned operation) {
    LayoutBuilder builder;
    auto layout=plan_paged_kv_pool(builder,{2,4,1,PagedKVPlaneOrder::PageMajor,{{DType::I8,16,2}}});
    DeviceBuffer backing(builder.finish(256));
    PagedKVPool pool({backing.p,backing.bytes},layout);
    DeviceContext device;
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto expected=bytes(pool.page_bytes(0),0,0);
    pool.copy_page_from_host(0,0,expected.data(),device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    HostKVArchive archive(plan_host_kv_archive(pool,1,256),mode);
    const std::array<std::int32_t,1> source{0};
    archive.writeback(pool,0,0,source,64,device.stream);
    archive.synchronize();
    const auto plan=plan_host_kv_staging(archive.layout(),2);
    DeviceBuffer staging(plan.capacity_bytes);
    PinnedHostBuffer observed(expected.size());
    HostKVTransferFaultInjection fault;
    fault.reject_consumer_wait=operation==0;
    fault.reject_consumer_release=operation==1;
    fault.reject_producer_record=operation==2;
    HostKVTransferEngine engine(archive,{staging.p,staging.bytes},plan,fault);
    const auto old=engine.prefetch(0,0,0,1,0);
    std::atomic<bool> borrowed_done{false};
    auto caller=operation==2 ? device.stream : device.load_stream;
    if (operation==1) engine.wait(old,caller);
    CUDA_CHECK(cudaLaunchHostFunc(caller,[](void* flag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        static_cast<std::atomic<bool>*>(flag)->store(true);
    },&borrowed_done));
    std::shared_future<void> canceled;
    std::atomic<bool> producer_done{false};
    if (operation!=2) {
        CUDA_CHECK(cudaLaunchHostFunc(device.stream,[](void* flag) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            static_cast<std::atomic<bool>*>(flag)->store(true);
        },&producer_done));
        canceled=engine.writeback(pool,0,0,source,64,device.stream);
    }
    bool typed=false;
    try {
        if (operation==0) engine.wait(old,caller);
        else if (operation==1) engine.release(old,caller);
        else (void)engine.writeback(pool,0,0,source,64,caller);
    } catch (const CudaTransferError& error) {
        typed=error.status()==cudaErrorInvalidValue && error.operation();
    }
    const bool canceled_retired_without_quiesce=!canceled.valid() ||
        canceled.wait_for(std::chrono::seconds(2))==std::future_status::ready;
    const bool producer_drained_before_quiesce=operation==2 ? borrowed_done.load() : producer_done.load();
    const auto failed=engine.quiesce();
    const bool borrowed_drained=borrowed_done.load();
    const auto caller_status=cudaStreamQuery(caller);
    // Drain before assertions even if a regression drops a dependency.
    CUDA_CHECK(cudaStreamSynchronize(caller));
    require(typed && failed.failed && failed.cuda_status==cudaErrorInvalidValue,
            "public CUDA rejection must poison owner and retain typed status");
    require(canceled_retired_without_quiesce && producer_drained_before_quiesce,
            "exceptional future and rejected producer submission retain source until producer drains");
    require(borrowed_drained && caller_status==cudaSuccess && !failed.unrecoverable,
            "failed public event submission must retain and drain caller stream");
    if (canceled.valid()) {
        require(canceled.wait_for(std::chrono::milliseconds(0))==std::future_status::ready,
                "public poison retires queued or active writeback promise exactly once");
        rejects([&] { canceled.get(); },"public poison has exceptional queued completion");
    }
    rejects([&] { (void)engine.prefetch(0,0,0,1,0); },"public poison rejects new work");
    const auto ring_bytes=engine.pinned_ring_bytes();
    (void)engine.recover();
    require(!engine.failure_state().failed && archive.frontier(0)==64 &&
            engine.pinned_ring_bytes()==ring_bytes,
            "public rejection recovery preserves archive authority and resource count");
    rejects([&] { engine.wait(old,caller); },"public rejection recovery invalidates old tickets");
    const auto current=engine.prefetch(0,0,0,1,0);
    engine.wait(current,device.load_stream);
    CUDA_CHECK(cudaMemcpyAsync(observed.data(),engine.staged(current).data,expected.size(),
                               cudaMemcpyDeviceToHost,device.load_stream));
    engine.release(current,device.load_stream);
    engine.synchronize();
    require(std::memcmp(observed.data(),expected.data(),expected.size())==0,
            "public CUDA rejection recovery transfers independent byte oracle");
    cudaStream_t temporary_producer=nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&temporary_producer,cudaStreamNonBlocking));
    engine.writeback(pool,0,0,source,64,temporary_producer).get();
    CUDA_CHECK(cudaStreamDestroy(temporary_producer));
    engine.synchronize();
    require(!engine.failure_state().failed,"completed writeback releases producer stream lifetime");
}

void released_ticket_borrowers_survive_canceled_slot_reuse(HostArchiveMode mode) {
    LayoutBuilder builder;
    auto layout=plan_paged_kv_pool(builder,{2,4,1,PagedKVPlaneOrder::PageMajor,
        {{DType::I8,16,2},{DType::I8,16,2}}});
    DeviceBuffer backing(builder.finish(256));
    PagedKVPool pool({backing.p,backing.bytes},layout);
    DeviceContext device;
    CUDA_CHECK(cudaDeviceSynchronize());
    for(unsigned plane=0;plane<2;++plane) {
        auto value=bytes(pool.page_bytes(plane),plane,0);
        pool.copy_page_from_host(plane,0,value.data(),device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
    }
    HostKVArchive archive(plan_host_kv_archive(pool,2,256),mode);
    const std::array<std::int32_t,1> source{0};
    archive.writeback(pool,0,0,source,64,device.stream);
    archive.synchronize();
    auto plan=plan_host_kv_staging(archive.layout(),2);
    DeviceBuffer staging(plan.capacity_bytes);
    PinnedHostBuffer observed(pool.page_bytes(1));
    HostKVTransferEngine engine(archive,{staging.p,staging.bytes},plan,{3,0});
    auto a=engine.prefetch(0,0,0,1,0);
    auto b=engine.prefetch(0,1,0,1,2*pool.page_bytes(0));
    engine.wait(a,device.stream);
    engine.wait(b,device.load_stream);
    const auto borrowed=engine.staged(b);
    CUDA_CHECK(cudaLaunchHostFunc(device.stream,[](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
    },nullptr));
    engine.release(a,device.stream);
    std::atomic<bool> b_done{false};
    CUDA_CHECK(cudaLaunchHostFunc(device.load_stream,[](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(600));
    },nullptr));
    CUDA_CHECK(cudaLaunchHostFunc(device.load_stream,[](void* flag) {
        static_cast<std::atomic<bool>*>(flag)->store(true);
    },&b_done));
    engine.release(b,device.load_stream);
    require(!b_done.load(), "long borrower must still be pending before replacement jobs enqueue");
    auto fault=engine.prefetch(0,1,0,1,0);
    auto canceled=engine.prefetch(0,0,0,1,2*pool.page_bytes(0));
    require(fault.slot==a.slot && canceled.slot==b.slot,
            "reproducer must replace both released ticket metadata slots");
    const auto failed=engine.quiesce();
    const bool drained_before_return=b_done.load();
    const auto completion_before_return=cudaStreamQuery(device.load_stream);
    // Always drain before assertions so the RED case cannot free borrowed bytes.
    CUDA_CHECK(cudaStreamSynchronize(device.load_stream));
    require(failed.failed && failed.completed_h2d_copies==3,
            "reproducer must inject after faulting real third copy");
    require(drained_before_return && completion_before_return==cudaSuccess,
            "quiesce lost a consumed event when queued released-slot reuse was canceled");
    const auto expected=bytes(observed.size(),1,0);
    CUDA_CHECK(cudaMemcpyAsync(observed.data(),borrowed.data,borrowed.bytes,
                               cudaMemcpyDeviceToHost,device.load_stream));
    CUDA_CHECK(cudaStreamSynchronize(device.load_stream));
    require(std::memcmp(observed.data(),expected.data(),expected.size())==0,
            "canceled slot reuse must preserve independent original consumer byte oracle");
    (void)engine.recover();
    rejects([&] { engine.wait(canceled,device.load_stream); },
            "canceled reuse ticket remains invalid after recovery");
    auto previous=engine.prefetch(0,0,0,1,0);
    engine.wait(previous,device.stream);
    engine.release(previous,device.stream);
    engine.synchronize();
    auto current=engine.prefetch(0,1,0,1,0);
    require(current.slot==previous.slot,"healthy regression must reuse the released event slot");
    engine.wait(current,device.load_stream);
    const auto current_bytes=engine.staged(current);
    CUDA_CHECK(cudaMemcpyAsync(observed.data(),current_bytes.data,current_bytes.bytes,
                               cudaMemcpyDeviceToHost,device.load_stream));
    b_done.store(false);
    CUDA_CHECK(cudaLaunchHostFunc(device.load_stream,[](void* flag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
        static_cast<std::atomic<bool>*>(flag)->store(true);
    },&b_done));
    // Current ticket intentionally has no release event. Its earlier generation's
    // recorded consumed event cannot establish completion of this new borrower.
    const auto healthy=engine.quiesce();
    const bool current_drained=b_done.load();
    const auto current_completion=cudaStreamQuery(device.load_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.load_stream));
    require(!healthy.failed && current_drained && current_completion==cudaSuccess,
            "old consumed event must not hide the current unreleased consumer stream");
    require(std::memcmp(observed.data(),expected.data(),expected.size())==0,
            "current unreleased consumer retains independent plane byte oracle");
    engine.release(current,device.load_stream);
    engine.synchronize();
}

void second_worker_restart_failure_is_retryable(HostArchiveMode mode) {
    LayoutBuilder builder;
    auto layout=plan_paged_kv_pool(builder,{2,4,1,PagedKVPlaneOrder::PageMajor,{{DType::I8,16,2}}});
    DeviceBuffer backing(builder.finish(256));
    PagedKVPool pool({backing.p,backing.bytes},layout);
    DeviceContext device;
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto expected=bytes(pool.page_bytes(0),0,0);
    pool.copy_page_from_host(0,0,expected.data(),device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    HostKVArchive archive(plan_host_kv_archive(pool,1,256),mode);
    const std::array<std::int32_t,1> source{0};
    archive.writeback(pool,0,0,source,64,device.stream);
    archive.synchronize();
    auto plan=plan_host_kv_staging(archive.layout(),2);
    DeviceBuffer staging(plan.capacity_bytes);
    PinnedHostBuffer observed(pool.page_bytes(0));
    HostKVTransferFaultInjection fault;
    fault.h2d_copy_after=1;
    fault.second_worker_start_failure_attempt=2;
    HostKVTransferEngine engine(archive,{staging.p,staging.bytes},plan,fault);
    const auto ring_bytes=engine.pinned_ring_bytes();
    auto old=engine.prefetch(0,0,0,1,0);
    require(engine.quiesce().failed,"startup regression begins from actual DMA failure");
    bool restart_failed=false;
    try { (void)engine.recover(); }
    catch(const std::system_error& error) {
        restart_failed=error.code()==std::make_error_code(std::errc::resource_unavailable_try_again);
    }
    require(restart_failed,"second worker creation must inject the intended resource failure");
    require(engine.failure_state().failed,"partial restart must never publish healthy failure state");
    rejects([&] { (void)engine.prefetch(0,0,0,1,0); },"partial restart rejects job submission");
    require(engine.quiesce().failed,"partial startup workers must be joined and quiesce retryable");
    const auto previous=engine.recover();
    require(previous.failed && !engine.failure_state().failed,
            "retry after partial startup must restart both workers without joinable reassignment");
    require(engine.pinned_ring_bytes()==ring_bytes,"startup retry retains fixed pinned ring");
    rejects([&] { engine.wait(old,device.stream); },"startup retry invalidates earlier tickets");
    auto current=engine.prefetch(0,0,0,1,0);
    engine.wait(current,device.stream);
    CUDA_CHECK(cudaMemcpyAsync(observed.data(),engine.staged(current).data,observed.size(),
                               cudaMemcpyDeviceToHost,device.stream));
    engine.release(current,device.stream);
    engine.synchronize();
    require(std::memcmp(observed.data(),expected.data(),expected.size())==0,
            "both restarted workers preserve independent archived byte oracle");
    // Exercise the second worker as well as H2D after the retry.
    engine.writeback(pool,0,0,source,64,device.stream).get();
    engine.synchronize();
}

void constructor_flags_allocation_failure_unwinds(HostArchiveMode mode) {
    LayoutBuilder builder;
    auto layout=plan_paged_kv_pool(builder,{2,4,1,PagedKVPlaneOrder::PageMajor,{{DType::I8,16,2}}});
    DeviceBuffer backing(builder.finish(256));
    PagedKVPool pool({backing.p,backing.bytes},layout);
    CUDA_CHECK(cudaDeviceSynchronize());
    HostKVArchive archive(plan_host_kv_archive(pool,1,256),mode);
    auto plan=plan_host_kv_staging(archive.layout(),2);
    DeviceBuffer staging(plan.capacity_bytes);
    for(const bool producer_flags : {false,true}) {
        HostKVTransferFaultInjection fault;
        fault.fail_consumed_flags_allocation=!producer_flags;
        fault.fail_producer_borrowed_flags_allocation=producer_flags;
        bool allocation_failed=false;
        try { HostKVTransferEngine engine(archive,{staging.p,staging.bytes},plan,fault); }
        catch(const std::bad_alloc&) { allocation_failed=true; }
        require(allocation_failed,producer_flags ?
            "injected producer-borrowed flags allocation must unwind as an ordinary bad_alloc" :
            "injected consumed flags allocation must unwind as an ordinary bad_alloc");
        // Failed construction must detach the archive owner and destroy only valid
        // initialized resources, so reconstructing on the same archive is legal.
        HostKVTransferEngine retry(archive,{staging.p,staging.bytes},plan);
        retry.synchronize();
    }
}

void delayed_writeback_fences_do_not_pin_ticket_generations() {
    LayoutBuilder builder;
    PagedKVPoolSpec spec{2,64,1,PagedKVPlaneOrder::PageMajor,{}};
    for(unsigned layer=0;layer<16;++layer)
        spec.planes.insert(spec.planes.end(),4,{DType::I8,16,2});
    auto layout=plan_paged_kv_pool(builder,spec);
    DeviceBuffer backing(builder.finish(256));
    PagedKVPool pool({backing.p,backing.bytes},layout);
    DeviceContext device;
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::array<std::int32_t,1> source{0};
    for(unsigned plane=0;plane<64;++plane) {
        auto value=bytes(pool.page_bytes(plane),plane,0);
        pool.copy_page_from_host(plane,0,value.data(),device.load_stream);
        CUDA_CHECK(cudaStreamSynchronize(device.load_stream));
    }
    HostKVArchive archive(plan_host_kv_archive(pool,4,4096),HostArchiveMode::Pinned);
    for(unsigned layer=0;layer<16;++layer)
        archive.writeback(pool,layer,0,source,64,device.load_stream);
    archive.synchronize();
    auto plan=plan_host_kv_staging(archive.layout(),4);
    require(plan.ticket_capacity==30,"ticket regression must exercise actual bounded capacity30");
    DeviceBuffer staging(plan.capacity_bytes);
    PinnedHostBuffer observed(pool.page_bytes(0));
    std::atomic<bool> paused{false},resume{false};
    HostKVTransferFaultInjection hook;
    hook.pause_before_h2d_copy=46;
    hook.h2d_paused=&paused;
    hook.resume_h2d=&resume;
    HostKVTransferEngine engine(archive,{staging.p,staging.bytes},plan,hook);
    struct ResumeWorker {
        std::atomic<bool>& signal;
        ~ResumeWorker() { signal.store(true); }
    } unblock{resume};
    // Populate every slot once, then release it. This establishes all event
    // generations without relying on CPU/DMA scheduling to consume free slots.
    std::vector<HostKVTransferTicket> warm;
    for(unsigned n=0;n<30;++n)
        warm.push_back(engine.prefetch(15,0,0,1,n*pool.page_bytes(0)));
    for(auto ticket:warm) {
        engine.wait(ticket,device.load_stream);
        engine.release(ticket,device.load_stream);
    }
    engine.synchronize();
    CUDA_CHECK(cudaLaunchHostFunc(device.stream,[](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    },nullptr));
    std::vector<std::shared_future<void>> writes;
    for(unsigned layer=0;layer<15;++layer) {
        auto read=engine.prefetch(15,0,0,1,layer*pool.page_bytes(0));
        engine.wait(read,device.load_stream);
        engine.release(read,device.load_stream);
        writes.push_back(engine.writeback(pool,layer,0,source,64,device.stream));
    }
    // The old implementation holds15ready-event references behind the delayed
    // producer. Fifteen queued replacements hold the other15consumed references.
    // The independent FIFO archive fence must let the next prefetch submit.
    std::vector<HostKVTransferTicket> queued;
    for(unsigned n=0;n<15;++n)
        queued.push_back(engine.prefetch(15,0,0,1,(30+n)*pool.page_bytes(0)));
    while(!paused.load()) std::this_thread::yield();
    bool accepted=true;
    HostKVTransferTicket extra;
    try { extra=engine.prefetch(15,0,0,1,60*pool.page_bytes(0)); }
    catch(const std::runtime_error&) { accepted=false; }
    resume.store(true);
    engine.synchronize();
    require(accepted,"delayed writeback must not exhaust released ticket generations at capacity30");
    for(auto& write:writes) write.get();
    engine.wait(extra,device.load_stream);
    CUDA_CHECK(cudaMemcpyAsync(observed.data(),engine.staged(extra).data,observed.size(),
                               cudaMemcpyDeviceToHost,device.load_stream));
    engine.release(extra,device.load_stream);
    engine.synchronize();
    const auto expected=bytes(observed.size(),60,0);
    require(std::memcmp(observed.data(),expected.data(),expected.size())==0,
            "independent layer bytes survive delayed fence/ticket pressure");
}

void canceled_archive_fence_recovers(HostArchiveMode mode) {
    LayoutBuilder builder;
    auto layout=plan_paged_kv_pool(builder,{2,4,1,PagedKVPlaneOrder::PageMajor,
        {{DType::I8,16,2},{DType::I8,16,2}}});
    DeviceBuffer backing(builder.finish(256));
    PagedKVPool pool({backing.p,backing.bytes},layout);
    DeviceContext device;
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::array<std::int32_t,1> source{0};
    for(unsigned p=0;p<2;++p) {
        const auto value=bytes(pool.page_bytes(p),p,0);
        pool.copy_page_from_host(p,0,value.data(),device.load_stream);
        CUDA_CHECK(cudaStreamSynchronize(device.load_stream));
    }
    HostKVArchive archive(plan_host_kv_archive(pool,2,256),mode);
    archive.writeback(pool,0,0,source,64,device.load_stream);
    archive.synchronize();
    const auto plan=plan_host_kv_staging(archive.layout(),2);
    DeviceBuffer staging(plan.capacity_bytes);
    PinnedHostBuffer observed(pool.page_bytes(0));
    std::atomic<bool> paused{false},resume{false},producer_done{false};
    HostKVTransferFaultInjection hook;
    hook.h2d_copy_after=hook.pause_before_h2d_copy=1;
    hook.h2d_paused=&paused;
    hook.resume_h2d=&resume;
    HostKVTransferEngine engine(archive,{staging.p,staging.bytes},plan,hook);
    struct ResumeWorker {
        std::atomic<bool>& signal;
        ~ResumeWorker() { signal.store(true); }
    } unblock{resume};
    const auto old=engine.prefetch(0,0,0,1,0);
    while(!paused.load()) std::this_thread::yield();
    CUDA_CHECK(cudaLaunchHostFunc(device.stream,[](void* pointer) {
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        static_cast<std::atomic<bool>*>(pointer)->store(true);
    },&producer_done));
    const auto write=engine.writeback(pool,0,0,source,64,device.stream);
    // The H2D worker cannot record this writeback's fence before the actual
    // injected first copy fails. D2H must observe poison, not an older event.
    resume.store(true);
    const auto failed=engine.quiesce();
    require(failed.failed && !failed.archive_bytes_uncertain && producer_done.load(),
            "canceled archive fence drains producer without trusting unrecorded generation");
    require(write.wait_for(std::chrono::milliseconds(0))==std::future_status::ready,
            "canceled archive fence retires writeback completion");
    rejects([&] { write.get(); },"canceled fence has exceptional completion");
    engine.recover();
    rejects([&] { engine.wait(old,device.load_stream); },"canceled fence recovery invalidates old ticket");
    const auto read=engine.prefetch(0,0,0,1,0);
    engine.wait(read,device.load_stream);
    CUDA_CHECK(cudaMemcpyAsync(observed.data(),engine.staged(read).data,observed.size(),
                               cudaMemcpyDeviceToHost,device.load_stream));
    engine.release(read,device.load_stream);
    engine.synchronize();
    const auto expected=bytes(observed.size(),0,0);
    require(std::memcmp(observed.data(),expected.data(),expected.size())==0,
            "canceled fence recovery preserves authoritative archive bytes");
    engine.writeback(pool,0,0,source,64,device.stream).get();
    engine.synchronize();
}
}
int main(int argc,char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { return 77; }
    try {
        cuda_failure_classification();
        if (argc == 2 && std::strcmp(argv[1], "--stronger-nonfatal-only") == 0) {
            stronger_cleanup_cause(1);
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--stronger-cause-only") == 0) {
            for (unsigned scenario = 0; scenario < 3; ++scenario) stronger_cleanup_cause(scenario);
            return 0;
        }
        for (unsigned scenario = 0; scenario < 3; ++scenario) stronger_cleanup_cause(scenario);
        if(argc==2 && std::strcmp(argv[1],"--public-cuda-rejection-only")==0) {
            for(auto mode : {HostArchiveMode::Pinned,HostArchiveMode::Pageable})
                for(unsigned operation=0;operation<3;++operation)
                    public_cuda_rejection_drains_and_recovers(mode,operation);
            return 0;
        }
        if(argc==2 && std::strcmp(argv[1],"--cuda-rejection-only")==0) {
            for(auto mode : {HostArchiveMode::Pinned,HostArchiveMode::Pageable}) {
                actual_copy_failure_and_restart(mode,false,true);
                actual_copy_failure_and_restart(mode,true,true);
                actual_copy_failure_and_restart(mode,false,true,PagedKVPlaneOrder::HeadMajor);
                actual_copy_failure_and_restart(mode,true,true,PagedKVPlaneOrder::HeadMajor);
            }
            return 0;
        }
        if(argc==2 && std::strcmp(argv[1],"--ticket-generation-only")==0) {
            delayed_writeback_fences_do_not_pin_ticket_generations();
            return 0;
        }
        if(argc==2 && std::strcmp(argv[1],"--startup-recovery-only")==0) {
            for(auto mode : {HostArchiveMode::Pinned,HostArchiveMode::Pageable})
                second_worker_restart_failure_is_retryable(mode);
            return 0;
        }
        if(argc==2 && std::strcmp(argv[1],"--constructor-unwind-only")==0) {
            for(auto mode : {HostArchiveMode::Pinned,HostArchiveMode::Pageable})
                constructor_flags_allocation_failure_unwinds(mode);
            return 0;
        }
        for(auto mode : {HostArchiveMode::Pinned,HostArchiveMode::Pageable}) {
            for(unsigned operation=0;operation<3;++operation)
                public_cuda_rejection_drains_and_recovers(mode,operation);
            constructor_flags_allocation_failure_unwinds(mode);
            second_worker_restart_failure_is_retryable(mode);
            canceled_archive_fence_recovers(mode);
        }
        delayed_writeback_fences_do_not_pin_ticket_generations();
        for(auto mode : {HostArchiveMode::Pinned,HostArchiveMode::Pageable})
            released_ticket_borrowers_survive_canceled_slot_reuse(mode);
        staging_plan_contract();
        maximum_fragmentation(HostArchiveMode::Pinned);
        maximum_fragmentation(HostArchiveMode::Pageable);
        for (auto mode : {HostArchiveMode::Pinned, HostArchiveMode::Pageable}) {
            exercise(mode, PagedKVPlaneOrder::PageMajor);
            exercise(mode, PagedKVPlaneOrder::HeadMajor);
            actual_copy_failure_and_restart(mode,false);
            actual_copy_failure_and_restart(mode,true);
            actual_copy_failure_and_restart(mode,false,true);
            actual_copy_failure_and_restart(mode,true,true);
            actual_copy_failure_and_restart(mode,false,true,PagedKVPlaneOrder::HeadMajor);
            actual_copy_failure_and_restart(mode,true,true,PagedKVPlaneOrder::HeadMajor);
        }
        std::cout << "PASS host KV transfer ring/direct, cross-layer prefetch, events and trim\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL host KV transfer engine: " << error.what() << '\n';
        return 1;
    }
}

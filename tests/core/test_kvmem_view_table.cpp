#include "core/kvmem/kv_view_table.h"

#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

using namespace ninfer::kvmem;

namespace {
void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

template <class F>
void rejects(F&& operation, const char* message) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::exception&) { rejected = true; }
    require(rejected, message);
}

void complete(KVViewTable& table, const KVViewWrite& write, std::uint32_t layers = 16) {
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        require(table.complete_writeback(layer, write.logical_page, write.epoch),
                "each layer completion must be accepted exactly once");
    }
}

void equal_snapshot(const KVViewSnapshot& a, const KVViewSnapshot& b) {
    require(a.frontier == b.frontier && a.generation == b.generation &&
                a.blocktable == b.blocktable && a.host_only == b.host_only &&
                a.resident.size() == b.resident.size(),
            "rejected operation changed snapshot");
    for (std::size_t i = 0; i < a.resident.size(); ++i) {
        require(a.resident[i] == b.resident[i], "rejected operation changed resident descriptor");
    }
}

void validate_snapshot(const KVViewSnapshot& snapshot, std::uint32_t slots) {
    require(snapshot.blocktable.size() == (snapshot.frontier + 63ULL) / 64,
            "blocktable must cover every readable page at exact frontier");
    std::set<std::uint32_t> logical;
    std::set<std::int32_t> physical;
    std::uint32_t previous = 0;
    bool first             = true;
    for (const auto& page : snapshot.resident) {
        require(first || page.logical_page > previous, "resident list must be ordered");
        first    = false;
        previous = page.logical_page;
        require(page.physical_slot >= 0 && static_cast<std::uint32_t>(page.physical_slot) < slots,
                "physical slot is outside the configured view");
        require(physical.insert(page.physical_slot).second, "two logical pages share a slot");
        require(logical.insert(page.logical_page).second &&
                    snapshot.blocktable.at(page.logical_page) == page.physical_slot,
                "resident list disagrees with blocktable");
    }
    require(std::is_sorted(snapshot.host_only.begin(), snapshot.host_only.end()),
            "host-only list must be ordered");
    for (auto page : snapshot.host_only) {
        require(logical.insert(page).second && snapshot.blocktable.at(page) == -1,
                "host-only descriptor disagrees with blocktable");
    }
    require(logical.size() == snapshot.blocktable.size(), "snapshot has a missing readable page");
}

void all_layers_gate_eviction() {
    KVViewTable table(512, 2, 0);
    const auto writes = table.begin_append(0, 128);
    require(writes.size() == 2, "append must return each written page");
    for (std::uint32_t layer = 0; layer < 15; ++layer) {
        require(table.complete_writeback(layer, 0, writes[0].epoch), "pending completion");
    }
    require(table.page(0)->state == KVViewState::DeviceOnly, "15 layers cannot publish Both");
    const auto before = table.snapshot();
    rejects([&] { (void)table.plan_restore(128); }, "restore cannot publish unarchived DeviceOnly pages");
    rejects([&] { (void)table.begin_append(128, 64); }, "DeviceOnly page must not be evicted");
    equal_snapshot(before, table.snapshot());
    require(table.complete_writeback(15, 0, writes[0].epoch), "last layer completion");
    require(table.page(0)->state == KVViewState::Both, "all layers must publish Both");
    const auto next = table.begin_append(128, 64);
    require(table.page(0)->state == KVViewState::HostOnly &&
                table.page(2)->physical_slot == writes[0].physical_slot,
            "old completed page must release its slot and keep its logical descriptor");
    require(next[0].logical_page == 2, "logical page number must be preserved");
    require(!table.complete_writeback(0, 0, writes[0].epoch),
            "evicted page must reject duplicate completion with matching epoch");
    validate_snapshot(table.snapshot(), 2);
}

void sink_and_oldest_are_protected() {
    KVViewTable table(1024, 4, 1);
    const auto writes = table.begin_append(0, 256);
    complete(table, writes[0]);
    complete(table, writes[2]);
    complete(table, writes[3]);
    const auto next = table.begin_append(256, 64);
    require(table.page(0)->state == KVViewState::Both &&
                table.page(1)->state == KVViewState::DeviceOnly &&
                table.page(2)->state == KVViewState::HostOnly &&
                next[0].physical_slot == writes[2].physical_slot,
            "evict oldest Both non-sink while skipping incomplete page");
    complete(table, writes[1]);
    complete(table, next[0]);
    (void)table.begin_append(320, 64);
    require(table.page(1)->state == KVViewState::HostOnly, "oldest newly completed page evicted");
    validate_snapshot(table.snapshot(), 4);
}

void touching_partial_page_invalidates_writeback() {
    KVViewTable table(512, 2, 0, 2);
    const auto original = table.begin_append(0, 65);
    complete(table, original[0], 2);
    require(table.complete_writeback(0, 1, original[1].epoch), "partial layer notification");
    const auto update = table.begin_append(65, 1);
    require(update.size() == 1 && update[0].logical_page == 1 &&
                update[0].physical_slot == original[1].physical_slot &&
                !(update[0].epoch == original[1].epoch),
            "partial append must issue a new revision");
    require(!table.complete_writeback(1, 1, original[1].epoch), "stale revision must be rejected");
    require(table.complete_writeback(1, 1, update[0].epoch), "new revision accepts notification");
    require(table.page(1)->state == KVViewState::DeviceOnly,
            "old layer completion cannot carry over");
    require(table.complete_writeback(0, 1, update[0].epoch), "new revision remaining layer");
    require(table.page(1)->state == KVViewState::Both, "new complete revision becomes Both");
}

void append_range_is_transactional_and_protected() {
    KVViewTable table(512, 2, 0, 1);
    auto writes = table.begin_append(0, 65);
    complete(table, writes[0], 1);
    complete(table, writes[1], 1);
    const auto before = table.snapshot();
    rejects([&] { (void)table.begin_append(65, 128); },
            "append cannot evict its own partial first page to fit an oversized range");
    equal_snapshot(before, table.snapshot());
    require(!table.begin_append(65, 0).size(), "empty append is a no-op");
    equal_snapshot(before, table.snapshot());
    const auto update = table.begin_append(65, 64);
    require(update.size() == 2 && table.page(0)->state == KVViewState::HostOnly &&
                update[0].physical_slot == writes[1].physical_slot,
            "current partial page must remain resident during append");
    require(table.page(1)->state == KVViewState::DeviceOnly, "modified Both returns DeviceOnly");
    require(!table.complete_writeback(0, 1, writes[1].epoch), "modified Both rejects old revision");
    validate_snapshot(table.snapshot(), 2);
}

void trim_invalidates_epochs_and_reuses_tail_slots() {
    KVViewTable table(1024, 4, 1, 2);
    const auto writes = table.begin_append(0, 256);
    complete(table, writes[0], 2);
    require(table.complete_writeback(0, 1, writes[1].epoch), "pre-trim layer");
    const auto generation = table.snapshot().generation;
    table.trim(70);
    require(table.snapshot().generation == generation + 1 && table.snapshot().frontier == 70,
            "trim must publish exact frontier and advance generation");
    require(table.page(1)->physical_slot == writes[1].physical_slot && !table.page(2),
            "trim retains partial last page and releases full trailing pages");
    require(!table.complete_writeback(1, 1, writes[1].epoch) &&
                !table.complete_writeback(0, 2, writes[2].epoch),
            "trim rejects old notifications");
    auto retained = *table.page(1);
    require(table.complete_writeback(1, 1, retained.epoch), "retained page new-generation layer");
    require(table.page(1)->state == KVViewState::DeviceOnly,
            "pre-trim completion cannot carry over");
    require(table.complete_writeback(0, 1, retained.epoch), "retained page remaining layer");
    require(table.page(1)->state == KVViewState::Both, "retained prefix can complete");
    const auto append = table.begin_append(70, 122);
    require(append.size() == 2 && append[1].physical_slot == writes[2].physical_slot,
            "trimmed tail slot is reusable");
    require(!table.complete_writeback(0, 2, writes[2].epoch),
            "reused logical page rejects old epoch");
    validate_snapshot(table.snapshot(), 4);
    const auto before = table.snapshot();
    rejects([&] { table.trim(193); }, "trim cannot grow frontier");
    equal_snapshot(before, table.snapshot());
    table.trim(64);
    require(!table.page(1), "aligned trim removes next page");
    table.trim(0);
    require(table.snapshot().resident.empty() && table.snapshot().blocktable.empty(), "trim zero");
}

void host_only_partial_page_requires_restore() {
    KVViewTable table(1024, 2, 0, 1);
    auto initial = table.begin_append(0, 128);
    complete(table, initial[0], 1);
    complete(table, initial[1], 1);
    (void)table.begin_append(128, 64);
    table.trim(10);
    require(table.page(0)->state == KVViewState::HostOnly, "trim retains host-only prefix");
    auto append = table.begin_append(10, 1);
    require(append[0].restore_from_host && append[0].logical_page == 0 &&
                table.page(0)->state == KVViewState::DeviceOnly,
            "writing host-only partial page allocates a slot and requests prefix restore");
    validate_snapshot(table.snapshot(), 2);
}

void original_logical_number_survives_many_evictions() {
    KVViewTable table(3002 * 64, 3, 1, 1);
    for (std::uint32_t page = 0; page <= 3000; ++page) {
        const auto writes = table.begin_append(page * 64, 64);
        require(writes[0].logical_page == page, "logical pages must never be compacted");
        complete(table, writes[0], 1);
    }
    const auto snapshot = table.snapshot();
    require(snapshot.blocktable.size() == 3001 && snapshot.resident[0].logical_page == 0 &&
                snapshot.resident[1].logical_page == 2999 &&
                snapshot.resident[2].logical_page == 3000,
            "view must consist of protected sink and most recent original logical pages");
    require(snapshot.host_only.size() == 2998 && snapshot.host_only.front() == 1 &&
                snapshot.host_only.back() == 2998,
            "host-only pages retain original ascending IDs");
    validate_snapshot(snapshot, 3);
}

void restore_rebuilds_sink_recent_and_partial_frontier() {
    KVViewTable table(1024, 3, 1, 1);
    std::vector<KVViewWrite> old;
    for (std::uint32_t base = 0; base < 384; base += 64) {
        old = table.begin_append(base, 64);
        complete(table, old[0], 1);
    }
    const auto captured_generation = table.snapshot().generation;
    const auto before = table.snapshot();
    const auto longer = table.plan_restore(321);
    require(longer.resident.size() == 3 && longer.resident[0].logical_page == 0 &&
                longer.resident[1].logical_page == 4 && longer.resident[2].logical_page == 5 &&
                longer.host_only == std::vector<std::uint32_t>({1, 2, 3}),
            "restore must retain sink and recent partial tail, with older original pages host-only");
    const auto plan = table.plan_restore(129);
    require(plan.frontier == 129 && plan.generation == captured_generation + 1,
            "restore plan exact partial frontier and fresh generation");
    require(plan.resident.size() == 3 && plan.resident[0].logical_page == 0 &&
                plan.resident[1].logical_page == 1 && plan.resident[2].logical_page == 2,
            "restore must hydrate checkpoint sink/recent/current page, not final live mapping");
    equal_snapshot(before, table.snapshot());
    table.install_restore(plan);
    validate_snapshot(table.snapshot(), 3);
    require(!table.complete_writeback(0, old[0].logical_page, old[0].epoch),
            "restore rejects stale writeback");
    const auto repeated = table.plan_restore(129);
    table.install_restore(repeated);
    require(table.snapshot().generation == captured_generation + 2,
            "equal frontier restore advances generation");
    rejects([&] { table.install_restore(plan); }, "stale restore plan rejected");
    rejects([&] { (void)table.plan_restore(130); }, "restore cannot grow archive frontier");
    const auto append = table.begin_append(129, 1);
    require(!append[0].restore_from_host, "restored partial tail already resident");
    complete(table, append[0], 1);
    table.reset();
    rejects([&] { table.install_restore(repeated); }, "reset invalidates restore plan");
}

void reset_and_invalid_inputs() {
    rejects([] { KVViewTable table(0, 1, 0); }, "zero context rejected");
    rejects([] { KVViewTable table(64, 0, 0); }, "zero view rejected");
    rejects([] { KVViewTable table(128, 1, 2); }, "sink larger than view rejected");
    rejects([] { KVViewTable table(64, 2, 2); }, "sink beyond logical context rejected");
    rejects([] { KVViewTable table(64, 1, 0, 0); }, "zero layers rejected");
    KVViewTable table(65, 2, 0, 2);
    const auto writes = table.begin_append(0, 65);
    require(!table.complete_writeback(2, 0, writes[0].epoch) &&
                !table.complete_writeback(0, 99, writes[0].epoch),
            "invalid layer/page notifications rejected");
    require(table.complete_writeback(0, 0, writes[0].epoch), "first notification");
    require(!table.complete_writeback(0, 0, writes[0].epoch), "duplicate notification rejected");
    require(table.page(0)->state == KVViewState::DeviceOnly,
            "duplicate cannot satisfy other layer");
    require(table.complete_writeback(1, 0, writes[0].epoch), "remaining layer");
    require(!table.complete_writeback(1, 0, writes[0].epoch),
            "completed notification duplicate rejected");
    auto before = table.snapshot();
    rejects([&] { (void)table.begin_append(64, 1); }, "noncontinuous append rejected");
    rejects([&] { (void)table.begin_append(65, 1); }, "context overflow rejected");
    rejects([&] { (void)table.begin_append(65, 0xffffffffU); }, "count overflow rejected");
    equal_snapshot(before, table.snapshot());
    table.reset();
    require(table.snapshot().generation == before.generation + 1 &&
                table.snapshot().frontier == 0 && !table.page(0),
            "reset releases all pages and advances generation");
    const auto fresh = table.begin_append(0, 64);
    require(!table.complete_writeback(0, 0, writes[0].epoch), "reset rejects prior incarnation");
    complete(table, fresh[0], 2);
    validate_snapshot(table.snapshot(), 2);
}

void partial_admission_failure_preserves_all_descriptors() {
    KVViewTable table(512, 3, 1, 1);
    const auto original = table.begin_append(0, 192);
    complete(table, original[0], 1);
    complete(table, original[1], 1);
    const auto before = table.snapshot();
    rejects([&] { (void)table.begin_append(192, 128); },
            "one completed victim cannot provide two required slots");
    equal_snapshot(before, table.snapshot());
    require(table.page(1)->state == KVViewState::Both,
            "a tentatively selected victim must stay resident after admission failure");
    complete(table, original[2], 1);
    const auto writes = table.begin_append(192, 128);
    require(writes.size() == 2 && writes[0].physical_slot == original[1].physical_slot &&
                writes[1].physical_slot == original[2].physical_slot,
            "multiple victims must be selected in ascending logical age");
    validate_snapshot(table.snapshot(), 3);

    KVViewTable sinks_only(128, 1, 1, 1);
    complete(sinks_only, sinks_only.begin_append(0, 64)[0], 1);
    const auto sink_snapshot = sinks_only.snapshot();
    rejects([&] { (void)sinks_only.begin_append(64, 64); },
            "even completed sink cannot be evicted");
    equal_snapshot(sink_snapshot, sinks_only.snapshot());
}
} // namespace

int main() {
    try {
        all_layers_gate_eviction();
        sink_and_oldest_are_protected();
        touching_partial_page_invalidates_writeback();
        append_range_is_transactional_and_protected();
        trim_invalidates_epochs_and_reuses_tail_slots();
        host_only_partial_page_requires_restore();
        original_logical_number_survives_many_evictions();
        restore_rebuilds_sink_recent_and_partial_frontier();
        reset_and_invalid_inputs();
        partial_admission_failure_preserves_all_descriptors();
        std::cout << "PASS: CPU KV view table (10 cases)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

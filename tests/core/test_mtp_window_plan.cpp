#include "targets/qwen3_6/impl/runtime/mtp_window_plan.h"
#include <iostream>
#include <stdexcept>
using namespace ninfer;
using namespace ninfer::targets::qwen3_6::detail;
void require(bool v, const char* message) {
    if (!v) throw std::runtime_error(message);
}
int main() {
    try {
        const auto p = plan_mtp_window(262144, 2048, TieredKVOptions{});
        require(p.physical_pages == 512 && p.sink_pages == 4 && p.recent_pages == 507 &&
                    p.guard_pages == 1,
                "32K must include sink and provisional guard inside 512 physical pages");
        auto o              = TieredKVOptions{};
        o.mtp_window_tokens = 256;
        bool rejected       = false;
        try {
            (void)plan_mtp_window(262144, 2048, o);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "small ring must reject a chunk that aliases its own writes");
        o.sink_tokens       = 0;
        o.mtp_window_tokens = 512;
        const auto small    = plan_mtp_window(262144, 128, o);
        require(small.physical_pages == 8 && small.recent_pages == 7 && small.guard_pages == 1,
                "zero sink ring preserves fixed physical budget");
        auto invalid=o; invalid.mtp_window_tokens=513;
        rejected=false;
        try { (void)plan_mtp_window(262144,128,invalid); } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"unaligned MTP ring must be rejected before loading");
        invalid=o; invalid.sink_tokens=448;
        rejected=false;
        try { (void)plan_mtp_window(262144,128,invalid); } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"sink cannot exhaust ring and guard budget");
        const auto short_context = plan_mtp_window(2048, 1024, TieredKVOptions{});
        require(short_context.physical_pages == 32 && short_context.guard_pages == 0,
                "context shorter than window needs no overwrite guard");
        const auto sink_only=plan_mtp_window(128,128,TieredKVOptions{});
        require(sink_only.sink_pages==2 && sink_only.recent_pages==0 && sink_only.guard_pages==0,
                "short context entirely in sink remains valid");
        std::cout << "MTP window planning passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

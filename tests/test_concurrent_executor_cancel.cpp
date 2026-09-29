#include "artifact/reader.h"
#include "runtime/engine/concurrent_executor.h"
#include "targets/qwen3_6/impl/frontend/test_access.h"

#include <cstdlib>
#include <future>
#include <iostream>

namespace {
using namespace ninfer;
using namespace ninfer::runtime;
namespace qwen = ninfer::targets::qwen3_6;

void require(bool result, const char* message) {
    if (!result) { throw std::runtime_error(message); }
}

// Exercise the real executor/output frontend, substituting only target compute and memory.
// No text/vision weights or GPU allocations are needed for these scheduling contracts.
struct Plan {
    RequestPlanSummary value;
    bool cancel_target = false;
    const RequestPlanSummary& summary() const { return value; }
};

struct Compute {
    bool later = false;
    std::array<bool, 2> active{};
    std::array<bool, 2> cancel_target{};
    std::array<const std::atomic_bool*, 2> cancellation{};
    std::array<TokenId, 1> token{198};
    std::atomic<int> aborts{0};
    std::promise<void> entered;

    AdmissionResources admission_capacity() const { return {2, 2, 0}; }
    std::string config_signature_slug() const { return {}; }
    void set_disk_state_cache(DiskStateCache*) {}
    Plan plan_request_base(const qwen::PreparedPrompt& prompt, const ResolvedExecutionOptions&) {
        const auto& data = qwen::FrontendTestAccess::inspect(prompt);
        return {.value = {.prompt_tokens = 2, .requested_output_tokens = 1,
                          .effective_output_tokens = 1, .effective_limit_reason = FinishReason::OutputLimit,
                          .transient_bytes = 256, .transient_alignment = 1,
                          .admission = {1, 1, 0}, .service_work_quanta = 2},
                .cancel_target = data.token_ids.front() == 198};
    }
    Plan plan_request_for_lane(std::uint32_t, const qwen::PreparedPrompt&, const Plan& base) { return base; }
    bool can_admit_lane(std::uint32_t lane, const Plan&) const { return !active.at(lane); }
    bool can_admit_lane_after_retained_eviction(std::uint32_t lane, const Plan& plan) const {
        return can_admit_lane(lane, plan);
    }
    bool has_retained_lane(std::uint32_t) const { return false; }
    std::uint32_t retained_lane_depth(std::uint32_t) const { return 0; }
    void evict_retained_lane(std::uint32_t) {}
    void block_until_cancelled(std::uint32_t lane) {
        entered.set_value();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!cancellation.at(lane)->load(std::memory_order_acquire)) {
            if (std::chrono::steady_clock::now() >= deadline) {
                throw std::runtime_error("cancel callback was not propagated to target compute");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        throw RequestCancelled();
    }
    PrefillStepResult start_prefill_lane(std::uint32_t lane, qwen::PreparedPrompt&&, Plan&& plan,
                                        TransientRegion transient, const std::atomic_bool* cancelled) {
        require(transient.size == 256 && cancelled != nullptr, "prefill inputs not propagated");
        active.at(lane) = true;
        cancel_target.at(lane) = plan.cancel_target;
        cancellation.at(lane) = cancelled;
        if (plan.cancel_target && !later) { block_until_cancelled(lane); }
        if (plan.cancel_target) {
            return {.summary = {.prompt_tokens = 2}, .processed_prompt_tokens = 1};
        }
        return {.summary = {.prompt_tokens = 2}, .round = {token},
                .processed_prompt_tokens = 2, .complete = true, .host_input_consumed = true};
    }
    PrefillStepResult advance_prefill_lane(std::uint32_t lane) {
        require(cancel_target.at(lane) && later, "unexpected later prefill");
        block_until_cancelled(lane);
        return {};
    }
    void resolve_prefill_lane(std::uint32_t lane, bool terminal) {
        require(terminal, "fixture should finish with its first token");
        active.at(lane) = false;
    }
    void abort_lane(std::uint32_t lane) { active.at(lane) = false; ++aborts; }
    BatchedGeneratedRound decode_batch(std::span<const std::uint32_t>, std::span<const RoundBudget>) {
        throw std::logic_error("fixture should not reach decode");
    }
    void resolve_pending_batch(std::span<const std::uint32_t>, std::span<const std::uint32_t>,
                               std::span<const std::uint8_t>, std::span<const std::uint8_t>) {}
    void set_token_mask_lane(std::uint32_t, std::span<const std::uint32_t>) {}
    GenerationTimings generation_timings_lane(std::uint32_t) const { return {}; }
    SpeculativeStats speculative_stats_lane(std::uint32_t) const { return {}; }
    CpuVisionCacheStats cpu_vision_cache_stats_lane(std::uint32_t) const { return {}; }
    void snapshot_lane_to_disk(std::uint32_t, DiskStateCache&) {}
    void snapshot_turn_checkpoint_to_disk(std::uint32_t, DiskStateCache&) {}
    MemorySummary memory_summary() const { return {}; }
    void reset_memory_peaks() {}
};

struct Memory {
    std::array<std::byte, 256> bytes{};
    bool active = false;
    int activations = 0;
    int deactivations = 0;
    void activate(std::size_t size, std::size_t) {
        require(!active && size == bytes.size(), "previous request leaked its transient owner");
        active = true;
        ++activations;
    }
    void deactivate() { require(active, "transient deactivated twice"); active = false; ++deactivations; }
    TransientRegion region() { return {bytes.data(), bytes.size(), 1}; }
    ArenaMemorySummary summary() const { return {256, active ? 256U : 0U, 256}; }
    void reset_peak() {}
};

struct TestPackage { using Program = Compute; using RequestBasePlan = Plan; using RequestPlan = Plan; };
struct Loaded { qwen::Frontend frontend; };
struct Instance {
    using Package = TestPackage;
    Compute* program;
    Loaded* loaded;
    Memory request_memory;
    KvCapacityResolution kv_capacity_resolution;
};

void check(const qwen::Frontend& frontend, bool later) {
    Compute compute;
    compute.later = later;
    auto entered = compute.entered.get_future();
    Loaded loaded{frontend};
    Instance instance{&compute, &loaded};
    EngineOptions options;
    options.max_concurrency = 2;
    options.max_pending_requests = 2;
    options.pending_timeout_ms = 10000;
    options.enable_prompt_cache = false;
    ConcurrentExecutor<Instance> executor(instance, options);
    const auto submit = [&](TokenId marker, HostInputLease lease = {}) {
        auto prompt = frontend.prepare_tokens({marker, 198});
        auto summary = prompt.summary();
        ResolvedRequestOptions request;
        request.execution.requested_output_tokens = 1;
        request.stop.include_model_defaults = false;
        return executor.submit(std::move(prompt), summary, 0.0, request, {}, std::move(lease));
    };
    auto owner = std::make_shared<int>(1);
    std::weak_ptr<int> weak = owner;
    auto first = submit(198, HostInputLease(owner));
    owner.reset();
    std::atomic<bool> cancel{false};
    auto result = std::async(std::launch::async, [&] {
        return first.wait(nullptr, CancellationView([&] { return cancel.load(); }));
    });
    if (entered.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        cancel.store(true);
        throw std::runtime_error("target never entered the intended prefill cancellation stage");
    }
    // This request is queued while the first target compute is blocked.
    auto second = submit(199);
    cancel.store(true);
    const auto cancelled = result.get();
    const auto completed = second.wait(nullptr, {});
    require(cancelled.finish_reason == FinishReason::Cancelled && cancelled.generated_token_ids.empty(),
            "target cancellation did not complete normally");
    require(completed.finish_reason == FinishReason::OutputLimit && completed.generated_token_ids.size() == 1,
            "cancelled target poisoned the queued request");
    require(executor.healthy() && weak.expired() && compute.aborts.load() == 1,
            "cancelled target leaked its input or failed the executor");
    // Flush the execution mutex before inspecting target/memory cleanup and published stats.
    require(executor.memory_summary().request_transient.used_bytes == 0, "transient owner leaked");
    const auto stats = executor.runtime_stats();
    require(stats.running_requests == 0 && stats.prefilling_requests == 0 && stats.waiting_requests == 0 &&
            instance.request_memory.activations == instance.request_memory.deactivations,
            "lane/transient/admission state was not cleaned up");
    auto third = submit(200);
    require(third.wait(nullptr, {}).finish_reason == FinishReason::OutputLimit,
            "admission capacity was not returned after cancellation");
}

qwen::Frontend load_frontend(const char* path) {
    artifact::Reader reader(path);
    const auto resource = [&](const char* name) {
        const auto bytes = reader.payload(name).data;
        return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    return qwen::make_frontend({
        .tokenizer_json = resource("frontend/tokenizer.json"),
        .tokenizer_config_json = resource("frontend/tokenizer_config.json"),
        .chat_template_jinja = resource("frontend/chat_template.jinja"),
        .generation_config_json = resource("frontend/generation_config.json"),
        .preprocessor_config_json = resource("frontend/preprocessor_config.json"),
        .video_preprocessor_config_json = resource("frontend/video_preprocessor_config.json")}, false);
}
} // namespace

int main() {
    const char* path = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    if (!path || !*path) { std::cout << "skip: NINFER_QWEN3_8_27B_WEIGHTS frontend fixture needed\n"; return 77; }
    try { const auto frontend = load_frontend(path); check(frontend, false); check(frontend, true); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}

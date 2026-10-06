#include <ninfer/engine.h>
#include <cstdlib>
#include <atomic>
#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "core/device.h"
#include "runtime/contract/sampling.h"
#include "runtime/engine/test_access.h"
#include "targets/registry.h"
#include "targets/qwen3_6/impl/runtime/test_access.h"

using namespace ninfer;
namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
ChatMessage message(std::string role, std::string text) {
    ChatMessage m;
    m.role = std::move(role);
    m.parts.push_back({.kind = MessagePartKind::Text, .text = std::move(text)});
    return m;
}
void exercise(const char* artifact, bool mtp) {
    EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 8192;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(8192); // non-Dense uses approved view floor
    options.kv_cache = KvCacheStorage::Int8Group64;
    options.kv_mode = KvMode::KVMem;
    options.kvmem.view_tokens = 1024;
    options.kvmem.sink_tokens = 64;
    options.kvmem.recent_tokens = 64;
    options.kvmem.gen_reserve_tokens = 128;
    options.kvmem.query_tokens = 4;
    options.kvmem.mtp_window_tokens = 512;
    options.kvmem.host_archive = mtp ? HostKVArchiveMode::Pinned : HostKVArchiveMode::Pageable;
    options.prefill_chunk = 128;
    options.enable_vision = false;
    // API default graph=true must automatically use eager KVMem.
    options.speculative.backend = mtp ? SpeculativeBackend::Mtp : SpeculativeBackend::None;
    options.speculative.draft_tokens = mtp ? 3 : 0;
    options.speculative.proposal_head = mtp ? ProposalHead::Optimized : ProposalHead::Full;
    Engine engine(options);
    require(engine.load_summary().prefill_chunk == 128, "explicit sparse inner chunk changed");
    const auto memory = engine.memory_summary();
    require(memory.cuda_graph_allowance_bytes == 0 && memory.cuda_graph_observed_bytes == 0,
            "API default graph=true allocated or captured graphs for KVMem");
    require(memory.kv_capacity == options.kvmem.view_tokens,
            "API loading bypassed the approved mode-aware resident capacity floor");
    RequestOptions request;
    request.execution.sampling.temperature = 0;
    request.execution.requested_output_tokens = 9;
    request.stop.include_model_defaults = false;
    request.output.raw = true;
    request.output.preserve_special_tokens = true;
    std::string archive;
    for (int i = 0; i < 200; ++i) archive += "Archived record " + std::to_string(i) + ": cedar.\n";
    PromptInput first;
    first.options.enable_thinking = false;
    first.current_input_message = 0;
    first.messages.push_back(message("user", archive + "Say cedar."));
    const auto run = [&](const PromptInput& prompt, bool reuse) {
        request.execution.allow_prefix_reuse = reuse;
        const auto before = engine.runtime_stats().computed_prefill_tokens;
        const auto result = engine.generate(engine.prepare(prompt), request);
        require(result.finish_reason == FinishReason::OutputLimit, "sparse generation ended early");
        require(engine.runtime_stats().computed_prefill_tokens - before + result.reused_prompt_tokens == result.prompt.prompt_tokens,
                "actual prefill progress lost prompt ordinals");
        return result;
    };
    const auto initial = run(first, false);
    require(initial.prompt.prompt_tokens > options.kvmem.view_tokens, "fixture did not require sparse publication");
    PromptInput second = first;
    second.messages.push_back(message("assistant", initial.content));
    second.current_input_message = second.messages.size();
    second.messages.push_back(message("user", "cedar"));
    second.messages.push_back(message("tool", "Current input tool envelope."));
    second.messages.push_back(message("user", "Say cedar."));
    const auto appended = run(second, true);
    require(appended.reused_prompt_tokens > 0, "sparse next exact prefill did not reuse resident continuation");
    const auto retry = run(second, true);
    require(retry.prefix_reuse_path == PrefixReusePath::RestoreTurnCheckpoint,
            "actual checkpoint restore missing in sparse Program");
    const auto retry_again = run(second, true);
    require(retry.generated_token_ids == retry_again.generated_token_ids,
            "query/Mean snapshot reuse changed deterministic sparse retry");
    // A tool-only event carries owning original Q across the evaluated prefix.
    PromptInput tool = second;
    tool.messages.push_back(message("assistant", retry.content));
    tool.current_input_message = tool.messages.size();
    tool.messages.push_back(message("tool", "Continue with cedar."));
    const auto continued = run(tool, true);
    require(continued.reused_prompt_tokens > 0, "tool-only continuation lost owning original query");
    const auto cold = run(second, false);
    require(cold.generated_token_ids == retry.generated_token_ids,
            "checkpoint Q recapture does not match complete cold sparse execution");
    // Real executor cancellation after output starts must discard any pending
    // Main transaction and leave the next whole bundle cold rebuild usable.
    struct CancelSink final : OutputSink {
        std::atomic_bool cancelled{false};
        void publish(OutputDelta) override { cancelled.store(true); }
    } sink;
    request.execution.requested_output_tokens = 64;
    const auto cancelled = engine.generate(engine.prepare(second), request, &sink,
        CancellationView([&] { return sink.cancelled.load(); }));
    require(cancelled.finish_reason == FinishReason::Cancelled, "real sparse cancellation did not propagate");
    require(engine.healthy(), "safe cancellation poisoned the sparse Engine");
    request.execution.requested_output_tokens = 1;
    const auto one = run(second, false);
    const auto exact = run(second, true);
    require(exact.reused_prompt_tokens == exact.prompt.prompt_tokens &&
        exact.generated_token_ids == one.generated_token_ids,
        "zero-suffix owning Q/accepted snapshot restore failed");
    const auto before_bad = engine.runtime_stats().computed_prefill_tokens;
    bool rejected = false;
    try { (void)engine.generate(engine.prepare_tokens({248045, 846, 198, 5834, 248046, 198}), request); }
    catch (const std::exception&) { rejected = true; }
    require(rejected && engine.runtime_stats().computed_prefill_tokens == before_bad,
            "KVMem raw input without role provenance reached expensive prefill");
    PromptInput unrelated;
    unrelated.options.enable_thinking = false;
    unrelated.current_input_message = 1;
    unrelated.messages.push_back(message("user", "Say birch."));
    unrelated.messages.push_back(message("tool", "Continue with birch."));
    (void)run(first, false); // retained long Q ordinals are outside this new short history
    const auto unrelated_cold = run(unrelated, false);
    (void)run(first, false);
    require(run(unrelated, true).generated_token_ids == unrelated_cold.generated_token_ids,
        "unrelated short tool-only history did not use its valid cold source query");
    PromptInput in_range = first;
    in_range.current_input_message = 1;
    in_range.messages[0] = message("user", archive + "Say birch.");
    in_range.messages.push_back(message("tool", "Continue with birch."));
    const auto changed_source_cold = run(in_range, false);
    (void)run(first, false);
    require(run(in_range, true).generated_token_ids == changed_source_cold.generated_token_ids,
        "in-range retained ordinals bypassed original source-prefix identity");
    std::cout << "actual KVMem unrelated tool cold query PASS mtp=" << mtp << '\n';
    std::cout << "actual KVMem Program PASS mtp=" << mtp << " prompt=" << initial.prompt.prompt_tokens
              << " checkpoint=" << retry.reused_prompt_tokens << '\n';
}

void exercise_failure_boundary(const char* artifact, bool mtp) {
    EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 8192;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(8192);
    options.kv_cache = KvCacheStorage::Int8Group64;
    options.kv_mode = KvMode::KVMem;
    options.kvmem.view_tokens = 1024;
    options.kvmem.sink_tokens = 64;
    options.kvmem.recent_tokens = 64;
    options.kvmem.gen_reserve_tokens = 128;
    options.kvmem.query_tokens = 4;
    options.kvmem.mtp_window_tokens = 512;
    options.kvmem.host_archive = mtp ? HostKVArchiveMode::Pinned : HostKVArchiveMode::Pageable;
    options.prefill_chunk = 128;
    options.enable_vision = false;
    options.speculative.backend = mtp ? SpeculativeBackend::Mtp : SpeculativeBackend::None;
    options.speculative.draft_tokens = mtp ? 3 : 0;
    options.speculative.proposal_head = mtp ? ProposalHead::Optimized : ProposalHead::Full;
    DeviceContext device;
    auto target = targets::construct_target(options, device);
    auto& instance = *std::get<std::unique_ptr<targets::Qwen3_6_27BInstance>>(target.active);
    auto& program = *instance.program;
    using Access = targets::qwen3_6::detail::ProgramTestAccess;
    auto& owner = Access::owner(program);
    runtime::ResolvedExecutionOptions execution;
    SamplingOverrides sampling;
    sampling.temperature = 0;
    execution.sampling = runtime::resolve_sampling(target.sampling_defaults, SamplingMode::NonThinking, sampling);
    execution.requested_output_tokens = 9;
    execution.allow_prefix_reuse = false;
    PromptInput prompt;
    prompt.current_input_message = 0;
    prompt.options.enable_thinking = false;
    std::string text;
    for (int i = 0; i < 200; ++i) text += "Archived record " + std::to_string(i) + ": cedar.\n";
    prompt.messages.push_back(message("user", text + "Say cedar."));
    const auto run = [&](bool inject_d2h, bool terminal = true) {
        auto prepared = instance.loaded->frontend.prepare(prompt);
        auto base = program.plan_request_base(prepared, execution);
        auto plan = program.plan_request_for_lane(0, prepared, base);
        instance.request_memory.activate(plan.summary().transient_bytes, plan.summary().transient_alignment);
        auto step = program.start_prefill_lane(0, std::move(prepared), std::move(plan), instance.request_memory.region());
        require(!step.complete, "fault fixture did not expose an actual inner prefill boundary");
        if (inject_d2h) {
            kvmem::HostKVTransferFaultInjection fault;
            fault.reject_d2h_submission_after = 1; // actual safe invalid-kind CUDA submission on valid pointers
            owner.set_transfer_test_fault(fault);
        }
        while (!step.complete) step = program.advance_prefill_lane(0);
        const auto token = step.round.tokens.front();
        program.resolve_prefill_lane(0, terminal);
        instance.request_memory.deactivate();
        return token;
    };
    const auto cold_token = run(false);
    // Resolve the real Main transaction with exactly the accepted count, even
    // on a full commit, a deliberately shortened commit, and a zero cancellation.
    for (int commit_case = 0; commit_case < (mtp ? 3 : 2); ++commit_case) {
        (void)run(false, false);
        const auto before = owner.frontier();
        const std::array<std::uint32_t, 1> lanes{0};
        const std::array<runtime::RoundBudget, 1> budgets{{
            {.generated_tokens_remaining = commit_case == 0 ? 1U : 8U}}};
        const auto round = program.decode_batch(lanes, budgets);
        const auto valid = owner.current_view().frontier - before;
        const auto produced = round.row_counts.empty() ? 1U : static_cast<std::uint32_t>(round.row_counts[0]);
        require(valid >= 1 && produced >= 1 && produced <= valid &&
            owner.sparse_capture()->index().frontier() == before,
            "real Main transaction published unaccepted or padded columns");
        const bool cancel = mtp ? commit_case == 2 : commit_case == 1;
        const std::array<std::uint32_t, 1> accepted{cancel ? 0U : 1U};
        const std::array<std::uint8_t, 1> terminal{1}, cancelled{static_cast<std::uint8_t>(cancel)};
        if (commit_case == 0)
            require(valid == 1 && produced == valid,
                "one-token remaining budget did not exercise a full actual-T commit below MTP padded storage");
        if (mtp && commit_case == 1)
            require(valid > accepted[0], "partial Main commit fixture did not expose speculative columns");
        program.resolve_pending_batch(lanes, accepted, terminal, cancelled);
        require(!owner.sparse_capture()->transaction_pending() &&
            owner.frontier() == (cancel ? 0U : before + accepted[0]) &&
            owner.sparse_capture()->index().frontier() == owner.frontier(),
            "real Program did not flush full/partial/zero accepted Main frontier before trim/reset");
        if (cancel) {
            const auto zero = Access::inspect(program);
            require(zero.gdn_zero && zero.positions_zero && zero.mtp_tags_empty &&
                !zero.owns_kv && !zero.hidden_valid && zero.ledger_empty,
                "zero acceptance/cancellation left a correlated provisional bundle");
        }
        std::cout << "actual KVMem Program commit PASS mtp=" << mtp << " case=" << commit_case
            << " valid=" << valid << " produced=" << produced << " accepted=" << accepted[0] << '\n';
    }
    bool cold_required = false;
    try { (void)run(true); }
    catch (const targets::qwen3_6::detail::KVMemColdResetRequired&) { cold_required = true; }
    require(cold_required, "actual uncertain D2H did not preserve the explicit whole-cold classification");
    instance.request_memory.deactivate();
    const auto cleared = Access::inspect(program);
    require(!owner.poisoned() && owner.frontier() == 0 && owner.sparse_capture()->index().frontier() == 0 &&
        !cleared.cleanup_failure && !cleared.owns_kv && cleared.execution_frontier == 0 &&
        cleared.ledger_frontier == 0 && cleared.ledger_empty && cleared.text_valid == 0 && cleared.mtp_valid == 0 &&
        !cleared.hidden_valid && !cleared.retained && !cleared.checkpoint_valid && !cleared.resume_valid &&
        !cleared.original_query && cleared.gdn_zero && cleared.positions_zero && cleared.mtp_tags_empty,
        "real Program uncertain D2H did not clear the correlated Main/GDN/MTP/hidden/position/ledger bundle");
    require(run(false) == cold_token, "whole-bundle cold rebuild changed deterministic real-model output");

    // Classify a fatal borrower without damaging the actual CUDA context.
    // A later successful drain must not authorize reset or another request.
    const auto prior = owner.current_view();
    const auto prior_state = Access::inspect(program);
    unsigned drains = 0;
    auto borrower = owner.sparse_capture()->borrow_query([&] {
        if (++drains == 1) throw kvmem::CudaTransferError(cudaErrorIllegalAddress,
            "Product test fatal Q borrower", false);
    });
    unsigned later_generic_drains = 0;
    auto later_generic = owner.sparse_capture()->borrow_query([&] {
        if (++later_generic_drains == 1) throw std::runtime_error("later generic Product borrower");
    });
    bool fatal_preserved = false;
    try { (void)run(false); }
    catch (const kvmem::CudaTransferError& error) {
        fatal_preserved = error.status() == cudaErrorIllegalAddress && !error.drain_failed();
    }
    require(fatal_preserved && owner.poisoned(), "Program masked the original fatal capture classification");
    owner.sparse_capture()->drain();
    require(drains >= 2 && later_generic_drains >= 2 && !borrower.valid() && !later_generic.valid(),
        "fatal fixture failed to prove every borrower attempted and later successfully drained");
    const auto blocked = Access::inspect(program);
    require(blocked.cleanup_failure && owner.frontier() == prior.frontier &&
        owner.current_view().generation == prior.generation && blocked.owns_kv &&
        blocked.execution_frontier == prior_state.execution_frontier && blocked.text_valid == prior_state.text_valid &&
        blocked.mtp_valid == prior_state.mtp_valid && blocked.ledger_frontier == prior_state.ledger_frontier &&
        blocked.hidden_valid == prior_state.hidden_valid && blocked.gdn_zero == prior_state.gdn_zero,
        "failed fatal reset mutated the accepted correlated bundle");
    bool next_blocked = false;
    try { (void)run(false); }
    catch (const kvmem::CudaTransferError& error) { next_blocked = error.status() == cudaErrorIllegalAddress; }
    require(next_blocked && owner.current_view().generation == prior.generation,
        "later successful drain erased sticky typed failure or allowed the next request");
    std::cout << "actual KVMem Program failure boundary PASS mtp=" << mtp << '\n';
}

// Test scheduling adapter: all state, math, reset and recovery classification
// remain in the registered real Program. An armed start runs its first real
// unit, installs the existing actual CUDA transfer rejection, then runs its
// second real unit before returning to the executor's start boundary.
struct StartFaultProgram {
    targets::Qwen3_6_27B::Program& real;
    targets::qwen3_6::detail::TieredContext& owner;
    cudaStream_t compute;
    bool reject_in_start = false;
    bool reject_checked_sync = false;
    bool reject_inner_sync = false, reject_reset = false;
    cudaError_t returned_status = cudaSuccess;
    std::uint64_t failed_generation = 0;
    std::uint64_t failed_index_generation = 0, failed_capture_id = 0;
    targets::qwen3_6::detail::ProgramTestState before_reset;
    template<class... Args> auto start_prefill_lane(Args&&... args) {
        auto step = real.start_prefill_lane(std::forward<Args>(args)...);
        const bool program_sync = std::exchange(reject_checked_sync, false);
        const bool inner_sync = std::exchange(reject_inner_sync, false);
        const bool reset_failure = std::exchange(reject_reset, false);
        if (program_sync || inner_sync || reset_failure) {
            require(!step.complete, "checked-sync fixture requires an actual completed first prefill unit");
            if (reset_failure) {
                while (!step.complete) step = real.advance_prefill_lane(0);
                before_reset = targets::qwen3_6::detail::ProgramTestAccess::inspect(real);
                failed_index_generation = owner.sparse_capture()->index().generation();
                require(bool(owner.sparse_capture()->query()), "reset fixture requires completed owning Q");
                failed_capture_id = owner.sparse_capture()->query()->id();
            }
            int valid_host_memory = 0;
            returned_status = cudaMemcpyAsync(&valid_host_memory, &valid_host_memory,
                sizeof(valid_host_memory), static_cast<cudaMemcpyKind>(99), compute);
            require(returned_status != cudaSuccess && !kvmem::cuda_failure_is_fatal(returned_status),
                "checked-sync fixture did not receive a safe actual CUDA rejection");
            failed_generation = owner.current_view().generation;
            // Feed the REAL returned API rejection into the ACTUAL compute-drain
            // checker. Failed-sync placement is simulated without corrupting the
            // CUDA context; no production status/operation is intercepted.
            if (reset_failure)
                targets::qwen3_6::detail::ProgramTestAccess::fail_reset_after_hardware_prepare(real, returned_status);
            else if (inner_sync)
                owner.check_compute_drain(returned_status, "KVMem inner prefill compute drain");
            else targets::qwen3_6::detail::ProgramTestAccess::check_compute_drain(real, returned_status);
            throw std::logic_error("checked compute drain accepted a failing status");
        }
        if (!std::exchange(reject_in_start, false)) return step;
        require(!step.complete, "start adapter needs two real staged prefill units");
        kvmem::HostKVTransferFaultInjection fault;
        fault.reject_d2h_submission_after = 1;
        owner.set_transfer_test_fault(fault);
        return real.advance_prefill_lane(0);
    }
#define FORWARD_REAL_PROGRAM(method) \
    template<class... Args> decltype(auto) method(Args&&... args) { \
        return real.method(std::forward<Args>(args)...); \
    }
    FORWARD_REAL_PROGRAM(abort_lane)
    FORWARD_REAL_PROGRAM(admission_capacity)
    FORWARD_REAL_PROGRAM(advance_prefill_lane)
    FORWARD_REAL_PROGRAM(can_admit_lane)
    FORWARD_REAL_PROGRAM(can_admit_lane_after_retained_eviction)
    FORWARD_REAL_PROGRAM(can_continue_after_gpu_failure)
    FORWARD_REAL_PROGRAM(config_signature_slug)
    FORWARD_REAL_PROGRAM(cpu_vision_cache_stats_lane)
    FORWARD_REAL_PROGRAM(decode_batch)
    FORWARD_REAL_PROGRAM(evict_retained_lane)
    FORWARD_REAL_PROGRAM(generation_timings_lane)
    FORWARD_REAL_PROGRAM(gpu_failure_after_cleanup)
    FORWARD_REAL_PROGRAM(has_retained_lane)
    FORWARD_REAL_PROGRAM(memory_summary)
    FORWARD_REAL_PROGRAM(plan_request_base)
    FORWARD_REAL_PROGRAM(plan_request_for_lane)
    FORWARD_REAL_PROGRAM(reset_memory_peaks)
    FORWARD_REAL_PROGRAM(resolve_pending_batch)
    FORWARD_REAL_PROGRAM(resolve_prefill_lane)
    FORWARD_REAL_PROGRAM(retained_lane_depth)
    FORWARD_REAL_PROGRAM(set_disk_state_cache)
    FORWARD_REAL_PROGRAM(set_token_mask_lane)
    FORWARD_REAL_PROGRAM(snapshot_lane_to_disk)
    FORWARD_REAL_PROGRAM(snapshot_turn_checkpoint_to_disk)
    FORWARD_REAL_PROGRAM(speculative_stats_lane)
#undef FORWARD_REAL_PROGRAM
};
struct StartFaultPackage {
    using Program = StartFaultProgram;
    using RequestBasePlan = targets::Qwen3_6_27B::RequestBasePlan;
    using RequestPlan = targets::Qwen3_6_27B::RequestPlan;
};
struct StartFaultInstance {
    using Package = StartFaultPackage;
    std::unique_ptr<targets::LoadedQwen3_6_27B>& loaded;
    runtime::KvCapacityResolution& kv_capacity_resolution;
    runtime::RequestMemory& request_memory;
    StartFaultProgram* program;
};

void exercise_executor_recovery(const char* artifact, bool mtp, unsigned checked_sync = 0, bool late_final = false, bool query_only = false, bool stronger_only = false) {
    EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 8192;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(8192);
    options.kv_cache = KvCacheStorage::Int8Group64;
    options.kv_mode = KvMode::KVMem;
    options.kvmem.view_tokens = 1024;
    options.kvmem.sink_tokens = 64;
    options.kvmem.recent_tokens = 64;
    options.kvmem.gen_reserve_tokens = 128;
    options.kvmem.query_tokens = 4;
    options.kvmem.mtp_window_tokens = 512;
    options.kvmem.host_archive = mtp ? HostKVArchiveMode::Pinned : HostKVArchiveMode::Pageable;
    options.prefill_chunk = 128;
    options.enable_vision = false;
    options.speculative.backend = mtp ? SpeculativeBackend::Mtp : SpeculativeBackend::None;
    options.speculative.draft_tokens = mtp ? 3 : 0;
    options.speculative.proposal_head = mtp ? ProposalHead::Optimized : ProposalHead::Full;
    DeviceContext device;
    auto target = targets::construct_target(options, device);
    auto& instance = *std::get<std::unique_ptr<targets::Qwen3_6_27BInstance>>(target.active);
    using Executor = runtime::ConcurrentExecutor<StartFaultInstance>;
    using ProgramAccess = targets::qwen3_6::detail::ProgramTestAccess;
    using ExecutorAccess = runtime::ConcurrentExecutorTestAccess;
    auto& owner = ProgramAccess::owner(*instance.program);
    StartFaultProgram scheduling{*instance.program, owner, device.stream};
    StartFaultInstance scheduling_instance{instance.loaded, instance.kv_capacity_resolution,
        instance.request_memory, &scheduling};
    Executor executor(scheduling_instance, options); // the actual Engine executor + registered real Program
    runtime::ResolvedRequestOptions request;
    SamplingOverrides sampling;
    sampling.temperature = 0;
    request.execution.sampling = runtime::resolve_sampling(target.sampling_defaults, SamplingMode::NonThinking, sampling);
    request.execution.requested_output_tokens = 9;
    request.stop.include_model_defaults = false;
    request.output.raw = true;
    request.output.preserve_special_tokens = true;
    PromptInput first;
    first.current_input_message = 0;
    first.options.enable_thinking = false;
    std::string history;
    for (int i = 0; i < 200; ++i) history += "Archived record " + std::to_string(i) + ": cedar.\n";
    first.messages.push_back(message("user", history + "Say cedar."));
    const auto submit = [&](const PromptInput& prompt, bool reuse) {
        auto prepared = instance.loaded->frontend.prepare(prompt);
        const auto summary = prepared.summary();
        auto resolved = request;
        resolved.execution.allow_prefix_reuse = reuse;
        return executor.submit(std::move(prepared), summary, 0, std::move(resolved));
    };
    const auto run = [&](const PromptInput& prompt, bool reuse, OutputSink* sink = nullptr,
                         const CancellationView& cancellation = CancellationView{}) {
        return submit(prompt, reuse).wait(sink, cancellation);
    };
    const auto baseline = run(first, false);
    if ((!checked_sync && !late_final && !stronger_only) || query_only) {
        PromptInput multi = first;
        multi.messages.push_back(message("assistant", baseline.content));
        multi.current_input_message = multi.messages.size();
        multi.messages.push_back(message("user", "Cedar cedar cedar cedar."));
        multi.messages.push_back(message("tool", "Intermediate cedar result."));
        multi.messages.push_back(message("user", "a"));
        const auto multi_result = run(multi, false);
        const auto source = owner.sparse_capture()->query()->provenance();
        require(source.ordinals.size() == 4 &&
            source.ordinals.back() > source.ordinals[source.ordinals.size() - 2] + 1,
            "multi-user query fixture did not span the intervening nonuser event");
        PromptInput tool = multi;
        tool.messages.push_back(message("assistant", multi_result.content));
        tool.current_input_message = tool.messages.size();
        tool.messages.push_back(message("tool", "Cedar tool continuation."));
        const auto warm = run(tool, true);
        const auto warm_id = owner.sparse_capture()->query()->id();
        require(owner.sparse_capture()->query()->provenance().ordinals == source.ordinals &&
            owner.sparse_capture()->query()->provenance().source_prefix == source.source_prefix,
            "warm tool continuation lost original multi-user query");
        const auto cold = run(tool, false);
        const auto& recaptured = owner.sparse_capture()->query();
        require(recaptured->provenance().ordinals == source.ordinals &&
            recaptured->provenance().source_prefix == source.source_prefix && recaptured->id() != warm_id,
            "cold tool continuation discarded valid original cross-span query or reused stale capture");
        require(cold.generated_token_ids == warm.generated_token_ids,
            "multi-user tool cold recapture output differs from warm continuation");
        std::cout << "actual KVMem original multi-user query cold PASS mtp=" << mtp << '\n';
        if (query_only) return;
    }
    if (late_final) {
        std::atomic<bool> paused{false}, resume{false}, exact_complete{false};
        std::optional<Executor::Submission> failed, queued;
        ExecutorAccess::with_execution_lock(executor, [&] {
            failed.emplace(submit(first, false));
            ExecutorAccess::start_one_pending(executor);
            while (owner.frontier() + options.prefill_chunk < baseline.prompt.prompt_tokens)
                ExecutorAccess::advance_one_prefill(executor);
            require(owner.frontier() == 2048, "late-final fixture did not reach the actual final short chunk");
            ExecutorAccess::advance_one_prefill(executor); // measured actual turn checkpoint: 53 + 4
            std::cout << "late fixture observed after staged unit frontier=" << owner.frontier()
                      << " prompt=" << baseline.prompt.prompt_tokens << std::endl;
            require(owner.frontier() == 2101, "late-final fixture did not reach actual checkpoint before final chunk");
            kvmem::HostKVTransferFaultInjection fault;
            fault.reject_d2h_submission_after = 64; // 16 FA layers x four INT8 planes, final layer/plane
            fault.pause_after_d2h_plane = 64;
            fault.d2h_paused = &paused; fault.resume_d2h = &resume;
            fault.selection_after_exact = &exact_complete;
            owner.set_transfer_test_fault(fault);
            queued.emplace(submit(first, false));
        });
        bool original_cold = false;
        try { (void)failed->wait(nullptr, {}); }
        catch (const targets::qwen3_6::detail::KVMemColdResetRequired&) { original_cold = true; }
        catch (const kvmem::CudaTransferError& error) {
            std::cout << "late writeback RED observer=selection-drain exact_frontier=2105 paused="
                << paused.load() << " after_exact=" << exact_complete.load() << " status="
                << int(error.status()) << " drain_failed=" << error.drain_failed() << std::endl;
            throw;
        }
        require(paused.load() && exact_complete.load(), "late-final failure preceded completed exact capture/select drain");
        require(original_cold && executor.healthy(), "late-final FA rejection lost whole-cold classification or executor health");
        require(queued->wait(nullptr, {}).generated_token_ids == baseline.generated_token_ids,
            "late-final FA rejection did not preserve queued cold output parity");
        std::cout << "actual KVMem executor late final FA PASS mtp=" << mtp << " base=2101 valid=4 plane=64\n";
        return;
    }
    if (checked_sync) {
        // Baseline exercises real normal checked stream synchronization first.
        ExecutorAccess::with_execution_lock(executor, [&] {
            scheduling.reject_checked_sync = checked_sync == 1;
            scheduling.reject_inner_sync = checked_sync == 2;
            scheduling.reject_reset = checked_sync == 3;
        });
        bool typed_failed_drain = false;
        try { (void)run(first, false); }
        catch (const kvmem::CudaTransferError& error) {
            typed_failed_drain = error.status() == scheduling.returned_status && error.drain_failed() &&
                std::string_view(error.operation()) == (checked_sync == 1 ? "KVMem Program compute drain" :
                    checked_sync == 2 ? "KVMem inner prefill compute drain" : "KVMem whole bundle reset final drain");
        }
        (void)executor.memory_summary();
        require(typed_failed_drain && !executor.healthy() &&
            owner.current_view().generation == scheduling.failed_generation,
            "checked compute drain aborted/masked its typed request cause or reset sticky generation");
        if (checked_sync == 3) {
            const auto state = ProgramAccess::inspect(*instance.program);
            require(owner.frontier() == baseline.prompt.prompt_tokens &&
                owner.sparse_capture()->index().frontier() == owner.frontier() &&
                owner.sparse_capture()->index().generation() == scheduling.failed_index_generation &&
                owner.sparse_capture()->query() && owner.sparse_capture()->query()->valid() &&
                owner.sparse_capture()->query()->id() == scheduling.failed_capture_id &&
                state.text_valid == scheduling.before_reset.text_valid &&
                state.mtp_valid == scheduling.before_reset.mtp_valid &&
                state.owns_kv == scheduling.before_reset.owns_kv &&
                state.hidden_valid == scheduling.before_reset.hidden_valid &&
                state.resume_valid == scheduling.before_reset.resume_valid &&
                state.checkpoint_valid == scheduling.before_reset.checkpoint_valid &&
                state.original_query == scheduling.before_reset.original_query && !state.ledger_empty,
                "new whole-bundle reset failure falsely published clean accepted host metadata");
        }
        ExecutorAccess::with_execution_lock(executor, [&] {
            owner.drain(); // later actual successful all-lane drain cannot license reset
            instance.program->abort_lane(0);
        });
        bool unavailable = false;
        try { (void)submit(first, false); }
        catch (const RequestError& error) { unavailable = error.kind() == RequestErrorKind::Unavailable; }
        require(unavailable && owner.current_view().generation == scheduling.failed_generation,
            "a later successful drain allowed failed-sync generation reset or a new request");
        std::cout << "actual KVMem executor checked sync PASS actual_status=" <<
            static_cast<int>(scheduling.returned_status) << " placement=" << checked_sync
            << " mtp=" << mtp << " simulated_drain_placement=1\n";
        return;
    }
    if (!stronger_only) {
    ExecutorAccess::with_execution_lock(executor, [&] { scheduling.reject_in_start = true; });
    bool start_error = false;
    try { (void)run(first, false); }
    catch (const targets::qwen3_6::detail::KVMemColdResetRequired&) { start_error = true; }
    require(start_error && executor.healthy(), "actual executor start boundary did not preserve safe original failure");
    require(run(first, false).generated_token_ids == baseline.generated_token_ids,
        "actual executor start boundary did not permit a complete cold next request");
    std::cout << "actual KVMem executor start recovery PASS mtp=" << mtp << '\n';
    PromptInput next = first;
    next.messages.push_back(message("assistant", baseline.content));
    next.current_input_message = next.messages.size();
    next.messages.push_back(message("user", "Say cedar again."));
    for (const std::uint64_t rejection : {1ULL, 129ULL}) {
        PromptInput current = next;
        std::string long_text;
        for (int i = 0; i < 400; ++i) long_text += "Current cedar record.\n";
        current.messages.back().parts[0].text = long_text + "Say cedar.";
        std::optional<Executor::Submission> failed;
        std::optional<Executor::Submission> queued;
        ExecutorAccess::with_execution_lock(executor, [&] {
            failed.emplace(submit(current, true));
            ExecutorAccess::start_one_pending(executor);
            kvmem::HostKVTransferFaultInjection fault;
            fault.reject_d2h_submission_after = rejection;
            owner.set_transfer_test_fault(fault);
            // A queued ordinary request must survive the failed C1 GPU unit.
            if (rejection > 1) queued.emplace(submit(first, false));
        });
        bool original_cold_error = false;
        try { (void)failed->wait(nullptr, {}); }
        catch (const targets::qwen3_6::detail::KVMemColdResetRequired&) { original_cold_error = true; }
        require(original_cold_error && executor.healthy(),
            "real executor masked a safe original request error or terminated after complete cold cleanup");
        if (!queued) {
            ExecutorAccess::with_execution_lock(executor, [&] {
                const auto cleared = ProgramAccess::inspect(*instance.program);
                require(owner.frontier() == 0 && cleared.gdn_zero && cleared.positions_zero &&
                    cleared.mtp_tags_empty && !cleared.owns_kv && cleared.ledger_empty && !cleared.hidden_valid,
                    "executor did not settle the whole cold bundle before publishing its original failure");
            });
        }
        const auto recovered = queued ? queued->wait(nullptr, {}) : run(first, false);
        require(recovered.generated_token_ids == baseline.generated_token_ids && executor.healthy(),
            "queued/next real executor request did not complete cold with deterministic baseline output");
        (void)executor.memory_summary(); // wait for completed-slot removal and stats publication
        const auto stats = executor.runtime_stats();
        require(stats.running_requests == 0 && stats.prefilling_requests == 0 && stats.waiting_requests == 0,
            "recoverable GPU unit leaked an executor slot/transient/pending request");
        std::cout << "actual KVMem executor recovery PASS mtp=" << mtp << " reject_plane=" << rejection << '\n';
    }
    request.execution.requested_output_tokens = 64;
    std::optional<Executor::Submission> failed_decode;
    ExecutorAccess::with_execution_lock(executor, [&] {
        failed_decode.emplace(submit(first, false));
        ExecutorAccess::start_one_pending(executor);
        ExecutorAccess::finish_prefill(executor);
        kvmem::HostKVTransferFaultInjection fault;
        fault.reject_d2h_submission_after = 1;
        owner.set_transfer_test_fault(fault);
    });
    bool decode_error = false;
    try { (void)failed_decode->wait(nullptr, {}); }
    catch (const targets::qwen3_6::detail::KVMemColdResetRequired&) { decode_error = true; }
    require(decode_error && executor.healthy(),
        "real decode GPU-unit failure did not preserve the request error and reusable executor");
    request.execution.requested_output_tokens = 9;
    require(run(first, false).generated_token_ids == baseline.generated_token_ids,
        "real executor decode failure did not leave a complete cold next request");
    std::cout << "actual KVMem executor decode recovery PASS mtp=" << mtp << '\n';
    }

    // A fatal Q drain encountered during cancellation/its next GPU boundary is
    // handed off from noexcept target cleanup to the failed request, never success.
    unsigned drains = 0;
    unsigned generic_drains = 0;
    std::optional<kvmem::QueryCaptureBorrow> generic_borrower;
    std::optional<kvmem::QueryCaptureBorrow> borrower;
    request.execution.requested_output_tokens = 64;
    std::optional<Executor::Submission> failed_cancel;
    ExecutorAccess::with_execution_lock(executor, [&] {
        failed_cancel.emplace(submit(first, false));
        ExecutorAccess::start_one_pending(executor);
        ExecutorAccess::finish_prefill(executor);
        generic_borrower.emplace(owner.sparse_capture()->borrow_query([&] {
            if (++generic_drains == 1) throw std::runtime_error("executor earlier generic Q cleanup");
        }));
        borrower.emplace(owner.sparse_capture()->borrow_query([&] {
            if (++drains == 1) throw kvmem::CudaTransferError(cudaErrorIllegalAddress,
                "executor test fatal Q cleanup", false);
        }));
        ExecutorAccess::cancel_active(executor);
    });
    bool fatal_original = false;
    try { (void)failed_cancel->wait(nullptr, {}); }
    catch (const kvmem::CudaTransferError& error) { fatal_original = error.status() == cudaErrorIllegalAddress; }
    (void)executor.memory_summary(); // wait for worker terminal cleanup before inspecting health
    require(fatal_original && !executor.healthy(),
        "fatal target cleanup reported cancellation/success or left public executor health reusable");
    const auto generation = owner.current_view().generation;
    ExecutorAccess::with_execution_lock(executor, [&] { owner.sparse_capture()->drain(); });
    require(drains >= 2 && generic_drains >= 2 && borrower && !borrower->valid() &&
        generic_borrower && !generic_borrower->valid(), "later successful fatal borrower drain missing");
    bool sticky_typed = false;
    try { owner.sparse_capture()->throw_if_unrecoverable(); }
    catch (const kvmem::CudaTransferError& error) {
        sticky_typed = error.status() == cudaErrorIllegalAddress && !error.drain_failed() &&
            std::string_view(error.operation()) == "executor test fatal Q cleanup";
    }
    require(sticky_typed, "later successful all-borrower drain cleared or masked first typed fatal cause");
    bool unavailable = false;
    try { (void)submit(first, false); }
    catch (const RequestError& error) { unavailable = error.kind() == RequestErrorKind::Unavailable; }
    require(unavailable && owner.current_view().generation == generation,
        "fatal real executor allowed a later request or reset sticky owner metadata");
    std::cout << "actual KVMem executor fatal cleanup PASS mtp=" << mtp << '\n';
}
}
int main(int argc, char** argv) {
    const auto artifact = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    if (!artifact || !*artifact) return 77;
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--only-stronger") {
            exercise_executor_recovery(artifact, false, 0, false, false, true);
            exercise_executor_recovery(artifact, true, 0, false, false, true);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--only-query") {
            exercise_executor_recovery(artifact, false, 0, false, true);
            exercise_executor_recovery(artifact, true, 0, false, true);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--only-late") {
            exercise_executor_recovery(artifact, false, false, true);
            exercise_executor_recovery(artifact, true, false, true);
            return 0;
        }
        const bool executor_only = argc == 2 && std::string_view(argv[1]) == "--only-executor";
        if (!executor_only) {
            exercise(artifact, false);
            exercise(artifact, true);
            exercise_failure_boundary(artifact, false);
            exercise_failure_boundary(artifact, true);
        }
        exercise_executor_recovery(artifact, false);
        exercise_executor_recovery(artifact, true);
        exercise_executor_recovery(artifact, false, true);
        exercise_executor_recovery(artifact, false, 2);
        exercise_executor_recovery(artifact, false, 3);
        exercise_executor_recovery(artifact, true, 3);
        exercise_executor_recovery(artifact, false, false, true);
        exercise_executor_recovery(artifact, true, false, true);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

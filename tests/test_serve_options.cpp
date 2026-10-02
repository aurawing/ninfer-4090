#include "serve/serve_options.h"
#include "serve/translate.h"

#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

std::optional<ServeOptions> accepts(std::vector<std::string> arguments) {
    try { return parse(std::move(arguments)); }
    catch (const std::invalid_argument&) { return std::nullopt; }
}

bool rejects_vision_options(const std::vector<std::string>& flags) {
    std::vector<std::string> arguments{"ninfer-serve", "model.ninfer"};
    arguments.insert(arguments.end(), flags.begin(), flags.end());
    try {
        (void)parse(std::move(arguments));
    } catch (const std::invalid_argument&) { return true; }
    std::cerr << "accepted Vision flags:";
    for (const std::string& flag : flags) { std::cerr << ' ' << flag; }
    std::cerr << '\n';
    return false;
}

} // namespace

int main() {
    int failures = 0;
    const auto mtp_window=accepts({"ninfer-serve","model.ninfer","--kvmem-mtp-window","8192"});
    failures += check(mtp_window && mtp_window->kvmem.mtp_window_tokens==8192,"serve MTP window option lost");
    const auto named_tiered = parse({"ninfer-serve", "model.ninfer", "--kv-mode", "tiered-exact",
                                    "--kv-dtype", "int8", "--kvmem-view-tokens", "16384",
                                    "--kvmem-sink-tokens", "0"});
    failures += check(named_tiered.kv_mode == ninfer::KvMode::TieredExact &&
                          named_tiered.kvmem.view_tokens == 16384 && named_tiered.kvmem.sink_tokens == 0,
                      "server must preserve documented tiered view/sink options");
    const auto tiered = accepts({"ninfer-serve", "model.ninfer", "--kv-mode", "tiered-exact",
                                 "--kv-dtype", "int8", "--kvmem-view", "8192",
                                 "--kvmem-host-archive", "pinned"});
    failures += check(tiered && tiered->kv_mode == ninfer::KvMode::TieredExact &&
                          tiered->kvmem.view_tokens == 8192 &&
                          tiered->kvmem.host_archive == ninfer::HostKVArchiveMode::Pinned,
                      "server tiered configuration was not preserved");
    failures += check(!accepts({"ninfer-serve", "model.ninfer", "--kv-mode", "invalid"}) &&
                          !accepts({"ninfer-serve", "model.ninfer", "--kvmem-host-archive", "invalid"}),
                      "server accepted invalid tiered configuration");

    const ServeOptions defaults = parse({"ninfer-serve", "model.ninfer"});
    failures += check(defaults.vision_cpu_cache_mib == 128, "server CPU cache default mismatch");
    failures += check(parse({"ninfer-serve", "model.ninfer", "--vision-mmproj", "mmproj.gguf", "--vision-cpu-cache-mib", "0"}).vision_cpu_cache_mib == 0,
                      "server CPU cache zero must disable caching");
    failures += check(parse({"ninfer-serve", "model.ninfer", "--vision-cpu-cache-mib", "64", "--vision-mmproj", "mmproj.gguf"}).vision_cpu_cache_mib == 64,
                      "server CPU cache parsing depends on argument order");
    for (const std::string& value : {std::string("-1"), std::string("4294967296"), std::string("+1"), std::string(" 1"), std::string("1x"), std::string("")}) {
        failures += check(rejects_vision_options({"--vision-mmproj", "mmproj.gguf", "--vision-cpu-cache-mib", value}), "invalid server CPU cache capacity accepted");
    }
    failures += check(rejects_vision_options({"--vision-device", "cuda", "--vision-cpu-cache-mib", "128"}) &&
                          rejects_vision_options({"--vision-cpu-cache-mib", "0"}), "server CUDA accepted CPU cache tuning");
    failures += check(serve_usage_text("ninfer-serve").find("--vision-cpu-cache-mib") != std::string::npos,
                      "server help omits CPU cache option");
    failures += check(defaults.allow_prefix_reuse, "prefix reuse is not enabled by default");
    failures +=
        check(!defaults.preserve_thinking, "thinking history is unexpectedly preserved by default");
    failures += check(!defaults.enable_vision, "Vision is not disabled by default");
    failures += check(defaults.vision_device == ninfer::VisionDevice::Cuda &&
                          defaults.vision_mmproj_path.empty() &&
                          defaults.vision_cpu_threads == 6 &&
                          defaults.vision_cpu_memory_mib == 4096,
                      "default Vision device, projector, or CPU limits mismatch");

    for (const std::vector<std::string>& flags :
         std::vector<std::vector<std::string>>{{"--vision"},
                                               {"--vision-max-tokens", "1024"},
                                               {"--vision-limit", "1024"},
                                               {"--vision-device", "cuda"},
                                               {"--vision-device", "cuda", "--vision-device", "cuda"}}) {
        std::vector<std::string> arguments{"ninfer-serve", "model.ninfer"};
        arguments.insert(arguments.end(), flags.begin(), flags.end());
        const ServeOptions vision_cuda = parse(std::move(arguments));
        failures += check(vision_cuda.enable_vision &&
                              vision_cuda.vision_device == ninfer::VisionDevice::Cuda &&
                              vision_cuda.vision_mmproj_path.empty(),
                          "existing Vision flags did not preserve the CUDA default");
    }

    const ServeOptions vision_shortcut =
        parse({"ninfer-serve", "model.ninfer", "--vision-mmproj", "mmproj-BF16.gguf"});
    failures += check(vision_shortcut.enable_vision &&
                          vision_shortcut.vision_device == ninfer::VisionDevice::Cpu &&
                          vision_shortcut.vision_mmproj_path == "mmproj-BF16.gguf",
                      "--vision-mmproj did not enable CPU Vision");
    const auto embedded_cpu =
        accepts({"ninfer-serve", "model.ninfer", "--vision-device", "cpu"});
    failures += check(embedded_cpu && embedded_cpu->enable_vision &&
                          embedded_cpu->vision_device == ninfer::VisionDevice::Cpu &&
                          embedded_cpu->vision_mmproj_path.empty(),
                      "server rejected embedded CPU Vision without external mmproj");
    const auto embedded_cpu_tuned =
        accepts({"ninfer-serve", "model.ninfer", "--vision-cpu-threads", "8",
                 "--vision-device", "cpu"});
    failures += check(embedded_cpu_tuned && embedded_cpu_tuned->enable_vision &&
                          embedded_cpu_tuned->vision_device == ninfer::VisionDevice::Cpu &&
                          embedded_cpu_tuned->vision_cpu_threads == 8 &&
                          embedded_cpu_tuned->vision_mmproj_path.empty(),
                      "server rejected pathless CPU Vision tuning");
    const auto external_gpu =
        accepts({"ninfer-serve", "model.ninfer", "--vision-device", "cuda",
                 "--vision-mmproj", "mmproj-BF16.gguf"});
    failures += check(external_gpu && external_gpu->enable_vision &&
                          external_gpu->vision_device == ninfer::VisionDevice::Cuda &&
                          external_gpu->vision_mmproj_path == "mmproj-BF16.gguf",
                      "server rejected external GGUF on explicitly selected GPU");
    const auto external_gpu_reversed =
        accepts({"ninfer-serve", "model.ninfer", "--vision-mmproj", "mmproj-BF16.gguf",
                 "--vision-device", "cuda"});
    failures += check(external_gpu_reversed && external_gpu_reversed->enable_vision &&
                          external_gpu_reversed->vision_device == ninfer::VisionDevice::Cuda &&
                          external_gpu_reversed->vision_mmproj_path == "mmproj-BF16.gguf",
                      "server external GGUF GPU selection depends on option order");
    for (const char* next_flag : {"--vision", "--vision-device", "--vision-max-tokens",
                                  "--api-key", "-h"}) {
        bool missing_path_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--vision-mmproj", next_flag});
        } catch (const std::invalid_argument& error) {
            const std::string message = error.what();
            missing_path_rejected = message.find("--vision-mmproj") != std::string::npos &&
                                    message.find("path") != std::string::npos;
        }
        failures += check(missing_path_rejected,
                          "--vision-mmproj consumed the following option as its path");
    }
    for (const std::vector<std::string>& flags : std::vector<std::vector<std::string>>{
             {"--vision-device", "cpu", "--vision-mmproj", "mmproj-BF16.gguf",
              "--vision-cpu-threads", "8", "--vision-cpu-memory-mib", "2048", "--vision"},
             {"--vision-cpu-memory-mib", "2048", "--vision-cpu-threads", "8", "--vision",
              "--vision-mmproj", "mmproj-BF16.gguf", "--vision-device", "cpu"},
             {"--vision-mmproj", "mmproj-BF16.gguf", "--vision-cpu-threads", "8",
              "--vision-cpu-memory-mib", "2048", "--vision-max-tokens", "1024"},
             {"--vision-max-tokens", "1024", "--vision-cpu-memory-mib", "2048",
              "--vision-cpu-threads", "8", "--vision-mmproj", "mmproj-BF16.gguf"},
             {"--vision-device", "cpu", "--vision-device", "cpu", "--vision-mmproj", "mmproj-BF16.gguf",
              "--vision-cpu-threads", "8", "--vision-cpu-memory-mib", "2048"}}) {
        std::vector<std::string> arguments{"ninfer-serve", "model.ninfer"};
        arguments.insert(arguments.end(), flags.begin(), flags.end());
        const ServeOptions vision_cpu = parse(std::move(arguments));
        failures += check(vision_cpu.enable_vision &&
                              vision_cpu.vision_device == ninfer::VisionDevice::Cpu &&
                              vision_cpu.vision_mmproj_path == "mmproj-BF16.gguf" &&
                              vision_cpu.vision_cpu_threads == 8 &&
                              vision_cpu.vision_cpu_memory_mib == 2048,
                          "CPU Vision options depend on argument order");
    }
    for (const std::vector<std::string>& flags : std::vector<std::vector<std::string>>{
             {"--vision-device", "gpu"}, {"--vision-device", ""},
             {"--vision-mmproj", ""},
             {"--vision-device", "cpu", "--vision-mmproj", "--vision"},
             {"--vision-mmproj"}, {"--vision-device"},
             {"--vision-cpu-threads"}, {"--vision-cpu-memory-mib"},
             {"--vision-device", "cuda", "--vision-device", "cpu", "--vision-mmproj", "mmproj.gguf"},
             {"--vision-device", "cpu", "--vision-device", "cuda", "--vision-mmproj", "mmproj.gguf"},
             {"--vision-device", "cuda", "--vision-cpu-threads", "6"},
             {"--vision-cpu-threads", "6", "--vision-device", "cuda"},
             {"--vision-device", "cuda", "--vision-cpu-memory-mib", "4096"},
             {"--vision-cpu-memory-mib", "4096", "--vision-device", "cuda"},
             {"--vision-cpu-threads", "6"}, {"--vision-cpu-memory-mib", "4096"},
             {"--spec", "dflash", "--draft-tokens", "3", "--vision-mmproj", "mmproj.gguf"}}) {
        failures += check(rejects_vision_options(flags), "invalid CPU Vision combination was accepted");
    }
    for (const std::string& flag : {std::string("--vision-cpu-threads"),
                                    std::string("--vision-cpu-memory-mib")}) {
        for (const std::string& value : {std::string("0"), std::string("-1"),
                                         std::string("4294967296"),
                                         std::string("18446744073709551616"),
                                         std::string(" -18446744073709551615"),
                                         std::string("\t-18446744073709551615"),
                                         std::string("+1"), std::string(" 1"),
                                         std::string("1 "), std::string("1x"),
                                         std::string("invalid"), std::string("")}) {
            failures += check(rejects_vision_options({"--vision-mmproj", "mmproj.gguf", flag, value}),
                              "invalid CPU Vision unsigned limit was accepted");
        }
    }
    const ServeOptions decimal_limits =
        parse({"ninfer-serve", "model.ninfer", "--vision-mmproj", "mmproj.gguf",
               "--vision-cpu-threads", "001", "--vision-cpu-memory-mib", "0001"});
    failures += check(decimal_limits.vision_cpu_threads == 1 && decimal_limits.vision_cpu_memory_mib == 1,
                      "CPU Vision rejected decimal digits with leading zeros");
    failures += check(rejects_vision_options({"--vision-mmproj", "mmproj.gguf",
                                             "--vision-cpu-threads", "513"}),
                      "CPU Vision thread count exceeded the GGML threadpool limit");
    const ServeOptions vision_limits =
        parse({"ninfer-serve", "model.ninfer", "--vision-mmproj", "mmproj.gguf",
               "--vision-cpu-threads", "512", "--vision-cpu-memory-mib", "4294967295"});
    failures += check(vision_limits.vision_cpu_threads == ninfer::kMaximumVisionCpuThreads &&
                          vision_limits.vision_cpu_memory_mib == 4294967295U,
                      "CPU Vision parser did not preserve the thread or memory limit");
    for (const char* flag : {"--vision-device", "--vision-mmproj", "--vision-cpu-threads",
                             "--vision-cpu-memory-mib"}) {
        failures += check(serve_usage_text("ninfer-serve").find(flag) != std::string::npos,
                          "serve help omits a CPU Vision option");
    }
    failures += check(defaults.request_log_jsonl.empty(),
                      "request JSONL logging is not disabled by default");
    failures += check(defaults.log_stats_interval_ms == 5000,
                      "periodic throughput interval default mismatch");
    failures += check(defaults.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          defaults.kv_capacity.explicit_tokens == defaults.max_context,
                      "default KV capacity does not follow max context");
    failures += check(defaults.speculative.backend == ninfer::SpeculativeBackend::None,
                      "speculative decoding is not disabled by default");
    failures += check(defaults.response_store_max_records == kDefaultResponseStoreRecords &&
                          defaults.response_store_max_bytes == kDefaultResponseStoreBytes,
                      "Responses store defaults mismatch");
    failures += check(!defaults.model_id_override.has_value(),
                      "model id override is unexpectedly configured by default");
    failures += check(!defaults.default_reasoning_effort.has_value(),
                      "reasoning effort is unexpectedly configured by default");
    failures += check(
        !defaults.sampling_overrides.temperature && !defaults.sampling_overrides.top_p &&
            !defaults.sampling_overrides.top_k && !defaults.sampling_overrides.presence_penalty &&
            !defaults.sampling_overrides.frequency_penalty,
        "server defaults unexpectedly override registered model sampling");
    failures += check(!defaults.wddm_evictable_budget,
                      "wddm_evictable_budget is unexpectedly enabled by default");
    failures += check(resolve_public_model_id(defaults, "artifact-model") == "artifact-model",
                      "artifact model id was not selected by default");

    const ServeOptions wddm_opt_in =
        parse({"ninfer-serve", "model.ninfer", "--wddm-evictable-budget"});
    failures += check(wddm_opt_in.wddm_evictable_budget,
                      "--wddm-evictable-budget was not parsed correctly");

    const ServeOptions ui_disabled =
        parse({"ninfer-serve", "model.ninfer", "--no-ui"});
    failures += check(!ui_disabled.enable_ui, "--no-ui did not disable WebUI");

    const ServeOptions ui_enabled =
        parse({"ninfer-serve", "model.ninfer", "--ui"});
    failures += check(ui_enabled.enable_ui, "--ui did not enable WebUI");

    const ServeOptions rotor =
        parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "rk8v4"});
    failures += check(
        rotor.kv_cache == ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64,
        "--kv-dtype rk8v4 did not select rotated K8/V4 storage");
    failures += check(defaults.kv_cache == ninfer::KvCacheStorage::BFloat16,
                      "rk8v4 unexpectedly changed the default KV storage");

    const ServeOptions k4e8 =
        parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "rk4v4-e8"});
    failures += check(
        k4e8.kv_cache == ninfer::KvCacheStorage::RK4V4E8,
        "--kv-dtype rk4v4-e8 did not select RK4V4E8 storage");

    const ServeOptions k2e8 =
        parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "rk2v4-e8"});
    failures += check(
        k2e8.kv_cache == ninfer::KvCacheStorage::RK2V4E8,
        "--kv-dtype rk2v4-e8 did not select RK2V4E8 storage");

    const ServeOptions model_alias =
        parse({"ninfer-serve", "model.ninfer", "--model-id", "deployment-alias"});
    failures +=
        check(model_alias.model_id_override == "deployment-alias" &&
                  resolve_public_model_id(model_alias, "artifact-model") == "deployment-alias",
              "explicit model id did not override the artifact identity");

    bool empty_model_id_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--model-id", ""});
    } catch (const std::invalid_argument&) { empty_model_id_rejected = true; }
    failures += check(empty_model_id_rejected, "empty --model-id was accepted");

    const ServeOptions dflash = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash",
                                       "--draft-tokens", "15", "--lm-head-draft"});
    failures += check(dflash.speculative.backend == ninfer::SpeculativeBackend::DFlash,
                      "--spec dflash did not select DFlash");
    failures += check(dflash.speculative.draft_tokens == 15,
                      "--draft-tokens did not preserve the DFlash window");
    failures += check(dflash.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                      "--lm-head-draft did not select the optimized proposal head");

    bool dflash_vision_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--spec", "dflash", "--draft-tokens", "15",
                     "--vision"});
    } catch (const std::invalid_argument&) { dflash_vision_rejected = true; }
    failures += check(dflash_vision_rejected, "DFlash and Vision were accepted together");

    bool implicit_backend_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--draft-tokens", "3"});
    } catch (const std::invalid_argument&) { implicit_backend_rejected = true; }
    failures += check(implicit_backend_rejected, "--draft-tokens selected a backend implicitly");

    const ServeOptions configured = parse(
        {"ninfer-serve", "model.ninfer", "--no-prefix-reuse", "--vision", "--max-concurrency", "4",
         "--max-pending-requests", "12", "--pending-timeout-ms", "2500", "--max-context", "4096",
         "--kv-capacity", "8192", "--log-stats-interval-ms", "0", "--preserve-thinking"});
    failures += check(!configured.allow_prefix_reuse,
                      "--no-prefix-reuse did not disable server prefix reuse");
    failures += check(configured.enable_vision, "--vision did not enable Vision");
    failures +=
        check(configured.preserve_thinking, "--preserve-thinking did not reach serving options");
    failures +=
        check(configured.max_concurrency == 4, "--max-concurrency did not reach serving options");
    failures += check(configured.max_context == 4096 &&
                          configured.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          configured.kv_capacity.explicit_tokens == 8192,
                      "context and KV capacity options were not kept distinct");
    failures += check(configured.max_pending_requests == 12,
                      "--max-pending-requests did not reach serving options");
    failures += check(configured.pending_timeout_ms == 2500,
                      "--pending-timeout-ms did not reach serving options");
    failures += check(configured.log_stats_interval_ms == 0,
                      "--log-stats-interval-ms did not disable periodic reporting");

    const ServeOptions response_store =
        parse({"ninfer-serve", "model.ninfer", "--response-store-max-records", "42",
               "--response-store-max-mib", "8"});
    failures += check(response_store.response_store_max_records == 42 &&
                          response_store.response_store_max_bytes == (8ULL << 20),
                      "Responses store limits did not reach serving options");

    const ServeOptions sampling =
        parse({"ninfer-serve", "model.ninfer", "--temperature", "0", "--top-p", "0.9", "--top-k",
               "40", "--min-p", "0.1", "--presence-penalty", "1.25", "--frequency-penalty", "-0.5",
               "--seed", "0"});
    failures += check(sampling.sampling_overrides.temperature == 0.0F &&
                          sampling.sampling_overrides.top_p == 0.9F &&
                          sampling.sampling_overrides.top_k == 40 &&
                          sampling.sampling_overrides.min_p == 0.1F &&
                          sampling.sampling_overrides.presence_penalty == 1.25F &&
                          sampling.sampling_overrides.frequency_penalty == -0.5F &&
                          sampling.sampling_overrides.seed == 0,
                      "server sampling flags did not preserve explicit values and zeros");

    GenerationRequest request;
    request.max_tokens = 1;
    ninfer::PromptCapabilities prompt_capabilities;
    prompt_capabilities.enable_thinking = true;
    failures += check(to_request_options(request, defaults).execution.allow_prefix_reuse,
                      "default server policy did not reach Engine options");
    failures += check(!to_request_options(request, configured).execution.allow_prefix_reuse,
                      "disabled server policy did not reach Engine options");
    const ninfer::RequestOptions inherited_sampling = to_request_options(request, sampling);
    failures += check(inherited_sampling.execution.sampling.temperature == 0.0F &&
                          inherited_sampling.execution.sampling.top_p == 0.9F &&
                          inherited_sampling.execution.sampling.seed == 0,
                      "server sampling overrides did not reach Engine options");
    request.sampling.temperature = 1.1;
    failures += check(to_request_options(request, sampling).execution.sampling.temperature == 1.1F,
                      "request sampling override did not win over the server override");
    failures +=
        check(resolve_prompt_semantics(request, configured, prompt_capabilities).preserve_thinking,
              "server preserve-thinking default was not resolved");
    request.preserve_thinking = false;
    failures +=
        check(!resolve_prompt_semantics(request, configured, prompt_capabilities).preserve_thinking,
              "request preserve-thinking override did not win");

    const ServeOptions effort_low =
        parse({"ninfer-serve", "model.ninfer", "--reasoning-effort", "low"});
    failures += check(effort_low.default_reasoning_effort == RequestedReasoningEffort::Low,
                      "--reasoning-effort low was not parsed");

    const ServeOptions effort_med =
        parse({"ninfer-serve", "model.ninfer", "--reasoning-effort", "medium"});
    failures += check(effort_med.default_reasoning_effort == RequestedReasoningEffort::Medium,
                      "--reasoning-effort medium was not parsed");

    const ServeOptions effort_xhigh =
        parse({"ninfer-serve", "model.ninfer", "--reasoning-effort", "xhigh"});
    failures += check(effort_xhigh.default_reasoning_effort == RequestedReasoningEffort::XHigh,
                      "--reasoning-effort xhigh was not parsed");

    const ServeOptions thinking_effort_alias =
        parse({"ninfer-serve", "model.ninfer", "--thinking-effort", "low"});
    failures += check(thinking_effort_alias.default_reasoning_effort == RequestedReasoningEffort::Low,
                      "--thinking-effort alias was not parsed");

    bool invalid_effort_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--reasoning-effort", "ultra"});
    } catch (const std::invalid_argument&) { invalid_effort_rejected = true; }
    failures += check(invalid_effort_rejected, "invalid --reasoning-effort value was accepted");

    // Test default reasoning effort resolution in semantics
    ninfer::PromptCapabilities effort_caps;
    effort_caps.enable_thinking  = true;
    effort_caps.reasoning_effort = ninfer::ReasoningEffortCapabilities{
        .low            = true,
        .medium         = true,
        .xhigh          = true,
        .default_effort = ninfer::ReasoningEffort::XHigh,
    };
    GenerationRequest effort_req;
    failures += check(
        resolve_prompt_semantics(effort_req, effort_low, effort_caps).reasoning_effort ==
            ninfer::ReasoningEffort::Low,
        "server default reasoning effort (low) was not applied to request omitting effort");

    // Client explicit reasoning effort overrides server default
    effort_req.reasoning_effort = RequestedReasoningEffort::XHigh;
    failures += check(
        resolve_prompt_semantics(effort_req, effort_low, effort_caps).reasoning_effort ==
            ninfer::ReasoningEffort::XHigh,
        "client explicit reasoning effort did not override server default");

    // When client disables thinking, reasoning effort is not applied
    effort_req.reasoning_effort = std::nullopt;
    effort_req.enable_thinking  = false;
    const ResolvedPromptSemantics non_thinking_sem =
        resolve_prompt_semantics(effort_req, effort_low, effort_caps);
    failures += check(!non_thinking_sem.enable_thinking && !non_thinking_sem.reasoning_effort.has_value(),
                      "server default reasoning effort conflicted with client disabled thinking");

    failures +=
        check(serve_usage_text("ninfer-serve").find("--no-prefix-reuse") != std::string::npos,
              "serve help omits --no-prefix-reuse");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--preserve-thinking") != std::string::npos,
              "serve help omits --preserve-thinking");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--reasoning-effort") != std::string::npos,
              "serve help omits --reasoning-effort");
    failures += check(serve_usage_text("ninfer-serve").find("--vision") != std::string::npos,
                      "serve help omits --vision");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--log-stats-interval-ms") != std::string::npos,
              "serve help omits --log-stats-interval-ms");
    failures += check(serve_usage_text("ninfer-serve").find("--kv-capacity") != std::string::npos,
                      "serve help omits --kv-capacity");
    failures += check(serve_usage_text("ninfer-serve").find("--response-store-max-mib") !=
                          std::string::npos,
                      "serve help omits Responses store limits");
    failures +=
        check(serve_usage_text("ninfer-serve").find("identity.model_id") != std::string::npos,
              "serve help omits the artifact-derived model id default");

    failures += check(defaults.default_max_tokens == static_cast<int>(defaults.max_context),
                      "default_max_tokens did not default to max_context");

    const ServeOptions inherited =
        parse({"ninfer-serve", "model.ninfer", "--max-context", "16384"});
    failures += check(inherited.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          inherited.kv_capacity.explicit_tokens == 16384,
                      "omitted --kv-capacity did not follow --max-context");
    failures += check(inherited.default_max_tokens == 16384,
                      "omitted --default-max-tokens did not follow --max-context");

    const ServeOptions explicit_max_tokens =
        parse({"ninfer-serve", "model.ninfer", "--max-context", "16384", "--default-max-tokens", "2048"});
    failures += check(explicit_max_tokens.default_max_tokens == 2048,
                      "explicit --default-max-tokens was overridden by --max-context");

    const ServeOptions automatic = parse({"ninfer-serve", "model.ninfer", "--kv-capacity", "auto"});
    failures += check(automatic.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          automatic.kv_capacity.explicit_tokens == 0 &&
                          automatic.kv_capacity.automatic_headroom_bytes ==
                              ninfer::kDefaultKvCapacityHeadroomBytes,
                      "--kv-capacity auto did not select automatic sizing");

    const ServeOptions logged = parse({"ninfer-serve", "model.ninfer", "--request-log-jsonl",
                                       "requests.jsonl", "--api-key", "do-not-log"});
    failures += check(logged.request_log_jsonl == "requests.jsonl",
                      "--request-log-jsonl did not preserve its path");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--request-log-jsonl") != std::string::npos,
              "serve help omits --request-log-jsonl");
    bool secret_present    = false;
    bool redaction_present = false;
    for (const std::string& argument : logged.startup_argv) {
        secret_present    = secret_present || argument == "do-not-log";
        redaction_present = redaction_present || argument == "<redacted>";
    }
    failures += check(!secret_present, "startup argv retained the API key");
    failures += check(redaction_present, "startup argv omitted the API-key redaction marker");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}

#include <ninfer/engine.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

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
void exercise(const char* artifact, HostKVArchiveMode archive, bool mtp) {
    EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 4096;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(4096);
    options.kv_cache = KvCacheStorage::Int8Group64;
    options.kv_mode = KvMode::TieredExact;
    options.kvmem.view_tokens = 256;
    options.kvmem.sink_tokens = 64;
    options.kvmem.mtp_window_tokens = 512;
    options.kvmem.host_archive = archive;
    options.prefill_chunk = 128;
    options.enable_vision = false;
    options.use_cuda_graph = false;
    options.speculative.backend = mtp ? SpeculativeBackend::Mtp : SpeculativeBackend::None;
    options.speculative.draft_tokens = mtp ? 3 : 0;
    options.speculative.proposal_head = mtp ? ProposalHead::Optimized : ProposalHead::Full;
    Engine engine(options);
    PromptInput first;
    first.options.enable_thinking = false;
    first.options.preserve_thinking = true;
    std::string text = "A fictional note: the secret word is maple.\n";
    for (int i = 0; i < 400; ++i) text += "ordinary filler ";
    text += "\nReply with the secret word, then list numbers 1 through 100.";
    first.messages.push_back(message("user", text));
    auto run = [&](const PromptInput& input, bool reuse) {
        RequestOptions request;
        request.execution.requested_output_tokens = 16;
        request.execution.sampling.temperature = 0.0F;
        request.execution.allow_prefix_reuse = reuse;
        request.stop.include_model_defaults = false;
        request.output.raw = true;
        request.output.preserve_special_tokens = true;
        const auto before = engine.runtime_stats();
        auto result = engine.generate(engine.prepare(input), request);
        const auto computed = engine.runtime_stats().computed_prefill_tokens - before.computed_prefill_tokens;
        require(computed == result.prompt.prompt_tokens - result.reused_prompt_tokens,
                "reuse must compute only new prompt suffix tokens");
        return result;
    };
    const auto initial = run(first, false);
    PromptInput second = first;
    second.messages.push_back(message("assistant", initial.content));
    second.messages.push_back(message("user", "Repeat the secret word, then list numbers 1 through 20."));
    const auto appended = run(second, true);
    require(appended.prefix_reuse_path == PrefixReusePath::AppendAtFrontier && appended.reused_prompt_tokens > 0,
            "tiered retained resume must append at the saved frontier");
    const auto restored = run(second, true);
    require(restored.prefix_reuse_path == PrefixReusePath::RestoreTurnCheckpoint && restored.reused_prompt_tokens > 0,
            "tiered same-turn retry must restore the captured checkpoint");
    const auto repeated = run(second, true);
    require(repeated.prefix_reuse_path == PrefixReusePath::RestoreTurnCheckpoint,
            "turn checkpoint must remain valid through repeated restore");
    const auto cold = run(second, false);
    require(restored.generated_token_ids == cold.generated_token_ids &&
                repeated.generated_token_ids == cold.generated_token_ids,
            "short checkpoint rollback must preserve cold greedy output");
    RequestOptions single;
    single.execution.requested_output_tokens = 1;
    single.execution.sampling.temperature = 0.0F;
    single.stop.include_model_defaults = false;
    const std::vector<TokenId> short_prompt{248045, 846, 198, 5834, 248046, 198};
    const auto short_first = engine.generate(engine.prepare_tokens(short_prompt), single);
    const auto exact = engine.generate(engine.prepare_tokens(short_prompt), single);
    require(exact.prefix_reuse_path == PrefixReusePath::AppendAtFrontier &&
                exact.reused_prompt_tokens == short_prompt.size() &&
                exact.generated_token_ids == short_first.generated_token_ids,
            "terminal one-token request must retain short-context exact-hit continuation");
    std::cout << "resume/checkpoint archive=" << (archive == HostKVArchiveMode::Pinned ? "pinned" : "pageable")
              << " mtp=" << mtp << " appended_reuse=" << appended.reused_prompt_tokens
              << " checkpoint_reuse=" << restored.reused_prompt_tokens << '\n';
}
}
int main() {
    const auto* artifact = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    if (!artifact || !*artifact) {
        std::cout << "SKIP: NINFER_QWEN3_8_27B_WEIGHTS is not set\n";
        return 77;
    }
    try {
        exercise(artifact, HostKVArchiveMode::Pageable, true);
        exercise(artifact, HostKVArchiveMode::Pinned, true);
        exercise(artifact, HostKVArchiveMode::Pageable, false);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

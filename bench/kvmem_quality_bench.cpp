// Text-only local synthetic measurement through the public Engine API.
#include <ninfer/engine.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

using nlohmann::json;
using namespace ninfer;

static PromptInput input_of(const json& fixture) {
    PromptInput input;
    input.options.enable_thinking = false;
    input.options.preserve_thinking = true;
    const auto form = fixture.at("form").get<std::string>();
    const auto boundary = fixture.at("current_input_message").get<std::size_t>();
    if ((form != "single" && form != "historical") ||
        boundary != (form == "single" ? 0 : 2) ||
        fixture.at("messages").size() != (form == "single" ? 1 : 3))
        throw std::invalid_argument("frozen input form/boundary mismatch");
    input.current_input_message = boundary;
    for (const auto& source : fixture.at("messages")) {
        ChatMessage message;
        message.role = source.at("role").get<std::string>();
        MessagePart part;
        part.kind = MessagePartKind::Text;
        part.text = source.at("content").get<std::string>();
        message.parts.push_back(std::move(part));
        input.messages.push_back(std::move(message));
    }
    if (input.messages[0].role != "user" || input.messages.back().role != "user" ||
        (form == "historical" && input.messages[1].role != "assistant"))
        throw std::invalid_argument("frozen roles mismatch");
    return input;
}

static void set_filler(json& fixture, std::uint32_t lines) {
    auto& recipe = fixture.at("recipe");
    recipe["filler_lines"] = lines;
    std::string document = recipe.at("prefix").get<std::string>();
    const auto& units = recipe.at("filler_units");
    if (units.empty()) throw std::invalid_argument("empty neutral background pool");
    const auto& injections = recipe.at("injections");
    std::uint32_t previous = 0;
    for (std::size_t index = 0; index < injections.size(); ++index) {
        const auto position = static_cast<std::uint32_t>((index + 1) * lines / (injections.size() + 1));
        for (; previous < position; ++previous) document += units[previous % units.size()].get<std::string>();
        document += injections[index].get<std::string>() + "\n";
    }
    for (; previous < lines; ++previous) document += units[previous % units.size()].get<std::string>();
    document += recipe.at("suffix").get<std::string>();
    if (fixture.at("form") == "single") document += "\n\nQUESTION\n" + fixture.at("question").get<std::string>();
    fixture["messages"][0]["content"] = std::move(document);
}

static const char* reuse_name(PrefixReusePath value) {
    switch (value) {
        case PrefixReusePath::FullReset: return "FullReset";
        case PrefixReusePath::AppendAtFrontier: return "AppendAtFrontier";
        case PrefixReusePath::RestoreTurnCheckpoint: return "RestoreTurnCheckpoint";
        case PrefixReusePath::RestoreDiskCheckpoint: return "RestoreDiskCheckpoint";
    }
    throw std::logic_error("unknown public reuse path");
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "ninfer_kvmem_quality_bench REQUEST.json NEW_OUTPUT.jsonl\n"
                      << "operations: render (CPU parity), fit (neutral filler only), count, generate; public Engine, C=1, MTP3\n";
            return 0;
        }
        if (argc != 3) throw std::invalid_argument("usage: ninfer_kvmem_quality_bench REQUEST.json NEW_OUTPUT.jsonl");
        if (std::filesystem::exists(argv[2])) throw std::invalid_argument("output exists; inference retry/overwrite forbidden");
        std::ifstream source(argv[1]);
        if (!source) throw std::invalid_argument("request missing");
        const auto request = json::parse(source);
        const auto operation = request.at("operation").get<std::string>();
        if (operation != "generate" && operation != "count" && operation != "fit" && operation != "render")
            throw std::invalid_argument("unknown operation");
        // CPU-only oracle boundary: prove C++/Python neutral stream parity before any Engine load.
        if (operation == "render") {
            std::ofstream output(argv[2], std::ios::out | std::ios::binary);
            if (!output) throw std::runtime_error("cannot open new render output");
            for (auto fixture : request.at("cases")) {
                set_filler(fixture, fixture.at("recipe").at("filler_lines").get<std::uint32_t>());
                output << json{{"case", fixture}}.dump() << '\n' << std::flush;
                if (!output) throw std::runtime_error("render output write failed");
            }
            return 0;
        }
        const auto& profile = request.at("profile");
        EngineOptions options;
        options.artifact_path = request.at("model").get<std::string>();
        options.max_context = profile.at("context_cap").get<std::uint32_t>();
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.max_concurrency = 1;
        const auto dtype = profile.at("dtype").get<std::string>();
        if (dtype != "int8" && dtype != "rk4v4-e8") throw std::invalid_argument("unsupported frozen dtype");
        options.kv_cache = dtype == "int8" ? KvCacheStorage::Int8Group64 : KvCacheStorage::RK4V4E8;
        const auto mode = profile.at("mode").get<std::string>();
        if (mode != "kvmem" && mode != "tiered-exact" && mode != "dense") throw std::invalid_argument("unsupported frozen mode");
        options.kv_mode = mode == "kvmem" ? KvMode::KVMem : mode == "tiered-exact" ? KvMode::TieredExact : KvMode::Dense;
        options.kvmem.view_tokens = profile.at("requested_view_tokens").get<std::uint32_t>();
        options.speculative.backend = SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
        options.speculative.proposal_head = ProposalHead::Optimized;
        options.enable_vision = false;
        options.enable_prompt_cache = false;
        const auto denominator = profile.at("denominator").get<std::string>();
        const auto* env = std::getenv("NINFER_KVMEM_SCORE_ALL_PAGES");
        const bool all = env && std::string(env) == "1";
        if (mode == "kvmem" && (all != (denominator == "all-committed")))
            throw std::invalid_argument("denominator environment/profile mismatch");
        Engine engine(options);
        std::ofstream output(argv[2], std::ios::out | std::ios::binary);
        if (!output) throw std::runtime_error("cannot open new raw output");
        auto emit = [&](const json& row) { output << row.dump() << '\n' << std::flush; if (!output) throw std::runtime_error("raw output write failed"); };
        auto fixtures = request.at("cases");
        if (operation == "fit") {
            // Fit each paired document against BOTH full chat forms, sharing the same text.
            std::map<std::string, std::vector<std::size_t>> groups;
            for (std::size_t i = 0; i < fixtures.size(); ++i) groups[fixtures[i].at("document_id").get<std::string>()].push_back(i);
            for (const auto& [id, indices] : groups) {
                auto fits = [&](std::uint32_t lines) {
                    bool fit = true;
                    for (const auto index : indices) {
                        set_filler(fixtures[index], lines);
                        try {
                            const auto count = engine.count_tokens(input_of(fixtures[index]));
                            fixtures[index]["prompt_tokens"] = count;
                            const auto budget = fixtures[index].at("requested_output_tokens").get<std::uint32_t>();
                            if (std::uint64_t(count) + budget + 128 > options.max_context) fit = false;
                        } catch (const RequestError& error) {
                            if (error.kind() != RequestErrorKind::ContextLengthExceeded) throw;
                            fit = false;
                        }
                    }
                    return fit;
                };
                if (!fits(80)) throw std::runtime_error("minimum separated injection document exceeds context");
                std::uint32_t low = 80, high = options.max_context; // Filler line always has >=1 token.
                while (low + 1 < high) {
                    const auto middle = low + (high-low)/2;
                    if (fits(middle)) low = middle; else high = middle;
                }
                if (!fits(low)) throw std::logic_error("non-monotonic tokenizer fit");
                for (const auto index : indices) emit({{"case_id", fixtures[index].at("id")}, {"infrastructure_status", "ok"}, {"case", fixtures[index]}});
            }
            return 0;
        }
        for (const auto& fixture : fixtures) {
            const auto id = fixture.at("id").get<std::string>();
            std::cerr << "[quality-case-begin] " << id << '\n' << std::flush;
            try {
                if (fixture.at("context_cap").get<std::uint32_t>() != options.max_context)
                    throw std::invalid_argument("case/profile context mismatch");
                const auto input = input_of(fixture);
                if (operation == "count") {
                    auto counted = fixture;
                    counted["prompt_tokens"] = engine.count_tokens(input);
                    emit({{"case_id", id}, {"infrastructure_status", "ok"}, {"case", counted}});
                } else {
                    if (!engine.healthy()) throw std::runtime_error("Engine unhealthy after previous failure");
                    RequestOptions generation;
                    generation.execution.allow_prefix_reuse = true;
                    generation.execution.requested_output_tokens = fixture.at("requested_output_tokens").get<std::uint32_t>();
                    generation.execution.sampling.temperature = 0;
                    generation.execution.sampling.top_k = 0;
                    generation.execution.sampling.top_p = 1;
                    generation.execution.sampling.min_p = 0;
                    generation.execution.sampling.presence_penalty = 0;
                    generation.execution.sampling.frequency_penalty = 0;
                    generation.execution.sampling.seed = 20261006;
                    generation.stop.include_model_defaults = fixture.at("include_model_defaults").get<bool>();
                    auto prepared = engine.prepare(input);
                    if (prepared.summary().prompt_tokens != fixture.at("prompt_tokens").get<std::uint32_t>())
                        throw std::invalid_argument("frozen actual tokenizer count mismatch");
                    const auto before = engine.runtime_stats();
                    const auto result = engine.generate(std::move(prepared), generation);
                    const auto after = engine.runtime_stats();
                    const auto& t = result.timings;
                    const auto committed = after.committed_decode_tokens - before.committed_decode_tokens;
                    json metrics = {{"prompt_tokens", result.prompt.prompt_tokens},
                        {"generated_tokens", result.generated_token_ids.size()}, {"reasoning_tokens", result.reasoning_tokens},
                        {"reused_prompt_tokens", result.reused_prompt_tokens}, {"prefix_reuse_path", reuse_name(result.prefix_reuse_path)},
                        {"computed_prefill_tokens", after.computed_prefill_tokens-before.computed_prefill_tokens},
                        {"committed_decode_tokens", committed}, {"prepare_seconds", t.prepare_seconds},
                        {"first_token_seconds", t.first_token_seconds}, {"vision_seconds", t.vision_seconds},
                        {"prefill_seconds", t.prefill_seconds}, {"decode_seconds", t.decode_seconds}, {"total_seconds", t.total_seconds},
                        {"decode_tokens_per_second", t.decode_seconds > 0 ? json(committed/t.decode_seconds) : json(nullptr)},
                        {"mtp_drafted_tokens", result.speculative.drafted_tokens}, {"mtp_accepted_tokens", result.speculative.accepted_tokens},
                        {"mtp_rounds", result.speculative.rounds}, {"mtp_fallback_steps", result.speculative.fallback_steps},
                        {"mtp_accepted_per_position", result.speculative.accepted_per_position},
                        {"mtp_acceptance", result.speculative.drafted_tokens ? json(double(result.speculative.accepted_tokens)/result.speculative.drafted_tokens) : json(nullptr)},
                        {"requested_view_tokens", options.kvmem.view_tokens}, {"resolved_main_kv_capacity_tokens", engine.memory_summary().kv_capacity},
                        {"finish_reason", static_cast<int>(result.finish_reason)}, {"prefill_chunk", engine.load_summary().prefill_chunk}};
                    emit({{"case_id", id}, {"infrastructure_status", "ok"}, {"metrics", metrics},
                          {"content", result.content}, {"reasoning", result.reasoning}, {"generated_token_ids", result.generated_token_ids}});
                }
            } catch (const std::exception& error) {
                emit({{"case_id", id}, {"infrastructure_status", "error"}, {"error", error.what()}});
            }
            std::cerr << "[quality-case-end] " << id << '\n' << std::flush;
        }
        return 0; // Validator failure is not a harness process/infrastructure failure.
    } catch (const std::exception& error) {
        std::cerr << "[quality-infrastructure-error] " << error.what() << '\n';
        return 2;
    }
}

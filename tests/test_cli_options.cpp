#include "../apps/cli/options.h"

#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ninfer::cli::Options parse(const std::vector<std::string>& flags) {
    std::vector<std::string> arguments{"ninfer", "model.ninfer", "--prompt", "Hello"};
    arguments.insert(arguments.end(), flags.begin(), flags.end());
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

std::optional<ninfer::cli::Options> accepts(const std::vector<std::string>& flags) {
    try { return parse(flags); }
    catch (const std::invalid_argument&) { return std::nullopt; }
}

bool rejects(const std::vector<std::string>& flags) {
    try {
        (void)parse(flags);
    } catch (const std::invalid_argument&) { return true; }
    std::cerr << "accepted Vision flags:";
    for (const std::string& flag : flags) { std::cerr << ' ' << flag; }
    std::cerr << '\n';
    return false;
}

} // namespace

int main() {
    int failures = 0;
    const ninfer::EngineOptions engine_defaults;
    failures += check(!engine_defaults.enable_vision &&
                          engine_defaults.vision_device == ninfer::VisionDevice::Cuda &&
                          engine_defaults.vision_mmproj_path.empty() &&
                          engine_defaults.vision_cpu_threads == 6 &&
                          engine_defaults.vision_cpu_memory_mib == 4096,
                      "Engine Vision defaults mismatch");
    const auto defaults = parse({});
    failures += check(engine_defaults.vision_cpu_cache_mib == 128 && defaults.vision_cpu_cache_mib == 128,
                      "CPU cache must default to 128 MiB");
    failures += check(parse({"--vision-mmproj", "mmproj.gguf", "--vision-cpu-cache-mib", "0"}).vision_cpu_cache_mib == 0,
                      "CPU cache zero must disable caching");
    failures += check(parse({"--vision-cpu-cache-mib", "64", "--vision-mmproj", "mmproj.gguf"}).vision_cpu_cache_mib == 64,
                      "CPU cache parsing must be independent of option order");
    for (const std::string& value : {std::string("-1"), std::string("4294967296"), std::string("+1"), std::string(" 1"), std::string("1x"), std::string("")}) {
        failures += check(rejects({"--vision-mmproj", "mmproj.gguf", "--vision-cpu-cache-mib", value}), "invalid CPU cache capacity accepted");
    }
    failures += check(rejects({"--vision-device", "cuda", "--vision-cpu-cache-mib", "128"}) &&
                          rejects({"--vision-cpu-cache-mib", "0"}), "CUDA accepted CPU cache tuning");
    failures += check(ninfer::cli::usage_text("ninfer").find("--vision-cpu-cache-mib") != std::string::npos,
                      "CLI help omits CPU cache tuning");
    failures += check(!defaults.enable_vision &&
                          defaults.vision_device == ninfer::VisionDevice::Cuda &&
                          defaults.vision_mmproj_path.empty() && defaults.vision_max_tokens == 8192 &&
                          defaults.vision_cpu_threads == 6 && defaults.vision_cpu_memory_mib == 4096,
                      "CLI Vision defaults mismatch");
    for (const std::vector<std::string>& flags : std::vector<std::vector<std::string>>{
             {"--vision"}, {"--vision-max-tokens", "1024"}, {"--vision-limit", "1024"},
             {"--vision-device", "cuda"}, {"--vision-device", "cuda", "--vision-device", "cuda"}}) {
        const auto options = parse(flags);
        failures += check(options.enable_vision && options.vision_device == ninfer::VisionDevice::Cuda &&
                              options.vision_mmproj_path.empty(),
                          "existing CLI Vision flags did not preserve CUDA behavior");
    }
    failures += check(parse({"--vision-max-tokens", "1024"}).vision_max_tokens == 1024 &&
                          parse({"--vision-limit", "1024"}).vision_max_tokens == 1024,
                      "CLI Vision token limit or alias was lost");
    const auto shortcut = parse({"--vision-mmproj", "mmproj-BF16.gguf"});
    failures += check(shortcut.enable_vision && shortcut.vision_device == ninfer::VisionDevice::Cpu &&
                          shortcut.vision_mmproj_path == "mmproj-BF16.gguf",
                      "CLI --vision-mmproj did not enable CPU Vision");
    const auto embedded_cpu = accepts({"--vision-device", "cpu"});
    failures += check(embedded_cpu && embedded_cpu->enable_vision &&
                          embedded_cpu->vision_device == ninfer::VisionDevice::Cpu &&
                          embedded_cpu->vision_mmproj_path.empty(),
                      "CLI rejected embedded CPU Vision without external mmproj");
    const auto embedded_cpu_tuned =
        accepts({"--vision-cpu-threads", "8", "--vision-device", "cpu"});
    failures += check(embedded_cpu_tuned && embedded_cpu_tuned->enable_vision &&
                          embedded_cpu_tuned->vision_device == ninfer::VisionDevice::Cpu &&
                          embedded_cpu_tuned->vision_cpu_threads == 8 &&
                          embedded_cpu_tuned->vision_mmproj_path.empty(),
                      "CLI rejected pathless CPU Vision tuning");
    const auto external_gpu =
        accepts({"--vision-device", "cuda", "--vision-mmproj", "mmproj-BF16.gguf"});
    failures += check(external_gpu && external_gpu->enable_vision &&
                          external_gpu->vision_device == ninfer::VisionDevice::Cuda &&
                          external_gpu->vision_mmproj_path == "mmproj-BF16.gguf",
                      "CLI rejected external GGUF on explicitly selected GPU");
    const auto external_gpu_reversed =
        accepts({"--vision-mmproj", "mmproj-BF16.gguf", "--vision-device", "cuda"});
    failures += check(external_gpu_reversed && external_gpu_reversed->enable_vision &&
                          external_gpu_reversed->vision_device == ninfer::VisionDevice::Cuda &&
                          external_gpu_reversed->vision_mmproj_path == "mmproj-BF16.gguf",
                      "CLI external GGUF GPU selection depends on option order");
    for (const char* next_flag : {"--vision", "--vision-device", "--vision-max-tokens",
                                  "--api-key", "-h"}) {
        bool missing_path_rejected = false;
        try {
            (void)parse({"--vision-mmproj", next_flag});
        } catch (const std::invalid_argument& error) {
            const std::string message = error.what();
            missing_path_rejected = message.find("--vision-mmproj") != std::string::npos &&
                                    message.find("path") != std::string::npos;
        }
        failures += check(missing_path_rejected,
                          "CLI --vision-mmproj consumed the following option as its path");
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
        const auto options = parse(flags);
        failures += check(options.enable_vision && options.vision_device == ninfer::VisionDevice::Cpu &&
                              options.vision_mmproj_path == "mmproj-BF16.gguf" &&
                              options.vision_cpu_threads == 8 && options.vision_cpu_memory_mib == 2048,
                          "CLI CPU Vision options depend on argument order");
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
        failures += check(rejects(flags), "invalid CLI CPU Vision combination was accepted");
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
            failures += check(rejects({"--vision-mmproj", "mmproj.gguf", flag, value}),
                              "invalid CLI CPU Vision unsigned limit was accepted");
        }
    }
    const auto decimal_limits = parse({"--vision-mmproj", "mmproj.gguf", "--vision-cpu-threads", "001",
                                       "--vision-cpu-memory-mib", "0001"});
    failures += check(decimal_limits.vision_cpu_threads == 1 &&
                          decimal_limits.vision_cpu_memory_mib == 1,
                      "CLI CPU Vision rejected decimal digits with leading zeros");
    failures += check(rejects({"--vision-mmproj", "mmproj.gguf", "--vision-cpu-threads", "513"}),
                      "CLI CPU Vision thread count exceeded the GGML threadpool limit");
    const auto limits = parse({"--vision-mmproj", "mmproj.gguf", "--vision-cpu-threads", "512",
                               "--vision-cpu-memory-mib", "4294967295"});
    failures += check(limits.vision_cpu_threads == ninfer::kMaximumVisionCpuThreads &&
                          limits.vision_cpu_memory_mib == 4294967295U,
                      "CLI CPU Vision parser did not preserve the thread or memory limit");
    for (const char* flag : {"--vision-device", "--vision-mmproj", "--vision-cpu-threads",
                             "--vision-cpu-memory-mib"}) {
        failures += check(ninfer::cli::usage_text("ninfer").find(flag) != std::string::npos,
                          "CLI help omits a CPU Vision option");
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}

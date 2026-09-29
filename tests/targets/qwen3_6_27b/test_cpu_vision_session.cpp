#include "targets/qwen3_6_27b/impl/variant.h"
#define NINFER_QWEN36_VARIANT ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/vision_context.h"

#include <iostream>
#include <algorithm>
#include <stdexcept>

namespace {
using namespace ninfer;
using namespace ninfer::targets::qwen3_6;
using namespace ninfer::targets::qwen3_6::detail::qwen3_6_27b_runtime;

void require(bool result, const char* message) {
    if (!result) { throw std::runtime_error(message); }
}

class Encoder final : public CpuVisionEncoder {
public:
    int calls = 0;
    void validate(const CpuVisionInput& input) const override {
        (void)validate_cpu_vision_input(input);
    }
    std::vector<std::uint16_t> encode(const CpuVisionInput& input,
                                     const std::function<bool()>& cancelled) override {
        validate(input);
        if (cancelled && cancelled()) { throw CpuVisionCancelled(); }
        ++calls;
        // Independent BF16 oracle: 1.0 is 0x3f80; 2.0 is 0x4000.
        return std::vector<std::uint16_t>(5120, calls == 1 ? 0x3f80 : 0x4000);
    }
    std::uint64_t identity_hash() const noexcept override { return 1234; }
    std::size_t weight_bytes() const noexcept override { return 0; }
};

PreparedPromptData prompt() {
    PreparedPromptData result;
    result.token_ids.resize(8);
    result.patches.resize(8 * 1536, 0.0F);
    result.vision_items = {
        {.modality = PromptModality::Image, .grid = {1, 2, 2}, .patch_begin = 0, .patch_count = 4},
        {.modality = PromptModality::Image, .grid = {1, 2, 2}, .patch_begin = 4, .patch_count = 4},
    };
    return result;
}

VisionPrefillPlan plan() {
    auto control = std::make_shared<VisionControl>();
    control->items = {
        {.modality = PromptModality::Image, .grid = {1, 2, 2}, .patch_begin = 0,
         .patch_count = 4, .merged_count = 1},
        {.modality = PromptModality::Image, .grid = {1, 2, 2}, .patch_begin = 4,
         .patch_count = 4, .merged_count = 1},
    };
    return {.control = std::move(control), .uses = {{1, 3, 0}, {4, 6, 1}}};
}

void check(DeviceContext& device) {
    LoadedModelData model;
    model.features.vision = true;
    model.features.vision_device = VisionDevice::Cpu;
    auto encoder = std::make_shared<Encoder>();
    model.cpu_vision = encoder;
    DeviceBuffer output(5120 * sizeof(std::uint16_t));
    WorkspaceArena work(256); // Entire encoder must work without a GPU vision workspace.
    auto prepared = prompt();
    const auto media_plan = plan();
    const runtime::TransientRegion transient{static_cast<std::byte*>(output.p), output.bytes, 256};
    schedule::VisionPrefillSession session(device, model, work, prepared, media_plan, transient);
    auto chunk = session.prepare_chunk(0, 8);
    require(chunk.length == 4 && encoder->calls == 1, "first media chunk cap/encode changed");
    std::vector<std::uint16_t> copied(5120);
    output.copy_to_host(copied.data(), output.bytes);
    require(std::ranges::all_of(copied, [](auto value) { return value == 0x3f80; }),
            "BF16 CPU output upload is corrupt or unfinished");
    (void)session.prepare_chunk(2, 2);
    require(encoder->calls == 1 && !session.release_consumed_media_payload(),
            "split chunk reencoded media or released payload early");
    (void)session.prepare_chunk(4, 4);
    output.copy_to_host(copied.data(), output.bytes);
    require(encoder->calls == 2 && copied.front() == 0x4000 && copied.back() == 0x4000,
            "second media item was not encoded/uploaded");
    require(session.release_consumed_media_payload() && prepared.patches.empty() &&
            !session.release_consumed_media_payload(), "final payload release contract changed");
    require(session.elapsed_seconds() >= 0.0 && work.used() == 0,
            "CPU encoder used GPU workspace or produced invalid timing");
    const auto uncached_stats = session.cpu_vision_cache_stats();
    require(uncached_stats.hits == 0 && uncached_stats.misses == 2 && uncached_stats.encode_calls == 2,
            "uncached CPU media counters changed");

    auto cached_encoder = std::make_shared<Encoder>();
    model.cpu_vision = make_cached_cpu_vision_encoder(cached_encoder, 5120 * sizeof(std::uint16_t));
    prepared = prompt();
    schedule::VisionPrefillSession cached(device, model, work, prepared, media_plan, transient);
    (void)cached.prepare_chunk(0, 8);
    (void)cached.prepare_chunk(2, 2);
    (void)cached.prepare_chunk(4, 4);
    const auto cached_stats = cached.cpu_vision_cache_stats();
    require(cached_encoder->calls == 1 && cached_stats.hits == 1 && cached_stats.misses == 1 &&
            cached_stats.encode_calls == 1, "runtime did not hash identical processed images or count hits");
    output.copy_to_host(copied.data(), output.bytes);
    require(copied.front() == 0x3f80 && copied.back() == 0x3f80,
            "cache hit upload did not preserve BF16 embeddings");
    prepared = prompt();
    schedule::VisionPrefillSession repeated(device, model, work, prepared, media_plan, transient);
    (void)repeated.prepare_chunk(0, 8);
    (void)repeated.prepare_chunk(4, 4);
    const auto repeated_stats = repeated.cpu_vision_cache_stats();
    require(cached_encoder->calls == 1 && repeated_stats.hits == 2 && repeated_stats.misses == 0 &&
            repeated_stats.encode_calls == 0, "model cache or per-request counters were not retained/reset");

    // Restore the v1 fixture for its existing cancellation/upload assertions.
    model.cpu_vision = encoder;
    const std::vector<std::uint16_t> previous_output(5120, 0x4000);
    CUDA_CHECK(cudaMemcpy(output.p, previous_output.data(), output.bytes, cudaMemcpyHostToDevice));

    prepared = prompt();
    schedule::VisionPrefillSession cancelled(device, model, work, prepared, media_plan, transient,
                                              [] { return true; });
    bool aborted = false;
    try { (void)cancelled.prepare_chunk(0, 8); }
    catch (const runtime::RequestCancelled&) { aborted = true; }
    require(aborted && encoder->calls == 2 && !cancelled.release_consumed_media_payload(),
            "cancelled encoder consumed an incomplete media item");
    output.copy_to_host(copied.data(), output.bytes);
    require(copied.front() == 0x4000, "cancelled encode overwrote device embeddings");
}
} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { return 77; }
    try { DeviceContext device(0); check(device); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}

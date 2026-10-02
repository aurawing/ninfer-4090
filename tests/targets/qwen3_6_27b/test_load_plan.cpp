#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact_fixture.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <ninfer/targets/qwen3_6_27b/package.h>

#include <bit>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::targets::qwen3_6_27b::Package;
using namespace ninfer::targets::qwen3_6_27b::detail;

std::filesystem::path artifact_path(const char* environment, const char* filename) {
    if (const char* value = std::getenv(environment); value != nullptr && *value != '\0') {
        return value;
    }
    const std::filesystem::path default_path = std::filesystem::path(NINFER_SOURCE_DIR) / "out" / filename;
    if (std::filesystem::is_regular_file(default_path)) {
        return default_path;
    }
    const std::filesystem::path fallback_llm = std::filesystem::path("C:/Users/Henrik/Desktop/LLM") / filename;
    if (std::filesystem::is_regular_file(fallback_llm)) {
        return fallback_llm;
    }
    return default_path;
}

ninfer::targets::qwen3_6::StartupFeatures all_features() {
    return {
        .vision        = true,
        .speculative   = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
}

std::size_t dflash2_device_objects(const ninfer::artifact::Reader& reader,
                                   const ninfer::artifact::MaterializationPlan& plan) {
    return static_cast<std::size_t>(
        std::ranges::count_if(plan.device_objects, [&](const auto& item) {
            return ninfer::artifact::object_name(reader.objects().at(item.object.index))
                .starts_with("dflash2/");
        }));
}

int verify_legacy_dflash2_compatibility(const std::filesystem::path& path, WeightsProfile profile) {
    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan = bind_artifact(binder, profile, all_features());
    if (dflash2_device_objects(reader, plan.materialization) != 0) {
        std::cerr << "legacy artifact unexpectedly bound DFlash2 on device: " << path << '\n';
        return 1;
    }
    auto cpu_features = all_features();
    cpu_features.vision_device = ninfer::VisionDevice::Cpu;
    cpu_features.vision_mmproj_path = "external-mmproj.gguf";
    ninfer::artifact::Binder cpu_binder(reader);
    const auto cpu_plan = bind_artifact(cpu_binder, profile, cpu_features);
    if (cpu_plan.materialization.device_capacity_bytes >= plan.materialization.device_capacity_bytes) {
        std::cerr << "CPU vision did not reduce device weight capacity\n";
        return 1;
    }
    for (const auto& item : cpu_plan.materialization.device_objects) {
        if (ninfer::artifact::object_name(reader.objects().at(item.object.index)).starts_with("vision/")) {
            std::cerr << "CPU mode placed vision weights on device\n";
            return 1;
        }
    }
    if (cpu_plan.bindings.features != cpu_features ||
        cpu_plan.materialization.host_objects.size() != plan.materialization.host_objects.size()) {
        std::cerr << "CPU mode changed frontend host bindings or frozen settings\n";
        return 1;
    }
    auto embedded_cpu_features = cpu_features;
    embedded_cpu_features.vision_mmproj_path.clear();
    ninfer::artifact::Binder embedded_cpu_binder(reader);
    const auto embedded_cpu_plan = bind_artifact(embedded_cpu_binder, profile, embedded_cpu_features);
    if (!embedded_cpu_plan.bindings.vision ||
        embedded_cpu_plan.materialization.device_capacity_bytes !=
            cpu_plan.materialization.device_capacity_bytes) {
        std::cerr << "embedded CPU Vision did not retain validated host-readable bindings\n";
        return 1;
    }
    auto external_gpu_features = all_features();
    external_gpu_features.vision_mmproj_path = "external-mmproj.gguf";
    ninfer::artifact::Binder external_gpu_binder(reader);
    const auto external_gpu_plan = bind_artifact(external_gpu_binder, profile, external_gpu_features);
    if (external_gpu_plan.materialization.device_capacity_bytes !=
            cpu_plan.materialization.device_capacity_bytes) {
        std::cerr << "external GGML CUDA Vision still allocated embedded vision weights\n";
        return 1;
    }
    return 0;
}

int verify_optional_vision_inventory() {
    using Json = nlohmann::json;
    const auto directory = [](bool partial) {
        Json objects = Json::array({{{"name", "frontend"},
                                     {"kind", "resource"},
                                     {"encoding", "raw-bytes-v1"},
                                     {"offset", 0},
                                     {"bytes", 1}}});
        if (partial) {
            objects.push_back({{"name", "vision/merger/fc2_bias"},
                               {"kind", "tensor"},
                               {"shape", {1}},
                               {"format", "BF16"},
                               {"layout", "contiguous-le-v1"},
                               {"offset", 256},
                               {"bytes", 2}});
        }
        return Json{{"identity", {{"model_id", "fixture"}, {"weights_id", "fixture"}}},
                    {"objects", std::move(objects)}};
    };
    for (const bool partial : {false, true}) {
        auto fixture = ninfer::test::artifact_fixture::write_fixture(directory(partial), "vision-inventory");
        ninfer::artifact::Reader reader(fixture.path);
        ninfer::artifact::Binder binder(reader);
        binder.validate_only(binder.require_resource(
            "frontend", ninfer::artifact::ResourceEncoding::RawBytesV1));
        if (bind_optional_vision(binder, ninfer::artifact::TensorPlacement::ValidateOnly)) {
            std::cerr << "partial or absent Vision inventory was treated as complete\n";
            return 1;
        }
        bool finish_rejected = false;
        try { (void)binder.finish(); }
        catch (const ninfer::artifact::ArtifactError&) { finish_rejected = true; }
        if (finish_rejected != partial) {
            std::cerr << "optional Vision inventory did not distinguish absent from partial\n";
            return 1;
        }
    }
    return 0;
}

int verify_dflash2_bundle(const std::filesystem::path& path, WeightsProfile profile) {
    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan = bind_artifact(binder, profile, all_features());
    if (dflash2_device_objects(reader, plan.materialization) != 0) {
        std::cerr << "DFlash2 bundle materialized on device instead of ValidateOnly: " << path << '\n';
        return 1;
    }
    return 0;
}

int verify_groupwise(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    if (Package::resolve_weights(reader.identity()) != WeightsProfile::GroupwiseInt) {
        std::cerr << "groupwise identity resolved to the wrong profile\n";
        return 1;
    }
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan =
        bind_artifact(binder, WeightsProfile::GroupwiseInt, all_features());
    if (plan.materialization.object_count != 1124 ||
        plan.materialization.device_objects.size() != 1118 ||
        plan.materialization.host_objects.size() != 6 ||
        plan.materialization.device_capacity_bytes == 0) {
        std::cerr << "groupwise materialization plan is incomplete\n";
        return 1;
    }
    if (plan.bindings.token_embedding.format != NumericFormat::Q6G64_F16S ||
        plan.bindings.output_head.format != NumericFormat::Q6G64_F16S) {
        std::cerr << "groupwise vocabulary endpoints have the wrong storage profile\n";
        return 1;
    }
    for (const TextLayerPlan& layer : plan.bindings.text_layers) {
        if (layer.is_full_attention) {
            if (!std::holds_alternative<SplitAttentionProjectionPlan>(layer.attention.projection)) {
                std::cerr << "groupwise attention parent boundary changed\n";
                return 1;
            }
        } else if (!std::holds_alternative<SplitGdnInputProjectionPlan>(
                       layer.gdn.input_projection)) {
            std::cerr << "groupwise GDN parent boundary changed\n";
            return 1;
        }
        if (layer.mlp.gate_up.format != NumericFormat::Q4G64_F16S ||
            layer.mlp.down.format != NumericFormat::Q5G64_F16S) {
            std::cerr << "groupwise MLP storage profile changed\n";
            return 1;
        }
    }
    return 0;
}

int verify_rejection() {
    try {
        (void)Package::resolve_weights({"qwen3.6-27b", "unknown"});
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        if (message.find("qwen3.6-27b/unknown") != std::string::npos) { return 0; }
    }
    std::cerr << "unknown weights identity was not rejected with the full identity\n";
    return 1;
}

int verify_profile_mismatch_rejection() {
    ninfer::DeviceContext device(0);
    ninfer::EngineOptions options;
    options.max_context    = 128;
    options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(128);
    options.prefill_chunk  = 128;
    options.use_cuda_graph = false;
    auto planner = Package::make_sequence_planner(device, options, WeightsProfile::GroupwiseInt);
    const std::uint32_t pages = planner.capacity_curve().minimum_main_page_groups;
    const auto planned_bytes = planner.capacity_curve().reservation_bytes(pages);
    auto moved = std::move(planner);
    if (planner.capacity_curve().minimum_main_page_groups != 0 ||
        moved.capacity_curve().reservation_bytes(pages) != planned_bytes) {
        std::cerr << "planner move did not transfer ownership and preserve the capacity curve\n";
        return 1;
    }
    bool empty_rejected = false;
    try { (void)std::move(planner).finalize(pages); }
    catch (const std::logic_error&) { empty_rejected = true; }
    if (!empty_rejected) {
        std::cerr << "moved-from planner unexpectedly finalized\n";
        return 1;
    }
    auto sequence = std::move(moved).finalize(pages);
    if (sequence.device_reservation_bytes() != planned_bytes) {
        std::cerr << "moved planner finalized a different reservation\n";
        return 1;
    }
    RuntimeModelView empty_model;
    try {
        (void)ninfer::targets::qwen3_6::create_program<Variant>(
            empty_model, WeightsProfile::GroupwiseIntW8Endpoints, std::move(sequence), device);
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find("weights profile") != std::string::npos) { return 0; }
    }
    std::cerr << "mismatched load/sequence weights profiles were not rejected\n";
    return 1;
}

int verify_ggml_cuda_graph_reservation() {
    ninfer::DeviceContext device(0);
    ninfer::EngineOptions options;
    options.max_context    = 128;
    options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(128);
    options.prefill_chunk  = 128;
    options.enable_vision  = true;
    options.use_cuda_graph = true;
    const auto reservation = [&] {
        auto planner = Package::make_sequence_planner(device, options, WeightsProfile::GroupwiseInt);
        const auto pages = planner.capacity_curve().minimum_main_page_groups;
        return std::move(planner).finalize(pages).device_reservation_bytes();
    };
    const std::size_t native_bytes = reservation();
    options.vision_mmproj_path = "external-vision.gguf";
    const std::size_t ggml_cuda_bytes = reservation();
    if (ggml_cuda_bytes != native_bytes + 64ULL * 1024ULL * 1024ULL) {
        std::cerr << "GGML CUDA Vision did not reserve its additional CUDA Graph allowance\n";
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (const int result = verify_optional_vision_inventory(); result != 0) { return result; }
    if (argc == 2 && std::string_view(argv[1]) == "--vision-inventory") { return 0; }
    if (const int result = verify_rejection(); result != 0) { return result; }
    if (const int result = verify_profile_mismatch_rejection(); result != 0) { return result; }
    if (const int result = verify_ggml_cuda_graph_reservation(); result != 0) { return result; }

    const std::filesystem::path groupwise =
        artifact_path("NINFER_QWEN3_6_27B_WEIGHTS", "qwen3_6_27b.ninfer");
    const std::filesystem::path qwen38_groupwise =
        artifact_path("NINFER_QWEN3_8_27B_WEIGHTS", "qwen3_8_27b.ninfer");
    const std::filesystem::path qwen38_dflash2 =
        artifact_path("NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS", "qwen3_8_27b_dflash2.ninfer");

    bool verified_any = false;
    if (std::filesystem::is_regular_file(groupwise)) {
        if (const int result = verify_groupwise(groupwise); result != 0) { return result; }
        if (const int result =
                verify_legacy_dflash2_compatibility(groupwise, WeightsProfile::GroupwiseInt);
            result != 0) {
            return result;
        }
        verified_any = true;
    }

    if (std::filesystem::is_regular_file(qwen38_groupwise)) {
        if (const int result = verify_legacy_dflash2_compatibility(
                qwen38_groupwise, WeightsProfile::GroupwiseIntW8Endpoints);
            result != 0) {
            return result;
        }
        verified_any = true;
    }

    if (std::filesystem::is_regular_file(qwen38_dflash2)) {
        if (const int result = verify_dflash2_bundle(qwen38_dflash2,
                                                     WeightsProfile::GroupwiseIntW8Endpoints);
            result != 0) {
            return result;
        }
        verified_any = true;
    }

    if (!verified_any) {
        std::cerr << "skip: real 27B artifact is required: groupwise=" << groupwise << '\n';
        return 77;
    }
    return 0;
}

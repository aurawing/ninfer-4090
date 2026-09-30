#pragma once

#include "targets/qwen3_6_27b/impl/load/bindings.h"

namespace ninfer::targets::qwen3_6_27b::detail {

std::shared_ptr<qwen3_6::CpuVisionEncoder> make_embedded_vision_encoder(
    artifact::Binder& binder, const VisionBindingPlan& vision,
    std::uint32_t threads, std::size_t memory_budget_bytes);

} // namespace ninfer::targets::qwen3_6_27b::detail

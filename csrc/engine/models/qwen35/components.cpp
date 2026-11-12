#include "engine/models/source.h"
#include "engine/ops.h"
#include "engine/models/qwen35/components.h"

#include "engine/models/common/load.h"
#include "engine/weights/loader.h"
#include "engine/models/qwen35/load.h"
#include "engine/models/qwen35/mtp.h"
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"

#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/weights/placement.h"
#include "engine/weights/cache.h"

#include <iostream>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::qwen35 {

tyloi::engine::Metrics Components::metrics() const {
    const auto component_state = state();
    const auto quantization = cuda_kv_quantization();
    return {
        {"kv_memory_supported", 0.0},
        {"kv_quantization_supported", 1.0},
        {"kv_quantization_bits", quantization.bits},
        {"vision_declared", double(component_state.vision_declared)},
        {"vision_supported", double(component_state.vision_supported)},
        {"vision_available", double(component_state.vision_available)},
        {"mtp_declared", double(component_state.mtp_declared)},
        {"mtp_supported", double(component_state.mtp_supported)},
        {"mtp_available", double(component_state.mtp_available)}};
}

void Components::describe(tyloi::engine::EngineInfo& info) const {
    const auto component_state = state();
    info.multimodal = component_state.vision_available;
    const auto caps = cuda_runtime_capabilities(
        graph, plan, component_state, info.multimodal, false);
    info.duplex = caps.full_duplex;
    info.capabilities = {graph.architecture, caps.text,
        caps.image_input, caps.video_input, caps.audio_input, caps.audio_output,
        caps.full_duplex, caps.mtp, "model-graph+cuda-adapters"};
}

Components Components::load(
    Model& model,
    const ::tyloi::weights::Loader& weights, ::tyloi::weights::Cache& cache,
    LayerPlacementConfig& placement,
    int& cpu_layers,
    Executor& executor,
    const int64_t& kv_capacity,
    tyloi::ops::Trace* const& trace,
    const int& trace_scope,
    bool optional,
    bool cpu_offload,
    bool preallocate) {
    Components components{model.graph, cuda_model_plan(model.graph)};
    if (!optional) return components;
    const bool mtp_allowed =
        !placement.enabled() && cpu_layers == 0 && !cpu_offload &&
        cache.transfer() == ::tyloi::weights::Cache::Transfer::automatic;

    load_vision(model, weights, options::vision_resident());
    components.vision_available = model.vision.has_value();

    if (components.plan.predictor == CudaPredictorAdapter::qwen35) {
        if (mtp_allowed && model.metadata.num_experts == 0 &&
            model.supports_speculation()) {
            TransformerWeightLoader loader{weights};
            auto predictor = load_predictor(
                loader, model, cuda_kv_quantization());
            if (predictor) {
                // Mtp retains these references, so they must outlive the result.
                components.mtp = std::make_unique<Mtp>(
                    std::move(*predictor), executor.fork(),
                    kv_capacity, trace, trace_scope, preallocate);
            }
        } else {
            std::cerr
                << "qwen_mtp unavailable: CUDA adapter requires dense GPU-resident Qwen blocks\n";
        }
    } else if (components.plan.predictor != CudaPredictorAdapter::none) {
        throw std::runtime_error("CUDA predictor adapter is unsupported for Qwen");
    }
    return components;
}

} // namespace tyloi::cuda::qwen35

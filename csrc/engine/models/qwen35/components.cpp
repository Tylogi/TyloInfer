#include "engine/models/source.h"
#include "engine/ops.h"
#include "engine/models/qwen35/components.h"

#include "engine/models/qwen35/load.h"
#include "engine/models/qwen35/mtp.h"
#include "engine/models/qwen35/state.h"
#include "engine/models/session/cache.h"
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"

#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/options.h"
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
    const ::tyloi::weights::Loader& weights,
    const LayerPlacementConfig& placement,
    int cpu_layers,
    Executor& executor,
    std::unique_ptr<tyloi::Executor>& predictor_execution,
    bool optional,
    bool cpu_offload,
    bool preallocate) {
    Components components{model.graph, cuda_model_plan(model.graph)};
    if (!optional) return components;
    load_components(model, weights, placement, cpu_layers, cpu_offload,
        options::vision_resident(), preallocate);
    components.vision_available = model.vision.has_value();
    components.mtp = model.mtp.get();
    if (components.mtp) {
        predictor_execution = executor.fork(model.device.index);
    }
    return components;
}

void restore_session(State& state, tyloi::models::Mtp* mtp, const TextSessionState& snapshot,
        const std::function<void()>& drain) {
    tyloi::cuda::internal::CudaSessionOps::restore(
        static_cast<tyloi::models::State&>(state), snapshot, drain, mtp);
}

} // namespace tyloi::cuda::qwen35

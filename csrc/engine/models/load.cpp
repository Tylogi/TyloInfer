#include "engine/models/load.h"

#include "engine/models/causal_model.h"
#include "engine/models/gemma4/load.h"
#include "engine/models/qwen35/load.h"
#include "engine/models/source.h"
#include "engine/models/source/model_graph.h"

#include <stdexcept>
#include <utility>

namespace tyloi::models {

std::unique_ptr<CausalModel> load(
    std::shared_ptr<const ModelSource> source,
    std::string_view definition,
    const weights::Loader& weights,
    LayerPlacementConfig& placement,
    LoadOptions options) {
    const auto model_graph = graph(*source);
    if (model_graph.backbone == "gemma4") {
        return ::tyloi::cuda::gemma4::load(
            std::move(source), definition, options.context, weights, placement,
            options.gpu_layers, options.expert_parallel);
    }
    if (model_graph.backbone == "qwen3_5" || model_graph.backbone == "generic_qwen") {
        int cpu_layers = 0;
        auto model = ::tyloi::cuda::qwen35::load(
            std::move(source), definition, weights, placement, options, cpu_layers);
        ::tyloi::cuda::qwen35::load_components(
            *model, weights, placement, cpu_layers, false, options.vision_resident, false);
        return model;
    }
    throw std::runtime_error(
        "unsupported model backbone: " + model_graph.backbone);
}

} // namespace tyloi::models

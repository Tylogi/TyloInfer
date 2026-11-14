#include "engine/models/load.h"

#include "engine/models/causal_model.h"
#include "engine/models/deepseek_v41/load.h"
#include "engine/models/gemma4/load.h"
#include "engine/models/nemotron_h/load.h"
#include "engine/models/qwen35/load.h"
#include "engine/models/qwen4_exp/load.h"
#include "engine/models/source.h"
#include "engine/models/source/model_graph.h"

#include <stdexcept>
#include <utility>

namespace tyloi::models {

void load(
    std::unique_ptr<CausalModel>& owner,
    std::shared_ptr<const ModelSource> source,
    std::string_view definition,
    const weights::Loader& weights,
    LayerPlacementConfig& placement,
    LoadOptions options) {
    const auto model_graph = graph(*source);
    if (model_graph.backbone == "gemma4") {
        if (!options.cpu_offload_layers.empty())
            throw std::invalid_argument("Gemma4 does not support --cpu-offload-layers");
        ::tyloi::cuda::gemma4::load(
            owner, std::move(source), definition, options.context, weights, placement,
            options.gpu_layers, options.expert_parallel);
        return;
    }
    if (model_graph.backbone == "qwen3_5" || model_graph.backbone == "generic_qwen") {
        int cpu_layers = 0;
        auto model = ::tyloi::cuda::qwen35::load(
            std::move(source), definition, weights, placement, options, cpu_layers);
        owner = std::move(model);
        if (options.optional_components)
            ::tyloi::cuda::qwen35::load_components(
                static_cast<::tyloi::cuda::qwen35::Model&>(*owner), weights, placement,
                cpu_layers, !options.cpu_offload_layers.empty(), options.vision_resident, false);
        return;
    }
    if (model_graph.backbone == "deepseek_v41") {
        ::tyloi::cuda::deepseek_v41::load(owner, std::move(source), definition, weights, placement, options);
        return;
    }
    if (model_graph.backbone == "qwen4_exp") {
        ::tyloi::cuda::qwen4_exp::load(owner, std::move(source), definition, weights, placement, options);
        return;
    }
    if (model_graph.backbone == "nemotron_h") {
        ::tyloi::cuda::nemotron_h::load(owner, std::move(source), definition, weights, placement, options);
        return;
    }
    throw std::runtime_error(
        "unsupported model backbone: " + model_graph.backbone);
}

std::unique_ptr<CausalModel> load(
    std::shared_ptr<const ModelSource> source, std::string_view definition,
    const weights::Loader& weights, LayerPlacementConfig& placement, LoadOptions options) {
    std::unique_ptr<CausalModel> owner;
    load(owner, std::move(source), definition, weights, placement, std::move(options));
    return owner;
}

} // namespace tyloi::models

#include "engine/models/load.h"

#include "engine/models/causal_model.h"
#include "engine/models/qwen35/load.h"
#include "engine/models/qwen4_exp/load.h"
#include "engine/models/source/model.h"
#include "engine/models/source/model_graph.h"
#include "engine/weights/loader.h"

#if TYLOINFER_HAS_CUDA
#include "engine/models/glm_dsa/load.h"
#include "engine/models/glm5_next/load.h"
#include "engine/models/deepseek_v4/load.h"
#include "engine/models/deepseek_v41/load.h"
#include "engine/models/gemma4/load.h"
#include "engine/models/nemotron_h/load.h"
#include "engine/models/minicpmo45/text/load.h"
#include "engine/models/minicpmo45/model.h"
#endif

#include <stdexcept>
#include <string>
#include <utility>

namespace {

enum class ModelFamily {
    unsupported,
    qwen35,
    qwen4_exp,
#if TYLOINFER_HAS_CUDA
    glm_dsa,
    glm5_next,
    deepseek_v4,
    gemma4,
    deepseek_v41,
    nemotron_h,
    minicpmo45,
    minicpmo_tts,
#endif
};

ModelFamily select_family(const tyloi::ModelGraph& model_graph, tyloi::Device target) {
    const auto qwen_family = [&] {
        if (model_graph.backbone == "qwen3_5" || model_graph.backbone == "generic_qwen")
            return ModelFamily::qwen35;
        if (model_graph.backbone == "qwen4_exp") return ModelFamily::qwen4_exp;
        return ModelFamily::unsupported;
    }();
    if (target.is_cuda()) {
#if TYLOINFER_HAS_CUDA
        if (qwen_family != ModelFamily::unsupported) return qwen_family;
        if (model_graph.backbone == "glm_dsa") return ModelFamily::glm_dsa;
        if (model_graph.backbone == "glm5_next") return ModelFamily::glm5_next;
        if (model_graph.backbone == "deepseek_v4") return ModelFamily::deepseek_v4;
        if (model_graph.backbone == "gemma4") return ModelFamily::gemma4;
        if (model_graph.backbone == "deepseek_v41") return ModelFamily::deepseek_v41;
        if (model_graph.backbone == "nemotron_h") return ModelFamily::nemotron_h;
        if (model_graph.backbone == "minicpmo45") return ModelFamily::minicpmo45;
        if (model_graph.backbone == "minicpmo_tts") return ModelFamily::minicpmo_tts;
#endif
        return ModelFamily::unsupported;
    }
    if (target.is_metal()) {
#if TYLOINFER_HAS_METAL
        return qwen_family;
#else
        return ModelFamily::unsupported;
#endif
    }
    return ModelFamily::unsupported;
}

std::string unavailable_reason_for_graph(
    const tyloi::ModelGraph& model_graph, tyloi::Device target) {
    if (target.is_cpu())
        return "CPU execution has no model loader in this build";
    if (target.is_cuda()) {
#if !TYLOINFER_HAS_CUDA
        return "CUDA execution backend is unavailable in this build";
#endif
    } else if (target.is_metal()) {
#if !TYLOINFER_HAS_METAL
        return "Metal execution backend is unavailable in this build";
#endif
    } else {
        return "requested execution device is unavailable in this build";
    }
    if (select_family(model_graph, target) == ModelFamily::unsupported)
        return "model backbone '" + model_graph.backbone +
            "' has no loader for the selected execution backend";
    return {};
}

} // namespace

namespace tyloi::models {

std::string unavailable_reason(const ModelSource& source, Device target) {
    try {
        return unavailable_reason_for_graph(graph(source), target);
    } catch (const std::exception& error) {
        return std::string("invalid model graph: ") + error.what();
    }
}

void load(
    std::unique_ptr<CausalModel>& owner,
    std::shared_ptr<const ModelSource> source,
    std::string_view definition,
    const weights::Loader& weights,
    LayerPlacementConfig& placement,
    LoadOptions options) {
    const auto model_graph = graph(*source);
    const auto family = select_family(model_graph, weights.placement.target);
    if (!options.dflash2_draft_path.empty() && ((family != ModelFamily::qwen35 && family != ModelFamily::qwen4_exp) ||
            !weights.placement.target.is_cuda() || !options.optional_components))
        throw std::invalid_argument("DFlash2 requires a CUDA Qwen3.5 or Qwen4 target");
    switch (family) {
#if TYLOINFER_HAS_CUDA
    case ModelFamily::glm_dsa:
        ::tyloi::cuda::glm_dsa::load(owner, std::move(source), definition, weights, placement, options);
        return;
    case ModelFamily::glm5_next:
        ::tyloi::cuda::glm5_next::load(owner, std::move(source), definition, weights, placement, options);
        return;
    case ModelFamily::deepseek_v4:
        ::tyloi::cuda::deepseek_v4::load(owner, std::move(source), definition, weights, placement, options);
        return;
    case ModelFamily::gemma4:
        if (!options.cpu_offload_layers.empty())
            throw std::invalid_argument("Gemma4 does not support --cpu-offload-layers");
        ::tyloi::cuda::gemma4::load(
            owner, std::move(source), definition, options.context, weights, placement,
            options.gpu_layers, options.expert_parallel);
        return;
#endif
    case ModelFamily::qwen35: {
        int cpu_layers = 0;
        auto model = ::tyloi::models::qwen35::load(
            std::move(source), definition, weights, placement, options, cpu_layers);
        owner = std::move(model);
        if (options.optional_components)
            ::tyloi::models::qwen35::load_components(
                static_cast<::tyloi::models::qwen35::Model&>(*owner), weights, placement,
                cpu_layers, !options.cpu_offload_layers.empty(), options.vision_resident, options.preallocate,
                options.dflash2_draft_path);
        return;
    }
    case ModelFamily::qwen4_exp:
        qwen4_exp::load(owner, std::move(source), definition, weights, placement, options);
        return;
#if TYLOINFER_HAS_CUDA
    case ModelFamily::deepseek_v41:
        ::tyloi::cuda::deepseek_v41::load(owner, std::move(source), definition, weights, placement, options);
        return;
    case ModelFamily::nemotron_h:
        ::tyloi::cuda::nemotron_h::load(owner, std::move(source), definition, weights, placement, options);
        return;
    case ModelFamily::minicpmo45:
        ::tyloi::cuda::minicpmo45::text::load(owner, std::move(source), definition, weights, placement, options);
        return;
    case ModelFamily::minicpmo_tts:
        ::tyloi::cuda::minicpmo45::load(owner, std::move(source), definition, weights, placement, options);
        return;
#endif
    case ModelFamily::unsupported:
        break;
    }
    throw std::runtime_error(unavailable_reason_for_graph(model_graph, weights.placement.target));
}

std::unique_ptr<CausalModel> load(
    std::shared_ptr<const ModelSource> source, std::string_view definition,
    const weights::Loader& weights, LayerPlacementConfig& placement, LoadOptions options) {
    std::unique_ptr<CausalModel> owner;
    load(owner, std::move(source), definition, weights, placement, std::move(options));
    return owner;
}

} // namespace tyloi::models

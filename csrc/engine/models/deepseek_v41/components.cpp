#include "engine/models/deepseek_v41/components.h"

#include "engine/models/deepseek_v41/dspark.h"
#include "engine/models/deepseek_v41/ops.h"
#include "engine/models/deepseek_v41/vision.h"
#include "engine/models/common/mtp.h"

#include <stdexcept>

namespace tyloi::cuda::deepseek_v41 {

std::unique_ptr<tyloi::models::Mtp> load_components(
        Model& model, const tyloi::weights::Loader& weights, bool load_optional_components
        ) {
    if (!load_optional_components) return {};
    for (const auto& component : model.graph.components)
        if (component.kind != "text" && component.kind != "vision" && component.kind != "predictor")
            throw std::runtime_error("unsupported DeepSeek-V4.1 component: " + component.kind);
    if (model.graph.has_component("vision")) {
        model.vision = std::make_unique<deepseek_v41_runtime::Vision>(
            deepseek_v41_runtime::Vision::load(model.source, weights,
                model.config));
    }
    if (!model.graph.has_component("predictor")) return {};
    TYLOINFER_RUNTIME_CHECK(model.supports_suffix_speculation() && model.tables.engram_parameters,
        "invalid DeepSeek-V4.1 DSpark model");
    return deepseek_v41_runtime::load_dspark_if_present(
        weights, model.config, model.embed, model.lm_head,
        model.metadata.final_logit_softcapping);
}

} // namespace tyloi::cuda::deepseek_v41

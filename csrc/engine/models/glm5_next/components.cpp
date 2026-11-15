#include "engine/models/glm5_next/components.h"

#include "engine/models/glm5_next/forward.h"
#include "engine/models/glm5_next/model.h"
#include "engine/models/glm5_next/mtp.h"
#include "engine/weights/loader.h"

#include <stdexcept>
#include <utility>

namespace tyloi::cuda::glm5_next {

void load_components(
        tyloi::cuda::glm5_next::Model& model,
        const tyloi::weights::Loader& weights,
        bool load_optional_components) {
    if (!load_optional_components) return;
    if (!model.graph.has_component("predictor")) return;
    TYLOINFER_RUNTIME_CHECK(
        tyloi::cuda::glm5_next::supports_speculation(model),
        "invalid Flash-Next predictor model");
    auto predictor = std::make_unique<tyloi::cuda::glm5_next::Mtp>(
        model.embed, weights.placement.target.index);
    auto* candidate = predictor.get();
    model.mtp = std::move(predictor);
    if (!candidate->load_weights(weights, model.config)) model.mtp.reset();
}

} // namespace tyloi::cuda::glm5_next

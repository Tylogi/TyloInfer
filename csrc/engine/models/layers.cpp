#include "engine/models/layers.h"
#include "engine/models/load.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/qwen35/layers.h"
#if TYLOINFER_HAS_CUDA
#include "engine/models/gemma4/layers.h"
#endif

#include <stdexcept>
#include <utility>

namespace tyloi::models {

void load_layers(std::unique_ptr<Layers>& owner, std::shared_ptr<const ModelSource> source,
        std::string_view definition, const weights::Loader& weights,
        LayerPlacementConfig& placement, const LoadOptions& options) {
    if (owner) throw std::invalid_argument("layer owner must be empty");
    if (!source) throw std::invalid_argument("layer-wise loading requires a model source");
    const auto name = graph(*source).backbone;
    if (name == "qwen3_5" || name == "generic_qwen")
        return qwen35::load_layers(owner, std::move(source), definition, weights, placement, options);
#if TYLOINFER_HAS_CUDA
    if (name == "gemma4")
        return cuda::gemma4::load_layers(owner, std::move(source), definition, weights, placement, options);
#endif
    throw std::runtime_error("unsupported layer-wise model: " + name);
}

} // namespace tyloi::models

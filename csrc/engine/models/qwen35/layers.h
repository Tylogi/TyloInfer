#pragma once

#include "engine/models/layers.h"
#include "engine/models/qwen35/model.h"
#include "engine/models/qwen35/state.h"

namespace tyloi::cuda::qwen35 {

Tensor forward_layer(const Layer&, LayerState&, Tensor, const tyloi::models::AttentionStep&,
    std::int64_t confirmed_prefix, const tyloi::ops::RopeTable&,
    const tyloi::models::ForwardOptions&, bool transposed);

/** Assemble Qwen layer-wise computation with bounded input and output weight reads. */
void load_layers(std::unique_ptr<tyloi::models::Layers>& owner,
    std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
    const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
    const tyloi::models::LoadOptions& options);

} // namespace tyloi::cuda::qwen35

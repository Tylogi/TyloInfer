#pragma once

#include "engine/ops.h"

#include <cstdint>
#include <memory>
#include <string_view>

struct LayerPlacementConfig;

namespace tyloi {
class ModelSource;
namespace weights { struct Loader; }

namespace models {
class CausalModel;

struct LoadOptions {
    std::int64_t context = 0;
    int gpu_layers = -1;
    bool expert_parallel = false;
    bool recurrent_transposed = false;
    bool vision_resident = false;
    bool mirror_scalars = true;
    tyloi::ops::YarnScaling yarn;
};

std::unique_ptr<CausalModel> load(
    std::shared_ptr<const ModelSource> source, std::string_view definition,
    const weights::Loader& weights, LayerPlacementConfig& placement,
    LoadOptions options = {});

} // namespace models
} // namespace tyloi

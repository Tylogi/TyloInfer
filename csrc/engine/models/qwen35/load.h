#pragma once
#include "engine/models/source.h"

#include "engine/models/qwen35/model.h"
#include "engine/models/qwen35/predictor.h"
#include "engine/ops.h"
#include "engine/state/kv_view.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

struct LayerPlacementConfig;
struct TransformerWeightLoader;
namespace tyloi::weights { struct Loader; }
namespace tyloi::models { struct LoadOptions; }

namespace tyloi::cuda::qwen35 {

struct LoadOptions {
    int gpu_layers = -1;
    bool expert_parallel = false;
    bool mirror_scalars = true;
    bool recurrent_transposed = false;
    bool load_layers = true;
    bool load_input_output = true;
};

Model read(std::shared_ptr<const tyloi::ModelSource> source,
           std::string_view definition, std::int64_t context);
std::unique_ptr<Model> load(std::shared_ptr<const tyloi::ModelSource> source,
                            std::string_view definition, const ::tyloi::weights::Loader& weights,
                            LayerPlacementConfig& placement, const tyloi::models::LoadOptions& options,
                            int& cpu_layers);
void load(Model& model, const ::tyloi::weights::Loader& weights,
          LayerPlacementConfig& placement, const tyloi::ops::YarnScaling& yarn,
          int& cpu_layers, const LoadOptions& options);

void load_vision(Model& model, const ::tyloi::weights::Loader& weights, bool resident = false);

Layer load_layer(const ::tyloi::weights::Loader&, const Config&,
                 int index, const std::string& type, bool mirror_scalars = true,
                 std::string_view tensor_root = "model");

std::optional<tyloi::models::qwen35::Predictor> load_predictor(
    TransformerWeightLoader&, const Config&, tyloi::state::KvQuantization);

} // namespace tyloi::cuda::qwen35

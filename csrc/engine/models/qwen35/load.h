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
namespace tyloi::weights { struct Loader; }
namespace tyloi::models { struct LoadOptions; }

namespace tyloi::models::qwen35 {

struct LoadOptions {
    int gpu_layers = -1;
    bool expert_parallel = false;
    bool mirror_scalars = true;
    bool recurrent_transposed = false;
    bool load_layers = true;
    bool load_input_output = true;
    tyloi::state::KvQuantization kv_quantization;
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
void load_components(Model& model, const ::tyloi::weights::Loader& weights,
                     const LayerPlacementConfig& placement, int cpu_layers, bool cpu_offload,
                     bool resident, bool preallocate);

Layer load_layer(const ::tyloi::weights::Loader&, const Config&,
                 int index, const std::string& type, bool mirror_scalars = true,
                 std::string_view tensor_root = "model", tyloi::state::KvQuantization quantization = {});
void load_layer(Layer&, const ::tyloi::weights::Loader&, const Config&,
                int index, const std::string& type, bool mirror_scalars = true,
                std::string_view tensor_root = "model", tyloi::state::KvQuantization quantization = {});

} // namespace tyloi::models::qwen35

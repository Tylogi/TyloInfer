#pragma once

#include "engine/models/minicpmo45/layer.h"
#include "engine/models/source/model_config.h"
#include "engine/ops/ops.h"
#include "engine/models/metadata.h"
#include "engine/models/causal_model.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/source/model.h"

#include <memory>
#include <unordered_map>
#include <vector>

struct LayerPlacementConfig;
namespace tyloi::weights { struct Loader; }
namespace tyloi::models { struct LoadOptions; }

namespace tyloi::cuda::minicpmo45 {

// The standalone TTS checkpoint uses offset RMS weights, unlike composite Tts.
struct Model
    : tyloi::models::CausalModel
{
    tyloi::models::Info info() const override;
    MfqModelCapabilities capabilities() const override;
    tyloi::engine::Metrics metrics() const override;
    std::unique_ptr<tyloi::models::State> make_state() const override;
    tyloi::StepSequence<tyloi::models::Prepared> prepare(const std::vector<std::int64_t>&,
        const MfqMultimodalInput&, tyloi::models::State&) const override;
    Tensor embedding(const Tensor&, int primary = 0) const override;
    Tensor normalize(const Tensor&, std::int64_t batch, std::int64_t tokens) const override;
    Tensor softcap(const Tensor&) const override;
    Tensor logits(const Tensor&, bool capped = true, tyloi::models::LogitPrecision precision = tyloi::models::LogitPrecision::model) const override;
    Tensor forward(tyloi::models::Input, tyloi::models::State&,
                   const tyloi::models::ForwardOptions& = {}) const override;
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    tyloi::models::ModelConfig config;
    tyloi::models::CausalLmMetadata metadata;
    tyloi::ops::RopeTable rope, cpu_rope;
    std::unordered_map<int, tyloi::ops::RopeTable> device_ropes;
    ::tyloi::Tensor embed, lm_head;
    ::tyloi::Tensor output_norm;
    std::vector<Layer> layers;
};

Model read(std::shared_ptr<const tyloi::ModelSource>, std::string_view definition, int64_t context);
void load(Model&, const tyloi::weights::Loader&, const LayerPlacementConfig&, int cpu_layers,
          const tyloi::ops::YarnScaling&);
void load(std::unique_ptr<tyloi::models::CausalModel>&, std::shared_ptr<const tyloi::ModelSource>,
          std::string_view definition, const tyloi::weights::Loader&, LayerPlacementConfig&,
          const tyloi::models::LoadOptions&);

} // namespace tyloi::cuda::minicpmo45

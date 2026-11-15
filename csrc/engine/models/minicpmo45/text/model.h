#pragma once

#include "engine/models/minicpmo45/layer.h"
#include "engine/models/minicpmo45/config.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/models/metadata.h"
#include "engine/models/causal_model.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/source.h"

#include <memory>
#include <unordered_map>
#include <vector>

struct MiniCPMO45VisionEncoder;
struct MiniCPMO45Resampler;
struct MiniCPMO45AudioEncoder;
namespace tyloi::cuda::minicpmo45 { struct Tts; }

namespace tyloi::cuda::minicpmo45::text {
using Config = tyloi::models::minicpmo45::Config;

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
    Config config;
    tyloi::models::CausalLmMetadata metadata;
    int primary = 0;
    ::tyloi::Tensor embed, lm_head;
    ::tyloi::Tensor output_norm;
    tyloi::ops::RopeTable rope, cpu_rope;
    std::unordered_map<int, tyloi::ops::RopeTable> device_ropes;
    std::vector<Layer> layers;
    std::shared_ptr<MiniCPMO45VisionEncoder> vision;
    std::shared_ptr<MiniCPMO45Resampler> resampler;
    std::shared_ptr<MiniCPMO45AudioEncoder> audio;
    std::shared_ptr<Tts> tts;
};
} // namespace tyloi::cuda::minicpmo45::text

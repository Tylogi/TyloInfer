#pragma once

#include "engine/models/glm_dsa/config.h"
#include "engine/models/metadata.h"
#include "engine/models/causal_model.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/source/model.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/models/common/ffn.h"
#include "engine/weights/moe/experts.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace tyloi::cuda::glm_dsa {
using DenseTensor = ::tyloi::DenseTensor;
using Config = tyloi::models::glm_dsa::Config;

struct Layer {
    Config config;
    int layer = -1;
    bool full_indexer = false;
    int device = 0;
    ::tyloi::Tensor attn_norm, ffn_norm, q_a_norm, kv_a_norm, index_k_norm, index_k_bias;
    std::vector<::tyloi::Tensor> input_proj, q_proj;
    ::tyloi::Tensor embed_q, unembed_out;
    ::tyloi::Tensor o_proj;
    ::tyloi::models::Ffn ffn;
};

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
    Tensor normalize(const Tensor&, std::int64_t, std::int64_t) const override;
    Tensor softcap(const Tensor&) const override;
    Tensor logits(const Tensor&, bool = true, tyloi::models::LogitPrecision precision = tyloi::models::LogitPrecision::model) const override;
    Tensor forward(tyloi::models::Input, tyloi::models::State&,
                   const tyloi::models::ForwardOptions& = {}) const override;
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    Config config;
    tyloi::models::CausalLmMetadata metadata;
    ::tyloi::Tensor embed, lm_head;
    ::tyloi::Tensor output_norm;
    tyloi::ops::RopeTable rope;
    std::unordered_map<int, tyloi::ops::RopeTable> device_ropes;
    std::vector<Layer> layers;
};
} // namespace tyloi::cuda::glm_dsa

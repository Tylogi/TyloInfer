#pragma once

#include "engine/models/minicpmo45/layer.h"
#include "engine/models/minicpmo45/config.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/models/metadata.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/models/causal_model.h"
#endif
#include "engine/models/common/capabilities.h"
#include "engine/models/source.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace tyloi::cuda::minicpmo45::text {
using Config = tyloi::models::minicpmo45::Config;

struct Model
#ifdef TYLOINFER_NATIVE_CUDA
    : tyloi::models::CausalModel
#endif
{
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::models::Info info() const override;
    MfqModelCapabilities capabilities() const override;
    tyloi::engine::Metrics metrics() const override;
    std::unique_ptr<tyloi::models::State> make_state() const override;
    tyloi::StepSequence<tyloi::models::Prepared> prepare(const std::vector<std::int64_t>&,
        const MfqMultimodalInput&, tyloi::models::State&) const override;
    Tensor embedding(const Tensor&, int primary = 0) const override;
    Tensor normalize(const Tensor&, std::int64_t batch, std::int64_t tokens) const override;
    Tensor softcap(const Tensor&) const override;
    Tensor logits(const Tensor&, bool capped = true) const override;
    Tensor forward(tyloi::models::Input, tyloi::models::State&,
                   const tyloi::models::ForwardOptions& = {}) const override;
#endif
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    Config config;
    tyloi::models::CausalLmMetadata metadata;
    ::tyloi::Tensor embed, lm_head;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor output_norm;
#else
    DenseTensor output_norm;
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::ops::RopeTable rope, cpu_rope;
    std::unordered_map<int, tyloi::ops::RopeTable> device_ropes;
#else
    RopeCache rope, cpu_rope;
    std::unordered_map<int, RopeCache> device_ropes;
#endif
    std::vector<Layer> layers;
};
} // namespace tyloi::cuda::minicpmo45::text

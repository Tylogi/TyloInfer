#pragma once

#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/models/common/attention.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/models/nemotron_h/config.h"
#include "engine/models/metadata.h"
#include "engine/models/common/capabilities.h"
#include "engine/models/source.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/models/causal_model.h"
#endif

#include <array>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

namespace tyloi::cuda::nemotron_h {

using DenseTensor = ::tyloi::DenseTensor;
using Config = tyloi::models::nemotron_h::Config;

struct Mamba {
    int layer = -1;
    int64_t mamba_heads = 0, mamba_head_dim = 0, ssm_state_size = 0;
    int64_t mamba_groups = 0, conv_kernel = 0;
    double rms_norm_eps = 1e-6, time_step_limit_lo = 0.0, time_step_limit_hi = 0.0;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor mixer_norm;
#else
    DenseTensor mixer_norm;
#endif
    ::tyloi::Tensor in_proj;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor conv_weight, conv_bias, a_log, d, dt_bias, scan_norm;
#else
    DenseTensor conv_weight, conv_bias, a_log, d, dt_bias, scan_norm;
#endif
    ::tyloi::Tensor out_proj;
};

struct Moe {
    int layer = -1;
    double rms_norm_eps = 1e-6;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor ffn_norm;
#else
    DenseTensor ffn_norm;
#endif
    Ffn ffn;
};

struct Layer {
    std::variant<Mamba, FullAttention, Moe> weights;
    int device = 0;
};

struct Model
#ifdef TYLOINFER_NATIVE_CUDA
    : tyloi::models::CausalModel
#endif
{
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
    std::vector<Layer> layers;
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::models::Info info() const override;
    bool supports_speculation() const noexcept override;
    MfqModelCapabilities capabilities() const override;
    tyloi::engine::Metrics metrics() const override;
    std::unique_ptr<tyloi::models::State> make_state() const override;
    tyloi::StepSequence<tyloi::models::Prepared> prepare(const std::vector<std::int64_t>&,
        const MfqMultimodalInput&, tyloi::models::State&) const override;
    ::tyloi::Tensor embedding(const tyloi::Tensor& ids, int primary = 0) const override;
    ::tyloi::Tensor normalize(const tyloi::Tensor&, std::int64_t, std::int64_t) const override;
    ::tyloi::Tensor softcap(const tyloi::Tensor&) const override;
    ::tyloi::Tensor logits(const tyloi::Tensor&, bool capped = true) const override;
    ::tyloi::Tensor forward(tyloi::models::Input, tyloi::models::State&,
        const tyloi::models::ForwardOptions& = {}) const override;
#endif
};


} // namespace tyloi::cuda::nemotron_h

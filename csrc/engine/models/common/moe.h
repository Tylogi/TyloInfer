#pragma once

#include "engine/models/common/mlp.h"

namespace tyloi::models {

struct Moe {
    // One fused Gate/Up bank or separate Gate and Up banks, in that order.
    std::vector<Tensor> gate_up;
    Tensor down, router, router_bias, shared_gate, latent_in, latent_out;
    Tensor hash_ids;
    Mlp shared;
    ops::MoeRouting routing;
    ops::MlpActivation activation = ops::MlpActivation::silu_gated;
    std::int64_t top_k = 0;
    double swiglu_limit = 0;
    int layer = -1;
};

Tensor moe(const Moe& weights, const Tensor& input, bool serial_experts = false,
           const Tensor& ids = {}, const Tensor& router_bias = {});

} // namespace tyloi::models

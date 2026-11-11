#pragma once

#include "engine/models/common/mlp.h"

namespace tyloi::models {

struct Moe {
    // One fused Gate/Up bank or separate Gate and Up banks, in that order.
    std::vector<Tensor> gate_up;
    Tensor down, router, router_bias, shared_gate, latent_in, latent_out;
    Mlp shared;
    ops::MoeRouting routing;
    std::int64_t top_k = 0;
    double swiglu_limit = 0;
    int layer = -1;
};

Tensor moe(const Moe& weights, const Tensor& input, bool serial_experts = false);

} // namespace tyloi::models

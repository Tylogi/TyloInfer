#pragma once

#include "config.h"

namespace mfq::models::lfm3::d1 {

// The checkpoint shapes are shared between the CUDA and Metal executors.
template<class Load>
void load_layer_weights(const Config& config, std::size_t layer, Load load) {
    const std::int64_t hidden = config.hidden;
    const std::int64_t intermediate = config.intermediate;
    load("operator_norm.weight", {hidden});
    load("ffn_norm.weight", {hidden});
    if (config.sandwich) {
        load("post_operator_norm.weight", {hidden});
        load("post_ffn_norm.weight", {hidden});
    }
    load("feed_forward.w1.weight", {intermediate, hidden});
    load("feed_forward.w3.weight", {intermediate, hidden});
    load("feed_forward.w2.weight", {hidden, intermediate});
    if (config.layers.at(layer) == "conv") {
        load("conv.conv.weight", {hidden, 1, config.convolution});
        load("conv.in_proj.weight", {3 * hidden, hidden});
        load("conv.out_proj.weight", {hidden, hidden});
        return;
    }
    const auto query = std::int64_t(config.heads) * config.head_dim;
    const auto key = std::int64_t(config.kv_heads) * config.head_dim;
    load("self_attn.q_proj.weight", {query, hidden});
    load("self_attn.k_proj.weight", {key, hidden});
    load("self_attn.v_proj.weight", {key, hidden});
    load("self_attn.gate_proj.weight", {config.heads, hidden});
    load("self_attn.o_proj.weight", {hidden, query});
    load("self_attn.q_norm.weight", {config.head_dim});
    load("self_attn.k_norm.weight", {config.head_dim});
}

} // namespace mfq::models::lfm3::d1

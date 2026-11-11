#pragma once

#include "engine/models/qwen35/config.h"
#include "engine/ops.h"
#include "engine/models/common/load.h"
#include "engine/models/common/ffn.h"

namespace tyloi::models::qwen35 {

struct LinearWeightNames {
    std::string qkv, qk, value, gate, alpha, beta;
};

template <class Loader>
models::Ffn load_ffn(Loader& ops, const Config& config, int layer, std::string_view tensor_root) {
    const std::string prefix =
        std::string(tensor_root) + ".block." + std::to_string(layer) + ".mlp.";
    if (!ops.has(prefix + "experts.gate_up.weight") &&
        !ops.has(prefix + "experts.gate.weight") &&
        !ops.has(prefix + "experts.up.weight") &&
        !ops.has(prefix + "experts.down.weight")) {
        return models::load_dense_ffn<Mlp>(ops, config, prefix);
    }
    if (config.num_experts <= 0 || config.num_experts_per_tok <= 0 ||
        config.moe_intermediate_size <= 0 || config.shared_expert_intermediate_size <= 0) {
        throw std::runtime_error("Qwen MoE config fields are missing");
    }

    Moe ffn;
    ffn.layer = layer;
    const bool split = models::split_experts(ops, prefix, layer);
    if (split) {
        ffn.gate_up.push_back(ops.expert(ExpertProjection::gate, prefix + "experts.gate.weight", layer));
        ffn.gate_up.push_back(ops.expert(ExpertProjection::up, prefix + "experts.up.weight", layer));
    } else {
        ffn.gate_up.push_back(ops.expert(ExpertProjection::gate_up, prefix + "experts.gate_up.weight", layer));
    }
    ffn.down = ops.expert(ExpertProjection::down, prefix + "experts.down.weight", layer);
    if (models::has_moe_latent(ops, prefix, layer)) {
        ffn.latent_in = ops.linear(prefix + "latent.down.weight");
        ffn.latent_out = ops.linear(prefix + "latent.up.weight");
    }
    ffn.router = ops.router_parameter(prefix + "router.weight");
    if (ops.has(prefix + "router.bias"))
        ffn.router_bias = ops.router_parameter(prefix + "router.bias");
    ffn.shared.down = ops.linear(prefix + "shared_expert.down.weight");
    ffn.shared.gate_up = ops.gate_up({prefix + "shared_expert.gate.weight",
                                     prefix + "shared_expert.up.weight"}, ffn.shared.down);
    ffn.shared_gate = ops.router_parameter(prefix + "shared_expert.router.weight");
    ffn.top_k = config.num_experts_per_tok;
    ffn.routing = config.routing();
    const bool routed_gate_shapes =
        split
            ? ffn.gate_up[0] && ffn.gate_up[1] && ffn.gate_up[0].size(0) == config.num_experts &&
                  ffn.gate_up[1].size(0) == config.num_experts &&
                  ffn.gate_up[0].size(2) == config.hidden_size &&
                  ffn.gate_up[1].size(2) == config.hidden_size &&
                  ffn.gate_up[0].size(1) == config.moe_intermediate_size &&
                  ffn.gate_up[1].size(1) == config.moe_intermediate_size
            : ffn.gate_up[0] && ffn.gate_up[0].size(0) == config.num_experts &&
                  ffn.gate_up[0].size(2) == config.hidden_size &&
                  ffn.gate_up[0].size(1) == 2 * config.moe_intermediate_size;
    if (!routed_gate_shapes || !ffn.down || ffn.down.size(0) != config.num_experts ||
        ffn.down.size(2) != config.moe_intermediate_size ||
        ffn.down.size(1) != config.hidden_size || ops.shape(ffn.router).size() != 2 ||
        ops.shape(ffn.router)[0] != config.num_experts ||
        ops.shape(ffn.router)[1] != config.hidden_size || ops.shape(ffn.shared_gate).size() != 2 ||
        ops.shape(ffn.shared_gate)[0] != 1 || ops.shape(ffn.shared_gate)[1] != config.hidden_size) {
        throw std::runtime_error("Qwen MoE tensor shapes disagree with config at layer " +
                                 std::to_string(layer));
    }
    return ffn;
}

} // namespace tyloi::models::qwen35

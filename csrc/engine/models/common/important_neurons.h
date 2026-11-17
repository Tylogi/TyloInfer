#pragma once

#include "engine/weights/loader.h"
#include "engine/models/common/mlp.h"
#include "engine/weights/source/read.h"
#include <memory>
#include <stdexcept>
#include <string>

namespace tyloi::models {
inline void load_important_branch(const tyloi::weights::Loader& weights,
    int64_t hidden_size, int64_t intermediate_size, Mlp& f,
    const std::string& down_name,
    const std::string& gate_name, const std::string& up_name) {
    const std::string down_high = down_name + ".in_high";
    const std::string gate_high = gate_name + ".in_high";
    const std::string up_high = up_name + ".in_high";
    const bool has_down = tyloi::has_tensor(weights.source, down_high);
    const bool has_gate = tyloi::has_tensor(weights.source, gate_high);
    const bool has_up = tyloi::has_tensor(weights.source, up_high);
    if (!has_down && !has_gate && !has_up) {
        return;
    }
    if (!has_down || !has_gate || !has_up) {
        throw std::runtime_error(
            "important-neuron FFN requires matching gate/up/down .in_high records");
    }
    auto high = std::make_unique<Mlp>();
    high->down = weights.weight(down_high, TensorParallelAxis::Input);
    high->gate_up = weights.gate_up({gate_high, up_high}, high->down);
    high->swiglu_limit = f.swiglu_limit;
    high->activation = f.activation;

    if (f.gate_up.size() != 2 || high->gate_up.size() != 2 ||
        f.gate_up[0].size(0) != f.gate_up[1].size(0) || high->gate_up[0].size(0) != high->gate_up[1].size(0) ||
        f.down.size(0) != hidden_size || high->down.size(0) != hidden_size ||
        f.down.size(1) != f.gate_up[0].size(0) ||
        high->down.size(1) != high->gate_up[0].size(0) ||
        f.down.size(1) + high->down.size(1) != intermediate_size) {
        throw std::runtime_error("important-neuron FFN tensor shapes disagree with model config");
    }
    f.important_neurons = std::move(high);
}

} // namespace tyloi::models

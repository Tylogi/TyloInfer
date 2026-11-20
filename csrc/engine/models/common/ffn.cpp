#include "engine/models/common/ffn.h"
#include "engine/ops/ffn.h"

#include <stdexcept>
#include <type_traits>

namespace tyloi::models {
namespace {

Tensor dense_ffn(const Mlp& weights, const Tensor& input, bool bf16) {
    if (!weights.down) throw std::runtime_error("FFN down projection is missing");
    if (bf16)
        return ops::Ffn::bf16(input, weights.gate_up, weights.down,
            weights.activation, weights.swiglu_limit);
    if (weights.important_neurons) {
        const auto& extra = *weights.important_neurons;
        if (!extra.down) throw std::runtime_error("FFN down projection is missing");
        return ops::Ffn::additive(input, weights.gate_up, weights.down,
            weights.activation, weights.swiglu_limit, extra.gate_up, extra.down,
            extra.activation, extra.swiglu_limit);
    }
    return ops::Ffn::dense(input, weights.gate_up, weights.down,
        weights.activation, weights.swiglu_limit);
}

Tensor project_ffn(const Mlp& weights, const Tensor& input, bool bf16, bool) {
    return dense_ffn(weights, input, bf16);
}

Tensor project_ffn(const Moe& weights, const Tensor& input, bool bf16, bool serial_experts) {
    if (bf16) throw std::runtime_error("BF16 routed FFN is unsupported");
    return moe(weights, input, serial_experts);
}

template <class Weights>
Tensor full(const Weights& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset, bool bf16, bool trace,
    ops::Trace* trace_output, bool serial_experts = false) {
    const auto batch = residual.size(0), tokens = residual.size(1), width = residual.size(2);
    auto pair = ops::residual_norm(residual, attention_output, norm, epsilon, offset, bf16);
    auto input = ops::reshape(pair[1], {batch * tokens, width});
    if (trace && trace_output)
        trace_output->emplace_back("qwen.ffn_norm", ops::clone(ops::contiguous(
            ops::cast(ops::reshape(input, {batch, tokens, width}), ScalarType::float32))));
    if constexpr (!std::is_same_v<Weights, Moe>) {
        if (!bf16 && !weights.important_neurons) {
            auto flat_residual = ops::reshape(pair[0], {batch * tokens, width});
            if (auto fused = ops::Ffn::residual(input, flat_residual, weights.gate_up,
                    weights.down, weights.activation, weights.swiglu_limit))
                return ops::reshape(*fused, {batch, tokens, width});
        }
    }
    auto output = ops::reshape(project_ffn(weights, input, bf16, serial_experts), {batch, tokens, width});
    if (trace && trace_output)
        trace_output->emplace_back("qwen.ffn_output", ops::clone(ops::contiguous(
            ops::cast(output, ScalarType::float32))));
    return ops::residual_add(pair[0], output);
}

template <class Weights>
Tensor linear(const Weights& weights, const std::array<Tensor, 2>& pair, bool serial_experts) {
    const auto& residual = pair[0];
    const auto batch = residual.size(0), tokens = residual.size(1), width = residual.size(2);
    auto input = ops::reshape(pair[1], {batch * tokens, width});
    if constexpr (!std::is_same_v<Weights, Moe>) {
        if (!weights.important_neurons) {
            auto flat_residual = ops::reshape(pair[0], {batch * tokens, width});
            if (auto fused = ops::Ffn::residual(input, flat_residual, weights.gate_up,
                    weights.down, weights.activation, weights.swiglu_limit, true))
                return ops::reshape(*fused, {batch, tokens, width});
        }
    }
    auto output = ops::reshape(project_ffn(weights, input, false, serial_experts), {batch, tokens, width});
    return ops::recurrent_add(pair[0], output);
}

} // namespace

Tensor mlp(const Mlp& weights, const Tensor& input) {
    if (weights.gate_up.empty() || !weights.down)
        throw std::logic_error("MLP weights are missing");
    return dense_ffn(weights, input, false);
}

Tensor full_ffn(const Mlp& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset, bool bf16, bool trace,
    ops::Trace* trace_output) {
    return full(weights, norm, residual, attention_output,
        epsilon, offset, bf16, trace, trace_output);
}

Tensor full_ffn(const Ffn& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset, bool bf16, bool trace,
    ops::Trace* trace_output, bool serial_experts) {
    return std::visit([&](const auto& value) {
        return full(value, norm,
            residual, attention_output, epsilon, offset, bf16, trace, trace_output,
            serial_experts);
    }, weights);
}

Tensor linear_ffn(const Ffn& weights, const std::array<Tensor, 2>& prepared, bool serial_experts) {
    return std::visit([&](const auto& value) {
        return linear(value, prepared, serial_experts);
    }, weights);
}

Tensor feed_forward(const Ffn& weights, const Tensor& input,
                    const Tensor& ids, const Tensor& router_bias) {
    return std::visit([&](const auto& value) -> Tensor {
        using Weight = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Weight, Mlp>)
            return mlp(value, input);
        else
            return moe(value, input, false, ids, router_bias);
    }, weights);
}

} // namespace tyloi::models

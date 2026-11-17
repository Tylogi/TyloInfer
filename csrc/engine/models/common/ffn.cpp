#include "engine/models/common/ffn_internal.h"

#include <type_traits>

namespace tyloi::models {

Tensor ffn_internal::feed_forward(const Mlp& weights, const Tensor& input, bool bf16, bool) {
    return dense_ffn(weights, input, bf16);
}

Tensor mlp(const Mlp& weights, const Tensor& input) {
    if (weights.gate_up.empty() || !weights.down)
        throw std::logic_error("MLP weights are missing");
    return ffn_internal::dense_ffn(weights, input, false);
}

Tensor full_ffn(const Mlp& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset, bool bf16, bool trace,
    ops::Trace* trace_output) {
    return ffn_internal::full(weights, norm, residual, attention_output,
        epsilon, offset, bf16, trace, trace_output);
}

Tensor linear_ffn(const Mlp& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset) {
    return ffn_internal::linear(weights, norm, residual, attention_output,
        epsilon, offset);
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

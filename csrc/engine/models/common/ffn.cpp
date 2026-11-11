#include "engine/models/common/ffn_internal.h"

namespace tyloi::models {

Tensor ffn_internal::feed_forward(const Mlp& weights, const Tensor& input, bool bf16) {
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
    return ffn_internal::full(weights, false, norm, residual, attention_output,
        epsilon, offset, bf16, trace, trace_output);
}

Tensor linear_ffn(const Mlp& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset) {
    return ffn_internal::linear(weights, false, norm, residual, attention_output,
        epsilon, offset);
}

} // namespace tyloi::models

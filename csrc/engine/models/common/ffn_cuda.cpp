#include "engine/models/common/ffn_internal.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"

#ifdef TYLOINFER_NATIVE_CUDA
namespace tyloi::models {

Tensor ffn_internal::feed_forward(const cuda::Ffn& weights, const Tensor& input, bool bf16) {
    if (weights.is_moe && !bf16) return weights.forward(input.dense());
    return dense_ffn(weights, input, bf16);
}

Tensor full_ffn(const cuda::Ffn& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset, bool bf16, bool trace,
    ops::Trace* trace_output) {
    return ffn_internal::full(weights, weights.is_moe, norm, residual, attention_output,
        epsilon, offset, bf16, trace, trace_output);
}

Tensor linear_ffn(const cuda::Ffn& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset) {
    return ffn_internal::linear(weights, weights.is_moe, norm, residual, attention_output,
        epsilon, offset);
}

} // namespace tyloi::models
#endif

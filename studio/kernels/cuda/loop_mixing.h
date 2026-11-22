#pragma once

#include "mfq_tensor_backend.h"

namespace mfq::cuda {

// [rows, states, hidden] and [queries, hidden] -> [rows, queries, hidden].
mfq_tensor_backend::Tensor residual_depth_mix(const mfq_tensor_backend::Tensor& states,
    const mfq_tensor_backend::Tensor& queries, float rms_epsilon);

mfq_tensor_backend::Tensor residual_depth_mix(const std::vector<mfq_tensor_backend::Tensor>& bank,
    const mfq_tensor_backend::Tensor& hidden,const mfq_tensor_backend::Tensor& queries,float rms_epsilon);

mfq_tensor_backend::Tensor normalize_loop_attention_projection(const mfq_tensor_backend::Tensor& input,
    const mfq_tensor_backend::Tensor& weights,float threshold_or_epsilon,bool clip);
mfq_tensor_backend::Tensor blend_loop_message(const mfq_tensor_backend::Tensor& current,
    const mfq_tensor_backend::Tensor& message,const mfq_tensor_backend::Tensor& gain,
    const mfq_tensor_backend::Tensor& forget);

// Causal depthwise convolution with a residual and optional learned taps per head.
// History [batch, taps-1, channels] advances by input.size(1) tokens.
mfq_tensor_backend::Tensor residual_short_convolution(mfq_tensor_backend::Tensor& history,
    const mfq_tensor_backend::Tensor& input, const mfq_tensor_backend::Tensor& weights,
    const mfq_tensor_backend::Tensor& dynamic_taps = {});
void retain_convolution_prefix(mfq_tensor_backend::Tensor& history,
    const mfq_tensor_backend::Tensor& previous,const mfq_tensor_backend::Tensor& inputs,
    const mfq_tensor_backend::Tensor& accepted);

}  // namespace mfq::cuda

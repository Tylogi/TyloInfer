#pragma once

#include "mfq_tensor_backend.h"

namespace mfq::cuda {

struct IndependentHcMix {
    mfq_tensor_backend::Tensor branch;
    mfq_tensor_backend::Tensor post;
};

struct TwoStreamMhcMix {
    mfq_tensor_backend::Tensor branch;
    mfq_tensor_backend::Tensor post;
    mfq_tensor_backend::Tensor combination;
};

// The router projects all streams jointly. With one scale the operator only
// reduces the streams for the prediction head. Two scales also emit post gates.
IndependentHcMix independent_hc_prepare(
    const mfq_tensor_backend::Tensor& streams, const mfq_tensor_backend::Tensor& function,
    const mfq_tensor_backend::Tensor& scale, const mfq_tensor_backend::Tensor& bias,
    float rms_epsilon, float gate_epsilon, float post_magnitude);

mfq_tensor_backend::Tensor independent_hc_merge(
    const mfq_tensor_backend::Tensor& streams, const mfq_tensor_backend::Tensor& branch,
    const mfq_tensor_backend::Tensor& post);

TwoStreamMhcMix two_stream_mhc_prepare(
    const mfq_tensor_backend::Tensor& streams, const mfq_tensor_backend::Tensor& function,
    const mfq_tensor_backend::Tensor& bias, float epsilon, float pre_scale,
    int sinkhorn_iterations, bool fixed_identity);

mfq_tensor_backend::Tensor two_stream_mhc_merge(
    const mfq_tensor_backend::Tensor& streams, const mfq_tensor_backend::Tensor& branch,
    const TwoStreamMhcMix& mix);

}  // namespace mfq::cuda

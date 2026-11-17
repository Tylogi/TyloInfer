#pragma once

#include "backend.h"

namespace tyloi::cuda::attention_ops {
using DenseTensor = ::tyloi::DenseTensor;

inline DenseTensor rms_norm(const DenseTensor& value,
                            const ::tyloi::Tensor& weight_tensor, double eps) {
    auto f = value.to(::tyloi::kFloat32);
    const auto& weight = weight_tensor.dense();
    return (f * ::tyloi::cuda::rsqrt((f * f).mean(-1, true) + eps) * weight.to(::tyloi::kFloat32))
        .to(value.scalar_type());
}

} // namespace tyloi::cuda::attention_ops

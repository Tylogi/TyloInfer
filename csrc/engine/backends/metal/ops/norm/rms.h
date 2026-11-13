#pragma once

#include "engine/tensor.h"

namespace tyloi::metal::ops {
Tensor rms_norm(const Tensor& input, const Tensor& weight, double eps, double offset, ScalarType output_type);

} // namespace tyloi::metal::ops

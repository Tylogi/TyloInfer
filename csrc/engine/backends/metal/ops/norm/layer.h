#pragma once

#include "engine/tensor.h"

namespace tyloi::metal::ops {
Tensor layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias, double epsilon);

} // namespace tyloi::metal::ops

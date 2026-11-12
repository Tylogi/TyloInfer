#pragma once

#include "engine/tensor.h"

namespace tyloi::metal::ops {

Tensor clone(const Tensor& value);
Tensor cast(const Tensor& value, ScalarType type);

}  // namespace tyloi::metal::ops

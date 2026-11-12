#pragma once

#include "engine/tensor.h"

#include <mlx/array.h>

namespace tyloi::metal {

// Arrays stay lazy; borrowed references live as long as the Tensor representation.
tyloi::Tensor tensor(mlx::core::array value);
const mlx::core::array& array(const tyloi::Tensor& value);

}  // namespace tyloi::metal

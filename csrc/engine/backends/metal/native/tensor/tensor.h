#pragma once

#include "engine/tensor.h"

#include <mlx/array.h>

namespace tyloi::metal {

// Arrays stay lazy; borrowed references live as long as the Tensor representation.
tyloi::Tensor tensor(mlx::core::array value);
const mlx::core::array& array(const tyloi::Tensor& value);
void replace(Tensor& value, mlx::core::array replacement);
mlx::core::Dtype dtype(ScalarType type);
ScalarType scalar_type(mlx::core::Dtype dtype);
std::vector<std::shared_ptr<TensorStorage>> backings(const mlx::core::array& value);
Tensor allocate(std::span<const std::int64_t> shape, ScalarType type);
Tensor copy_from_host(const void* data, std::span<const std::int64_t> shape, ScalarType type);
Tensor copy(const Tensor& value, Device target);

}  // namespace tyloi::metal

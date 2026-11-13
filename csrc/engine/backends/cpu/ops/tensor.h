#pragma once

#include "engine/tensor.h"

namespace tyloi::cpu::ops {
Tensor empty(std::span<const std::int64_t> shape, const TensorOptions& options);
Tensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options);
Tensor clone(const Tensor& source);
void copy_into(Tensor destination, const Tensor& source);
void fill(Tensor& value, double fill_value);
}

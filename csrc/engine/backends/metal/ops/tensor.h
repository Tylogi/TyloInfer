#pragma once

#include "engine/tensor.h"

#include <span>

namespace tyloi::metal::ops {

Tensor copy(const Tensor& value, Device target);
Tensor clone(const Tensor& value);
Tensor cast(const Tensor& value, ScalarType type);
Tensor reshape(const Tensor& value, std::span<const std::int64_t> shape);
Tensor transpose(const Tensor& value, std::int64_t first, std::int64_t second);
Tensor slice(const Tensor& value, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step = 1);
Tensor contiguous(const Tensor& value);

}  // namespace tyloi::metal::ops

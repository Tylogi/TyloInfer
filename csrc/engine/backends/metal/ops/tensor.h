#pragma once

#include "engine/tensor.h"

#include <span>

namespace tyloi::metal::ops {

Tensor empty(std::span<const std::int64_t> shape, const TensorOptions& options);
Tensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options);
void copy_into(Tensor destination, const Tensor& source);
void fill(Tensor& value, double number);

Tensor copy(const Tensor& value, Device target);
Tensor clone(const Tensor& value);
Tensor cast(const Tensor& value, ScalarType type);
Tensor reshape(const Tensor& value, std::span<const std::int64_t> shape);
Tensor transpose(const Tensor& value, std::int64_t first, std::int64_t second);
Tensor slice(const Tensor& value, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step = 1);
Tensor update_slice(const Tensor& destination, const Tensor& source, std::int64_t dim, std::int64_t start);
Tensor contiguous(const Tensor& value);
Tensor cat(std::span<const Tensor> values, std::int64_t dim);
Tensor index_select(const Tensor& value, std::int64_t dim, const Tensor& indices);
void index_copy(Tensor& value, std::int64_t dim, const Tensor& indices, const Tensor& source);

Tensor add(const Tensor& left, const Tensor& right);
Tensor multiply(const Tensor& left, const Tensor& right);
Tensor matmul(const Tensor& left, const Tensor& right);
Tensor add(const Tensor& value, std::int64_t offset);
Tensor divide(const Tensor& value, double number);
Tensor scale(const Tensor& value, double number);
Tensor softcap(const Tensor& value, double number);
Tensor gelu(const Tensor& value, bool approximate);
Tensor silu(const Tensor& value);
Tensor relu_squared(const Tensor& value);
Tensor abs(const Tensor& value);
Tensor square(const Tensor& value);
Tensor exp(const Tensor& value);
Tensor sqrt(const Tensor& value);
Tensor logical_not(const Tensor& value);
Tensor sum(const Tensor& value, std::int64_t dim, bool keepdim);
Tensor mean(const Tensor& value, std::int64_t dim, bool keepdim);
Tensor amax(const Tensor& value, std::int64_t dim, bool keepdim);
Tensor argmax(const Tensor& value, std::int64_t dim, bool keepdim);
Tensor all(const Tensor& value, std::int64_t dim, bool keepdim);
Tensor any(const Tensor& value, std::int64_t dim, bool keepdim);
Tensor clamp(const Tensor& value, double minimum, double maximum);
Tensor expand(const Tensor& value, std::span<const std::int64_t> shape);
Tensor unsqueeze(const Tensor& value, std::int64_t dim);
Tensor select(const Tensor& value, std::int64_t dim, std::int64_t index);

}  // namespace tyloi::metal::ops

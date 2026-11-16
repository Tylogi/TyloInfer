#pragma once

#include "engine/tensor/tensor.h"

namespace tyloi::metal::ops {

Tensor dense_project(const Tensor& input, const Tensor& weight);
Tensor dense_embedding(const Tensor& ids, const Tensor& weight);

} // namespace tyloi::metal::ops

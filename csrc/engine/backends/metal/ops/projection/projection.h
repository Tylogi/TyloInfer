#pragma once

#include "engine/tensor.h"

#include <span>
#include <vector>

namespace tyloi::metal::ops {

Tensor project(const Tensor& input, const Tensor& weight);
Tensor embedding(const Tensor& ids, const Tensor& weight);
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias);
Tensor affine(const Tensor& input, const Tensor& weight, const Tensor& bias);
Tensor project(const Tensor& input, const Tensor& weight, const Tensor& ids);
Tensor head_project(const Tensor& input, const Tensor& weight);
bool grouped(std::span<const Tensor> weights);
std::vector<Tensor> project_group(const Tensor& input, std::span<const Tensor> weights);
std::vector<Tensor> project_group(const Tensor& input, std::span<const Tensor> weights, std::size_t prefix);

}  // namespace tyloi::metal::ops

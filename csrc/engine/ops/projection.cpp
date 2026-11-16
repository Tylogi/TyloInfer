#include "engine/ops/ops.h"

#if TYLOINFER_HAS_CUDA
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#endif
#if TYLOINFER_HAS_METAL
#include "engine/backends/metal/ops/projection/projection.h"
#endif
#include <algorithm>
#include <limits>

namespace tyloi::ops {
#if TYLOINFER_HAS_CUDA
namespace {
const DenseTensor& dense(const Tensor& value) {
    const auto* result = value.representation<DenseTensor>();
    if (!result || result->is_encoded()) throw std::invalid_argument("operation requires a dense tensor");
    return *result;
}
Tensor owner(DenseTensor value) {
    return std::make_shared<DenseTensor>(std::move(value));
}
std::vector<Tensor> owners(std::vector<DenseTensor> values) {
    std::vector<Tensor> result;
    result.reserve(values.size());
    for (auto& value : values) result.push_back(owner(std::move(value)));
    return result;
}
void device(Device value) {
    if (!value.is_cpu() && !value.is_cuda())
        throw std::invalid_argument("unsupported tensor device");
    if (value.is_cuda()) (void)cuda::Executor::current();
}
void input(const Tensor& value_tensor) {
    const auto& value = dense(value_tensor);
    if (!value.defined()) throw std::invalid_argument("undefined tensor");
    value.require_dense();
    device(value.device());
}
void compatible(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    input(left);
    input(right);
    if (left.device() != right.device())
        throw std::invalid_argument("tensor devices do not match");
}
void require_cuda(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    if (!value.device().is_cuda()) throw std::invalid_argument("operation requires a CUDA tensor");
}
} // namespace
#endif

Tensor embedding(const Tensor& ids_tensor, const ::tyloi::Tensor& weight) {
#if TYLOINFER_HAS_METAL
    if ((ids_tensor.defined() && ids_tensor.device().is_metal()) ||
        (weight.defined() && weight.device().is_metal()))
        return metal::ops::embedding(ids_tensor, weight);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& ids = ::tyloi::ops::dense(ids_tensor);
    input(ids);
    return owner(cuda::ops::embedding(ids, weight));
#else
    throw std::invalid_argument("embedding is not supported on this device");
#endif
}

Tensor project(const Tensor& value_tensor, const ::tyloi::Tensor& weight) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()) ||
        (weight.defined() && weight.device().is_metal()))
        return metal::ops::project(value_tensor, weight);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::project(value, weight));
#else
    throw std::invalid_argument("project is not supported on this device");
#endif
}

Tensor head_project(const Tensor& value, const Tensor& weight) {
#if TYLOINFER_HAS_METAL
    if ((value.defined() && value.device().is_metal()) ||
        (weight.defined() && weight.device().is_metal()))
        return metal::ops::head_project(value, weight);
#endif
#if TYLOINFER_HAS_CUDA
    input(value);
    require_cuda(value);
    if (!weight || !weight.defined() || value.dim() != 4 || weight.dim() != 3 ||
        value.numel() <= 0 || value.size(2) <= 0 || value.size(2) > std::numeric_limits<int>::max() ||
        value.size(3) <= 0 || value.size(2) != weight.size(0) || value.size(3) != weight.size(2))
        throw std::invalid_argument("head-wise projection geometry is incompatible");
    return owner(cuda::ops::head_project(dense(value), weight));
#else
    throw std::invalid_argument("head_project is not supported on this device");
#endif
}

Tensor project(const Tensor& value, const Tensor& weight, const Tensor& ids) {
#if TYLOINFER_HAS_METAL
    if ((value.defined() && value.device().is_metal()) ||
        (weight.defined() && weight.device().is_metal()) ||
        (ids.defined() && ids.device().is_metal()))
        return metal::ops::project(value, weight, ids);
#endif
#if TYLOINFER_HAS_CUDA
    compatible(value, ids);
    return owner(cuda::ops::project(dense(value), weight, dense(ids)));
#else
    throw std::invalid_argument("project is not supported on this device");
#endif
}

Tensor linear(const Tensor& value_tensor, const ::tyloi::Tensor& weight, const Tensor& bias) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()) ||
        (weight.defined() && weight.device().is_metal()) ||
        (bias.defined() && bias.device().is_metal()))
        return metal::ops::linear(value_tensor, weight, bias);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    const auto* bias_dense = bias.defined() ? &::tyloi::ops::dense(bias) : nullptr;
    if (bias_dense) input(*bias_dense);
    return owner(cuda::ops::linear(value, weight, bias_dense));
#else
    throw std::invalid_argument("linear is not supported on this device");
#endif
}

Tensor affine(const Tensor& value_tensor, const ::tyloi::Tensor& weight, const Tensor& bias) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()) ||
        (weight.defined() && weight.device().is_metal()) ||
        (bias.defined() && bias.device().is_metal()))
        return metal::ops::affine(value_tensor, weight, bias);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    const auto* bias_dense = bias.defined() ? &::tyloi::ops::dense(bias) : nullptr;
    if (bias_dense) input(*bias_dense);
    return owner(cuda::ops::affine(value, weight, bias_dense));
#else
    throw std::invalid_argument("affine is not supported on this device");
#endif
}

std::vector<Tensor> project_group(const Tensor& value_tensor, std::span<const Tensor> group) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()) ||
        std::any_of(group.begin(), group.end(), [](const Tensor& weight) {
            return weight.defined() && weight.device().is_metal();
        }))
        return metal::ops::project_group(value_tensor, group);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owners(cuda::ops::project_group(value, group));
#else
    throw std::invalid_argument("project_group is not supported on this device");
#endif
}

bool grouped(std::span<const Tensor> weights) {
#if TYLOINFER_HAS_METAL
    if (std::any_of(weights.begin(), weights.end(), [](const Tensor& weight) {
            return weight.defined() && weight.device().is_metal();
        }))
        return metal::ops::grouped(weights);
#endif
#if TYLOINFER_HAS_CUDA
    return cuda::ops::grouped(weights);
#else
    return false;
#endif
}

std::vector<Tensor> project_group(const Tensor& value_tensor, std::span<const Tensor> weights,
                                  std::size_t prefix) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()) ||
        std::any_of(weights.begin(), weights.end(), [](const Tensor& weight) {
            return weight.defined() && weight.device().is_metal();
        }))
        return metal::ops::project_group(value_tensor, weights, prefix);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owners(cuda::ops::project_group(value, weights, prefix));
#else
    throw std::invalid_argument("project_group is not supported on this device");
#endif
}

} // namespace tyloi::ops

#include "engine/ops.h"

#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/tensor.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/norm/layer.h"
#endif
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
#include "engine/backends/metal/ops/tensor.h"
#include "engine/backends/metal/ops/norm/rms.h"
#include "engine/backends/metal/ops/norm/layer.h"
#endif

namespace tyloi::ops {
#ifdef TYLOINFER_NATIVE_CUDA
namespace {
const DenseTensor& dense(const Tensor& value) {
    const auto* result = value.representation<DenseTensor>();
    if (!result || result->is_encoded()) throw std::invalid_argument("operation requires a dense tensor");
    return *result;
}
Tensor owner(DenseTensor value) {
    return std::make_shared<DenseTensor>(std::move(value));
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

Tensor add(const Tensor& left_tensor, const Tensor& right_tensor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((left_tensor.defined() && left_tensor.device().is_metal()) ||
        (right_tensor.defined() && right_tensor.device().is_metal()))
        return metal::ops::add(left_tensor, right_tensor);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::add(left, right));
#else
    throw std::invalid_argument("add is not supported on this device");
#endif
}

Tensor multiply(const Tensor& left_tensor, const Tensor& right_tensor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((left_tensor.defined() && left_tensor.device().is_metal()) ||
        (right_tensor.defined() && right_tensor.device().is_metal()))
        return metal::ops::multiply(left_tensor, right_tensor);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::multiply(left, right));
#else
    throw std::invalid_argument("multiply is not supported on this device");
#endif
}

Tensor matmul(const Tensor& left_tensor, const Tensor& right_tensor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((left_tensor.defined() && left_tensor.device().is_metal()) ||
        (right_tensor.defined() && right_tensor.device().is_metal()))
        return metal::ops::matmul(left_tensor, right_tensor);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::matmul(left, right));
#else
    throw std::invalid_argument("matmul is not supported on this device");
#endif
}

Tensor rms_norm(const Tensor& value_tensor, const Tensor& weight_tensor, double eps, double offset, ScalarType output_type) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()) ||
        (weight_tensor.defined() && weight_tensor.device().is_metal()))
        return metal::ops::rms_norm(value_tensor, weight_tensor, eps, offset, output_type);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& weight = ::tyloi::ops::dense(weight_tensor);
    input(value);
    input(weight);
    if (value.device() != weight.device() || value.dim() == 0 || weight.dim() == 0 ||
        value.size(-1) <= 0 || weight.numel() != value.size(-1) ||
        weight.size(-1) != value.size(-1))
        throw std::invalid_argument("rms_norm input and weight are incompatible");
    if (output_type != ScalarType::float32 && output_type != ScalarType::float16 &&
        output_type != ScalarType::bfloat16)
        throw std::invalid_argument("unsupported rms_norm output type");
    if (!value.is_cuda()) {
        auto normalized = qwen_rms_norm(value.reshape({-1, value.size(-1)}).to(kFloat32),
                                        weight_tensor, eps, offset);
        return owner(normalized.to(output_type).contiguous().reshape(value.sizes()));
    }
    return owner(cuda::ops::rms_norm(value, weight, eps, offset, output_type));
#else
    throw std::invalid_argument("rms_norm is not supported on this device");
#endif
}

Tensor layer_norm(const Tensor& value_tensor, const Tensor& weight_tensor, const Tensor& bias_tensor, double epsilon) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()) ||
        (weight_tensor.defined() && weight_tensor.device().is_metal()) ||
        (bias_tensor.defined() && bias_tensor.device().is_metal()))
        return metal::ops::layer_norm(value_tensor, weight_tensor, bias_tensor, epsilon);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& weight = ::tyloi::ops::dense(weight_tensor);
    const auto& bias = ::tyloi::ops::dense(bias_tensor);
    compatible(value, weight);
    compatible(value, bias);
    if (value.dim() == 0 || weight.dim() != 1 || bias.dim() != 1 ||
        weight.size(0) != value.size(-1) || bias.size(0) != value.size(-1))
        throw std::invalid_argument("layer_norm input and weights are incompatible");
    require_cuda(value);
    return owner(cuda::ops::layer_norm(value, weight, bias, epsilon));
#else
    throw std::invalid_argument("layer_norm is not supported on this device");
#endif
}

Tensor gelu(const Tensor& value_tensor, bool approximate) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::gelu(value_tensor, approximate);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::gelu(value, approximate));
#else
    throw std::invalid_argument("gelu is not supported on this device");
#endif
}

Tensor silu(const Tensor& value_tensor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::silu(value_tensor);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::silu(value));
#else
    throw std::invalid_argument("silu is not supported on this device");
#endif
}

Tensor relu_squared(const Tensor& value_tensor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::relu_squared(value_tensor);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::relu_squared(value));
#else
    throw std::invalid_argument("relu_squared is not supported on this device");
#endif
}

Tensor softcap(const Tensor& value_tensor, double cap) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::softcap(value_tensor, cap);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    if (cap > 0.0) require_cuda(value);
    return owner(cuda::ops::softcap(value, cap));
#else
    throw std::invalid_argument("softcap is not supported on this device");
#endif
}

Tensor scale(const Tensor& value_tensor, double factor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::scale(value_tensor, factor);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::scale(value, factor));
#else
    throw std::invalid_argument("scale is not supported on this device");
#endif
}

Tensor divide(const Tensor& value_tensor, double divisor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::divide(value_tensor, divisor);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::divide(value, divisor));
#else
    throw std::invalid_argument("divide is not supported on this device");
#endif
}

Tensor sum(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::sum(value_tensor, dim, keepdim);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::sum(value, dim, keepdim));
#else
    throw std::invalid_argument("sum is not supported on this device");
#endif
}

Tensor add(const Tensor& value_tensor, std::int64_t offset) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::add(value_tensor, offset);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::add(value, offset));
#else
    throw std::invalid_argument("add is not supported on this device");
#endif
}

Tensor mean(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::mean(value_tensor, dim, keepdim);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::mean(value, dim, keepdim));
#else
    throw std::invalid_argument("mean is not supported on this device");
#endif
}

Tensor amax(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::amax(value_tensor, dim, keepdim);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::amax(value, dim, keepdim));
#else
    throw std::invalid_argument("amax is not supported on this device");
#endif
}

Tensor argmax(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::argmax(value_tensor, dim, keepdim);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::argmax(value, dim, keepdim));
#else
    throw std::invalid_argument("argmax is not supported on this device");
#endif
}

Tensor all(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::all(value_tensor, dim, keepdim);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::all(value, dim, keepdim));
#else
    throw std::invalid_argument("all is not supported on this device");
#endif
}

Tensor any(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::any(value_tensor, dim, keepdim);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::any(value, dim, keepdim));
#else
    throw std::invalid_argument("any is not supported on this device");
#endif
}

Tensor abs(const Tensor& value_tensor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::abs(value_tensor);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::abs(value));
#else
    throw std::invalid_argument("abs is not supported on this device");
#endif
}

Tensor square(const Tensor& value_tensor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::square(value_tensor);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::square(value));
#else
    throw std::invalid_argument("square is not supported on this device");
#endif
}

Tensor exp(const Tensor& value_tensor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::exp(value_tensor);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::exp(value));
#else
    throw std::invalid_argument("exp is not supported on this device");
#endif
}

Tensor sqrt(const Tensor& value_tensor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::sqrt(value_tensor);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::sqrt(value));
#else
    throw std::invalid_argument("sqrt is not supported on this device");
#endif
}

Tensor logical_not(const Tensor& value_tensor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::logical_not(value_tensor);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::logical_not(value));
#else
    throw std::invalid_argument("logical_not is not supported on this device");
#endif
}

Tensor clamp(const Tensor& value_tensor, double minimum, double maximum) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::clamp(value_tensor, minimum, maximum);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::clamp(value, minimum, maximum));
#else
    throw std::invalid_argument("clamp is not supported on this device");
#endif
}

} // namespace tyloi::ops

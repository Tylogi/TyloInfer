#include "engine/ops.h"

#if TYLOINFER_HAS_CUDA
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/tensor.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/norm/layer.h"
#endif
#if TYLOINFER_HAS_METAL
#include "engine/backends/metal/ops/tensor.h"
#include "engine/backends/metal/ops/norm/rms.h"
#include "engine/backends/metal/ops/norm/layer.h"
#endif

namespace tyloi::ops {
#if TYLOINFER_HAS_CUDA
namespace {
const DenseTensor& dense(const Tensor& value) {
    const auto* result = value.representation<DenseTensor>();
    if (!result || result->is_encoded()) throw std::invalid_argument("operation requires a dense tensor");
    return *result;
}
DenseTensor& dense(Tensor& value) {
    auto* result = value.representation<DenseTensor>();
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
#if TYLOINFER_HAS_METAL
    if ((left_tensor.defined() && left_tensor.device().is_metal()) ||
        (right_tensor.defined() && right_tensor.device().is_metal()))
        return metal::ops::add(left_tensor, right_tensor);
#endif
#if TYLOINFER_HAS_CUDA
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
#if TYLOINFER_HAS_METAL
    if ((left_tensor.defined() && left_tensor.device().is_metal()) ||
        (right_tensor.defined() && right_tensor.device().is_metal()))
        return metal::ops::multiply(left_tensor, right_tensor);
#endif
#if TYLOINFER_HAS_CUDA
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
#if TYLOINFER_HAS_METAL
    if ((left_tensor.defined() && left_tensor.device().is_metal()) ||
        (right_tensor.defined() && right_tensor.device().is_metal()))
        return metal::ops::matmul(left_tensor, right_tensor);
#endif
#if TYLOINFER_HAS_CUDA
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
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()) ||
        (weight_tensor.defined() && weight_tensor.device().is_metal()))
        return metal::ops::rms_norm(value_tensor, weight_tensor, eps, offset, output_type);
#endif
#if TYLOINFER_HAS_CUDA
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
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()) ||
        (weight_tensor.defined() && weight_tensor.device().is_metal()) ||
        (bias_tensor.defined() && bias_tensor.device().is_metal()))
        return metal::ops::layer_norm(value_tensor, weight_tensor, bias_tensor, epsilon);
#endif
#if TYLOINFER_HAS_CUDA
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
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::gelu(value_tensor, approximate);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::gelu(value, approximate));
#else
    throw std::invalid_argument("gelu is not supported on this device");
#endif
}

Tensor silu(const Tensor& value_tensor) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::silu(value_tensor);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::silu(value));
#else
    throw std::invalid_argument("silu is not supported on this device");
#endif
}

Tensor relu_squared(const Tensor& value_tensor) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::relu_squared(value_tensor);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::relu_squared(value));
#else
    throw std::invalid_argument("relu_squared is not supported on this device");
#endif
}

Tensor softcap(const Tensor& value_tensor, double cap) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::softcap(value_tensor, cap);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    if (cap > 0.0) require_cuda(value);
    return owner(cuda::ops::softcap(value, cap));
#else
    throw std::invalid_argument("softcap is not supported on this device");
#endif
}

Tensor scale(const Tensor& value_tensor, double factor) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::scale(value_tensor, factor);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::scale(value, factor));
#else
    throw std::invalid_argument("scale is not supported on this device");
#endif
}

Tensor divide(const Tensor& value_tensor, double divisor) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::divide(value_tensor, divisor);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::divide(value, divisor));
#else
    throw std::invalid_argument("divide is not supported on this device");
#endif
}

Tensor sum(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::sum(value_tensor, dim, keepdim);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::sum(value, dim, keepdim));
#else
    throw std::invalid_argument("sum is not supported on this device");
#endif
}

Tensor add(const Tensor& value_tensor, std::int64_t offset) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::add(value_tensor, offset);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::add(value, offset));
#else
    throw std::invalid_argument("add is not supported on this device");
#endif
}

Tensor mean(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::mean(value_tensor, dim, keepdim);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::mean(value, dim, keepdim));
#else
    throw std::invalid_argument("mean is not supported on this device");
#endif
}

Tensor amax(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::amax(value_tensor, dim, keepdim);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::amax(value, dim, keepdim));
#else
    throw std::invalid_argument("amax is not supported on this device");
#endif
}

Tensor argmax(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::argmax(value_tensor, dim, keepdim);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::argmax(value, dim, keepdim));
#else
    throw std::invalid_argument("argmax is not supported on this device");
#endif
}

Tensor all(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::all(value_tensor, dim, keepdim);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::all(value, dim, keepdim));
#else
    throw std::invalid_argument("all is not supported on this device");
#endif
}

Tensor any(const Tensor& value_tensor, std::int64_t dim, bool keepdim) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::any(value_tensor, dim, keepdim);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::any(value, dim, keepdim));
#else
    throw std::invalid_argument("any is not supported on this device");
#endif
}

Tensor abs(const Tensor& value_tensor) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::abs(value_tensor);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::abs(value));
#else
    throw std::invalid_argument("abs is not supported on this device");
#endif
}

Tensor square(const Tensor& value_tensor) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::square(value_tensor);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::square(value));
#else
    throw std::invalid_argument("square is not supported on this device");
#endif
}

Tensor exp(const Tensor& value_tensor) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::exp(value_tensor);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::exp(value));
#else
    throw std::invalid_argument("exp is not supported on this device");
#endif
}

Tensor sqrt(const Tensor& value_tensor) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::sqrt(value_tensor);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::sqrt(value));
#else
    throw std::invalid_argument("sqrt is not supported on this device");
#endif
}

Tensor logical_not(const Tensor& value_tensor) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::logical_not(value_tensor);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::logical_not(value));
#else
    throw std::invalid_argument("logical_not is not supported on this device");
#endif
}

Tensor clamp(const Tensor& value_tensor, double minimum, double maximum) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::clamp(value_tensor, minimum, maximum);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::clamp(value, minimum, maximum));
#else
    throw std::invalid_argument("clamp is not supported on this device");
#endif
}

Tensor clamp_min(const Tensor& value_tensor, double minimum) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::clamp_min(value_tensor, minimum);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::clamp_min(value, minimum));
#else
    throw std::invalid_argument("clamp_min is not supported on this device");
#endif
}

Tensor clamp_max(const Tensor& value_tensor, double maximum) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::clamp_max(value_tensor, maximum);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::clamp_max(value, maximum));
#else
    throw std::invalid_argument("clamp_max is not supported on this device");
#endif
}

Tensor remainder(const Tensor& value_tensor, double divisor) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::remainder(value_tensor, divisor);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::remainder(value, divisor));
#else
    throw std::invalid_argument("remainder is not supported on this device");
#endif
}

Tensor eq(const Tensor& left_tensor, const Tensor& right_tensor) {
#if TYLOINFER_HAS_METAL
    if ((left_tensor.defined() && left_tensor.device().is_metal()) ||
        (right_tensor.defined() && right_tensor.device().is_metal()))
        return metal::ops::eq(left_tensor, right_tensor);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::eq(left, right));
#else
    throw std::invalid_argument("eq is not supported on this device");
#endif
}

Tensor eq(const Tensor& value_tensor, double other) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::eq(value_tensor, other);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::eq(value, other));
#else
    throw std::invalid_argument("eq is not supported on this device");
#endif
}

Tensor ne(const Tensor& left_tensor, const Tensor& right_tensor) {
#if TYLOINFER_HAS_METAL
    if ((left_tensor.defined() && left_tensor.device().is_metal()) ||
        (right_tensor.defined() && right_tensor.device().is_metal()))
        return metal::ops::ne(left_tensor, right_tensor);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return owner(cuda::ops::ne(left, right));
#else
    throw std::invalid_argument("ne is not supported on this device");
#endif
}

Tensor ne(const Tensor& value_tensor, double other) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::ne(value_tensor, other);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::ne(value, other));
#else
    throw std::invalid_argument("ne is not supported on this device");
#endif
}

bool equal(const Tensor& left_tensor, const Tensor& right_tensor) {
#if TYLOINFER_HAS_METAL
    if ((left_tensor.defined() && left_tensor.device().is_metal()) ||
        (right_tensor.defined() && right_tensor.device().is_metal()))
        return metal::ops::equal(left_tensor, right_tensor);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    compatible(left, right);
    require_cuda(left);
    return cuda::ops::equal(left, right);
#else
    throw std::invalid_argument("equal is not supported on this device");
#endif
}

Tensor masked_select(const Tensor& value_tensor, const Tensor& mask_tensor) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()) ||
        (mask_tensor.defined() && mask_tensor.device().is_metal()))
        return metal::ops::masked_select(value_tensor, mask_tensor);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& mask = ::tyloi::ops::dense(mask_tensor);
    compatible(value, mask);
    require_cuda(value);
    return owner(cuda::ops::masked_select(value, mask));
#else
    throw std::invalid_argument("masked_select is not supported on this device");
#endif
}

Tensor gather(const Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()) ||
        (indices_tensor.defined() && indices_tensor.device().is_metal()))
        return metal::ops::gather(value_tensor, dim, indices_tensor);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    compatible(value, indices);
    require_cuda(value);
    return owner(cuda::ops::gather(value, dim, indices));
#else
    throw std::invalid_argument("gather is not supported on this device");
#endif
}

Tensor repeat(const Tensor& value_tensor, std::span<const std::int64_t> repeats) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::repeat(value_tensor, repeats);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::repeat(value, repeats));
#else
    throw std::invalid_argument("repeat is not supported on this device");
#endif
}

Tensor repeat_interleave(const Tensor& value_tensor, std::int64_t repeats, std::int64_t dim) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::repeat_interleave(value_tensor, repeats, dim);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::repeat_interleave(value, repeats, dim));
#else
    throw std::invalid_argument("repeat_interleave is not supported on this device");
#endif
}

Tensor tril(const Tensor& value_tensor, std::int64_t diagonal) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::tril(value_tensor, diagonal);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    require_cuda(value);
    return owner(cuda::ops::tril(value, diagonal));
#else
    throw std::invalid_argument("tril is not supported on this device");
#endif
}

Tensor masked_fill(const Tensor& value_tensor, const Tensor& mask_tensor, double fill_value) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()) ||
        (mask_tensor.defined() && mask_tensor.device().is_metal()))
        return metal::ops::masked_fill(value_tensor, mask_tensor, fill_value);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& mask = ::tyloi::ops::dense(mask_tensor);
    compatible(value, mask);
    require_cuda(value);
    return owner(cuda::ops::masked_fill(value, mask, fill_value));
#else
    throw std::invalid_argument("masked_fill is not supported on this device");
#endif
}

void masked_fill_inplace(Tensor& value_tensor, const Tensor& mask_tensor, double fill_value) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()) ||
        (mask_tensor.defined() && mask_tensor.device().is_metal())) {
        metal::ops::masked_fill_inplace(value_tensor, mask_tensor, fill_value);
        return;
    }
#endif
#if TYLOINFER_HAS_CUDA
    auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& mask = ::tyloi::ops::dense(mask_tensor);
    compatible(value, mask);
    require_cuda(value);
    cuda::ops::masked_fill_inplace(value, mask, fill_value);
#else
    throw std::invalid_argument("masked_fill_inplace is not supported on this device");
#endif
}

void scatter_add(Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor, const Tensor& source_tensor) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()) ||
        (indices_tensor.defined() && indices_tensor.device().is_metal()) ||
        (source_tensor.defined() && source_tensor.device().is_metal())) {
        metal::ops::scatter_add(value_tensor, dim, indices_tensor, source_tensor);
        return;
    }
#endif
#if TYLOINFER_HAS_CUDA
    auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    const auto& source = ::tyloi::ops::dense(source_tensor);
    compatible(value, indices);
    compatible(value, source);
    require_cuda(value);
    cuda::ops::scatter_add(value, dim, indices, source);
#else
    throw std::invalid_argument("scatter_add is not supported on this device");
#endif
}

void scatter(Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor, const Tensor& source_tensor) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()) ||
        (indices_tensor.defined() && indices_tensor.device().is_metal()) ||
        (source_tensor.defined() && source_tensor.device().is_metal())) {
        metal::ops::scatter(value_tensor, dim, indices_tensor, source_tensor);
        return;
    }
#endif
#if TYLOINFER_HAS_CUDA
    auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    const auto& source = ::tyloi::ops::dense(source_tensor);
    compatible(value, indices);
    compatible(value, source);
    require_cuda(value);
    cuda::ops::scatter(value, dim, indices, source);
#else
    throw std::invalid_argument("scatter is not supported on this device");
#endif
}

void index_fill(Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor, double fill_value) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()) ||
        (indices_tensor.defined() && indices_tensor.device().is_metal())) {
        metal::ops::index_fill(value_tensor, dim, indices_tensor, fill_value);
        return;
    }
#endif
#if TYLOINFER_HAS_CUDA
    auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    compatible(value, indices);
    require_cuda(value);
    cuda::ops::index_fill(value, dim, indices, fill_value);
#else
    throw std::invalid_argument("index_fill is not supported on this device");
#endif
}

std::array<Tensor, 3> qkv_norm(std::array<Tensor, 3> values, std::array<Tensor, 3> weights,
                               double eps, double offset, ScalarType output_type) {
#if TYLOINFER_HAS_METAL
    bool on_metal = false;
    for (std::size_t i = 0; i < values.size(); ++i)
        on_metal |= (values[i].defined() && values[i].device().is_metal()) ||
                    (weights[i].defined() && weights[i].device().is_metal());
    if (on_metal) {
        if (output_type != ScalarType::float32 && output_type != ScalarType::bfloat16)
            throw std::invalid_argument("unsupported qkv_norm output type");
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (!values[i].defined() || values[i].is_encoded())
                throw std::invalid_argument("qkv_norm requires all values to be unencoded");
            if (!weights[i].defined()) continue;
            if (values[i].dim() == 0 || weights[i].dim() == 0 || weights[i].is_encoded() ||
                values[i].size(-1) <= 0 || weights[i].numel() != values[i].size(-1))
                throw std::invalid_argument("qkv_norm input and weight are incompatible");
            auto weight = metal::ops::reshape(weights[i], std::array{values[i].size(-1)});
            values[i] = metal::ops::rms_norm(values[i], weight, eps, offset, output_type);
        }
        return values;
    }
#endif
#if TYLOINFER_HAS_CUDA
    if (output_type != ScalarType::float32 && output_type != ScalarType::bfloat16)
        throw std::invalid_argument("unsupported qkv_norm output type");
    std::array<const DenseTensor*, 3> dense_weights{};
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!values[i].defined()) throw std::invalid_argument("qkv_norm requires all values");
        input(values[i]);
        if (weights[i].defined()) {
            dense_weights[i] = &dense(weights[i]);
            compatible(values[i], weights[i]);
            if (values[i].dim() == 0 || weights[i].dim() == 0 ||
                values[i].size(-1) <= 0 || weights[i].numel() != values[i].size(-1))
                throw std::invalid_argument("qkv_norm input and weight are incompatible");
        }
    }
    for (std::size_t i = 0; i < values.size(); ++i)
        if (weights[i].defined()) require_cuda(values[i]);
    (void)cuda::Executor::current();
    std::array<DenseTensor, 3> dense_values{dense(values[0]), dense(values[1]), dense(values[2])};
    auto result = cuda::ops::qkv_norm(std::move(dense_values), dense_weights, eps, offset, output_type);
    return {owner(std::move(result[0])), owner(std::move(result[1])), owner(std::move(result[2]))};
#else
    throw std::invalid_argument("qkv_norm is not supported on this device");
#endif
}

} // namespace tyloi::ops

namespace tyloi::ops {
Tensor log_negative(const Tensor& value) {
#if TYLOINFER_HAS_METAL
    if (value.defined() && value.is_metal()) return metal::ops::log_negative(value);
#endif
#if TYLOINFER_HAS_CUDA
    input(value);
    return owner(cuda::log(-dense(value)));
#else
    throw std::invalid_argument("log_negative requires a compiled device");
#endif
}
} // namespace tyloi::ops

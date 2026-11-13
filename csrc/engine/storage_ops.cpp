#include "engine/ops.h"

#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/tensor.h"
#include "engine/backends/cuda/native/tensor/tensor.h"
#include <algorithm>
#include <cstring>
#else
#include "engine/backends/cpu/ops/tensor.h"
#endif
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
#include "engine/backends/metal/ops/tensor.h"
#endif

namespace tyloi::ops {
#ifndef TYLOINFER_NATIVE_CUDA
struct DeviceScope::Impl {
    explicit Impl(Device target) {
        if (!(target.is_cpu() && target.index == 0)
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
            && !(target.is_metal() && target.index == 0)
#endif
        ) throw std::invalid_argument("unsupported execution device");
    }
};
DeviceScope::DeviceScope(Device target) : impl_(std::make_unique<Impl>(target)) {}
DeviceScope::~DeviceScope() = default;
#else
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
    const auto& value = ::tyloi::ops::dense(value_tensor);
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
}
#endif

Tensor copy(const Tensor& value_tensor, Device target) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if (target.is_metal() || (value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::copy(value_tensor, target);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    device(target);
    return owner(cuda::ops::copy(value, target));
#else
    if (!value_tensor.defined() || value_tensor.is_encoded() ||
        !value_tensor.representation<DenseTensor>() ||
        value_tensor.device() != Device{DeviceType::cpu, 0} || target != Device{DeviceType::cpu, 0})
        throw std::invalid_argument("copy requires a dense CPU:0 tensor and CPU:0 target");
    return value_tensor;
#endif
}

void copy_into(Tensor destination_tensor, const Tensor& source_tensor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((destination_tensor.defined() && destination_tensor.device().is_metal()) ||
        (source_tensor.defined() && source_tensor.device().is_metal())) {
        if (!destination_tensor.defined() || !source_tensor.defined() || destination_tensor.device() != source_tensor.device())
            throw std::invalid_argument("tensor devices do not match");
        metal::ops::copy_into(destination_tensor, source_tensor);
        return;
    }
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    auto& destination = ::tyloi::ops::dense(destination_tensor);
    const auto& source = ::tyloi::ops::dense(source_tensor);
    input(destination);
    input(source);
    cuda::ops::copy_into(destination, source);
#else
    cpu::ops::copy_into(destination_tensor, source_tensor);
#endif
}

Tensor clone(const Tensor& value_tensor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if (value_tensor.defined() && value_tensor.device().is_metal()) return metal::ops::clone(value_tensor);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::clone(value));
#else
    return cpu::ops::clone(value_tensor);
#endif
}

Tensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if (options.target_device().is_metal()) return metal::ops::zeros(shape, options);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    DeviceScope scope(options.target_device());
    return owner(cuda::ops::zeros(shape, options));
#else
    return cpu::ops::zeros(shape, options);
#endif
}

Tensor empty(std::span<const std::int64_t> shape, const TensorOptions& options) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if (options.target_device().is_metal()) return metal::ops::empty(shape, options);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    DeviceScope scope(options.target_device());
    return owner(cuda::empty(shape, options));
#else
    return cpu::ops::empty(shape, options);
#endif
}

void fill(Tensor& value_tensor, double fill_value) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if (value_tensor.defined() && value_tensor.device().is_metal()) { metal::ops::fill(value_tensor, fill_value); return; }
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    auto& value = ::tyloi::ops::dense(value_tensor);
    if (!value.defined()) throw std::invalid_argument("undefined tensor");
    DeviceScope scope(value.device());
    if (value.is_cuda()) {
        cuda::ops::fill(value, fill_value);
        return;
    }
    if (!value.is_contiguous())
        throw std::invalid_argument("CPU fill requires a contiguous tensor");
    if (fill_value == 0.0) {
        std::memset(value.data_ptr(), 0, value.nbytes());
        return;
    }
    switch (value.scalar_type()) {
        case ScalarType::boolean:
            std::fill_n(value.data_ptr<bool>(), value.numel(), fill_value != 0.0);
            break;
        case ScalarType::uint8:
            std::fill_n(value.data_ptr<std::uint8_t>(), value.numel(), static_cast<std::uint8_t>(fill_value));
            break;
        case ScalarType::int8:
            std::fill_n(value.data_ptr<std::int8_t>(), value.numel(), static_cast<std::int8_t>(fill_value));
            break;
        case ScalarType::int16:
            std::fill_n(value.data_ptr<std::int16_t>(), value.numel(), static_cast<std::int16_t>(fill_value));
            break;
        case ScalarType::int32:
            std::fill_n(value.data_ptr<std::int32_t>(), value.numel(), static_cast<std::int32_t>(fill_value));
            break;
        case ScalarType::int64:
            std::fill_n(value.data_ptr<std::int64_t>(), value.numel(), static_cast<std::int64_t>(fill_value));
            break;
        case ScalarType::float16:
            std::fill_n(value.data_ptr<std::uint16_t>(), value.numel(),
                cuda::encode_float16(static_cast<float>(fill_value)));
            break;
        case ScalarType::bfloat16:
            std::fill_n(value.data_ptr<std::uint16_t>(), value.numel(),
                cuda::encode_bfloat16(static_cast<float>(fill_value)));
            break;
        case ScalarType::float32:
            std::fill_n(value.data_ptr<float>(), value.numel(), static_cast<float>(fill_value));
            break;
        case ScalarType::float64:
            std::fill_n(value.data_ptr<double>(), value.numel(), fill_value);
            break;
        default:
            throw std::invalid_argument("CPU fill for this scalar type is not implemented yet");
    }
#else
    cpu::ops::fill(value_tensor, fill_value);
#endif
}


Tensor cast(const Tensor& value_tensor, ScalarType type) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if (value_tensor.defined() && value_tensor.device().is_metal()) return metal::ops::cast(value_tensor, type);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::cast(value, type));
#else
    if (!value_tensor.defined() || value_tensor.is_encoded() ||
        !value_tensor.device().is_cpu() || value_tensor.device().index != 0 ||
        !value_tensor.representation<DenseTensor>())
        throw std::invalid_argument("operation requires a dense CPU tensor");
    if (value_tensor.scalar_type() != type)
        throw std::invalid_argument("CPU cast conversion is not supported");
    return value_tensor;
#endif
}

Tensor slice(const Tensor& value_tensor, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if (value_tensor.defined() && value_tensor.device().is_metal()) return metal::ops::slice(value_tensor, dim, start, end, step);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::slice(value, dim, start, end, step));
#else
    const auto* value = value_tensor.representation<DenseTensor>();
    if (!value_tensor.defined() || value_tensor.is_encoded() || !value ||
        !value_tensor.device().is_cpu() || value_tensor.device().index != 0)
        throw std::invalid_argument("operation requires a dense CPU tensor");
    return std::make_shared<DenseTensor>(tyloi::slice(*value, dim, start, end, step));
#endif
}

Tensor update_slice(const Tensor& destination, const Tensor& source, std::int64_t dim, std::int64_t start) {
    if (!destination.defined() || !source.defined() || destination.is_encoded() || source.is_encoded())
        throw std::invalid_argument("slice update requires defined dense tensors");
    if (destination.device() != source.device() || destination.scalar_type() != source.scalar_type() ||
        destination.dim() != source.dim())
        throw std::invalid_argument("slice update tensors are incompatible");
    const auto rank = destination.dim();
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank)
        throw std::invalid_argument("slice update dimension is invalid");
    for (std::int64_t axis = 0; axis < rank; ++axis)
        if (axis != dim && destination.size(axis) != source.size(axis))
            throw std::invalid_argument("slice update shape is incompatible");
    const auto extent = source.size(dim), size = destination.size(dim);
    if (start < 0 || start > size || extent < 0 || extent > size - start)
        throw std::invalid_argument("slice update exceeds destination");
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if (destination.device().is_metal()) return metal::ops::update_slice(destination, source, dim, start);
#endif
    auto target = slice(destination, dim, start, start + extent, 1);
    copy_into(target, source);
    return destination;
}

Tensor cat(std::span<const Tensor> values, std::int64_t dim) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    for (const auto& value : values)
        if (value.defined() && value.device().is_metal()) return metal::ops::cat(values, dim);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    if (values.empty()) throw std::invalid_argument("cat requires at least one tensor");
    std::vector<DenseTensor> dense_values;
    dense_values.reserve(values.size());
    for (const auto& value : values) {
        if (!value.defined()) throw std::invalid_argument("cat requires defined tensors");
        if (dense_values.empty()) input(value);
        else compatible(dense_values.front(), value);
        dense_values.push_back(dense(value));
    }
    return owner(cuda::ops::cat(dense_values, dim));
#else
    throw std::invalid_argument("cat is not supported");
#endif
}

Tensor index_select(const Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()) ||
        (indices_tensor.defined() && indices_tensor.device().is_metal()))
        return metal::ops::index_select(value_tensor, dim, indices_tensor);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    compatible(value, indices);
    require_cuda(value);
    return owner(cuda::ops::index_select(value, dim, indices));
#else
    throw std::invalid_argument("index select is not supported");
#endif
}

void index_copy(Tensor& value_tensor, std::int64_t dim, const Tensor& indices_tensor, const Tensor& source_tensor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if (value_tensor.defined() && value_tensor.device().is_metal()) {
        metal::ops::index_copy(value_tensor, dim, indices_tensor, source_tensor);
        return;
    }
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    auto& value = ::tyloi::ops::dense(value_tensor);
    const auto& indices = ::tyloi::ops::dense(indices_tensor);
    const auto& source = ::tyloi::ops::dense(source_tensor);
    compatible(value, indices);
    compatible(value, source);
    require_cuda(value);
    cuda::ops::index_copy(value, dim, indices, source);
#else
    throw std::invalid_argument("indexed update is not supported");
#endif
}

} // namespace tyloi::ops

namespace tyloi {
Tensor Tensor::relocated(Device target) const {
    if (!impl_) return {};
    if (device().is_metal() || target.is_metal()) return ops::copy(*this, target);
    if (const auto* value = representation<DenseTensor>()) {
        return value->defined() && value->device() != target
            ? ops::copy(*this, target)
            : Tensor{*value};
    }
    return Tensor{impl_->relocated(target)};
}
} // namespace tyloi

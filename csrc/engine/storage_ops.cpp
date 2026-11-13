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
}
#endif

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

} // namespace tyloi::ops

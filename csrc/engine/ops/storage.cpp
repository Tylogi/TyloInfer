#include "engine/ops/ops.h"
#include "engine/backends/cpu/ops/tensor.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>

#if TYLOINFER_HAS_CUDA
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/tensor.h"
#include "engine/backends/cuda/native/tensor/tensor.h"
#include "engine/backends/cuda/native/device.h"
#endif
#if TYLOINFER_HAS_METAL
#include "engine/backends/metal/ops/tensor.h"
#endif

namespace tyloi::ops {
struct DeviceScope::Impl {
#if TYLOINFER_HAS_CUDA
    std::optional<cuda::DeviceGuard> guard;
#endif
    explicit Impl(Device target) {
        if (target.is_cpu() && target.index == 0) return;
#if TYLOINFER_HAS_CUDA
        if (target.is_cuda() && target.index >= 0) {
            guard.emplace(target.index);
            return;
        }
#endif
#if TYLOINFER_HAS_METAL
        if (target.is_metal() && target.index == 0) return;
#endif
        throw std::invalid_argument("unsupported execution device");
    }
};
DeviceScope::DeviceScope(Device target) : impl_(std::make_unique<Impl>(target)) {}
DeviceScope::~DeviceScope() = default;

#if !TYLOINFER_HAS_CUDA
namespace {
const DenseTensor& cpu_view(const Tensor& tensor) {
    const auto* value = tensor.representation<DenseTensor>();
    if (!value || !value->defined() || value->is_encoded() ||
        value->device() != Device{DeviceType::cpu, 0})
        throw std::invalid_argument("view requires a dense CPU:0 tensor");
    return *value;
}
} // namespace
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
#if TYLOINFER_HAS_METAL
    if (target.is_metal() || (value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::copy(value_tensor, target);
#endif
#if TYLOINFER_HAS_CUDA
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
#if TYLOINFER_HAS_METAL
    if ((destination_tensor.defined() && destination_tensor.device().is_metal()) ||
        (source_tensor.defined() && source_tensor.device().is_metal())) {
        metal::ops::copy_into(destination_tensor, source_tensor);
        return;
    }
#endif
#if TYLOINFER_HAS_CUDA
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
#if TYLOINFER_HAS_METAL
    if (value_tensor.defined() && value_tensor.device().is_metal()) return metal::ops::clone(value_tensor);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::clone(value));
#else
    return cpu::ops::clone(value_tensor);
#endif
}

Tensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options) {
#if TYLOINFER_HAS_METAL
    if (options.target_device().is_metal()) return metal::ops::zeros(shape, options);
#endif
#if TYLOINFER_HAS_CUDA
    DeviceScope scope(options.target_device());
    return owner(cuda::ops::zeros(shape, options));
#else
    return cpu::ops::zeros(shape, options);
#endif
}

Tensor empty(std::span<const std::int64_t> shape, const TensorOptions& options) {
#if TYLOINFER_HAS_METAL
    if (options.target_device().is_metal()) return metal::ops::empty(shape, options);
#endif
#if TYLOINFER_HAS_CUDA
    DeviceScope scope(options.target_device());
    return owner(cuda::empty(shape, options));
#else
    return cpu::ops::empty(shape, options);
#endif
}

void fill(Tensor& value_tensor, double fill_value) {
#if TYLOINFER_HAS_METAL
    if (value_tensor.defined() && value_tensor.device().is_metal()) { metal::ops::fill(value_tensor, fill_value); return; }
#endif
#if TYLOINFER_HAS_CUDA
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
    if (value_tensor.defined() && value_tensor.device().is_cpu()) return cpu::ops::cast(value_tensor, type);
#if TYLOINFER_HAS_METAL
    if (value_tensor.defined() && value_tensor.device().is_metal()) return metal::ops::cast(value_tensor, type);
#endif
#if TYLOINFER_HAS_CUDA
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
#if TYLOINFER_HAS_METAL
    if (value_tensor.defined() && value_tensor.device().is_metal()) return metal::ops::slice(value_tensor, dim, start, end, step);
#endif
#if TYLOINFER_HAS_CUDA
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
#if TYLOINFER_HAS_METAL
    if (destination.device().is_metal()) return metal::ops::update_slice(destination, source, dim, start);
#endif
    auto target = slice(destination, dim, start, start + extent, 1);
    copy_into(target, source);
    return destination;
}

Tensor cat(std::span<const Tensor> values, std::int64_t dim) {
    if (!values.empty() && values.front().defined() && values.front().device().is_cpu())
        return cpu::ops::cat(values, dim);
#if TYLOINFER_HAS_METAL
    for (const auto& value : values)
        if (value.defined() && value.device().is_metal()) return metal::ops::cat(values, dim);
#endif
#if TYLOINFER_HAS_CUDA
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
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()) ||
        (indices_tensor.defined() && indices_tensor.device().is_metal()))
        return metal::ops::index_select(value_tensor, dim, indices_tensor);
#endif
#if TYLOINFER_HAS_CUDA
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
#if TYLOINFER_HAS_METAL
    if (value_tensor.defined() && value_tensor.device().is_metal()) {
        metal::ops::index_copy(value_tensor, dim, indices_tensor, source_tensor);
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
    cuda::ops::index_copy(value, dim, indices, source);
#else
    throw std::invalid_argument("indexed update is not supported");
#endif
}

Tensor transpose(const Tensor& value_tensor, std::int64_t first, std::int64_t second) {
#if TYLOINFER_HAS_METAL
    if (value_tensor.defined() && value_tensor.device().is_metal())
        return metal::ops::transpose(value_tensor, first, second);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::transpose(value, first, second));
#else
    return std::make_shared<DenseTensor>(tyloi::transpose(cpu_view(value_tensor), first, second));
#endif
}

Tensor contiguous(const Tensor& value_tensor) {
#if TYLOINFER_HAS_METAL
    if (value_tensor.defined() && value_tensor.device().is_metal())
        return metal::ops::contiguous(value_tensor);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(cuda::ops::contiguous(value));
#else
    const auto& value = cpu_view(value_tensor);
    return value.is_contiguous() ? value_tensor : cpu::ops::clone(value_tensor);
#endif
}

Tensor reshape(const Tensor& value_tensor, std::span<const std::int64_t> shape) {
#if TYLOINFER_HAS_METAL
    if (value_tensor.defined() && value_tensor.device().is_metal())
        return metal::ops::reshape(value_tensor, shape);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::reshape(value, shape));
#else
    return std::make_shared<DenseTensor>(tyloi::reshape(cpu_view(value_tensor), shape));
#endif
}

Tensor expand(const Tensor& value_tensor, std::span<const std::int64_t> shape) {
#if TYLOINFER_HAS_METAL
    if (value_tensor.defined() && value_tensor.device().is_metal())
        return metal::ops::expand(value_tensor, shape);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::expand(value, shape));
#else
    return std::make_shared<DenseTensor>(tyloi::expand(cpu_view(value_tensor), shape));
#endif
}

Tensor select(const Tensor& value_tensor, std::int64_t dim, std::int64_t index) {
#if TYLOINFER_HAS_METAL
    if (value_tensor.defined() && value_tensor.device().is_metal())
        return metal::ops::select(value_tensor, dim, index);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::select(value, dim, index));
#else
    return std::make_shared<DenseTensor>(tyloi::select(cpu_view(value_tensor), dim, index));
#endif
}

Tensor unsqueeze(const Tensor& value_tensor, std::int64_t dim) {
#if TYLOINFER_HAS_METAL
    if (value_tensor.defined() && value_tensor.device().is_metal())
        return metal::ops::unsqueeze(value_tensor, dim);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owner(tyloi::unsqueeze(value, dim));
#else
    return std::make_shared<DenseTensor>(tyloi::unsqueeze(cpu_view(value_tensor), dim));
#endif
}

Tensor tensor(std::span<const float> values, Device target) {
#if TYLOINFER_HAS_METAL
    if (target.is_metal())
        return metal::ops::tensor(values, target);
#endif
#if TYLOINFER_HAS_CUDA
    device(target);
    return owner(cuda::ops::tensor(values, target));
#else
    if (target != Device{DeviceType::cpu, 0}) throw std::invalid_argument("tensor requires CPU:0 or a compiled device");
    if (values.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("tensor extent exceeds int64 limit");
    auto result = cpu::ops::empty(std::array{static_cast<std::int64_t>(values.size())},
        TensorOptions().dtype(kFloat32).device(target));
    if (!values.empty()) std::memcpy(result.dense().data_ptr(), values.data(), result.nbytes());
    return result;
#endif
}

Tensor tensor(std::span<const std::int32_t> values, Device target) {
#if TYLOINFER_HAS_METAL
    if (target.is_metal())
        return metal::ops::tensor(values, target);
#endif
#if TYLOINFER_HAS_CUDA
    device(target);
    return owner(cuda::ops::tensor(values, target));
#else
    if (target != Device{DeviceType::cpu, 0}) throw std::invalid_argument("tensor requires CPU:0 or a compiled device");
    if (values.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("tensor extent exceeds int64 limit");
    auto result = cpu::ops::empty(std::array{static_cast<std::int64_t>(values.size())},
        TensorOptions().dtype(kInt32).device(target));
    if (!values.empty()) std::memcpy(result.dense().data_ptr(), values.data(), result.nbytes());
    return result;
#endif
}

Tensor tensor(std::span<const std::int64_t> values, Device target) {
#if TYLOINFER_HAS_METAL
    if (target.is_metal())
        return metal::ops::tensor(values, target);
#endif
#if TYLOINFER_HAS_CUDA
    device(target);
    return owner(cuda::ops::tensor(values, target));
#else
    if (target != Device{DeviceType::cpu, 0}) throw std::invalid_argument("tensor requires CPU:0 or a compiled device");
    if (values.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("tensor extent exceeds int64 limit");
    auto result = cpu::ops::empty(std::array{static_cast<std::int64_t>(values.size())},
        TensorOptions().dtype(kInt64).device(target));
    if (!values.empty()) std::memcpy(result.dense().data_ptr(), values.data(), result.nbytes());
    return result;
#endif
}

Tensor tensor(std::span<const std::uint8_t> values, Device target) {
#if TYLOINFER_HAS_METAL
    if (target.is_metal())
        return metal::ops::tensor(values, target);
#endif
#if TYLOINFER_HAS_CUDA
    device(target);
    return owner(cuda::ops::tensor(values, target));
#else
    if (target != Device{DeviceType::cpu, 0}) throw std::invalid_argument("tensor requires CPU:0 or a compiled device");
    if (values.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("tensor extent exceeds int64 limit");
    auto result = cpu::ops::empty(std::array{static_cast<std::int64_t>(values.size())},
        TensorOptions().dtype(kUInt8).device(target));
    if (!values.empty()) std::memcpy(result.dense().data_ptr(), values.data(), result.nbytes());
    return result;
#endif
}

Tensor arange(std::int64_t first, std::int64_t last, Device target) {
#if TYLOINFER_HAS_METAL
    if (target.is_metal())
        return metal::ops::arange(first, last, target);
#endif
#if TYLOINFER_HAS_CUDA
    device(target);
    return owner(cuda::ops::arange(first, last, target));
#else
    if (target != Device{DeviceType::cpu, 0}) throw std::invalid_argument("arange requires CPU:0 or a compiled device");
    const auto count = last > first ? static_cast<std::uint64_t>(last) - static_cast<std::uint64_t>(first) : 0;
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("arange extent exceeds int64 limit");
    auto result = cpu::ops::empty(std::array{static_cast<std::int64_t>(count)}, TensorOptions().dtype(kInt64).device(target));
    if (count) std::iota(result.dense().data_ptr<std::int64_t>(), result.dense().data_ptr<std::int64_t>() + count, first);
    return result;
#endif
}

Tensor full(std::span<const std::int64_t> shape, std::int64_t value, Device target) {
#if TYLOINFER_HAS_METAL
    if (target.is_metal())
        return metal::ops::full(shape, value, target);
#endif
#if TYLOINFER_HAS_CUDA
    device(target);
    return owner(cuda::ops::full(shape, value, target));
#else
    if (target != Device{DeviceType::cpu, 0}) throw std::invalid_argument("full requires CPU:0 or a compiled device");
    auto result = cpu::ops::empty(shape, TensorOptions().dtype(kInt64).device(target));
    if (result.numel()) std::fill_n(result.dense().data_ptr<std::int64_t>(), result.numel(), value);
    return result;
#endif
}

Tensor from_blob(void* data, std::span<const std::int64_t> shape, const TensorOptions& options) {
#if TYLOINFER_HAS_METAL
    if (options.target_device().is_metal())
        throw std::invalid_argument("Metal from_blob cannot borrow external storage; use tensor or copy");
#endif
#if TYLOINFER_HAS_CUDA
    device(options.target_device());
    return owner(cuda::ops::from_blob(data, shape, options));
#else
    if (options.target_device() != Device{DeviceType::cpu, 0})
        throw std::invalid_argument("from_blob requires CPU:0 or a compiled device");
    auto view = make_contiguous_view(data, shape, options.scalar_type(), options.target_device());
    const auto bytes = view.nbytes();
    if (bytes && !data) throw std::invalid_argument("from_blob requires data for a nonempty view");
    auto storage = std::make_shared<TensorStorage>();
    storage->base = data;
    storage->bytes = bytes;
    storage->device = options.target_device();
    storage->pinned = options.pinned();
    return Tensor{DenseTensor(std::move(storage), view)};
#endif
}

Tensor cast(const Tensor& value_tensor, ScalarType type, Device target) {
#if TYLOINFER_HAS_METAL
    if ((value_tensor.defined() && value_tensor.device().is_metal()) || target.is_metal()) {
        if (!value_tensor.defined() || value_tensor.is_encoded())
            throw std::invalid_argument("targeted Metal cast requires an unencoded tensor");
        // Convert on Metal so a CPU target does not require an extra host numeric implementation.
        auto device_value = value_tensor.device().is_metal() ? value_tensor : metal::ops::copy(value_tensor, target);
        return metal::ops::copy(metal::ops::cast(device_value, type), target);
    }
#endif
#if TYLOINFER_HAS_CUDA
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    device(target);
    return owner(cuda::ops::cast(value, type, target));
#else
    return copy(cast(value_tensor, type), target);
#endif
}

std::vector<Tensor> chunk(const Tensor& value_tensor, std::int64_t count, std::int64_t dim) {
#if TYLOINFER_HAS_CUDA
#if TYLOINFER_HAS_METAL
    if (!(value_tensor.defined() && value_tensor.device().is_metal())) {
#endif
    const auto& value = ::tyloi::ops::dense(value_tensor);
    input(value);
    return owners(tyloi::chunk(value, count, dim));
#if TYLOINFER_HAS_METAL
    }
#endif
#endif
    if (!value_tensor.defined() || value_tensor.is_encoded()) throw std::invalid_argument("chunk requires an unencoded tensor");
    if (count <= 0) throw std::invalid_argument("chunk count must be positive");
    const auto rank = value_tensor.dim();
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank) throw std::out_of_range("tensor dimension is out of range");
    const auto extent = value_tensor.size(dim);
    const auto width = extent / count + (extent % count != 0);
    std::vector<Tensor> result;
    for (std::int64_t first = 0; first < extent;) {
        const auto last = first + std::min(width, extent - first);
        result.push_back(slice(value_tensor, dim, first, last));
        first = last;
    }
    return result;
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

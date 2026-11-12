#include "tensor.h"

#include "engine/backends/metal/native/stream.h"

#include <mlx/backend/metal/device.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace tyloi::metal {

ScalarType scalar_type(mlx::core::Dtype dtype) {
    namespace mx = mlx::core;
    if (dtype == mx::bool_) return ScalarType::boolean;
    if (dtype == mx::uint8) return ScalarType::uint8;
    if (dtype == mx::uint16) return ScalarType::uint16;
    if (dtype == mx::uint32) return ScalarType::uint32;
    if (dtype == mx::uint64) return ScalarType::uint64;
    if (dtype == mx::int8) return ScalarType::int8;
    if (dtype == mx::int16) return ScalarType::int16;
    if (dtype == mx::int32) return ScalarType::int32;
    if (dtype == mx::int64) return ScalarType::int64;
    if (dtype == mx::float16) return ScalarType::float16;
    if (dtype == mx::bfloat16) return ScalarType::bfloat16;
    if (dtype == mx::float32) return ScalarType::float32;
    throw std::invalid_argument("unsupported MLX tensor dtype");
}

namespace {

template <typename Data>
auto allocation_buffer(Data& data) {
    if constexpr (requires { data.buffer(); }) return data.buffer();
    else return data.buffer;
}

class Array final : public TensorImpl {
public:
    explicit Array(mlx::core::array value)
        : value_(std::move(value)), dtype_(metal::scalar_type(value_.dtype())) {}

    bool defined() const noexcept override { return true; }
    std::int64_t dim() const noexcept override { return value_.ndim(); }
    std::int64_t numel() const override {
        if (value_.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
            throw std::overflow_error("MLX tensor element count overflow");
        return static_cast<std::int64_t>(value_.size());
    }
    std::size_t nbytes() const override { return value_.nbytes(); }
    std::size_t backing_nbytes() const override {
        return backing_bytes(backings());
    }
    std::int64_t size(std::int64_t dimension) const override {
        if (dimension < 0) dimension += dim();
        if (dimension < 0 || dimension >= dim()) throw std::out_of_range("tensor dimension is out of range");
        return value_.shape(static_cast<int>(dimension));
    }
    ScalarType scalar_type() const noexcept override { return dtype_; }
    const Device& device() const noexcept override { return device_; }
    bool is_contiguous() const noexcept override { return value_.flags().row_contiguous; }
    bool is_encoded() const noexcept override { return false; }
    bool is_pinned() const noexcept override { return false; }
    std::vector<const DenseTensor*> buffers() const override { return {}; }
    std::vector<std::shared_ptr<TensorStorage>> backings() const override {
        std::vector<std::shared_ptr<TensorStorage>> storages;
        std::vector<mlx::core::array> worklist{value_};
        std::unordered_set<std::uintptr_t> visited;
        while (!worklist.empty()) {
            auto node = std::move(worklist.back());
            worklist.pop_back();
            if (!visited.insert(node.id()).second) continue;
            auto data = node.data_shared_ptr();
            if (data) {
                auto buffer = allocation_buffer(*data);
                if (buffer.ptr()) {
                    auto storage = std::make_shared<TensorStorage>();
                    storage->owner = std::shared_ptr<void>(data, buffer.ptr());
                    storage->bytes = static_cast<MTL::Buffer*>(buffer.ptr())->allocatedSize();
                    storage->device = device_;
                    storages.push_back(std::move(storage));
                    continue;
                }
            }
            const auto& inputs = node.inputs();
            worklist.insert(worklist.end(), inputs.rbegin(), inputs.rend());
        }
        return storages;
    }
    void record_stream(std::uintptr_t, std::shared_ptr<void>) const override {
        throw std::runtime_error("Metal tensor stream recording is not supported");
    }
    std::shared_ptr<TensorImpl> relocated(Device device) const override {
        if (device == device_) return std::make_shared<Array>(value_);
        return metal::copy(metal::tensor(value_), device).implementation();
    }

    const mlx::core::array& value() const noexcept { return value_; }

private:
    mlx::core::array value_;
    ScalarType dtype_;
    Device device_{DeviceType::metal, 0};
};

}  // namespace

mlx::core::Dtype dtype(ScalarType type) {
    namespace mx = mlx::core;
    switch (type) {
        case ScalarType::boolean: return mx::bool_;
        case ScalarType::uint8: return mx::uint8;
        case ScalarType::uint16: return mx::uint16;
        case ScalarType::uint32: return mx::uint32;
        case ScalarType::uint64: return mx::uint64;
        case ScalarType::int8: return mx::int8;
        case ScalarType::int16: return mx::int16;
        case ScalarType::int32: return mx::int32;
        case ScalarType::int64: return mx::int64;
        case ScalarType::float16: return mx::float16;
        case ScalarType::bfloat16: return mx::bfloat16;
        case ScalarType::float32: return mx::float32;
        default: throw std::invalid_argument("unsupported Metal tensor dtype");
    }
}

namespace {

struct Layout {
    mlx::core::Shape shape;
    std::size_t elements;
    std::size_t bytes;
};

Layout layout(std::span<const std::int64_t> shape, ScalarType type) {
    (void)dtype(type);
    if (shape.size() > kMaximumTensorRank) throw std::invalid_argument("tensor rank exceeds maximum");
    Layout result{{}, 1, 0};
    bool empty = false;
    for (auto extent : shape) {
        if (extent < 0 || extent > std::numeric_limits<int>::max())
            throw std::invalid_argument("invalid Metal tensor extent");
        result.shape.push_back(static_cast<int>(extent));
        empty |= extent == 0;
    }
    if (empty) result.elements = 0;
    else for (auto extent : shape) {
        if (result.elements > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) / extent)
            throw std::overflow_error("tensor element count overflow");
        result.elements *= static_cast<std::size_t>(extent);
    }
    const auto width = scalar_size(type);
    if (result.elements > std::numeric_limits<std::size_t>::max() / width)
        throw std::overflow_error("tensor byte count overflow");
    result.bytes = result.elements * width;
    return result;
}

}  // namespace

Tensor copy_from_host(const void* data, std::span<const std::int64_t> shape, ScalarType type) {
    auto dimensions = layout(shape, type);
    if (dimensions.bytes && !data) throw std::invalid_argument("null tensor data");
    auto result = mlx::core::array(
        mlx::core::allocator::malloc(dimensions.bytes), std::move(dimensions.shape), dtype(type));
    if (dimensions.bytes) std::memcpy(result.data<std::byte>(), data, dimensions.bytes);
    return tensor(std::move(result));
}

Tensor copy(const Tensor& value, Device target) {
    const Device cpu{DeviceType::cpu, 0};
    const Device gpu{DeviceType::metal, 0};
    if (!value.defined() || value.is_encoded() || (target != cpu && target != gpu) ||
        (value.device() != cpu && value.device() != gpu))
        throw std::invalid_argument("copy requires CPU 0 or Metal 0 dense tensors");
    const auto* dense = value.representation<DenseTensor>();
    const auto* metal_array = value.representation<Array>();
    if ((value.device() == cpu && !dense) || (value.device() == gpu && !metal_array))
        throw std::invalid_argument("unsupported tensor representation");
    if (value.device() == target) return value;

    if (dense) {
        const auto& view = dense->view_descriptor();
        if (view.rank > kMaximumTensorRank) throw std::invalid_argument("tensor rank exceeds maximum");
        const auto shape = std::span(view.sizes.data(), view.rank);
        const auto dimensions = layout(shape, view.scalar_type);
        if (dimensions.bytes && !view.data) throw std::invalid_argument("null tensor data");
        if (!dimensions.bytes || view.is_contiguous()) return copy_from_host(view.data, shape, view.scalar_type);
        auto result = mlx::core::array(
            mlx::core::allocator::malloc(dimensions.bytes), dimensions.shape, dtype(view.scalar_type));
        auto* output = result.data<std::byte>();
        const auto* input = static_cast<const std::byte*>(view.data);
        const auto width = scalar_size(view.scalar_type);
        for (std::size_t index = 0; index < dimensions.elements; ++index) {
            auto remaining = index;
            __int128 offset = 0;
            for (std::size_t axis = view.rank; axis > 0; --axis) {
                const auto extent = static_cast<std::size_t>(view.sizes[axis - 1]);
                offset += static_cast<__int128>(remaining % extent) * view.strides[axis - 1];
                remaining /= extent;
            }
            offset *= width;
            if (offset < std::numeric_limits<std::ptrdiff_t>::min() ||
                offset > std::numeric_limits<std::ptrdiff_t>::max())
                throw std::overflow_error("tensor stride offset overflow");
            std::memcpy(output + index * width, input + static_cast<std::ptrdiff_t>(offset), width);
        }
        return tensor(std::move(result));
    }

    const auto& input = metal_array->value();
    std::vector<std::int64_t> shape(input.shape().begin(), input.shape().end());
    const auto dimensions = layout(shape, value.scalar_type());
    const auto view = make_contiguous_view(nullptr, shape, value.scalar_type(), cpu);
    auto source = mlx::core::contiguous(input);
    metal::eval(source);
    auto storage = std::make_shared<TensorStorage>();
    if (dimensions.bytes) {
        storage->owner = std::shared_ptr<void>(::operator new(dimensions.bytes), [](void* data) { ::operator delete(data); });
        storage->base = storage->owner.get();
        std::memcpy(storage->base, source.data<std::byte>(), dimensions.bytes);
    }
    storage->bytes = dimensions.bytes;
    storage->device = cpu;
    return DenseTensor(storage, view);
}

tyloi::Tensor tensor(mlx::core::array value) {
    return tyloi::Tensor{std::make_shared<Array>(std::move(value))};
}

const mlx::core::array& array(const tyloi::Tensor& value) {
    const auto* representation = value.representation<Array>();
    if (!representation) throw std::invalid_argument("tensor is not an MLX array");
    return representation->value();
}

}  // namespace tyloi::metal

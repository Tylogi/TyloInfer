#include "tensor.h"

#include <mlx/backend/metal/device.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::metal {
namespace {

template <typename Data>
auto allocation_buffer(Data& data) {
    if constexpr (requires { data.buffer(); }) return data.buffer();
    else return data.buffer;
}

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
        const auto storage = backings();
        return storage.empty() ? 0 : storage.front()->bytes;
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
    std::vector<const DenseTensor*> buffers() const override { return {}; }
    std::vector<std::shared_ptr<TensorStorage>> backings() const override {
        auto data = value_.data_shared_ptr();
        if (!data) return {};
        auto buffer = allocation_buffer(*data);
        if (!buffer.ptr()) return {};
        auto storage = std::make_shared<TensorStorage>();
        storage->owner = std::shared_ptr<void>(data, buffer.ptr());
        storage->bytes = static_cast<MTL::Buffer*>(buffer.ptr())->allocatedSize();
        storage->device = device_;
        return {std::move(storage)};
    }
    void record_stream(std::uintptr_t, std::shared_ptr<void>) const override {
        throw std::runtime_error("Metal tensor stream recording is not supported");
    }
    std::shared_ptr<TensorImpl> relocated(Device device) const override {
        if (device != device_) throw std::invalid_argument("Metal tensor relocation is not supported");
        return std::make_shared<Array>(value_);
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

tyloi::Tensor tensor(mlx::core::array value) {
    return tyloi::Tensor{std::make_shared<Array>(std::move(value))};
}

const mlx::core::array& array(const tyloi::Tensor& value) {
    const auto* representation = value.representation<Array>();
    if (!representation) throw std::invalid_argument("tensor is not an MLX array");
    return representation->value();
}

}  // namespace tyloi::metal

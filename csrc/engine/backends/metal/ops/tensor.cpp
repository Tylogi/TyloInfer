#include "tensor.h"

#include "engine/backends/metal/native/tensor/tensor.h"

#include <mlx/mlx.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tyloi::metal::ops {
namespace {

const mlx::core::array& input(const Tensor& value) {
    if (!value.defined() || value.device() != Device{DeviceType::metal, 0} || value.is_encoded()) {
        throw std::invalid_argument("expected a defined, unencoded Metal 0 tensor");
    }
    return metal::array(value);
}

int dimension(std::int64_t dim, std::size_t rank) {
    if (rank > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("Metal tensor rank exceeds int32 limit");
    const auto count = static_cast<std::int64_t>(rank);
    if (dim < 0) dim += count;
    if (dim < 0 || dim >= count) throw std::out_of_range("tensor dimension is out of range");
    return static_cast<int>(dim);
}

}  // namespace

Tensor clone(const Tensor& value) {
    const auto& source = input(value);
    if (source.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("Metal clone tensor element count exceeds int32 limit");
    }
    // MLX may elide copy() when the result is otherwise identical to its input.
    // Cache snapshots must own a distinct allocation because live cache updates
    // can write through aliases. Appending and then slicing off one sentinel forces
    // a new backing allocation without applying arithmetic to the tensor values.
    auto flat = mlx::core::reshape(source, {-1});
    auto storage = mlx::core::concatenate(
        {flat, mlx::core::zeros({1}, source.dtype())},
        0);
    return metal::tensor(mlx::core::reshape(
        mlx::core::slice(storage, {0}, {static_cast<int>(source.size())}),
        source.shape()));
}

Tensor cast(const Tensor& value, ScalarType type) {
    const auto& source = input(value);
    if (value.scalar_type() == type) return value;
    const auto target = metal::dtype(type);
    return metal::tensor(mlx::core::astype(source, target));
}

Tensor reshape(const Tensor& value, std::span<const std::int64_t> shape) {
    const auto& source = input(value);
    if (shape.size() > kMaximumTensorRank) throw std::invalid_argument("tensor rank exceeds maximum");
    mlx::core::Shape target;
    target.reserve(shape.size());
    bool inferred = false;
    for (const auto extent : shape) {
        if (extent < -1 || extent > std::numeric_limits<int>::max() || (extent == -1 && inferred))
            throw std::invalid_argument("invalid reshape extent");
        inferred |= extent == -1;
        target.push_back(static_cast<int>(extent));
    }
    return metal::tensor(mlx::core::reshape(source, target));
}

Tensor transpose(const Tensor& value, std::int64_t first, std::int64_t second) {
    const auto& source = input(value);
    const auto a = dimension(first, source.ndim());
    const auto b = dimension(second, source.ndim());
    return metal::tensor(mlx::core::swapaxes(source, a, b));
}

Tensor slice(const Tensor& value, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step) {
    const auto& source = input(value);
    if (step <= 0 || step > std::numeric_limits<int>::max())
        throw std::invalid_argument("Metal slice requires a positive int32 step");
    const auto selected = dimension(dim, source.ndim());
    const std::int64_t extent = source.shape(selected);
    auto first = std::clamp(start < 0 ? start + extent : start, std::int64_t{0}, extent);
    auto last = std::clamp(end < 0 ? end + extent : end, std::int64_t{0}, extent);
    last = std::max(first, last);
    mlx::core::Shape starts(source.ndim(), 0);
    mlx::core::Shape stops = source.shape();
    mlx::core::Shape strides(source.ndim(), 1);
    starts[selected] = static_cast<int>(first);
    stops[selected] = static_cast<int>(last);
    strides[selected] = static_cast<int>(step);
    return metal::tensor(mlx::core::slice(source, starts, stops, strides));
}

Tensor contiguous(const Tensor& value) {
    const auto& source = input(value);
    if (value.is_contiguous()) return value;
    return metal::tensor(mlx::core::contiguous(source));
}

}  // namespace tyloi::metal::ops

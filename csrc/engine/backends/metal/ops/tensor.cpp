#include "tensor.h"

#include "engine/backends/metal/native/tensor/tensor.h"

#include <mlx/mlx.h>

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

}  // namespace tyloi::metal::ops

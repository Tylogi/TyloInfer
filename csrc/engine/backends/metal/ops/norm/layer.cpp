#include "layer.h"
#include "engine/backends/metal/native/tensor/tensor.h"

#include <mlx/mlx.h>
#include <optional>
#include <stdexcept>

namespace tyloi::metal::ops {

Tensor layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias, double epsilon) {
    const Device device{DeviceType::metal, 0};
    if (!input.defined() || !weight.defined() || !bias.defined() || input.is_encoded() ||
        weight.is_encoded() || bias.is_encoded() || input.device() != device ||
        weight.device() != device || bias.device() != device || input.dim() == 0 ||
        weight.dim() != 1 || bias.dim() != 1 || input.size(-1) <= 0 ||
        weight.size(0) != input.size(-1) || bias.size(0) != input.size(-1))
        throw std::invalid_argument("layer_norm input and weights are incompatible");
    const auto floating = [](ScalarType type) {
        return type == ScalarType::float16 || type == ScalarType::bfloat16 || type == ScalarType::float32;
    };
    if (!floating(input.scalar_type()) || !floating(weight.scalar_type()) || !floating(bias.scalar_type()))
        throw std::invalid_argument("layer_norm requires floating tensors");
    const auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
    const float eps = static_cast<float>(epsilon);
    const auto output_type = metal::dtype(input.scalar_type());
    auto source = mlx::core::astype(metal::array(input), mlx::core::float32, stream);
    auto scale = mlx::core::astype(metal::array(weight), mlx::core::float32, stream);
    auto shift = mlx::core::astype(metal::array(bias), mlx::core::float32, stream);
    auto result = mlx::core::fast::layer_norm(source, std::optional<mlx::core::array>(scale),
        std::optional<mlx::core::array>(shift), eps, stream);
    return metal::tensor(mlx::core::astype(result, output_type, stream));
}

} // namespace tyloi::metal::ops

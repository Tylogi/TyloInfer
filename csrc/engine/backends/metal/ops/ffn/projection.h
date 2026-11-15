#pragma once

#include "engine/backends/metal/native/tensor/tensor.h"
#include "engine/weights/encoding/metal.h"
#include "backends/metal/runtime/mlx_tensor.h"

#include <stdexcept>

namespace tyloi::metal {
inline MlxLinear ffn_projection(const Tensor& weight) {
    if (const auto* encoded = weight.representation<encoding::MetalTensor>()) return encoded->linear();
    if (!weight || !weight.is_metal() || weight.dim() != 2)
        throw std::invalid_argument("invalid Metal FFN projection");
    return MlxLinear(array(weight));
}
} // namespace tyloi::metal

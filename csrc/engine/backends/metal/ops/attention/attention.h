#pragma once

#include <mlx/mlx.h>
#include <optional>

namespace mfq::metal {
mlx::core::array scaled_dot_product_attention(
    const mlx::core::array& query, const mlx::core::array& key, const mlx::core::array& value,
    bool causal = true, float scale = 0.0f,
    const std::optional<mlx::core::array>& mask = std::nullopt);
}

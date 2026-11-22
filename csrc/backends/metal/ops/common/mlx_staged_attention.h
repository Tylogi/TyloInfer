#pragma once

#include <mlx/mlx.h>
#include <optional>

namespace mfq::metal {
std::optional<mlx::core::array> mlx_staged_attention(
    const mlx::core::array& query, const mlx::core::array& key,
    const mlx::core::array& value, bool causal, float scale,
    const std::optional<mlx::core::array>& mask);

// The table is [batch, query, head, distance]. Positions beyond the table
// have zero relative bias before subtracting the query's sink logit.
mlx::core::array mlx_attention_relative_bias(const mlx::core::array& table, const mlx::core::array& sink_logits,
    const mlx::core::array& key_positions, int first_row, int count,
    int first_position, int window, mlx::core::Dtype dtype);
}

#pragma once

#include "engine/state/kv_view.h"

#include <mlx/mlx.h>
#include <optional>
#include <utility>

namespace mfq::metal {

// The caller validates topology and reserves capacity, then publishes both returned physical arrays.
std::pair<mlx::core::array, mlx::core::array> mlx_kv_append(
    const tyloi::state::KvView& kv, const mlx::core::array& key,
    const mlx::core::array& value, int position);
std::pair<mlx::core::array, mlx::core::array> mlx_kv_view(
    const tyloi::state::KvView& kv, int position);
mlx::core::array mlx_kv_attention(const mlx::core::array& query,
    const tyloi::state::KvView& kv, int position, bool causal = true);

int mlx_kv_packed_width(int dimension, int bits);
mlx::core::array mlx_kv_encode(const mlx::core::array& input, int bits, bool value);
mlx::core::array mlx_kv_decode(const mlx::core::array& input, int dimension, int bits,
    bool value, mlx::core::Dtype dtype = mlx::core::float16);
mlx::core::array mlx_kv_sparse_attention(const mlx::core::array& query,
    const mlx::core::array& key, const mlx::core::array& value,
    const std::optional<mlx::core::array>& blocks, tyloi::state::KvQuantization quantization,
    int position, int query_offset, int ratio, int budget,
    const std::optional<mlx::core::array>& key_mask = std::nullopt);

}

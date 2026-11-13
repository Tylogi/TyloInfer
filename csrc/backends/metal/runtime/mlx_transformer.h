#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxYarnScaling {
    bool enabled = false;
    double factor = 1.0;
    double beta_fast = 32.0;
    double beta_slow = 1.0;
    std::int64_t original_max_position_embeddings = 0;
    double attention_factor = 1.0;
};

std::int64_t qwen_yarn_capacity(std::int64_t native_context, double maximum_factor);
MlxYarnScaling qwen_yarn_scaling(const MlxYarnScaling& settings,
    std::int64_t native_context, std::int64_t requested_context, double maximum_factor);
double yarn_frequency(int pair, int dimension, double theta, const MlxYarnScaling& scaling);

// Model-neutral adjacent-pair RoPE primitives. Architecture adapters provide
// the configured YaRN parameters but do not own the table builder or kernel.
std::pair<mlx::core::array, mlx::core::array> mlx_yarn_tables(
    int dimension,
    int length,
    float theta,
    const MlxYarnScaling& scaling = {});

mlx::core::array mlx_rope_adjacent(
    const mlx::core::array& value,
    int rotary_dimension,
    const mlx::core::array& cosine,
    const mlx::core::array& sine,
    bool inverse = false);

mlx::core::array mlx_rope_adjacent(
    const mlx::core::array& value,
    const mlx::core::array& cosine,
    const mlx::core::array& sine,
    bool inverse = false);

mlx::core::array apply_rope(
    const mlx::core::array& input,
    int rotary_dimension,
    float base,
    int offset,
    const MlxYarnScaling& scaling = {});

// Apply rotate-half RoPE using explicit token positions. positions accepts
// [tokens], [1,tokens], or [3,tokens]. Three-axis positions select temporal,
// height, and width frequencies according to sections. With interleaved=true,
// the pair layout is THWTHW... (with each axis bounded by its section size);
// otherwise it is the contiguous T...H...W... layout.
mlx::core::array apply_rope(
    const mlx::core::array& input,
    const mlx::core::array& positions,
    int rotary_dimension,
    float base,
    const std::vector<std::int64_t>& sections = {},
    bool interleaved = false,
    const MlxYarnScaling& scaling = {});

// Return the visible rows of a [batch,window,...] circular cache in
// chronological order. `position` is the absolute position immediately after
// the last committed row.
mlx::core::array mlx_circular_cache_history(
    const mlx::core::array& cache,
    int position);

} // namespace mfq::metal

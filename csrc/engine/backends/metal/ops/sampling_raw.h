#pragma once

#include <mlx/mlx.h>

namespace tyloi::metal {
struct MlxTopKDistribution { mlx::core::array sampled, indices, probabilities; };
mlx::core::array sample_greedy(const mlx::core::array& logits);
mlx::core::array sample_softmax(const mlx::core::array& logits, const mlx::core::array& random,
    double temperature = 1.0);
mlx::core::array sample_top_k_top_p(const mlx::core::array& logits, const mlx::core::array& random,
    double temperature, int top_k, double top_p = 1.0);
mlx::core::array sample_nucleus(const mlx::core::array& logits, const mlx::core::array& random,
    double temperature, double top_p);
MlxTopKDistribution sample_top_k_distribution(const mlx::core::array& logits,
    const mlx::core::array& random, double temperature, int top_k, double top_p = 1.0);
mlx::core::array sample_token_counts_add(const mlx::core::array& counts, const mlx::core::array& tokens);
mlx::core::array sample_apply_penalties(const mlx::core::array& logits, const mlx::core::array& counts,
    double presence = 0.0, double frequency = 0.0, double repetition = 1.0);
} // namespace tyloi::metal

#pragma once

#include "engine/backends/metal/ops/sampling_raw.h"

#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include <mlx/mlx.h>

namespace tyloi::metal {

struct MlxSamplingParams {
    double temperature = 0.0;
    int top_k = 0;
    double top_p = 1.0;
    double presence_penalty = 0.0;
    double frequency_penalty = 0.0;
    double repetition_penalty = 1.0;
    bool enable_mtp = true;
    int mtp_max_draft_tokens = 3;
    std::uint64_t seed = 0;
    bool warmup = false;

    bool greedy() const noexcept {
        return temperature <= 0.0 || top_k == 1;
    }

    bool has_penalties() const noexcept {
        return presence_penalty != 0.0 ||
            frequency_penalty != 0.0 ||
            repetition_penalty != 1.0;
    }
};

// Select the final time step from [batch,tokens,vocab] logits and return
// [batch,vocab]. Generation adapters must not carry private copies of this
// shape policy.
mlx::core::array mlx_last_token_logits(
    const mlx::core::array& logits,
    int vocab);

void mlx_validate_token_set(
    std::span<const std::int64_t> token_ids,
    int vocab);

bool mlx_token_in_set(
    std::span<const std::int64_t> token_ids,
    std::int64_t token) noexcept;

// Stateless policy; an omitted random input starts a fresh stream at params.seed.
mlx::core::array sample(const mlx::core::array& logits,
    const MlxSamplingParams& params = {},
    const std::optional<mlx::core::array>& random = std::nullopt);

// Exact host-visible distribution used by stochastic speculative decoding.
// It applies the same temperature/top-k/top-p policy as ordinary sampling.
// The expensive path is confined to MTP verification, where proposal and
// target probabilities are required by the acceptance rule.
std::vector<float> host_sampling_distribution(
    const mlx::core::array& logits,
    const MlxSamplingParams& params);

std::int32_t sample_host_distribution(
    const std::vector<float>& probabilities,
    double uniform);

class MlxSampler {
public:
    explicit MlxSampler(MlxSamplingParams params = {});

    const MlxSamplingParams& params() const noexcept {
        return params_;
    }

    void reset_seed(std::uint64_t seed);

    // Advance the persistent RNG when an adapter sampled one or more tokens
    // before handing generation to the common MTP engine.
    void discard_random(std::uint64_t draws);

    // Sample raw logits using this sampler's persistent seeded RNG stream.
    mlx::core::array sample(const mlx::core::array& logits);

    // Apply configured penalties using counts, then sample the adjusted logits.
    mlx::core::array sample(
        const mlx::core::array& logits,
        const mlx::core::array& counts);

    mlx::core::array apply_penalties(
        const mlx::core::array& logits,
        const mlx::core::array& counts) const;

    double next_uniform();

private:
    mlx::core::array next_random(const mlx::core::array& logits);

    MlxSamplingParams params_;
    std::mt19937_64 rng_;
};

} // namespace tyloi::metal

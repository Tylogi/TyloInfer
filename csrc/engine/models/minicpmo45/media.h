#pragma once

#include "engine/models/common/check.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <span>
#include <utility>
#include <vector>

namespace tyloi::models::minicpmo45 {

struct MediaBound {
    int64_t batch = 0, source = 0, begin = 0, end = 0;
};

inline std::vector<MediaBound> media_bounds(std::span<const int64_t> values) {
    require_model(values.size() % 4 == 0, "media bounds require four columns");
    std::vector<MediaBound> result;
    result.reserve(values.size() / 4);
    for (size_t i = 0; i < values.size(); i += 4) {
        MediaBound bound{values[i], values[i + 1], values[i + 2], values[i + 3]};
        require_model(bound.batch >= 0 && bound.source >= 0 && bound.begin >= 0 &&
                          bound.end > bound.begin,
                      "invalid media bound");
        result.push_back(bound);
    }
    return result;
}

template <class Tensor> struct MultimodalResult {
    Tensor vision_states, image_embeddings, audio_embeddings, input_embeddings;
};

struct TtsSampling {
    int64_t eos_token = 6561, minimum_steps = 50, top_k = 25, minimum_keep = 0;
    double temperature = 0.8, top_p = 0.85, repetition_penalty = 1.05, min_p = 0.0;
    void validate(int64_t steps, int64_t vocabulary) const {
        require_model(steps > 0 && minimum_steps >= 0 && eos_token >= 0 && eos_token < vocabulary &&
                          std::isfinite(temperature) && temperature > 0.0 && top_p > 0.0 &&
                          top_p <= 1.0 && min_p >= 0.0 && min_p <= 1.0 && top_k >= 3 &&
                          top_k <= vocabulary && std::isfinite(repetition_penalty) &&
                          repetition_penalty > 0.0 && minimum_keep >= 0 && minimum_keep <= top_k,
                      "MiniCPM-o TTS generation limits are invalid");
    }
};

// Official TTS filters before temperature scaling; the reference evaluator
// scales first and keeps at least minimum_keep tokens. These orders differ
// from the text sampler and must not be interchanged.
template <class Ops>
auto sample_tts(Ops &ops, typename Ops::DenseTensor logits, const TtsSampling &config,
                std::span<const typename Ops::DenseTensor> history, int64_t step, bool evaluator) {
    if (evaluator)
        logits = ops.temperature(std::move(logits), config.temperature);
    if (!history.empty() && config.repetition_penalty != 1.0)
        logits =
            ops.penalties(std::move(logits), history.last(std::min<size_t>(16, history.size())),
                          config.repetition_penalty);
    if (step < config.minimum_steps)
        ops.mask_eos(logits, config.eos_token);
    if (evaluator)
        return ops.reference(std::move(logits), config.top_k, config.top_p, config.minimum_keep);
    logits = ops.top_k(std::move(logits), config.top_k);
    if (config.top_p < 1.0)
        logits = ops.top_p(std::move(logits), config.top_p);
    if (config.min_p > 0.0)
        logits = ops.min_p(std::move(logits), config.min_p);
    return ops.sample(ops.temperature(std::move(logits), config.temperature));
}

struct PatchPositions {
    std::vector<int64_t> ids;
    bool all_active = true;
};
inline PatchPositions patch_positions(int64_t batch, int64_t patches, int64_t side,
                                      const int64_t *sizes, const bool *mask) {
    require_model(batch > 0 && patches > 0 && side > 0, "invalid patch geometry");
    PatchPositions result{std::vector<int64_t>(batch * patches, 0), true};
    for (int64_t b = 0; b < batch; ++b) {
        const auto height = sizes[2 * b], width = sizes[2 * b + 1];
        require_model(height > 0 && width > 0 && height <= patches / width,
                      "MiniCPM-o target patch size is invalid");
        int64_t active = 0;
        for (int64_t patch = 0; patch < patches; ++patch) {
            if (!mask[b * patches + patch]) {
                result.all_active = false;
                continue;
            }
            require_model(active < height * width,
                          "MiniCPM-o patch mask has too many active entries");
            const auto row = std::min(side - 1, (active / width) * side / height);
            const auto column = std::min(side - 1, (active % width) * side / width);
            result.ids[b * patches + patch] = row * side + column;
            ++active;
        }
        require_model(active == height * width,
                      "MiniCPM-o patch mask active count disagrees with target size");
    }
    return result;
}

inline int64_t pooled_audio_length(int64_t frames) {
    const auto convolved = (frames - 1) / 2 + 1;
    const auto pooled = (convolved - 5) / 5 + 1;
    require_model(frames > 0 && pooled > 0, "MiniCPM-o audio length is too short");
    return pooled;
}
inline constexpr int64_t audio_chunk_frames = 50;

} // namespace tyloi::models::minicpmo45

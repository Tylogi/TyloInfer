#pragma once

#include "mlx_resource_telemetry.h"
#include "engine/state/kv_view.h"

#include <cstddef>
#include <cstdint>
#include <optional>
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

struct MlxKvCacheSnapshot {
    int batch = 0;
    int heads = 0;
    int maximum_sequence = 0;
    int head_dimension = 0;
    int capacity = 0;
    int position = 0;
    mlx::core::Dtype dtype = mlx::core::float16;
    mlx::core::array key = mlx::core::array(0.0f);
    mlx::core::array value = mlx::core::array(0.0f);
    tyloi::state::KvQuantization quantization;

    std::size_t nbytes() const noexcept {
        return key.nbytes() + value.nbytes();
    }
};

class MlxRmsNorm {
public:
    MlxRmsNorm(
        mlx::core::array weight,
        float eps = 1e-6f,
        float weight_offset = 0.0f);

    mlx::core::array operator()(
        const mlx::core::array& input,
        std::optional<mlx::core::Dtype> output_dtype = std::nullopt) const;

    int width() const noexcept {
        return width_;
    }
    const mlx::core::array& weight() const noexcept {
        return weight_;
    }
    float eps() const noexcept {
        return eps_;
    }

private:
    mlx::core::array weight_;
    float eps_;
    int width_;
};

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

class MlxKvCache {
public:
    MlxKvCache(
        int batch,
        int heads,
        int maximum_sequence,
        int head_dimension,
        int initial_capacity = 16,
        mlx::core::Dtype dtype = mlx::core::float16,
        bool quantization_allowed = false);

    void reset() noexcept {
        position_ = 0;
    }

    void materialize();

    void append_only(const mlx::core::array& key, const mlx::core::array& value);
    bool quantized() const noexcept { return kv_.quantization.enabled(); }
    const tyloi::state::KvView& storage() const noexcept { return kv_; }

    // Reserve cache rows without materializing an indexed update. Specialized
    // decode primitives use this when the projection/post-processing kernel
    // writes the new K/V row directly into the persistent Metal allocation.
    void reserve_append(int tokens);

    // Discard an appended speculative suffix. Storage is intentionally kept;
    // the next append overwrites the now-invisible rows.
    void trim(int tokens);

    // Session snapshots own a compact copy of the visible prefix. Restoring
    // recreates the original allocation capacity without aliasing the saved
    // arrays, so a resumed decode cannot mutate another session snapshot.
    MlxKvCacheSnapshot snapshot(bool detached = true) const;
    void restore_snapshot(const MlxKvCacheSnapshot& snapshot);
    // A freshly decoded persistent snapshot has no other owner. Adopt its
    // compact K/V allocations directly; the normal growth path reserves more
    // capacity on the first append when needed.
    void restore_snapshot(MlxKvCacheSnapshot&& snapshot);

    int position() const noexcept {
        return position_;
    }
    int capacity() const noexcept { return static_cast<int>(kv_.physical_key.size(2)); }
    const mlx::core::array& key_storage() const;
    const mlx::core::array& value_storage() const;

private:
    void ensure_capacity(int required);
    void set_storage(mlx::core::array key, mlx::core::array value);
    std::pair<mlx::core::array, mlx::core::array> physical_view() const;

    int maximum_sequence_;
    tyloi::state::KvView kv_;
    int position_ = 0;
    MlxResourceTelemetry resources_;
};

} // namespace mfq::metal

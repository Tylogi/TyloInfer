#include "sampling.h"
#include "sampling_raw.h"
#include "tensor.h"
#include "engine/backends/metal/native/tensor/tensor.h"
#include "engine/backends/metal/native/stream.h"

#include <mlx/mlx.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tyloi::metal::ops {
namespace {
using mlx::core::CompileOptions;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;
mlx::core::Stream gpu_stream() { return mlx::core::default_stream(mlx::core::Device::gpu); }
struct MlxTopKDistribution { array sampled; array indices; array probabilities; };
constexpr int kThreads = 256;
constexpr int kMaxTopK = 1024;
constexpr int kDirectTopK = 64;
constexpr int kHierarchicalTopK = 128;
constexpr int kTopKBlock = 1024;
constexpr int kGreedyBlock = 4096;

constexpr const char* kGreedyReductionHeader = R"METAL(
inline void mfq_greedy_reduce(thread float& value, thread int& id, uint lane, uint warp,
    threadgroup float* values, threadgroup int* ids) {
    for (uint stride = 16u; stride; stride >>= 1u) {
        float other = simd_shuffle_down(value, stride);
        int index = simd_shuffle_down(id, stride);
        if (lane + stride < 32u && (other > value || (other == value && index < id))) {
            value = other; id = index;
        }
    }
    if (lane == 0u) { values[warp] = value; ids[warp] = id; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0u) {
        value = lane < 8u ? values[lane] : -FLT_MAX;
        id = lane < 8u ? ids[lane] : 0;
        for (uint stride = 16u; stride; stride >>= 1u) {
            float other = simd_shuffle_down(value, stride);
            int index = simd_shuffle_down(id, stride);
            if (lane + stride < 32u && (other > value || (other == value && index < id))) {
                value = other; id = index;
            }
        }
    }
}
)METAL";

constexpr const char* kGreedyTilesSource = R"METAL(
    uint tile = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint tid = thread_index_in_threadgroup;
    float best = -FLT_MAX;
    int id = 0;
    for (uint token = tile * uint(BLOCK) + tid;
        token < min(uint(VOCAB), (tile + 1u) * uint(BLOCK)); token += 256u) {
        float value = float(logits[ulong(row) * uint(VOCAB) + token]);
        value = isnan(value) ? -FLT_MAX : value;
        if (value > best || (value == best && int(token) < id)) { best = value; id = int(token); }
    }
    threadgroup float values[8];
    threadgroup int ids[8];
    mfq_greedy_reduce(best, id, thread_index_in_simdgroup, simdgroup_index_in_threadgroup, values, ids);
    if (tid == 0u) {
        scores[ulong(row) * uint(TILES) + tile] = best;
        indices[ulong(row) * uint(TILES) + tile] = id;
    }
)METAL";

constexpr const char* kGreedyTilesFinalSource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    float best = -FLT_MAX;
    int id = 0;
    for (uint tile = tid; tile < uint(TILES); tile += 256u) {
        float value = scores[ulong(row) * uint(TILES) + tile];
        int index = indices[ulong(row) * uint(TILES) + tile];
        if (value > best || (value == best && index < id)) { best = value; id = index; }
    }
    threadgroup float values[8];
    threadgroup int ids[8];
    mfq_greedy_reduce(best, id, thread_index_in_simdgroup, simdgroup_index_in_threadgroup, values, ids);
    if (tid == 0u) output[row] = id;
)METAL";

constexpr const char* kGreedySource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    threadgroup float values[256];
    threadgroup int indices[256];
    float best = -FLT_MAX;
    int best_index = 0;
    uint offset = row * uint(VOCAB);
    for (uint token = tid; token < uint(VOCAB); token += 256u) {
        float value = float(logits[offset + token]);
        value = isnan(value) ? -FLT_MAX : value;
        if (value > best || (value == best && int(token) < best_index)) {
            best = value;
            best_index = int(token);
        }
    }
    values[tid] = best;
    indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            float other = values[tid + stride];
            int other_index = indices[tid + stride];
            if (
                other > values[tid]
                || (other == values[tid] && other_index < indices[tid])
            ) {
                values[tid] = other;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        output[row] = indices[0];
    }
)METAL";

constexpr const char* kSoftmaxSource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }
    threadgroup float reduction[256];
    threadgroup float maximum;
    uint offset = row * uint(VOCAB);
    float local_max = -FLT_MAX;
    for (uint token = tid; token < uint(VOCAB); token += 256u) {
        float value = float(logits[offset + token]) / params[0];
        value = isnan(value) ? -FLT_MAX : value;
        local_max = max(local_max, value);
    }
    reduction[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            reduction[tid] = max(reduction[tid], reduction[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        maximum = reduction[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint token = tid; token < uint(VOCAB); token += 256u) {
        float value = float(logits[offset + token]) / params[0];
        value = isnan(value) ? -FLT_MAX : value;
        local_sum += exp(value - maximum);
    }
    reduction[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            reduction[tid] += reduction[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        float uniform = clamp(random[row], 0.0f, 0.99999994f);
        float target = uniform * reduction[0];
        float cumulative = 0.0f;
        int chosen = int(VOCAB) - 1;
        for (uint token = 0u; token < uint(VOCAB); ++token) {
            float value = float(logits[offset + token]) / params[0];
            value = isnan(value) ? -FLT_MAX : value;
            cumulative += exp(value - maximum);
            if (cumulative >= target) {
                chosen = int(token);
                break;
            }
        }
        output[row] = chosen;
    }
)METAL";

constexpr const char* kTopKSource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;

    // Each lane scans its vocabulary stripe once and keeps a stable, sorted
    // local top-k.  The previous implementation scanned the entire vocabulary
    // once per output rank and also searched the already-selected ranks.
    float local_values[TOP_K];
    int local_indices[TOP_K];
    for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
        local_values[rank] = -INFINITY;
        local_indices[rank] = int(VOCAB);
    }

    uint offset = row * uint(VOCAB);
    for (uint token = tid; token < uint(VOCAB); token += 256u) {
        float value = float(logits[offset + token]) / params[0];
        value = isnan(value) ? -FLT_MAX : value;
        int token_index = int(token);
        uint last = uint(TOP_K) - 1u;
        bool enters =
            value > local_values[last]
            || (
                value == local_values[last]
                && token_index < local_indices[last]
            );
        if (!enters) {
            continue;
        }

        uint insert = last;
        while (insert > 0u) {
            float previous_value = local_values[insert - 1u];
            int previous_index = local_indices[insert - 1u];
            bool before =
                value > previous_value
                || (
                    value == previous_value
                    && token_index < previous_index
                );
            if (!before) {
                break;
            }
            local_values[insert] = previous_value;
            local_indices[insert] = previous_index;
            --insert;
        }
        local_values[insert] = value;
        local_indices[insert] = token_index;
    }

    // Merge the 32 sorted lane lists in each SIMD group.  Only TOP_K values
    // per SIMD group survive, so the threadgroup merge remains small even for
    // a 248k-token vocabulary.
    threadgroup float simd_values[8 * TOP_K];
    threadgroup int simd_indices[8 * TOP_K];
    uint local_rank = 0u;
    for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
        float best = local_values[local_rank];
        int best_index = local_indices[local_rank];
        for (uint stride = 16u; stride > 0u; stride >>= 1u) {
            float other = simd_shuffle_down(best, stride);
            int other_index = simd_shuffle_down(best_index, stride);
            if (
                lane < stride
                && (
                    other > best
                    || (other == best && other_index < best_index)
                )
            ) {
                best = other;
                best_index = other_index;
            }
        }
        best = simd_broadcast_first(best);
        best_index = simd_broadcast_first(best_index);
        if (local_indices[local_rank] == best_index) {
            ++local_rank;
        }
        if (lane == 0u) {
            uint destination =
                simd_group * uint(TOP_K) + rank;
            simd_values[destination] = best;
            simd_indices[destination] = best_index;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float top_values[TOP_K];
    threadgroup int top_indices[TOP_K];
    threadgroup float probabilities[TOP_K];

    if (tid == 0u) {
        // Merge the eight SIMD-group lists.  Ties remain deterministic:
        // greater logit first, then the lower token id.
        uint simd_ranks[8] = {
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        };
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            float best = -INFINITY;
            int best_index = int(VOCAB);
            uint best_group = 0u;
            for (uint group = 0u; group < 8u; ++group) {
                uint candidate =
                    group * uint(TOP_K) + simd_ranks[group];
                float value = simd_values[candidate];
                int index = simd_indices[candidate];
                if (
                    value > best
                    || (value == best && index < best_index)
                ) {
                    best = value;
                    best_index = index;
                    best_group = group;
                }
            }
            top_values[rank] = best;
            top_indices[rank] = best_index;
            ++simd_ranks[best_group];
        }

        float maximum = top_values[0];
        float total = 0.0f;
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            float probability = exp(top_values[rank] - maximum);
            probabilities[rank] = probability;
            total += probability;
        }
        uint keep = uint(TOP_K);
        float keep_sum = total;
        if (params[1] > 0.0f && params[1] < 1.0f) {
            float cutoff = params[1] * total;
            float cumulative = 0.0f;
            for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
                cumulative += probabilities[rank];
                if (cumulative >= cutoff) {
                    keep = rank + 1u;
                    keep_sum = cumulative;
                    break;
                }
            }
        }
        float uniform = clamp(random[row], 0.0f, 0.99999994f);
        float target = uniform * keep_sum;
        float cumulative = 0.0f;
        int chosen = top_indices[keep - 1u];
        for (uint rank = 0u; rank < keep; ++rank) {
            cumulative += probabilities[rank];
            if (cumulative >= target) {
                chosen = top_indices[rank];
                break;
            }
        }
        output[row] = chosen;
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            uint destination = row * uint(TOP_K) + rank;
            indices_out[destination] = top_indices[rank];
            probabilities_out[destination] = rank < keep
                ? probabilities[rank] / keep_sum
                : 0.0f;
        }
    }
)METAL";

// Exact row-wise top-k selection for the common 65..128 range. Each pass
// sorts independent 1024-value tiles and retains 128 candidates. Repeating
// the pass cannot discard a member of the global top-k, while avoiding the
// full-vocabulary argpartition/argsort graph used by the generic fallback.
constexpr const char* kHierarchicalTopKHeader = R"METAL(
inline void mfq_prune_sorted_candidates(threadgroup float* values,
    threadgroup int* indices, uint tid) {
    for (uint chunk = 128u; chunk <= 512u; chunk <<= 1u) {
        uint pairs = 1024u / (2u * chunk);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint item = tid; item < pairs * 64u; item += 256u) {
            uint pair = item / 64u;
            uint offset = item % 64u;
            uint left = pair * 2u * chunk + chunk + offset;
            uint right = pair * 2u * chunk + chunk + 127u - offset;
            float value = values[left];
            int index = indices[left];
            values[left] = values[right];
            indices[left] = indices[right];
            values[right] = value;
            indices[right] = index;
        }
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint logical = tid; logical < pairs * 256u; logical += 256u) {
                uint other = logical ^ stride;
                if (other <= logical) continue;
                uint pair = logical / 256u;
                uint offset = logical % 256u;
                uint other_offset = other % 256u;
                uint left = pair * 2u * chunk + (offset & 127u) + (offset >= 128u ? chunk : 0u);
                uint right = pair * 2u * chunk + (other_offset & 127u) + (other_offset >= 128u ? chunk : 0u);
                float lv = values[left], rv = values[right];
                int li = indices[left], ri = indices[right];
                if (rv > lv || (rv == lv && ri < li)) {
                    values[left] = rv; values[right] = lv;
                    indices[left] = ri; indices[right] = li;
                }
            }
        }
    }
}
)METAL";

constexpr const char* kHierarchicalTopKFirstSource = R"METAL(
    constexpr uint BLOCK = 1024u;
    constexpr uint KEEP = 128u;
    uint group = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint tid = thread_index_in_threadgroup;
    threadgroup float tile_values[BLOCK];
    threadgroup int tile_indices[BLOCK];

    uint lane = thread_index_in_simdgroup;
    uint warp = simdgroup_index_in_threadgroup;
    float4 values;
    int4 indices;
    for (uint item = 0u; item < 4u; ++item) {
        uint index = group * BLOCK + warp * KEEP + lane * 4u + item;
        float value = index < uint(COUNT) ? float(logits[ulong(row) * uint(COUNT) + index]) : -INFINITY;
        values[item] = isnan(value) ? -INFINITY : value;
        indices[item] = index < uint(COUNT) ? int(index) : INT_MAX;
    }
    for (uint width = 2u; width <= KEEP; width <<= 1u) {
        for (uint stride = width >> 1u; stride > 0u; stride >>= 1u) {
            float4 partners;
            int4 partner_indices;
            for (uint item = 0u; item < 4u; ++item) {
                partners[item] = stride >= 4u ? simd_shuffle_xor(values[item], stride >> 2u) : values[item ^ stride];
                partner_indices[item] = stride >= 4u ? simd_shuffle_xor(indices[item], stride >> 2u) : indices[item ^ stride];
            }
            for (uint item = 0u; item < 4u; ++item) {
                uint logical = lane * 4u + item;
                bool better = ((logical & width) == 0u) == ((logical & stride) == 0u);
                bool partner_before = partners[item] > values[item] ||
                    (partners[item] == values[item] && partner_indices[item] < indices[item]);
                bool own_before = values[item] > partners[item] ||
                    (values[item] == partners[item] && indices[item] < partner_indices[item]);
                if ((better && partner_before) || (!better && own_before)) {
                    values[item] = partners[item];
                    indices[item] = partner_indices[item];
                }
            }
        }
    }
    for (uint item = 0u; item < 4u; ++item) {
        uint destination = warp * KEEP + lane * 4u + item;
        tile_values[destination] = values[item];
        tile_indices[destination] = indices[item];
    }
    mfq_prune_sorted_candidates(tile_values, tile_indices, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < KEEP) {
        ulong output = ulong(row) * ((uint(COUNT) + BLOCK - 1u) / BLOCK) * KEEP + group * KEEP + tid;
        scores_out[output] = tile_values[tid];
        indices_out[output] = tile_indices[tid];
    }
)METAL";

constexpr const char* kHierarchicalTopKMergeSource = R"METAL(
    constexpr uint BLOCK = 1024u;
    constexpr uint KEEP = 128u;
    uint group = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint tid = thread_index_in_threadgroup;
    threadgroup float tile_values[BLOCK];
    threadgroup int tile_indices[BLOCK];

    uint begin = group * BLOCK;
    for (uint offset = tid; offset < BLOCK; offset += 256u) {
        uint index = begin + offset;
        tile_values[offset] = index < uint(COUNT)
            ? scores_in[ulong(row) * uint(COUNT) + index]
            : -INFINITY;
        tile_indices[offset] = index < uint(COUNT)
            ? indices_in[ulong(row) * uint(COUNT) + index]
            : INT_MAX;
    }

    mfq_prune_sorted_candidates(tile_values, tile_indices, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < KEEP) {
        ulong output = ulong(row) * ((uint(COUNT) + BLOCK - 1u) / BLOCK) * KEEP + group * KEEP + tid;
        scores_out[output] = tile_values[tid];
        indices_out[output] = tile_indices[tid];
    }
)METAL";

constexpr const char* kHierarchicalTopKFinalSource = R"METAL(
    constexpr uint BLOCK = 1024u;
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    threadgroup float tile_values[BLOCK];
    threadgroup int tile_indices[BLOCK];

    for (uint index = tid; index < BLOCK; index += 256u) {
        tile_values[index] = index < uint(COUNT)
            ? scores[ulong(row) * uint(COUNT) + index]
            : -INFINITY;
        tile_indices[index] = index < uint(COUNT)
            ? indices[ulong(row) * uint(COUNT) + index]
            : INT_MAX;
    }
    mfq_prune_sorted_candidates(tile_values, tile_indices, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float inverse_temperature = 1.0f / params[0];
        float maximum = tile_values[0] * inverse_temperature;
        float probabilities[TOP_K];
        float total = 0.0f;
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            float probability = exp(
                tile_values[rank] * inverse_temperature - maximum);
            probabilities[rank] = probability;
            total += probability;
        }
        uint keep = uint(TOP_K);
        float keep_sum = total;
        if (params[1] > 0.0f && params[1] < 1.0f) {
            float cutoff = params[1] * total;
            float cumulative = 0.0f;
            for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
                cumulative += probabilities[rank];
                if (cumulative >= cutoff) {
                    keep = rank + 1u;
                    keep_sum = cumulative;
                    break;
                }
            }
        }
        float uniform = clamp(random[row], 0.0f, 0.99999994f);
        float target = uniform * keep_sum;
        float cumulative = 0.0f;
        int chosen = tile_indices[keep - 1u];
        for (uint rank = 0u; rank < keep; ++rank) {
            cumulative += probabilities[rank];
            if (cumulative >= target) {
                chosen = tile_indices[rank];
                break;
            }
        }
        output[row] = chosen;
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            indices_out[ulong(row) * uint(TOP_K) + rank] = tile_indices[rank];
            probabilities_out[ulong(row) * uint(TOP_K) + rank] = rank < keep
                ? probabilities[rank] / keep_sum
                : 0.0f;
        }
    }
)METAL";

constexpr const char* kSortedSource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }
    threadgroup float reduction[256];
    uint offset = row * uint(COUNT);
    float maximum = scores[offset];
    float local_sum = 0.0f;
    for (uint rank = tid; rank < uint(COUNT); rank += 256u) {
        local_sum += exp(scores[offset + rank] - maximum);
    }
    reduction[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            reduction[tid] += reduction[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        float total = reduction[0];
        uint keep = uint(COUNT);
        float keep_sum = total;
        if (params[0] > 0.0f && params[0] < 1.0f) {
            float cutoff = params[0] * total;
            float cumulative = 0.0f;
            for (uint rank = 0u; rank < uint(COUNT); ++rank) {
                cumulative += exp(scores[offset + rank] - maximum);
                if (cumulative >= cutoff) {
                    keep = rank + 1u;
                    keep_sum = cumulative;
                    break;
                }
            }
        }
        float uniform = clamp(random[row], 0.0f, 0.99999994f);
        float target = uniform * keep_sum;
        float cumulative = 0.0f;
        int chosen = indices[offset + keep - 1u];
        for (uint rank = 0u; rank < keep; ++rank) {
            cumulative += exp(scores[offset + rank] - maximum);
            if (cumulative >= target) {
                chosen = indices[offset + rank];
                break;
            }
        }
        output[row] = chosen;
    }
)METAL";

constexpr const char* kTokenCountsSource = R"METAL(
    uint token = thread_position_in_grid.x;
    if (token >= uint(VOCAB)) {
        return;
    }
    int value = counts[token];
    for (uint index = 0u; index < uint(token_ids_shape[0]); ++index) {
        value += int(token_ids[index] == long(token));
    }
    output[token] = value;
)METAL";

constexpr const char* kPenaltiesSource = R"METAL(
    uint index = thread_position_in_grid.x;
    if (index >= uint(logits_shape[0]) * uint(VOCAB)) {
        return;
    }
    uint token = index % uint(VOCAB);
    int count = counts[token];
    float value = float(logits[index]);
    if (count != 0) {
        if (params[2] != 1.0f) {
            value = value < 0.0f
                ? value * params[2]
                : value / params[2];
        }
        value -= params[0] + params[1] * float(count);
    }
    output[index] = T(value);
)METAL";

mlx::core::fast::CustomKernelFunction make_kernel(
    const char* name,
    std::vector<std::string> inputs,
    const char* source,
    bool fast_math = true) {
    CompileOptions options;
    if (fast_math) {
        options.math_mode = MathMode::Fast;
    }
    return mlx::core::fast::metal_kernel(
        name,
        inputs,
        {"output"},
        source,
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& greedy_kernel() {
    static const auto kernel =
        make_kernel("mfq_cpp_sample_greedy", {"logits"}, kGreedySource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& greedy_tiles_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel("mfq_cpp_sample_greedy_tiles", {"logits"},
            {"scores", "indices"}, kGreedyTilesSource, kGreedyReductionHeader, true, false, options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& greedy_tiles_final_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel("mfq_cpp_sample_greedy_tiles_final", {"scores", "indices"},
            {"output"}, kGreedyTilesFinalSource, kGreedyReductionHeader, true, false, options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& softmax_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_sample_softmax",
        {"logits", "random", "params"},
        kSoftmaxSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& top_k_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_sample_top_k_top_p",
            {"logits", "random", "params"},
            {"output", "indices_out", "probabilities_out"},
            kTopKSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
hierarchical_top_k_first_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_sample_top_k_128_first",
            {"logits"},
            {"scores_out", "indices_out"},
            kHierarchicalTopKFirstSource,
            kHierarchicalTopKHeader,
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
hierarchical_top_k_merge_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_sample_top_k_128_merge",
            {"scores_in", "indices_in"},
            {"scores_out", "indices_out"},
            kHierarchicalTopKMergeSource,
            kHierarchicalTopKHeader,
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
hierarchical_top_k_final_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_sample_top_k_128_final",
            {"scores", "indices", "random", "params"},
            {"output", "indices_out", "probabilities_out"},
            kHierarchicalTopKFinalSource,
            kHierarchicalTopKHeader,
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& sorted_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_sample_sorted_top_p",
        {"scores", "indices", "random", "params"},
        kSortedSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& token_counts_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_sample_token_counts_add",
        {"counts", "token_ids"},
        kTokenCountsSource,
        false);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& penalties_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_sample_apply_penalties",
        {"logits", "counts", "params"},
        kPenaltiesSource);
    return kernel;
}

struct LogitsView {
    array values;
    Shape prefix;
    int rows;
    int vocab;
};

int checked_int(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(
                    std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            std::string("sampling ") + name + " exceeds MLX limits");
    }
    return static_cast<int>(value);
}

LogitsView normalize_logits(const array& logits) {
    if (logits.ndim() < 1 || logits.shape(-1) <= 0) {
        throw std::invalid_argument(
            "sampling logits must end in a non-empty vocabulary");
    }
    auto values = logits;
    if (values.dtype() != mlx::core::float16 &&
        values.dtype() != mlx::core::float32 &&
        values.dtype() != mlx::core::bfloat16) {
        values = mlx::core::astype(values, mlx::core::float32, gpu_stream());
    }
    values = mlx::core::contiguous(values, gpu_stream());
    const int vocab = values.shape(-1);
    if (values.size() % static_cast<std::size_t>(vocab) != 0) {
        throw std::invalid_argument("invalid sampling logits shape");
    }
    const int rows = checked_int(
        values.size() / static_cast<std::size_t>(vocab),
        "row count");
    if (rows <= 0) {
        throw std::invalid_argument("sampling logits cannot have zero rows");
    }
    Shape prefix(values.shape().begin(), values.shape().end() - 1);
    values = mlx::core::reshape(values, Shape{rows, vocab}, gpu_stream());
    return {
        std::move(values),
        std::move(prefix),
        rows,
        vocab,
    };
}

array normalize_random(const array& random, int rows) {
    auto values = mlx::core::astype(random, mlx::core::float32, gpu_stream());
    values = mlx::core::contiguous(
        mlx::core::reshape(
            values,
            Shape{checked_int(values.size(), "random value count")}, gpu_stream()), gpu_stream());
    if (values.size() != static_cast<std::size_t>(rows)) {
        throw std::invalid_argument(
            "sampling random values must contain one item per row");
    }
    return values;
}

std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
base_templates(const LogitsView& view) {
    return {
        {"T", view.values.dtype()},
        {"ROWS", view.rows},
        {"VOCAB", view.vocab},
    };
}

MlxTopKDistribution run_direct_top_k_distribution(
    const LogitsView& view,
    const array& random,
    double temperature,
    int top_k,
    double top_p) {
    const array params(
        {
            static_cast<float>(temperature),
            static_cast<float>(top_p),
        },
        mlx::core::float32);
    auto templates = base_templates(view);
    templates.push_back({"TOP_K", top_k});
    auto outputs = top_k_kernel()(
        {view.values, random, params},
        {
            Shape{view.rows},
            Shape{view.rows, top_k},
            Shape{view.rows, top_k},
        },
        {
            mlx::core::int32,
            mlx::core::int32,
            mlx::core::float32,
        },
        {view.rows * kThreads, 1, 1},
        {kThreads, 1, 1},
        std::move(templates),
        std::nullopt,
        false,
        gpu_stream());
    auto sample_shape = view.prefix;
    auto distribution_shape = view.prefix;
    distribution_shape.push_back(top_k);
    return {
        mlx::core::reshape(
            std::move(outputs.at(0)),
            std::move(sample_shape), gpu_stream()),
        mlx::core::reshape(
            std::move(outputs.at(1)),
            distribution_shape, gpu_stream()),
        mlx::core::reshape(
            std::move(outputs.at(2)),
            std::move(distribution_shape), gpu_stream()),
    };
}

array run_sorted(
    const LogitsView& view,
    const array& random,
    double temperature,
    int top_k,
    double top_p) {
    auto scores =
        mlx::core::divide(mlx::core::astype(view.values, mlx::core::float32, gpu_stream()),
        array(static_cast<float>(temperature)), gpu_stream());
    scores = mlx::core::where(
        mlx::core::isnan(scores, gpu_stream()),
        array(-std::numeric_limits<float>::infinity()),
        scores, gpu_stream());
    const int count =
        top_k <= 0 ? view.vocab : std::min(top_k, view.vocab);
    auto order = [&]() {
        if (count >= view.vocab) {
            return mlx::core::flip(
            mlx::core::argsort(scores, -1, gpu_stream()),
            -1, gpu_stream());
        }
        const int start = view.vocab - count;
        auto candidates = mlx::core::slice(
            mlx::core::argpartition(scores, start, -1, gpu_stream()),
            Shape{0, start},
            Shape{view.rows, view.vocab}, gpu_stream());
        candidates = mlx::core::take_along_axis(
            candidates,
            mlx::core::argsort(candidates, -1, gpu_stream()),
            -1, gpu_stream());
        auto candidate_scores = mlx::core::take_along_axis(
            scores,
            candidates,
            -1, gpu_stream());
        auto candidate_order = mlx::core::argsort(
            mlx::core::negative(candidate_scores, gpu_stream()),
            -1, gpu_stream());
        return mlx::core::take_along_axis(
            candidates,
            candidate_order,
            -1, gpu_stream());
    }();
    order = mlx::core::contiguous(
        mlx::core::astype(order, mlx::core::int32, gpu_stream()), gpu_stream());
    auto ordered = mlx::core::contiguous(
        mlx::core::take_along_axis(scores, order, -1, gpu_stream()), gpu_stream());
    const array params(
        {static_cast<float>(top_p)},
        mlx::core::float32);
    auto outputs = sorted_kernel()(
        {ordered, order, random, params},
        {Shape{view.rows}},
        {mlx::core::int32},
        {view.rows * kThreads, 1, 1},
        {kThreads, 1, 1},
        {
            {"ROWS", view.rows},
            {"COUNT", count},
        },
        std::nullopt,
        false,
        gpu_stream());
    return mlx::core::reshape(outputs.front(), view.prefix, gpu_stream());
}

MlxTopKDistribution run_hierarchical_top_k_distribution(
    const LogitsView& view,
    const array& random,
    double temperature,
    int top_k,
    double top_p) {
    if (top_k < 1 ||
        top_k > kHierarchicalTopK) {
        throw std::invalid_argument(
            "hierarchical top-k requires top_k <= 128");
    }
    int count = view.vocab;
    int groups = (count + kTopKBlock - 1) / kTopKBlock;
    auto selected = hierarchical_top_k_first_kernel()(
        {view.values},
        {
            Shape{view.rows, groups * kHierarchicalTopK},
            Shape{view.rows, groups * kHierarchicalTopK},
        },
        {mlx::core::float32, mlx::core::int32},
        {groups * kThreads, view.rows, 1},
        {kThreads, 1, 1},
        {{"COUNT", count}},
        std::nullopt,
        false,
        gpu_stream());
    auto scores = std::move(selected.at(0));
    auto indices = std::move(selected.at(1));
    count = groups * kHierarchicalTopK;

    while (count > kTopKBlock) {
        groups = (count + kTopKBlock - 1) / kTopKBlock;
        selected = hierarchical_top_k_merge_kernel()(
            {scores, indices},
            {
                Shape{view.rows, groups * kHierarchicalTopK},
                Shape{view.rows, groups * kHierarchicalTopK},
            },
            {mlx::core::float32, mlx::core::int32},
            {groups * kThreads, view.rows, 1},
            {kThreads, 1, 1},
            {{"COUNT", count}},
            std::nullopt,
            false,
            gpu_stream());
        scores = std::move(selected.at(0));
        indices = std::move(selected.at(1));
        count = groups * kHierarchicalTopK;
    }

    const array params(
        {
            static_cast<float>(temperature),
            static_cast<float>(top_p),
        },
        mlx::core::float32);
    auto outputs = hierarchical_top_k_final_kernel()(
        {scores, indices, random, params},
        {
            Shape{view.rows},
            Shape{view.rows, top_k},
            Shape{view.rows, top_k},
        },
        {
            mlx::core::int32,
            mlx::core::int32,
            mlx::core::float32,
        },
        {view.rows * kThreads, 1, 1},
        {kThreads, 1, 1},
        {
            {"COUNT", count},
            {"TOP_K", top_k},
        },
        std::nullopt,
        false,
        gpu_stream());
    auto sample_shape = view.prefix;
    auto distribution_shape = view.prefix;
    distribution_shape.push_back(top_k);
    return {
        mlx::core::reshape(
            std::move(outputs.at(0)),
            std::move(sample_shape), gpu_stream()),
        mlx::core::reshape(
            std::move(outputs.at(1)),
            distribution_shape, gpu_stream()),
        mlx::core::reshape(
            std::move(outputs.at(2)),
            std::move(distribution_shape), gpu_stream()),
    };
}

array greedy_array(const array& logits) {
    auto view = normalize_logits(logits);
    if (view.vocab > kGreedyBlock) {
        const int tiles = (view.vocab - 1) / kGreedyBlock + 1;
        auto partials = greedy_tiles_kernel()({view.values},
            {Shape{view.rows, tiles}, Shape{view.rows, tiles}}, {mlx::core::float32, mlx::core::int32},
            {tiles * kThreads, view.rows, 1}, {kThreads, 1, 1},
            {{"T", view.values.dtype()}, {"VOCAB", view.vocab}, {"BLOCK", kGreedyBlock}, {"TILES", tiles}},
            std::nullopt, false, gpu_stream());
        auto outputs = greedy_tiles_final_kernel()(partials, {Shape{view.rows}}, {mlx::core::int32},
            {view.rows * kThreads, 1, 1}, {kThreads, 1, 1}, {{"TILES", tiles}}, std::nullopt, false, gpu_stream());
        return mlx::core::reshape(outputs.front(), std::move(view.prefix), gpu_stream());
    }
    auto outputs = greedy_kernel()(
        {view.values},
        {Shape{view.rows}},
        {mlx::core::int32},
        {view.rows * kThreads, 1, 1},
        {kThreads, 1, 1},
        {{"T", view.values.dtype()}, {"VOCAB", view.vocab}},
        std::nullopt,
        false,
        gpu_stream());
    return mlx::core::reshape(
        outputs.front(),
        std::move(view.prefix), gpu_stream());
}

array softmax_array(
    const array& logits,
    const array& random,
    double temperature) {
    if (!std::isfinite(temperature) || temperature <= 0.0) {
        throw std::invalid_argument(
            "temperature must be finite and positive");
    }
    auto view = normalize_logits(logits);
    auto uniforms = normalize_random(random, view.rows);
    const array params(
        {static_cast<float>(temperature)},
        mlx::core::float32);
    auto outputs = softmax_kernel()(
        {view.values, uniforms, params},
        {Shape{view.rows}},
        {mlx::core::int32},
        {view.rows * kThreads, 1, 1},
        {kThreads, 1, 1},
        base_templates(view),
        std::nullopt,
        false,
        gpu_stream());
    return mlx::core::reshape(
        outputs.front(),
        std::move(view.prefix), gpu_stream());
}

array top_k_array(
    const array& logits,
    const array& random,
    double temperature,
    int top_k,
    double top_p) {
    if (!std::isfinite(temperature) || temperature <= 0.0) {
        throw std::invalid_argument(
            "temperature must be finite and positive");
    }
    auto view = normalize_logits(logits);
    if (top_k < 1 || top_k > std::min(view.vocab, kMaxTopK)) {
        throw std::invalid_argument(
            "top_k must be in [1,min(vocab,1024)]");
    }
    auto uniforms = normalize_random(random, view.rows);
    if (view.rows == 1 && view.vocab > kTopKBlock &&
        top_k <= kHierarchicalTopK) {
        return run_hierarchical_top_k_distribution(
            view,
            uniforms,
            temperature,
            top_k,
            top_p).sampled;
    }
    if (top_k > kDirectTopK) {
        return run_sorted(
            view,
            uniforms,
            temperature,
            top_k,
            top_p);
    }

    return run_direct_top_k_distribution(
        view,
        uniforms,
        temperature,
        top_k,
        top_p).sampled;
}

array counts_array(
    const array& counts,
    const array& tokens) {
    auto current = mlx::core::contiguous(
        mlx::core::reshape(
            mlx::core::astype(counts, mlx::core::int32, gpu_stream()),
            Shape{checked_int(counts.size(), "token count size")}, gpu_stream()), gpu_stream());
    if (current.size() == 0) {
        throw std::invalid_argument("token counts cannot be empty");
    }
    auto token_ids = mlx::core::contiguous(
        mlx::core::reshape(
            tokens,
            Shape{checked_int(tokens.size(), "token id count")}, gpu_stream()), gpu_stream());
    const int vocab = checked_int(current.size(), "vocabulary size");
    auto outputs = token_counts_kernel()(
        {current, token_ids},
        {Shape{vocab}},
        {mlx::core::int32},
        {vocab, 1, 1},
        {std::min(kThreads, vocab), 1, 1},
        {
            {"VOCAB", vocab},
        },
        std::nullopt,
        false,
        gpu_stream());
    return outputs.front();
}

array penalties_array(
    const array& logits,
    const array& counts,
    double presence_penalty,
    double frequency_penalty,
    double repetition_penalty) {
    const double parameters[] = {
        presence_penalty,
        frequency_penalty,
        repetition_penalty,
    };
    if (!std::all_of(
            std::begin(parameters),
            std::end(parameters),
            [](double value) { return std::isfinite(value); })) {
        throw std::invalid_argument(
            "sampling penalties must be finite");
    }
    if (repetition_penalty <= 0.0) {
        throw std::invalid_argument(
            "repetition_penalty must be positive");
    }
    auto view = normalize_logits(logits);
    auto token_counts = mlx::core::contiguous(
        mlx::core::reshape(
            mlx::core::astype(counts, mlx::core::int32, gpu_stream()),
            Shape{checked_int(counts.size(), "penalty count size")}, gpu_stream()), gpu_stream());
    if (token_counts.size() !=
        static_cast<std::size_t>(view.vocab)) {
        throw std::invalid_argument(
            "penalty counts must match the vocabulary size");
    }
    const array params(
        {
            static_cast<float>(presence_penalty),
            static_cast<float>(frequency_penalty),
            static_cast<float>(repetition_penalty),
        },
        mlx::core::float32);
    const int size = checked_int(
        view.values.size(),
        "penalty logits size");
    auto outputs = penalties_kernel()(
        {view.values, token_counts, params},
        {Shape{view.rows, view.vocab}},
        {view.values.dtype()},
        {size, 1, 1},
        {std::min(kThreads, size), 1, 1},
        {
            {"T", view.values.dtype()},
            {"VOCAB", view.vocab},
        },
        std::nullopt,
        false,
        gpu_stream());
    Shape output_shape = std::move(view.prefix);
    output_shape.push_back(view.vocab);
    return mlx::core::reshape(
        outputs.front(),
        std::move(output_shape), gpu_stream());
}


} // namespace
namespace {
const array& checked(const Tensor& value) {
    if (!value.defined() || value.device() != Device{DeviceType::metal, 0} || value.is_encoded() || !value.is_contiguous())
        throw std::invalid_argument("sampling requires a contiguous, unencoded Metal 0 tensor");
    return metal::array(value);
}

const array& checked_logits(const Tensor& logits) {
    const auto& value = checked(logits);
    if (logits.dim() != 2 || logits.size(0) <= 0 || logits.size(1) <= 0 ||
        (logits.scalar_type() != kFloat32 && logits.scalar_type() != kFloat16 && logits.scalar_type() != kBFloat16))
        throw std::invalid_argument("sampling logits must be [rows,vocab] in float32, float16, or bfloat16");
    return value;
}

void checked_counts(const Tensor& counts, std::int64_t vocab) {
    (void)checked(counts);
    if (counts.scalar_type() != kInt32 || counts.dim() != 1 || counts.size(0) != vocab)
        throw std::invalid_argument("sampling counts must be int32 [vocab]");
}
} // namespace

void decode_commit(const Tensor& next, Tensor& generated, Tensor& step, Tensor& input,
                   Tensor& position, Tensor& length) {
    const std::array<const Tensor*, 6> operands{&next, &generated, &step, &input, &position, &length};
    for (const auto* value : operands) {
        (void)checked(*value);
        if (value->device() != next.device() || value->scalar_type() != kInt64)
            throw std::invalid_argument("decode commit requires contiguous int64 tensors on the same device");
    }
    if (next.numel() != 1 || generated.numel() < 1 || step.numel() != 1 ||
        input.numel() != 1 || position.numel() != 1 || length.numel() != 1)
        throw std::invalid_argument("invalid decode commit geometry");
    auto flat = mlx::core::reshape(metal::array(generated), {-1}, gpu_stream());
    auto updated = mlx::core::put_along_axis(flat, mlx::core::reshape(metal::array(step), {1}, gpu_stream()),
        mlx::core::reshape(metal::array(next), {1}, gpu_stream()), 0, gpu_stream());
    updated = mlx::core::reshape(updated, metal::array(generated).shape(), gpu_stream());
    metal::eval(updated);
    const auto write = [](Tensor& destination, const array& value) {
        metal::ops::copy_into(destination, metal::tensor(mlx::core::contiguous(value, gpu_stream())));
    };
    write(generated, updated);
    write(input, metal::array(next));
    const array one(std::int64_t(1), mlx::core::int64);
    write(position, mlx::core::add(metal::array(position), one, gpu_stream()));
    write(length, mlx::core::add(metal::array(length), one, gpu_stream()));
    write(step, mlx::core::add(metal::array(step), one, gpu_stream()));
}

Tensor sample_greedy(const Tensor& logits) {
    return metal::tensor(mlx::core::astype(greedy_array(checked_logits(logits)), mlx::core::int64, gpu_stream()));
}

Tensor sample_stochastic(const Tensor& logits, const Tensor& random, double temperature,
                         std::int64_t top_k, double top_p) {
    const auto& values = checked_logits(logits);
    const auto& uniforms = checked(random);
    if (random.device() != logits.device() || random.scalar_type() != kFloat32 || random.numel() != logits.size(0))
        throw std::invalid_argument("sampling random must be float32 with one value per row on the logit device");
    if (top_k < 0 || top_k > kMaxTopK) throw std::invalid_argument("sampling top_k must be in [0,1024]");
    if (!std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0)
        throw std::invalid_argument("top_p must be in (0,1]");
    if (!std::isfinite(temperature) || temperature <= 0.0)
        throw std::invalid_argument("temperature must be finite and positive");
    auto result = [&] {
        if (top_k > 0) return top_k_array(values, uniforms, temperature, static_cast<int>(top_k), top_p);
        if (top_p >= 1.0) return softmax_array(values, uniforms, temperature);
        auto view = normalize_logits(values);
        auto normalized = normalize_random(uniforms, view.rows);
        return run_sorted(view, normalized, temperature, 0, top_p);
    }();
    return metal::tensor(mlx::core::astype(result, mlx::core::int64, gpu_stream()));
}

void add_token_counts(Tensor& counts, const Tensor& ids) {
    const auto& values = checked(counts);
    const auto& tokens = checked(ids);
    if (counts.dim() != 1 || counts.scalar_type() != kInt32 || ids.scalar_type() != kInt64 || ids.device() != counts.device())
        throw std::invalid_argument("sampling requires int32 counts [vocab] and int64 token IDs on the same device");
    if (!ids.numel() || !counts.numel()) return;
    auto result = metal::tensor(counts_array(values, tokens));
    metal::ops::copy_into(counts, result);
}

void apply_penalties(Tensor& logits, const Tensor& counts, double presence, double frequency, double repetition) {
    const auto& values = checked_logits(logits);
    if (logits.size(0) != 1) throw std::invalid_argument("penalty logits must be [1,vocab]");
    checked_counts(counts, logits.size(1));
    auto result = metal::tensor(penalties_array(values, metal::array(counts), presence, frequency, repetition));
    metal::ops::copy_into(logits, result);
}

std::pair<std::vector<float>, std::vector<std::int64_t>> read_topk(
        const Tensor& logits, std::int64_t count, bool force_compact) {
    (void)force_compact;
    const auto& values = checked_logits(logits);
    if (count < 1 || count > 64 || count > logits.size(1))
        throw std::invalid_argument("top-k count must be in [1,min(vocab,64)]");
    const int rows = checked_int(static_cast<std::size_t>(logits.size(0)), "row count");
    const int k = static_cast<int>(count);
    auto scores = mlx::core::astype(values, mlx::core::float32, gpu_stream());
    auto order = mlx::core::argsort(mlx::core::negative(scores, gpu_stream()), -1, gpu_stream());
    order = mlx::core::slice(order, Shape{0, 0}, Shape{rows, k}, gpu_stream());
    auto selected = mlx::core::contiguous(mlx::core::take_along_axis(scores, order, -1, gpu_stream()), gpu_stream());
    auto indices = mlx::core::contiguous(mlx::core::astype(order, mlx::core::int64, gpu_stream()), gpu_stream());
    metal::eval(std::vector<array>{selected, indices});
    const auto size = static_cast<std::size_t>(rows) * static_cast<std::size_t>(k);
    return {{selected.data<float>(), selected.data<float>() + size},
            {indices.data<std::int64_t>(), indices.data<std::int64_t>() + size}};
}
} // namespace tyloi::metal::ops

namespace tyloi::metal {

mlx::core::array sample_greedy(const mlx::core::array& logits) {
    return ops::greedy_array(logits);
}

mlx::core::array sample_softmax(const mlx::core::array& logits, const mlx::core::array& random,
                                double temperature) {
    return ops::softmax_array(logits, random, temperature);
}

mlx::core::array sample_top_k_top_p(const mlx::core::array& logits, const mlx::core::array& random,
                                    double temperature, int top_k, double top_p) {
    if (!std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0)
        throw std::invalid_argument("top_p must be in (0,1]");
    return ops::top_k_array(logits, random, temperature, top_k, top_p);
}

mlx::core::array sample_nucleus(const mlx::core::array& logits, const mlx::core::array& random,
                                double temperature, double top_p) {
    if (!std::isfinite(temperature) || temperature <= 0.0)
        throw std::invalid_argument("temperature must be finite and positive");
    if (!std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0)
        throw std::invalid_argument("top_p must be in (0,1]");
    auto view = ops::normalize_logits(logits);
    auto uniforms = ops::normalize_random(random, view.rows);
    return ops::run_sorted(view, uniforms, temperature, 0, top_p);
}

MlxTopKDistribution sample_top_k_distribution(const mlx::core::array& logits,
        const mlx::core::array& random, double temperature, int top_k, double top_p) {
    if (!std::isfinite(temperature) || temperature <= 0.0)
        throw std::invalid_argument("temperature must be finite and positive");
    if (!std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0)
        throw std::invalid_argument("top_p must be in (0,1]");
    auto view = ops::normalize_logits(logits);
    if (top_k < 1 || top_k > std::min(view.vocab, ops::kHierarchicalTopK))
        throw std::invalid_argument("compact top_k must be in [1,min(vocab,128)]");
    auto uniforms = ops::normalize_random(random, view.rows);
    auto result = top_k > ops::kDirectTopK
        ? ops::run_hierarchical_top_k_distribution(view, uniforms, temperature, top_k, top_p)
        : ops::run_direct_top_k_distribution(view, uniforms, temperature, top_k, top_p);
    return {std::move(result.sampled), std::move(result.indices), std::move(result.probabilities)};
}

mlx::core::array sample_token_counts_add(const mlx::core::array& counts, const mlx::core::array& tokens) {
    return ops::counts_array(counts, mlx::core::astype(tokens, mlx::core::int32, ops::gpu_stream()));
}

mlx::core::array sample_apply_penalties(const mlx::core::array& logits, const mlx::core::array& counts,
                                        double presence, double frequency, double repetition) {
    return ops::penalties_array(logits, counts, presence, frequency, repetition);
}

} // namespace tyloi::metal

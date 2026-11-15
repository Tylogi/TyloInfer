#include "engine/ops/ple.h"
#include "engine/ops.h"
#include "engine/backends/metal/native/tensor/tensor.h"
#include "engine/backends/metal/ops/common/mlx_linear_attention.h"
#include "backends/metal/models/flash_next/qwen4_ops.h"

#include <mlx/mlx.h>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace tyloi::ops {
namespace mx = mlx::core;

Tensor ple(const Tensor& input, const Tensor& embeddings, const PleWeights& w,
        Tensor& convolution, bool cache, std::int64_t hidden, std::int64_t streams,
        std::int64_t dilation, double eps, std::vector<Tensor>* checkpoints,
        std::size_t checkpoint_offset) {
    if (!input.is_metal() || input.dim() != 3 || !embeddings.is_metal() || embeddings.dim() != 3 ||
        input.size(0) != embeddings.size(0) || input.size(1) != embeddings.size(1) ||
        input.size(0) <= 0 || input.size(1) <= 0 || input.size(0) > std::numeric_limits<int>::max() ||
        input.size(1) > std::numeric_limits<int>::max() ||
        hidden <= 0 || streams <= 0 || hidden > std::numeric_limits<int>::max() / streams ||
        input.size(2) != hidden * streams || dilation <= 0 || dilation > std::numeric_limits<int>::max())
        throw std::invalid_argument("invalid Metal PLE input geometry");
    const auto batch = int(input.size(0)), tokens = int(input.size(1));
    const auto channels = int(hidden * streams);
    auto key_value = project_group(embeddings, std::array<Tensor, 2>{w.key, w.value});
    auto key = mx::reshape(metal::qwen4_grouped_rms_norm(metal::array(key_value[0]),
        metal::array(w.key_norm), int(hidden), float(eps)), {batch, tokens, int(streams), int(hidden)});
    auto query = mx::reshape(metal::qwen4_grouped_rms_norm(metal::array(input),
        metal::array(w.query_norm), int(hidden), float(eps)), {batch, tokens, int(streams), int(hidden)});
    auto score = mx::sum(mx::astype(key, mx::float32) * mx::astype(query, mx::float32), -1) /
        std::sqrt(float(hidden));
    auto sign = mx::astype(mx::less(mx::array(0.0f), score), mx::float32) -
        mx::astype(mx::less(score, mx::array(0.0f)), mx::float32);
    auto root = sign * mx::sqrt(mx::maximum(mx::abs(score), mx::array(1e-6f)));
    auto gated = mx::reshape(mx::expand_dims(mx::sigmoid(root), -1) *
        mx::expand_dims(mx::astype(metal::array(key_value[1]), mx::float32), -2),
        {batch, tokens, channels});
    auto normalized = metal::qwen4_grouped_rms_norm(gated, metal::array(w.conv_norm), int(hidden), float(eps));
    std::optional<mx::array> prior;
    if (cache && convolution) prior = metal::array(convolution);

    if (checkpoints) {
        if (!cache || !prior || prior->ndim() != 3 || checkpoint_offset > checkpoints->size() ||
            std::size_t(tokens - 1) > checkpoints->size() - checkpoint_offset)
            throw std::invalid_argument("Metal PLE prefix history is unavailable");
        const int lookback = prior->shape(1);
        auto joined = mx::concatenate({*prior, normalized}, 1);
        for (int count = 1; count < tokens; ++count) {
            auto saved = mx::contiguous(mx::slice(joined, {0, count, 0},
                {batch, count + lookback, channels}));
            metal::replace(checkpoints->at(checkpoint_offset + std::size_t(count - 1)), std::move(saved));
        }
    }
    auto result = metal::cached_depthwise_conv_silu(normalized, metal::array(w.conv), prior, int(dilation));
    if (cache) {
        if (convolution) metal::replace(convolution, std::move(result.state));
        else convolution = metal::tensor(std::move(result.state));
    }
    return metal::tensor(mx::astype(gated + result.output, metal::array(input).dtype()));
}

} // namespace tyloi::ops

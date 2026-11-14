#pragma once

#include "engine/tensor.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tyloi::ops {
struct PleWeights {
    Tensor key, value, key_norm, query_norm, conv_norm, conv;
};

// Compute PLE from prepared embedding rows and update request-owned convolution history.
Tensor ple(const Tensor& input, const Tensor& embeddings, const PleWeights& weights,
    Tensor& convolution, bool cache, std::int64_t hidden, std::int64_t streams,
    std::int64_t dilation, double eps, std::vector<Tensor>* checkpoints = nullptr,
    std::size_t checkpoint_offset = 0);
}

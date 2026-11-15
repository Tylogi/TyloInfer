#pragma once

#include "engine/tensor.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace tyloi::metal::ops {

void decode_commit(const Tensor& next, Tensor& generated, Tensor& step, Tensor& input, Tensor& position, Tensor& length);
Tensor sample_greedy(const Tensor& logits);
Tensor sample_stochastic(const Tensor& logits, const Tensor& random, double temperature, std::int64_t top_k, double top_p);
void add_token_counts(Tensor& counts, const Tensor& ids);
void apply_penalties(Tensor& logits, const Tensor& counts, double presence, double frequency, double repetition);
std::pair<std::vector<float>, std::vector<std::int64_t>> read_topk(const Tensor& logits, std::int64_t count, bool force_compact);

} // namespace tyloi::metal::ops

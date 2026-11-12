#pragma once

#include "engine/generation/mtp/policy.h"
#include "engine/requests/sampling.h"
#include "engine/tensor.h"

#include <vector>

namespace tyloi::engine::mtp {

std::vector<float> probabilities(const Tensor& logits, const MfqSamplingParams& parameters);
std::vector<CompactDistribution> compact_rows(
    const Tensor& logits, const MfqSamplingParams& parameters, bool force_compact = false);
std::vector<CompactDistribution> compact_rows_from_packed(
    const Tensor& packed, const MfqSamplingParams& parameters);

} // namespace tyloi::engine::mtp

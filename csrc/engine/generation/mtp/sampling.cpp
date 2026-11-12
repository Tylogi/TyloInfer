#include "engine/generation/mtp/sampling.h"
#include "engine/ops.h"

#include <cstdint>
#include <utility>

namespace tyloi::engine::mtp {
namespace {

std::vector<CompactDistribution> distributions(
        const std::pair<std::vector<float>, std::vector<std::int64_t>>& topk,
        const MfqSamplingParams& parameters) {
    const auto count = static_cast<std::size_t>(parameters.top_k);
    std::vector<CompactDistribution> result;
    result.reserve(topk.first.size() / count);
    for (std::size_t offset = 0; offset < topk.first.size(); offset += count) {
        result.push_back(compact_distribution_from_topk(
            topk.first.data() + offset, topk.second.data() + offset,
            parameters.top_k, parameters.temperature, parameters.top_p));
    }
    return result;
}

} // namespace

std::vector<float> probabilities(const Tensor& logits, const MfqSamplingParams& parameters) {
    return distribution(ops::read_floats(logits), parameters.temperature,
                        parameters.top_k, parameters.top_p);
}

std::vector<CompactDistribution> compact_rows(
        const Tensor& logits, const MfqSamplingParams& parameters, bool force_compact) {
    return distributions(ops::read_topk(logits, parameters.top_k, force_compact), parameters);
}

std::vector<CompactDistribution> compact_rows_from_packed(
        const Tensor& packed, const MfqSamplingParams& parameters) {
    return distributions(ops::unpack_topk(packed, parameters.top_k), parameters);
}

} // namespace tyloi::engine::mtp

#include "engine/ops/ffn.h"

#include "engine/backends/metal/ops/ffn/projection.h"
#include "engine/backends/metal/ops/moe.h"
#include "backends/metal/runtime/mlx_packed_expert_cache.h"
#include "backends/metal/runtime/mlx_projection_batch.h"
#include "backends/metal/runtime/mlx_tq_expert_cache.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace tyloi::ops {
namespace {

bool preserves_activation(const Tensor& bank) {
    const auto* expert = bank.representation<metal::ExpertBank>();
    if (!expert) return false;
    const auto* cache = std::get_if<std::shared_ptr<metal::MlxPackedExpertCache>>(&expert->cache());
    return cache && (*cache)->preserves_activation_dtype();
}

template<class Work>
Tensor rows(const Tensor& input, std::int64_t count, Work work) {
    if (count <= 0 || input.size(0) <= count) return work(input);
    std::vector<Tensor> output;
    for (std::int64_t start = 0; start < input.size(0); start += count)
        output.push_back(work(slice(input, 0, start, std::min(start + count, input.size(0)))));
    return cat(output, 0);
}

} // namespace

Tensor Ffn::project(const Tensor& input, const Tensor& weight, std::int64_t projection_rows) {
    if (!input || !weight || input.dim() != 2 || input.size(0) <= 0 || input.size(1) <= 0)
        throw std::invalid_argument("Metal FFN projection requires a nonempty [rows,width] input");
    return rows(input, projection_rows, [&](const Tensor& part) {
        return ::tyloi::ops::project(part, weight);
    });
}

ScalarType Ffn::input_type(const Tensor& bank, ScalarType input) {
    if (!bank) throw std::invalid_argument("Metal FFN input type requires an expert bank");
    return preserves_activation(bank) ? input : kFloat16;
}

} // namespace tyloi::ops

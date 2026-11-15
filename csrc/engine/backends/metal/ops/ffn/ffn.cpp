#include "engine/ops/ffn.h"

#include "engine/backends/metal/ops/ffn/projection.h"
#include "engine/backends/metal/ops/moe.h"
#include "engine/weights/moe/batch.h"
#include "backends/metal/runtime/mlx_packed_expert_cache.h"
#include "backends/metal/runtime/mlx_projection_batch.h"
#include "backends/metal/runtime/mlx_tq_expert_cache.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace tyloi::ops {
namespace {

bool preserve_activation(const Tensor& bank) {
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

Tensor shared_ffn(const FfnWeights& weights, const Tensor& input, std::int64_t count) {
    auto gate = metal::ffn_projection(weights.gate_up[0]);
    auto up = metal::ffn_projection(weights.gate_up[1]);
    auto down = metal::ffn_projection(weights.down);
    metal::MlxProjectionBatch projections({&gate, &up});
    return rows(input, count, [&](const Tensor& part) {
        return metal::tensor(down(projections.swiglu(metal::array(part), static_cast<float>(weights.swiglu_limit))));
    });
}

Tensor qwen_ffn(const FfnWeights& weights, const Tensor& input, const Tensor& bias,
                std::int64_t projection_rows) {
    std::vector<std::int64_t> shape;
    for (std::int64_t i = 0; i < input.dim(); ++i) shape.push_back(input.size(i));
    auto source = reshape(input, {-1, weights.qwen_hidden});
    if (!preserve_activation(weights.gate_up.front())) source = cast(source, kFloat16);
    MoeRouting routing;
    routing.activation = weights.moe_use_sigmoid ? RouterActivation::sigmoid :
        weights.moe_use_sqrt_softplus ? RouterActivation::sqrt_softplus : RouterActivation::softmax;
    routing.normalize = weights.moe_normalize;
    routing.delayed_softmax = weights.moe_delayed_softmax;
    routing.scale = weights.moe_router_scale;
    std::int64_t cache_rows = 0;
    if (weights.qwen_tq) {
        const auto& bank = *weights.gate_up.front().representation<metal::ExpertBank>();
        const auto& cache = *std::get<std::shared_ptr<metal::MlxTqExpertCache>>(bank.cache());
        if (cache.limit_bytes() < cache.maximum()) cache_rows = 1;
    }
    auto output = rows(source, cache_rows, [&](const Tensor& part) {
        auto logits = rows(part, projection_rows, [&](const Tensor& chunk) {
            return project(chunk, weights.moe_router);
        });
        auto selected = topk(logits, weights.moe_top_k, routing, bias);
        tyloi::weights::ExpertBatch batch(weights.gate_up, weights.down, selected.ids);
        auto shared = shared_ffn(*weights.shared, part, projection_rows);
        ExpertSpec spec{weights.moe_shared_gate, weights.moe_latent_in, weights.moe_latent_out,
                        weights.swiglu_limit, weights.moe_layer, true};
        auto experts = routed(part, cast(part, kFloat32), selected.weights, batch, spec);
        return combine_experts(experts, selected.weights, shared);
    });
    return reshape(output, shape);
}

} // namespace

FfnWeights qwen4_ffn_weights(FfnWeights weights) {
    if (!weights.is_moe || !weights.target.is_metal() || weights.qwen_hidden <= 0 ||
        weights.qwen_moe_width <= 0 || weights.moe_top_k <= 0 || !weights.moe_router ||
        weights.moe_router.dim() != 2 || weights.moe_router.size(1) != weights.qwen_hidden ||
        weights.moe_top_k > weights.moe_router.size(0) || !weights.moe_shared_gate ||
        !weights.shared || weights.shared->gate_up.size() != 2 ||
        !weights.shared->gate_up[0] || !weights.shared->gate_up[1] || !weights.shared->down ||
        (weights.gate_up.size() != 1 && weights.gate_up.size() != 2) || !weights.down)
        throw std::invalid_argument("incomplete Metal Qwen4 FFN weights");
    for (const auto& bank : weights.gate_up)
        if (!bank || bank.dim() != 3 || bank.size(0) != weights.moe_router.size(0))
            throw std::invalid_argument("invalid Metal Qwen4 expert bank");
    const auto* gate = weights.gate_up.front().representation<metal::ExpertBank>();
    const auto* up = weights.gate_up.size() == 2
        ? weights.gate_up.back().representation<metal::ExpertBank>() : nullptr;
    const auto* down = weights.down.representation<metal::ExpertBank>();
    if (gate || up || down) {
        if (!gate || !up || !down || gate->role() != metal::ExpertRole::gate ||
            up->role() != metal::ExpertRole::up || down->role() != metal::ExpertRole::down ||
            gate->cache() != up->cache() || gate->cache() != down->cache() ||
            gate->layer() != up->layer() || gate->layer() != down->layer())
            throw std::invalid_argument("incompatible Metal Qwen4 expert cache banks");
        const auto width = weights.moe_latent_in ? weights.moe_latent_in.size(0) : weights.qwen_hidden;
        std::visit([&](const auto& cache) {
            cache->validate_layer(gate->layer(), static_cast<int>(weights.moe_router.size(0)),
                                  static_cast<int>(width), static_cast<int>(weights.qwen_moe_width));
            cache->prepare_layer(gate->layer(), weights.moe_top_k, static_cast<float>(weights.swiglu_limit));
        }, gate->cache());
    }
    weights.qwen_tq = false;
    if (gate) weights.qwen_tq = std::holds_alternative<std::shared_ptr<metal::MlxTqExpertCache>>(gate->cache());
    weights.policy = FfnPolicy::qwen4;
    return weights;
}

Tensor ffn(const FfnWeights& weights, const Tensor& input, const Tensor& ids,
           const Tensor& router_bias, std::int64_t projection_rows) {
    if (!input || !input.is_metal() || input.dim() < 2 || input.size(-1) <= 0)
        throw std::invalid_argument("invalid Metal FFN input");
    if (weights.policy == FfnPolicy::qwen4) {
        if (input.size(-1) != weights.qwen_hidden || ids || router_bias)
            throw std::invalid_argument("invalid Qwen4 FFN input or routing override");
        return qwen_ffn(weights, input, router_bias, projection_rows);
    }
    if (weights.is_moe)
        throw std::invalid_argument("Metal routed FFN requires the Qwen4 policy");
    if (weights.important_neurons)
        return additive_mlp(input, weights.gate_up, weights.down, weights.activation, weights.swiglu_limit,
            weights.important_neurons->gate_up, weights.important_neurons->down,
            weights.important_neurons->activation, weights.important_neurons->swiglu_limit);
    return mlp(input, weights.gate_up, weights.down, weights.activation, weights.swiglu_limit);
}

Tensor batch_ffn(const FfnWeights& weights, const Tensor& input) {
    return ffn(weights, input);
}

} // namespace tyloi::ops

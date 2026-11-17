#include "engine/models/common/moe.h"
#include "engine/models/common/ffn_internal.h"
#include "engine/weights/moe/batch.h"
#include "engine/weights/moe/experts.h"

#include <stdexcept>

namespace tyloi::models {
namespace {

Tensor token_ids_for_row(const Tensor& ids, std::int64_t row) {
    if (!ids) return {};
    auto flat = ids.dim() == 1 ? ids : ops::reshape(ids, {-1});
    return ops::slice(flat, 0, row, row + 1);
}

Tensor routed(const Moe& weights, const Tensor& input, const Tensor& bias,
              const Tensor& token_ids, std::int64_t row_offset,
              std::int64_t row_total) {
    auto input_f32 = ops::measure("moe.input_f32", input.device(), [&] {
        return ops::cast(input, kFloat32);
    });
    auto logits = ops::measure("moe.router", input.device(), [&] {
        return weights.router.is_encoded()
            ? ops::project(input_f32, weights.router)
            : ops::matmul(input_f32, ops::transpose(weights.router, 0, 1));
    });
    auto selected = ops::measure("moe.route", input.device(), [&] {
        if (weights.hash_ids) {
            if (!token_ids)
                throw std::invalid_argument("hash-routed MoE requires token ids");
            return ops::hash_route(logits, weights.hash_ids, token_ids, weights.routing.scale);
        }
        return ops::topk(logits, weights.top_k, weights.routing, bias);
    });

    // Build the route batch before shared-expert work. This keeps expert cache
    // admission and the selected route lifetime in the weight-side object.
    tyloi::weights::ExpertBatch batch(
        weights.gate_up, weights.down, selected.ids, row_offset, row_total,
        weights.activation);

    auto shared = ops::measure("moe.shared", input.device(), [&] {
        return mlp(weights.shared, input);
    });
    shared = ops::cast(ops::contiguous(ops::reshape(shared, {input.size(0), input.size(1)})), kFloat16);

    auto routed_input = input;
    if (weights.latent_in)
        routed_input = ops::project(routed_input, weights.latent_in);
    const ops::ExpertSpec spec{weights.activation, weights.swiglu_limit, weights.layer};
    auto experts = ops::Ffn::routed(routed_input, selected.weights, batch, spec);
    if (weights.latent_out) {
        if (!experts.reduced)
            experts.values = ops::weighted_reduce(experts.values, selected.weights);
        experts.values = ops::project(experts.values, weights.latent_out);
        experts.reduced = true;
    }

    Tensor shared_logits;
    if (weights.shared_gate) {
        shared_logits = ops::measure("moe.shared_gate", input.device(), [&] {
            return ops::project(input_f32, weights.shared_gate);
        });
    }
    return ops::Ffn::combine(experts, selected.weights, shared, shared_logits);
}

} // namespace

Tensor moe(const Moe& weights, const Tensor& input, bool serial_experts,
           const Tensor& ids, const Tensor& router_bias) {
    if (!input || input.dim() != 2 || input.size(0) <= 0 || input.size(1) <= 0)
        throw std::invalid_argument("MoE input must be nonempty [rows,width]");
    if (!weights.router || weights.router.dim() != 2 ||
        weights.router.size(1) != input.size(1) || weights.top_k <= 0 ||
        weights.top_k > weights.router.size(0) || !weights.shared.down ||
        weights.shared.gate_up.empty() || !weights.down || weights.gate_up.empty() ||
        (weights.gate_up.size() != 1 && weights.gate_up.size() != 2))
        throw std::runtime_error("incomplete MoE FFN state");
    if (bool(weights.latent_in) != bool(weights.latent_out))
        throw std::runtime_error("MoE latent projections must be provided as a pair");
    if (weights.hash_ids && !ids)
        throw std::invalid_argument("hash-routed MoE requires token ids");

    const Tensor& bias = router_bias ? router_bias : weights.router_bias;
    if (bias && !weights.hash_ids &&
        !((bias.dim() == 1 && bias.size(0) == weights.router.size(0)) ||
          (bias.dim() == 2 && bias.size(0) == input.size(0) &&
           bias.size(1) == weights.router.size(0))))
        throw std::invalid_argument("MoE router correction bias shape mismatch");

    auto half = input.scalar_type() == kFloat16 ? input :
        ops::measure("ffn.input_cast", input.device(), [&] { return ops::cast(input, kFloat16); });
    half = ops::contiguous(half);
    auto flat_ids = ids ? ops::reshape(ids, {-1}) : Tensor{};

    bool cached = false;
    if (serial_experts && input.size(0) > 1) {
        cached = tyloi::weights::cached(weights.down);
        for (const auto& bank : weights.gate_up)
            cached = cached || tyloi::weights::cached(bank);
    }
    if (!cached)
        return routed(weights, half, bias, flat_ids, 0, input.size(0));

    // Bound cache admission to one token's selected experts while retaining
    // the model-level serial_experts contract used by existing callers.
    std::vector<Tensor> outputs;
    outputs.reserve(input.size(0));
    for (std::int64_t row = 0; row < input.size(0); ++row) {
        auto values = ops::contiguous(ops::slice(half, 0, row, row + 1));
        auto row_bias = bias && bias.dim() == 2 ? ops::slice(bias, 0, row, row + 1) : bias;
        auto row_ids = token_ids_for_row(flat_ids, row);
        outputs.push_back(routed(weights, values, row_bias, row_ids, row, input.size(0)));
    }
    return ops::cat(outputs, 0);
}

Tensor ffn_internal::feed_forward(const Moe& weights, const Tensor& input, bool bf16, bool serial_experts) {
    if (bf16) throw std::runtime_error("BF16 routed FFN is unsupported");
    return moe(weights, input, serial_experts);
}

Tensor full_ffn(const Ffn& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset, bool bf16, bool trace,
    ops::Trace* trace_output, bool serial_experts) {
    return std::visit([&](const auto& value) {
        return ffn_internal::full(value, norm,
            residual, attention_output, epsilon, offset, bf16, trace, trace_output,
            serial_experts);
    }, weights);
}

Tensor linear_ffn(const Ffn& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset, bool serial_experts) {
    return std::visit([&](const auto& value) {
        return ffn_internal::linear(value, norm,
            residual, attention_output, epsilon, offset, serial_experts);
    }, weights);
}

} // namespace tyloi::models

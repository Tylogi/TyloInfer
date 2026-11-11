#include "engine/models/common/moe.h"
#include "engine/models/common/ffn_internal.h"
#include "engine/weights/moe/batch.h"
#include "engine/weights/moe/experts.h"

#include <stdexcept>

namespace tyloi::models {
namespace {

Tensor routed(const Moe& weights, const Tensor& input, const Tensor& bias) {
    auto input_f32 = ops::measure("moe.input_f32", input.device(), [&] {
        return ops::cast(input, kFloat32);
    });
    auto logits = ops::measure("moe.router", input.device(), [&] {
        return weights.router.is_encoded()
            ? ops::project(input_f32, weights.router)
            : ops::matmul(input_f32, ops::transpose(weights.router, 0, 1));
    });
    auto selected = ops::measure("moe.topk", input.device(), [&] {
        return ops::topk(logits, weights.top_k, weights.routing, bias);
    });
    tyloi::weights::ExpertBatch batch(weights.gate_up, weights.down, selected.ids);
    auto shared = ops::measure("moe.shared", input.device(), [&] {
        return mlp(weights.shared, input);
    });
    shared = ops::cast(ops::contiguous(ops::reshape(shared, {input.size(0), input.size(1)})), kFloat16);
    const ops::ExpertSpec spec{weights.shared_gate, weights.latent_in, weights.latent_out,
                               weights.swiglu_limit, weights.layer};
    auto experts = ops::routed(input, input_f32, selected.weights, batch, spec);
    return ops::combine_experts(experts, selected.weights, shared);
}

} // namespace

Tensor moe(const Moe& weights, const Tensor& input, bool serial_experts) {
    if (!input || input.dim() != 2 || input.size(0) <= 0 || input.size(1) <= 0)
        throw std::invalid_argument("MoE input must be nonempty [rows,width]");
    if (!weights.router || weights.router.dim() != 2 ||
        weights.router.size(1) != input.size(1) || weights.top_k <= 0 ||
        weights.top_k > weights.router.size(0) || !weights.shared_gate ||
        !weights.shared.down || weights.shared.gate_up.empty() || !weights.down ||
        (weights.gate_up.size() != 1 && weights.gate_up.size() != 2))
        throw std::runtime_error("incomplete MoE FFN state");
    const auto& bias = weights.router_bias;
    if (bias && !((bias.dim() == 1 && bias.size(0) == weights.router.size(0)) ||
        (bias.dim() == 2 && bias.size(0) == input.size(0) &&
         bias.size(1) == weights.router.size(0))))
        throw std::invalid_argument("MoE router correction bias shape mismatch");
    auto half = input.scalar_type() == kFloat16 ? input :
        ops::measure("ffn.input_cast", input.device(), [&] { return ops::cast(input, kFloat16); });
    half = ops::contiguous(half);
    bool cached = false;
    if (serial_experts && input.size(0) > 1) {
        cached = tyloi::weights::cached(weights.down);
        for (const auto& bank : weights.gate_up)
            cached = cached || tyloi::weights::cached(bank);
    }
    if (!cached) return routed(weights, half, bias);

    // Bound cache admission to one token's selected experts.
    std::vector<Tensor> outputs;
    outputs.reserve(input.size(0));
    for (std::int64_t row = 0; row < input.size(0); ++row) {
        auto values = ops::contiguous(ops::slice(half, 0, row, row + 1));
        auto row_bias = bias && bias.dim() == 2 ? ops::slice(bias, 0, row, row + 1) : bias;
        outputs.push_back(routed(weights, values, row_bias));
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

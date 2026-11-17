#pragma once

#include "engine/ops/ops.h"

namespace tyloi::ops {

// Numerical operations borrow their operands for the call; models retain weights.
struct Ffn {
    Ffn() = delete;

    static Tensor dense(const Tensor& input, std::span<const Tensor> gate_up,
        const Tensor& down, MlpActivation activation, double limit);
    static Tensor bf16(const Tensor& input, std::span<const Tensor> gate_up,
        const Tensor& down, MlpActivation activation, double limit);
    static Tensor additive(const Tensor& input, std::span<const Tensor> gate_up,
        const Tensor& down, MlpActivation activation, double limit,
        std::span<const Tensor> extra_gate_up, const Tensor& extra_down,
        MlpActivation extra_activation, double extra_limit);
    static std::optional<Tensor> residual(const Tensor& input, const Tensor& residual,
        std::span<const Tensor> gate_up, const Tensor& down,
        MlpActivation activation, double limit, bool recurrent = false);

    // Select the supported activation dtype without changing the model's input.
    static ScalarType input_type(const Tensor& bank, ScalarType input);
    // Matrix projection with the supplied calibration row geometry.
    static Tensor project(const Tensor& input, const Tensor& weight,
        std::int64_t projection_rows = 0);
    // Input is [rows,width]; route weights are F32. The batch supplies selected
    // Gate/Up/Down banks; activation/clamp and F32 route accumulation stay fused.
    static RoutedExperts routed(const Tensor& input, const Tensor& route_weights,
        weights::ExpertBatch& batch, const ExpertSpec& spec);
    // Optional F32 [rows,1] logits sigmoid-gate the shared output; absence means addition.
    static Tensor combine(const RoutedExperts& experts,
        const Tensor& route_weights, const Tensor& shared, const Tensor& shared_logits = {});

    // Selected experts plus a scalar-gated shared MLP. Routing is already resolved.
    // Gate projection uses exactly gate_input's precision on either backend.
    // CUDA fusion requires the same F16 source and zero clamp; other combinations fail explicitly.
    static Tensor routed_shared(const Tensor& input, const Tensor& route_weights,
        weights::ExpertBatch& batch, std::span<const Tensor> shared_gate_up,
        const Tensor& shared_down, const Tensor& gate_input, const Tensor& gate_weight,
        const ExpertSpec& spec, std::int64_t projection_rows = 0);
};

} // namespace tyloi::ops

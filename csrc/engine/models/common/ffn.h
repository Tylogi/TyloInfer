#pragma once

#include "engine/models/common/mlp.h"
#include "engine/models/common/moe.h"

#include <variant>

namespace tyloi::models {

using Ffn = std::variant<Mlp, Moe>;

Tensor full_ffn(const Ffn& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset, bool bf16, bool trace,
    ops::Trace* trace_output = nullptr, bool serial_experts = false);

Tensor full_ffn(const Mlp& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset, bool bf16, bool trace,
    ops::Trace* trace_output = nullptr);

Tensor linear_ffn(const Mlp& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset);

Tensor linear_ffn(const Ffn& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset, bool serial_experts = false);

} // namespace tyloi::models

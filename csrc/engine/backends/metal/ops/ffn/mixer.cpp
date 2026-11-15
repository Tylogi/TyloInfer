#include "engine/ops/residual.h"

#include "engine/backends/metal/ops/ffn/projection.h"
#include "engine/backends/metal/native/allocation.h"
#include "engine/backends/metal/ops/gguf/native/mlx_native_gguf.h"
#include "backends/metal/models/flash_next/qwen4_ops.h"

#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>

namespace tyloi::ops {
namespace {

Residual::Result result(metal::MlxQwen4GatedResidualPre values) {
    return {metal::tensor(std::move(values.branch)), metal::tensor(std::move(values.residual)),
        values.injection ? metal::tensor(std::move(*values.injection)) : Tensor{}};
}

} // namespace

struct Residual::Impl {
    Tensor norm;
    metal::MlxLinear down, up;
    std::optional<metal::MlxLinear> injection;
    std::optional<std::array<metal::MlxNativeGgufWeight, 2>> native;
    std::optional<mlx::core::array> native_norm, native_injection;
    ResidualConfig config;

    Impl(ResidualWeights weights, ResidualConfig options)
        : norm(std::move(weights.norm)), down(metal::ffn_projection(weights.down)), up(metal::ffn_projection(weights.up)),
          config(options) {
        if (!config.target.is_metal() || config.hidden <= 0 || config.streams <= 1 ||
            !std::isfinite(config.eps) || config.eps <= 0 || !norm || !norm.is_metal() ||
            norm.dim() != 1 || norm.size(0) != config.hidden * config.streams ||
            down.input_size() != config.hidden * config.streams ||
            up.input_size() != down.output_size() || up.output_size() != down.input_size() ||
            config.final == bool(weights.injection))
            throw std::invalid_argument("invalid Metal gated residual weights");
        if (weights.injection) {
            injection.emplace(metal::ffn_projection(weights.injection));
            if (injection->input_size() != down.input_size() || injection->output_size() != config.streams)
                throw std::invalid_argument("invalid Metal residual injection weights");
        }
        if ((config.native_hyper || (down.affine_weight_ref() && up.affine_weight_ref())) &&
            config.hidden == 2560 && config.streams == 4 &&
            static_cast<float>(config.eps) == 1e-6f) {
            const auto convert = [](const metal::MlxLinear& value) -> std::optional<metal::MlxNativeGgufWeight> {
                if (const auto* dense = value.dense_weight_ref()) return metal::mlx_native_gguf_float(*dense);
                if (const auto* gguf = value.gguf_weight_ref(); gguf && gguf->native_weight_ref())
                    return *gguf->native_weight_ref();
                if (const auto* affine = value.affine_weight_ref(); affine && affine->native_weight_ref())
                    return *affine->native_weight_ref();
                return std::nullopt;
            };
            auto lower = convert(down), upper = convert(up);
            if (lower && upper && (!injection || injection->dense_weight_ref() ||
                    injection->gguf_weight_ref() || injection->affine_weight_ref())) {
                native.emplace(std::array{*lower, *upper});
                native_norm.emplace(mlx::core::astype(metal::array(norm), mlx::core::float32) + mlx::core::array(1.f));
                if (injection) {
                    if (const auto* dense = injection->dense_weight_ref())
                        native_injection.emplace(mlx::core::astype(*dense, mlx::core::float32));
                    else if (const auto* gguf = injection->gguf_weight_ref())
                        native_injection.emplace(gguf->dequantize(mlx::core::float32));
                    else native_injection.emplace(injection->affine_weight_ref()->dequantize(mlx::core::float32));
                }
                native_norm->eval();
                metal::AllocationScope::record(*native_norm);
                if (native_injection) { native_injection->eval(); metal::AllocationScope::record(*native_injection); }
            }
        }
    }

    bool use_native(const mlx::core::array& value) const {
        const auto rows = value.size() / value.shape(-1);
        const bool single = down.affine_weight_ref() || (down.dense_weight_ref() && injection);
        const bool small = injection && injection->dense_weight_ref() && down.dense_weight_ref() &&
            up.dense_weight_ref() && down.output_size() == 320 && rows <= 4;
        return native && !(small || (single && rows == 1));
    }

    metal::MlxQwen4GatedResidualPre pre(const mlx::core::array& value) const {
        if (use_native(value)) {
            auto values = metal::mlx_native_gguf_hyper(value, *native_norm, (*native)[0], (*native)[1],
                native_injection ? &*native_injection : nullptr);
            std::optional<mlx::core::array> gate;
            if (native_injection) {
                auto shape = value.shape(); shape.back() = static_cast<int>(config.streams);
                gate.emplace(mlx::core::reshape(values[1], shape));
                if (const auto* affine = down.affine_weight_ref())
                    gate = mlx::core::astype(*gate, affine->activation_dtype());
            }
            return {std::move(values[0]), value, std::move(gate)};
        }
        return finish(metal::qwen4_gated_residual_pre(value, metal::array(norm), down, up, injection,
            config.hidden, config.streams, config.eps));
    }

    metal::MlxQwen4GatedResidualPre finish(metal::MlxQwen4GatedResidualPre values) const {
        if (native && values.residual.size() > std::size_t(values.residual.shape(-1))) {
            values.branch = mlx::core::astype(values.branch, mlx::core::bfloat16);
            if (values.injection) values.injection = mlx::core::astype(*values.injection, mlx::core::float32);
        }
        return values;
    }
};

Residual::Residual() = default;
Residual::Residual(ResidualWeights weights, ResidualConfig config)
    : impl_(std::make_unique<Impl>(std::move(weights), config)) {}
Residual::~Residual() = default;
Residual::Residual(Residual&&) noexcept = default;
Residual& Residual::operator=(Residual&&) noexcept = default;

Residual::Result Residual::pre(const Tensor& input, ResidualRole) const {
    if (!impl_) throw std::logic_error("gated-residual operator is not initialized");
    return result(impl_->pre(metal::array(input)));
}

Residual::Result Residual::pre_after(const Tensor& input, const Tensor& residual, const Tensor& gate,
                                     ResidualRole) const {
    if (!impl_) throw std::logic_error("gated-residual operator is not initialized");
    if (impl_->use_native(metal::array(residual)))
        return result(impl_->pre(metal::qwen4_gated_residual_post(metal::array(input), metal::array(residual),
            metal::array(gate), impl_->config.streams)));
    return result(impl_->finish(metal::qwen4_gated_residual_pre_after(metal::array(input), metal::array(residual),
        metal::array(gate), metal::array(impl_->norm), impl_->down, impl_->up, impl_->injection,
        impl_->config.hidden, impl_->config.streams, impl_->config.eps)));
}

Tensor Residual::post(const Tensor& input, const Tensor& residual, const Tensor& gate) const {
    if (!impl_) throw std::logic_error("gated-residual operator is not initialized");
    return metal::tensor(metal::qwen4_gated_residual_post(metal::array(input), metal::array(residual),
        metal::array(gate), impl_->config.streams));
}

bool Residual::can_chain(const Tensor& input) const {
    return impl_ && input && input.is_metal() && input.dim() >= 2 &&
        input.size(-1) == impl_->config.hidden * impl_->config.streams;
}

} // namespace tyloi::ops

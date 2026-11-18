#include "engine/models/common/linear.h"
#include "engine/ops/ops.h"

#include <stdexcept>

namespace tyloi::models {

bool LinearAttention::verify_window(std::int64_t batch, std::int64_t tokens) const {
    return !split_in_proj && !dense_out_proj && (!conv_bias || !conv_bias.numel()) &&
        batch == 1 && tokens >= 2 && tokens <= 8 &&
        key_dim == 128 && value_dim == 128 && conv_kernel == 4;
}

LinearOutput LinearAttention::forward(
    const Tensor& input, LinearView state, LinearInputs* captured, const Tensor& ffn_norm) const {
    if (!dense_ab_tail && split_in_proj && !split_dense_zab && !z_proj)
        throw std::runtime_error("linear attention Z projection is missing");
    if (!dense_out_proj && !out_proj)
        throw std::runtime_error("linear attention output projection is missing");
    const auto B = input.size(0), T = input.size(1), H = input.size(2);
    const auto nv = value_heads, dv = value_dim, vsz = nv * dv;
    const auto normalize = [&] {
        ops::ProfileRange range("linear.attn_norm", input.device());
        return ops::rms_norm(input, attn_norm, rms_norm_eps, norm_weight_offset);
    };
    Tensor xn;
    Tensor qkv, qk, value, z, alpha, beta;
    std::array<ops::ScaledProjection, 2> scaled;
    const bool prefill = input.is_cuda() && !split_in_proj && !captured &&
        T > 1 && key_dim == dv && dv > 0 && dv <= 256 && key_heads > 0 && nv % key_heads == 0 &&
        conv_kernel >= 2 && conv_kernel <= 8 && (!conv_bias || !conv_bias.numel());
    const bool readout = input.is_cuda() && !captured && T >= 512 && ffn_norm &&
        input.scalar_type() == kFloat16 && !dense_out_proj;
    if (split_in_proj) {
        xn = normalize();
        auto shared = !split_dense_zab && ab_is_nint
            ? ops::linear_shared(xn, qkv_proj, z_proj, ab_nint_proj) : std::nullopt;
        if (shared) {
            qk = (*shared)[0]; value = (*shared)[1]; z = (*shared)[2];
            if (shared->size() == 5) {
                alpha = (*shared)[3]; beta = (*shared)[4];
            } else {
                ops::ProfileRange range("linear.ab_proj", input.device());
                auto ab = ops::project_group(xn, ab_nint_proj);
                alpha = ab[0]; beta = ab[1];
            }
        } else {
            {
                ops::ProfileRange range("linear.qkv_proj", input.device());
                auto parts = ops::project_group(xn, qkv_proj);
                qk = parts[0]; value = parts[1];
            }
            if (split_dense_zab) {
                ops::ProfileRange range("linear.zab_proj", input.device());
                auto zab = ops::reshape(ops::project(ops::reshape(xn, {B * T, H}), zab_proj),
                                        {B, T, vsz + 2 * nv});
                z = ops::slice(zab, -1, 0, vsz);
                alpha = ops::slice(zab, -1, vsz, vsz + nv);
                beta = ops::slice(zab, -1, vsz + nv, vsz + 2 * nv);
            } else {
                {
                    ops::ProfileRange range("linear.z_proj", input.device());
                    z = ops::project(xn, z_proj);
                }
                ops::ProfileRange range("linear.ab_proj", input.device());
                if (ab_is_nint) {
                    auto ab = ops::project_group(xn, ab_nint_proj);
                    alpha = ab[0]; beta = ab[1];
                } else {
                    auto ab = ops::reshape(ops::project(ops::reshape(xn, {B * T, H}), ab_proj),
                                           {B, T, 2 * nv});
                    alpha = ops::slice(ab, -1, 0, nv);
                    beta = ops::slice(ab, -1, nv, 2 * nv);
                }
            }
        }
    } else {
        auto parts = ops::project_group_rms(input, attn_norm, in_proj,
            rms_norm_eps, norm_weight_offset, xn,
            std::span(scaled).first(prefill && T >= 256 ? (readout ? 2 : 1) : 0));
        if (parts.empty()) {
            xn = normalize();
            ops::ProfileRange range("linear.in_proj", input.device());
            parts = ops::project_group(xn, in_proj);
        }
        if (!scaled[0]) qkv = ops::cast(parts[0], kFloat32);
        z = parts[1];
        if (dense_ab_tail) {
            ops::ProfileRange range("linear.ab_proj", input.device());
            auto ab = ops::reshape(ops::project(ops::reshape(xn, {B * T, H}), ab_proj),
                                   {B, T, 2 * nv});
            alpha = ops::slice(ab, -1, 0, nv);
            beta = ops::slice(ab, -1, nv, 2 * nv);
        } else {
            alpha = parts[2]; beta = parts[3];
        }
    }
    const bool verify = captured && verify_window(B, T);
    if (verify || (!captured && T == 1 && key_dim == value_dim &&
        !split_in_proj && !dense_out_proj && (!conv_bias || !conv_bias.numel()))) {
        auto output = ops::linear_decode(qkv, ops::reshape(z, {B, T, vsz}),
            ops::reshape(alpha, {B, T, nv}), ops::reshape(beta, {B, T, nv}),
            state.convolution, state.recurrent, conv_weight, dt_bias, a_log, linear_norm,
            out_proj, key_heads, nv, dv, rms_norm_eps, state.transposed, tiled_v_heads, verify);
        if (output) {
            if (captured) *captured = {{}, {}, qkv, {}, {}, B, T, false, output->updates};
            return {input, ops::reshape(output->output, {B, T, H}), {}};
        }
    }
    std::array<Tensor, 2> gates;
    std::array<Tensor, 3> qkv_values;
    auto prepared = prefill ? ops::linear_prefill(state.convolution, qkv, scaled[0],
        ops::reshape(alpha, {B, T, nv}), ops::reshape(beta, {B, T, nv}),
        conv_weight, dt_bias, a_log, key_heads, nv, dv, rms_norm_eps) : std::nullopt;
    if (prepared) {
        qkv_values = std::move(prepared->qkv);
        gates = std::move(prepared->gates);
    } else {
        if (scaled[0]) qkv = ops::reshape(ops::materialize(scaled[0]),
                                            {B, T, 2 * key_heads * key_dim + vsz});
        {
            ops::ProfileRange range("linear.gates_fused", input.device());
            gates = ops::linear_gates(ops::reshape(alpha, {B, T, nv}),
                                     ops::reshape(beta, {B, T, nv}), dt_bias, a_log);
        }
        if (captured) *captured = {qk, value, qkv, gates[0], gates[1], B, T, split_in_proj, {}};
        qkv_values = ops::linear_conv(state.convolution, split_in_proj ? qk : qkv,
            value, conv_weight, conv_bias, key_heads, nv, key_dim, dv, rms_norm_eps);
    }
    scaled[0] = {};
    Tensor y;
    {
        ops::ProfileRange range("linear.gdn", input.device());
        y = ops::linear_recur(qkv_values[0], qkv_values[1], qkv_values[2],
            gates[0], gates[1], state.recurrent, state.transposed, tiled_v_heads);
    }
    if (readout) {
        auto result = ops::linear_readout(y, z ? ops::reshape(z, {B, T, vsz}) : Tensor{},
                                          scaled[1], linear_norm, out_proj,
                                          input, ffn_norm, rms_norm_eps, norm_weight_offset);
        if (result) return {std::move((*result)[0]), {}, std::move((*result)[1])};
    }
    if (scaled[1]) z = ops::reshape(ops::materialize(scaled[1]), {B, T, vsz});
    {
        ops::ProfileRange range("linear.out_norm", input.device());
        y = ops::rms_norm(y, linear_norm, rms_norm_eps, 0.0);
    }
    y = ops::reshape(ops::contiguous(ops::transpose(y, 1, 2)), {B, T, vsz});
    Tensor output;
    {
        ops::ProfileRange range(dense_out_proj ? "linear.out_proj_gate_dense" : "linear.out_proj_gate",
                                input.device());
        output = ops::silu_project(y, ops::reshape(z, {B, T, vsz}),
                                   dense_out_proj ? out_proj_dense : out_proj);
    }
    return {input, ops::reshape(output, {B, T, H}), {}};
}

void LinearAttention::replay(LinearView state, const LinearInputs& inputs,
                             std::int64_t retained_tokens) const {
    if (retained_tokens <= 0 || retained_tokens > inputs.tokens || inputs.batch <= 0 ||
        inputs.split != split_in_proj)
        throw std::runtime_error("invalid linear-attention recurrent replay");
    if (inputs.updates.defined()) {
        ops::linear_replay(state.convolution, state.recurrent, inputs.qkv, inputs.updates,
                           retained_tokens, state.transposed);
        return;
    }
    auto first = ops::slice(inputs.split ? inputs.qk : inputs.qkv, 1, 0, retained_tokens);
    auto value = inputs.split ? ops::slice(inputs.value, 1, 0, retained_tokens) : Tensor{};
    auto qkv = ops::linear_conv(state.convolution, first, value, conv_weight, conv_bias,
        key_heads, value_heads, key_dim, value_dim, rms_norm_eps, true);
    ops::linear_recur(qkv[0], qkv[1], qkv[2],
        ops::contiguous(ops::slice(inputs.gate, 2, 0, retained_tokens)),
        ops::contiguous(ops::slice(inputs.beta, 2, 0, retained_tokens)),
        state.recurrent, state.transposed, tiled_v_heads);
}

} // namespace tyloi::models

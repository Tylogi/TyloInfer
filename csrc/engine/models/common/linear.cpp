#include "engine/models/common/linear.h"
#include "engine/ops.h"

#include <stdexcept>

namespace tyloi::models {

std::array<Tensor, 2> LinearAttention::forward(
    const Tensor& input, LinearView state, LinearInputs* captured) const {
    if (!dense_ab_tail && split_in_proj && !split_dense_zab && !z_proj)
        throw std::runtime_error("linear attention Z projection is missing");
    if (!dense_out_proj && !out_proj)
        throw std::runtime_error("linear attention output projection is missing");
    const auto B = input.size(0), T = input.size(1), H = input.size(2);
    const auto nv = value_heads, dv = value_dim, vsz = nv * dv;
    Tensor xn;
    {
        ops::ProfileRange range("linear.attn_norm", input.device());
        xn = ops::rms_norm(input, attn_norm, rms_norm_eps, norm_weight_offset);
    }
    Tensor qkv, qk, value, z, alpha, beta;
    if (split_in_proj) {
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
        std::vector<Tensor> parts;
        {
            ops::ProfileRange range("linear.in_proj", input.device());
            parts = ops::project_group(xn, in_proj);
        }
        qkv = ops::cast(parts[0], kFloat32); z = parts[1];
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
    std::array<Tensor, 2> gates;
    {
        ops::ProfileRange range("linear.gates_fused", input.device());
        gates = ops::linear_gates(ops::reshape(alpha, {B, T, nv}),
                                 ops::reshape(beta, {B, T, nv}), dt_bias, a_log);
    }
    if (captured) *captured = {qk, value, qkv, gates[0], gates[1], B, T, split_in_proj};
    auto qkv_values = ops::linear_conv(state.convolution, split_in_proj ? qk : qkv,
        value, conv_weight, conv_bias, key_heads, nv, key_dim, dv, rms_norm_eps);
    Tensor y;
    {
        ops::ProfileRange range("linear.gdn", input.device());
        y = ops::linear_recur(qkv_values[0], qkv_values[1], qkv_values[2],
            gates[0], gates[1], state.recurrent, state.transposed, tiled_v_heads);
    }
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
    return {input, ops::reshape(output, {B, T, H})};
}

void LinearAttention::replay(LinearView state, const LinearInputs& inputs,
                             std::int64_t retained_tokens) const {
    if (retained_tokens <= 0 || retained_tokens > inputs.tokens || inputs.batch <= 0 ||
        inputs.split != split_in_proj)
        throw std::runtime_error("invalid linear-attention recurrent replay");
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

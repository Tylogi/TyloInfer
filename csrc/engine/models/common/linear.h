#pragma once

#include "engine/tensor.h"

#include <array>
#include <cstdint>
#include <vector>

namespace tyloi::models {

struct LinearInputs {
    Tensor qk, value, qkv, gate, beta;
    std::int64_t batch = 0, tokens = 0;
    bool split = false;
};

struct LinearView {
    Tensor& convolution;
    Tensor& recurrent;
    bool transposed;
};

struct LinearAttention {
    std::int64_t key_heads = 0, value_heads = 0, key_dim = 0, value_dim = 0;
    std::int64_t conv_kernel = 0;
    double rms_norm_eps = 1e-6, norm_weight_offset = 1.0;
    Tensor attn_norm, conv_weight, conv_bias, dt_bias, a_log, linear_norm;
    bool split_in_proj = false, split_dense_zab = false, dense_ab_tail = false;
    bool ab_is_nint = false, dense_out_proj = false, tiled_v_heads = false;
    std::vector<Tensor> in_proj, qkv_proj, ab_nint_proj;
    Tensor ab_proj, zab_proj, z_proj, out_proj, out_proj_dense;

    std::array<Tensor, 2> forward(const Tensor& input, LinearView state,
                                  LinearInputs* captured = nullptr) const;
    void replay(LinearView state, const LinearInputs& inputs, std::int64_t retained_tokens) const;
};

} // namespace tyloi::models

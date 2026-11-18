#include "engine/ops/ops.h"
#include "engine/ops/ffn.h"
#include "engine/backends/metal/native/tensor/tensor.h"
#include "engine/backends/metal/ops/attention/attention.h"
#include "engine/backends/metal/ops/attention/kv.h"
#include "engine/backends/metal/ops/common/mlx_linear_attention.h"
#include "engine/backends/metal/ops/common/mlx_moe_ops.h"
#include "engine/weights/encoding/metal.h"
#include "backends/metal/runtime/mlx_tensor.h"

#include <mlx/mlx.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace tyloi::ops {
namespace mx = mlx::core;
namespace {
using metal::array;
using metal::tensor;

mx::array narrow_last(const mx::array& x, int first, int count) {
    auto start = mx::Shape(x.ndim(), 0), stop = x.shape();
    start.back() = first;
    stop.back() = first + count;
    return mx::slice(x, start, stop);
}
}

static Tensor mlp_impl(const Tensor& input, std::span<const Tensor> weights, const Tensor& down,
                       MlpActivation activation, double limit) {
    if (weights.empty() || (activation != MlpActivation::relu_squared && weights.size() != 2))
        throw std::invalid_argument("invalid MLP projection weights");
    auto gate = project(input, weights[0]);
    Tensor hidden;
    if (activation == MlpActivation::relu_squared) hidden = relu_squared(gate);
    else {
        auto up = project(input, weights[1]);
        if (activation == MlpActivation::gelu_gated) hidden = multiply(gelu(gate), up);
        else {
            if (limit > 0) {
                gate = clamp_max(cast(gate, kFloat32), limit);
                up = clamp(cast(up, kFloat32), -limit, limit);
            }
            hidden = multiply(silu(gate), up);
            if (limit > 0) hidden = cast(hidden, kFloat16);
        }
    }
    return project(hidden, down);
}
Tensor Ffn::dense(const Tensor& input, std::span<const Tensor> weights, const Tensor& down,
                  MlpActivation activation, double limit) {
    return mlp_impl(input, weights, down, activation, limit);
}
Tensor Ffn::bf16(const Tensor& value, std::span<const Tensor> weights, const Tensor& down,
                 MlpActivation activation, double limit) {
    return cast(mlp_impl(value, weights, down, activation, limit), kBFloat16);
}
Tensor Ffn::additive(const Tensor& value, std::span<const Tensor> weights, const Tensor& down,
                     MlpActivation activation, double limit, std::span<const Tensor> extra,
                     const Tensor& extra_down, MlpActivation extra_activation, double extra_limit) {
    return add(Ffn::dense(value, weights, down, activation, limit),
               Ffn::dense(value, extra, extra_down, extra_activation, extra_limit));
}
std::optional<Tensor> Ffn::residual(const Tensor&, const Tensor&, std::span<const Tensor>, const Tensor&,
                                    MlpActivation, double, bool) { return std::nullopt; }
std::optional<std::vector<Tensor>> linear_shared(const Tensor&, std::span<const Tensor>,
                                                  const Tensor&, std::span<const Tensor>) { return std::nullopt; }
Tensor silu_project(const Tensor& input, const Tensor& gate, const Tensor& weight) {
    return project(multiply(input, silu(gate)), weight);
}
Tensor sigmoid_project(const Tensor& input, const Tensor& gate, const Tensor& weight) {
    return project(tensor(array(input) * mx::sigmoid(array(gate))), weight);
}

Tensor axial_rope(const Tensor& input, const std::vector<std::int32_t>& positions, double theta) {
    if (input.dim() != 3 || input.size(0) <= 0 || input.size(2) <= 0 ||
        input.size(2) % 4 != 0 ||
        positions.size() != static_cast<std::size_t>(input.size(0) * 2))
        throw std::invalid_argument("grid-ViT axial RoPE geometry mismatch");
    const auto tokens = input.size(0), width = input.size(2), half = width / 2;
    const auto frequencies = half / 2;
    std::vector<float> cosine(static_cast<std::size_t>(tokens * width));
    std::vector<float> sine(cosine.size());
    for (std::int64_t token = 0; token < tokens; ++token) {
        for (int axis = 0; axis < 2; ++axis) {
            const auto position = static_cast<float>(positions[static_cast<std::size_t>(token * 2 + axis)]);
            for (std::int64_t index = 0; index < frequencies; ++index) {
                const auto inverse = std::pow(static_cast<float>(theta),
                    -static_cast<float>(2 * index) / static_cast<float>(half));
                const auto angle = position * inverse;
                const auto column = axis * frequencies + index;
                for (int repeat = 0; repeat < 2; ++repeat) {
                    const auto offset = static_cast<std::size_t>(token * width + repeat * half + column);
                    cosine[offset] = std::cos(angle);
                    sine[offset] = std::sin(angle);
                }
            }
        }
    }
    const std::array<std::int64_t, 3> shape{tokens, 1, width};
    auto cos = metal::copy_from_host(cosine.data(), shape, kFloat32);
    auto sin = metal::copy_from_host(sine.data(), shape, kFloat32);
    auto source = mx::astype(array(input), mx::float32);
    auto halves = mx::split(source, 2, -1);
    auto rotated = mx::concatenate({-halves.at(1), halves.at(0)}, -1);
    return tensor(mx::contiguous(mx::astype(source * array(cos) + rotated * array(sin), array(input).dtype())));
}

Tensor segmented_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                           const std::vector<std::int32_t>& lengths) {
    if (query.dim() != 3 || key.dim() != 3 || value.dim() != 3 ||
        query.size(0) != key.size(0) || query.size(0) != value.size(0) ||
        query.size(1) != key.size(1) || query.size(1) != value.size(1) ||
        query.size(2) != key.size(2) || query.size(2) != value.size(2) ||
        query.size(0) <= 0 || query.size(1) <= 0 || query.size(2) <= 0)
        throw std::invalid_argument("grid-ViT attention geometry mismatch");
    auto q = mx::astype(array(query), mx::float16);
    auto k = mx::astype(array(key), mx::float16);
    auto v = mx::astype(array(value), mx::float16);
    std::vector<mx::array> outputs;
    outputs.reserve(lengths.size());
    std::int64_t offset = 0;
    for (const auto length : lengths) {
        if (length <= 0 || offset + length > query.size(0))
            throw std::invalid_argument("grid-ViT attention segment is invalid");
        const auto segment = [offset, length](const mx::array& source) {
            auto part = mx::slice(source, {static_cast<int>(offset), 0, 0},
                {static_cast<int>(offset + length), source.shape(1), source.shape(2)});
            return mx::expand_dims(mx::transpose(part, {1, 0, 2}), 0);
        };
        auto attended = metal::scaled_dot_product_attention(segment(q), segment(k), segment(v), false,
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(query.size(2)))));
        outputs.push_back(mx::transpose(mx::squeeze(attended, 0), {1, 0, 2}));
        offset += length;
    }
    if (outputs.empty() || offset != query.size(0))
        throw std::invalid_argument("grid-ViT attention segments do not cover every patch");
    return tensor(mx::contiguous(outputs.size() == 1 ? outputs.front() : mx::concatenate(outputs, 0)));
}

std::array<Tensor, 3> qkv_norm(std::array<Tensor, 3> values, std::array<Tensor, 3> weights,
                               double eps, double offset, ScalarType output_type) {
    for (size_t i = 0; i < values.size(); ++i)
        if (weights[i].defined()) values[i] = rms_norm(values[i], weights[i], eps, offset, output_type);
    return values;
}
Tensor add_residual(const Tensor& residual, const Tensor& value, ResidualMode) {
    return add(residual, cast(value, residual.scalar_type()));
}
Tensor residual_add(const Tensor& residual, const Tensor& value) { return add_residual(residual, value); }
Tensor recurrent_add(const Tensor& residual, const Tensor& value) { return add_residual(residual, value); }
std::array<Tensor, 2> residual_norm(const Tensor& residual, const Tensor& value,
        const Tensor& weight, double eps, double offset, bool bf16) {
    auto sum = add_residual(residual, value);
    return {sum, rms_norm(sum, weight, eps, offset, bf16 ? kBFloat16 : kFloat32)};
}
std::array<Tensor, 2> recurrent_norm(const Tensor& residual, const Tensor& value,
        const Tensor& weight, double eps, double offset) {
    auto sum = add_residual(residual, value);
    return {sum, rms_norm(sum, weight, eps, offset)};
}
std::optional<ResidualNorms> residual_norms(const Tensor&, const Tensor&, const Tensor&,
        const Tensor&, const Tensor&, const Tensor&, double) { return std::nullopt; }

std::array<Tensor, 2> linear_gates(const Tensor& alpha, const Tensor& beta,
                                   const Tensor& bias, const Tensor& a_log) {
    auto result = metal::gated_delta_gates(array(alpha), array(beta), array(bias), -mx::exp(array(a_log)));
    return {tensor(std::move(result.gate)), tensor(std::move(result.beta))};
}
std::optional<LinearDecode> linear_decode(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
        Tensor&, Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
        std::int64_t, std::int64_t, std::int64_t, double, bool, bool, bool) {
    return std::nullopt;
}
std::vector<Tensor> project_group_rms(const Tensor&, const Tensor&, std::span<const Tensor>,
        double, double, Tensor&) {
    return {};
}
void linear_replay(Tensor&, Tensor&, const Tensor&, const Tensor&, std::int64_t, bool) {
    throw std::logic_error("recorded GDN replay requires CUDA");
}
std::array<Tensor, 3> linear_conv(Tensor& state, const Tensor& input, const Tensor& value,
        const Tensor& weight, const Tensor& bias, std::int64_t nk, std::int64_t nv,
        std::int64_t dk, std::int64_t dv, double eps, bool) {
    const int qk_width = static_cast<int>(2 * nk * dk);
    const auto& x = array(input);
    auto qk = value.defined() ? x : narrow_last(x, 0, qk_width);
    auto v = value.defined() ? array(value) : narrow_last(x, qk_width, static_cast<int>(nv * dv));
    std::optional<mx::array> b;
    if (bias.defined() && bias.numel()) b = array(bias);
    auto result = metal::linear_conv_qkv(array(state), qk, v, array(weight),
        static_cast<int>(nk), static_cast<int>(nv), static_cast<int>(dk), static_cast<int>(dv), b,
        static_cast<float>(eps));
    metal::replace(state, std::move(result.state));
    return {tensor(std::move(result.query)), tensor(std::move(result.key)), tensor(std::move(result.value))};
}
Tensor linear_recur(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& gate,
                    const Tensor& beta, Tensor& state, bool transposed, bool tiled) {
    auto result = metal::gated_delta_net(array(q), array(k), array(v), array(gate), array(beta),
                                          array(state), transposed, tiled);
    state = tensor(std::move(result.state));
    return tensor(std::move(result.output));
}


SelectedExperts topk(const Tensor& logits, std::int64_t count, const MoeRouting& routing, const Tensor& bias) {
    std::optional<mx::array> b;
    if (bias.defined()) b = array(bias);
    auto result = metal::moe_topk(array(logits), static_cast<int>(count),
        routing.activation == RouterActivation::sigmoid,
        routing.activation == RouterActivation::sqrt_softplus,
        routing.normalize, routing.delayed_softmax, b, std::nullopt, 1e-20f,
        static_cast<float>(routing.scale), false);
    return {tensor(std::move(result.ids)), tensor(std::move(result.weights))};
}

SelectedExperts hash_route(const Tensor&, const Tensor&, const Tensor&, double) {
    throw std::invalid_argument("Metal hash routing is unsupported");
}

Tensor norm_rope_kv(const AttentionInput&, const state::KvView&, const RopeTable&,
                    const Tensor&, const Tensor&, double, double) { return {}; }
Tensor rope_kv(const AttentionInput&, const state::KvView&, const RopeTable&) { return {}; }
std::pair<Tensor, Tensor> append_kv(state::KvView kv, const Tensor& key, const Tensor& value,
                                    const Tensor&, std::int64_t start, bool) {
    if (kv.paged() || kv.ring) throw std::invalid_argument("Metal attention requires contiguous KV storage");
    if (key.size(0) != 1 || kv.batch != 1)
        throw std::invalid_argument("Metal KV writes currently require a single request");
    if (start < 0 || start + key.size(2) > kv.physical_key.size(2))
        throw std::invalid_argument("Metal KV write exceeds capacity");
    auto next = metal::mlx_kv_append(kv, array(key), array(value), static_cast<int>(start));
    metal::replace(kv.physical_key, std::move(next.first));
    metal::replace(kv.physical_value, std::move(next.second));
    auto visible = metal::mlx_kv_view(kv, static_cast<int>(start + key.size(2)));
    return {tensor(std::move(visible.first)), tensor(std::move(visible.second))};
}
Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& mask, double scale) {
    std::optional<mx::array> m;
    if (mask.defined()) m = array(mask);
    return tensor(metal::scaled_dot_product_attention(array(q), array(k), array(v), false,
                                                       static_cast<float>(scale), m));
}
AttentionResult attention(const AttentionInput& request, const state::KvView& kv, const AttentionConfig& config) {
    if (kv.paged() || kv.ring) throw std::invalid_argument("Metal attention requires contiguous KV storage");
    const auto batch = static_cast<int>(request.query.size(0));
    const auto tokens = static_cast<int>(request.query.size(2));
    const auto length = static_cast<int>(kv.visible_key.size(2));
    auto positions = request.write_positions.device() == request.query.device()
        ? request.write_positions : copy(request.write_positions, request.query.device());
    auto qpos = array(cast(positions, kInt32));
    if (positions.dim() == 1) qpos = mx::reshape(qpos, {1, tokens, 1});
    else if (positions.dim() == 2) qpos = mx::reshape(qpos, {batch, tokens, 1});
    else throw std::invalid_argument("Metal attention write positions must have rank one or two");
    auto kpos = mx::reshape(mx::arange(0, length, 1, mx::int32), {1, 1, length});
    auto allowed = mx::less_equal(kpos, qpos);
    if (config.sliding) {
        if (config.window <= 0) throw std::invalid_argument("invalid sliding attention window");
        allowed = mx::logical_and(allowed, mx::greater_equal(kpos, qpos - mx::array(int(config.window - 1))));
    }
    if (request.sequence_lengths.defined()) {
        auto lengths = request.sequence_lengths.device() == request.query.device()
            ? request.sequence_lengths : copy(request.sequence_lengths, request.query.device());
        allowed = mx::logical_and(allowed, mx::less(kpos, mx::reshape(array(cast(lengths, kInt32)), {batch, 1, 1})));
    }
    if (request.attention_mask.defined()) {
        auto valid = request.attention_mask.device() == request.query.device()
            ? request.attention_mask : copy(request.attention_mask, request.query.device());
        if (valid.dim() != 2 || valid.size(0) != batch || valid.size(1) < length)
            throw std::invalid_argument("Metal attention mask must cover [batch, visible tokens]");
        auto m = slice(valid, 1, 0, length);
        allowed = mx::logical_and(allowed, mx::expand_dims(mx::not_equal(array(m), mx::array(0)), 1));
        allowed = mx::where(mx::any(allowed, -1, true), allowed, mx::array(true));
    }
    auto mask = mx::expand_dims(mx::where(allowed, mx::array(0.0f),
        mx::array(-std::numeric_limits<float>::infinity())), 1);
    mask = mx::astype(mask, array(request.query).dtype());
    return {tensor(metal::scaled_dot_product_attention(array(request.query), array(kv.visible_key),
             array(kv.visible_value), false, static_cast<float>(config.scale), mask)), false};
}
std::int64_t attention_parts(std::int64_t, Device) { return 1; }



} // namespace tyloi::ops

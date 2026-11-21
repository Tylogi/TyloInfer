#include "mlx_hyper_connection.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mfq::metal {
using namespace mlx::core;

namespace {
array sinkhorn(array logits, int iterations) {
    if (logits.shape(-1) == 2 && default_device() == Device::gpu) {
        static const auto kernel = fast::metal_kernel("hyper_connection_sinkhorn2",
            {"logits"}, {"result"}, R"METAL(
            const uint row = thread_position_in_grid.x;
            if (row >= uint(logits_shape[0])) return;
            const auto x = logits + row * 4;
            float a = x[0], b = x[1], c = x[2], d = x[3];
            const auto logaddexp = [](float u, float v) {
                return max(u, v) + metal::log(1.f + metal::exp(-abs(u - v)));
            };
            for (int i = 0; i < ITERATIONS; ++i) {
                float l = logaddexp(a, b), r = logaddexp(c, d);
                a -= l; b -= l; c -= r; d -= r;
                l = logaddexp(a, c); r = logaddexp(b, d);
                a -= l; b -= r; c -= l; d -= r;
            }
            auto out = result + row * 4;
            out[0] = metal::exp(a); out[1] = metal::exp(b);
            out[2] = metal::exp(c); out[3] = metal::exp(d);
            )METAL");
        const auto shape = logits.shape();
        const int rows = shape[0] * shape[1];
        logits = contiguous(reshape(logits, Shape{rows, 4}));
        auto mixed = kernel({logits}, {logits.shape()}, {float32}, {rows, 1, 1},
            {std::min(rows, 128), 1, 1}, {{"ITERATIONS", iterations}},
            std::nullopt, false, {}).front();
        return reshape(mixed, shape);
    }
    for (int iteration = 0; iteration < iterations; ++iteration) {
        logits = logits - logsumexp(logits, -1, true);
        logits = logits - logsumexp(logits, -2, true);
    }
    return exp(logits);
}
}

std::optional<array> mlx_hyper_connection_decode_projection(const array& residual, const array& weight, float epsilon) {
    if (residual.ndim() != 4 || weight.ndim() != 2 || weight.dtype() != float32 ||
        residual.shape(0) <= 0 || residual.shape(1) <= 0 || residual.shape(2) <= 0 ||
        residual.shape(0) > 6 / residual.shape(1) || !std::isfinite(epsilon) || epsilon <= 0 ||
        (residual.dtype() != float16 && residual.dtype() != bfloat16) || default_device() != Device::gpu)
        return std::nullopt;
    const int width = weight.shape(1), outputs = weight.shape(0);
    if (width <= 0 || width % 1024 || width / residual.shape(2) != residual.shape(3) ||
        width % residual.shape(2) || outputs <= 0 || outputs > 128)
        return std::nullopt;
    const int rows = residual.shape(0) * residual.shape(1);
    const int threads = width >= 8192 && rows <= 3 ? (rows == 1 ? 1024 : 512) : 256;
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        return fast::metal_kernel("tyloinfer_hyper_decode_projection", {"residual", "weight", "params"},
            {"output"}, R"METAL(
            const uint row = threadgroup_position_in_grid.x / OUTPUTS;
            const uint projection = threadgroup_position_in_grid.x % OUTPUTS;
            const uint tid = thread_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint group = simdgroup_index_in_threadgroup;
            float product = 0.0f, square = 0.0f;
            for (uint column = tid * 4u; column < uint(WIDTH); column += THREADS * 4u) {
                const auto source = residual + ulong(row) * WIDTH + column;
                const float4 x(source[0], source[1], source[2], source[3]);
                const auto w = reinterpret_cast<device const float4*>(weight + projection * WIDTH)[column / 4u];
                product += dot(x, w);
                square += dot(x, x);
            }
            product = simd_sum(product);
            square = simd_sum(square);
            threadgroup float products[THREADS / 32u], squares[THREADS / 32u];
            if (lane == 0u) { products[group] = product; squares[group] = square; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (group == 0u) {
                product = simd_sum(lane < THREADS / 32u ? products[lane] : 0.0f);
                square = simd_sum(lane < THREADS / 32u ? squares[lane] : 0.0f);
                if (lane == 0u) output[row * OUTPUTS + projection] =
                    product * metal::precise::rsqrt(square / float(WIDTH) + params[0]);
            }
            )METAL", "", true, false, options);
    }();
    return kernel({contiguous(residual), contiguous(weight), array({epsilon})},
        {{residual.shape(0), residual.shape(1), outputs}}, {float32},
        {rows * outputs * threads, 1, 1}, {threads, 1, 1},
        {{"WIDTH", width}, {"OUTPUTS", outputs}, {"THREADS", threads}}, std::nullopt, false, {}).front();
}

MlxHyperConnection::MlxHyperConnection(array weight, array bias,
    MlxHyperConnectionConfig config, std::optional<array> projection_scale)
    : weight_(contiguous(astype(weight, float32))),
      prefill_weight_(contiguous(transpose(weight_))),
      bias_(astype(bias, float32)),
      config_(config),
      projection_scale_(std::move(projection_scale)) {
    const int streams = config_.streams;
    if (streams < 2 || streams > 32 || config_.sinkhorn_iterations < 0 ||
        !std::isfinite(config_.norm_epsilon) || config_.norm_epsilon <= 0 ||
        !std::isfinite(config_.gate_epsilon) || config_.gate_epsilon < 0 ||
        !std::isfinite(config_.pre_scale) || !std::isfinite(config_.post_scale)) {
        throw std::invalid_argument("invalid hyper-connection parameters");
    }
    const int channels = 2 * streams + (config_.sinkhorn_iterations ? streams * streams : 0);
    if (weight_.ndim() != 2 || weight_.shape(1) <= 0 || weight_.shape(1) % streams ||
        weight_.shape(0) != channels || bias_.shape() != Shape{channels} ||
        (projection_scale_ && projection_scale_->shape() != Shape{channels})) {
        throw std::invalid_argument("invalid hyper-connection projection geometry");
    }
    eval(weight_, prefill_weight_);
}

MlxHyperConnectionState MlxHyperConnection::collapse(const array& residual) const {
    const int streams = config_.streams;
    if (residual.ndim() != 4 || residual.shape(0) <= 0 || residual.shape(1) <= 0 ||
        residual.shape(2) != streams ||
        residual.shape(3) != weight_.shape(1) / streams) {
        throw std::invalid_argument("invalid hyper-connection residual geometry");
    }
    const int batch = residual.shape(0);
    const int tokens = residual.shape(1);
    auto mixes = [&] {
        if (auto projected = mlx_hyper_connection_decode_projection(residual, weight_, config_.norm_epsilon))
            return std::move(*projected);
        const auto flat = reshape(astype(residual, float32), {batch, tokens, weight_.shape(1)});
        return matmul(fast::rms_norm(flat, std::nullopt, config_.norm_epsilon), prefill_weight_);
    }();
    if (projection_scale_) mixes = mixes * *projection_scale_;
    mixes = mixes + bias_;
    auto pre = sigmoid(slice(mixes, {0, 0, 0}, {batch, tokens, streams}));
    auto post = sigmoid(slice(mixes, {0, 0, streams}, {batch, tokens, 2 * streams})) *
        array(config_.post_scale);
    if (config_.gate_epsilon != 0) {
        pre = pre + array(config_.gate_epsilon);
        post = post + array(config_.gate_epsilon);
    }
    auto branch = squeeze(matmul(expand_dims(astype(pre, residual.dtype()), -2), residual), -2);
    if (config_.pre_scale != 1.0f) branch = branch * array(config_.pre_scale, branch.dtype());

    std::optional<array> combination;
    if (config_.sinkhorn_iterations) {
        auto log_mix = reshape(slice(mixes, {0, 0, 2 * streams},
            {batch, tokens, 2 * streams + streams * streams}), {batch, tokens, streams, streams});
        combination = sinkhorn(std::move(log_mix), config_.sinkhorn_iterations);
    }
    return {std::move(branch), std::move(post), std::move(combination)};
}

array MlxHyperConnection::expand(const array& residual, const array& branch,
    const MlxHyperConnectionState& state) {
    if (state.combination && residual.ndim() == 4 && residual.size() > 0 &&
        residual.size() <= std::size_t(std::numeric_limits<int>::max()) && default_device() == Device::gpu &&
        residual.dtype() == branch.dtype() && (residual.dtype() == float16 || residual.dtype() == bfloat16) &&
        state.combination->shape() == Shape{residual.shape(0), residual.shape(1), residual.shape(2), residual.shape(2)} &&
        state.post.shape() == Shape{residual.shape(0), residual.shape(1), residual.shape(2)} &&
        branch.shape() == Shape{residual.shape(0), residual.shape(1), residual.shape(3)}) {
        const int streams = residual.shape(2), width = residual.shape(3);
        static const auto kernel = [] {
            CompileOptions options;
            options.math_mode = MathMode::Safe;
            return fast::metal_kernel("tyloinfer_hyper_connection_expand",
                {"residual", "branch", "post", "combination"}, {"output"}, R"MSL(
                const uint i = thread_position_in_grid.x;
                const ulong count = ulong(residual_shape[0]) * residual_shape[1] * STREAMS * WIDTH;
                if (i >= count) return;
                const uint row = i / (STREAMS * WIDTH);
                const uint stream = (i / WIDTH) % STREAMS;
                const uint channel = i % WIDTH;
                float mixed = 0.0f;
                for (uint k = 0; k < STREAMS; ++k) {
                    const T coefficient = T(combination[(ulong(row) * STREAMS + stream) * STREAMS + k]);
                    const uint source = (row * STREAMS + k) * WIDTH + channel;
                    mixed = metal::fma(float(coefficient), float(residual[source]), mixed);
                }
                const T gate = T(post[row * STREAMS + stream]);
                const T delta = T(float(gate) * float(branch[row * WIDTH + channel]));
                output[i] = T(float(T(mixed)) + float(delta));
                )MSL", "", true, false, options);
        }();
        const auto size = static_cast<int>(residual.size());
        return kernel({residual, branch, state.post, *state.combination}, {residual.shape()}, {residual.dtype()},
            {size, 1, 1}, {std::min(size, 256), 1, 1},
            {{"T", residual.dtype()}, {"WIDTH", width}, {"STREAMS", streams}},
            std::nullopt, false, {}).front();
    }
    const auto mixed = state.combination ? matmul(astype(*state.combination, residual.dtype()), residual) : residual;
    return mixed + expand_dims(astype(state.post, branch.dtype()), -1) * expand_dims(branch, -2);
}

}  // namespace mfq::metal

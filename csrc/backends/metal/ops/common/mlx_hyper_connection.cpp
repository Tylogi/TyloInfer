#include "mlx_hyper_connection.h"

#include <algorithm>
#include <cmath>
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

MlxHyperConnection::MlxHyperConnection(array weight, array bias,
    MlxHyperConnectionConfig config, std::optional<array> projection_scale)
    : weight_(contiguous(transpose(astype(weight, float32)))),
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
    if (weight_.ndim() != 2 || weight_.shape(0) <= 0 || weight_.shape(0) % streams ||
        weight_.shape(1) != channels || bias_.shape() != Shape{channels} ||
        (projection_scale_ && projection_scale_->shape() != Shape{channels})) {
        throw std::invalid_argument("invalid hyper-connection projection geometry");
    }
    weight_.eval();
}

MlxHyperConnectionState MlxHyperConnection::collapse(const array& residual) const {
    const int streams = config_.streams;
    if (residual.ndim() != 4 || residual.shape(0) <= 0 || residual.shape(1) <= 0 ||
        residual.shape(2) != streams ||
        residual.shape(3) != weight_.shape(0) / streams) {
        throw std::invalid_argument("invalid hyper-connection residual geometry");
    }
    const int batch = residual.shape(0);
    const int tokens = residual.shape(1);
    const auto flat = reshape(astype(residual, float32), {batch, tokens, weight_.shape(0)});
    auto mixes = matmul(fast::rms_norm(flat, std::nullopt, config_.norm_epsilon), weight_);
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
    const auto mixed = state.combination ? matmul(astype(*state.combination, residual.dtype()), residual) : residual;
    return mixed + expand_dims(astype(state.post, branch.dtype()), -1) * expand_dims(branch, -2);
}

}  // namespace mfq::metal

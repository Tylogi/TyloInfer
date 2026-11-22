#include "mlx_hyper_connection.h"
#include "mlx_transformer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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

struct DecodeGateEpilogue {
    const array& scale;
    const array& bias;
    float post_scale;
    float gate_epsilon;
};

struct DecodeResidualUpdate {
    const array& branch;
    const array& post;
    const MlxRmsNorm* norm;
};

struct DecodeProjection {
    array values;
    std::optional<array> residual;
};

std::optional<DecodeProjection> decode_projection(const array& residual, const array& weight, float epsilon,
    const DecodeGateEpilogue* epilogue, const DecodeResidualUpdate* update = nullptr) {
    if (residual.ndim() != 4 || weight.ndim() != 2 || weight.dtype() != float32 ||
        residual.shape(0) <= 0 || residual.shape(1) <= 0 || residual.shape(2) <= 0 ||
        residual.shape(0) > (epilogue ? 8 : 6) / residual.shape(1) || !std::isfinite(epsilon) || epsilon <= 0 ||
        (residual.dtype() != float16 && residual.dtype() != bfloat16) || default_device() != Device::gpu)
        return std::nullopt;
    const int width = weight.shape(1), outputs = weight.shape(0);
    if (width <= 0 || width % 1024 || width / residual.shape(2) != residual.shape(3) ||
        width % residual.shape(2) || outputs <= 0 || outputs > 128)
        return std::nullopt;
    const int rows = residual.shape(0) * residual.shape(1);
    const int threads = width >= 8192 && rows <= 3 ? (rows == 1 ? 1024 : 512) : 256;
    constexpr const char* source = R"METAL(
            const uint row = threadgroup_position_in_grid.x / OUTPUTS;
            const uint projection = threadgroup_position_in_grid.x % OUTPUTS;
            const uint tid = thread_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint group = simdgroup_index_in_threadgroup;
#if TYLOINFER_HC_UPDATE_NORM
            threadgroup bfloat normalized[3072];
            threadgroup float branch_squares[32];
            threadgroup float inverse;
            const uint first = tid * 4u;
            float4 branch_values(0.0f);
            float sum = 0.0f;
            if (first < 3072u) {
                branch_values = float4(*reinterpret_cast<device const bfloat4*>(branch + first));
                #pragma clang loop unroll(full)
                for (uint i = 0u; i < 4u; ++i) sum += branch_values[i] * branch_values[i];
            }
            sum = simd_sum(sum);
            if (lane == 0u) branch_squares[group] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (group == 0u) {
                const float total = simd_sum(branch_squares[lane]);
                if (lane == 0u) inverse = precise::rsqrt(total / 3072.0f + branch_eps[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (first < 3072u) {
                #pragma clang loop unroll(full)
                for (uint i = 0u; i < 4u; ++i)
                    normalized[first + i] = bfloat((branch_values[i] * inverse) * branch_weight[first + i]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
            float product = 0.0f, square = 0.0f;
            for (uint column = tid * 4u; column < uint(WIDTH); column += THREADS * 4u) {
                const auto source = residual + ulong(row) * WIDTH + column;
#if TYLOINFER_HC_UPDATE_RESIDUAL
                const uint stream = column / 3072u;
                const uint channel = column % 3072u;
                const float coefficient = float(bfloat(post[row * 4u + stream]));
#if TYLOINFER_HC_UPDATE_NORM
                const auto delta = float4(*reinterpret_cast<threadgroup const bfloat4*>(normalized + channel));
#else
                const auto delta = float4(*reinterpret_cast<device const bfloat4*>(branch + row * 3072u + channel));
#endif
                const auto prior = float4(*reinterpret_cast<device const bfloat4*>(source));
                // Preserve the BF16 rounding between multiplication and addition.
                const auto next = bfloat4(prior + float4(bfloat4(delta * coefficient)));
                if (projection == 0u)
                    *reinterpret_cast<device bfloat4*>(updated + ulong(row) * WIDTH + column) = next;
                const float4 x(float(next.x), float(next.y), float(next.z), float(next.w));
#else
                const float4 x(source[0], source[1], source[2], source[3]);
#endif
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
                if (lane == 0u) {
                    float value = product * metal::precise::rsqrt(square / float(WIDTH) + params[0]);
                    if constexpr (GATES) {
#pragma clang fp contract(off)
                        value = value * scales[projection];
                        value = value + bias[projection];
                        value = 1.f / (1.f + metal::exp(-value));
                        if (projection >= OUTPUTS / 2u) value *= params[1];
                        if (params[2] != 0.f) value += params[2];
                    }
                    output[row * OUTPUTS + projection] = value;
                }
            }
            )METAL";
    const auto make_kernel = [source](bool update_residual, bool update_norm) {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        std::vector<std::string> inputs{"residual", "weight", "params", "scales", "bias"};
        std::vector<std::string> outputs{"output"};
        if (update_residual) {
            inputs.insert(inputs.end(), {"branch", "post"});
            outputs.push_back("updated");
        }
        if (update_norm) inputs.insert(inputs.end(), {"branch_weight", "branch_eps"});
        std::string defines = update_residual ? "#define TYLOINFER_HC_UPDATE_RESIDUAL 1\n"
                                              : "#define TYLOINFER_HC_UPDATE_RESIDUAL 0\n";
        defines += update_norm ? "#define TYLOINFER_HC_UPDATE_NORM 1\n" : "#define TYLOINFER_HC_UPDATE_NORM 0\n";
        const char* name = update_norm ? "tyloinfer_hyper_norm_expand_collapse"
            : update_residual ? "tyloinfer_hyper_expand_collapse" : "tyloinfer_hyper_decode_projection";
        return fast::metal_kernel(name, inputs, outputs, source, defines, true, false, options);
    };
    static const auto projection_kernel = make_kernel(false, false);
    static const auto residual_kernel = make_kernel(true, false);
    static const auto normalized_kernel = make_kernel(true, true);
    std::vector<array> inputs{contiguous(residual), contiguous(weight),
        array({epsilon, epilogue ? epilogue->post_scale : 1.f, epilogue ? epilogue->gate_epsilon : 0.f}),
        epilogue ? epilogue->scale : weight, epilogue ? epilogue->bias : weight};
    std::vector<Shape> shapes{{residual.shape(0), residual.shape(1), outputs}};
    std::vector<Dtype> dtypes{float32};
    if (update) {
        inputs.insert(inputs.end(), {contiguous(update->branch), contiguous(update->post)});
        shapes.push_back(residual.shape());
        dtypes.push_back(residual.dtype());
    }
    const auto* norm = update ? update->norm : nullptr;
    if (norm) inputs.insert(inputs.end(), {norm->weight(), array({norm->eps()}, float32)});
    const auto& kernel = norm ? normalized_kernel : update ? residual_kernel : projection_kernel;
    auto result = kernel(inputs, shapes, dtypes,
        {rows * outputs * threads, 1, 1}, {threads, 1, 1},
        {{"WIDTH", width}, {"OUTPUTS", outputs}, {"THREADS", threads}, {"GATES", epilogue != nullptr}},
        std::nullopt, false, {});
    std::optional<array> updated;
    if (update) updated = std::move(result[1]);
    return DecodeProjection{std::move(result[0]), std::move(updated)};
}

array decode_collapse_norm(const array& residual, const array& pre, const MlxRmsNorm& norm) {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        return fast::metal_kernel("hy4_ihc_collapse_norm_exact", {"residual", "pre", "weights", "eps"}, {"output"}, R"METAL(
        {
        #pragma clang fp reassociate(off)
            const uint row = threadgroup_position_in_grid.x;
            const uint tid = thread_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint group = simdgroup_index_in_threadgroup;
            const uint column = tid * 4u;
            float4 partial(0.0f);
            #pragma clang loop unroll(full)
            for (uint stream = 0u; stream < 4u; ++stream) {
                const auto v = float4(*reinterpret_cast<device const bfloat4*>(
                    residual + (row * 4u + stream) * 3072u + column));
                partial = fma(v, float4(float(bfloat(pre[row * 4u + stream]))), partial);
            }
            // Round the collapsed stream before calculating its RMS.
            const auto value = float4(bfloat4(partial));
            float squares = 0.0f;
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 4u; ++i) squares += value[i] * value[i];
            squares = simd_sum(squares);
            threadgroup float groups[24];
            threadgroup float inverse;
            if (lane == 0u) groups[group] = squares;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (group == 0u) {
                const float total = simd_sum(lane < 24u ? groups[lane] : 0.0f);
                if (lane == 0u) inverse = precise::rsqrt(total / float(residual_shape[3]) + eps[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 4u; ++i)
                output[row * 3072u + column + i] = bfloat((value[i] * inverse) * weights[column + i]);
        }
        )METAL", "", true, false, options);
    }();
    const int batch = residual.shape(0), tokens = residual.shape(1);
    return kernel({contiguous(residual), contiguous(pre), norm.weight(), array({norm.eps()}, float32)},
        {{batch, tokens, 3072}}, {bfloat16}, {batch * tokens * 768, 1, 1}, {768, 1, 1},
        {}, std::nullopt, false, {}).front();
}

bool compact_hy4_residual(const array& residual, bool packed_decode) {
    return residual.ndim() == 4 && residual.shape(0) > 0 && residual.shape(1) > 0 &&
        residual.shape(0) <= (packed_decode ? 8 : 1) / residual.shape(1) &&
        residual.shape(2) == 4 && residual.shape(3) == 3072 && residual.dtype() == bfloat16;
}

array collapse_branch(const array& residual, const array& pre, float scale, const MlxRmsNorm* branch_norm,
    bool packed_decode) {
    if (branch_norm && compact_hy4_residual(residual, packed_decode) &&
        pre.shape() == Shape{residual.shape(0), residual.shape(1), 4} &&
        branch_norm->width() == 3072 && scale == 1.f && default_device() == Device::gpu)
        return decode_collapse_norm(residual, pre, *branch_norm);
    auto branch = squeeze(matmul(expand_dims(astype(pre, residual.dtype()), -2), residual), -2);
    if (scale != 1.f) branch = branch * array(scale, branch.dtype());
    return branch_norm ? (*branch_norm)(branch) : branch;
}
}  // namespace

std::optional<array> mlx_hyper_connection_decode_projection(const array& residual, const array& weight, float epsilon) {
    if (auto projected = decode_projection(residual, weight, epsilon, nullptr)) return std::move(projected->values);
    return std::nullopt;
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

MlxHyperConnectionState MlxHyperConnection::collapse(const array& residual, const MlxRmsNorm* branch_norm,
    bool packed_decode) const {
    const int streams = config_.streams;
    if (residual.ndim() != 4 || residual.shape(0) <= 0 || residual.shape(1) <= 0 ||
        residual.shape(2) != streams ||
        residual.shape(3) != weight_.shape(1) / streams) {
        throw std::invalid_argument("invalid hyper-connection residual geometry");
    }
    const int batch = residual.shape(0);
    const int tokens = residual.shape(1);
    std::optional<DecodeGateEpilogue> epilogue;
    if (!config_.sinkhorn_iterations && projection_scale_ && compact_hy4_residual(residual, packed_decode)) {
        epilogue.emplace(*projection_scale_, bias_, config_.post_scale, config_.gate_epsilon);
    }
    auto projected = decode_projection(residual, weight_, config_.norm_epsilon, epilogue ? &*epilogue : nullptr);
    const bool fused_gates = projected && epilogue;
    auto mixes = [&] {
        if (projected) return std::move(projected->values);
        const auto flat = reshape(astype(residual, float32), {batch, tokens, weight_.shape(1)});
        return matmul(fast::rms_norm(flat, std::nullopt, config_.norm_epsilon), prefill_weight_);
    }();
    if (!fused_gates) {
        if (projection_scale_) mixes = mixes * *projection_scale_;
        mixes = mixes + bias_;
    }
    auto pre = slice(mixes, {0, 0, 0}, {batch, tokens, streams});
    auto post = slice(mixes, {0, 0, streams}, {batch, tokens, 2 * streams});
    if (!fused_gates) {
        pre = sigmoid(pre);
        post = sigmoid(post) * array(config_.post_scale);
        if (config_.gate_epsilon != 0) {
            pre = pre + array(config_.gate_epsilon);
            post = post + array(config_.gate_epsilon);
        }
    }
    auto branch = collapse_branch(residual, pre, config_.pre_scale, branch_norm, packed_decode);

    std::optional<array> combination;
    if (config_.sinkhorn_iterations) {
        auto log_mix = reshape(slice(mixes, {0, 0, 2 * streams},
            {batch, tokens, 2 * streams + streams * streams}), {batch, tokens, streams, streams});
        combination = sinkhorn(std::move(log_mix), config_.sinkhorn_iterations);
    }
    return {std::move(branch), std::move(post), std::move(combination)};
}

std::pair<array, MlxHyperConnectionState> MlxHyperConnection::expand_and_collapse(
    const array& residual, const array& branch, const MlxHyperConnectionState& state,
    const MlxRmsNorm* branch_norm, bool packed_decode, const MlxRmsNorm* update_norm) const {
    if (!state.combination && !config_.sinkhorn_iterations && projection_scale_ &&
        config_.streams == 4 && weight_.shape(1) == 12288 && compact_hy4_residual(residual, packed_decode) &&
        branch.shape() == Shape{residual.shape(0), residual.shape(1), 3072} &&
        state.post.shape() == Shape{residual.shape(0), residual.shape(1), 4} && branch.dtype() == bfloat16 &&
        (!update_norm || (residual.shape(0) == 1 && residual.shape(1) == 1 && update_norm->width() == 3072))) {
        const DecodeGateEpilogue epilogue{*projection_scale_, bias_, config_.post_scale, config_.gate_epsilon};
        const DecodeResidualUpdate update{branch, state.post, update_norm};
        if (auto projected = decode_projection(residual, weight_, config_.norm_epsilon, &epilogue, &update)) {
            auto updated = std::move(*projected->residual);
            const int batch = residual.shape(0), tokens = residual.shape(1);
            const auto pre = slice(projected->values, {0, 0, 0}, {batch, tokens, 4});
            auto post = slice(projected->values, {0, 0, 4}, {batch, tokens, 8});
            auto collapsed = collapse_branch(updated, pre, config_.pre_scale, branch_norm, packed_decode);
            return {std::move(updated), {std::move(collapsed), std::move(post), std::nullopt}};
        }
    }
    auto updated = expand(residual, update_norm ? (*update_norm)(branch) : branch, state);
    auto collapsed = collapse(updated, branch_norm, packed_decode);
    return {std::move(updated), std::move(collapsed)};
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

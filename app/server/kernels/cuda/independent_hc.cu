#include "app/server/kernels/cuda/independent_hc.h"

#include <climits>
#include <cmath>
#include <type_traits>

namespace mfq::cuda {
namespace {

namespace tb = mfq_tensor_backend;
constexpr int kThreads = 256;
constexpr int kTileWidth = 1024;

template<class T> __device__ float load(T value) { return static_cast<float>(value); }
template<> __device__ float load(half value) { return __half2float(value); }
template<> __device__ float load(__nv_bfloat16 value) { return __bfloat162float(value); }

template<class T> __device__ T rounded(float value) { return static_cast<T>(value); }
template<> __device__ half rounded(float value) { return __float2half_rn(value); }
template<> __device__ __nv_bfloat16 rounded(float value) { return __float2bfloat16_rn(value); }

__device__ float warp_sum(float value) {
    for (int offset = 16; offset; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

// Split the wide, few-output projection across SMs even for a single token.
// One warp also accumulates the squared norm while reading the same input.
template<class T> __global__ void projection_partials(
    const T* streams, const float* function, float* partials, int width, int gates, int tiles) {
    const int row = blockIdx.x / tiles;
    const int first = (blockIdx.x % tiles) * kTileWidth;
    const int end = first + min(kTileWidth, width - first);
    const int lane = threadIdx.x % 32;
    const T* input = streams + int64_t(row) * width;
    float* output = partials + int64_t(blockIdx.x) * (gates + 1);
    for (int gate = threadIdx.x / 32; gate < gates; gate += kThreads / 32) {
        const float* weight = function + int64_t(gate) * width;
        float dot = 0;
        float squares = 0;
        for (int column = first + lane; column < end; column += 32) {
            const float value = load(input[column]);
            dot = fmaf(value, weight[column], dot);
            if (gate == 0) squares = fmaf(value, value, squares);
        }
        dot = warp_sum(dot);
        if (gate == 0) squares = warp_sum(squares);
        if (lane == 0) {
            output[gate] = dot;
            if (gate == 0) output[gates] = squares;
        }
    }
}

__device__ void normalize_log_pair(float& a, float& b) {
    const float delta = a - b;
    const float normalizer = -log1pf(expf(-fabsf(delta)));
    a = normalizer + fminf(0.0f, delta);
    b = normalizer - fmaxf(0.0f, delta);
}

template<class T, bool Matrix = false> __global__ void prepare_branch(
    const T* streams, const float* partials, const float* scale, const float* bias,
    T* branch, float* post, int width, int count, int hidden, int tiles,
    float rms_epsilon, float gate_epsilon, float post_magnitude,
    float pre_scale = 1.0f, float* combination = nullptr, int iterations = 0) {
    extern __shared__ float pre[];
    __shared__ float inverse_rms;
    const int row = blockIdx.x;
    const int sigmoid_gates = post ? 2 * count : count;
    const int gates = sigmoid_gates + (Matrix && combination ? 4 : 0);
    const float* row_partials = partials + int64_t(row) * tiles * (gates + 1);
    if (threadIdx.x == 0) {
        float squares = 0;
        for (int tile = 0; tile < tiles; ++tile)
            squares += row_partials[int64_t(tile) * (gates + 1) + gates];
        inverse_rms = rsqrtf(squares / float(width) + rms_epsilon);
    }
    __syncthreads();
    for (int gate = threadIdx.x; gate < sigmoid_gates; gate += blockDim.x) {
        float dot = 0;
        for (int tile = 0; tile < tiles; ++tile)
            dot += row_partials[int64_t(tile) * (gates + 1) + gate];
        const bool is_post = gate >= count;
        float logit = dot * inverse_rms;
        if constexpr (Matrix) logit += bias[gate];
        else logit = fmaf(logit, scale[is_post], bias[gate]);
        const float sigmoid = 1.0f / (1.0f + expf(-logit));
        if (is_post) post[int64_t(row) * count + gate - count] = fmaf(post_magnitude, sigmoid, gate_epsilon);
        else pre[gate] = sigmoid + gate_epsilon;
    }
    if constexpr (Matrix) {
        if (combination && threadIdx.x == 0) {
            float logits[4]{};
            for (int tile = 0; tile < tiles; ++tile)
                for (int weight = 0; weight < 4; ++weight)
                    logits[weight] += row_partials[int64_t(tile) * (gates + 1) + sigmoid_gates + weight];
            for (int weight = 0; weight < 4; ++weight)
                logits[weight] = fmaf(logits[weight], inverse_rms, bias[sigmoid_gates + weight]);
            for (int iteration = 0; iteration < iterations; ++iteration) {
                normalize_log_pair(logits[0], logits[1]);
                normalize_log_pair(logits[2], logits[3]);
                normalize_log_pair(logits[0], logits[2]);
                normalize_log_pair(logits[1], logits[3]);
            }
            for (int weight = 0; weight < 4; ++weight)
                combination[int64_t(row) * 4 + weight] = expf(logits[weight]);
        }
    }
    __syncthreads();
    const T* input = streams + int64_t(row) * width;
    for (int feature = threadIdx.x; feature < hidden; feature += blockDim.x) {
        float sum = 0;
        for (int stream = 0; stream < count; ++stream)
            sum = fmaf(pre[stream], load(input[stream * hidden + feature]), sum);
        branch[int64_t(row) * hidden + feature] = rounded<T>(sum * pre_scale);
    }
}

template<class T, bool Matrix = false> __global__ void merge_streams(
    const T* streams, const T* branch, const float* post, T* output,
    int64_t elements, int count, int hidden, const float* combination = nullptr) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const int64_t stream = index / hidden;
    const auto row = stream / count;
    const auto feature = index % hidden;
    const float value = load(branch[row * hidden + feature]);
    float residual = load(streams[index]);
    if constexpr (Matrix) {
        const auto* coeff = combination + stream * 2;
        const auto residual_base = row * 2 * hidden + feature;
        residual = fmaf(coeff[0], load(streams[residual_base]), coeff[1] * load(streams[residual_base + hidden]));
    }
    output[index] = rounded<T>(fmaf(post[stream], value, residual));
}

bool floating_activation(tb::ScalarType type) {
    return type == tb::kFloat16 || type == tb::kBFloat16 || type == tb::kFloat32;
}

bool float_parameter(const tb::Tensor& parameter, const tb::Tensor& streams) {
    return parameter.is_cuda() && parameter.is_contiguous() && parameter.device() == streams.device() &&
           parameter.scalar_type() == tb::kFloat32;
}

bool valid_streams(const tb::Tensor& streams) {
    return streams.is_cuda() && streams.is_contiguous() && floating_activation(streams.scalar_type()) &&
           streams.dim() == 4 && streams.numel() > 0 && streams.size(2) <= kThreads &&
           streams.size(3) <= INT_MAX / streams.size(2);
}

template<bool Matrix> TwoStreamMhcMix prepare_routes(
    const tb::Tensor& streams, const tb::Tensor& function, const tb::Tensor& scale, const tb::Tensor& bias,
    float rms_epsilon, float gate_epsilon, float post_magnitude, float pre_scale, int iterations, bool fixed_identity) {
    TYLOINFER_RUNTIME_CHECK(valid_streams(streams) && (!Matrix || streams.size(2) == 2),
        "hyperconnection streams must have a compatible CUDA [B,T,S,H] layout");
    const auto batch = streams.size(0), tokens = streams.size(1);
    const int count = int(streams.size(2)), hidden = int(streams.size(3)), width = count * hidden;
    const int64_t rows = batch * tokens;
    const int tiles = (width - 1) / kTileWidth + 1;
    const bool with_post = Matrix || scale.numel() == 2;
    const bool with_matrix = Matrix && !fixed_identity;
    const int gates = (with_post ? 2 * count : count) + (with_matrix ? 4 : 0);
    TYLOINFER_RUNTIME_CHECK(
        rows <= INT_MAX / tiles && float_parameter(function, streams) && function.dim() == 2 &&
            function.size(0) == gates && function.size(1) == width &&
            (Matrix || (float_parameter(scale, streams) && (with_post || scale.numel() == 1))) &&
            float_parameter(bias, streams) && bias.numel() == gates &&
            std::isfinite(rms_epsilon) && rms_epsilon > 0 && std::isfinite(gate_epsilon) && gate_epsilon >= 0 &&
            std::isfinite(post_magnitude) && post_magnitude > 0 && std::isfinite(pre_scale) && pre_scale > 0 &&
            (!with_matrix || iterations > 0), "invalid hyperconnection router geometry or parameters");
    tb::Tensor partials = tb::empty({rows, tiles, gates + 1}, function.options());
    tb::Tensor branch = tb::empty({batch, tokens, hidden}, streams.options());
    tb::Tensor post = with_post ? tb::empty({batch, tokens, count}, function.options()) : tb::Tensor{};
    tb::Tensor matrix = with_matrix ? tb::empty({batch, tokens, 4}, function.options()) : tb::Tensor{};
    auto stream = mfq_current_cuda_stream();
    const auto launch = [&](auto sample) {
        using T = decltype(sample);
        projection_partials<<<int(rows) * tiles, kThreads, 0, stream>>>(
            streams.data_ptr<T>(), function.data_ptr<float>(), partials.data_ptr<float>(), width, gates, tiles);
        TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
        prepare_branch<T, Matrix><<<int(rows), kThreads, count * sizeof(float), stream>>>(
            streams.data_ptr<T>(), partials.data_ptr<float>(), Matrix ? nullptr : scale.data_ptr<float>(), bias.data_ptr<float>(),
            branch.data_ptr<T>(), with_post ? post.data_ptr<float>() : nullptr, width, count, hidden, tiles,
            rms_epsilon, gate_epsilon, post_magnitude, pre_scale,
            with_matrix ? matrix.data_ptr<float>() : nullptr, iterations);
    };
    if (streams.scalar_type() == tb::kFloat16) launch(half{});
    else if (streams.scalar_type() == tb::kBFloat16) launch(__nv_bfloat16{});
    else launch(float{});
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return {std::move(branch), std::move(post), std::move(matrix)};
}

tb::Tensor apply_routes(const tb::Tensor& streams, const tb::Tensor& branch, const tb::Tensor& post,
                        const tb::Tensor& combination) {
    TYLOINFER_RUNTIME_CHECK(valid_streams(streams), "hyperconnection streams must be contiguous CUDA [B,T,S,H]");
    const auto batch = streams.size(0), tokens = streams.size(1);
    const auto count = streams.size(2), hidden = streams.size(3), elements = streams.numel();
    TYLOINFER_RUNTIME_CHECK(
        branch.is_cuda() && branch.is_contiguous() && branch.device() == streams.device() &&
            branch.scalar_type() == streams.scalar_type() && branch.dim() == 3 && branch.size(0) == batch &&
            branch.size(1) == tokens && branch.size(2) == hidden && float_parameter(post, streams) &&
            post.dim() == 3 && post.size(0) == batch && post.size(1) == tokens && post.size(2) == count &&
            (elements - 1) / kThreads < INT_MAX &&
            (!combination.defined() || (count == 2 && float_parameter(combination, streams) &&
                combination.dim() == 3 && combination.size(0) == batch && combination.size(1) == tokens &&
                combination.size(2) == 4)), "invalid hyperconnection branch or post gate geometry");
    auto output = tb::empty_like(streams);
    const auto grid = int((elements - 1) / kThreads + 1);
    const auto stream = mfq_current_cuda_stream();
    const auto launch = [&](auto sample) {
        using T = decltype(sample);
        if (combination.defined())
            merge_streams<T, true><<<grid, kThreads, 0, stream>>>(streams.data_ptr<T>(), branch.data_ptr<T>(),
                post.data_ptr<float>(), output.data_ptr<T>(), elements, int(count), int(hidden), combination.data_ptr<float>());
        else
            merge_streams<T, false><<<grid, kThreads, 0, stream>>>(streams.data_ptr<T>(), branch.data_ptr<T>(),
                post.data_ptr<float>(), output.data_ptr<T>(), elements, int(count), int(hidden));
    };
    if (streams.scalar_type() == tb::kFloat16) launch(half{});
    else if (streams.scalar_type() == tb::kBFloat16) launch(__nv_bfloat16{});
    else launch(float{});
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace

IndependentHcMix independent_hc_prepare(const tb::Tensor& streams, const tb::Tensor& function,
    const tb::Tensor& scale, const tb::Tensor& bias, float rms_epsilon, float gate_epsilon, float post_magnitude) {
    auto result = prepare_routes<false>(streams, function, scale, bias, rms_epsilon, gate_epsilon, post_magnitude, 1.f, 0, true);
    return {std::move(result.branch), std::move(result.post)};
}

tb::Tensor independent_hc_merge(const tb::Tensor& streams, const tb::Tensor& branch, const tb::Tensor& post) {
    return apply_routes(streams, branch, post, {});
}

TwoStreamMhcMix two_stream_mhc_prepare(const tb::Tensor& streams, const tb::Tensor& function,
    const tb::Tensor& bias, float epsilon, float pre_scale, int sinkhorn_iterations, bool fixed_identity) {
    return prepare_routes<true>(streams, function, {}, bias, epsilon, 0.f, 2.f, pre_scale, sinkhorn_iterations, fixed_identity);
}

tb::Tensor two_stream_mhc_merge(const tb::Tensor& streams, const tb::Tensor& branch, const TwoStreamMhcMix& mix) {
    return apply_routes(streams, branch, mix.post, mix.combination);
}

}  // namespace mfq::cuda

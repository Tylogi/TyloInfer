#include "mlx_transformer.h"
#include "mlx_resident_budget.h"
#include "engine/backends/metal/ops/attention/kv.h"
#include "engine/config.h"
#include "engine/state/kv_layout.h"

#include "engine/ops.h"
#include "engine/backends/metal/native/tensor/tensor.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;
using mlx::core::CompileOptions;
using mlx::core::MathMode;

const mlx::core::fast::CustomKernelFunction&
fused_fp16_rms_norm_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_rms_f16_f32_f16_v2",
            {"x", "w"},
            {"y"},
            R"METAL(
    constexpr uint N_READS = 4u;
    constexpr uint PASSES = (WIDTH + THREADS * N_READS - 1u) / (THREADS * N_READS);
    constexpr uint SIMD_SIZE = 32u;
    uint lid = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    ulong row_offset = ulong(threadgroup_position_in_grid.x) * WIDTH;
    float4 values[PASSES];
    float accumulator = 0.0f;
#pragma unroll
    for (uint pass = 0u; pass < PASSES; ++pass) {
        uint base = (pass * THREADS + lid) * N_READS;
        values[pass] = base < WIDTH
            ? float4(*reinterpret_cast<device const half4*>(x + row_offset + base)) : float4(0.0f);
#pragma unroll
        for (uint index = 0u; index < N_READS; ++index)
            accumulator += values[pass][index] * values[pass][index];
    }
    accumulator = simd_sum(accumulator);

    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float inverse_rms[1];
    if (lane == 0u) local_sums[simd_group] = accumulator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        accumulator = simd_sum(lane < THREADS / SIMD_SIZE ? local_sums[lane] : 0.0f);
        if (lane == 0u) {
            inverse_rms[0] = metal::precise::rsqrt(
                accumulator / float(WIDTH) + as_type<float>(uint(EPS_BITS)));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
    for (uint pass = 0u; pass < PASSES; ++pass) {
        uint base = (pass * THREADS + lid) * N_READS;
        if (base < WIDTH) {
            float4 packed_weight = *reinterpret_cast<device const float4*>(w + base);
            half4 normalized = half4(packed_weight * (values[pass] * inverse_rms[0]));
            if constexpr (OUTPUT_BF16)
                *reinterpret_cast<device bfloat4*>(y + row_offset + base) = bfloat4(normalized);
            else
                *reinterpret_cast<device half4*>(y + row_offset + base) = normalized;
        }
    }
)METAL",
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

constexpr const char* kMropeSource = R"METAL(
    const uint TOKENS = uint(positions_shape[1]);
    const uint SIZE = uint(x_shape[0]) * uint(DIM);
    uint index = thread_position_in_grid.x;
    if (index >= uint(SIZE)) {
        return;
    }

    uint column = index % uint(DIM);
    if (column >= uint(ROTARY_DIM)) {
        y[index] = x[index];
        return;
    }

    constexpr uint HALF = uint(ROTARY_DIM / 2);
    uint row = index / uint(DIM);
    uint token = row % uint(TOKENS);
    uint batch = row / uint(HEADS * TOKENS);
    uint pair = column < HALF ? column : column - HALF;

    uint axis = 0u;
    if (S0 + S1 + S2 > 0 && POS_AXES == 3) {
        if (INTERLEAVED != 0) {
            uint residue = pair % 3u;
            if (residue == 1u && pair < uint(S1 * 3)) {
                axis = 1u;
            } else if (residue == 2u && pair < uint(S2 * 3)) {
                axis = 2u;
            }
        } else {
            axis = pair < uint(S0)
                ? 0u
                : (pair < uint(S0 + S1) ? 1u : 2u);
        }
    }

    int position = POSITION_BATCH != 0
        ? positions[batch * uint(TOKENS) + token]
        : positions[axis * uint(TOKENS) + token];
    float exponent =
        -2.0f * float(pair) / float(ROTARY_DIM);
    float frequency = pow(params[0], exponent);
    if (YARN != 0) {
        frequency = params[2 + pair];
    }
    float angle = float(position) * frequency;
    float cosine = cos(angle);
    float sine = sin(angle);
    if (YARN != 0) { cosine *= params[1]; sine *= params[1]; }

    uint row_offset = row * uint(DIM);
    float first = float(x[row_offset + pair]);
    float second = float(x[row_offset + pair + HALF]);
    float value = column < HALF
        ? first * cosine - second * sine
        : second * cosine + first * sine;
    y[index] = T(value);
)METAL";

mlx::core::fast::CustomKernelFunction make_mrope_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_mrope",
        {"x", "positions", "params"},
        {"y"},
        kMropeSource,
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& mrope_kernel() {
    static const auto kernel = make_mrope_kernel();
    return kernel;
}

constexpr const char* kAdjacentRopeSource = R"METAL(
    const uint TOKENS = uint(cos_values_shape[0]);
    const uint SIZE = uint(x_shape[0]) * uint(DIM);
    uint index = thread_position_in_grid.x;
    if (index >= uint(SIZE)) {
        return;
    }
    uint column = index % uint(DIM);
    uint row = index / uint(DIM);
    constexpr uint PREFIX = uint(DIM - ROTARY);
    if (column < PREFIX) {
        y[index] = x[index];
        return;
    }

    uint rotary_column = column - PREFIX;
    uint pair = rotary_column >> 1u;
    uint token = (row / uint(HEADS)) % uint(TOKENS);
    float cosine = float(cos_values[token * uint(PAIRS) + pair]);
    float sine = float(sin_values[token * uint(PAIRS) + pair]);
    if (INVERSE != 0) {
        sine = -sine;
    }
    uint pair_base = row * uint(DIM) + PREFIX + (pair << 1u);
    float first = float(x[pair_base]);
    float second = float(x[pair_base + 1u]);
    float result = (rotary_column & 1u) == 0u
        ? first * cosine - second * sine
        : first * sine + second * cosine;
    y[index] = T(result);
)METAL";

const mlx::core::fast::CustomKernelFunction& adjacent_rope_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_adjacent_rope",
            {"x", "cos_values", "sin_values"},
            {"y"},
            kAdjacentRopeSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

array floating(const array& input, mlx::core::Dtype dtype) {
    auto result = input;
    if (result.dtype() != dtype) {
        result = mlx::core::astype(result, dtype);
    }
    return mlx::core::contiguous(result);
}

void require_attention_shape(const array& value, const char* name) {
    if (value.ndim() != 4) {
        throw std::runtime_error(
            std::string(name) + " must have [batch,heads,tokens,head_dim] shape");
    }
}

} // namespace

std::int64_t qwen_yarn_capacity(std::int64_t native_context, double maximum_factor) {
    if (native_context <= 0 || !std::isfinite(maximum_factor) || maximum_factor < 1.0 ||
        maximum_factor * native_context > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("invalid Qwen YaRN context limit");
    }
    return static_cast<std::int64_t>(std::floor(native_context * maximum_factor));
}

MlxYarnScaling qwen_yarn_scaling(const MlxYarnScaling& settings,
    std::int64_t native_context, std::int64_t requested_context, double maximum_factor) {
    if (requested_context <= 0 || requested_context > qwen_yarn_capacity(native_context, maximum_factor)) {
        throw std::invalid_argument("context exceeds the YaRN extension limit");
    }
    auto result = settings;
    result.original_max_position_embeddings = native_context;
    result.enabled = requested_context > native_context;
    result.factor = std::max(1.0, double(requested_context) / native_context);
    result.attention_factor = result.enabled ? 1.0 + 0.1 * std::log(result.factor) : 1.0;
    return result;
}

double yarn_frequency(int pair, int dimension, double theta, const MlxYarnScaling& scaling) {
    if (dimension <= 0 || dimension % 2 || pair < 0 || pair >= dimension / 2 ||
        !std::isfinite(theta) || theta <= 1.0) throw std::invalid_argument("invalid YaRN frequency");
    double frequency = std::pow(theta, -2.0 * pair / dimension);
    if (!scaling.enabled) return frequency;
    if (!std::isfinite(scaling.factor) || scaling.factor < 1.0 ||
        scaling.original_max_position_embeddings <= 0 ||
        !std::isfinite(scaling.beta_fast) || !std::isfinite(scaling.beta_slow) ||
        scaling.beta_slow <= 0.0 || scaling.beta_fast <= scaling.beta_slow ||
        !std::isfinite(scaling.attention_factor) || scaling.attention_factor <= 0.0) {
        throw std::invalid_argument("invalid YaRN scaling");
    }
    const auto correction = [&](double rotations) {
        return dimension * std::log(scaling.original_max_position_embeddings / (rotations * 2.0 * std::acos(-1.0))) / (2.0 * std::log(theta));
    };
    const double low = std::max(0.0, std::floor(correction(scaling.beta_fast)));
    double high = std::min(double(dimension - 1), std::ceil(correction(scaling.beta_slow)));
    if (low == high) high += 0.001;
    const double ramp = std::clamp((pair - low) / (high - low), 0.0, 1.0);
    return frequency * (1.0 - ramp + ramp / scaling.factor);
}

std::pair<array, array> mlx_yarn_tables(
    int dimension,
    int length,
    float theta,
    const MlxYarnScaling& scaling) {
    if (dimension <= 0 || dimension % 2 != 0 || length <= 0 ||
        !std::isfinite(theta) || theta <= 0.0f) {
        throw std::invalid_argument("invalid YaRN table parameters");
    }
    const int pairs = dimension / 2;
    std::vector<float> frequency(pairs);
    for (int pair = 0; pair < pairs; ++pair) {
        frequency[pair] = 1.0f / std::pow(
            theta,
            static_cast<float>(2 * pair) / static_cast<float>(dimension));
    }
    if (scaling.enabled &&
        scaling.original_max_position_embeddings > 0) {
        if (!std::isfinite(scaling.factor) || scaling.factor <= 0.0 ||
            !std::isfinite(scaling.beta_fast) || scaling.beta_fast <= 0.0 ||
            !std::isfinite(scaling.beta_slow) || scaling.beta_slow <= 0.0) {
            throw std::invalid_argument("invalid YaRN scaling");
        }
        const auto correction = [&](double rotations) {
            return static_cast<double>(dimension) *
                std::log(
                    static_cast<double>(
                        scaling.original_max_position_embeddings) /
                    (rotations * 2.0 * std::acos(-1.0))) /
                (2.0 * std::log(static_cast<double>(theta)));
        };
        const int low = std::max(
            static_cast<int>(std::floor(correction(scaling.beta_fast))), 0);
        double high = std::min(
            std::ceil(correction(scaling.beta_slow)),
            static_cast<double>(dimension - 1));
        if (static_cast<double>(low) == high) {
            high += 0.001;
        }
        for (int pair = 0; pair < pairs; ++pair) {
            const float ramp = std::clamp(
                static_cast<float>(
                    (static_cast<double>(pair) - low) / (high - low)),
                0.0f,
                1.0f);
            const float smooth = 1.0f - ramp;
            frequency[pair] =
                frequency[pair] / static_cast<float>(scaling.factor) *
                    (1.0f - smooth) +
                frequency[pair] * smooth;
        }
    }
    std::vector<float> cosine(
        static_cast<std::size_t>(length) * pairs);
    std::vector<float> sine(cosine.size());
    for (int position = 0; position < length; ++position) {
        for (int pair = 0; pair < pairs; ++pair) {
            const float angle = static_cast<float>(position) * frequency[pair];
            cosine[position * pairs + pair] = std::cos(angle);
            sine[position * pairs + pair] = std::sin(angle);
        }
    }
    return {
        array(cosine.begin(), Shape{length, pairs}),
        array(sine.begin(), Shape{length, pairs}),
    };
}

array mlx_rope_adjacent(
    const array& value,
    int rotary_dimension,
    const array& cosine,
    const array& sine,
    bool inverse) {
    if (value.ndim() < 2 || value.ndim() > 4 || value.shape(-1) <= 0 ||
        rotary_dimension <= 0 || rotary_dimension > value.shape(-1) ||
        rotary_dimension % 2 != 0 || cosine.shape() != sine.shape() ||
        cosine.shape(-1) != rotary_dimension / 2) {
        throw std::invalid_argument("invalid adjacent RoPE input");
    }

    auto source = value;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::bfloat16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::contiguous(source);
    const int dimension = source.shape(-1);
    const int tokens = source.ndim() == 2 ? source.shape(0) : source.shape(1);
    const int heads = source.ndim() == 4 ? source.shape(2) : 1;
    const int pairs = rotary_dimension / 2;
    if (tokens <= 0 || heads <= 0 ||
        cosine.size() != static_cast<std::size_t>(tokens) * pairs) {
        throw std::invalid_argument(
            "adjacent RoPE table is not token-broadcastable");
    }
    auto cos_values = mlx::core::contiguous(
        mlx::core::astype(cosine, mlx::core::float32));
    auto sin_values = mlx::core::contiguous(
        mlx::core::astype(sine, mlx::core::float32));
    if (source.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("adjacent RoPE tensor exceeds Metal limits");
    }
    const int grid = static_cast<int>(source.size());
    auto outputs = adjacent_rope_kernel()(
        {
            mlx::core::reshape(source, Shape{grid / dimension, dimension}),
            mlx::core::reshape(cos_values, Shape{tokens, pairs}),
            mlx::core::reshape(sin_values, Shape{tokens, pairs}),
        },
        {source.shape()},
        {source.dtype()},
        {grid, 1, 1},
        {std::min(256, grid), 1, 1},
        {
            {"T", source.dtype()},
            {"DIM", dimension},
            {"ROTARY", rotary_dimension},
            {"PAIRS", pairs},
            {"HEADS", heads},
            {"INVERSE", inverse ? 1 : 0},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array mlx_rope_adjacent(
    const array& value,
    const array& cosine,
    const array& sine,
    bool inverse) {
    if (value.ndim() == 0) {
        throw std::invalid_argument("invalid adjacent RoPE input");
    }
    return mlx_rope_adjacent(
        value, value.shape(-1), cosine, sine, inverse);
}

MlxRmsNorm::MlxRmsNorm(
    array weight,
    float eps,
    float weight_offset)
    : weight_(mlx::core::contiguous(
          mlx::core::astype(std::move(weight), mlx::core::float32)) +
          weight_offset),
      eps_(eps),
      width_(weight_.ndim() == 1 ? weight_.shape(0) : 0) {
    if (width_ <= 0) {
        throw std::runtime_error(
            "RMSNorm weight must be a non-empty vector");
    }
    if (!std::isfinite(eps_) || eps_ <= 0.0f ||
        !std::isfinite(weight_offset)) {
        throw std::runtime_error("invalid RMSNorm parameters");
    }
    weight_.eval();
}

array MlxRmsNorm::operator()(const array& input, std::optional<mlx::core::Dtype> output_dtype) const {
    if (input.ndim() == 0 || input.shape(-1) != width_) {
        throw std::runtime_error("RMSNorm input width mismatch");
    }
    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::bfloat16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    const auto result_dtype = output_dtype.value_or(source.dtype());
    if (result_dtype != mlx::core::float16 && result_dtype != mlx::core::bfloat16 && result_dtype != mlx::core::float32)
        throw std::invalid_argument("RMSNorm output dtype must be floating point");
    static const bool fused_enabled = [] {
        const auto* setting =
            std::getenv("TYLOINFER_METAL_FUSED_RMS_CAST");
        return setting == nullptr || std::strcmp(setting, "0") != 0;
    }();
    const auto rows = source.size() / std::size_t(width_);
    if (fused_enabled && source.dtype() == mlx::core::float16 &&
        result_dtype != mlx::core::float32 && width_ >= 1024 && width_ <= 8192 && width_ % 128 == 0 && rows >= 1 && rows <= 8) {
        const int threads = std::min(width_ / 4, 1024);
        auto contiguous = mlx::core::contiguous(
            mlx::core::reshape(source, Shape{int(rows), width_}));
        return fused_fp16_rms_norm_kernel()(
            {contiguous, weight_},
            {source.shape()},
            {result_dtype},
            {threads * int(rows), 1, 1},
            {threads, 1, 1},
            {{"WIDTH", width_}, {"THREADS", threads}, {"EPS_BITS", std::bit_cast<std::int32_t>(eps_)},
             {"OUTPUT_BF16", result_dtype == mlx::core::bfloat16}},
            std::nullopt,
            false,
            {}).front();
    }
    auto normalized = mlx::core::fast::rms_norm(
        source,
        std::optional<array>(weight_),
        eps_);
    // MLX promotes an FP16 activation to FP32 when the RMSNorm weight is
    // FP32. Keep the stored/offset weight in FP32 for accuracy, but preserve
    // the model activation dtype at the operator boundary.
    auto rounded = normalized.dtype() == source.dtype()
        ? normalized
        : mlx::core::astype(normalized, source.dtype());
    return rounded.dtype() == result_dtype ? rounded : mlx::core::astype(rounded, result_dtype);
}

array apply_rope(
    const array& input,
    int rotary_dimension,
    float base,
    int offset,
    const MlxYarnScaling& scaling) {
    if (input.ndim() < 2 ||
        rotary_dimension <= 0 ||
        rotary_dimension > input.shape(-1) ||
        rotary_dimension % 2 != 0 ||
        !std::isfinite(base) ||
        base <= 0.0f ||
        offset < 0) {
        throw std::runtime_error("invalid RoPE input or parameters");
    }
    if (scaling.enabled) {
        return apply_rope(input, mlx::core::arange(offset, offset + input.shape(-2), mlx::core::int32),
            rotary_dimension, base, {}, false, scaling);
    }
    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::bfloat16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    return mlx::core::fast::rope(
        source,
        rotary_dimension,
        false,
        base,
        1.0f,
        offset);
}

array apply_rope(
    const array& input,
    const array& positions,
    int rotary_dimension,
    float base,
    const std::vector<std::int64_t>& sections,
    bool interleaved,
    const MlxYarnScaling& scaling) {
    if (input.ndim() < 2 ||
        input.shape(-2) <= 0 ||
        input.shape(-1) <= 0 ||
        rotary_dimension <= 0 ||
        rotary_dimension > input.shape(-1) ||
        rotary_dimension % 2 != 0 ||
        !std::isfinite(base) ||
        base <= 0.0f) {
        throw std::runtime_error(
            "invalid explicit-position RoPE input or parameters");
    }

    const int tokens = input.shape(-2);
    int position_axes = 0;
    bool position_batch = false;
    if (positions.ndim() == 1) {
        position_axes = 1;
    } else if (positions.ndim() == 2 &&
               positions.shape(0) == input.shape(0) &&
               sections.empty()) {
        position_axes = 1;
        position_batch = positions.shape(0) > 1;
    } else if (positions.ndim() == 2 &&
               (positions.shape(0) == 1 ||
                positions.shape(0) == 3)) {
        position_axes = positions.shape(0);
    } else {
        throw std::runtime_error(
            "RoPE positions must have [tokens], [1,tokens], "
            "[batch,tokens], or [3,tokens] shape");
    }
    if (positions.shape(-1) != tokens) {
        throw std::runtime_error(
            "RoPE position length must match the input token count");
    }

    int section_values[3] = {0, 0, 0};
    if (!sections.empty()) {
        if (sections.size() != 3) {
            throw std::runtime_error(
                "MRoPE sections must contain exactly three entries");
        }
        std::int64_t total = 0;
        for (std::size_t index = 0; index < sections.size(); ++index) {
            const auto section = sections[index];
            if (section < 0 ||
                section > std::numeric_limits<int>::max()) {
                throw std::runtime_error(
                    "MRoPE sections must be nonnegative int values");
            }
            section_values[index] = static_cast<int>(section);
            total += section;
        }
        if (total != rotary_dimension / 2) {
            throw std::runtime_error(
                "MRoPE sections must sum to rotary_dim / 2");
        }
    } else if (interleaved) {
        throw std::runtime_error(
            "interleaved MRoPE requires three section sizes");
    }

    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::bfloat16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::contiguous(source);
    auto position_values = positions;
    if (position_values.dtype() != mlx::core::int32) {
        position_values =
            mlx::core::astype(position_values, mlx::core::int32);
    }
    position_values = mlx::core::contiguous(position_values);

    if (source.size() >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "explicit-position RoPE Metal grid exceeds MLX limits");
    }
    const int size = static_cast<int>(source.size());
    std::vector<float> rope_parameters{base};
    if (scaling.enabled) {
        rope_parameters.push_back(static_cast<float>(scaling.attention_factor));
        for (int pair = 0; pair < rotary_dimension / 2; ++pair)
            rope_parameters.push_back(static_cast<float>(yarn_frequency(pair, rotary_dimension, base, scaling)));
    }
    const array parameters(rope_parameters.begin(), Shape{static_cast<int>(rope_parameters.size())});
    std::vector<std::pair<
        std::string,
        mlx::core::fast::TemplateArg>> templates{
        {"T", source.dtype()},
        {"DIM", source.shape(-1)},
        {"HEADS", source.ndim() >= 3 ? source.shape(-3) : 1},
        {"ROTARY_DIM", rotary_dimension},
        {"POS_AXES", position_axes},
        {"POSITION_BATCH", static_cast<int>(position_batch)},
        {"S0", section_values[0]},
        {"S1", section_values[1]},
        {"S2", section_values[2]},
        {"INTERLEAVED", static_cast<int>(interleaved)},
        {"YARN", static_cast<int>(scaling.enabled)},
    };
    auto outputs = mrope_kernel()(
        {
            mlx::core::reshape(source, Shape{size / source.shape(-1), source.shape(-1)}),
            mlx::core::reshape(position_values,
                Shape{static_cast<int>(position_values.size() / tokens), tokens}),
            parameters,
        },
        {source.shape()},
        {source.dtype()},
        {size, 1, 1},
        {std::min(256, size), 1, 1},
        std::move(templates),
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array mlx_circular_cache_history(
    const array& cache,
    int position) {
    if (cache.ndim() < 2 || cache.shape(1) <= 0 || position < 0) {
        throw std::invalid_argument(
            "invalid circular cache history request");
    }
    const int window = cache.shape(1);
    const int length = std::min(position, window);
    if (position <= window) {
        Shape begin(cache.ndim(), 0);
        Shape end = cache.shape();
        end[1] = length;
        return mlx::core::slice(cache, std::move(begin), std::move(end));
    }
    auto indices = mlx::core::remainder(
        mlx::core::arange(
            position - length,
            position,
            1,
            mlx::core::int32),
        array(window, mlx::core::int32));
    return mlx::core::take(cache, indices, 1);
}

MlxKvCache::MlxKvCache(
    int batch,
    int heads,
    int maximum_sequence,
    int head_dimension,
    int initial_capacity,
    mlx::core::Dtype dtype,
    bool quantization_allowed)
    : maximum_sequence_(maximum_sequence) {
    const auto quantization = quantization_allowed ? tyloi::engine::kv_quantization() : tyloi::state::KvQuantization{};
    if (batch <= 0 || heads <= 0 || maximum_sequence <= 0 || head_dimension <= 0 ||
        initial_capacity < 0 || (dtype != mlx::core::float16 &&
        dtype != mlx::core::bfloat16 && dtype != mlx::core::float32))
        throw std::runtime_error("invalid KV cache dimensions or dtype");
    const auto capacity = std::min(initial_capacity, maximum_sequence);
    const auto logical_dtype = tyloi::metal::scalar_type(dtype);
    const auto layout = tyloi::state::kv_layout(batch, heads, capacity, head_dimension,
        false, logical_dtype, quantization);
    MlxResidentBudgetScope::reserve(layout.bytes);
    auto next = tyloi::state::allocate_kv(batch, heads, capacity, head_dimension, false,
        tyloi::Device{tyloi::DeviceType::metal, 0}, logical_dtype, quantization);
    kv_ = std::move(next);
    resources_.set({kv_.physical_key.nbytes() + kv_.physical_value.nbytes(), static_cast<std::size_t>(batch)});
}

const array& MlxKvCache::key_storage() const { return tyloi::metal::array(kv_.physical_key); }
const array& MlxKvCache::value_storage() const { return tyloi::metal::array(kv_.physical_value); }

void MlxKvCache::set_storage(array key, array value) {
    auto next_key = tyloi::metal::tensor(std::move(key));
    auto next_value = tyloi::metal::tensor(std::move(value));
    kv_.physical_key = std::move(next_key);
    kv_.physical_value = std::move(next_value);
}

void MlxKvCache::materialize() {
    mlx::core::eval(key_storage(), value_storage());
}

void MlxKvCache::ensure_capacity(int required) {
    if (required <= capacity()) {
        return;
    }
    if (required > maximum_sequence_) {
        throw std::runtime_error("KV cache exceeds maximum sequence length");
    }
    const auto old_capacity = capacity();
    const auto next_capacity = tyloi::state::kv_capacity(old_capacity, required, maximum_sequence_);
    MlxResidentBudgetScope::reserve(tyloi::state::kv_layout(kv_.batch, kv_.heads, next_capacity,
        kv_.head_dim, false, kv_.dtype, kv_.quantization).bytes);
    auto next = tyloi::state::grow_kv(kv_, next_capacity, old_capacity);
    kv_ = std::move(next);
    resources_.set({key_storage().nbytes() + value_storage().nbytes(), static_cast<std::size_t>(kv_.batch)});
}

void MlxKvCache::append_only(const array& key, const array& value) {
    require_attention_shape(key, "KV key");
    require_attention_shape(value, "KV value");
    if (key.shape() != value.shape() ||
        key.shape(0) != static_cast<int>(kv_.batch) ||
        key.shape(1) != static_cast<int>(kv_.heads) ||
        key.shape(3) != static_cast<int>(kv_.head_dim)) {
        throw std::runtime_error("KV append shape mismatch");
    }
    const int tokens = key.shape(2);
    if (tokens < 0 || position_ > maximum_sequence_ - tokens) {
        throw std::runtime_error("KV append exceeds maximum sequence length");
    }
    ensure_capacity(position_ + tokens);
    if (tokens > 0) {
        auto next = mlx_kv_append(kv_, key, value, position_);
        set_storage(std::move(next.first), std::move(next.second));
        position_ += tokens;
    }
}

void MlxKvCache::reserve_append(int tokens) {
    if (quantized()) throw std::runtime_error("direct KV writes cannot target a quantized cache");
    if (tokens <= 0 || position_ > maximum_sequence_ - tokens) {
        throw std::runtime_error("invalid KV cache reservation");
    }
    ensure_capacity(position_ + tokens);
    position_ += tokens;
}

void MlxKvCache::trim(int tokens) {
    if (tokens < 0 || tokens > position_) {
        throw std::runtime_error("invalid KV cache trim");
    }
    position_ -= tokens;
}

std::pair<array, array> MlxKvCache::physical_view() const {
    const Shape start{0, 0, 0, 0};
    const Shape stop{
        static_cast<int>(kv_.batch),
        static_cast<int>(kv_.heads),
        position_,
        key_storage().shape(3),
    };
    return {
        mlx::core::slice(key_storage(), start, stop),
        mlx::core::slice(value_storage(), start, Shape{static_cast<int>(kv_.batch), static_cast<int>(kv_.heads), position_, value_storage().shape(3)}),
    };
}

MlxKvCacheSnapshot MlxKvCache::snapshot(bool detached) const {
    if (position_ <= 0 || position_ > capacity()) {
        throw std::runtime_error(
            "cannot snapshot an empty or inconsistent KV cache");
    }
    auto visible = physical_view();
    auto key = detached ? mlx::core::array(tyloi::metal::array(
        tyloi::ops::clone(tyloi::metal::tensor(visible.first)))) : visible.first;
    auto value = detached ? mlx::core::array(tyloi::metal::array(
        tyloi::ops::clone(tyloi::metal::tensor(visible.second)))) : visible.second;
    mlx::core::eval(key, value);
    return MlxKvCacheSnapshot{
        static_cast<int>(kv_.batch),
        static_cast<int>(kv_.heads),
        maximum_sequence_,
        static_cast<int>(kv_.head_dim),
        capacity(),
        position_,
        key_storage().dtype(),
        std::move(key),
        std::move(value),
        kv_.quantization,
    };
}

void MlxKvCache::restore_snapshot(
    const MlxKvCacheSnapshot& snapshot) {
    if (snapshot.batch != static_cast<int>(kv_.batch) || snapshot.heads != static_cast<int>(kv_.heads) ||
        snapshot.maximum_sequence != maximum_sequence_ ||
        snapshot.head_dimension != static_cast<int>(kv_.head_dim) ||
        snapshot.quantization != kv_.quantization || snapshot.dtype != key_storage().dtype() || snapshot.capacity <= 0 ||
        snapshot.capacity > maximum_sequence_ || snapshot.position <= 0 ||
        snapshot.position > snapshot.capacity || snapshot.key.ndim() != 4 ||
        snapshot.value.shape() != Shape{static_cast<int>(kv_.batch), static_cast<int>(kv_.heads), snapshot.position, value_storage().shape(3)} ||
        snapshot.key.shape() != Shape{
            static_cast<int>(kv_.batch), static_cast<int>(kv_.heads), snapshot.position, key_storage().shape(3)} ||
        snapshot.key.dtype() != key_storage().dtype() || snapshot.value.dtype() != value_storage().dtype()) {
        throw std::runtime_error("KV cache snapshot topology mismatch");
    }
    const auto allocation = tyloi::state::kv_layout(static_cast<int>(kv_.batch), static_cast<int>(kv_.heads), snapshot.capacity,
        static_cast<int>(kv_.head_dim), false, kv_.dtype, kv_.quantization).bytes;
    const auto copies = tyloi::state::kv_layout(static_cast<int>(kv_.batch), static_cast<int>(kv_.heads), snapshot.position,
        static_cast<int>(kv_.head_dim), false, kv_.dtype, kv_.quantization).bytes;
    if (copies > std::numeric_limits<std::size_t>::max() - allocation)
        throw std::overflow_error("KV tensor size overflow");
    MlxResidentBudgetScope::reserve(allocation + copies);
    auto next = tyloi::state::allocate_kv(kv_.batch, kv_.heads, snapshot.capacity, kv_.head_dim,
        false, tyloi::Device{tyloi::DeviceType::metal, 0}, kv_.dtype, kv_.quantization);
    next.physical_key = tyloi::ops::update_slice(next.physical_key,
        tyloi::ops::clone(tyloi::metal::tensor(snapshot.key)), 2, 0);
    next.physical_value = tyloi::ops::update_slice(next.physical_value,
        tyloi::ops::clone(tyloi::metal::tensor(snapshot.value)), 2, 0);
    mlx::core::eval(tyloi::metal::array(next.physical_key), tyloi::metal::array(next.physical_value));
    kv_ = std::move(next);
    position_ = snapshot.position;
    resources_.set({key_storage().nbytes() + value_storage().nbytes(), static_cast<std::size_t>(kv_.batch)});
}

void MlxKvCache::restore_snapshot(MlxKvCacheSnapshot&& snapshot) {
    if (snapshot.batch != static_cast<int>(kv_.batch) || snapshot.heads != static_cast<int>(kv_.heads) ||
        snapshot.maximum_sequence != maximum_sequence_ ||
        snapshot.head_dimension != static_cast<int>(kv_.head_dim) ||
        snapshot.quantization != kv_.quantization || snapshot.dtype != key_storage().dtype() || snapshot.capacity <= 0 ||
        snapshot.capacity > maximum_sequence_ || snapshot.position <= 0 ||
        snapshot.position > snapshot.capacity || snapshot.key.ndim() != 4 ||
        snapshot.value.shape() != Shape{static_cast<int>(kv_.batch), static_cast<int>(kv_.heads), snapshot.position, value_storage().shape(3)} ||
        snapshot.key.shape() != Shape{
            static_cast<int>(kv_.batch), static_cast<int>(kv_.heads), snapshot.position, key_storage().shape(3)} ||
        snapshot.key.dtype() != key_storage().dtype() || snapshot.value.dtype() != value_storage().dtype()) {
        throw std::runtime_error("KV cache snapshot topology mismatch");
    }
    set_storage(std::move(snapshot.key), std::move(snapshot.value));
    position_ = snapshot.position;
    resources_.set({key_storage().nbytes() + value_storage().nbytes(), static_cast<std::size_t>(kv_.batch)});
}

} // namespace mfq::metal

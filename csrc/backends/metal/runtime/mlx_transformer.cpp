#include "mlx_transformer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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

} // namespace mfq::metal

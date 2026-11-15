#include "tail.h"
#include "adjacent.h"
#include "engine/backends/metal/native/tensor/tensor.h"

#include <mlx/mlx.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyloi::metal::ops {
namespace {
using mlx::core::array;
using mlx::core::Shape;
using mlx::core::CompileOptions;
using mlx::core::MathMode;
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

array adjacent_rope(
    const array& value,
    int rotary_dimension,
    const array& cosine,
    const array& sine,
    bool inverse) {
    if (value.ndim() < 2 || value.ndim() > 4 || value.shape(-1) <= 0 ||
        rotary_dimension <= 0 || rotary_dimension > value.shape(-1) ||
        rotary_dimension % 2 != 0 || cosine.ndim() == 0 || cosine.shape() != sine.shape() ||
        cosine.shape(-1) != rotary_dimension / 2) {
        throw std::invalid_argument("invalid adjacent RoPE input");
    }

    const auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
    auto source = value;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::bfloat16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16, stream);
    }
    source = mlx::core::contiguous(source, stream);
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
        mlx::core::astype(cosine, mlx::core::float32, stream), stream);
    auto sin_values = mlx::core::contiguous(
        mlx::core::astype(sine, mlx::core::float32, stream), stream);
    if (source.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("adjacent RoPE tensor exceeds Metal limits");
    }
    const int grid = static_cast<int>(source.size());
    auto outputs = adjacent_rope_kernel()(
        {
            mlx::core::reshape(source, Shape{grid / dimension, dimension}, stream),
            mlx::core::reshape(cos_values, Shape{tokens, pairs}, stream),
            mlx::core::reshape(sin_values, Shape{tokens, pairs}, stream),
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
        stream);
    return std::move(outputs.front());
}

::tyloi::ops::TailRopeTable tail_rope_table(std::int64_t max_positions, double base,
        std::int64_t compress_ratio, Device device, double compress_base,
        std::int64_t original_positions, double factor, double beta_fast, double beta_slow) {
    const Device gpu{DeviceType::metal, 0};
    const auto positions = std::max<std::int64_t>(1, max_positions);
    const double theta = compress_ratio > 0 && compress_base > 0 ? compress_base : base;
    if (device != gpu || positions > INT32_MAX / 32 || !std::isfinite(theta) || theta <= 0 ||
        (compress_ratio > 0 && original_positions > 0 &&
         (theta == 1 || !std::isfinite(factor) || factor <= 0 || !std::isfinite(beta_fast) ||
          !std::isfinite(beta_slow) || beta_fast <= 0 || beta_slow <= 0)))
        throw std::invalid_argument("invalid Metal tail RoPE table parameters");
    double low = 0, high = 0;
    if (compress_ratio > 0 && original_positions > 0) {
        const auto correction = [&](double rotations) {
            return 64.0 * std::log(original_positions / (rotations * 2.0 * std::acos(-1.0))) / (2.0 * std::log(theta));
        };
        low = std::max(0.0, std::floor(correction(beta_fast)));
        high = std::min(63.0, std::ceil(correction(beta_slow)));
        if (high == low) high += 0.001;
    }
    std::vector<float> frequency(32);
    for (int j = 0; j < 32; ++j) {
        frequency[j] = float(std::pow(theta, -2.0 * j / 64.0));
        if (compress_ratio > 0 && original_positions > 0) {
            float ramp = float(std::clamp((j - low) / (high - low), 0.0, 1.0));
            frequency[j] = frequency[j] / float(factor) * ramp + frequency[j] * (1.0f - ramp);
        }
    }
    const auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
    auto position = mlx::core::reshape(
        mlx::core::arange(int(positions), mlx::core::float32, stream), {int(positions), 1}, stream);
    auto frequencies = array(frequency.begin(), Shape{1, 32});
    auto angles = mlx::core::multiply(position, frequencies, stream);
    auto sine = mlx::core::sin(angles, stream);
    return {metal::tensor(mlx::core::cos(angles, stream)), metal::tensor(sine),
            metal::tensor(mlx::core::negative(sine, stream))};
}

Tensor tail_rope(const Tensor& input, const Tensor& positions,
        const ::tyloi::ops::TailRopeTable& table, bool inverse) {
    const Device gpu{DeviceType::metal, 0};
    const auto& sine = inverse ? table.negative_sin : table.sin;
    if (!input.defined() || input.is_encoded() || input.device() != gpu || input.dim() != 4 ||
        input.numel() <= 0 || input.numel() > INT32_MAX || input.size(-1) < 64 || input.size(-1) % 2 ||
        (input.scalar_type() != ScalarType::float16 && input.scalar_type() != ScalarType::float32) ||
        !positions.defined() || positions.is_encoded() || (!positions.is_cpu() && positions.device() != gpu) ||
        positions.numel() != input.size(2) || !table.cos.defined() || !sine.defined() ||
        table.cos.device() != gpu || sine.device() != gpu || table.cos.scalar_type() != ScalarType::float32 ||
        sine.scalar_type() != ScalarType::float32 || table.cos.dim() != 2 || sine.dim() != 2 ||
        table.cos.size(0) <= 0 || table.cos.size(1) < 32 || table.cos.size(0) != sine.size(0) ||
        table.cos.size(1) != sine.size(1))
        throw std::invalid_argument("invalid Metal tail RoPE input");
    static const auto kernel = mlx::core::fast::metal_kernel("tyloi_tail_rope",
        {"x", "positions", "cosine", "sine"}, {"y"}, R"METAL(
    uint i = thread_position_in_grid.x;
    if (i >= uint(SIZE)) return;
    uint column = i % uint(DIM), row = i / uint(DIM);
    if (column < uint(DIM - 64)) { y[i] = x[i]; return; }
    uint pair = (column - uint(DIM - 64)) / 2;
    uint token = row % uint(TOKENS);
    long index = positions[token] * long(STRIDE) + long(pair);
    float c = cosine[index], s = sine[index];
    uint offset = row * uint(DIM) + uint(DIM - 64) + pair * 2;
    float a = float(x[offset]), b = float(x[offset + 1]);
    y[i] = T((column & 1) ? b * c + a * s : a * c - b * s);
)METAL", "", true, false);
    auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
    auto x = mlx::core::contiguous(metal::array(input), stream);
    auto pos = metal::array(positions.device() == gpu ? positions : metal::copy(positions, gpu));
    pos = mlx::core::contiguous(mlx::core::astype(pos, mlx::core::int64, stream), stream);
    auto cosine = mlx::core::contiguous(metal::array(table.cos), stream);
    auto sin_values = mlx::core::contiguous(metal::array(sine), stream);
    auto result = kernel({x, pos, cosine, sin_values}, {x.shape()}, {x.dtype()},
        {int(input.numel()), 1, 1}, {std::min(256, int(input.numel())), 1, 1},
        {{"T", x.dtype()}, {"SIZE", int(input.numel())}, {"DIM", int(input.size(-1))},
         {"TOKENS", int(input.size(2))}, {"STRIDE", int(table.cos.size(1))}}, std::nullopt, false, stream);
    return metal::tensor(std::move(result.front()));
}
} // namespace tyloi::metal::ops

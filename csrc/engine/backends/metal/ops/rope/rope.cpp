#include "rope.h"
#include "engine/backends/metal/native/tensor/tensor.h"
#include <mlx/mlx.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <array>
#include <stdexcept>
#include <vector>

namespace tyloi::metal::ops {
namespace {
using mlx::core::array;
using mlx::core::Shape;
const Device gpu{DeviceType::metal, 0};

const mlx::core::fast::CustomKernelFunction& kernel() {
    static const auto value = mlx::core::fast::metal_kernel("tyloi_rope_table",
        {"x", "pos", "cosine", "sine", "inverse"}, {"y"}, R"METAL(
    uint i = thread_position_in_grid.x;
    if (i >= uint(SIZE)) return;
    uint col = i % uint(DIM), row = i / uint(DIM);
    if (col >= uint(ROTARY)) { y[i] = x[i]; return; }
    constexpr uint HALF = ROTARY / 2;
    uint pair = col % HALF, token = row % uint(TOKENS);
    uint axis = 0;
    if (GRID && AXES > 1) {
        uint ordered = INTERLEAVED ? uint(inverse[pair]) : pair;
        axis = ordered < uint(S0) ? 0 : (ordered < uint(S0 + S1) ? 1 : 2);
        if (axis >= uint(AXES)) axis = 0;
    }
    uint batch = BATCHES ? row / uint(TOKENS * ROWS_PER_BATCH) : 0;
    long p = pos[(GRID ? axis : batch) * uint(TOKENS) + token];
    p = max(long(0), min(long(TABLE_LEN - 1), p));
    float c = cosine[p * HALF + pair], s = sine[p * HALF + pair];
    float a = float(x[row * DIM + pair]), b = float(x[row * DIM + pair + HALF]);
    if (BF16) {
        c = float(bfloat(c)); s = float(bfloat(s));
        float ac = float(bfloat(a * c)), bs = float(bfloat(b * s));
        float bc = float(bfloat(b * c)), as = float(bfloat(a * s));
        y[i] = T(col < HALF ? float(bfloat(ac - bs)) : float(bfloat(bc + as)));
    } else y[i] = T(col < HALF ? a * c - b * s : b * c + a * s);
)METAL", "", true, false);
    return value;
}
}

::tyloi::ops::RopeTable rope_table(std::int64_t positions, std::int64_t dimension, double base,
        Device device, std::int64_t frequency_dim, std::int64_t active_pairs) {
    if (device != gpu || positions <= 0 || positions > INT32_MAX || dimension <= 0 ||
        dimension > INT32_MAX || dimension % 2 || !std::isfinite(base) || base <= 0 ||
        active_pairs > dimension / 2 || positions > INT32_MAX / (dimension / 2))
        throw std::invalid_argument("invalid Metal RoPE table parameters");
    const auto half = dimension / 2;
    const auto denominator = frequency_dim > 0 ? frequency_dim : dimension;
    if (active_pairs < 0) active_pairs = half;
    std::vector<float> frequency(half);
    for (std::int64_t j = 0; j < half; ++j)
        frequency[j] = j < active_pairs ? float(std::pow(base, -2.0 * j / denominator)) : 0.0f;
    const auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
    auto position = mlx::core::reshape(
        mlx::core::arange(int(positions), mlx::core::float32, stream), {int(positions), 1}, stream);
    auto frequencies = array(frequency.begin(), Shape{1, int(half)});
    auto angles = mlx::core::multiply(position, frequencies, stream);
    ::tyloi::ops::RopeTable result;
    result.cos = metal::tensor(mlx::core::cos(angles, stream));
    result.sin = metal::tensor(mlx::core::sin(angles, stream));
    result.rotary_dim = dimension;
    return result;
}

Tensor rope(const Tensor& input, const Tensor& positions, const ::tyloi::ops::RopeTable& table,
        bool bf16, bool grid) {
    if (!input.defined() || input.is_encoded() || input.device() != gpu || input.dim() < 2 ||
        input.size(-2) <= 0 || input.size(-1) <= 0 || input.numel() <= 0 || input.numel() > INT32_MAX ||
        (input.scalar_type() != ScalarType::float16 && input.scalar_type() != ScalarType::bfloat16 &&
         input.scalar_type() != ScalarType::float32) || !positions.defined() || positions.is_encoded() ||
        (!positions.is_cpu() && positions.device() != gpu) || positions.dim() < 1 || positions.dim() > 2 ||
        positions.size(-1) != input.size(-2) || table.rotary_dim <= 0 || table.rotary_dim % 2 ||
        table.rotary_dim > input.size(-1) || !table.cos.defined() || !table.sin.defined() ||
        table.cos.device() != gpu || table.sin.device() != gpu || table.cos.scalar_type() != ScalarType::float32 ||
        table.sin.scalar_type() != ScalarType::float32 || table.cos.dim() != 2 || table.sin.dim() != 2 ||
        table.cos.size(0) <= 0 || table.cos.size(0) > INT32_MAX || table.cos.size(1) != table.rotary_dim / 2 ||
        table.cos.size(0) != table.sin.size(0) || table.cos.size(1) != table.sin.size(1))
        throw std::invalid_argument("invalid Metal RoPE input");
    const auto tokens = input.size(-2), rows = input.numel() / input.size(-1) / tokens;
    int sections[3] = {};
    const bool has_sections = table.sections.defined() && table.sections.numel() != 0;
    if (grid && has_sections) {
        const auto* dense = table.sections.representation<DenseTensor>();
        if (!dense || !dense->is_cpu() || !dense->is_contiguous() || dense->scalar_type() != ScalarType::int64 || dense->numel() != 3)
            throw std::invalid_argument("RoPE sections must be CPU int64[3]");
        const auto* values = dense->data_ptr<std::int64_t>();
        for (int i = 0; i < 3; ++i) {
            if (values[i] < 0 || values[i] > table.rotary_dim / 2) throw std::invalid_argument("invalid RoPE section");
            sections[i] = int(values[i]);
        }
        if (sections[0] + sections[1] + sections[2] != table.rotary_dim / 2)
            throw std::invalid_argument("RoPE sections do not match rotary dimension");
    }
    const int axes = positions.dim() == 1 ? 1 : int(positions.size(0));
    const int batches = !(grid && has_sections) && positions.dim() == 2 ? axes : 0;
    if (axes <= 0 || (grid && has_sections && axes != 3) || (batches && rows % batches))
        throw std::invalid_argument("invalid RoPE positions");
    const bool interleaved = grid && table.interleaved_order.defined();
    if (interleaved && (!has_sections || !table.interleaved_inverse.defined() ||
        table.interleaved_inverse.device() != gpu || table.interleaved_inverse.scalar_type() != ScalarType::int64 ||
        table.interleaved_inverse.numel() != table.rotary_dim / 2))
        throw std::invalid_argument("invalid RoPE interleaving");
    auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
    auto pos = metal::array(positions.device() == gpu ? positions : metal::copy(positions, gpu));
    auto x = mlx::core::contiguous(mlx::core::astype(metal::array(input), bf16 ? mlx::core::bfloat16 : mlx::core::float32, stream), stream);
    pos = mlx::core::contiguous(mlx::core::astype(pos, mlx::core::int64, stream), stream);
    auto inverse = interleaved ? mlx::core::contiguous(metal::array(table.interleaved_inverse), stream) : pos;
    auto cosine = mlx::core::contiguous(metal::array(table.cos), stream);
    auto sine = mlx::core::contiguous(metal::array(table.sin), stream);
    auto output = kernel()({x, pos, cosine, sine, inverse}, {x.shape()}, {x.dtype()},
        {int(input.numel()), 1, 1}, {std::min(256, int(input.numel())), 1, 1},
        {{"T", x.dtype()}, {"SIZE", int(input.numel())}, {"DIM", int(input.size(-1))},
         {"ROTARY", int(table.rotary_dim)}, {"TOKENS", int(tokens)}, {"AXES", axes},
         {"BATCHES", batches}, {"ROWS_PER_BATCH", int(batches ? rows / batches : rows)},
         {"TABLE_LEN", int(table.cos.size(0))}, {"GRID", int(grid && has_sections)},
         {"S0", sections[0]}, {"S1", sections[1]}, {"INTERLEAVED", int(interleaved)}, {"BF16", int(bf16)}},
        std::nullopt, false, stream);
    return metal::tensor(std::move(output.front()));
}
using ::tyloi::ops::RopeTable;
using ::tyloi::ops::YarnScaling;
namespace mx = mlx::core;

namespace {
RopeTable build_rope(std::int64_t positions, std::int64_t dimension, double base, Device device,
                     std::int64_t frequency_dim, std::int64_t active_pairs,
                     bool official, YarnScaling yarn) {
    if (!device.is_metal() || positions <= 0 || dimension <= 0 || dimension % 2 ||
        !std::isfinite(base) || base <= 0 || active_pairs > dimension / 2)
        throw std::invalid_argument("invalid Metal RoPE table parameters");
    const auto denominator = frequency_dim > 0 ? frequency_dim : dimension;
    const auto half = dimension / 2;
    if (active_pairs < 0) active_pairs = half;
    if (yarn.enabled() && (yarn.original_context <= 0 || !std::isfinite(yarn.factor) ||
        !std::isfinite(yarn.beta_fast) || !std::isfinite(yarn.beta_slow) || yarn.beta_slow <= 0 ||
        yarn.beta_fast <= yarn.beta_slow || base <= 1))
        throw std::invalid_argument("invalid Qwen YaRN scaling");
    double low = 0, high = 1;
    if (yarn.enabled()) {
        const auto correction = [&](double rotations) {
            return denominator * std::log(yarn.original_context / (rotations * 2 * std::acos(-1.0))) /
                   (2 * std::log(base));
        };
        low = std::max(0.0, std::floor(correction(yarn.beta_fast)));
        high = std::min(double(denominator - 1), std::ceil(correction(yarn.beta_slow)));
        if (low == high) high += 0.001;
    }
    std::vector<float> frequencies(static_cast<size_t>(half));
    for (std::int64_t i = 0; i < half; ++i) {
        double f = official ? 1.0f / std::pow(float(base), float(2 * i) / float(denominator))
                            : std::pow(base, -2.0 * i / denominator);
        if (yarn.enabled()) {
            auto ramp = std::clamp((i - low) / (high - low), 0.0, 1.0);
            f = std::pow(base, -2.0 * i / denominator) * (1 - ramp + ramp / yarn.factor);
        }
        frequencies[static_cast<size_t>(i)] = i < active_pairs ? float(f) : 0.0f;
    }
    auto freq = metal::copy_from_host(frequencies.data(), std::array<std::int64_t, 1>{half}, kFloat32);
    auto pos = mx::arange(0, static_cast<int>(positions), 1, mx::float32);
    auto angle = mx::expand_dims(pos, 1) * mx::expand_dims(metal::array(freq), 0);
    const float factor = yarn.enabled() ? float(1 + 0.1 * std::log(yarn.factor)) : 1.0f;
    RopeTable result;
    result.cos = metal::tensor(mx::cos(angle) * factor);
    result.sin = metal::tensor(mx::sin(angle) * factor);
    result.rotary_dim = dimension;
    return result;
}
}
RopeTable make_rope(std::int64_t positions, std::int64_t dimension, double base, Device device,
                    std::span<const std::int64_t> sections, bool interleaved,
                    bool official, YarnScaling yarn) {
    auto result = build_rope(positions, dimension, base, device, 0, -1, official, yarn);
    if (sections.empty()) return result;
    if (sections.size() != 3 || std::accumulate(sections.begin(), sections.end(), std::int64_t{}) != dimension / 2 ||
        std::any_of(sections.begin(), sections.end(), [](auto n) { return n < 0; }))
        throw std::invalid_argument("invalid multi-axis RoPE sections");
    std::vector<std::int64_t> execution(sections.begin(), sections.end());
    if (interleaved) {
        std::array<std::vector<std::int64_t>, 3> pairs;
        for (std::int64_t pair = 0; pair < dimension / 2; ++pair) {
            const auto residue = pair % 3;
            const int axis = residue == 1 && pair < sections[1] * 3 ? 1 :
                             residue == 2 && pair < sections[2] * 3 ? 2 : 0;
            pairs[axis].push_back(pair);
        }
        std::vector<std::int64_t> order;
        for (size_t i = 0; i < pairs.size(); ++i) {
            execution[i] = pairs[i].size();
            order.insert(order.end(), pairs[i].begin(), pairs[i].end());
        }
        std::vector<std::int64_t> inverse(order.size());
        for (size_t i = 0; i < order.size(); ++i) inverse[order[i]] = i;
        result.interleaved_order = ::tyloi::ops::tensor(std::span<const std::int64_t>(order), device);
        result.interleaved_inverse = ::tyloi::ops::tensor(std::span<const std::int64_t>(inverse), device);
    }
    result.sections = ::tyloi::ops::tensor(std::span<const std::int64_t>(execution), Device{DeviceType::cpu});
    return result;
}

}

#include "mlx_tensor.h"
#include "mlx_moe_ops.h"
#include "tyloinfer/channel_fp8.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

namespace mx = mlx::core;
namespace fp8 = mfq::channel_fp8;

float round_e4m3(float x) {
    const auto magnitude = std::abs(x);
    int nearest = 0;
    for (int code = 1; code <= 126; ++code) {
        const float candidate = std::abs(fp8::decode(static_cast<std::uint8_t>(code)) - magnitude);
        const float current = std::abs(fp8::decode(static_cast<std::uint8_t>(nearest)) - magnitude);
        if (candidate < current || (candidate == current && code % 2 == 0))
            nearest = code;
    }
    return std::copysign(fp8::decode(static_cast<std::uint8_t>(nearest)), x);
}

void check_fused_quantize() {
    constexpr int tokens = 65, width = 5120;
    std::vector<float> samples(tokens * width);
    for (int row = 0; row < tokens; ++row) {
        for (int col = 0; col < width; ++col) {
            const auto magnitude = row % 3 == 0 ? 448.f : (1.f + row) * .09375f;
            samples[row * width + col] = std::sin((row + 1) * (col + 3) * .00713f) * magnitude;
        }
        if (row % 3 == 0) {
            samples[row * width] = 448.f;
            samples[row * width + 1] = 1.0625f;
            samples[row * width + 2] = -1.1875f;
            samples[row * width + 3] = .0009765625f;
        }
    }
    std::fill(samples.end() - width, samples.end(), 0.f);
    const auto source = mx::astype(mx::array(samples.data(), {tokens, width}), mx::bfloat16);
    const auto actual = mfq::metal::MlxChannelFp8Weight::quantize(source);
    const auto x = mx::astype(source, mx::float32);
    const auto scales = mx::maximum(mx::max(mx::abs(x), -1, true), mx::array(1e-12f)) / mx::array(448.f);
    const auto scaled = mx::clip(x / scales, mx::array(-448.f), mx::array(448.f));
    const auto expected = mx::from_fp8(mx::to_fp8(scaled), mx::bfloat16);
    if (!mx::all(actual.values == expected).item<bool>() ||
        !mx::all(mx::abs(actual.scales - scales) <= mx::abs(scales) * 1e-7f).item<bool>())
        throw std::runtime_error("fused channel FP8 rounding differs from MLX E4M3");
}

void check_channel_fp8(bool dynamic) {
    constexpr int width = 128, outputs = 64;
    const auto layout = fp8::layout(outputs, width, fp8::ScaleKind::Bf16, dynamic);
    const auto prefix = fp8::prefix(layout);
    std::vector<std::uint8_t> blob(layout.bytes);
    std::memcpy(blob.data(), prefix.data(), prefix.size());
    auto* values = blob.data() + fp8::header_bytes;
    std::vector<float> scales(outputs);
    for (int row = 0; row < outputs; ++row) {
        for (int column = 0; column < width; ++column) {
            const int index = row * width + column;
            values[index] = static_cast<std::uint8_t>(24 + (index * 37) % 103 + (index % 2) * 128);
        }
        const auto bits = static_cast<std::uint16_t>(0x3b80 + (row % 4) * 32);
        blob[layout.scales + 2 * row] = static_cast<std::uint8_t>(bits & 255);
        blob[layout.scales + 2 * row + 1] = static_cast<std::uint8_t>(bits >> 8);
        scales[row] = (1.f + (row % 4) * .25f) / 256.f;
    }

    const auto weight = mfq::metal::MlxChannelFp8Weight::from_blob(blob);
    for (const auto dtype : {mx::float32, mx::bfloat16}) {
        for (const int tokens : {1, 7, 65}) {
            std::vector<float> input(tokens * width);
            for (int i = 0; i < tokens * width; ++i)
                input[i] = std::sin(i * .0313f) * std::pow(0.5f, (i / width) % 5);
            if (tokens > 1) std::fill(input.end() - width, input.end(), 0.f);

            const auto source = mx::astype(mx::array(input.data(), {1, tokens, width}), dtype);
            auto reference_input = mx::astype(source, mx::float32);
            auto actual = mx::astype(weight.matmul(source), mx::float32);
            mx::eval(reference_input, actual);
            const auto relative_tolerance = dtype == mx::bfloat16 ? .009 : .001;
            for (int token = 0; token < tokens; ++token) {
                const auto* row = reference_input.data<float>() + token * width;
                float scale = 1.f;
                std::vector<float> quantized(row, row + width);
                if (dynamic) {
                    float maximum = 1e-12f;
                    for (const auto v : quantized) maximum = std::max(maximum, std::abs(v));
                    scale = maximum / 448.f;
                    for (auto& v : quantized) v = round_e4m3(v / scale);
                }
                for (int out = 0; out < outputs; ++out) {
                    double sum = 0.0;
                    for (int k = 0; k < width; ++k)
                        sum += double(quantized[k]) * fp8::decode(values[out * width + k]);
                    const auto expected = sum * scale * scales[out];
                    const auto value = actual.data<float>()[token * outputs + out];
                    if (!std::isfinite(value) || std::abs(expected - value) > .001 + std::abs(expected) * relative_tolerance)
                        throw std::runtime_error("Metal channel FP8 projection differs from the E4M3 reference");
                }
            }
        }
    }

    if (dynamic) {
        constexpr int tokens = 65;
        const auto input = mx::astype(
            mx::reshape(mx::sin(mx::arange(tokens * width, mx::float32) * .011f), {1, tokens, width}),
            mx::bfloat16);
        const auto gate = weight.matmul(input);
        blob[layout.scales] ^= 32;
        const auto up = mfq::metal::MlxChannelFp8Weight::from_blob(blob);
        const auto expected = mfq::metal::moe_swiglu_pair(gate, up.matmul(input));
        const auto fused = weight.fused_swiglu(up, input);
        if (!fused || !mx::all(*fused == expected).item<bool>())
            throw std::runtime_error("fused channel FP8 SwiGLU changed output bits");
        blob[layout.scales] ^= 32;
    }

    auto rows = weight.embedding(mx::array({1, outputs - 1}, mx::int32), mx::float32);
    rows.eval();
    for (int k = 0; k < width; ++k)
        if (rows.data<float>()[k] != fp8::decode(values[width + k]) * scales[1] ||
            rows.data<float>()[width + k] != fp8::decode(values[(outputs - 1) * width + k]) * scales.back())
            throw std::runtime_error("Metal channel FP8 row lookup differs from the E4M3 reference");
}

} // namespace

int main() {
    mx::set_default_device(mx::Device::gpu);
    check_fused_quantize();
    check_channel_fp8(false);
    check_channel_fp8(true);
}

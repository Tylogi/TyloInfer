#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace mfq::channel_fp8 {
inline constexpr std::string_view dtype = "FP8-E4M3-CHANNEL";
inline constexpr std::size_t header_bytes = 24;
enum class ScaleKind : std::uint8_t { Bf16 = 1, F16 = 2, F32 = 3 };
struct Layout {
    std::int64_t outputs = 0, width = 0;
    ScaleKind scale_kind = ScaleKind::Bf16;
    bool dynamic_activations = false;
    std::size_t scales = 0, bytes = 0;
};
inline std::size_t scale_bytes(ScaleKind kind) {
    switch (kind) {
        case ScaleKind::Bf16: case ScaleKind::F16: return 2;
        case ScaleKind::F32: return 4;
    }
    throw std::invalid_argument("invalid channel FP8 scale dtype");
}
inline Layout layout(std::int64_t outputs, std::int64_t width, ScaleKind kind, bool dynamic) {
    const auto item = scale_bytes(kind);
    if (outputs <= 0 || width <= 0 || outputs > INT32_MAX || width > INT32_MAX ||
        std::uint64_t(outputs) > (std::numeric_limits<std::size_t>::max() - header_bytes) /
            (std::uint64_t(width) + item))
        throw std::invalid_argument("invalid channel FP8 shape");
    const auto offset = header_bytes + std::size_t(outputs) * width;
    return {outputs, width, kind, dynamic, offset, offset + std::size_t(outputs) * item};
}
inline std::vector<std::byte> prefix(const Layout& value) {
    std::vector<std::byte> result(header_bytes);
    std::memcpy(result.data(), "F8CH", 4);
    result[4] = std::byte{1}; result[5] = std::byte(value.dynamic_activations);
    result[6] = std::byte(value.scale_kind);
    for (int i = 0; i < 8; ++i) {
        result[8+i] = std::byte(std::uint64_t(value.outputs) >> (8*i));
        result[16+i] = std::byte(std::uint64_t(value.width) >> (8*i));
    }
    return result;
}
inline Layout parse(const std::byte* data, std::size_t bytes) {
    if (!data || bytes < header_bytes || std::memcmp(data, "F8CH", 4) ||
        data[4] != std::byte{1} || std::to_integer<unsigned>(data[5]) > 1 || data[7] != std::byte{})
        throw std::invalid_argument("invalid channel FP8 header");
    std::uint64_t outputs = 0, width = 0;
    for (int i = 0; i < 8; ++i) {
        outputs |= std::uint64_t(std::to_integer<unsigned>(data[8+i])) << (8*i);
        width |= std::uint64_t(std::to_integer<unsigned>(data[16+i])) << (8*i);
    }
    if (outputs > INT32_MAX || width > INT32_MAX)
        throw std::invalid_argument("channel FP8 shape exceeds the execution ABI");
    auto result = layout(std::int64_t(outputs), std::int64_t(width),
        static_cast<ScaleKind>(data[6]), data[5] != std::byte{});
    if (bytes != result.bytes) throw std::invalid_argument("channel FP8 payload size mismatch");
    return result;
}
inline float decode(std::uint8_t code) {
    const auto magnitude = code & 127u;
    if (magnitude == 127u) return std::numeric_limits<float>::quiet_NaN();
    const int exponent = magnitude >> 3, mantissa = magnitude & 7;
    const float value = exponent ? std::ldexp(1.f + float(mantissa) / 8.f, exponent - 7)
        : std::ldexp(float(mantissa), -9);
    return code & 128u ? -value : value;
}
} // namespace mfq::channel_fp8

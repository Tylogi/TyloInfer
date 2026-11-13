#include "rms.h"
#include "engine/backends/metal/native/tensor/tensor.h"

#include <mlx/mlx.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>

namespace tyloi::metal::ops {
namespace {
using mlx::core::array;
using mlx::core::Shape;
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

} // namespace

Tensor rms_norm(const Tensor& input, const Tensor& weight, double eps, double offset, ScalarType output_type) {
    const Device device{DeviceType::metal, 0};
    if (!input.defined() || !weight.defined() || input.is_encoded() || weight.is_encoded() ||
        input.device() != device || weight.device() != device || input.dim() == 0 || weight.dim() == 0 ||
        input.size(-1) <= 0 || weight.numel() != input.size(-1) || weight.size(-1) != input.size(-1))
        throw std::invalid_argument("rms_norm input and weight are incompatible");
    if (output_type != ScalarType::float16 && output_type != ScalarType::bfloat16 && output_type != ScalarType::float32)
        throw std::invalid_argument("unsupported rms_norm output type");
    const float epsilon = static_cast<float>(eps), weight_offset = static_cast<float>(offset);
    if (!std::isfinite(eps) || !std::isfinite(epsilon) || epsilon <= 0 ||
        !std::isfinite(offset) || !std::isfinite(weight_offset))
        throw std::invalid_argument("invalid RMSNorm parameters");
    if (input.size(-1) > std::numeric_limits<int>::max()) throw std::overflow_error("RMSNorm width exceeds int32");
    const int width = static_cast<int>(input.size(-1));
    const auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
    auto source = metal::array(input);
    auto scale = mlx::core::add(mlx::core::contiguous(mlx::core::astype(
        mlx::core::reshape(metal::array(weight), Shape{width}, stream), mlx::core::float32, stream), stream),
        array(weight_offset), stream);
    if (source.dtype() != mlx::core::float16 && source.dtype() != mlx::core::bfloat16 && source.dtype() != mlx::core::float32)
        source = mlx::core::astype(source, mlx::core::float16, stream);
    const auto result_dtype = metal::dtype(output_type);
    static const bool fused_enabled = [] {
        const auto* setting = std::getenv("TYLOINFER_METAL_FUSED_RMS_CAST");
        return setting == nullptr || std::strcmp(setting, "0") != 0;
    }();
    const auto rows = source.size() / std::size_t(width);
    if (fused_enabled && source.dtype() == mlx::core::float16 && result_dtype != mlx::core::float32 &&
        width >= 1024 && width <= 8192 && width % 128 == 0 && rows >= 1 && rows <= 8) {
        const int threads = std::min(width / 4, 1024);
        auto contiguous = mlx::core::contiguous(mlx::core::reshape(source, Shape{int(rows), width}, stream), stream);
        return metal::tensor(fused_fp16_rms_norm_kernel()(
            {contiguous, scale}, {source.shape()}, {result_dtype}, {threads * int(rows), 1, 1},
            {threads, 1, 1}, {{"WIDTH", width}, {"THREADS", threads}, {"EPS_BITS", std::bit_cast<std::int32_t>(epsilon)},
            {"OUTPUT_BF16", result_dtype == mlx::core::bfloat16}}, std::nullopt, false, stream).front());
    }
    auto normalized = mlx::core::fast::rms_norm(source, std::optional<array>(scale), epsilon, stream);
    auto rounded = normalized.dtype() == source.dtype() ? normalized : mlx::core::astype(normalized, source.dtype(), stream);
    return metal::tensor(rounded.dtype() == result_dtype ? rounded : mlx::core::astype(rounded, result_dtype, stream));
}
} // namespace tyloi::metal::ops

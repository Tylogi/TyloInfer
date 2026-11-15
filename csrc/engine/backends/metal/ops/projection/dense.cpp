#include "dense.h"
#include "engine/backends/metal/native/tensor/tensor.h"

#include <mlx/mlx.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::metal::ops {
namespace {
using mlx::core::array;
using mlx::core::Shape;
using mlx::core::CompileOptions;
using mlx::core::MathMode;

const Device gpu{DeviceType::metal, 0};
const auto stream = [] { return mlx::core::default_stream(mlx::core::Device::gpu); };

// Decode verification presents two through six hidden states at once.  MLX's
// general GEMM path does not reuse a dense weight row efficiently at this M,
// so one SIMD group owns an output and accumulates every input row while the
// weight vector is resident in registers.
constexpr const char* kDenseSmallM = R"METAL(
    constexpr uint K_LANES_VALUE = uint(K_LANES);
    constexpr uint SIMD_GROUPS_VALUE = uint(SIMD_GROUPS);
    constexpr uint OUTPUTS_PER_SIMD = 32u / K_LANES_VALUE;
    constexpr uint OUTPUTS_PER_TG =
        SIMD_GROUPS_VALUE * OUTPUTS_PER_SIMD;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES_VALUE - 1u);
    uint simd_output = lane / K_LANES_VALUE;
    uint output_index =
        threadgroup_position_in_grid.y * OUTPUTS_PER_TG
        + simd_group * OUTPUTS_PER_SIMD + simd_output;
    uint output = min(output_index, uint(OUT) - 1u);

    float accumulators[M];
    for (uint row = 0u; row < uint(M); ++row) {
        accumulators[row] = 0.0f;
    }

    uint weight_base = output * uint(K);
    for (uint vector = k_lane;
         vector < uint(K) / 4u;
         vector += K_LANES_VALUE) {
        uint column = vector * 4u;
        vec<T, 4> packed_weight = *(device const vec<T, 4>*)(
            weight + weight_base + column);
        float4 weight_values = float4(packed_weight);
        for (uint row = 0u; row < uint(M); ++row) {
            vec<T, 4> packed_input = *(device const vec<T, 4>*)(
                x + row * uint(K) + column);
            accumulators[row] += dot(float4(packed_input), weight_values);
        }
    }

    for (uint row = 0u; row < uint(M); ++row) {
        if (K_LANES_VALUE >= 32u) {
            accumulators[row] += simd_shuffle_down(accumulators[row], 16u);
        }
        if (K_LANES_VALUE >= 16u) {
            accumulators[row] += simd_shuffle_down(accumulators[row], 8u);
        }
        if (K_LANES_VALUE >= 8u) {
            accumulators[row] += simd_shuffle_down(accumulators[row], 4u);
        }
        accumulators[row] += simd_shuffle_down(accumulators[row], 2u);
        accumulators[row] += simd_shuffle_down(accumulators[row], 1u);
        if (k_lane == 0u && output_index < uint(OUT)) {
            y[row * uint(OUT) + output_index] = T(accumulators[row]);
        }
    }
)METAL";

// M=2..6 verifier GEMV with the same 32-lane reduction geometry as MLX's
// one-token large-output GEMV. Four output rows share every activation load
// inside each SIMD group, while each weight is reused across all M inputs.
//
// Scheduling derived from oMLX 0.6.4 verify_qmv.py.
// Copyright © 2026 Apple Inc.  Licensed under Apache-2.0.
constexpr const char* kDenseSmallMExact = R"METAL(
    constexpr uint OUTPUTS_PER_SIMD = 4u;
    constexpr uint SIMD_GROUPS = 8u;
    constexpr uint OUTPUTS_PER_TG = OUTPUTS_PER_SIMD * SIMD_GROUPS;
    constexpr uint VALUES_PER_LANE = 4u;
    constexpr uint K_BLOCK = VALUES_PER_LANE * 32u;

    uint simd_group = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint output_base =
        threadgroup_position_in_grid.x * OUTPUTS_PER_TG
        + simd_group * OUTPUTS_PER_SIMD;
    if (output_base >= uint(OUT)) {
        return;
    }

    uint k_base = lane * VALUES_PER_LANE;
    float accum[M][OUTPUTS_PER_SIMD] = {{0.0f}};
    for (uint block = 0u; block < uint(K); block += K_BLOCK) {
        float input_values[M][VALUES_PER_LANE];
        for (uint row = 0u; row < uint(M); ++row) {
            for (uint column = 0u;
                 column < VALUES_PER_LANE;
                 ++column) {
                input_values[row][column] =
                    float(x[row * uint(K) + k_base + column]);
            }
        }
        for (uint result = 0u;
             result < OUTPUTS_PER_SIMD;
             ++result) {
            device const T* row_weight =
                weight + (output_base + result) * uint(K) + k_base;
            float weight_values[VALUES_PER_LANE];
            for (uint column = 0u;
                 column < VALUES_PER_LANE;
                 ++column) {
                weight_values[column] = float(row_weight[column]);
            }
            for (uint row = 0u; row < uint(M); ++row) {
                for (uint column = 0u;
                     column < VALUES_PER_LANE;
                     ++column) {
                    accum[row][result] +=
                        weight_values[column] * input_values[row][column];
                }
            }
        }
        k_base += K_BLOCK;
    }

    for (uint row = 0u; row < uint(M); ++row) {
        for (uint result = 0u;
             result < OUTPUTS_PER_SIMD;
             ++result) {
            for (ushort step = 16u; step >= 1u; step >>= 1u) {
                accum[row][result] +=
                    simd_shuffle_down(accum[row][result], step);
            }
            if (lane == 0u) {
                y[row * uint(OUT) + output_base + result] =
                    T(accum[row][result]);
            }
        }
    }
)METAL";

const mlx::core::fast::CustomKernelFunction& dense_small_m_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_dense_small_m_m2_6",
            {"weight", "x"},
            {"y"},
            kDenseSmallM,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
dense_small_m_exact_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_dense_small_m_exact_m2_6",
            {"weight", "x"},
            {"y"},
            kDenseSmallMExact,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

array dense_small_m_matmul(
    const array& weight,
    const array& input,
    int rows,
    int input_size,
    int output_size) {
    const auto* layout = std::getenv(
        "TYLOINFER_METAL_DENSE_SMALL_M_LAYOUT");
    const bool exact =
        input_size % 128 == 0 &&
        output_size >= 4096 &&
        output_size % 32 == 0 &&
        (layout == nullptr || std::strcmp(layout, "legacy") != 0);
    if (exact) {
        auto source = mlx::core::reshape(
            input.flags().row_contiguous
                ? input
                : mlx::core::contiguous(input, stream()),
            Shape{rows, input_size}, stream());
        auto outputs = dense_small_m_exact_kernel()(
            {weight, std::move(source)},
            {Shape{rows, output_size}},
            {weight.dtype()},
            {
                ((output_size + 31) / 32) * 256,
                1,
                1,
            },
            {256, 1, 1},
            {
                {"T", weight.dtype()},
                {"M", rows},
                {"K", input_size},
                {"OUT", output_size},
            },
            std::nullopt,
            false,
            stream());
        Shape output_shape = input.shape();
        output_shape.back() = output_size;
        return mlx::core::reshape(
            std::move(outputs.front()),
            std::move(output_shape), stream());
    }
    constexpr int simd_groups = 4;
    const int k_lanes = input_size >= 16384 ? 16 : 32;
    const int outputs_per_threadgroup = simd_groups * 32 / k_lanes;
    auto source = mlx::core::reshape(
        input.flags().row_contiguous ? input : mlx::core::contiguous(input, stream()),
        Shape{rows, input_size}, stream());
    auto outputs = dense_small_m_kernel()(
        {weight, std::move(source)},
        {Shape{rows, output_size}},
        {weight.dtype()},
        {
            simd_groups * 32,
            (output_size + outputs_per_threadgroup - 1) /
                outputs_per_threadgroup,
            1,
        },
        {simd_groups * 32, 1, 1},
        {
            {"T", weight.dtype()},
            {"M", rows},
            {"K", input_size},
            {"OUT", output_size},
            {"K_LANES", k_lanes},
            {"SIMD_GROUPS", simd_groups},
        },
        std::nullopt,
        false,
        stream());
    Shape output_shape = input.shape();
    output_shape.back() = output_size;
    return mlx::core::reshape(
        std::move(outputs.front()),
        std::move(output_shape), stream());
}

array dense_decode_consistent_matmul(
    const array& weight,
    const array& input,
    int rows,
    int input_size,
    int output_size) {
    // This is the same fallback used by oMLX's decode-consistency patch:
    // express M independent projections as a batch of matrix-vector
    // products. MLX consequently keeps its M=1 GEMV reduction for every row
    // instead of selecting a width-M GEMM kernel.
    auto source = mlx::core::reshape(
        input.flags().row_contiguous
            ? input
            : mlx::core::contiguous(input, stream()),
        Shape{rows, input_size}, stream());
    auto projected = mlx::core::matmul(
        weight,
        mlx::core::expand_dims(std::move(source), -1, stream()), stream());
    projected = mlx::core::squeeze(std::move(projected), -1, stream());
    Shape output_shape = input.shape();
    output_shape.back() = output_size;
    return mlx::core::reshape(
        std::move(projected),
        std::move(output_shape), stream());
}

// Adapted from oMLX's Qwen router GEMV topology (Apache-2.0): spread one
// router row over each SIMD group instead of packing a narrow 512-row GEMV
// into the stock matmul schedule. MFQ supplies ordinary dense FP16/BF16
// weights here; no external quantized layout is part of this ABI.
constexpr const char* kDenseRouterLogitsSource = R"METAL(
    constexpr uint SIMD_GROUPS = 4u;
    uint lane = thread_index_in_simdgroup;
    uint row = threadgroup_position_in_grid.y * SIMD_GROUPS
        + simdgroup_index_in_threadgroup;
    if (row >= uint(EXPERTS)) {
        return;
    }

    float accumulator = 0.0f;
    device const activation_t* row_weight =
        weight + row * uint(K);
    for (uint column = lane * 4u;
         column < uint(K);
         column += 128u) {
        activation4_t activation =
            *(device const activation4_t*)(input + column);
        activation4_t values =
            *(device const activation4_t*)(row_weight + column);
        accumulator = fma(float(values[0]), float(activation[0]), accumulator);
        accumulator = fma(float(values[1]), float(activation[1]), accumulator);
        accumulator = fma(float(values[2]), float(activation[2]), accumulator);
        accumulator = fma(float(values[3]), float(activation[3]), accumulator);
    }
    accumulator = simd_sum(accumulator);
    if (lane == 0u) {
        output[row] = activation_t(accumulator);
    }
)METAL";

mlx::core::fast::CustomKernelFunction make_router_kernel(
    const char* name, std::string source) {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(name, {"input", "weight"}, {"output"},
        std::move(source), "", true, false, options);
}

const mlx::core::fast::CustomKernelFunction&
dense_router_logits_kernel(mlx::core::Dtype dtype) {
    static const auto fp16_kernel = make_router_kernel(
        "mfq_cpp_moe_dense_router_logits_f16",
        std::string(
            "using activation_t = half;\n"
            "using activation4_t = half4;\n") +
            kDenseRouterLogitsSource);
    static const auto bf16_kernel = make_router_kernel(
        "mfq_cpp_moe_dense_router_logits_bf16",
        std::string(
            "using activation_t = bfloat;\n"
            "using activation4_t = bfloat4;\n") +
            kDenseRouterLogitsSource);
    return dtype == mlx::core::bfloat16 ? bf16_kernel : fp16_kernel;
}

bool moe_dense_router_logits_supported(
    const array& input,
    const array& weight) noexcept {
    const bool dense16 = weight.dtype() == mlx::core::float16 ||
        weight.dtype() == mlx::core::bfloat16;
    if (!dense16 || weight.ndim() != 2 ||
        !weight.flags().row_contiguous || input.ndim() == 0) {
        return false;
    }
    const int experts = weight.shape(0);
    const int width = weight.shape(1);
    return (experts == 256 || experts == 384 || experts == 512) &&
        width > 64 && width < 16 * experts && width % 128 == 0 &&
        input.shape(-1) == width &&
        input.size() == static_cast<std::size_t>(width);
}

array moe_dense_router_logits(
    const array& input,
    const array& weight) {
    if (!moe_dense_router_logits_supported(input, weight)) {
        throw std::invalid_argument(
            "dense router GEMV requires one row, 256/384/512 experts, "
            "and contiguous FP16/BF16 weights");
    }
    auto source = mlx::core::contiguous(mlx::core::reshape(
        input.dtype() == weight.dtype()
            ? input
            : mlx::core::astype(input, weight.dtype(), stream()),
        Shape{weight.shape(1)}, stream()), stream());
    const int experts = weight.shape(0);
    auto outputs = dense_router_logits_kernel(weight.dtype())(
        {std::move(source), weight},
        {Shape{1, experts}},
        {weight.dtype()},
        {32, experts, 1},
        {32, 4, 1},
        {
            {"K", weight.shape(1)},
            {"EXPERTS", experts},
        },
        std::nullopt,
        false,
        stream());
    return std::move(outputs.front());
}


} // namespace

Tensor dense_project(const Tensor& input, const Tensor& weight) {
    const auto floating = [](ScalarType type) {
        return type == ScalarType::float16 || type == ScalarType::bfloat16 || type == ScalarType::float32;
    };
    if (!input.defined() || input.is_encoded() || input.device() != gpu || input.dim() == 0 ||
        !weight.defined() || weight.is_encoded() || weight.device() != gpu || weight.dim() != 2 ||
        weight.size(0) <= 0 || weight.size(1) <= 0 || input.size(-1) != weight.size(1) ||
        !floating(input.scalar_type()) || !floating(weight.scalar_type()))
        throw std::invalid_argument("invalid Metal dense projection input");
    const auto& dense = metal::array(weight);
    const auto& original = metal::array(input);
    auto source = original.dtype() == dense.dtype() ? original : mlx::core::astype(original, dense.dtype(), stream());
    const int input_size = dense.shape(1), output_size = dense.shape(0);
    const auto rows = source.size() / static_cast<std::size_t>(input_size);
    if (rows == 1 && moe_dense_router_logits_supported(source, dense)) {
        auto shape = source.shape();
        shape.back() = output_size;
        return metal::tensor(mlx::core::reshape(moe_dense_router_logits(source, dense), std::move(shape), stream()));
    }
    if (rows >= 2 && rows <= 6 && input_size % 4 == 0 && dense.size() >= 65536 && dense.flags().row_contiguous) {
        const auto* layout = std::getenv("TYLOINFER_METAL_DENSE_SMALL_M_LAYOUT");
        const bool legacy = layout != nullptr && std::strcmp(layout, "legacy") == 0;
        if (!legacy && output_size < 4096)
            return metal::tensor(dense_decode_consistent_matmul(dense, source, int(rows), input_size, output_size));
        if (dense.dtype() == mlx::core::float16 || dense.dtype() == mlx::core::bfloat16)
            return metal::tensor(dense_small_m_matmul(dense, source, int(rows), input_size, output_size));
    }
    return metal::tensor(mlx::core::matmul(source, mlx::core::transpose(dense, {1, 0}, stream()), stream()));
}

Tensor dense_embedding(const Tensor& ids, const Tensor& weight) {
    if (!ids.defined() || ids.is_encoded() || (ids.device() != gpu && !ids.is_cpu()) ||
        (ids.scalar_type() != ScalarType::int32 && ids.scalar_type() != ScalarType::int64) ||
        !weight.defined() || weight.is_encoded() || weight.device() != gpu || weight.dim() != 2 ||
        weight.size(0) <= 0 || weight.size(1) <= 0)
        throw std::invalid_argument("invalid Metal dense embedding input");
    const auto selected = ids.device() == gpu ? ids : metal::copy(ids, gpu);
    return metal::tensor(mlx::core::take(metal::array(weight), metal::array(selected), 0, stream()));
}

} // namespace tyloi::metal::ops

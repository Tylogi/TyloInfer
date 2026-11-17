#include "engine/backends/cuda/ops/projection/channel_fp8.h"

#include "engine/backends/cuda/kernels/projection/prefill/lt.cuh"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/ops/projection/prefill.h"
#include "engine/weights/encoding/fp8/channel.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace tyloi::cuda {
namespace {

__device__ float decode(unsigned char code) {
    const unsigned magnitude = code & 127u;
    const unsigned sign = unsigned(code & 128u) << 24;
    if (magnitude == 127u) return __uint_as_float(sign | 0x7fc00000u);
    const float value = magnitude < 8u
        ? float(magnitude) * .001953125f
        : __uint_as_float((magnitude + 960u) << 20);
    return __uint_as_float(__float_as_uint(value) | sign);
}

template <class T>
__global__ void prepare_input(
        const float* x, T* codes, float* scales, int width, float operand_scale) {
    __shared__ float maxima[8];
    const auto row = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    float maximum = 0;
    for (int k = threadIdx.x; k < width; k += 256)
        maximum = fmaxf(maximum, fabsf(x[std::int64_t(row) * width + k]));
    for (int stride = 16; stride; stride /= 2)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, stride));
    if (!lane) maxima[warp] = maximum;
    __syncthreads();
    if (!warp) {
        maximum = lane < 8 ? maxima[lane] : 0.f;
        for (int stride = 16; stride; stride /= 2)
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, stride));
        if (!lane) maxima[0] = maximum;
    }
    __syncthreads();
    // A zero row uses unit scale. Clamp only to keep division finite for subnormals.
    const float scale = maxima[0] == 0
        ? 1.f : fmaxf(maxima[0] / 448.f, 1.17549435e-38f);
    if (threadIdx.x == 0) scales[row] = scale;
    for (int k = threadIdx.x; k < width; k += 256) {
        const auto index = std::int64_t(row) * width + k;
        codes[index] = T(decode(__nv_cvt_float_to_fp8(
            x[index] / scale, __NV_SATFINITE, __NV_E4M3)) * operand_scale);
    }
}

__global__ void project(
        const unsigned char* weights, const float* scales, const float* x,
        const float* token_scales, float* output, int width, int outputs) {
    const int row = blockIdx.x * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int token = blockIdx.y;
    if (row >= outputs) return;
    float sum = 0;
    for (int k = lane; k < width; k += 32)
        sum = fmaf(decode(weights[std::int64_t(row) * width + k]),
            x[std::int64_t(token) * width + k], sum);
    for (int delta = 16; delta; delta /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (lane == 0)
        output[std::int64_t(token) * outputs + row] =
            sum * scales[row] * (token_scales ? token_scales[token] : 1.f);
}

template <class T>
__global__ void decode_tile(
        const unsigned char* weights, T* output, std::int64_t count,
        float operand_scale) {
    for (auto i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += std::int64_t(blockDim.x) * gridDim.x)
        output[i] = T(decode(weights[i]) * operand_scale);
}

template <class Input, class Output>
__global__ void restore_scales(
        const Input* product, Output* output, const float* weights,
        const float* tokens, int columns, int rows, float product_scale) {
    for (int column = blockIdx.x * blockDim.x + threadIdx.x;
         column < columns; column += blockDim.x * gridDim.x) {
        const float weight = weights[column];
        for (int row = blockIdx.y; row < rows; row += gridDim.y) {
            const float token = tokens ? tokens[row] : 1.f;
            const auto i = std::int64_t(row) * columns + column;
            output[i] = Output((float(product[i]) * product_scale) * (weight * token));
        }
    }
}

__global__ void select_rows(
        const unsigned char* weights, const float* scales,
        const std::int64_t* ids, float* output, int width, int rows,
        std::int64_t count) {
    for (auto i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += std::int64_t(blockDim.x) * gridDim.x) {
        const auto row = ids[i / width];
        // Invalid indices must not read outside the encoded weight storage.
        output[i] = row >= 0 && row < rows
            ? decode(weights[row * width + i % width]) * scales[row]
            : nanf("");
    }
}

int blocks(std::int64_t count) {
    return int(std::min<std::int64_t>(65535, (count + 255) / 256));
}

dim3 restore_grid(std::int64_t columns, std::int64_t rows, int device) {
    static thread_local std::unordered_map<int, int> processors;
    auto found = processors.find(device);
    if (found == processors.end()) {
        int count = 0;
        TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(
            &count, cudaDevAttrMultiProcessorCount, device));
        found = processors.emplace(device, std::max(1, count)).first;
    }
    // Reuse each block across rows instead of launching a block per 256 values.
    const auto x = std::min<std::int64_t>(65535, (columns + 255) / 256);
    return dim3(unsigned(x), unsigned(std::min<std::int64_t>(
        rows, (found->second * 8 + x - 1) / x)));
}

int operand_exponent(std::int64_t width) {
    int bits = 0;
    for (auto remaining = std::uint64_t(width - 1); remaining; remaining >>= 1) ++bits;
    return 9 + std::max(0, (bits - 13) / 2);
}

bool fp16_accumulation(std::int64_t width) {
    const auto* accumulation = std::getenv("TYLOINFER_FP8_FP16_ACCUM");
    return operand_exponent(width) <= 15 &&
        !(accumulation && std::strcmp(accumulation, "0") == 0);
}

float rounded_fp8(float value) {
    const float magnitude = std::min(448.f, std::abs(value));
    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (tyloi::channel_fp8::decode(mid) < magnitude) lo = mid + 1;
        else hi = mid;
    }
    if (lo && (magnitude - tyloi::channel_fp8::decode(lo - 1) <
                   tyloi::channel_fp8::decode(lo) - magnitude ||
               (magnitude - tyloi::channel_fp8::decode(lo - 1) ==
                    tyloi::channel_fp8::decode(lo) - magnitude &&
                ((lo - 1) & 1) == 0))) --lo;
    return std::copysign(tyloi::channel_fp8::decode(lo), value);
}

template <class Measure>
auto measure(CudaProfiler* profiler, const char* name, Measure&& operation)
    -> decltype(operation()) {
    return profiler ? profiler->measure(name, std::forward<Measure>(operation))
                    : operation();
}

} // namespace

ChannelFp8Input prepare_channel_fp8_input(
        const DenseTensor& input, CudaProfiler* profiler) {
    if (!input.is_cuda() || input.dim() < 1 || input.size(-1) < 1 ||
        input.numel() / input.size(-1) <= 8 ||
        (input.scalar_type() != kFloat16 &&
         input.scalar_type() != kBFloat16 &&
         input.scalar_type() != kFloat32)) {
        throw std::invalid_argument(
            "shared FP8 input requires a CUDA floating prefill matrix");
    }
    const MfqCudaGuard guard(input.get_device());
    const auto stream = mfq_get_current_cuda_stream(input.get_device());
    const auto width = input.size(-1);
    const auto tokens = input.numel() / width;
    if (tokens > INT32_MAX || width > INT32_MAX)
        throw std::invalid_argument("shared FP8 input exceeds the kernel ABI");
    auto x = measure(profiler, "fp8.input_cast", [&] {
        return input.reshape({tokens, width}).to(kFloat32).contiguous();
    });
    ChannelFp8Input result{
        input,
        empty({tokens, width}, input.options().dtype(kFloat16)),
        empty({tokens}, input.options().dtype(kFloat32))};
    const auto scale = fp16_accumulation(width)
        ? std::ldexp(1.f, -operand_exponent(width)) : 1.f;
    for (const auto& tensor : {input, x, result.values, result.scales})
        mfq_cuda_record_stream(tensor, stream);
    measure(profiler, "fp8.input_quant", [&] {
        prepare_input<<<unsigned(tokens), 256, 0, stream.stream()>>>(
            x.data_ptr<float>(), result.values.data_ptr<__half>(),
            result.scales.data_ptr<float>(), int(width), scale);
        return 0;
    });
    TYLOINFER_CUDA_CHECK(cudaGetLastError());
    return result;
}

DenseTensor channel_fp8_projection(
        const ChannelFp8Tensor& weight, DenseTensor input,
        CudaProfiler* profiler, const ChannelFp8Input* prepared) {
    const auto measure_operation = [profiler](const char* name, auto&& operation) {
        return measure(profiler, name, std::forward<decltype(operation)>(operation));
    };
    if (input.dim() < 1 || input.size(-1) != weight.neuron_len ||
        input.device() != weight.values.device() ||
        (input.scalar_type() != kFloat16 &&
         input.scalar_type() != kBFloat16 &&
         input.scalar_type() != kFloat32)) {
        throw std::invalid_argument("channel FP8 input shape, dtype or device mismatch");
    }
    const auto dtype = input.scalar_type();
    auto shape = input.sizes().vec();
    shape.back() = weight.out;
    if (prepared &&
        (!weight.dynamic_activations || !input.is_cuda() ||
         !prepared->source.defined() ||
         prepared->source.data_ptr() != input.data_ptr() ||
         prepared->source.sizes() != input.sizes() ||
         prepared->source.device() != input.device() ||
         prepared->source.scalar_type() != input.scalar_type() ||
         !prepared->values.defined() || !prepared->scales.defined() ||
         prepared->values.device() != input.device() ||
         prepared->scales.device() != input.device() ||
         prepared->values.scalar_type() != kFloat16 ||
         prepared->scales.scalar_type() != kFloat32 ||
         !prepared->values.is_contiguous() ||
         !prepared->scales.is_contiguous() || prepared->values.dim() != 2 ||
         prepared->values.size(1) != weight.neuron_len ||
         prepared->values.numel() != input.numel() ||
         prepared->scales.numel() != input.numel() / weight.neuron_len)) {
        throw std::invalid_argument("shared FP8 input does not match its source tensor");
    }
    auto x = prepared ? prepared->values : measure_operation("fp8.input_cast", [&] {
        return input.reshape({-1, weight.neuron_len}).to(kFloat32).contiguous();
    });
    const auto tokens = x.size(0);
    if (tokens > INT32_MAX)
        throw std::invalid_argument("channel FP8 token count exceeds the GEMM ABI");
    // Scale exact FP8 codes by powers of two so an all-positive dot remains
    // below 2^14. Exponents through 15 retain the smallest FP8 code.
    const int exponent = operand_exponent(weight.neuron_len);
    const bool half_accum = weight.values.is_cuda() &&
        weight.dynamic_activations && tokens > 8 &&
        fp16_accumulation(weight.neuron_len);
    const float operand_scale = half_accum ? std::ldexp(1.f, -exponent) : 1.f;
    const float product_scale = half_accum ? std::ldexp(1.f, 2 * exponent) : 1.f;
    auto output = empty({tokens, weight.out},
        x.options().dtype(half_accum ? kFloat16 : kFloat32));
    if (!tokens) return output.to(dtype).reshape(shape);

    if (!weight.values.is_cuda()) {
        const auto* weights = weight.values.data_ptr<unsigned char>();
        const auto* scales_data = weight.scales.data_ptr<float>();
        if (weight.dynamic_activations) x = x.clone();
        auto* source = x.data_ptr<float>();
        auto* destination = output.data_ptr<float>();
        for (std::int64_t token = 0; token < tokens; ++token) {
            float scale = 1;
            if (weight.dynamic_activations) {
                float maximum = 0;
                for (std::int64_t k = 0; k < weight.neuron_len; ++k)
                    maximum = std::max(maximum,
                        std::abs(source[token * weight.neuron_len + k]));
                scale = maximum == 0
                    ? 1.f : std::max(maximum / 448.f, std::numeric_limits<float>::min());
                for (std::int64_t k = 0; k < weight.neuron_len; ++k)
                    source[token * weight.neuron_len + k] = rounded_fp8(
                        source[token * weight.neuron_len + k] / scale);
            }
            for (std::int64_t row = 0; row < weight.out; ++row) {
                float sum = 0;
                for (std::int64_t k = 0; k < weight.neuron_len; ++k)
                    sum += tyloi::channel_fp8::decode(
                        weights[row * weight.neuron_len + k]) *
                        source[token * weight.neuron_len + k];
                destination[token * weight.out + row] = sum * scales_data[row] * scale;
            }
        }
        return output.to(dtype).reshape(shape);
    }

    MfqCudaGuard guard(weight.values.get_device());
    const auto stream = mfq_get_current_cuda_stream(weight.values.get_device());
    DenseTensor token_scales = prepared ? prepared->scales : DenseTensor{};
    if (weight.dynamic_activations && !prepared) {
        auto quantized = empty(x.sizes(), x.options().dtype(
            tokens > 8 ? kFloat16 : kFloat32));
        token_scales = empty({tokens}, x.options());
        measure_operation("fp8.input_quant", [&] {
            if (tokens > 8) {
                prepare_input<<<unsigned(tokens), 256, 0, stream.stream()>>>(
                    x.data_ptr<float>(), quantized.data_ptr<__half>(),
                    token_scales.data_ptr<float>(), int(weight.neuron_len), operand_scale);
            } else {
                prepare_input<<<unsigned(tokens), 256, 0, stream.stream()>>>(
                    x.data_ptr<float>(), quantized.data_ptr<float>(),
                    token_scales.data_ptr<float>(), int(weight.neuron_len), 1.f);
            }
            return 0;
        });
        mfq_cuda_record_stream(x, stream);
        x = std::move(quantized);
    }
    for (const auto& tensor : {weight.values, weight.scales, x, output})
        mfq_cuda_record_stream(tensor, stream);
    const auto* token_scale = token_scales.defined()
        ? token_scales.data_ptr<float>() : nullptr;
    if (token_scales.defined()) mfq_cuda_record_stream(token_scales, stream);

    if (tokens <= 8) {
        project<<<dim3(unsigned((weight.out + 3) / 4), unsigned(tokens)),
            128, 0, stream.stream()>>>(
            weight.values.data_ptr<unsigned char>(), weight.scales.data_ptr<float>(),
            x.data_ptr<float>(), token_scale, output.data_ptr<float>(),
            int(weight.neuron_len), int(weight.out));
    } else {
        // Restore channel, token, and exact power-of-two scales after GEMM.
        const auto gemm_type = weight.dynamic_activations ? kFloat16 : kFloat32;
        auto operands = measure_operation("fp8.gemm_input", [&] {
            return x.to(gemm_type).contiguous();
        });
        mfq_cuda_record_stream(operands, stream);
        const auto row_bytes = std::uint64_t(weight.neuron_len) * operands.element_size();
        const auto tile = prefill_weight_rows(
            weight.out, row_bytes, 1, weight.values.get_device());
        if (tile < 1)
            throw std::runtime_error("insufficient workspace for channel FP8 projection");
        auto decoded = empty({tile, weight.neuron_len}, operands.options());
        mfq_cuda_record_stream(decoded, stream);
        auto& blas = default_context(weight.values.get_device())->blas(stream.stream());
        const float alpha = 1, beta = 0;
        const __half half_alpha = __float2half(1.f), half_beta = __float2half(0.f);
        for (std::int64_t first = 0; first < weight.out; first += tile) {
            const auto count = std::min(tile, weight.out - first);
            const auto elements = count * weight.neuron_len;
            const auto* packed = weight.values.data_ptr<unsigned char>() +
                first * weight.neuron_len;
            measure_operation("fp8.weight_decode", [&] {
                if (gemm_type == kFloat16)
                    decode_tile<<<blocks(elements), 256, 0, stream.stream()>>>(
                        packed, decoded.data_ptr<__half>(), elements, operand_scale);
                else
                    decode_tile<<<blocks(elements), 256, 0, stream.stream()>>>(
                        packed, decoded.data_ptr<float>(), elements, 1.f);
                return 0;
            });
            const auto type = gemm_type == kFloat16 ? CUDA_R_16F : CUDA_R_32F;
            auto* destination = static_cast<std::byte*>(output.data_ptr()) +
                first * output.element_size();
            const auto plan = gemm_type == kFloat16
                ? prefill::prepare_gemm(operands.data_ptr(), decoded.data_ptr(),
                    destination, tokens, count, weight.neuron_len, weight.out,
                    stream.stream(), half_accum)
                : nullptr;
            DenseTensor scratch;
            if (plan && plan->workspace) {
                scratch = empty({std::int64_t(plan->workspace)},
                    weight.values.options().dtype(kUInt8));
                mfq_cuda_record_stream(scratch, stream);
            }
            const auto gemm_name = profiler && profiler->enabled
                ? "fp8.gemm." + std::to_string(count) + "x" +
                    std::to_string(weight.neuron_len) : "fp8.gemm";
            measure_operation(gemm_name.c_str(), [&] {
                if (plan) {
                    prefill::gemm(*plan, operands.data_ptr(), decoded.data_ptr(),
                        destination, scratch.defined() ? scratch.data_ptr() : nullptr,
                        stream.stream());
                } else {
                    const auto status = cublasGemmEx(
                        blas.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                        int(count), int(tokens), int(weight.neuron_len),
                        half_accum ? static_cast<const void*>(&half_alpha) : &alpha,
                        decoded.data_ptr(), type, int(weight.neuron_len),
                        operands.data_ptr(), type, int(weight.neuron_len),
                        half_accum ? static_cast<const void*>(&half_beta) : &beta,
                        destination, half_accum ? CUDA_R_16F : CUDA_R_32F,
                        int(weight.out), half_accum ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    if (status != CUBLAS_STATUS_SUCCESS)
                        throw std::runtime_error("channel FP8 compatibility GEMM failed");
                }
                return 0;
            });
            if (const auto* trace = std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");
                trace && trace[0] == '1') {
                static bool reported = false;
                if (!reported) {
                    reported = true;
                    std::fprintf(stderr,
                        "channel_fp8_gemm fp16_accumulation=%d cublaslt=%d "
                        "workspace_bytes=%zu operand_exponent=%d\n",
                        int(half_accum), int(bool(plan)), plan ? plan->workspace : 0,
                        half_accum ? exponent : 0);
                }
            }
        }
        auto restored = output.scalar_type() == dtype
            ? output : empty(output.sizes(), output.options().dtype(dtype));
        mfq_cuda_record_stream(restored, stream);
        measure_operation("fp8.scale_restore", [&] {
            const auto restore = [&](auto* target) {
                const auto grid = restore_grid(weight.out, tokens, weight.values.get_device());
                if (half_accum)
                    restore_scales<<<grid, 256, 0, stream.stream()>>>(
                        output.data_ptr<__half>(), target, weight.scales.data_ptr<float>(),
                        token_scale, int(weight.out), int(tokens), product_scale);
                else
                    restore_scales<<<grid, 256, 0, stream.stream()>>>(
                        output.data_ptr<float>(), target, weight.scales.data_ptr<float>(),
                        token_scale, int(weight.out), int(tokens), 1.f);
            };
            if (dtype == kFloat16) restore(restored.data_ptr<__half>());
            else if (dtype == kBFloat16) restore(restored.data_ptr<__nv_bfloat16>());
            else restore(restored.data_ptr<float>());
            return 0;
        });
        output = std::move(restored);
    }
    TYLOINFER_CUDA_CHECK(cudaGetLastError());
    return measure_operation("fp8.output_cast", [&] {
        return output.to(dtype).reshape(shape);
    });
}

DenseTensor channel_fp8_rows(const ChannelFp8Tensor& weight, DenseTensor ids) {
    if (ids.scalar_type() != kInt64 || ids.device() != weight.values.device())
        throw std::invalid_argument(
            "channel FP8 row indices must be int64 on the weight device");
    auto shape = ids.sizes().vec();
    shape.push_back(weight.neuron_len);
    ids = ids.contiguous();
    auto output = empty(shape, weight.scales.options());
    if (!output.numel()) return output;
    if (!weight.values.is_cuda()) {
        for (std::int64_t i = 0; i < ids.numel(); ++i) {
            const auto row = ids.data_ptr<std::int64_t>()[i];
            if (row < 0 || row >= weight.out)
                throw std::out_of_range("channel FP8 row is out of bounds");
            for (std::int64_t k = 0; k < weight.neuron_len; ++k)
                output.data_ptr<float>()[i * weight.neuron_len + k] =
                    tyloi::channel_fp8::decode(
                        weight.values.data_ptr<unsigned char>()[
                            row * weight.neuron_len + k]) *
                    weight.scales.data_ptr<float>()[row];
        }
    } else {
        MfqCudaGuard guard(weight.values.get_device());
        const auto stream = mfq_get_current_cuda_stream(weight.values.get_device());
        for (const auto& tensor : {weight.values, weight.scales, ids, output})
            mfq_cuda_record_stream(tensor, stream);
        select_rows<<<blocks(output.numel()), 256, 0, stream.stream()>>>(
            weight.values.data_ptr<unsigned char>(), weight.scales.data_ptr<float>(),
            ids.data_ptr<std::int64_t>(), output.data_ptr<float>(),
            int(weight.neuron_len), int(weight.out), output.numel());
        TYLOINFER_CUDA_CHECK(cudaGetLastError());
    }
    return output;
}

DenseTensor dequant_channel_fp8(const ChannelFp8Tensor& weight) {
    auto ids = arange(weight.out, weight.values.options().dtype(kInt64));
    return channel_fp8_rows(weight, std::move(ids));
}

} // namespace tyloi::cuda

#include "rotation.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/kernels/hadamard/transform.h"
#include "components/ggml/include/ggml.h"
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstring>
#include <stdexcept>


namespace tyloi::cuda::gguf {

namespace {
// Same normalized Sylvester butterfly as Strata's CPU hadamard_rotate
// (MIT, third_party/strata.LICENSE). The basis is per weight, with no global
// model state; adjoint and intermediate dtype rounding are explicit.
void rotate_cpu(float* values, std::int64_t rows, std::int64_t width, int block,
        const std::vector<std::int8_t>& signs, bool adjoint) {
    const float normalization = 1.f / std::sqrt(float(block));
    for (std::int64_t row = 0; row < rows; ++row) {
        auto* x = values + row * width;
        if (!adjoint && !signs.empty()) for (std::int64_t i = 0; i < width; ++i) x[i] *= signs[i];
        for (std::int64_t start = 0; start < width; start += block)
            for (int half = 1; half < block; half *= 2)
                for (int first = 0; first < block; first += 2 * half)
                    for (int i = 0; i < half; ++i) {
                        const float a = x[start + first + i], b = x[start + first + half + i];
                        x[start + first + i] = a + b; x[start + first + half + i] = a - b;
                    }
        for (std::int64_t i = 0; i < width; ++i)
            x[i] *= normalization * (adjoint && !signs.empty() ? float(signs[i]) : 1.f);
    }
}
void rotate_gpu(const GgufRotation& rotation, const ::tyloi::DenseTensor& input, const ::tyloi::DenseTensor& output, bool adjoint) {
    const MfqCudaGuard guard(input.get_device());
    const auto stream = mfq_get_current_cuda_stream(input.get_device());
    TYLOINFER_RUNTIME_CHECK(rotation.signs_cpu.empty() || (rotation.signs_device.defined() && rotation.signs_device.device() == input.device()),
        "GGUF Hadamard signs are not on the activation device");
    for (const auto& tensor : {input, output}) mfq_cuda_record_stream(tensor, stream);
    const std::int8_t* signs = nullptr;
    if (rotation.signs_device.defined()) { mfq_cuda_record_stream(rotation.signs_device, stream); signs = rotation.signs_device.data_ptr<std::int8_t>(); }
    const auto width = input.size(-1), rows = input.numel() / width;
    if (input.scalar_type() == ::tyloi::kFloat16)
        tyloi::cuda::hadamard::apply_f16(input.data_ptr(), signs, output.data_ptr(), rows, width, rotation.block, adjoint, stream.stream());
    else tyloi::cuda::hadamard::apply_f32(input.data_ptr<float>(), signs, output.data_ptr<float>(), rows, width, rotation.block, adjoint, stream.stream());
}
}


void apply_rotation_inplace(const GgufRotation& r, ::tyloi::DenseTensor input, bool adjoint) {
    TYLOINFER_RUNTIME_CHECK(r.block > 0 && !(r.block & (r.block - 1)) && input.dim() >= 1 && input.is_contiguous() && input.size(-1) > 0 &&
        input.size(-1) % r.block == 0 && (r.signs_cpu.empty() || int64_t(r.signs_cpu.size()) == input.size(-1)) &&
        (input.scalar_type() == ::tyloi::kFloat32 || input.scalar_type() == ::tyloi::kFloat16),
        "GGUF Hadamard activation geometry/precision differs");
    const auto width = input.size(-1), rows = input.numel() / width;
    if (!rows) return;
    if (input.is_cuda()) {
        rotate_gpu(r, input, input, adjoint);
    } else {
        if (input.scalar_type() == ::tyloi::kFloat32) rotate_cpu(input.data_ptr<float>(), rows, width, r.block, r.signs_cpu, adjoint);
        else {
            auto values = input.to(::tyloi::kFloat32).contiguous();
            rotate_cpu(values.data_ptr<float>(), rows, width, r.block, r.signs_cpu, adjoint);
            input.copy_(values.to(::tyloi::kFloat16));
        }
    }
}
::tyloi::DenseTensor apply_rotation(const GgufRotation& r, ::tyloi::DenseTensor input, bool adjoint) {
    auto source = (input.scalar_type() == ::tyloi::kFloat16 ? input : input.to(::tyloi::kFloat32)).contiguous();
    TYLOINFER_RUNTIME_CHECK(r.block > 0 && !(r.block & (r.block - 1)) && source.dim() >= 1 && source.size(-1) > 0 &&
        source.size(-1) % r.block == 0 && (r.signs_cpu.empty() || int64_t(r.signs_cpu.size()) == source.size(-1)),
        "GGUF Hadamard activation geometry differs");
    if (source.is_cuda()) {
        auto result = ::tyloi::cuda::empty_like(source);
        if (source.numel()) rotate_gpu(r, source, result, adjoint);
        return result;
    }
    auto result = source.clone(); apply_rotation_inplace(r,result, adjoint);
    return result;
}


} // namespace tyloi::cuda::gguf

#include "engine/weights/prepare/channel_fp8.h"

#include "engine/backends/cuda/native/device.h"
#include "engine/backends/cuda/native/tensor/tensor.h"
#include "engine/weights/encoding/fp8/channel.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {

float decode_scale(
        const std::uint8_t* source,
        tyloi::channel_fp8::ScaleKind kind) {
    if (kind == tyloi::channel_fp8::ScaleKind::F32) {
        float value = 0.0f;
        std::memcpy(&value, source, sizeof(value));
        return value;
    }
    std::uint16_t raw = 0;
    std::memcpy(&raw, source, sizeof(raw));
    if (kind == tyloi::channel_fp8::ScaleKind::Bf16) {
        const auto bits = std::uint32_t(raw) << 16;
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    __half half;
    std::memcpy(&half, &raw, sizeof(raw));
    return static_cast<float>(half);
}

} // namespace

tyloi::ChannelFp8Tensor tyloi::weights::load_channel_fp8(
        const std::vector<std::uint8_t>& payload,
        tyloi::Device target) {
    namespace fp = tyloi::channel_fp8;
    if ((!target.is_cpu() && !target.is_cuda()) ||
            (target.is_cpu() && target.index != 0)) {
        throw std::invalid_argument("channel FP8 weights require CPU:0 or CUDA");
    }
    const auto image = fp::parse(
        reinterpret_cast<const std::byte*>(payload.data()), payload.size());

    ChannelFp8Tensor result;
    result.out = image.outputs;
    result.neuron_len = image.width;
    result.dynamic_activations = image.dynamic_activations;
    result.values = ::tyloi::cuda::from_blob(
        const_cast<std::uint8_t*>(payload.data()) + fp::header_bytes,
        {image.outputs, image.width},
        ::tyloi::TensorOptions().device(::tyloi::kCPU).dtype(::tyloi::kUInt8)).clone();
    result.scales = ::tyloi::cuda::empty(
        {image.outputs},
        ::tyloi::TensorOptions().device(::tyloi::kCPU).dtype(::tyloi::kFloat32));

    auto* scales = result.scales.data_ptr<float>();
    for (std::int64_t row = 0; row < image.outputs; ++row) {
        const auto* source = payload.data() + image.scales +
            static_cast<std::size_t>(row) * fp::scale_bytes(image.scale_kind);
        const auto value = decode_scale(source, image.scale_kind);
        if (!(value >= 0) || !std::isfinite(value)) {
            throw std::invalid_argument(
                "channel FP8 scales must be finite and nonnegative");
        }
        scales[row] = value;
    }
    for (std::size_t i = fp::header_bytes; i < image.scales; ++i) {
        if ((payload[i] & 127u) == 127u) {
            throw std::invalid_argument("channel FP8 weights contain NaN codes");
        }
    }

    if (target.is_cuda()) {
        ::tyloi::cuda::DeviceGuard guard(target.index);
        result.values = result.values.to(target).contiguous();
        result.scales = result.scales.to(target).contiguous();
    }
    return result;
}

#include "projection.h"

#include "dense.h"
#include "engine/backends/metal/native/tensor/tensor.h"
#include "engine/backends/metal/ops/tensor.h"
#include "engine/weights/encoding/metal.h"
#include "backends/metal/runtime/mlx_tensor.h"

#include <mlx/mlx.h>

#include <stdexcept>
#include <utility>

namespace tyloi::metal::ops {
namespace {

const mlx::core::array& input(const Tensor& value) {
    if (!value.defined() || value.device() != Device{DeviceType::metal, 0} || value.is_encoded())
        throw std::invalid_argument("projection requires a defined, unencoded Metal 0 tensor");
    return metal::array(value);
}

bool floating(mlx::core::Dtype type) {
    return type == mlx::core::float16 || type == mlx::core::bfloat16 || type == mlx::core::float32;
}

mlx::core::Stream gpu_stream() {
    return mlx::core::default_stream(mlx::core::Device::gpu);
}

}  // namespace

Tensor project(const Tensor& value, const Tensor& weight) {
    if (const auto* encoded = weight.representation<encoding::MetalTensor>())
        return metal::tensor(encoded->linear()(input(value)));
    return dense_project(value, weight);
}

Tensor embedding(const Tensor& ids, const Tensor& weight) {
    if (const auto* encoded = weight.representation<encoding::MetalTensor>()) {
        const auto selected = ids.is_cpu() ? metal::copy(ids, Device{DeviceType::metal, 0}) : ids;
        return metal::tensor(encoded->embedding()(input(selected), mlx::core::float16));
    }
    if (!floating(input(weight).dtype())) throw std::invalid_argument("Metal embedding requires floating weights");
    return cast(dense_embedding(ids, weight), kFloat16);
}

Tensor linear(const Tensor& value, const Tensor& weight, const Tensor& bias) {
    const auto& source = input(value);
    auto projected = project(value, weight);
    auto output = input(projected);
    const auto stream = gpu_stream();
    if (bias.defined())
        output = mlx::core::add(output, mlx::core::astype(input(bias), output.dtype(), stream), stream);
    return metal::tensor(mlx::core::contiguous(mlx::core::astype(output, source.dtype(), stream), stream));
}

Tensor affine(const Tensor& value, const Tensor& weight, const Tensor& bias) {
    auto projected = project(value, weight);
    if (!bias.defined()) return projected;
    const auto& output = input(projected); const auto stream = gpu_stream();
    return metal::tensor(mlx::core::add(output, mlx::core::astype(input(bias), output.dtype(), stream), stream));
}

Tensor project(const Tensor& value, const Tensor& weight, const Tensor& ids) {
    const auto& x = input(value); const auto& w = input(weight); const auto& selected = input(ids);
    if (!floating(x.dtype()) || !floating(w.dtype()) || w.ndim() != 3 || selected.ndim() != 2 ||
        (selected.dtype() != mlx::core::int32 && selected.dtype() != mlx::core::int64) ||
        (x.ndim() != 2 && x.ndim() != 3) || x.shape(0) != selected.shape(0) ||
        x.shape(x.ndim() - 1) != w.shape(2) || (x.ndim() == 3 && x.shape(1) != selected.shape(1)))
        throw std::invalid_argument("invalid Metal routed projection geometry or dtype");
    const auto stream = gpu_stream();
    auto matrices = mlx::core::take(w, selected, 0, stream);
    auto source = mlx::core::astype(x, w.dtype(), stream);
    if (x.ndim() == 2) source = mlx::core::expand_dims(source, 1, stream);
    auto result = mlx::core::matmul(matrices, mlx::core::expand_dims(source, -1, stream), stream);
    return metal::tensor(mlx::core::squeeze(result, -1, stream));
}

Tensor head_project(const Tensor& value, const Tensor& weight) {
    const auto& x = input(value); const auto& w = input(weight);
    if (!floating(x.dtype()) || !floating(w.dtype()) || x.ndim() != 4 || w.ndim() != 3 ||
        !x.size() || x.shape(2) <= 0 || x.shape(3) <= 0 ||
        x.shape(2) != w.shape(0) || x.shape(3) != w.shape(2))
        throw std::invalid_argument("invalid Metal head projection geometry or dtype");
    const auto stream = gpu_stream();
    auto source = mlx::core::expand_dims(mlx::core::astype(x, w.dtype(), stream), -1, stream);
    return metal::tensor(mlx::core::squeeze(mlx::core::matmul(w, source, stream), -1, stream));
}

bool grouped(std::span<const Tensor>) { return false; }

std::vector<Tensor> project_group(const Tensor& value, std::span<const Tensor> weights) {
    if (weights.empty()) throw std::invalid_argument("empty projection group");
    std::vector<Tensor> outputs;
    outputs.reserve(weights.size());
    for (const auto& weight : weights) outputs.push_back(project(value, weight));
    return outputs;
}

std::vector<Tensor> project_group(const Tensor& value, std::span<const Tensor> weights, std::size_t prefix) {
    if (!prefix || prefix > weights.size()) throw std::invalid_argument("invalid grouped projection prefix");
    auto outputs = project_group(value, weights);
    if (prefix > 1) {
        for (std::size_t i = 1; i < prefix; ++i)
            if (outputs[i].scalar_type() != outputs[0].scalar_type())
                outputs[i] = cast(outputs[i], outputs[0].scalar_type());
        auto combined = cat(std::span<const Tensor>(outputs.data(), prefix), -1);
        outputs.erase(outputs.begin(), outputs.begin() + prefix);
        outputs.insert(outputs.begin(), std::move(combined));
    }
    return outputs;
}

}  // namespace tyloi::metal::ops

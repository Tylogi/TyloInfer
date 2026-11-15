#include "engine/ops.h"

#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/tensor/tensor.h"
#endif
#include <algorithm>
#include <cstring>

namespace tyloi::ops {
namespace {
template<class T>
std::vector<T> host_values(const Tensor& value, ScalarType type) {
    auto host = contiguous(copy(cast(value, type), Device{kCPU, 0}));
    const auto& data = host.dense();
    std::vector<T> result(static_cast<std::size_t>(data.numel()));
    if (!result.empty()) std::copy_n(data.data_ptr<T>(), result.size(), result.data());
    return result;
}
#ifdef TYLOINFER_NATIVE_CUDA
const DenseTensor& dense(const Tensor& value) {
    const auto* result = value.representation<DenseTensor>();
    if (!result || result->is_encoded()) throw std::invalid_argument("operation requires a dense tensor");
    return *result;
}
Tensor owner(DenseTensor value) {
    return std::make_shared<DenseTensor>(std::move(value));
}
void device(Device value) {
    if (!value.is_cpu() && !value.is_cuda())
        throw std::invalid_argument("unsupported tensor device");
    if (value.is_cuda()) (void)cuda::Executor::current();
}
void input(const Tensor& value_tensor) {
    const auto& value = dense(value_tensor);
    if (!value.defined()) throw std::invalid_argument("undefined tensor");
    value.require_dense();
    device(value.device());
}
#endif
} // namespace

std::vector<float> read_floats(const Tensor& tensor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if (tensor.defined() && tensor.device().is_metal()) return host_values<float>(tensor, kFloat32);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = dense(tensor);
    input(value);
    auto host = value.to(kFloat32).cpu().contiguous();
    std::vector<float> result(static_cast<std::size_t>(host.numel()));
    if (!result.empty()) std::copy_n(host.data_ptr<float>(), result.size(), result.data());
    return result;
#else
    return host_values<float>(tensor, kFloat32);
#endif
}

std::vector<std::int64_t> read_integers(const Tensor& tensor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if (tensor.defined() && tensor.device().is_metal()) return host_values<std::int64_t>(tensor, kInt64);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = dense(tensor);
    input(value);
    auto host = (value.scalar_type() == kInt64 ? value : value.to(kInt64)).cpu().contiguous();
    std::vector<std::int64_t> result(static_cast<std::size_t>(host.numel()));
    if (!result.empty()) std::copy_n(host.data_ptr<std::int64_t>(), result.size(), result.data());
    return result;
#else
    return host_values<std::int64_t>(tensor, kInt64);
#endif
}

std::pair<std::vector<float>, std::vector<std::int64_t>> unpack_topk(
        const Tensor& tensor, std::int64_t count) {
    const auto& packets = tensor;
    if (!packets.defined() || packets.is_encoded()) throw std::invalid_argument("top-k packets must be defined and unencoded");
    if (packets.dim() != 2 || packets.size(0) < 1 || packets.scalar_type() != kInt64 ||
        count < 1 || count > 64 || packets.size(1) < count)
        throw std::invalid_argument("top-k packets must be int64 [rows,stride] with stride >= count and count in [1,64]");
    auto storage = contiguous(copy(packets, Device{kCPU, 0}));
    const auto& host = storage.dense();
    const auto size = static_cast<std::size_t>(host.size(0)) * static_cast<std::size_t>(count);
    std::vector<float> values(size);
    std::vector<std::int64_t> indices(size);
    const auto* data = host.data_ptr<std::int64_t>();
    for (std::int64_t row = 0; row < host.size(0); ++row) {
        for (std::int64_t column = 0; column < count; ++column) {
            const auto packet = static_cast<std::uint64_t>(data[row * host.size(1) + column]);
            const auto raw = static_cast<std::uint32_t>(packet >> 32);
            const auto offset = static_cast<std::size_t>(row * count + column);
            std::memcpy(&values[offset], &raw, sizeof(raw));
            indices[offset] = static_cast<std::uint32_t>(packet);
        }
    }
    return {std::move(values), std::move(indices)};
}

Tensor constrain(const Tensor& logits_tensor, const MfqTokenConstraintPtr& constraint) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if (logits_tensor.defined() && logits_tensor.device().is_metal()) {
        if (!constraint) throw std::invalid_argument("token constraint is absent");
        auto masked = contiguous(copy(cast(logits_tensor, kFloat32), Device{kCPU, 0}));
        constraint->apply(masked.dense().data_ptr<float>(), static_cast<std::size_t>(masked.numel()));
        return masked;
    }
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& logits = dense(logits_tensor);
    input(logits);
    auto masked = logits.to(kCPU, kFloat32).contiguous();
    constraint->apply(masked.data_ptr<float>(), static_cast<std::size_t>(masked.numel()));
    return owner(std::move(masked));
#else
    if (!constraint) throw std::invalid_argument("token constraint is absent");
    auto masked = contiguous(copy(cast(logits_tensor, kFloat32), Device{kCPU, 0}));
    constraint->apply(masked.dense().data_ptr<float>(), static_cast<std::size_t>(masked.numel()));
    return masked;
#endif
}

std::int64_t item(const Tensor& value_tensor) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if (value_tensor.defined() && value_tensor.device().is_metal()) {
        if (!value_tensor.defined() || value_tensor.numel() != 1)
            throw std::invalid_argument("item() requires a defined scalar tensor");
        return host_values<std::int64_t>(value_tensor, kInt64).front();
    }
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = dense(value_tensor);
    input(value);
    return value.item<std::int64_t>();
#else
    if (!value_tensor.defined() || value_tensor.numel() != 1)
        throw std::invalid_argument("item() requires a defined scalar tensor");
    return host_values<std::int64_t>(value_tensor, kInt64).front();
#endif
}

} // namespace tyloi::ops

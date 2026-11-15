#include "engine/ops.h"

#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/backends/cuda/ops/rope/tail.h"
#endif
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
#include "engine/backends/metal/ops/rope/rope.h"
#include "engine/backends/metal/ops/rope/tail.h"
#endif
#include <cmath>

namespace tyloi::ops {
#ifdef TYLOINFER_NATIVE_CUDA
namespace {
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
void compatible(const Tensor& left_tensor, const Tensor& right_tensor) {
    const auto& left = ::tyloi::ops::dense(left_tensor);
    const auto& right = ::tyloi::ops::dense(right_tensor);
    input(left);
    input(right);
    if (left.device() != right.device())
        throw std::invalid_argument("tensor devices do not match");
}
void require_cuda(const Tensor& value_tensor) {
    const auto& value = ::tyloi::ops::dense(value_tensor);
    if (!value.device().is_cuda()) throw std::invalid_argument("operation requires a CUDA tensor");
}
} // namespace
#endif

RopeTable rope_table(std::int64_t positions, std::int64_t dimension, double base, Device target,
                     std::int64_t frequency_dim, std::int64_t active_pairs) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if (target.is_metal())
        return metal::ops::rope_table(positions, dimension, base, target, frequency_dim, active_pairs);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    device(target);
    if (!target.is_cuda()) throw std::invalid_argument("RoPE table requires a CUDA device");
    if (positions <= 0 || dimension <= 0 || dimension % 2 != 0 ||
        !std::isfinite(base) || base <= 0.0 || active_pairs > dimension / 2)
        throw std::invalid_argument("invalid RoPE table parameters");
    return cuda::ops::rope_table(positions, dimension, base, target, frequency_dim, active_pairs);
#else
    throw std::invalid_argument("rope_table is not supported on this device");
#endif
}

Tensor rope(const Tensor& value_tensor, const Tensor& positions_tensor, const RopeTable& table, bool bf16, bool grid) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value_tensor.defined() && value_tensor.device().is_metal()))
        return metal::ops::rope(value_tensor, positions_tensor, table, bf16, grid);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& value = dense(value_tensor);
    const auto& positions = dense(positions_tensor);
    if (!table.cos || !table.sin) throw std::invalid_argument("RoPE table requires cosine and sine");
    compatible(value, table.cos);
    compatible(value, table.sin);
    for (const auto& item : {table.sections, table.empty_sections,
                             table.interleaved_order, table.interleaved_inverse})
        if (item.defined()) (void)dense(item);
    input(positions);
    if (value.dim() == 0 || table.rotary_dim <= 0 || table.rotary_dim % 2 != 0 ||
        table.rotary_dim > value.size(-1))
        throw std::invalid_argument("invalid RoPE rotary dimension");
    require_cuda(value);
    return owner(cuda::ops::rope(value, positions, table, bf16, grid));
#else
    throw std::invalid_argument("rope is not supported on this device");
#endif
}

TailRopeTable tail_rope_table(std::int64_t max_positions, double base, std::int64_t compress_ratio,
        Device target, double compress_base, std::int64_t original_positions,
        double factor, double beta_fast, double beta_slow) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if (target.is_metal())
        return metal::ops::tail_rope_table(max_positions, base, compress_ratio, target, compress_base, original_positions, factor, beta_fast, beta_slow);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    device(target);
    if (!target.is_cuda() || target.index < 0)
        throw std::invalid_argument("tail RoPE table requires a CUDA device");
    return cuda::ops::tail_rope_table(max_positions, base, compress_ratio, target,
        compress_base, original_positions, factor, beta_fast, beta_slow);
#else
    throw std::invalid_argument("tail_rope_table is not supported on this device");
#endif
}

Tensor tail_rope(const Tensor& value, const Tensor& positions,
        const TailRopeTable& table, bool inverse) {
#ifdef TYLOINFER_NATIVE_MLX_RUNTIME
    if ((value.defined() && value.device().is_metal()))
        return metal::ops::tail_rope(value, positions, table, inverse);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& x = dense(value);
    const auto& pos = dense(positions);
    device(x.device());
    if (!x.device().is_cuda() || x.device().index < 0)
        throw std::invalid_argument("tail RoPE requires a CUDA device");
    if (!table.cos || !(inverse ? table.negative_sin : table.sin))
        throw std::invalid_argument("tail RoPE table requires cosine and sine");
    return owner(cuda::ops::tail_rope(x, pos, table, inverse));
#else
    throw std::invalid_argument("tail_rope is not supported on this device");
#endif
}

} // namespace tyloi::ops

#include "engine/ops.h"

#if TYLOINFER_HAS_CUDA
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/tensor.h"
#include "engine/backends/cuda/native/tensor/tensor.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#endif
#if TYLOINFER_HAS_METAL
#include "engine/backends/metal/ops/sampling.h"
#endif

#include <tuple>

namespace tyloi::ops {
#if TYLOINFER_HAS_CUDA
namespace {
const DenseTensor& dense(const Tensor& value) {
    const auto* result = value.representation<DenseTensor>();
    if (!result || result->is_encoded()) throw std::invalid_argument("operation requires a dense tensor");
    return *result;
}
DenseTensor& dense(Tensor& value) {
    auto* result = value.representation<DenseTensor>();
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

Tensor sample_greedy(const Tensor& logits_tensor) {
#if TYLOINFER_HAS_METAL
    if ((logits_tensor.defined() && logits_tensor.device().is_metal()))
        return metal::ops::sample_greedy(logits_tensor);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& logits = dense(logits_tensor);
    input(logits);
    require_cuda(logits);
    return owner(::sample_greedy_cuda(logits));
#else
    throw std::invalid_argument("sample_greedy is not supported on this device");
#endif
}

Tensor sample_stochastic(const Tensor& logits_tensor, const Tensor& random_tensor,
                                          double temperature, std::int64_t top_k, double top_p) {
#if TYLOINFER_HAS_METAL
    if ((logits_tensor.defined() && logits_tensor.device().is_metal()) ||
        (random_tensor.defined() && random_tensor.device().is_metal()))
        return metal::ops::sample_stochastic(logits_tensor, random_tensor, temperature, top_k, top_p);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& logits = dense(logits_tensor);
    const auto& random = dense(random_tensor);
    compatible(logits, random);
    require_cuda(logits);
    if (top_k > 0)
        return owner(::sample_top_k_top_p_cuda(logits, random, temperature, top_k, top_p));
    return owner(::sample_softmax_cuda(logits, random, temperature));
#else
    throw std::invalid_argument("sample_stochastic is not supported on this device");
#endif
}

void add_token_counts(Tensor& counts_tensor, const Tensor& ids_tensor) {
#if TYLOINFER_HAS_METAL
    if ((counts_tensor.defined() && counts_tensor.device().is_metal()) ||
        (ids_tensor.defined() && ids_tensor.device().is_metal())) {
        metal::ops::add_token_counts(counts_tensor, ids_tensor);
        return;
    }
#endif
#if TYLOINFER_HAS_CUDA
    auto& counts = dense(counts_tensor);
    const auto& ids = dense(ids_tensor);
    compatible(counts, ids);
    require_cuda(counts);
    ::sample_token_counts_add_cuda(counts, ids);
#else
    throw std::invalid_argument("add_token_counts is not supported on this device");
#endif
}

void apply_penalties(Tensor& logits_tensor, const Tensor& counts_tensor,
                     double presence, double frequency, double repetition) {
#if TYLOINFER_HAS_METAL
    if ((logits_tensor.defined() && logits_tensor.device().is_metal()) ||
        (counts_tensor.defined() && counts_tensor.device().is_metal())) {
        metal::ops::apply_penalties(logits_tensor, counts_tensor, presence, frequency, repetition);
        return;
    }
#endif
#if TYLOINFER_HAS_CUDA
    auto& logits = dense(logits_tensor);
    const auto& counts = dense(counts_tensor);
    compatible(logits, counts);
    require_cuda(logits);
    ::sample_apply_penalties_cuda(logits, counts, presence, frequency, repetition);
#else
    throw std::invalid_argument("apply_penalties is not supported on this device");
#endif
}

std::pair<std::vector<float>, std::vector<std::int64_t>> read_topk(
        const Tensor& tensor, std::int64_t count, bool force_compact) {
#if TYLOINFER_HAS_METAL
    if ((tensor.defined() && tensor.device().is_metal()))
        return metal::ops::read_topk(tensor, count, force_compact);
#endif
#if TYLOINFER_HAS_CUDA
    const auto& logits = dense(tensor);
    input(logits);
    require_cuda(logits);
    if (logits.dim() != 2 || logits.size(0) < 1 || count < 1 || count > 64 || count > logits.size(1))
        throw std::invalid_argument("top-k logits must be [rows,vocab] with count in [1,min(vocab,64)]");
    const auto* option = force_compact ? nullptr : cuda::env_lookup("TYLOINFER_MTP_RADIX_TOPK");
    if (force_compact || (option && option[0] == '1' && logits.size(0) <= 32))
        return unpack_topk(owner(::sample_top_k_compact_cuda(logits, count)), count);
    auto selected = ::tyloi::cuda::topk(logits, count, -1, true, true);
    auto values = std::get<0>(selected).to(kCPU, kFloat32).contiguous();
    auto indices = std::get<1>(selected).to(kCPU, kInt64).contiguous();
    const auto size = static_cast<std::size_t>(values.numel());
    return {{values.data_ptr<float>(), values.data_ptr<float>() + size},
            {indices.data_ptr<std::int64_t>(), indices.data_ptr<std::int64_t>() + size}};
#else
    throw std::invalid_argument("read_topk is not supported on this device");
#endif
}

void decode_commit(const Tensor& next, Tensor& generated, Tensor& step,
                   Tensor& input_tensor, Tensor& position, Tensor& length) {
#if TYLOINFER_HAS_METAL
    if (next.defined() && next.device().is_metal()) {
        metal::ops::decode_commit(next, generated, step, input_tensor, position, length);
        return;
    }
#endif
#if TYLOINFER_HAS_CUDA
    input(next);
    require_cuda(next);
    const std::array<const Tensor*, 6> operands{&next, &generated, &step, &input_tensor, &position, &length};
    for (const auto* value : operands) compatible(next, *value);
    for (const auto* value : operands) {
        const auto& item = dense(*value);
        if (item.scalar_type() != kInt64 || !item.is_contiguous())
            throw std::invalid_argument("decode commit requires contiguous int64 tensors");
    }
    if (dense(next).numel() != 1 || dense(generated).numel() < 1 || dense(step).numel() != 1 ||
        dense(input_tensor).numel() != 1 || dense(position).numel() != 1 || dense(length).numel() != 1)
        throw std::invalid_argument("invalid decode commit geometry");
    cuda::ops::decode_commit(dense(next), dense(generated), dense(step), dense(input_tensor), dense(position), dense(length));
#else
    throw std::invalid_argument("decode commit requires supported graph execution");
#endif
}

} // namespace tyloi::ops

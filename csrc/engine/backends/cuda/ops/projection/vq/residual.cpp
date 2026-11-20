#include <stdexcept>
#include "backend.h"
#include "engine/backends/cuda/kernels/projection/nepq/residual_raw.h"

namespace {
void validate_residual(
    const ::tyloi::DenseTensor & dictionary,
    const ::tyloi::DenseTensor & first,
    const ::tyloi::DenseTensor & second,
    int64_t width,
    int64_t position_bits,
    int64_t block_vectors) {
    if (!(dictionary.is_cuda() && dictionary.scalar_type() == ::tyloi::kFloat16 &&
        dictionary.is_contiguous() && dictionary.dim() == 2 &&
        dictionary.size(0) == 1024 && dictionary.size(1) == 8))
        throw std::runtime_error("NEPQ-A dictionary must be CUDA contiguous fp16 [1024,8]");
    if (!(first.is_cuda() && first.scalar_type() == ::tyloi::kInt16 &&
        first.is_contiguous() && first.dim() == 2))
        throw std::runtime_error("NEPQ-A first records must be CUDA contiguous int16 rank-2");
    if (!(second.is_cuda() && second.scalar_type() == ::tyloi::kInt16 &&
        second.is_contiguous() && second.sizes() == first.sizes()))
        throw std::runtime_error("NEPQ-A second records must be int16 and match the first-record layout");
    if (!(dictionary.device() == first.device() &&
        dictionary.device() == second.device()))
        throw std::runtime_error("NEPQ-A residual tensors must share one CUDA device");
    if (!(width > 0 && width % 8 == 0))
        throw std::runtime_error("NEPQ-A width must be divisible by 8");
    if (!(position_bits >= 1 && position_bits <= 5 &&
        block_vectors >= 2 && block_vectors <= 32))
        throw std::runtime_error("invalid NEPQ-A residual profile");
    const int64_t expected_blocks = (width / 8 + block_vectors - 1) / block_vectors;
    if (!(first.size(1) == expected_blocks))
        throw std::runtime_error("NEPQ-A block count mismatch");
}

}

/** Add a sparse residual matrix product to the supplied output. */
::tyloi::DenseTensor nepq_sparse_residual_matmul_cuda(
    ::tyloi::DenseTensor dictionary,
    ::tyloi::DenseTensor first,
    ::tyloi::DenseTensor second,
    ::tyloi::DenseTensor input,
    int64_t position_bits,
    int64_t block_vectors,
    ::tyloi::DenseTensor output) {
    if (!(input.is_cuda() && input.scalar_type() == ::tyloi::kFloat16 &&
        input.is_contiguous() && input.dim() == 2))
        throw std::runtime_error("NEPQ-A input must be CUDA contiguous fp16 rank-2");
    if (!(output.is_cuda() && output.scalar_type() == ::tyloi::kFloat16 &&
        output.is_contiguous() && output.dim() == 2 &&
        output.size(0) == input.size(0) && output.size(1) == first.size(0)))
        throw std::runtime_error("NEPQ-A output must be CUDA contiguous fp16 [M,rows]");
    validate_residual(
        dictionary, first, second, input.size(1), position_bits, block_vectors);
    if (!(dictionary.device() == input.device() && input.device() == output.device()))
        throw std::runtime_error("NEPQ-A matmul tensors must share one CUDA device");
    MfqCudaGuard guard(input.device());
    const int64_t total = input.size(0) * first.size(0);
    cudaStream_t stream = mfq_current_cuda_stream();
    tyloi::cuda::nepq_residual_raw::matmul(
        dictionary.data_ptr(), first.data_ptr<int16_t>(), second.data_ptr<int16_t>(),
        input.data_ptr(), output.data_ptr(), static_cast<int>(first.size(0)),
        static_cast<int>(input.size(1)), static_cast<int>(input.size(0)),
        static_cast<int>(first.size(1)), static_cast<int>(position_bits),
        static_cast<int>(block_vectors), total, stream);
    if (!(cudaGetLastError() == cudaSuccess))
        throw std::runtime_error("NEPQ-A residual matmul kernel launch failed");
    return output;
}


/** Add decoded sparse residual vectors to the supplied weight matrix. */
::tyloi::DenseTensor nepq_sparse_residual_dequant_cuda(
    ::tyloi::DenseTensor dictionary,
    ::tyloi::DenseTensor first,
    ::tyloi::DenseTensor second,
    int64_t position_bits,
    int64_t block_vectors,
    ::tyloi::DenseTensor weight) {
    if (!(weight.is_cuda() && weight.scalar_type() == ::tyloi::kFloat16 &&
        weight.is_contiguous() && weight.dim() == 2 &&
        weight.size(0) == first.size(0)))
        throw std::runtime_error("NEPQ-A weight must be CUDA contiguous fp16 [rows,K]");
    validate_residual(
        dictionary, first, second, weight.size(1), position_bits, block_vectors);
    if (!(dictionary.device() == weight.device()))
        throw std::runtime_error("NEPQ-A dequant tensors must share one CUDA device");
    MfqCudaGuard guard(weight.device());
    const int64_t total = first.numel();
    cudaStream_t stream = mfq_current_cuda_stream();
    tyloi::cuda::nepq_residual_raw::dequant(
        dictionary.data_ptr(), first.data_ptr<int16_t>(), second.data_ptr<int16_t>(),
        weight.data_ptr(), static_cast<int>(weight.size(1)),
        static_cast<int>(first.size(1)), static_cast<int>(position_bits),
        static_cast<int>(block_vectors), total, stream);
    if (!(cudaGetLastError() == cudaSuccess))
        throw std::runtime_error("NEPQ-A residual dequant kernel launch failed");
    return weight;
}


/** Add routed sparse residual products to the supplied output. */
::tyloi::DenseTensor nepq_sparse_residual_grouped_cuda(
    ::tyloi::DenseTensor dictionary,
    ::tyloi::DenseTensor first,
    ::tyloi::DenseTensor second,
    ::tyloi::DenseTensor input,
    ::tyloi::DenseTensor route_ids,
    ::tyloi::DenseTensor expert_local,
    int64_t out_per_expert,
    int64_t position_bits,
    int64_t block_vectors,
    ::tyloi::DenseTensor output) {
    if (!(input.is_cuda() && input.scalar_type() == ::tyloi::kFloat16 &&
        input.is_contiguous() && (input.dim() == 2 || input.dim() == 3)))
        throw std::runtime_error("NEPQ-A routed input must be CUDA contiguous fp16 rank-2 or rank-3");
    if (!(route_ids.is_cuda() && route_ids.scalar_type() == ::tyloi::kInt32 &&
        route_ids.is_contiguous() && route_ids.dim() == 2))
        throw std::runtime_error("NEPQ-A route IDs must be CUDA contiguous int32 rank-2");
    if (!(expert_local.is_cuda() && expert_local.scalar_type() == ::tyloi::kInt32 &&
        expert_local.is_contiguous() && expert_local.dim() == 1))
        throw std::runtime_error("NEPQ-A expert map must be CUDA contiguous int32 rank-1");
    if (!(output.is_cuda() && output.scalar_type() == ::tyloi::kFloat16 &&
        output.is_contiguous() && output.dim() == 3 &&
        output.size(0) == route_ids.size(0) &&
        output.size(1) == route_ids.size(1) &&
        output.size(2) == out_per_expert))
        throw std::runtime_error("NEPQ-A grouped output shape mismatch");
    const int64_t width = input.size(-1);
    validate_residual(dictionary, first, second, width, position_bits, block_vectors);
    if (!(dictionary.device() == input.device() &&
        input.device() == route_ids.device() &&
        input.device() == expert_local.device() &&
        input.device() == output.device()))
        throw std::runtime_error("NEPQ-A grouped tensors must share one CUDA device");
    if (!(first.size(0) % out_per_expert == 0))
        throw std::runtime_error("NEPQ-A expert row count is not divisible by output width");
    if (!(input.size(0) == route_ids.size(0) &&
        (input.dim() == 2 || input.size(1) == route_ids.size(1))))
        throw std::runtime_error("NEPQ-A routed input leading dimensions mismatch");
    MfqCudaGuard guard(input.device());
    const int64_t total = route_ids.numel() * out_per_expert;
    cudaStream_t stream = mfq_current_cuda_stream();
    tyloi::cuda::nepq_residual_raw::grouped(
        dictionary.data_ptr(), first.data_ptr<int16_t>(), second.data_ptr<int16_t>(),
        input.data_ptr(), route_ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),
        output.data_ptr(), static_cast<int>(route_ids.size(0)),
        static_cast<int>(route_ids.size(1)), static_cast<int>(out_per_expert),
        static_cast<int>(width), static_cast<int>(first.size(1)),
        static_cast<int>(position_bits), static_cast<int>(block_vectors),
        input.dim() == 3, expert_local.numel() != 0,
        static_cast<int>(expert_local.numel()),
        static_cast<int>(first.size(0) / out_per_expert), total, stream);
    if (!(cudaGetLastError() == cudaSuccess))
        throw std::runtime_error("NEPQ-A residual grouped kernel launch failed");
    return output;
}

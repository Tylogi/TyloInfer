#include <stdexcept>
#include "backend.h"
#include "engine/backends/cuda/kernels/hadamard/transform.h"

/** Apply the signed block Hadamard transform to CUDA activations. */
::tyloi::DenseTensor nepq_hadamard_input_cuda(
    ::tyloi::DenseTensor input,
    ::tyloi::DenseTensor signs,
    int64_t block_size) {
    if (!(input.is_cuda() && input.scalar_type() == ::tyloi::kFloat16 &&
        input.is_contiguous() && input.dim() == 2))
        throw std::runtime_error("NEPQ Hadamard input must be CUDA contiguous fp16 rank-2");
    if (!(signs.is_cuda() && signs.scalar_type() == ::tyloi::kInt8 &&
        signs.is_contiguous() && signs.dim() == 1))
        throw std::runtime_error("NEPQ Hadamard signs must be CUDA contiguous int8 rank-1");
    if (!(block_size >= 2 && block_size <= 2048 &&
        (block_size & (block_size - 1)) == 0))
        throw std::runtime_error("NEPQ Hadamard block must be a power of two in [2,2048]");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    if (!(input.device() == signs.device()))
        throw std::runtime_error("NEPQ Hadamard tensors must share one CUDA device");
    if (!(signs.numel() == width))
        throw std::runtime_error("NEPQ Hadamard sign width mismatch");
    if (!(width % block_size == 0))
        throw std::runtime_error("NEPQ Hadamard block must divide K");
    MfqCudaGuard guard(input.device());
    auto output = ::tyloi::cuda::empty_like(input);
    cudaStream_t stream = mfq_current_cuda_stream();
    tyloi::cuda::hadamard::apply_f16(input.data_ptr(), signs.data_ptr<int8_t>(),
        output.data_ptr(), rows, width, static_cast<int>(block_size), false, stream);
    if (!(cudaGetLastError() == cudaSuccess))
        throw std::runtime_error("NEPQ Hadamard input kernel launch failed");
    return output;
}


/** Apply the adjoint signed block Hadamard transform to CUDA activations. */
::tyloi::DenseTensor nepq_hadamard_adjoint_cuda(
    ::tyloi::DenseTensor input,
    ::tyloi::DenseTensor signs,
    int64_t block_size) {
    if (!(input.is_cuda() && input.scalar_type() == ::tyloi::kFloat16 &&
        input.is_contiguous() && input.dim() == 2))
        throw std::runtime_error("NEPQ Hadamard adjoint input must be CUDA contiguous fp16 rank-2");
    if (!(signs.is_cuda() && signs.scalar_type() == ::tyloi::kInt8 &&
        signs.is_contiguous() && signs.dim() == 1))
        throw std::runtime_error("NEPQ Hadamard adjoint signs must be CUDA contiguous int8 rank-1");
    if (!(block_size >= 2 && block_size <= 2048 &&
        (block_size & (block_size - 1)) == 0))
        throw std::runtime_error("NEPQ Hadamard adjoint block must be a power of two in [2,2048]");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    if (!(signs.numel() == width && width % block_size == 0))
        throw std::runtime_error("NEPQ Hadamard adjoint geometry mismatch");
    if (!(input.device() == signs.device()))
        throw std::runtime_error("NEPQ Hadamard adjoint tensors must share one CUDA device");
    MfqCudaGuard guard(input.device());
    auto output = ::tyloi::cuda::empty_like(input);
    if (rows == 0) {
        return output;
    }
    tyloi::cuda::hadamard::apply_f16(input.data_ptr(), signs.data_ptr<int8_t>(),
        output.data_ptr(), rows, width, static_cast<int>(block_size), true,
        mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

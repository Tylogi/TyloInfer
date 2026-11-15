#include "engine/backends/cuda/ops/attention/kv_storage.h"

#include "engine/backends/cuda/native/tensor/backend.h"

#include <cstdint>
#include <stdexcept>

namespace tyloi::cuda::ops {

void bind_kv_chunk(Tensor keys, Tensor values, std::size_t index,
        const Tensor& key_chunk, const Tensor& value_chunk) {
    if (!keys.defined() || !values.defined() || !key_chunk.defined() || !value_chunk.defined() ||
            !keys.is_cuda() || values.device() != keys.device() ||
            key_chunk.device() != keys.device() || value_chunk.device() != keys.device() ||
            keys.scalar_type() != kInt64 || values.scalar_type() != kInt64 ||
            keys.dim() != 1 || values.dim() != 1 || !keys.is_contiguous() || !values.is_contiguous() ||
            index >= static_cast<std::size_t>(keys.size(0)) ||
            index >= static_cast<std::size_t>(values.size(0)) ||
            !key_chunk.is_contiguous() || !value_chunk.is_contiguous())
        throw std::invalid_argument("invalid paged KV chunk pointer binding");

    auto& key_table = keys.dense();
    auto& value_table = values.dense();
    const auto key_pointer = reinterpret_cast<std::int64_t>(key_chunk.dense().data_ptr());
    const auto value_pointer = reinterpret_cast<std::int64_t>(value_chunk.dense().data_ptr());
    MfqCudaGuard guard(keys.device().index);
    const auto stream = mfq_get_current_cuda_stream();
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(key_table.data_ptr<std::int64_t>() + index,
        &key_pointer, sizeof(key_pointer), cudaMemcpyHostToDevice, stream));
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(value_table.data_ptr<std::int64_t>() + index,
        &value_pointer, sizeof(value_pointer), cudaMemcpyHostToDevice, stream));
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace tyloi::cuda::ops

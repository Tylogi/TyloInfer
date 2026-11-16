#include "engine/ops/kv_storage.h"

#if TYLOINFER_HAS_CUDA
#include "engine/backends/cuda/ops/attention/kv/storage.h"
#endif

#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::ops {

Tensor storage_view(const Tensor& arena, std::size_t offset,
        std::span<const std::int64_t> shape, ScalarType dtype) {
    const auto* dense = arena.representation<DenseTensor>();
    const auto width = scalar_size(dtype);
    if (!dense || !dense->defined() || dense->is_encoded() || !dense->is_contiguous() ||
        dense->scalar_type() != kUInt8 || dense->dim() != 1 || !width ||
        (!dense->device().is_cuda() && dense->device() != Device{DeviceType::cpu, 0}) || offset % width)
        throw std::invalid_argument("storage view requires a contiguous CPU or CUDA byte arena");
    auto view = make_contiguous_view(nullptr, shape, dtype, dense->device());
    if (offset > dense->nbytes() || view.nbytes() > dense->nbytes() - offset ||
        offset > std::numeric_limits<std::size_t>::max() - dense->byte_offset())
        throw std::out_of_range("storage view exceeds its arena");
    return Tensor{dense->alias(view, dense->byte_offset() + offset)};
}

void bind_kv_chunk(Tensor keys, Tensor values, std::size_t index,
        const Tensor& key_chunk, const Tensor& value_chunk) {
#if TYLOINFER_HAS_CUDA
    cuda::ops::bind_kv_chunk(std::move(keys), std::move(values), index, key_chunk, value_chunk);
#else
    throw std::invalid_argument("KV chunk pointer binding requires CUDA");
#endif
}

std::size_t gather_kv(std::span<const KvGatherRow> rows, const KvGatherWorkspace& workspace) {
#if TYLOINFER_HAS_CUDA
    return cuda::ops::gather_kv(rows, workspace);
#else
    throw std::invalid_argument("fixed KV gather requires CUDA");
#endif
}

} // namespace tyloi::ops

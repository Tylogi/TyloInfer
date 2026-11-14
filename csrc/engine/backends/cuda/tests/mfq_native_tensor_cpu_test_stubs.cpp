#include "tensor.h"

#include <stdexcept>

namespace tyloi::cuda {

DenseTensor empty_cuda(std::span<const std::int64_t>, const TensorOptions&) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}

DenseTensor empty_pinned(std::span<const std::int64_t>, const TensorOptions&) {
    throw std::runtime_error("pinned memory is unavailable in the host-only tensor test");
}

DenseTensor copy_or_convert_cuda(const DenseTensor&, const TensorOptions&) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}

void fill_cuda(DenseTensor&, double) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}

void copy_cuda(DenseTensor&, const DenseTensor&) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}

DenseTensor make_contiguous_cuda(const DenseTensor&) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}

DenseTensor mean_cuda(const DenseTensor&, std::int64_t, bool) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}

#define TYLOINFER_HOST_CUDA_STUB(name, signature) \
    DenseTensor name signature { \
        throw std::runtime_error("CUDA is unavailable in the host-only tensor test"); \
    }

TYLOINFER_HOST_CUDA_STUB(reduce_cuda,
    (const DenseTensor&, std::int64_t, bool, int))
TYLOINFER_HOST_CUDA_STUB(unary_cuda,
    (const DenseTensor&, int, double, double))
TYLOINFER_HOST_CUDA_STUB(binary_cuda,
    (const DenseTensor&, const DenseTensor&, int))
TYLOINFER_HOST_CUDA_STUB(scalar_binary_cuda,
    (const DenseTensor&, double, int, bool))
TYLOINFER_HOST_CUDA_STUB(index_select_cuda,
    (const DenseTensor&, std::int64_t, const DenseTensor&))
TYLOINFER_HOST_CUDA_STUB(gather_cuda,
    (const DenseTensor&, std::int64_t, const DenseTensor&))
TYLOINFER_HOST_CUDA_STUB(repeat_cuda,
    (const DenseTensor&, std::span<const std::int64_t>))
TYLOINFER_HOST_CUDA_STUB(repeat_interleave_cuda,
    (const DenseTensor&, std::int64_t, std::int64_t))
TYLOINFER_HOST_CUDA_STUB(masked_select_cuda,
    (const DenseTensor&, const DenseTensor&))

#undef TYLOINFER_HOST_CUDA_STUB

void scatter_cuda(DenseTensor&, std::int64_t, const DenseTensor&, const DenseTensor&, bool) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}
void index_copy_cuda(DenseTensor&, std::int64_t, const DenseTensor&, const DenseTensor&) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}
void index_fill_cuda(DenseTensor&, std::int64_t, const DenseTensor&, double) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}
void masked_fill_cuda(DenseTensor&, const DenseTensor&, double) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}

bool equal(const DenseTensor&, const DenseTensor&) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}
DenseTensor where(const DenseTensor&, const DenseTensor&, double) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}
DenseTensor arange(std::int64_t, const TensorOptions&) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}
DenseTensor operator<=(const DenseTensor&, const DenseTensor&) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}
DenseTensor operator+(const DenseTensor&, const DenseTensor&) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}
DenseTensor operator+(const DenseTensor&, double) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}

}  // namespace tyloi::cuda

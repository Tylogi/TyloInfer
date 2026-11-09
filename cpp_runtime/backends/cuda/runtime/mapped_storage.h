#pragma once
#include "mfq_cuda_context.h"
#include "mfq_tensor_backend.h"
#include <cstring>
#include <cstdlib>
#include <memory>
#include <stdexcept>
namespace mfq::cuda {
inline mfq_tensor_backend::Tensor map_pinned_tensor_storage(
        const mfq_tensor_backend::Tensor& cpu,int device) {
    namespace tb=mfq_tensor_backend;
    if(!cpu.is_cpu() || !cpu.is_contiguous() || !cpu.nbytes())
        throw std::invalid_argument("mapped storage requires nonempty contiguous host storage");
    struct Owner {
        HostBuffer host;
        std::size_t bytes;
        explicit Owner(std::size_t count):host(count,true),bytes(count) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            charge_tensor_host_bytes(bytes);
#endif
        }
        ~Owner() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            tensor_host_bytes.fetch_sub(bytes);
#endif
        }
    };
    MfqCudaGuard guard(device);
    auto owner=std::make_shared<Owner>(cpu.nbytes());
    std::memcpy(owner->host.data(),cpu.data_ptr(),cpu.nbytes());
    void* address=nullptr;
    TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&address,owner->host.data(),0));
    const auto gpu=tb::Device(tb::kCUDA,device);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    auto storage=std::make_shared<tb::TensorStorage>();
    storage->owner=std::move(owner);storage->base=address;storage->bytes=cpu.nbytes();storage->device=gpu;
    return tb::Tensor(storage,tb::make_contiguous_view(address,cpu.sizes(),cpu.scalar_type(),gpu));
#else
    return tb::from_blob(address,cpu.sizes(),[owner](void*){},cpu.options().device(gpu));
#endif
}

inline bool host_metadata_enabled(int device) {
    const auto* value=std::getenv("TYLOINFER_QSA_HOST_METADATA");
    if(!value || value[0]!='1')return false;
    int mapped=0,unified=0;
    TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&mapped,cudaDevAttrCanMapHostMemory,device));
    TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&unified,cudaDevAttrUnifiedAddressing,device));
    return mapped && unified;
}
inline mfq_tensor_backend::Tensor zero_mapped_metadata(
        std::int64_t rows,std::int64_t bytes,const mfq_tensor_backend::TensorOptions& options,int device) {
    namespace tb=mfq_tensor_backend;
    auto host=tb::zeros({rows,bytes},options.device(tb::kCPU).dtype(tb::kUInt8));
    return map_pinned_tensor_storage(host,device);
}
}

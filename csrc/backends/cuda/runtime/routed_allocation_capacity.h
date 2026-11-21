#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace mfq::cuda {
// Idle pages in the resident pool can serve another compressed image without
// an additional physical charge. Temporary workspace pools remain separate.
inline std::size_t routed_reusable_pool_bytes(const void* image) {
#if defined(TYLOINFER_NATIVE_CUDA_RUNTIME) && CUDA_VERSION >= 11030
    if(!image)return 0;
    CUmemoryPool pool=nullptr;
    if(cuPointerGetAttribute(&pool,CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
            reinterpret_cast<CUdeviceptr>(image))!=CUDA_SUCCESS || !pool)return 0;
    std::uint64_t used=0,reserved=0;
    const auto native=reinterpret_cast<cudaMemPool_t>(pool);
    if(cudaMemPoolGetAttribute(native,cudaMemPoolAttrUsedMemCurrent,&used)!=cudaSuccess ||
            cudaMemPoolGetAttribute(native,cudaMemPoolAttrReservedMemCurrent,&reserved)!=cudaSuccess) {
        (void)cudaGetLastError();return 0;
    }
    return reserved>used?reserved-used:0;
#else
    return 0;
#endif
}
}

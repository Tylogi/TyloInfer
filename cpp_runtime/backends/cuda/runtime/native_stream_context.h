#pragma once
#include "mfq_cuda_context.h"
#include <cuda.h>

namespace mfq::cuda {
// Runtime device queries can succeed on a worker without a current driver
// context. Bind the stream's owner and restore any context supplied by callers.
class NativeStreamContext {
    struct Functions {
        decltype(&cuStreamGetCtx) stream_context;
        decltype(&cuCtxGetCurrent) current;
        decltype(&cuCtxPushCurrent) push;
        decltype(&cuCtxPopCurrent) pop;
        template<class Function> static Function resolve(const char* name) {
            void* raw=nullptr;cudaDriverEntryPointQueryResult query=cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
            // Keep the two-argument cuStreamGetCtx ABI instead of selecting v2.
            TYLOINFER_NATIVE_CUDA_CHECK(cudaGetDriverEntryPointByVersion(name,&raw,12000,cudaEnableDefault,&query));
#else
            TYLOINFER_NATIVE_CUDA_CHECK(cudaGetDriverEntryPoint(name,&raw,cudaEnableDefault,&query));
#endif
            if(!raw || query!=cudaDriverEntryPointSuccess)
                throw std::runtime_error(std::string("CUDA context entry unavailable: ")+name);
            return reinterpret_cast<Function>(raw);
        }
        Functions():stream_context(resolve<decltype(stream_context)>("cuStreamGetCtx")),
            current(resolve<decltype(current)>("cuCtxGetCurrent")),
            push(resolve<decltype(push)>("cuCtxPushCurrent")),pop(resolve<decltype(pop)>("cuCtxPopCurrent")) {}
    };
    static const Functions& functions(){static const Functions value;return value;}
    bool pushed_=false;
    static void checked(CUresult result) {
        if(result!=CUDA_SUCCESS)throw std::runtime_error("CUDA stream context failed: "+std::to_string(result));
    }
public:
    explicit NativeStreamContext(cudaStream_t stream) {
        CUcontext owner=nullptr,current=nullptr;
        const auto& api=functions();
        checked(api.stream_context(reinterpret_cast<CUstream>(stream),&owner));
        checked(api.current(&current));
        if(owner!=current){checked(api.push(owner));pushed_=true;}
    }
    NativeStreamContext(const NativeStreamContext&)=delete;
    NativeStreamContext& operator=(const NativeStreamContext&)=delete;
    ~NativeStreamContext()noexcept {
        if(pushed_){CUcontext previous=nullptr;(void)functions().pop(&previous);}
    }
};
}

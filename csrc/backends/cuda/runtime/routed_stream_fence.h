#pragma once
#include "mfq_tensor_backend.h"
#include "mfq_cuda_context.h"
#include <initializer_list>
#include <stdexcept>

namespace mfq::cuda {
// Call after joining host submitters. Only the consumer and owned producer
// lanes may access the storage being retired; null producers are absent.
inline void synchronize_routed_streams(const MfqCudaStream& consumer,
        std::initializer_list<cudaStream_t> producers={}) {
    const MfqCudaGuard guard(consumer.device());
    const auto check=[](cudaStream_t stream) {
        cudaStreamCaptureStatus capture;
        TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stream,&capture));
        if(capture!=cudaStreamCaptureStatusNone)
            throw std::logic_error("routed storage cannot be retired during graph capture");
    };
    check(consumer.stream());
    for(const auto stream:producers)if(stream && stream!=consumer.stream())check(stream);
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(consumer.stream()));
    for(auto i=producers.begin();i!=producers.end();++i)
        if(*i && *i!=consumer.stream()) {
            bool duplicate=false;
            for(auto j=producers.begin();j!=i;++j)duplicate|=*j==*i;
            if(!duplicate)TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(*i));
        }
}
} // namespace mfq::cuda

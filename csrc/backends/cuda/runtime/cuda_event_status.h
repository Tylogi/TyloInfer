#pragma once
#include "mfq_cuda_context.h"

namespace mfq::cuda {
// The caller owns the event's device context. Only an unfinished event is
// pending; invalid handles and asynchronous execution errors must propagate.
inline bool cuda_event_ready(cudaEvent_t event) {
    const auto status=cudaEventQuery(event);
    if(status==cudaErrorNotReady) {
        (void)cudaGetLastError();
        return false;
    }
    TYLOINFER_NATIVE_CUDA_CHECK(status);
    return true;
}
} // namespace mfq::cuda

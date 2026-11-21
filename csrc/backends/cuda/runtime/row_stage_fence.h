#pragma once
#include "mfq_cuda_context.h"
#include "mfq_tensor_backend.h"
#include <memory>

namespace mfq::cuda {
// Host packets cannot be overwritten until their last eager consumer ends.
// Decode-window owners fence graph consumers before calling upload_graph().
class RowStageFence {
    std::unique_ptr<Event> done_;
    cudaStream_t stream_=nullptr;
    bool pending_=false;
public:
    void initialize() {if(!done_)done_=std::make_unique<Event>();}
    void before_write() {
        if(pending_)done_->synchronize();
        pending_=false;
    }
    void owner_fenced()noexcept {pending_=false;}
    void record_transfer(cudaStream_t stream) {
        try {initialize();done_->record(stream);}
        catch(...) {(void)cudaStreamSynchronize(stream);throw;}
        stream_=stream;pending_=true;
    }
    cudaStreamCaptureStatus join(cudaStream_t stream)const {
        cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(stream,&capture));
        if(pending_ && stream!=stream_)
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(stream,done_->get(),
                capture==cudaStreamCaptureStatusNone?0:cudaEventWaitExternal));
        return capture;
    }
    void record_consumer(cudaStream_t stream,cudaStreamCaptureStatus capture) {
        // Internal graph event records cannot be synchronized by host uploads.
        if(capture==cudaStreamCaptureStatusNone)record_transfer(stream);
    }
};
}

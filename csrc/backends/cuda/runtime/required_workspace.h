#pragma once
#include "row_stage.h"
#include "pinned_allocation_retry.h"
#include "qsa_ram_cache.h"
#include "decode_window.h"
#include "mfq_cuda_context.h"
#include "mfq_cuda_resident_allocations.h"
#include <cstdio>
#include <thread>

namespace mfq::cuda {
// This lease is acquired at a request/chunk planning boundary, before any
// mutable model state advances. Owners retire their finite GPU/CPU consumers
// before their buffers return to the allocator.
class RequiredWorkspaceLease {
    int device_;
    cudaStream_t stream_;
    std::thread::id thread_=std::this_thread::get_id();
    std::vector<std::shared_ptr<RowStage>> stages_;
public:
    RequiredWorkspaceLease(int device,cudaStream_t stream):device_(device),stream_(stream){}
    template<class Factory>
    std::shared_ptr<RowStage> acquire(const RowStageResources& plan,Factory&& factory) {
        if(thread_!=std::this_thread::get_id() || DecodeWindow::recording())
            throw std::logic_error("Required workspace admission must precede graph execution on the model thread");
        cudaStreamCaptureStatus capture;
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(stream_,&capture));
        if(capture!=cudaStreamCaptureStatusNone)
            throw std::logic_error("Required workspace admission cannot allocate during capture");
        std::fprintf(stderr,"native_required_workspace_plan role=%s rows=%llu width=%llu host_bytes=%zu pinned_bytes=%zu gpu_bytes=%zu before_state_advance=1\n",
            plan.role,static_cast<unsigned long long>(plan.rows),static_cast<unsigned long long>(plan.width),
            plan.host_bytes,plan.pinned_bytes,plan.gpu_bytes);
        std::shared_ptr<RowStage> stage;
        try {
            const ResidentAllocationScope resident;
            stage=allocate_with_pinned_recovery(plan.pinned_bytes,true,std::forward<Factory>(factory),
                [&](std::size_t bytes){return reclaim_qsa_pinned(device_,stream_,bytes);},[](std::size_t){});
        }catch(const std::exception& error) {
            std::fprintf(stderr,"native_required_workspace_failure role=%s pinned_bytes=%zu reason=%s\n",
                plan.role,plan.pinned_bytes,error.what());
            throw;
        }
        stages_.push_back(stage);
        return stage;
    }
};
} // namespace mfq::cuda

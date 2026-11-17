#pragma once
#include "mfq_cuda_context.h"
#include "runtime/routed_prefill_capacity.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfq::cuda {
inline std::size_t routed_prefill_quota(std::size_t experts,std::size_t budget,std::size_t payload) {
    return payload?std::min(experts,std::size_t(static_cast<long double>(experts)*std::min(budget,payload)/payload)):0;
}

// A non-owning stage also supports persistent layer images whose events are
// retained by their cache across asynchronous lookahead submissions.
class RoutedPrefillStageFence {
    cudaEvent_t gate_up_,ready_,consumed_;
public:
    RoutedPrefillStageFence(cudaEvent_t gate_up,cudaEvent_t ready,cudaEvent_t consumed)
        :gate_up_(gate_up),ready_(ready),consumed_(consumed) {}
    void begin_upload(cudaStream_t upload,bool has_consumer=true)const {
        if(has_consumer)TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(upload,consumed_,0));
    }
    void join(cudaStream_t compute,bool gate_up_only=false)const {
        TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(compute,gate_up_only?gate_up_:ready_,0));
    }
    void record_gate_up(cudaStream_t upload)const {TYLOINFER_CUDA_CHECK(cudaEventRecord(gate_up_,upload));}
    void record_ready(cudaStream_t upload)const {TYLOINFER_CUDA_CHECK(cudaEventRecord(ready_,upload));}
    void record_consumed(cudaStream_t compute)const {TYLOINFER_CUDA_CHECK(cudaEventRecord(consumed_,compute));}
    void wait_ready()const {TYLOINFER_CUDA_CHECK(cudaEventSynchronize(ready_));}
    void wait_consumed()const {TYLOINFER_CUDA_CHECK(cudaEventSynchronize(consumed_));}
};

// Two reusable weight stages, with separate Gate/Up and full-image readiness.
// Event ownership, lookahead, reuse and exceptional cleanup are format neutral.
class RoutedPrefillPipeline {
    cudaStream_t compute_,upload_;
    bool finished_=false;
    // Submit ready eager work before host staging can block. Completion
    // polling is optional and cannot serve as the driver's submission boundary.
    static void submit_eager(cudaStream_t stream) {
        cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
        TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stream,&capture));
        if(capture!=cudaStreamCaptureStatusNone)return;
        const auto status=cudaStreamQuery(stream);
        if(status!=cudaSuccess && status!=cudaErrorNotReady)TYLOINFER_CUDA_CHECK(status);
    }
public:
    static constexpr std::size_t depth=2;
    std::array<Event,depth> ready,gate_up_ready,consumed;
    RoutedPrefillPipeline(cudaStream_t compute,cudaStream_t upload):compute_(compute),upload_(upload){}
    RoutedPrefillPipeline(const RoutedPrefillPipeline&)=delete;
    RoutedPrefillPipeline& operator=(const RoutedPrefillPipeline&)=delete;
    ~RoutedPrefillPipeline(){finish();}
    RoutedPrefillStageFence stage(std::size_t group)const {
        const auto index=group%depth;
        return {gate_up_ready[index].get(),ready[index].get(),consumed[index].get()};
    }
    void begin_upload(std::size_t group) {
        stage(group).begin_upload(upload_,group>=depth);
    }
    void join(std::size_t group,bool gate_up_only=false) {
        stage(group).join(compute_,gate_up_only);
    }
    void finish()noexcept {
        if(std::exchange(finished_,true))return;
        try {
            Event done(cudaEventDisableTiming);done.record(compute_);
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(upload_,done.get(),0));
        }catch(...) {(void)cudaStreamSynchronize(compute_);}
    }
    // Phased adapters record Gate/Up and Down readiness at their actual
    // upload boundaries. Stage reuse and host lookahead follow the same order.
    template<class Acquire,class Compute,class Release>
    void run_phased(std::size_t groups,Acquire acquire,Compute compute,Release release) {
        const auto upload=[&](std::size_t group) {
            if(group>=groups)return;
            begin_upload(group);acquire(group);submit_eager(upload_);
        };
        try {
            if(groups)upload(0);
            for(std::size_t group=0;group<groups;++group) {
                // Submit ready work before the next host acquisition can wait
                // for storage or a pinned staging slot. Its upload still
                // overlaps the already submitted compute on the other stream.
                compute(group);stage(group).record_consumed(compute_);submit_eager(compute_);
                release(group);upload(group+1);
            }
        }catch(...) {finish();throw;}
        finish();
    }
    template<class Acquire,class Compute,class Release>
    void run(std::size_t groups,Acquire acquire,Compute compute,Release release) {
        run_phased(groups,[&](std::size_t group) {
            acquire(group);stage(group).record_ready(upload_);
        },std::move(compute),std::move(release));
    }
    template<class Acquire,class Compute>
    void run(std::size_t groups,Acquire acquire,Compute compute) {
        run(groups,std::move(acquire),std::move(compute),[](std::size_t){});
    }
};
} // namespace mfq::cuda

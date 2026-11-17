#pragma once
#include "cuda_execution.h"
#include "runtime/decode_window.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>

// The independent projection forks after its inputs are ready and joins
// before its output is consumed. Routed weights retain their own DMA lane.
template<class Main>
std::optional<mfq_tensor_backend::Tensor> try_independent_branch(CudaIndependentBranchExecutor* executor,
        const std::function<void()>& side,Main&& main,std::int64_t tokens,
        cudaStream_t transfer=nullptr) {
    auto* window=mfq::cuda::DecodeWindow::recording();
    if(!executor || !side || (window && window->layer_profiling()))return std::nullopt;
    const auto parent=mfq_get_current_cuda_stream();
    if(!executor->ensure(1,parent))return std::nullopt;
    const auto branch=executor->streams[0];
    if(branch.stream()==parent.stream() || branch.stream()==transfer)return std::nullopt;
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(parent.stream(),&capture));
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if(window)window->prepare_branch_stream(branch,executor->graph_pools[0]);
#endif
    TYLOINFER_CUDA_CHECK(cudaEventRecord(executor->ready,parent.stream()));
    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(branch.stream(),executor->ready,0));
    const char* phase="shared";
    try {
        {const MfqCudaStreamGuard guard(branch);side();}
        phase="shared_complete";
        TYLOINFER_CUDA_CHECK(cudaEventRecord(executor->completed[0],branch.stream()));
        phase="routed";
        auto output=main();
        phase="join";
        TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(parent.stream(),executor->completed[0],0));
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            static std::atomic<std::uint64_t> seen[2]{};
            const auto bit=std::uint64_t(1)<<std::min(tokens,std::int64_t(63));
            const int captured=capture!=cudaStreamCaptureStatusNone;
            if(!(seen[captured].fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
                "native_shared_stream actual_match=1 rows=%lld fork_join=1 shared_parallel_resident=1 captured=%d\n",
                static_cast<long long>(tokens),captured);
        }
        return output;
    }catch(...) {
        std::fprintf(stderr,"native_shared_stream_error phase=%s rows=%lld captured=%d\n",
            phase,static_cast<long long>(tokens),int(capture!=cudaStreamCaptureStatusNone));
        // Preserve the dependency even if shared/main preparation throws;
        // temporary branch allocations must not race the parent's recovery.
        (void)cudaEventRecord(executor->completed[0],branch.stream());
        (void)cudaStreamWaitEvent(parent.stream(),executor->completed[0],0);
        throw;
    }
}

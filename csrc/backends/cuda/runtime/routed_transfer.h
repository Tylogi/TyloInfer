#pragma once
#include "runtime/native_dma_batch.h"
#include "runtime/native_stream_signal.h"
#include "runtime/routed_completion_order.h"
#include "runtime/decode_window.h"
#include <algorithm>
#include <cstdlib>
#include <utility>

namespace mfq::cuda {
inline RoutedCompletionPosition record_routed_completion(RoutedCompletionClock& clock,
        int device,cudaStream_t stream) {
    // Implicit stream handles can identify different per-thread queues.
    if(!stream || stream==cudaStreamPerThread)return {};
    return clock.record(device,reinterpret_cast<std::uintptr_t>(stream));
}

inline int routed_dma_mode() {
    const auto* option=std::getenv("TYLOINFER_MOE_NATIVE_DMA_BATCH");
    return option?std::clamp(std::atoi(option),0,2):2;
}

inline void validate_routed_transfers(const std::vector<DmaCopy>& copies) {
    for(const auto& copy:copies)
        if(!copy.destination || !copy.source || !copy.bytes)
            throw std::invalid_argument("empty routed transfer field");
}

// Every source and destination remains owned until stream completion. The
// alias callback supplies registered device addresses for native batches.
template<class Alias,class Fallback>
bool submit_routed_transfers(const std::vector<DmaCopy>& copies,cudaStream_t stream,
        int mode,Alias&& alias,Fallback&& fallback) {
    // Validate the complete packet before either producer can submit work.
    // The fallback may be a prebuilt DMA graph or ordinary async copies.
    validate_routed_transfers(copies);
    if(NativeDmaBatch::submit(copies,stream,mode,std::forward<Alias>(alias)))return true;
    if(!copies.empty())std::forward<Fallback>(fallback)();
    return false;
}

template<class Alias>
bool submit_routed_transfers(const std::vector<DmaCopy>& copies,cudaStream_t stream,
        int mode,Alias&& alias) {
    return submit_routed_transfers(copies,stream,mode,std::forward<Alias>(alias),[&] {
        for(const auto& copy:copies)
            TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(copy.destination,copy.source,copy.bytes,cudaMemcpyDefault,stream));
    });
}

// Native batches address registered host memory through its device alias.
// Ordinary copies retain the original CPU address, in either direction.
template<class HostAlias>
bool submit_routed_host_transfers(const std::vector<DmaCopy>& copies,cudaMemcpyKind kind,
        cudaStream_t stream,int mode,HostAlias&& host_alias) {
    if(kind!=cudaMemcpyHostToDevice && kind!=cudaMemcpyDeviceToHost)
        throw std::invalid_argument("routed host transfer requires an explicit direction");
    validate_routed_transfers(copies);
    std::vector<DmaCopy> aliases;
    bool mapped=mode && !copies.empty() && NativeDmaBatch::available();
    if(mapped) {
        aliases=copies;
        for(auto& copy:aliases) {
            const auto* host=kind==cudaMemcpyHostToDevice?copy.source:copy.destination;
            const auto* address=host_alias(host,copy.bytes);
            if(!address){mapped=false;break;}
            if(kind==cudaMemcpyHostToDevice)copy.source=address;
            else copy.destination=const_cast<void*>(static_cast<const void*>(address));
        }
    }
    return submit_routed_transfers(mapped?aliases:copies,stream,mapped?mode:0,
        [](const void* pointer,std::size_t){return reinterpret_cast<CUdeviceptr>(pointer);},[&] {
            for(const auto& copy:copies)
                TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(copy.destination,copy.source,copy.bytes,kind,stream));
        });
}

// Stream writes and host callbacks both publish after preceding DMA. Neither
// requires the serving thread to synchronize the transfer stream.
inline bool enqueue_routed_completion(std::uint32_t* host,std::uint32_t* device,
        cudaStream_t stream,bool native=true) {
    if(!host)throw std::invalid_argument("missing routed completion flag");
    if(native && NativeStreamSignal::submit(device,stream))return true;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaLaunchHostFunc(stream,[](void* flag) {
        publish_mapped_flag(static_cast<std::uint32_t*>(flag));
    },host));
    return false;
}
}

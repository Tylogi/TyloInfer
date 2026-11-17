#pragma once
#include "mfq_cuda_context.h"
#include "runtime/routed_transfer.h"
#include "runtime/moe_pipeline.h"
#include "runtime/decode_window.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

// Outside the captured graph, the serving caller knows the current layer's
// source pointers. Submit copies to an independent, uncaptured stream and
// publish readiness in stream order. The captured graph retains fixed flag
// addresses; compressed expert bytes and destinations never change encoding.
class GgufDecodeDma {
    mfq::cuda::StreamHandle stream_;
    int native_mode_;
    std::mutex mutex_;
    std::vector<mfq::cuda::DmaCopy> phases_[2];
    bool stream_signals_=[] {
        const auto* option=std::getenv("TYLOINFER_GGUF_DECODE_DMA_SIGNAL");
        return !option || option[0]!='0';
    }();
    std::atomic<std::uint64_t> bytes_{0},copies_{0},batches_{0},signals_{0};
public:
    static bool default_enabled() {
        // Individual driver submissions can outweigh graph-copy savings for
        // small expert blobs. Older drivers retain the captured SM-copy path;
        // the explicit DMA override also exercises its individual-copy fallback.
        const auto* option=std::getenv("TYLOINFER_MOE_NATIVE_DMA_BATCH");
        return (!option || std::atoi(option)>0) && mfq::cuda::NativeDmaBatch::available();
    }
    explicit GgufDecodeDma(int device):stream_(mfq::cuda::stream_from_pool(false,device)),
        native_mode_(mfq::cuda::routed_dma_mode()) {if(!mfq::cuda::NativeDmaBatch::available())native_mode_=0;}
    void synchronize() {
        const std::lock_guard lock(mutex_);
        const mfq::cuda::DeviceGuard guard(stream_.device());
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream_.stream()));
    }
    ~GgufDecodeDma(){try{synchronize();}catch(...) {}}
    void submit(const mfq::cuda::MoeMappedCopyDescriptor* descriptors,std::size_t count,
            std::size_t split,std::uint32_t* gu_host,std::uint32_t* down_host,
            std::uint32_t* gu_device=nullptr,std::uint32_t* down_device=nullptr) {
        const std::lock_guard lock(mutex_);
        const mfq::cuda::DeviceGuard guard(stream_.device());
        for(auto& phase:phases_)phase.clear();
        for(std::size_t i=0;i<count;++i) {
            const auto& d=descriptors[i];
            if(!d.source || !d.destination || !d.bytes || d.offset>UINT64_MAX-d.bytes ||
                    (d.offset<split && d.bytes>split-d.offset))
                throw std::invalid_argument("invalid decode DMA descriptor or phase boundary");
            phases_[d.offset>=split].push_back({reinterpret_cast<void*>(d.destination),
                reinterpret_cast<const void*>(d.source),std::size_t(d.bytes)});
        }
        // An empty phase has no pending producer to wait for.
        for(int phase=0;phase<2;++phase) {
            const auto& copies=phases_[phase];
            auto* host=phase?down_host:gu_host;
            if(copies.empty()) {
                mfq::cuda::publish_mapped_flag(host);
                continue;
            }
            const bool batched=mfq::cuda::submit_routed_transfers(copies,stream_.stream(),native_mode_,
                [](const void* mapped,std::size_t){return reinterpret_cast<CUdeviceptr>(mapped);});
            if(mfq::cuda::enqueue_routed_completion(host,phase?down_device:gu_device,stream_.stream(),stream_signals_))
                signals_.fetch_add(1,std::memory_order_relaxed);
            if(batched)batches_.fetch_add(1,std::memory_order_relaxed);
            copies_.fetch_add(copies.size(),std::memory_order_relaxed);
            for(const auto& copy:copies)bytes_.fetch_add(copy.bytes,std::memory_order_relaxed);
        }
        // Submit this lane's new WDDM work before the caller waits for the
        // next layer's route publication on a different stream.
        mfq::cuda::check_stream_progress(stream_.stream());

    }
    std::uint64_t bytes()const{return bytes_.load(std::memory_order_relaxed);}
    std::uint64_t copies()const{return copies_.load(std::memory_order_relaxed);}
    std::uint64_t batches()const{return batches_.load(std::memory_order_relaxed);}
    std::uint64_t signals()const{return signals_.load(std::memory_order_relaxed);}
};

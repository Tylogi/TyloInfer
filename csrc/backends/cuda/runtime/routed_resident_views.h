#pragma once
#include "mfq_cuda_context.h"
#include "mfq_native_tensor.h"
#include "mfq_tensor_backend.h"
#include "runtime/decode_window.h"
#include "runtime/routed_layer_admission.h"
#include <array>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace mfq::cuda {
// Publish an expert only when all of its ordinary projections are available.
// Adapters retain their encoded view type; residency and publication are shared.
template<class View> struct RoutedResidentViews {
    static_assert(std::is_trivially_copyable_v<View>);
    std::array<std::uint64_t,3> sources;
    std::int64_t experts;
    int device;
    std::vector<View> raw;
    std::vector<std::uint8_t> valid;
    std::unique_ptr<HostBuffer> host;
    std::size_t admission_offset;
    void* mapped=nullptr;
    bool rebuild=false;
    static std::size_t storage_bytes(std::int64_t count) {
        if(count<1 || std::uint64_t(count)>(SIZE_MAX-127)/(3*sizeof(View)))
            throw std::overflow_error("resident expert view table exceeds addressable storage");
        return ((3*std::size_t(count)*sizeof(View)+63)&~std::size_t(63))+64;
    }
    static std::int64_t checked_experts(std::int64_t count){(void)storage_bytes(count);return count;}
    RoutedResidentViews(std::array<std::uint64_t,3> ids,std::int64_t count,int ordinal)
        :sources(ids),experts(checked_experts(count)),device(ordinal),raw(3*std::size_t(experts)),valid(raw.size()),
         host(std::make_unique<HostBuffer>(storage_bytes(count),true)),admission_offset(host->size()-64) {
        TYLOINFER_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(&mapped,host->data(),0));
        std::memset(host->data(),0,host->size());publish_mapped_flag(admission_host());
        charge_tensor_host_bytes(host->size()+raw.size()*sizeof(View)+valid.size());
    }
    ~RoutedResidentViews(){tensor_host_bytes.fetch_sub(host->size()+raw.size()*sizeof(View)+valid.size());}
    View* pointers(int role){return static_cast<View*>(host->data())+role*experts;}
    const View* device_pointers(int role)const{return static_cast<const View*>(mapped)+role*experts;}
    std::uint32_t* admission_host()const {
        return reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(host->data())+admission_offset);
    }
    std::uint32_t* admission_device()const {
        return reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(mapped)+admission_offset);
    }
    void update(int role,std::int64_t expert,View view,bool present) {
        if(role<0 || role>=3 || expert<0 || expert>=experts)throw std::out_of_range("resident expert view index");
        raw[role*experts+expert]=view;valid[role*experts+expert]=present;
        const bool complete=valid[expert] && valid[experts+expert] && valid[2*experts+expert];
        for(int bank=0;bank<3;++bank)pointers(bank)[expert]=complete?raw[bank*experts+expert]:View{};
    }
    void invalidate() {
        std::memset(host->data(),0,raw.size()*sizeof(View));rebuild=true;
    }
    void republish() {
        for(std::int64_t expert=0;expert<experts;++expert) {
            const bool complete=valid[expert] && valid[experts+expert] && valid[2*experts+expert];
            for(int role=0;role<3;++role)pointers(role)[expert]=complete?raw[role*experts+expert]:View{};
        }
        rebuild=false;
    }
};

template<class Epoch>
void arm_routed_layer_admission(Epoch& epoch,mfq_tensor_backend::Tensor& word,cudaStream_t stream) {
    if(epoch.views.empty())throw std::invalid_argument("layer admission requires resident views");
    if(!word.defined()) {
        word=mfq_tensor_backend::full({1},1,mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA,epoch.device)).dtype(mfq_tensor_backend::kInt32));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(epoch.views.front()->admission_host(),word.data_ptr(),
            sizeof(std::uint32_t),cudaMemcpyDeviceToHost,stream));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    epoch.ready_word=word;
    epoch.submitted_done=std::make_shared<Event>();
    epoch.arm();
}

template<class Epoch>
void publish_routed_admission_layers(const Epoch& epoch,const std::vector<std::uint32_t*>& flags,cudaStream_t stream) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    // Copy-engine landing words progress while a captured kernel waits on
    // mapped memory, including drivers that serialize compute submissions.
    for(auto* flag:flags)TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(flag,epoch.ready_word.data_ptr(),
        sizeof(std::uint32_t),cudaMemcpyDeviceToHost,stream));
    const auto progress=cudaStreamQuery(stream);
    if(progress!=cudaSuccess && progress!=cudaErrorNotReady)TYLOINFER_NATIVE_CUDA_CHECK(progress);
}
} // namespace mfq::cuda

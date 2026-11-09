#pragma once

#include "mfq_cuda_context.h"
#include "runtime/mapped_storage.h"
#include "mfq_tensor_backend.h"

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>

namespace mfq::cuda {

// A load owns its budget. Nested loads restore their caller's placement policy.
class DenseWeightPlacementScope {
    inline static thread_local DenseWeightPlacementScope* current_ = nullptr;
    DenseWeightPlacementScope* previous_ = nullptr;
    std::size_t remaining_ = 0;
    bool enabled_ = false;
public:
    explicit DenseWeightPlacementScope(bool dense,std::optional<std::size_t> bytes=std::nullopt)
        : previous_(current_) {
        if(dense && bytes) {
            remaining_=*bytes;enabled_=true;
        } else if(dense)if(const auto* text=std::getenv("TYLOINFER_DENSE_GPU_WEIGHT_BUDGET_BYTES")) {
            const auto end=text+std::strlen(text);
            const auto [parsed,error]=std::from_chars(text,end,remaining_);
            if(error!=std::errc{} || parsed!=end)
                throw std::invalid_argument("dense GPU weight budget must be a non-negative integer");
            enabled_=true;
        }
        current_=this;
    }
    ~DenseWeightPlacementScope() {current_=previous_;}
    DenseWeightPlacementScope(const DenseWeightPlacementScope&)=delete;
    DenseWeightPlacementScope& operator=(const DenseWeightPlacementScope&)=delete;
    static bool limited() { return current_ && current_->enabled_; }
    static bool place_on_host(std::size_t bytes) {
        if(!current_ || !current_->enabled_)return false;
        if(bytes>current_->remaining_)return true;
        current_->remaining_-=bytes;
        return false;
    }
};

inline bool host_mapped_weight(const mfq_tensor_backend::Tensor& tensor) {
    if(!tensor.defined() || !tensor.is_cuda())return false;
    cudaPointerAttributes attributes{};
    TYLOINFER_CUDA_CHECK(cudaPointerGetAttributes(&attributes,tensor.data_ptr()));
    return attributes.type==cudaMemoryTypeHost;
}

// The immutable host image stays pinned for every graph that references it.
inline mfq_tensor_backend::Tensor map_dense_weight_storage(
        const mfq_tensor_backend::Tensor& cpu,int device) {
    return map_pinned_tensor_storage(cpu,device);
}

inline mfq_tensor_backend::Tensor clone_mapped_weight_storage(const mfq_tensor_backend::Tensor& tensor) {
    namespace tb=mfq_tensor_backend;
    cudaPointerAttributes attributes{};
    TYLOINFER_CUDA_CHECK(cudaPointerGetAttributes(&attributes,tensor.data_ptr()));
    if(attributes.type!=cudaMemoryTypeHost || !attributes.hostPointer)
        throw std::invalid_argument("mapped weight clone requires a host allocation");
    const auto cpu=tb::from_blob(attributes.hostPointer,tensor.sizes(),
        tensor.options().device(tb::kCPU));
    return map_dense_weight_storage(cpu,tensor.get_device());
}

} // namespace mfq::cuda

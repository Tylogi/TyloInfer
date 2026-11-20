#pragma once
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include "tyloinfer/runtime.h"

namespace mfq::cuda {
struct DeferredPinnedAllocation:MfqResourceExhausted {
    std::size_t bytes;
    DeferredPinnedAllocation(std::size_t n,const char* message):MfqResourceExhausted(message),bytes(n){}
};
// Only retry a failed allocation before any consumer owns its storage. Quota
// reclamation must run on the model thread, never a graph/codec callback.
template<class Allocate,class Reclaim,class Defer>
auto allocate_with_pinned_recovery(std::size_t bytes,bool foreground,
        Allocate&& allocate,Reclaim&& reclaim,Defer&& defer) -> decltype(allocate()) {
    try{return allocate();}catch(const std::exception& error) {
        if(!std::string_view(error.what()).starts_with("CUDA pinned allocation failed:"))throw;
        if(!foreground) {
            defer(bytes);
            throw DeferredPinnedAllocation(bytes,error.what());
        }
        if(!reclaim(bytes))throw;
    }
    return allocate();
}
} // namespace mfq::cuda

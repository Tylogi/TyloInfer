#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace mfq::cuda {
// Follow Context's best-fit loan allocation before ordinary allocation.
// Independent intervals never contribute capacity to the same tensor.
inline bool routed_workspace_allocations_fit(const std::vector<std::size_t>& allocations,
        std::vector<std::size_t> intervals,std::size_t ordinary) {
    for(const auto bytes:allocations) {
        if(!bytes)continue;
        auto best=intervals.end();
        if(bytes<=SIZE_MAX-255) {
            const auto aligned=(bytes+255)&~std::size_t(255);
            for(auto it=intervals.begin();it!=intervals.end();++it)
                if(*it>=aligned && (best==intervals.end() || *it<*best))best=it;
            if(best!=intervals.end()) {*best-=aligned;continue;}
        }
        if(bytes>ordinary)return false;
        ordinary-=bytes;
    }
    return true;
}

// Context loans align each independent backing region to 256 bytes. A sum of
// separate holes is never a capacity guarantee for one allocation.
inline std::size_t routed_workspace_region_bytes(const void* pointer,std::size_t bytes)noexcept {
    const auto address=reinterpret_cast<std::uintptr_t>(pointer);
    const auto padding=(256-(address&255))&255;
    return bytes>padding?(bytes-padding)&~std::size_t(255):0;
}

// Reserve the complete ordinary working set before pinning an existing cache
// hole. Its large allocations can then fall back without depending on the
// distribution, fragmentation or stream eligibility of borrowed intervals.
// The optional loan covers only measured pressure, never a fixed arena size.
inline std::size_t routed_workspace_borrow_request(std::size_t required,std::size_t before,
        std::size_t ordinary,std::size_t largest_hole)noexcept {
    if(before>=required || ordinary<required || largest_hole<=256)return 0;
    const auto maximum=(largest_hole-256)&~std::size_t(255);
    if(!maximum)return 0;
    const auto pressure=required-before;
    const auto payload=pressure>=maximum?maximum:(pressure+255)&~std::size_t(255);
    // Keep room for alignment even when an arena hole starts at a 16-byte
    // boundary. DeviceImageArena rounds this request to a multiple of 16.
    return payload+256;
}

inline std::size_t routed_workspace_full_borrow_request(std::size_t required,std::size_t largest_hole)noexcept {
    if(!required || required>largest_hole)return 0;
    if(required>SIZE_MAX-511)return largest_hole;
    return std::min(largest_hole,((required+255)&~std::size_t(255))+256);
}

// The backend supplies storage and an owning stream handle. Successful loans
// give Context shared backing ownership: an abandoned output or exceptional
// live-tensor check must never release its backing allocation. Retirement
// returns the region after the last live allocation, without an owner cycle.
// Graph pools and resident allocations keep Context's existing bypass rules.
template<class Context,class Storage,class Stream>
class RoutedPrefillLoanState {
    std::shared_ptr<Context> context_;
    Stream stream_;
    std::shared_ptr<Storage> storage_;
    std::thread::id thread_=std::this_thread::get_id();
    bool active_=false;
    RoutedPrefillLoanState(std::shared_ptr<Context> context,Stream stream,Storage storage)
        :context_(std::move(context)),stream_(std::move(stream)),storage_(std::make_shared<Storage>(std::move(storage))) {}
    void check_thread()const {
        if(thread_!=std::this_thread::get_id())
            throw std::logic_error("prefill workspace loan belongs to another thread");
    }
public:
    static std::shared_ptr<RoutedPrefillLoanState> begin(std::shared_ptr<Context> context,
            Stream stream,Storage storage) {
        auto owner=std::shared_ptr<RoutedPrefillLoanState>(
            new RoutedPrefillLoanState(std::move(context),std::move(stream),std::move(storage)));
        owner->context_->begin_workspace_loan({{owner->storage_->data_ptr(),owner->storage_->nbytes()}},owner->stream_.stream(),owner->storage_);
        owner->active_=true;return owner;
    }
    ~RoutedPrefillLoanState(){if(active_)context_->retire_workspace_loan();}
    bool matches(const std::shared_ptr<Context>& context,const Stream& stream)const noexcept {
        return context_.get()==context.get() && stream_.stream()==stream.stream() && thread_==std::this_thread::get_id();
    }
    bool active()const noexcept{return active_;}
    const Stream& stream()const noexcept{return stream_;}
    std::size_t bytes()const noexcept {
        return active_?routed_workspace_region_bytes(storage_->data_ptr(),storage_->nbytes()):0;
    }
    void suspend() {
        check_thread();if(active_)context_->suspend_workspace_loan();
    }
    void finish() {
        check_thread();if(!active_)return;
        // The caller fences the owner stream before attempting this check.
        // Failure leaves both Context's backing owner and this handle intact.
        context_->end_workspace_loan();active_=false;
        storage_.reset();
    }
};
} // namespace mfq::cuda

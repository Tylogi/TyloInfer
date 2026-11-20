#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfq::cuda {
// Cache owners serialize metadata changes. Each layer releases its readers
// only after all three projection uploads have reached the same stream.
template<class View,class Completion,class Word> struct RoutedLayerAdmission {
    int device=-1;
    Word ready_word;
    std::shared_ptr<Completion> submitted_done;
    bool cancelled=false;
    std::exception_ptr error;
    std::vector<std::shared_ptr<View>> views;

    void arm() {
        for(const auto& view:views)*reinterpret_cast<volatile std::uint32_t*>(view->admission_host())=0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
    void require_active()const {
        if(error)std::rethrow_exception(error);
        if(cancelled)throw std::runtime_error("routed layer admission cancelled");
    }
    template<class Pending,class Publish> void release_unchanged(Pending&& pending,Publish&& publish) {
        if(cancelled)return;
        for(const auto& view:views)if(!pending(*view))publish(view->admission_host());
    }
    template<class Invalidate,class Publish> void abort(std::exception_ptr failure,Invalidate&& invalidate,Publish&& publish) {
        if(failure)error=std::move(failure);
        if(cancelled)return;
        cancelled=true;
        for(const auto& view:views)invalidate(*view);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        for(const auto& view:views)publish(view->admission_host());
    }
};

template<class Ready,class Pause,class Timeout>
void wait_routed_layer_admission(Ready&& ready,Pause&& pause,Timeout&& timed_out,
        std::chrono::steady_clock::duration limit=std::chrono::seconds(20)) {
    const auto started=std::chrono::steady_clock::now();
    std::uint32_t spins=0;
    while(!ready()) {
        pause();
        if((++spins&1023u)==0 && std::chrono::steady_clock::now()-started>=limit) {
            timed_out();
            throw std::runtime_error("routed layer admission timed out");
        }
    }
    std::atomic_thread_fence(std::memory_order_acquire);
}
} // namespace mfq::cuda

#pragma once
#include "mfq_tensor_backend.h"
#include <functional>
#include <memory>
#include <utility>
#include <vector>
struct Block;
namespace mfq::cuda {
struct LayerWeightBinding {
    mfq_tensor_backend::Tensor* tensor=nullptr;
    std::function<void(mfq_tensor_backend::Tensor)> assign;
};
// Compute one layer while copying the next. Immutable weights alone are
// rebound; attention caches and recurrent state retain their device storage.
class LayerWeightStream {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void leave() noexcept;
public:
    class Lease {
        LayerWeightStream* owner_=nullptr;
        friend class LayerWeightStream;
        explicit Lease(LayerWeightStream* owner):owner_(owner){}
    public:
        Lease()=default;
        Lease(Lease&& other)noexcept:owner_(std::exchange(other.owner_,nullptr)){}
        Lease(const Lease&)=delete;
        ~Lease(){if(owner_)owner_->leave();}
    };
    explicit LayerWeightStream(int device,bool timings=false);
    ~LayerWeightStream();
    void add(Block& block);
    void add(const void* identity,std::vector<LayerWeightBinding> bindings);
    void prepare();
    Lease enter(const void* identity);
    void prefetch_next() noexcept;
    void report_timings();
    std::size_t host_bytes()const;
    std::size_t device_bytes()const;
};
void prefetch_active_layer_weights() noexcept;
struct LayerWeightPrefetchKick {
    ~LayerWeightPrefetchKick(){prefetch_active_layer_weights();}
};
} // namespace mfq::cuda

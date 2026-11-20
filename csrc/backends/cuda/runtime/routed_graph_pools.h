#pragma once
#include "mfq_cuda_context.h"
#include "mfq_tensor_backend.h"
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace mfq::cuda {
// Cache owners join admission and hold their metadata lock before preparing
// a graph pool. Retain the execution stream until its pool has been released.
class RoutedGraphPools {
    struct Key {
        cudaStream_t stream;
        int device;
        bool operator==(const Key&)const=default;
    };
    struct Hash {
        std::size_t operator()(const Key& key)const noexcept {
            return std::hash<cudaStream_t>{}(key.stream)^(std::hash<int>{}(key.device)<<1);
        }
    };
    struct Pool {MfqCudaStream stream;std::shared_ptr<Context> context;};
    std::unordered_map<Key,Pool,Hash> pools_;
public:
    RoutedGraphPools()=default;
    RoutedGraphPools(const RoutedGraphPools&)=delete;
    RoutedGraphPools& operator=(const RoutedGraphPools&)=delete;
    std::shared_ptr<Context> prepare(MfqCudaStream stream) {
        const Key key{stream.stream(),stream.device()};
        if(const auto found=pools_.find(key);found!=pools_.end())return found->second.context;
        auto context=default_context(stream.device());context->begin_graph_pool(stream.stream(),true);
        try{pools_.emplace(key,Pool{stream,context});}
        catch(...){context->end_graph_pool(stream.stream());throw;}
        return context;
    }
    // One shared set of mutable pipeline buffers requires one execution lane.
    std::shared_ptr<Context> prepare_shared_buffers(MfqCudaStream stream) {
        if(!pools_.empty() && !pools_.count(Key{stream.stream(),stream.device()}))
            throw std::logic_error("shared graph buffers cannot cross execution streams");
        return prepare(std::move(stream));
    }
    void synchronize()const {
        for(const auto& [_,pool]:pools_) {
            const MfqCudaGuard guard(pool.stream.device());
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(pool.stream.stream()));
        }
    }
    void clear()noexcept {
        for(auto& [_,pool]:pools_)pool.context->end_graph_pool(pool.stream.stream());
        pools_.clear();
    }
    ~RoutedGraphPools(){clear();}
};
} // namespace mfq::cuda

#pragma once
#include "mfq_native_tensor.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

// Compressed images share backing pages. Callers order their reads before
// releasing leases; the arena never inserts waits behind future GPU work.
class DeviceImageArena {
    using Free=std::map<std::size_t,std::size_t>;
    struct State {
        mfq::cuda::Tensor storage;
        std::mutex mutex;
        Free free;
        std::atomic<std::size_t> used{0},largest{0};
        State(std::size_t bytes,int device):storage(mfq::cuda::empty({static_cast<int64_t>(bytes)},
                mfq::cuda::TensorOptions().device(mfq::cuda::Device{mfq::cuda::DeviceType::cuda,device})
                    .dtype(mfq::cuda::ScalarType::uint8))) {
            free.emplace(0,bytes);largest=bytes;
        }
        void update_largest()noexcept {
            std::size_t maximum=0;for(const auto& [_,bytes]:free)maximum=std::max(maximum,bytes);
            largest.store(maximum,std::memory_order_relaxed);
        }
    };
    struct Block {
        std::shared_ptr<State> state;
        Free::node_type interval;
        ~Block() {
            if(!state)return;
            std::lock_guard lock(state->mutex);
            state->used.fetch_sub(interval.mapped(),std::memory_order_relaxed);
            auto next=state->free.lower_bound(interval.key());
            if(next!=state->free.begin()) {
                const auto prior=std::prev(next);
                if(prior->first+prior->second==interval.key()) {
                    interval.key()=prior->first;interval.mapped()+=prior->second;state->free.erase(prior);
                }
            }
            if(next!=state->free.end() && interval.key()+interval.mapped()==next->first) {
                interval.mapped()+=next->second;state->free.erase(next);
            }
            state->free.insert(std::move(interval));state->update_largest();
        }
    };
    std::shared_ptr<State> state_;
public:
    DeviceImageArena(std::size_t bytes,int device) {
        if(!bytes || bytes>INT64_MAX)throw std::overflow_error("Device image arena size exceeds tensor limits");
        state_=std::make_shared<State>(bytes,device);
    }
    std::size_t bytes()const noexcept{return state_->storage.nbytes();}
    // A queued same-stream copy needs the backing pages, while its interval
    // may already be reused by later copies on that stream.
    std::shared_ptr<void> retain_backing()const noexcept{return state_;}
    std::size_t used_bytes()const noexcept{return state_->used.load(std::memory_order_relaxed);}
    std::size_t largest_available()const noexcept{return state_->largest.load(std::memory_order_relaxed);}
    bool owns(const mfq::cuda::Tensor& image)const noexcept {
        const auto base=reinterpret_cast<std::uintptr_t>(state_->storage.data_ptr());
        const auto address=reinterpret_cast<std::uintptr_t>(image.data_ptr());
        return address>=base && address-base<bytes();
    }
    mfq::cuda::Tensor allocate(std::size_t bytes) {
        if(!bytes || bytes>SIZE_MAX-15)return {};
        bytes=(bytes+15)&~std::size_t(15);
        if(state_->largest.load(std::memory_order_relaxed)<bytes)return {};
        auto block=std::make_shared<Block>();Free node;node.emplace(0,bytes);block->interval=node.extract(node.begin());
        auto storage=std::make_shared<mfq::cuda::TensorStorage>();
        std::size_t offset=0;
        {
            std::lock_guard lock(state_->mutex);
            auto found=std::find_if(state_->free.begin(),state_->free.end(),[&](const auto& item){return item.second>=bytes;});
            if(found==state_->free.end())return {};
            offset=found->first;
            if(found->second==bytes)state_->free.erase(found);
            else {
                auto rest=state_->free.extract(found);rest.key()+=bytes;rest.mapped()-=bytes;
                state_->free.insert(std::move(rest));
            }
            block->interval.key()=offset;block->state=state_;
            state_->used.fetch_add(bytes,std::memory_order_relaxed);state_->update_largest();
        }
        storage->owner=std::move(block);
        storage->base=static_cast<std::uint8_t*>(state_->storage.data_ptr())+offset;
        storage->bytes=bytes;storage->device=state_->storage.device();
        const std::array<int64_t,1> shape{static_cast<int64_t>(bytes)};
        return mfq::cuda::Tensor(storage,mfq::cuda::make_contiguous_view(storage->base,shape,
            mfq::cuda::ScalarType::uint8,storage->device));
    }
};

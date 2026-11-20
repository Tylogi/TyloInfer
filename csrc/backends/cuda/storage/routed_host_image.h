#pragma once
#include "storage/moe_pinned_expert_arena.h"
#include "mfq_tensor_backend.h"
#include "mfq_cuda_context.h"
#include <cstring>
#include <memory>
#include <stdexcept>

namespace mfq::cuda {
struct UnmappedExpertSnapshot {
    std::uint8_t* data(std::size_t)const{return nullptr;}
    const void* registration(std::size_t,std::size_t)const{return nullptr;}
    std::uint8_t* mapped_data(std::size_t,std::size_t,int)const {
        throw std::logic_error("expert image has no mapped source");
    }
};

// Immutable encoded bytes and their producer lifetime are independent of the
// projection's encoding. Snapshot providers expose bounded registered ranges.
template<class Snapshot=UnmappedExpertSnapshot> struct RoutedHostImage {
    std::shared_ptr<MoePinnedExpertArena::Block> block;
    std::size_t block_offset=0;
    mfq_tensor_backend::Tensor fallback;
    std::shared_ptr<Snapshot> snapshot;
    std::size_t snapshot_offset=0,bytes=0;
    std::shared_ptr<mfq::cuda::Event> ready;
    cudaStream_t producer=nullptr;
    bool ready_on_host=false;

    std::size_t size()const noexcept{return bytes;}
    std::uint8_t* data()const {
        return snapshot?snapshot->data(snapshot_offset):block?block->data()+block_offset:
            static_cast<std::uint8_t*>(fallback.data_ptr());
    }
    bool pinned()const {return block || (snapshot && snapshot->registration(snapshot_offset,bytes));}
    const void* registration_id()const {
        return block?block->allocation_id():snapshot?snapshot->registration(snapshot_offset,bytes):nullptr;
    }
    std::uint8_t* mapped_data(int device)const {
        if(block)return block->mapped_data()+block_offset;
        if(snapshot)return snapshot->mapped_data(snapshot_offset,bytes,device);
        throw std::logic_error("expert image is not registered");
    }
    std::size_t pageable_capacity()const noexcept {return fallback.defined()?fallback.nbytes():0;}
    void allocate(std::size_t count,std::shared_ptr<MoePinnedExpertArena::Block> pinned={}) {
        if(!count || count>INT64_MAX || (pinned && pinned->bytes()<count))
            throw std::invalid_argument("invalid expert host image allocation");
        auto ordinary=pinned?mfq_tensor_backend::Tensor{}:mfq_tensor_backend::empty({static_cast<int64_t>(count)},
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kUInt8));
        bytes=count;block=std::move(pinned);fallback=std::move(ordinary);block_offset=0;
    }
    bool pin(std::shared_ptr<MoePinnedExpertArena::Block> destination) {
        if(!destination || pinned())return false;
        if(destination->bytes()<bytes)throw std::invalid_argument("expert pin interval is too small");
        std::memcpy(destination->data(),data(),bytes);
        block=std::move(destination);block_offset=0;fallback={};snapshot.reset();return true;
    }
};
} // namespace mfq::cuda

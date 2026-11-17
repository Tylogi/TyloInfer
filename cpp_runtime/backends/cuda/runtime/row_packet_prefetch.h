#pragma once
#include "routed_admission_executor.h"
#include "row_stage.h"
#include "mfq_cuda_resident_allocations.h"
#include <array>
#include <atomic>
#include <cstring>
#include <functional>
#include <optional>

namespace mfq::cuda {
// One outstanding packet per stage bounds queued reads by live model stages.
// Workers touch CPU buffers only; the model thread retains all CUDA ownership.
class RowPacketPrefetch {
    std::future<void> ready_;
    std::vector<std::int64_t> ids_,shape_;
    std::vector<std::uint8_t> packet_;
    std::size_t used_=0;
    std::size_t charged_=0;
    bool prepared_=false;
    static RoutedAdmissionExecutor& worker() {
        static std::array<RoutedAdmissionExecutor,2> workers;
        static std::atomic<unsigned> next{0};
        return workers[next.fetch_add(1,std::memory_order_relaxed)%workers.size()];
    }
    void discard()noexcept {
        if(ready_.valid())ready_.wait();
        ready_={};prepared_=false;ids_.clear();shape_.clear();
    }
public:
    ~RowPacketPrefetch(){discard();tensor_host_bytes.fetch_sub(charged_);}
    template<class Prepare> void start(const std::vector<std::int64_t>& ids,
            const std::vector<std::int64_t>& shape,std::size_t bytes,Prepare&& prepare) {
        if(prepared_ && ids_==ids && shape_==shape)return;
        discard();packet_.resize(bytes);
        const auto capacity=packet_.capacity();
        if(capacity>charged_){charge_tensor_host_bytes(capacity-charged_);charged_=capacity;}
        ids_=ids;shape_=shape;
        ready_=worker().submit([this,prepare=std::forward<Prepare>(prepare)]() mutable {
            used_=prepare(packet_.data());
            if(used_>packet_.size())throw std::logic_error("prefetched row packet exceeds storage");
        });
        prepared_=true;
    }
    std::optional<std::size_t> consume(const std::vector<std::int64_t>& ids,
            const std::vector<std::int64_t>& shape,void* destination) {
        if(!prepared_ || ids_!=ids || shape_!=shape){discard();return {};}
        prepared_=false;ids_.clear();shape_.clear();
        if(ready_.valid())ready_.get();
        std::memcpy(destination,packet_.data(),used_);return used_;
    }
};

template<class Factory> std::function<mfq_tensor_backend::Tensor()> prefetch_table_rows(
        Factory&& factory,const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape) {
    std::shared_ptr<RowStage> stage;
    {const ResidentAllocationScope resident;stage=factory();}
    if(!stage->prefetch(ids,shape))return {};
    return [stage=std::move(stage),ids,shape] {
        stage->upload(ids,shape);stage->decode();return stage->output();
    };
}
}

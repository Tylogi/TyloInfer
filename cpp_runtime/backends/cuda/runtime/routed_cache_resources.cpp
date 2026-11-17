#include "runtime/routed_cache_resources.h"
#include "cuda_execution.h"
#include "tq.h"
#include "gguf_weight.h"
#include "mfq_cuda_context.h"
#include "runtime/routed_workspace_plan.h"
#include "runtime/routed_graph_pools.h"
#include <stdexcept>
#include <exception>
#include <array>
#include <algorithm>
#include <unordered_map>

namespace mfq::cuda {
namespace {
std::array<std::size_t,3> routed_source_weights(const CudaExecutionContext& execution) {
    return {gguf_expert_source_bytes(execution.gguf_expert_cache),
        tq_expert_source_bytes(execution.tq_expert_cache),
        moe_expert_source_bytes(execution.moe_expert_cache)};
}
}

bool has_routed_expert_cache(const CudaExecutionContext& execution) noexcept {
    return execution.moe_expert_cache || execution.gguf_expert_cache || execution.tq_expert_cache;
}

void finish_routed_expert_exchanges(const CudaExecutionContext& execution) {
    // Retire every owner before returning storage, including after a failed
    // admission. An earlier failure must not leave another worker using it.
    std::exception_ptr failure;
    const auto finish=[&](auto&& operation) {
        try {operation();}
        catch(...) {if(!failure)failure=std::current_exception();}
    };
    finish([&]{wait_gguf_expert_admission(execution.gguf_expert_cache);});
    finish([&]{wait_tq_expert_admission(execution.tq_expert_cache);});
    finish([&]{finish_moe_expert_exchanges(execution.moe_expert_cache);});
    if(failure)std::rethrow_exception(failure);
}

bool prepare_routed_graph_pool(const CudaExecutionContext& execution) {
    finish_routed_expert_exchanges(execution);
    bool prepared=prepare_moe_graph_pool(execution.moe_expert_cache);
    prepared=prepare_gguf_graph_pool(execution.gguf_expert_cache) || prepared;
    return prepare_tq_graph_pool(execution.tq_expert_cache) || prepared;
}

void prepare_execution_graph_pool(const CudaExecutionContext& execution,std::unique_ptr<RoutedGraphPools>& owned) {
    if(prepare_routed_graph_pool(execution))return;
    if(owned) {
        owned->prepare_shared_buffers(mfq_get_current_cuda_stream(mfq_current_cuda_device()));
        return;
    }
    auto pool=std::make_unique<RoutedGraphPools>();
    pool->prepare_shared_buffers(mfq_get_current_cuda_stream(mfq_current_cuda_device()));
    owned=std::move(pool);
}

void finalize_routed_weight_images(CudaExecutionContext& execution,std::size_t growth) {
    if(execution.config.moe_host_cache_bytes) {
        const auto weights=routed_source_weights(execution);
        if(std::count_if(weights.begin(),weights.end(),[](auto bytes){return bytes>0;})>1) {
            // Plan every immutable minimum before changing an admission cap.
            // Setters retain active leases and do not credit logical eviction
            // as physical headroom or allocate RAM when a cap grows.
            const std::array<std::size_t,3> minima{gguf_expert_host_minimum_bytes(execution.gguf_expert_cache),0,
                moe_expert_host_minimum_bytes(execution.moe_expert_cache)};
            const auto shares=routed_budget_shares(execution.config.moe_host_cache_bytes,weights,minima);
            resize_gguf_expert_host(execution.gguf_expert_cache,shares[0]);
            resize_tq_expert_host(execution.tq_expert_cache,shares[1]);
            resize_moe_expert_host(execution.moe_expert_cache,shares[2]);
        }
    }
    if(execution.config.moe_device_cache_bytes.has_value()) {
        const auto weights=routed_source_weights(execution);
        if(std::count_if(weights.begin(),weights.end(),[](auto bytes){return bytes>0;})>1)
            resize_routed_expert_cache(execution,*execution.config.moe_device_cache_bytes);
    }
    finalize_gguf_expert_cache(execution.gguf_expert_cache,growth);
    finalize_tq_expert_cache(execution.tq_expert_cache,growth);
}

MoePrefillWorkspaceBudget routed_prefill_workspace_budget(
        const CudaExecutionContext& execution,bool allow_borrow) {
    auto result=moe_prefill_workspace_budget(execution.moe_expert_cache,allow_borrow);
    if(allow_borrow)result.lendable_bytes+=gguf_prefill_lendable_bytes(execution.gguf_expert_cache)+
        tq_prefill_lendable_bytes(execution.tq_expert_cache);
    return result;
}

std::size_t routed_prefill_transfer_bytes(const CudaExecutionContext& execution) {
    return moe_prefill_transfer_bytes(execution.moe_expert_cache);
}

std::uint64_t routed_expert_streamed_bytes(const CudaExecutionContext& execution) {
    return gguf_expert_streamed_bytes(execution.gguf_expert_cache)+
        tq_expert_streamed_bytes(execution.tq_expert_cache)+moe_expert_streamed_bytes(execution.moe_expert_cache);
}

std::vector<std::pair<std::string,double>> routed_expert_memory_metrics(
        const CudaExecutionContext& execution) {
    std::vector<std::pair<std::string,double>> result;
    std::unordered_map<std::string,std::size_t> positions;
    double host_budget=0;
    const auto append=[&](const auto& values) {
        for(const auto& [name,value]:values) {
            if(name=="ram_expert_cache_budget_bytes" || name=="ram_expert_limit_bytes")host_budget+=value;
            const auto [position,inserted]=positions.emplace(name,result.size());
            if(inserted){result.emplace_back(name,value);continue;}
            auto& total=result[position->second].second;
            // These describe shared process capacity or a boolean state.
            if(name=="host_tensor_bytes" || name=="ram_expert_cache_budget_bytes" || name.ends_with("_enabled"))
                total=std::max(total,value);
            else total+=value;
        }
    };
    if(execution.gguf_expert_cache)append(gguf_expert_memory_metrics(execution.gguf_expert_cache));
    if(execution.tq_expert_cache)append(tq_expert_memory_metrics(execution.tq_expert_cache));
    if(execution.moe_expert_cache)append(moe_expert_memory_metrics(execution.moe_expert_cache));
    if(auto found=positions.find("expert_cache_budget_bytes");found!=positions.end())
        result[found->second].second=double(routed_expert_cache_budget_bytes(execution));
    if(execution.config.moe_host_cache_bytes) {
        const auto weights=routed_source_weights(execution);
        if(std::count_if(weights.begin(),weights.end(),[](auto bytes){return bytes>0;})>1) {
            const auto [found,inserted]=positions.emplace("ram_expert_cache_budget_bytes",result.size());
            if(inserted)result.emplace_back("ram_expert_cache_budget_bytes",host_budget);
            else result[found->second].second=host_budget;
        }
    }
    return result;
}

std::size_t routed_expert_cache_budget_bytes(const CudaExecutionContext& execution) {
    const auto weights=routed_source_weights(execution);
    return (weights[0]?gguf_expert_cache_budget_bytes(execution.gguf_expert_cache):0)+
        (weights[1]?tq_expert_cache_budget_bytes(execution.tq_expert_cache):0)+
        (weights[2]?moe_expert_cache_budget_bytes(execution.moe_expert_cache):0);
}

void resize_routed_expert_cache(CudaExecutionContext& execution,std::size_t bytes) {
    const auto weights=routed_source_weights(execution);
    const std::array<std::size_t,3> minima{0,0,moe_expert_cache_minimum_bytes(execution.moe_expert_cache)};
    const auto shares=routed_budget_shares(bytes,weights,minima);
    const std::array<std::size_t,3> previous{
        weights[0]?gguf_expert_cache_budget_bytes(execution.gguf_expert_cache):0,
        tq_expert_cache_budget_bytes(execution.tq_expert_cache),
        moe_expert_cache_budget_bytes(execution.moe_expert_cache)};
    const auto resize=[&](std::size_t index) {
        if(index==0)resize_gguf_expert_cache(execution.gguf_expert_cache,shares[index]);
        else if(index==1)resize_tq_expert_cache(execution.tq_expert_cache,shares[index]);
        else resize_moe_expert_cache(execution.moe_expert_cache,shares[index]);
    };
    // Return physical capacity before another adapter grows into that share.
    for(std::size_t i=0;i<shares.size();++i)if(shares[i]<=previous[i])resize(i);
    for(std::size_t i=0;i<shares.size();++i)if(shares[i]>previous[i])resize(i);
}

void reserve_routed_workspace(CudaExecutionContext& execution,std::size_t bytes) {
    const auto context=default_context(mfq_current_cuda_device());
    // A nested reservation can use an intact interval owned by the active
    // outer loan. Retiring its cache would invalidate that interval.
    if(context->workspace_largest_available(mfq_get_current_cuda_stream().stream())>=bytes)return;
    reserve_gguf_expert_workspace(execution.gguf_expert_cache,bytes);
    reserve_tq_expert_workspace(execution.tq_expert_cache,bytes);
    reserve_moe_workspace(execution.moe_expert_cache,bytes);
}

void constrain_routed_capture_workspace(CudaExecutionContext& execution,std::size_t bytes) {
    constrain_tq_capture_workspace(execution.tq_expert_cache,bytes);
}

void reserve_routed_capture_workspace(CudaExecutionContext& execution,std::size_t bytes) {
    reserve_routed_workspace(execution,bytes);constrain_routed_capture_workspace(execution,bytes);
}

void restore_routed_decode_workspace(CudaExecutionContext& execution,std::size_t growth,
        std::optional<std::size_t> transient) {
    prepare_moe_decode_residency(execution.moe_expert_cache);
    restore_gguf_decode_workspace(execution.gguf_expert_cache,growth,transient);
    restore_tq_decode_workspace(execution.tq_expert_cache,growth,transient);
}

void restore_routed_decode_workspace_after_capture(CudaExecutionContext& execution) {
    restore_gguf_decode_workspace_after_capture(execution.gguf_expert_cache);
    restore_tq_decode_workspace_after_capture(execution.tq_expert_cache);
}

std::size_t reclaim_routed_optional_host(CudaExecutionContext& execution,std::size_t allocation_bytes) {
    return reserve_gguf_expert_host(execution.gguf_expert_cache,allocation_bytes)+
        reserve_tq_expert_host(execution.tq_expert_cache,allocation_bytes)+
        reclaim_moe_optional_host(execution.moe_expert_cache,allocation_bytes);
}

std::size_t reclaim_routed_expert_duplicates(CudaExecutionContext& execution,std::size_t bytes) {
    auto released=reclaim_gguf_expert_duplicates(execution.gguf_expert_cache,bytes);
    if(released<bytes)released+=reclaim_tq_expert_duplicates(execution.tq_expert_cache,bytes-released);
    if(released<bytes)released+=reclaim_moe_expert_duplicates(execution.moe_expert_cache,bytes-released);
    return released;
}
std::size_t share_routed_expert_host(CudaExecutionContext& execution,std::size_t bytes) {
    auto released=share_gguf_expert_host(execution.gguf_expert_cache,bytes);
    released+=share_tq_expert_host(execution.tq_expert_cache,bytes);
    released+=share_moe_expert_host(execution.moe_expert_cache,bytes);
    return released;
}

RoutedCacheLearningScope::RoutedCacheLearningScope(CudaExecutionContext& execution,bool enabled)
        :legacy_(execution.moe_expert_cache),gguf_(execution.gguf_expert_cache),tq_(execution.tq_expert_cache),
         legacy_previous_(set_moe_reuse_learning(legacy_,enabled)),
         gguf_previous_(set_gguf_reuse_learning(gguf_,enabled)),tq_previous_(set_tq_reuse_learning(tq_,enabled)) {}
RoutedCacheLearningScope::~RoutedCacheLearningScope() {
        set_moe_reuse_learning(legacy_,legacy_previous_);
        set_gguf_reuse_learning(gguf_,gguf_previous_);set_tq_reuse_learning(tq_,tq_previous_);
    }
RoutedSpeculativeRoutes::RoutedSpeculativeRoutes(RoutedSpeculativeRoutes&&) noexcept=default;
RoutedSpeculativeRoutes::~RoutedSpeculativeRoutes(){discard();}
void RoutedSpeculativeRoutes::discard() noexcept {
        discard_moe_speculative_routes(legacy_);discard_gguf_speculative_routes(gguf_);
        discard_tq_speculative_routes(tq_);legacy_.reset();gguf_.reset();tq_.reset();
    }
void RoutedSpeculativeRoutes::begin(CudaExecutionContext& execution,int rows) {
        if(rows<1)throw std::invalid_argument("speculative route window must be positive");
        discard();legacy_=execution.moe_expert_cache;gguf_=execution.gguf_expert_cache;tq_=execution.tq_expert_cache;
        try {
            begin_moe_speculative_routes(legacy_,rows);begin_gguf_speculative_routes(gguf_,rows);
            begin_tq_speculative_routes(tq_,rows);
        }catch(...) {discard();throw;}
    }
void RoutedSpeculativeRoutes::resolve(int rows,bool cancelled) {
        if(cancelled){discard();return;}
        try {
            commit_moe_speculative_routes(legacy_,rows);commit_gguf_speculative_routes(gguf_,rows);
            commit_tq_speculative_routes(tq_,rows);
        }catch(...) {discard();throw;}
        legacy_.reset();gguf_.reset();tq_.reset();
    }

struct RoutedPrefillWorkspaceLoan::Implementation {
    int device=mfq_current_cuda_device();
    GgufPrefillWorkspaceLoan gguf;
    TqPrefillWorkspaceLoan tq;
    MoePrefillWorkspaceLoan legacy;
    std::shared_ptr<MoeExpertCache> legacy_cache;
    explicit Implementation(CudaExecutionContext& execution)
        :gguf(execution.gguf_expert_cache),tq(execution.tq_expert_cache),legacy(execution.moe_expert_cache),
         legacy_cache(execution.moe_expert_cache) {}
};
RoutedPrefillWorkspaceLoan::RoutedPrefillWorkspaceLoan(CudaExecutionContext& execution)
    :implementation_(std::make_unique<Implementation>(execution)) {}
RoutedPrefillWorkspaceLoan::~RoutedPrefillWorkspaceLoan() {
    if(!implementation_)return;
    try {
        const MfqCudaGuard guard(implementation_->device);
        try {finish();}catch(const std::exception& error) {
            std::fprintf(stderr,"routed prefill restore failed: %s\n",error.what());
        }
        implementation_.reset();
    }catch(...) {}
}

std::array<double,4> routed_expert_phase_timings(const CudaExecutionContext& execution) {
    const auto tq=tq_expert_transfer_timings(execution.tq_expert_cache);
    std::array<double,4> result{0,gguf_expert_fetch_ms(execution.gguf_expert_cache)+tq[0],tq[1],0};
    if(execution.moe_expert_cache)for(const auto& [name,value]:moe_expert_memory_metrics(execution.moe_expert_cache)) {
        if(name=="expert_route_wait_ms")result[0]+=value;
        else if(name=="expert_host_fetch_ms")result[1]+=value;
        else if(name=="expert_host_copy_ms")result[2]+=value;
        else if(name=="ssd_expert_read_wait_ms")result[3]+=value;
    }
    return result;
}
std::size_t RoutedPrefillWorkspaceLoan::begin(std::size_t bytes) {
    finish();
    auto& state=*implementation_;
    const MfqCudaGuard guard(state.device);
    try {
        const auto context=default_context(state.device);
        // A nested caller can already share another format's Context loan.
        // TQ records its independent reservation without opening a second one.
        auto bytes_lent=context->workspace_loan_active()?0:state.gguf.begin(bytes);
        bytes_lent+=state.tq.begin(bytes);
        if(context->workspace_loan_active()) {
            if(context->available_memory()<bytes && context->workspace_largest_available(mfq_current_cuda_stream())<bytes)
                reserve_moe_workspace(state.legacy_cache,bytes);
        } else bytes_lent+=state.legacy.begin(bytes);
        return bytes_lent;
    }catch(...) {
        const auto failure=std::current_exception();try{finish();}catch(...){}std::rethrow_exception(failure);
    }
}
void RoutedPrefillWorkspaceLoan::suspend() {
    auto& state=*implementation_;
    const MfqCudaGuard guard(state.device);
    state.gguf.suspend();state.tq.suspend();state.legacy.suspend();
}
void RoutedPrefillWorkspaceLoan::finish() {
    auto& state=*implementation_;std::exception_ptr failure;
    const MfqCudaGuard guard(state.device);
    const auto finish=[&](auto& owner) {try{owner.finish();}catch(...){if(!failure)failure=std::current_exception();}};
    finish(state.gguf);finish(state.tq);finish(state.legacy);
    if(failure)std::rethrow_exception(failure);
}
} // namespace mfq::cuda

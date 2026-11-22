#include "cuda_batching.h"

#include "mfq_cuda_sampling_ops.h"
#include "runtime/decode_window.h"
#include "runtime/execution_options.h"
#include "runtime/graph_pool_lease.h"
#include "runtime/routed_cache_resources.h"
#include "storage/text_session_cache.h"

#include <algorithm>
#include <iostream>
#include <numeric>

namespace mfq::cuda {
namespace tb=mfq_tensor_backend;
namespace spec=mfq::engine::mtp;

template<class Model,class Adapter>
struct StatefulCudaBatchOperations<Model,Adapter>::SpeculativeGraph {
    Tensor metadata,metadata_host,raw,logits;
    typename Adapter::Verification journals;
    hy4::EngramRows engram_rows;
    std::shared_ptr<GraphPoolLease> pool;
    std::unique_ptr<DecodeWindow> window;
};

BatchLogitCandidates rank_cuda_batch_logits(tb::Tensor logits,int64_t top_k) {
    logits=logits.reshape({-1,logits.size(-1)});
    auto [values,indices]=tb::topk(logits,std::min(top_k,logits.size(1)),-1,true,true);
    return {values.to(tb::kCPU,tb::kFloat32).contiguous(),indices.to(tb::kCPU,tb::kInt64).contiguous()};
}

int32_t BatchLogitCandidates::best(int64_t row) const {
    return int32_t(indices.data_ptr<int64_t>()[row*indices.size(1)]);
}

spec::CompactDistribution BatchLogitCandidates::distribution(int64_t row,const MfqSamplingParams& sampling) const {
    const auto stride=indices.size(1);
    return spec::compact_distribution_from_topk(values.data_ptr<float>()+row*stride,
        indices.data_ptr<int64_t>()+row*stride,int(std::min<int64_t>(stride,sampling.top_k)),
        sampling.temperature,sampling.top_p);
}

template<class Model,class Adapter>
StatefulCudaBatchOperations<Model,Adapter>::StatefulCudaBatchOperations(Model& model,CudaExecutionContext& execution,
                                               CudaContinuousBatchConfig config,const MtpModule* predictor)
    : model_(model),execution_(execution),config_(config),predictor_(predictor) {}
template<class Model,class Adapter>
StatefulCudaBatchOperations<Model,Adapter>::~StatefulCudaBatchOperations()=default;

template<class Model,class Adapter>
int64_t StatefulCudaBatchOperations<Model,Adapter>::vocab_size() const {return model_.vocab_size();}
template<class Model,class Adapter>
int64_t StatefulCudaBatchOperations<Model,Adapter>::max_context() const {return model_.max_position_embeddings();}

template<class Model,class Adapter>
mfq::engine::Metrics StatefulCudaBatchOperations<Model,Adapter>::metrics() const {
    return {
        {"continuous_batching_decode_batches",double(decode_steps_)},
        {"continuous_batching_decode_tokens",double(decode_tokens_)},
        {"continuous_batching_cuda_graph_captures",double(graph_captures_)},
        {"continuous_batching_cuda_graph_replays",double(graph_replays_)},
        {"continuous_batching_mtp_batches",double(speculative_batches_)},
        {"continuous_batching_mtp_max_batch",double(speculative_maximum_batch_)},
    };
}

template<class Model,class Adapter>
std::size_t StatefulCudaBatchOperations<Model,Adapter>::prepare(const std::shared_ptr<Request>& request) {
    if (predictor_ && request->sampling.enable_mtp) {
        request->predictor=predictor_->fork_session();
        if (!request->predictor) throw std::logic_error("CUDA batch predictor cannot create a request session");
        request->predictor->prepare_request(std::min<int64_t>(max_context(),request->prompt.size()+request->generation_limit),
            request->sampling.mtp_max_draft_tokens);
        request->mtp_stats.available=request->mtp_stats.used=true;
    }
    if (!prefix_cache_ || !prefix_cache_->persistent_prefix_enabled() || !request->prefix_limit) {
        request->prefix_limit=0;
        return 0;
    }
    request->prefix_limit=std::min(request->prefix_limit,request->prompt.size()-1);
    if (!request->prefix_limit) return 0;
    const MfqCudaGuard guard(execution_.layer_placement.primary_device());
    const auto active=decode_bound_?model_.cache_pos:0;
    suspend_decode();
    std::size_t restored=0;
    try {
        auto result=prefix_cache_->restore_best(model_,request->predictor.get(),{},request->prompt,request->prefix_limit);
        restored=result.tokens;
        request->predictor_hidden=std::move(result.mtp_last_target_hidden);
        if (restored) request->cache=Adapter::take(model_);
    } catch (...) {
        resume_decode(active);
        throw;
    }
    request->prefix_saved=restored==request->prefix_limit;
    resume_decode(active);
    return restored;
}

template<class Model,class Adapter>
void StatefulCudaBatchOperations<Model,Adapter>::commit_prefix(const std::shared_ptr<Request>& request) {
    if (!prefix_cache_ || request->prefix_saved || request->cache.layers.empty() || !request->prefix_limit ||
        request->prefill_offset!=int64_t(request->prefix_limit)) return;
    const MfqCudaGuard guard(execution_.layer_placement.primary_device());
    const auto active=decode_bound_?model_.cache_pos:0;
    suspend_decode();
    Adapter::bind(model_,std::move(request->cache),request->prefill_offset,false);
    try {
        const std::vector<int64_t> prefix(request->prompt.begin(),request->prompt.begin()+request->prefix_limit);
        auto state=model_.capture_text_session_state(prefix);
        if (request->predictor) {
            state.mtp=request->predictor->capture_session_state(request->prefill_offset,request->predictor_hidden);
            state.bytes+=state.mtp->bytes;
        }
        prefix_cache_->store({},std::move(state));
        request->prefix_saved=true;
    } catch (const std::exception& error) {
        std::cerr<<"runtime_cuda_batch_prefix_store action=skip error="<<error.what()<<'\n';
    }
    request->cache=Adapter::take(model_);
    resume_decode(active);
}

template<class Model,class Adapter>
typename StatefulCudaBatchOperations<Model,Adapter>::Sample StatefulCudaBatchOperations<Model,Adapter>::prefill(
        const std::shared_ptr<Request>& request,mfq::engine::PrefillChunk chunk) {
    const auto device=execution_.layer_placement.primary_device();
    const MfqCudaGuard guard(device);
    Adapter::bind(model_,std::move(request->cache),chunk.offset,false);
    if (!request->ids.defined()) {
        request->ids=tb::tensor(request->prompt,tb::TensorOptions().device(tb::Device(tb::kCUDA,device))
            .dtype(tb::kInt64)).reshape({1,-1}).contiguous();
        request->sampler.emplace(request->sampling);
        if (request->sampler->has_penalties())
            request->counts=SamplingOps::token_counts(request->ids,vocab_size());
    }
    internal::PrefillCudaTimer timer;
    Tensor hidden;
    const auto end=chunk.offset+chunk.count;
    const auto row_bytes=std::size_t(model_.hidden_size())*model_.hc_mult()*sizeof(float)*11;
    while (chunk.offset<end) {
        const bool decode_sized=chunk.count==1 && chunk.offset>0;
        if (!decode_sized && has_routed_expert_cache(execution_)) {
            restore_workspace_=true;
            chunk=plan_stateful_batch_prefill(execution_,chunk,row_bytes,[&] {
                graph_.reset();
                speculative_graphs_.clear();
                graph_logits_={};
                graph_pool_.reset();
            });
        }
        Tensor raw;
        const auto options=request->ids.options();
        auto length=chunk.count==1 && chunk.offset>0
            ?std::optional{tb::full({1},chunk.offset+1,options)}:std::nullopt;
        hidden=model_.hidden_forward(request->ids.narrow(1,chunk.offset,chunk.count).contiguous(),{},length,
            nullptr,{},request->predictor?&raw:nullptr);
        if (request->predictor) {
            request->predictor->append_target_context(raw,chunk.offset);
            request->predictor_hidden=raw.narrow(1,chunk.count-1,1).clone();
        }
        chunk.offset+=chunk.count;
        chunk.count=end-chunk.offset;
    }
    Tensor logits;
    if (end==request->ids.size(1)) logits=model_.logits_from_hidden(hidden.narrow(1,hidden.size(1)-1,1));
    TYLOINFER_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_current_cuda_stream()));
    request->prefill_ms+=timer.elapsed_ms();
    request->cache=Adapter::take(model_);
    if (!logits.defined()) return {};
    auto next=sample_logits(*request->sampler,std::move(logits),request->counts,request->token_constraint);
    const auto token=next.template item<int64_t>();
    request->ids={};
    return {std::move(next),token,{request->prompt.size(),request->prefill_ms,0,request->prefill_ms}};
}

template<class Model,class Adapter>
void StatefulCudaBatchOperations<Model,Adapter>::activate(const std::shared_ptr<Request>& request,const Sample& result) {
    const MfqCudaGuard guard(execution_.layer_placement.primary_device());
    const auto vacant=std::find(slots_.begin(),slots_.end(),nullptr);
    const auto slot=int64_t(vacant-slots_.begin());
    const auto count=std::max(int64_t(slots_.size()),slot+1);
    if (count>config_.max_sequences || request->slot>=0)
        throw std::logic_error("CUDA batch slot capacity exceeded");
    const auto requested=std::min(max_context(),request->cache_length+request->generation_limit);
    const auto planned=std::min(max_context(),(requested+1023)/1024*1024);
    const auto requested_width=request->predictor?1+std::clamp(request->sampling.mtp_max_draft_tokens,
        1,request->predictor->maximum_draft_depth()):1;
    if (count>int64_t(slots_.size()) || planned>planned_length_ || requested_width>verification_width_) {
        graph_.reset();
        speculative_graphs_.clear();
        graph_logits_={};
        staged_={};
        planned_length_=std::max(planned_length_,planned);
        verification_width_=std::max(verification_width_,int64_t(requested_width));
        if constexpr (std::is_same_v<Model,InklingCausalLm>) {
            std::vector<int64_t> lengths;
            for (const auto* active:slots_) lengths.push_back(active?active->cache_length:0);
            Adapter::reserve(model_,decode_cache_,request->cache,count,planned_length_,verification_width_-1,lengths);
        } else Adapter::reserve(model_,decode_cache_,request->cache,count,planned_length_);
        slots_.resize(count);
    }
    Adapter::assign(model_,decode_cache_,request->cache,slot,request->cache_length);
    request->cache={};
    request->slot=slot;
    slots_[slot]=request.get();
    request->first_token=result.token;
    accept(request,result);
}

template<class Model,class Adapter>
void StatefulCudaBatchOperations<Model,Adapter>::discard_prefill(const std::shared_ptr<Request>& request) {
    request->cache={};
    request->ids={};
}

template<class Model,class Adapter>
void StatefulCudaBatchOperations<Model,Adapter>::suspend_decode() {
    speculation_.interrupt();
    if (!decode_bound_) return;
    decode_cache_=Adapter::take(model_);
    decode_bound_=false;
}

template<class Model,class Adapter>
void StatefulCudaBatchOperations<Model,Adapter>::resume_decode(int64_t position) {
    if (!position || decode_bound_) return;
    Adapter::bind(model_,std::move(decode_cache_),position,true);
    decode_bound_=true;
}

template<class Model,class Adapter>
void StatefulCudaBatchOperations<Model,Adapter>::retire(const std::vector<std::shared_ptr<Request>>& retired,int64_t) {
    for (const auto& request:retired) if (request->slot>=0) {
        if constexpr (std::is_same_v<Model,Hy4CausalLm>)
            hy4::release_slot(model_,decode_cache_,request->slot,decode_bound_);
        slots_.at(request->slot)=nullptr;
        request->slot=-1;
    }
    if (std::any_of(slots_.begin(),slots_.end(),[](auto* request){return request!=nullptr;})) return;
    clear_slots();
}

template<class Model,class Adapter>
std::optional<mfq::engine::Generation> StatefulCudaBatchOperations<Model,Adapter>::generate_decode(
        const std::shared_ptr<Request>& request,mfq::engine::InferenceOutput& output,std::function<void()> schedule,
        std::function<bool()> ready,std::function<void()> check) {
    if (!request->predictor) return std::nullopt;
    return publish_cuda_batch_mtp(request,*request,output,std::move(schedule),std::move(ready),std::move(check));
}

mfq::engine::Generation publish_cuda_batch_mtp(
        std::shared_ptr<mfq::engine::BatchRequest> request,CudaBatchMtpState& state,
        mfq::engine::InferenceOutput& output,std::function<void()> schedule,
        std::function<bool()> ready,std::function<void()> check) {
    output.metrics.mtp=state.mtp_stats;
    request->pending_token=state.first_token;
    auto first=output.append({state.first_token});
    request->produced+=static_cast<int32_t>(first.token_ids.size());
    if (!output.stopped()) schedule();
    co_yield std::move(first);
    while (!output.stopped()) {
        while (!ready() && !output.stopped()) co_yield mfq::StepState::waiting;
        if (output.stopped()) co_return;
        check();
        auto tokens=std::exchange(state.verified_tokens,{});
        if (tokens.empty()) throw std::logic_error("CUDA batch verification returned no token");
        request->pending_token=tokens.back();
        auto delta=output.append(std::move(tokens));
        request->produced+=static_cast<int32_t>(delta.token_ids.size());
        const auto accepted=std::min(state.verified_drafts,delta.token_ids.size());
        state.mtp_stats.accepted_tokens+=accepted;
        for (size_t i=0;i<accepted;++i) ++state.mtp_stats.position_accepted[i];
        output.metrics.mtp=state.mtp_stats;
        if (!output.stopped()) schedule();
        co_yield std::move(delta);
    }
}

template<class Model,class Adapter>
typename StatefulCudaBatchOperations<Model,Adapter>::Decoded
StatefulCudaBatchOperations<Model,Adapter>::verify_speculative(State& state) {
    const auto device=execution_.layer_placement.primary_device();
    const auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kInt64);
    const auto host=options.device(tb::kCPU).pinned_memory(true);
    const auto batch=int64_t(slots_.size());
    const auto cycle_start=CudaBatchSpeculation::Clock::now();
    const auto budget=speculation_.depth(state);
    const auto parked_before=speculation_.parked_rounds();
    int64_t width=1,remaining=planned_length_;
    int64_t position=0;
    for (const auto& request:state.active) {
        remaining=std::min(remaining,planned_length_-request->cache_length);
        position=std::max(position,request->cache_length);
        if (request->eligible && request->predictor) width=std::max<int64_t>(width,1+std::min(
            budget,std::clamp(request->sampling.mtp_max_draft_tokens,1,request->predictor->maximum_draft_depth())));
    }
    width=std::min(width,remaining);
    if (width<1) throw std::logic_error("CUDA batch verification exceeds the model context");
    auto& graph=speculative_graphs_[width];
    if (!graph) {
        graph=std::make_unique<SpeculativeGraph>();
        ResidentAllocationScope persistent;
        graph->metadata=tb::empty({batch*(2*width+1)},options);
        graph->metadata_host=tb::empty({graph->metadata.numel()},host);
    }
    auto* data=graph->metadata_host.template data_ptr<int64_t>();
    std::fill_n(data,batch*width,0);
    auto* positions=data+batch*width;
    auto* lengths=positions+batch*width;
    for (int64_t row=0;row<batch;++row) {
        const auto* request=slots_.at(row);
        const auto start=request?request->cache_length:0;
        std::iota(positions+row*width,positions+(row+1)*width,start);
        lengths[row]=start+width;
    }
    struct Proposals {
        Request* request=nullptr;
        int64_t depth=0;
        Tensor previous,hidden;
        std::vector<int32_t> tokens;
        std::vector<spec::CompactDistribution> probabilities;
    };
    std::vector<Proposals> drafts(batch);
    int64_t mtp_rows=0,top_k=1;
    const auto draft_started=mfq::engine::Clock::now();
    for (const auto& request:state.active) {
        const auto lane=request->slot;
        data[lane*width]=request->pending_token;
        if (!request->eligible || !request->predictor) continue;
        auto& predictor=*request->predictor;
        if (predictor.cache_position()!=request->cache_length)
            throw std::logic_error("CUDA batch predictor position changed");
        ++mtp_rows;
        if (!request->sampler->greedy()) top_k=std::max<int64_t>(top_k,request->sampling.top_k);
        const auto cap=std::clamp(request->sampling.mtp_max_draft_tokens,1,predictor.maximum_draft_depth());
        auto& proposal=drafts[lane];
        proposal.request=request.get();
        proposal.depth=std::min<int64_t>({cap,width-1,request->generation_limit-request->produced-1});
        if (proposal.depth<=0) continue;
        proposal.previous=tb::full({1,1},request->pending_token,options);
        proposal.hidden=predictor.draft_block(MtpTarget(model_,lane),proposal.previous,int(proposal.depth));
    }
    for (int64_t step=0;step+1<width;++step) {
        std::vector<Tensor> hidden;
        std::vector<int64_t> lanes;
        for (int64_t lane=0;lane<batch;++lane) {
            auto& proposal=drafts[lane];
            if (proposal.depth<=step) continue;
            hidden.push_back(proposal.request->predictor->draft_next_hidden(
                proposal.hidden.narrow(1,step,1),proposal.previous));
            lanes.push_back(lane);
        }
        if (lanes.empty()) break;
        const auto candidates=rank_cuda_batch_logits(model_.logits_from_hidden(tb::cat(hidden,0)),top_k);
        for (int64_t row=0;row<int64_t(lanes.size());++row) {
            const auto lane=lanes[row];
            auto& proposal=drafts[lane];
            auto& request=*proposal.request;
            auto token=candidates.best(row);
            if (!request.sampler->greedy()) {
                auto distribution=candidates.distribution(row,request.sampling);
                token=spec::sample_compact(distribution,request.sampler->next_uniform());
                proposal.probabilities.push_back(std::move(distribution));
            }
            proposal.tokens.push_back(token);
            data[lane*width+step+1]=token;
            proposal.previous.fill_(token);
        }
    }
    const auto draft_ms=std::chrono::duration<double,std::milli>(mfq::engine::Clock::now()-draft_started).count();
    for (const auto& proposal:drafts) if (proposal.request && proposal.depth>0)
        proposal.request->mtp_stats.head_ms+=draft_ms;
    const std::span<const int64_t> host_ids{data,size_t(batch*width)};
    const std::span<const int64_t> host_positions{positions,size_t(batch*width)};
    if constexpr (std::is_same_v<Model,Hy4CausalLm>) {
        std::vector<uint8_t> active(batch);
        for (const auto& request:state.active) active[request->slot]=request->eligible;
        hy4::batch_engram_rows(model_,host_ids,host_positions,active,graph->engram_rows,width,true);
    }
    graph->metadata.copy_(graph->metadata_host);
    const auto ids=graph->metadata.narrow(0,0,batch*width).reshape({batch,width});
    const auto at=graph->metadata.narrow(0,batch*width,batch*width).reshape({batch,width});
    const auto ends=graph->metadata.narrow(0,2*batch*width,batch);
    model_.cache_pos=position;
    const auto target_started=mfq::engine::Clock::now();
    const bool graph_ready=bool(graph->window);
    if (!graph_ready) {
        const auto context=default_context(device);
        const auto reserve=runtime_options::workspace_reserve_bytes();
        const bool returned=has_routed_expert_cache(execution_) && context->available_memory()<reserve;
        if (returned) reserve_routed_capture_workspace(execution_,reserve);
        const auto stream=mfq_current_cuda_stream();
        graph->pool=GraphPoolLease::acquire(context,stream);
        graph->window=std::make_unique<DecodeWindow>(stream,runtime_options::layer_timings());
        graph->window->set_profile_geometry(
            std::is_same_v<Model,Hy4CausalLm>?"hy4_batch_verify":"inkling_batch_verify",int(batch*width));
        const auto all=tb::arange(batch,options);
        const auto before=Adapter::save(model_,all);
        bool recorded=false,complete=false;
        struct VerifyScope {
            Model& model;
            VerifyScope(Model& m,const hy4::EngramRows& rows):model(m) {
                Adapter::set_verification(model,true);
                if constexpr (std::is_same_v<Model,Hy4CausalLm>) hy4::set_staged_engram(model,&rows);
            }
            ~VerifyScope() {
                if constexpr (std::is_same_v<Model,Hy4CausalLm>) hy4::set_staged_engram(model,nullptr);
                Adapter::set_verification(model,false);
            }
        } verify(model_,graph->engram_rows);
        graph->window->capture([&] {
            complete=false;
            cudaStreamCaptureStatus status;
            TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stream,&status));
            recorded=status==cudaStreamCaptureStatusActive;
            const auto hidden=model_.hidden_forward(ids,at,ends,nullptr,at,&graph->raw,0,{planned_length_,0});
            graph->logits=model_.logits_from_hidden(hidden);
            complete=true;
        },[&] {
            if (complete) {
                auto journals=Adapter::take_verification(model_);
                if (recorded) graph->journals=std::move(journals);
            }
            Adapter::restore(model_,before,all);
        },[&]{graph->raw={};graph->logits={};});
        if (returned) restore_routed_decode_workspace_after_capture(execution_);
        ++graph_captures_;
    }
    RoutedCacheLearningScope learning(execution_,false);
    graph->window->run();
    ++graph_replays_;
    const auto candidates=rank_cuda_batch_logits(graph->logits,top_k);
    std::vector<int64_t> kept(batch);
    std::vector<int> observed;
    int64_t decoded=0;
    const auto target_ms=std::chrono::duration<double,std::milli>(mfq::engine::Clock::now()-target_started).count();
    for (const auto& request:state.active) if (request->eligible) {
        const auto lane=request->slot;
        const auto& proposed=drafts[lane];
        const auto rows=int64_t(proposed.tokens.size()+1);
        decoded+=rows;
        kept[lane]=1;
        if (!request->predictor) continue;
        const auto first=lane*width;
        spec::ChainVerification result;
        if (request->sampler->greedy()) {
            std::vector<int32_t> expected(rows);
            for (int64_t i=0;i<rows;++i) expected[i]=candidates.best(first+i);
            result=proposed.tokens.empty()?spec::ChainVerification{0,expected.front(),true}:
                spec::verify_greedy(proposed.tokens,expected);
        } else {
            std::vector<spec::CompactDistribution> target_rows;
            target_rows.reserve(rows);
            for (int64_t i=0;i<rows;++i) target_rows.push_back(candidates.distribution(first+i,request->sampling));
            std::vector<double> uniforms(proposed.tokens.size());
            std::generate(uniforms.begin(),uniforms.end(),[&]{return request->sampler->next_uniform();});
            result=proposed.tokens.empty()?spec::ChainVerification{0,spec::sample_compact(target_rows.front(),request->sampler->next_uniform()),true}:
                spec::verify_compact_chain(proposed.tokens,proposed.probabilities,target_rows,uniforms,request->sampler->next_uniform());
        }
        const auto accepted=result.accepted_drafts;
        kept[lane]+=int64_t(accepted);
        request->verified_drafts=accepted;
        request->verified_tokens.assign(proposed.tokens.begin(),proposed.tokens.begin()+accepted);
        request->verified_tokens.push_back(result.next_token);
        auto raw=graph->raw.narrow(0,lane,1).narrow(1,0,kept[lane]).contiguous();
        request->predictor->append_target_context(raw,request->cache_length);
        request->cache_length+=int64_t(accepted);
        auto& stats=request->mtp_stats;
        stats.target_ms+=target_ms;
        stats.selected_depth=int(proposed.tokens.size());
        ++stats.cycles;
        ++stats.depth_cycles[proposed.tokens.size()];
        stats.drafted_tokens+=proposed.tokens.size();
        stats.standard_tokens+=proposed.tokens.empty();
        for (size_t i=0;i<proposed.tokens.size();++i) ++stats.position_drafted[i];
        observed.push_back(int(accepted));
    }
    auto lengths_kept=tb::tensor(kept,options);
    if constexpr (std::is_same_v<Model,Hy4CausalLm>) {
        hy4::retain_batch_prefix(model_,graph->journals,lengths_kept,host_positions,kept,width);
        hy4::commit_batch_engram(model_,host_ids,host_positions,kept,width);
    } else inkling::retain_batch_prefix(model_,graph->journals,lengths_kept);
    speculation_.observe(int(width-1),observed,cycle_start,graph_ready,!state.prefilling.empty());
    const auto parked_after=speculation_.parked_rounds();
    const auto cycle_ms=std::chrono::duration<double,std::milli>(CudaBatchSpeculation::Clock::now()-cycle_start).count();
    for (const auto& request:state.active) if (request->eligible && request->predictor) {
        request->mtp_stats.measured_depth_ms[request->mtp_stats.selected_depth]+=cycle_ms;
        request->mtp_stats.park_count+=!parked_before && parked_after;
        request->mtp_stats.reentry_probes+=parked_before && !parked_after;
    }
    ++speculative_batches_;
    speculative_maximum_batch_=std::max(speculative_maximum_batch_,mtp_rows);
    ++decode_steps_;
    decode_tokens_+=decoded;
    return {graph->logits.select(1,0),{}};
}

template<class Model,class Adapter>
typename StatefulCudaBatchOperations<Model,Adapter>::Decoded StatefulCudaBatchOperations<Model,Adapter>::decode(State& state) {
    const auto device=execution_.layer_placement.primary_device();
    const MfqCudaGuard guard(device);
    if (restore_workspace_ && model_.restore_decode_workspace) model_.restore_decode_workspace();
    restore_workspace_=false;
    if (std::any_of(state.active.begin(),state.active.end(),[](const auto& request){return request->eligible && request->predictor;}))
        return verify_speculative(state);
    speculation_.interrupt();
    const auto batch=int64_t(slots_.size());
    auto options=tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kInt64);
    if (!metadata_.defined()) {
        const auto elements=3*int64_t(config_.max_sequences);
        metadata_=tb::empty({elements},options);
        metadata_host_=tb::empty({elements},options.device(tb::kCPU).pinned_memory(true));
    }
    auto* data=metadata_host_.data_ptr<int64_t>();
    std::fill_n(data,batch*2,0);
    std::fill_n(data+batch*2,batch,1);
    int64_t eligible=0,max_position=0,remaining=max_context();
    std::vector<int64_t> paused;
    std::vector<uint8_t> active_rows;
    if constexpr (std::is_same_v<Model,Hy4CausalLm>) active_rows.resize(batch);
    for (const auto& request:state.active) {
        const auto slot=request->slot;
        if (slot<0 || slot>=batch || slots_[slot]!=request.get())
            throw std::logic_error("request lost its CUDA batch slot");
        data[slot]=request->pending_token;
        data[batch+slot]=request->cache_length;
        data[2*batch+slot]=request->cache_length+1;
        max_position=std::max(max_position,request->cache_length);
        remaining=std::min(remaining,int64_t(request->generation_limit-request->produced));
        if (request->eligible) ++eligible;
        else paused.push_back(slot);
        if constexpr (std::is_same_v<Model,Hy4CausalLm>) active_rows[slot]=request->eligible;
    }
    model_.cache_pos=max_position;
    if constexpr (std::is_same_v<Model,Hy4CausalLm>)
        hy4::batch_engram_rows(model_,{data,size_t(batch)},{data+batch,size_t(batch)},active_rows,staged_);
    auto metadata=metadata_.narrow(0,0,3*batch);
    metadata.copy_(metadata_host_.narrow(0,0,3*batch));
    auto ids=metadata.narrow(0,0,batch).reshape({batch,1});
    auto positions=metadata.narrow(0,batch,batch).reshape({batch,1});
    auto lengths=metadata.narrow(0,2*batch,batch);
    Tensor pause_ids;
    Cache pause_state;
    if (!paused.empty()) {
        pause_ids=tb::tensor(paused,options);
        pause_state=Adapter::save(model_,pause_ids);
    }
    const auto forward=[&] {
        if constexpr (std::is_same_v<Model,Hy4CausalLm>) {
            struct Scope {
                Hy4CausalLm& model;
                explicit Scope(Hy4CausalLm& value,const hy4::EngramRows& stage):model(value) {
                    hy4::set_staged_engram(model,&stage);
                }
                ~Scope(){hy4::set_staged_engram(model,nullptr);}
            } use(model_,staged_);
            return model_.last_logits_static(ids,positions,lengths,{planned_length_,0});
        }
        return model_.last_logits_static(ids,positions,lengths,{planned_length_,0});
    };
    Tensor logits;
    if (config_.cuda_graph && mfq_cuda_graph_capture_supported() &&
        (graph_ || remaining>=config_.cuda_graph_minimum_tokens)) {
        if (!graph_) {
            const auto context=default_context(device);
            const auto reserve=runtime_options::workspace_reserve_bytes();
            const bool reserve_capture=has_routed_expert_cache(execution_) && context->available_memory()<reserve;
            if (reserve_capture) reserve_routed_capture_workspace(execution_,reserve);
            auto all=tb::arange(batch,options);
            auto before=Adapter::save(model_,all);
            const auto stream=mfq_current_cuda_stream();
            graph_pool_=GraphPoolLease::acquire(context,stream);
            graph_=std::make_unique<DecodeWindow>(stream,runtime_options::layer_timings());
            graph_->capture([&]{graph_logits_=forward();},[&]{Adapter::restore(model_,before,all);},
                [&]{graph_logits_={};});
            if (reserve_capture) restore_routed_decode_workspace_after_capture(execution_);
            ++graph_captures_;
        }
        graph_->run();
        ++graph_replays_;
        logits=graph_logits_;
    } else logits=forward();
    if (pause_ids.defined()) Adapter::restore(model_,pause_state,pause_ids);
    if constexpr (std::is_same_v<Model,Hy4CausalLm>) hy4::commit_batch_selections(model_,active_rows);
    ++decode_steps_;
    decode_tokens_+=eligible;
    return {std::move(logits),{}};
}

template<class Model,class Adapter>
typename StatefulCudaBatchOperations<Model,Adapter>::Sample StatefulCudaBatchOperations<Model,Adapter>::sample(
        const std::shared_ptr<Request>& request,const Decoded& state) {
    if (request->predictor) return {};
    auto next=sample_logits(*request->sampler,state.logits.narrow(0,request->slot,1),
        request->counts,request->token_constraint);
    const auto token=next.template item<int64_t>();
    return {std::move(next),token,{}};
}

template<class Model,class Adapter>
void StatefulCudaBatchOperations<Model,Adapter>::accept(const std::shared_ptr<Request>& request,const Sample& result) {
    if (request->counts.defined()) sample_token_counts_add_cuda(request->counts,result.next.contiguous());
}

template<class Model,class Adapter>
void StatefulCudaBatchOperations<Model,Adapter>::recover(State&) {clear_slots();}

template<class Model,class Adapter>
void StatefulCudaBatchOperations<Model,Adapter>::clear_slots() {
    graph_.reset();
    speculative_graphs_.clear();
    graph_logits_={};
    graph_pool_.reset();
    staged_={};
    slots_.clear();
    planned_length_=0;
    verification_width_=1;
    suspend_decode();
    decode_cache_={};
    model_.reset(1);
}

template class StatefulCudaBatchOperations<InklingCausalLm,InklingBatchAdapter>;
template class StatefulCudaBatchOperations<Hy4CausalLm,Hy4BatchAdapter>;

}  // namespace mfq::cuda

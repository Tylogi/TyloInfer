#include "ops.h"
#include "batch_state.h"
#include "engram.h"

#include "core/decode_graph.h"
#include "core/ffn.h"
#include "core/mtp.h"
#include "kernels/qsa_tq.h"
#include "qsa_indexer_cpu.h"
#include "mfq_cuda_attention_ops.h"
#include "mfq_cuda_kv_quantization.h"
#include "mfq_cuda_norm_ops.h"
#include "runtime/decode_window.h"
#include "runtime/execution_options.h"
#include "runtime/graph_pool_lease.h"
#include "runtime/kv_budget.h"
#include "runtime/kv_memory.h"
#include "storage/transformer_loader.h"
#include "storage/weight_loader.h"
#include "studio/kernels/cuda/index_pool.h"
#include "studio/kernels/cuda/qwen4_exp.h"
#include "tyloinfer/host_memory.h"

#include <cmath>
#include <bit>
#include <cstdio>
#include <iostream>
#include <map>
#include <optional>
#include <unordered_map>

namespace mfq::cuda {
namespace tb=mfq_tensor_backend;
using Tensor=tb::Tensor;

namespace hy4 {
struct DecodeGraphs {
    struct Cell {
        Tensor ids,positions,lengths,hidden,logits;
        EngramRows engram_rows;
        std::vector<std::optional<Verification>> journals;
        std::vector<Tensor> selections;
        std::shared_ptr<GraphPoolLease> pool;
        std::unique_ptr<DecodeWindow> window;
    };
    int64_t length=0;
    cudaStream_t stream=nullptr;
    std::vector<const void*> state;
    std::map<std::pair<int64_t,int64_t>,std::unique_ptr<Cell>> cells;
};

struct Shared {
    mfq::models::hy4_flash::Config config;
    RopeCache rope,sliding_rope;
    std::optional<EngramState> engram;
    std::shared_ptr<KvOffloadStore> offload;
    std::unique_ptr<DecodeGraphs> graphs;
    const EngramRows* staged_engram=nullptr;
    bool batch_verification=false,host_indexer=false;
    int cold_bits=0;
    int64_t storage_width() const {
        if (!cold_bits) return config.kv_rank+config.metadata.rotary_dim;
        return cuda_kv_packed_width(int(config.kv_rank),cold_bits)+
            cuda_kv_packed_width(int(config.metadata.rotary_dim),cold_bits);
    }
    tb::ScalarType storage_dtype() const {return cold_bits?tb::kInt32:tb::kFloat16;}

    Tensor encode_kv(const Tensor& rows) const {
        if (!cold_bits) return rows;
        return tb::cat({cuda_kv_encode(rows.narrow(-1,0,config.kv_rank),cold_bits,false),
            cuda_kv_encode(rows.narrow(-1,config.kv_rank,config.metadata.rotary_dim),cold_bits,false)},-1);
    }

    Tensor decode_kv(const Tensor& rows,const Tensor& lengths={}) const {
        if (!cold_bits) return rows;
        return cuda_kv_decode_mla(rows,int(config.kv_rank),int(config.metadata.rotary_dim),cold_bits,lengths);
    }
    std::vector<mfq::models::deepseek_v41::EngramHashSnapshot> batch_hashes;
    std::vector<uint8_t> active_rows;
};
}

namespace {

struct RouteWeights {Tensor function,scale,base;};

struct Hy4Block final : Block {
    std::shared_ptr<hy4::Shared> state;
    int64_t layer=0,heads=0;
    bool sliding=false,indexed=false;
    RouteWeights attention_hc,ffn_hc;
    Tensor attention_norm,attention_output_norm,ffn_norm,ffn_output_norm,q_norm,kv_norm;
    QuantLinear q_a,q_b,kv_a,gate,output;
    Tensor key_b,value_b,sinks;
    FFN ffn;
    hy4::EngramLayer engram;
    QuantLinear index_q,index_k,index_score;
    QuantLinearGroup index_pool_value_gate;
    Tensor index_key_norm,index_bias,index_pool_norm;
    Tensor kv,records,pending_values,pending_logits;
    std::vector<std::unique_ptr<KvOffloadSequence>> sequences;
    struct ColdStage {
        Tensor rows,indices,lengths,metadata,host_metadata,host_positions;
        std::shared_ptr<KvOffloadStore::GatherWorkspace> gather;
        std::vector<std::vector<int64_t>> requested;
        std::vector<KvOffloadSequence::GatherSlice> slices;
    };
    std::shared_ptr<ColdStage> eager_cold;
    bool expose_selection=false,reuse_draft_kv=false;
    Tensor last_selection;
    std::vector<Tensor> batch_selections;
    std::vector<hy4::DraftKvHistory> batch_draft_kv;
    std::optional<hy4::Verification> verification;

    void reset(int64_t batch) override {
        verification.reset();
        last_selection={};
        batch_selections.clear();
        batch_draft_kv.clear();
        sequences.clear();
        if (kv.defined() && kv.size(0)!=batch) kv={};
        if (records.defined() && records.size(0)!=batch) {
            records={};pending_values={};pending_logits={};
        }
    }

    bool supports_speculation() const noexcept override {return true;}
    void commit_speculative() override {verification.reset();}
    void rollback_speculative(int64_t keep) override {
        if (!verification) return;
        auto saved=std::move(*verification);
        verification.reset();
        const auto accepted=keep-saved.start;
        if (accepted<=0 || accepted>saved.positions.numel())
            throw std::logic_error("invalid Hy4 verification boundary");
        for (auto& sequence:sequences) sequence->truncate(keep);
        if (expose_selection)
            last_selection=saved.selections.narrow(1,accepted-1,1).contiguous();
        if (!indexed || keep%state->config.group_size==0) return;
        pending_values.copy_(saved.pending_values);
        pending_logits.copy_(saved.pending_logits);
        // Completed groups below the accepted boundary already have the
        // correct records. Recover just the unfinished group's inputs.
        (void)index_pool_append(saved.values.narrow(1,0,accepted).contiguous(),
            saved.logits.narrow(1,0,accepted).contiguous(),saved.positions.narrow(-1,0,accepted).contiguous(),
            pending_values,pending_logits,index_bias,index_pool_norm,state->rope.cos,state->rope.sin,
            int(records.size(1)),float(state->config.metadata.rms_norm_eps));
    }

    Tensor cold_attention(const Tensor& query,const Tensor& rows,const Tensor& indices,
                          int64_t start,double scale) {
        const auto batch=rows.size(0),tokens=rows.size(1),width=state->storage_width();
        const auto packed=state->encode_kv(rows);
        const auto stride=indices.size(2);
        if (sequences.empty()) for (int64_t i=0;i<batch;++i)
            sequences.push_back(std::make_unique<KvOffloadSequence>(state->offload,width,state->config.group_size,state->storage_dtype()));
        if (!start) {
            auto result=attention_mla_latent_selected_cuda(query,rows,indices,sinks,scale);
            for (int64_t i=0;i<batch;++i) sequences[i]->append(packed.select(0,i));
            return result;
        }
        std::vector<Tensor> outputs;
        outputs.reserve(batch);
        for (int64_t i=0;i<batch;++i) {
            auto& sequence=*sequences.at(i);
            if (sequence.position()>start) sequence.truncate(start);
            if (sequence.position()!=start) throw std::logic_error("Hy4 cold KV has a discontinuity");
            sequence.append(packed.select(0,i));
            auto selected=indices.select(0,i).to(tb::kCPU).contiguous();
            auto* positions=selected.data_ptr<int32_t>();
            std::vector<int64_t> unique;
            std::unordered_map<int32_t,int32_t> slots;
            for (int64_t j=0;j<tokens*stride;++j) if (positions[j]>=0) {
                auto [slot,added]=slots.try_emplace(positions[j],int32_t(unique.size()));
                if (added) unique.push_back(positions[j]);
                positions[j]=slot->second;
            }
            Tensor compact;
            if (unique.empty()) compact=tb::zeros({1,width},packed.options());
            else {
                const auto capacity=int64_t(std::bit_ceil(unique.size()));
                auto workspace=sequence.gather_workspace(capacity);
                sequence.gather_into(unique,*workspace);
                compact=workspace->rows.narrow(0,0,int64_t(unique.size()));
            }
            outputs.push_back(attention_mla_latent_selected_cuda(query.narrow(0,i,1),state->decode_kv(compact.unsqueeze(0)),
                selected.unsqueeze(0).to(rows.device()),sinks,scale));
        }
        return outputs.size()==1?std::move(outputs.front()):tb::cat(outputs,0);
    }

    Tensor staged_cold_attention(const Tensor& query,const Tensor& rows,const Tensor& indices,
        const Tensor& positions,int64_t planned,double scale,DecodeWindow* window) {
        const auto batch=rows.size(0),tokens=rows.size(1),width=rows.size(2);
        const auto slots=tokens*indices.size(2),capacity=std::min(slots,planned);
        const auto make_stage=[&] {
            ResidentAllocationScope resident;
            auto stage=std::make_shared<ColdStage>();
            stage->rows=tb::empty(rows.sizes(),rows.options());
            stage->metadata=tb::zeros({indices.numel()+batch},indices.options());
            stage->indices=stage->metadata.narrow(0,0,indices.numel()).reshape(indices.sizes());
            stage->lengths=stage->metadata.narrow(0,indices.numel(),batch);
            stage->host_metadata=tb::empty(stage->metadata.sizes(),indices.options().device(tb::kCPU).pinned_memory(true));
            stage->host_positions=tb::empty(positions.sizes(),positions.options().device(tb::kCPU).pinned_memory(true));
            stage->gather=state->offload->gather_workspace(batch*capacity,width,rows.scalar_type());
            stage->requested.resize(batch);
            stage->slices.reserve(batch);
            return stage;
        };
        std::shared_ptr<ColdStage> stage;
        if (window) stage=window->workspace<ColdStage>(this,make_stage);
        else {
            if (!eager_cold || eager_cold->indices.sizes()!=indices.sizes() ||
                eager_cold->gather->rows.size(0)!=batch*capacity) eager_cold=make_stage();
            stage=eager_cold;
        }
        stage->rows.copy_(rows);
        const auto stream=mfq_current_cuda_stream();
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(stage->host_metadata.data_ptr(),indices.data_ptr(),indices.nbytes(),cudaMemcpyDeviceToHost,stream));
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(stage->host_positions.data_ptr(),positions.data_ptr(),positions.nbytes(),cudaMemcpyDeviceToHost,stream));
        const auto gather=[this,stage,batch,tokens,slots,capacity,stream] {
            auto* selected=stage->host_metadata.data_ptr<int32_t>();
            auto* lengths=selected+batch*slots;
            const auto* offsets=stage->host_positions.data_ptr<int64_t>();
            stage->slices.clear();
            for (int64_t row=0;row<batch;++row) {
                auto* ids=selected+row*slots;
                if (!state->active_rows.empty() && !state->active_rows.at(row)) {
                    std::fill_n(ids,slots,-1);
                    lengths[row]=0;
                    continue;
                }
                auto& sequence=*sequences.at(row);
                const auto position=offsets[stage->host_positions.dim()==1?0:row*tokens];
                if (sequence.position()!=position) throw std::logic_error("Hy4 streamed decode KV has a discontinuity");
                sequence.append(stage->rows.select(0,row));
                auto& requested=stage->requested[row];
                requested.clear();
                if (tokens==1) {
                    for (int64_t i=0;i<slots;++i) if (ids[i]>=0) {
                        requested.push_back(ids[i]);
                        ids[i]=int32_t(requested.size()-1);
                    }
                } else {
                    std::unordered_map<int32_t,int32_t> compact;
                    for (int64_t i=0;i<slots;++i) if (ids[i]>=0) {
                        const auto [slot,added]=compact.try_emplace(ids[i],int32_t(requested.size()));
                        if (added) requested.push_back(ids[i]);
                        ids[i]=slot->second;
                    }
                }
                if (int64_t(requested.size())>capacity) throw std::logic_error("Hy4 selected KV exceeds its compact workspace");
                lengths[row]=int32_t(requested.size());
                stage->slices.push_back({&sequence,requested,row*capacity});
            }
            if (!stage->slices.empty()) stage->slices.front().sequence->gather_into(stage->slices,*stage->gather);
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(stage->metadata.data_ptr(),selected,stage->metadata.nbytes(),cudaMemcpyHostToDevice,stream));
        };
        if (window) window->host_barrier(gather);
        else {
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
            gather();
        }
        auto selected=state->decode_kv(stage->gather->rows.reshape({batch,capacity,width}),stage->lengths);
        if (reuse_draft_kv && state->batch_verification && verification) {
            verification->gathered_kv=selected;
            verification->gathered_indices=stage->indices;
        }
        return attention_mla_latent_selected_cuda(query,selected,stage->indices,sinks,scale);
    }

    std::vector<Tensor*> graph_warmup_state() override {return {&pending_values,&pending_logits};}

    Tensor allocate_growing(const Tensor& current,const Tensor& owner,int64_t needed,int64_t width,
        tb::ScalarType dtype,bool ring,int64_t position,bool host=false) const {
        const int64_t batch=owner.size(0);
        if (current.defined() && current.size(0)==batch && current.size(1)>=needed) return current;
        ResidentAllocationScope resident;
        auto options=owner.options().dtype(dtype);
        if (host) options=options.device(tb::kCPU).pinned_memory(true);
        auto updated=tb::zeros({batch,needed,width},options);
        if (!current.defined() || !position || current.size(0)!=batch) return updated;
        if (ring) {
            const auto first=std::max(int64_t(0),position-state->config.window+1);
            auto offsets=tb::arange(first,position,owner.options().dtype(tb::kInt64));
            updated.index_copy_(1,tb::remainder(offsets,needed),current.index_select(1,tb::remainder(offsets,current.size(1))));
        } else {
            auto count=std::min(position,current.size(1));
            updated.narrow(1,0,count).copy_(current.narrow(1,0,count));
        }
        return updated;
    }

    void reserve_decode(const Tensor& owner,int64_t position,int64_t planned) {
        const auto& config=state->config;
        const auto limit=config.metadata.max_position_embeddings;
        const auto width=config.kv_rank+config.metadata.rotary_dim;
        const auto overrun=int64_t(mfq::engine::mtp::kMaximumBlockwiseDraftDepth);
        const auto needed=std::min(limit,sliding?config.window+overrun:(planned+1023)/1024*1024);
        if (sliding || !state->offload)
            kv=allocate_growing(std::move(kv),owner,needed,width,tb::kFloat16,sliding,position);
        if (indexed) {
            const auto group=config.group_size;
            const auto records_needed=std::min((limit+group-1)/group,((planned+group-1)/group+255)/256*256);
            records=allocate_growing(std::move(records),owner,records_needed,config.index_width+2,
                tb::kInt16,false,position/group,state->host_indexer);
        }
    }

    Tensor rotary(const Tensor& value,const Tensor& positions,const RopeCache& rope) const {
        return dsv4_tail_rope_cuda(value.transpose(1,2),positions.contiguous(),
            rope.cos,rope.sin,state->config.metadata.rotary_dim);
    }

    Tensor select_groups_on_host(const Tensor& query,const Tensor& weights,const Tensor& positions,int64_t count) {
        struct Stage {Tensor query,weights,positions,groups,selected;};
        const auto batch=query.size(0),tokens=query.size(1);
        const auto make_stage=[&] {
            ResidentAllocationScope resident;
            const auto pinned=query.options().device(tb::kCPU).pinned_memory(true);
            auto stage=std::make_shared<Stage>();
            stage->query=tb::empty(query.sizes(),pinned);
            stage->weights=tb::empty(weights.sizes(),pinned);
            stage->positions=tb::empty(positions.sizes(),pinned.dtype(tb::kInt64));
            stage->groups=tb::empty({batch,tokens,count},pinned.dtype(tb::kInt32));
            stage->selected=tb::empty({batch,tokens,count},query.options().dtype(tb::kInt32));
            return stage;
        };
        auto* window=DecodeWindow::recording();
        auto stage=window?window->workspace<Stage>(&records,make_stage):make_stage();
        const auto stream=mfq_current_cuda_stream();
        const auto copy=[&](const Tensor& input,const Tensor& target) {
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(target.data_ptr(),input.data_ptr(),input.nbytes(),cudaMemcpyDeviceToHost,stream));
        };
        copy(query,stage->query);
        copy(weights,stage->weights);
        copy(positions,stage->positions);
        const auto rank=[this,stage,batch,tokens,count,stream] {
            const auto& config=state->config;
            const auto heads=config.index_heads,width=config.index_width,capacity=records.size(1);
            for (int64_t row=0;row<batch;++row) {
                if (!state->active_rows.empty() && !state->active_rows.at(row)) {
                    std::fill_n(stage->groups.data_ptr<int32_t>()+row*tokens*count,tokens*count,-1);
                    continue;
                }
                const auto* offsets=stage->positions.data_ptr<int64_t>()+(stage->positions.dim()==2?row*tokens:0);
                const auto* queries=stage->query.data_ptr<float>()+row*tokens*heads*width;
                const auto* weights=stage->weights.data_ptr<float>()+row*tokens*heads;
                const auto* packed=records.data_ptr<int16_t>()+row*capacity*(width+2);
                auto* selected=stage->groups.data_ptr<int32_t>()+row*tokens*count;
                constexpr int tile=16;
                for (int64_t t=0;t<tokens;t+=tile) mfq::cpu::index_pool_affine16_topk(
                    queries+t*heads*width,weights+t*heads,offsets+t,int(std::min<int64_t>(tile,tokens-t)),
                    int(heads),int(width),packed,int(capacity),int(config.group_size),config.keep_tail,int(count),
                    mfq_get_num_threads(),selected+t*count);
            }
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(stage->selected.data_ptr(),stage->groups.data_ptr(),
                stage->groups.nbytes(),cudaMemcpyHostToDevice,stream));
        };
        if (window) window->host_barrier(rank);
        else {
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
            rank();
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        return stage->selected;
    }

    Tensor select_groups(CudaExecutionContext& execution,const Tensor& hidden,const Tensor& low_query,
        const Tensor& positions,int64_t start,int64_t planned) {
        const auto& config=state->config;
        const auto b=hidden.size(0),t=hidden.size(1),w=config.index_width,g=config.group_size;
        const auto cap=std::min((config.metadata.max_position_embeddings+g-1)/g,((planned+g-1)/g+255)/256*256);
        records=allocate_growing(std::move(records),hidden,cap,w+2,tb::kInt16,false,start/g,state->host_indexer);
        if (!pending_values.defined()) {
            ResidentAllocationScope resident;
            pending_values=tb::zeros({b,g,w},hidden.options().dtype(tb::kFloat32));
            pending_logits=tb::zeros({b,g,w},hidden.options().dtype(tb::kFloat32));
        }
        auto indices=index_k.forward(execution,hidden);
        indices=rms_norm_cuda(indices.contiguous(),index_key_norm,config.metadata.rms_norm_eps);
        auto parts=index_pool_value_gate.forward(execution,indices);
        auto values=parts[0].to(tb::kFloat32).contiguous(),logits=parts[1].to(tb::kFloat32).contiguous();
        if (verification) {
            verification->pending_values=pending_values.clone();
            verification->pending_logits=pending_logits.clone();
            verification->values=values;
            verification->logits=logits;
        }
        auto complete=index_pool_append(values,logits,
            positions,pending_values,pending_logits,index_bias,index_pool_norm,state->rope.cos,state->rope.sin,
            int(records.size(1)),float(config.metadata.rms_norm_eps));
        auto* record_pointer=records.data_ptr();
        if (state->host_indexer)
            TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&record_pointer,records.data_ptr(),0));
        TYLOINFER_CUDA_CHECK(static_cast<cudaError_t>(tyloinfer::qsa::indexer_affine16_pack_at(
            complete.keys.data_ptr<float>(),false,w,static_cast<int16_t*>(record_pointer),complete.record_positions.data_ptr(),
            true,int(complete.record_positions.numel()),int(w),int(records.size(0)*records.size(1)),mfq_current_cuda_stream())));
        if (planned<=config.index_budget) return {};
        auto query=index_q.forward(execution,low_query).to(tb::kFloat32).reshape({b,t,config.index_heads,w});
        query=rotary(query,positions,state->rope).transpose(1,2).contiguous();
        auto learned=index_score.forward(execution,hidden).to(tb::kFloat32).contiguous();
        const auto selected=std::min(cap,config.index_budget/g-int64_t(config.keep_tail));
        if (state->host_indexer) return select_groups_on_host(query,learned,positions,selected);
        const auto workspace_row=std::size_t(cap)*sizeof(float)+
            tyloinfer::qsa::indexer_affine16_score_scratch(1,int(config.index_heads),int(w));
        // Leave the remainder of the workspace for TopK and live projections.
        const auto scratch_limit=runtime_options::workspace_reserve_bytes()/2;
        const auto tile=std::clamp<int64_t>(int64_t(scratch_limit/workspace_row),1,t);
        auto scores=tb::empty({1,tile,cap},hidden.options().dtype(tb::kFloat32));
        const auto bytes=tyloinfer::qsa::indexer_affine16_score_scratch(int(tile),int(config.index_heads),int(w));
        auto workspace=tb::empty({int64_t(bytes)},hidden.options().dtype(tb::kUInt8));
        const auto choose=[&](int64_t batch,int64_t offset,int64_t count) {
            const auto row=positions.dim()==1?positions:positions.select(0,batch);
            const auto at=row.narrow(0,offset,count).contiguous();
            auto block_scores=scores.narrow(1,0,count);
            TYLOINFER_CUDA_CHECK(static_cast<cudaError_t>(tyloinfer::qsa::index_pool_affine16_scores(
                query.select(0,batch).narrow(0,offset,count).data_ptr<float>(),records.select(0,batch).data_ptr<int16_t>(),
                learned.select(0,batch).narrow(0,offset,count).data_ptr<float>(),at.data_ptr<int64_t>(),
                block_scores.data_ptr<float>(),int(count),int(config.index_heads),int(w),int(cap),int(g),
                config.keep_tail,mfq_current_cuda_stream(),workspace.data_ptr(),bytes)));
            return mfq_qwen4_exp::live_topk_blocks(block_scores,at,g,selected).to(tb::kInt32);
        };
        if (b==1 && tile==t) return choose(0,0,t);
        auto groups=tb::empty({b,t,selected},positions.options().dtype(tb::kInt32));
        for (int64_t i=0;i<b;++i)
            for (int64_t offset=0;offset<t;offset+=tile) {
                const auto count=std::min(tile,t-offset);
                groups.narrow(0,i,1).narrow(1,offset,count).copy_(choose(i,offset,count));
            }
        return groups;
    }

    Tensor attend(CudaExecutionContext& execution,Tensor x,const Context& context) {
        const auto& config=state->config;
        const auto& c=config.metadata;
        const auto& rope=sliding?state->sliding_rope:state->rope;
        const auto b=x.size(0),t=x.size(1),start=context.cache_position;
        const auto planned=std::max(start+t,context.planned_kv_length);
        const auto& positions=context.cache_positions.has_value()?context.cache_positions.value():context.positions;
        auto low_query=rms_norm_cuda(q_a.forward(execution,x).contiguous(),q_norm,c.rms_norm_eps);
        auto query=q_b.forward(execution,low_query).to(tb::kFloat32).reshape({b,t,heads,c.head_dim});
        const auto head_nope=c.head_dim-c.rotary_dim;
        auto q_roped=rotary(query.narrow(3,head_nope,c.rotary_dim).contiguous(),context.positions,rope);
        auto q_plain=query.narrow(3,0,head_nope).transpose(1,2).to(key_b.scalar_type());
        auto absorbed=tb::matmul(q_plain,key_b.transpose(1,2)).to(tb::kFloat32);
        auto q=tb::cat({absorbed,q_roped},-1).contiguous();
        auto packed=kv_a.forward(execution,x).to(tb::kFloat32);
        auto k=rms_norm_cuda(packed.narrow(2,0,config.kv_rank).contiguous(),kv_norm,c.rms_norm_eps);
        auto k_rope=rotary(packed.narrow(2,config.kv_rank,c.rotary_dim).contiguous().unsqueeze(2),context.positions,rope).squeeze(1);
        auto rows=tb::cat({k,k_rope},-1).to(tb::kFloat16).contiguous();
        const bool offloaded=!sliding && state->offload;
        if (!offloaded) {
            const auto width=config.kv_rank+c.rotary_dim;
            const auto reserve=mfq::engine::mtp::kMaximumBlockwiseDraftDepth;
            const auto needed=std::min(c.max_position_embeddings,
                sliding?config.window+std::max(t-1,int64_t(reserve)):(planned+1023)/1024*1024);
            kv=allocate_growing(std::move(kv),rows,needed,width,tb::kFloat16,sliding,start);
            const auto physical=sliding?tb::remainder(positions,kv.size(1)):positions;
            if (physical.dim()==1) kv.index_copy_(1,physical,rows);
            else for (int64_t i=0;i<b;++i) {
                auto timeline=kv.select(0,i);
                timeline.index_copy_(0,physical.select(0,i),rows.select(0,i));
            }
        }
        auto groups=indexed?select_groups(execution,x,low_query,positions,start,planned):Tensor{};
        auto selected=mla_group_token_indices_cuda(groups,positions,b,t,config.group_size,
            sliding?std::min(config.window,planned):planned,sliding?kv.size(1):0,config.keep_tail);
        if (expose_selection) {
            last_selection=selected.narrow(1,t-1,1).contiguous();
            if (verification) verification->selections=selected;
        }
        const double scale=1/std::sqrt(double(c.head_dim));
        auto result=[&] {
            if (!offloaded) return attention_mla_latent_selected_cuda(q,kv,selected,sinks,scale);
            auto* window=DecodeWindow::recording();
            if (window || b>1) return staged_cold_attention(q,state->encode_kv(rows),selected,positions,planned,scale,window);
            return cold_attention(q,rows,selected,start,scale);
        }();
        result=tb::matmul(result.transpose(1,2).to(value_b.scalar_type()),value_b.transpose(1,2))
            .transpose(1,2).reshape({b,t,heads*config.value_head_dim});
        result=result*tb::sigmoid(gate.forward(execution,x));
        return rms_norm_cuda(output.forward(execution,result).contiguous(),attention_output_norm,c.rms_norm_eps);
    }

    Tensor forward_context(CudaExecutionContext& execution,Tensor streams,const Context& context,const RopeCache&) override try {
        const auto& config=state->config;
        if (context.confirmed_prefix>0 || state->batch_verification) {
            verification.emplace();
            verification->start=context.cache_position;
            verification->positions=context.cache_positions.value_or(context.positions);
        }
        if (state->engram && layer==0 && !state->staged_engram) {
            auto ids=context.host_token_ids.defined()?context.host_token_ids:context.token_ids.to(tb::kCPU).contiguous();
            if (context.confirmed_prefix>0) state->engram->hash->begin_speculative();
            state->engram->prepare(ids,context.cache_position);
        }
        if (engram.slot>=0) streams=execution.profiler.measure("hy4.engram",[&] {
            if (state->staged_engram) {
                const auto& [rows,valid]=state->staged_engram->at(engram.slot);
                return engram.forward_rows(execution,std::move(streams),rows,valid);
            }
            return engram.forward(execution,*state->engram,std::move(streams));
        });
        const auto mix=[&](const RouteWeights& route) {
            return independent_hc_prepare(streams,route.function,route.scale,route.base,
                float(config.metadata.rms_norm_eps),float(config.metadata.hc_eps),float(config.hc_magnitude));
        };
        const auto& c=config.metadata;
        auto a=mix(attention_hc);
        auto attended=execution.profiler.measure("hy4.attention",[&] {
            return attend(execution,rms_norm_cuda(a.branch,attention_norm,c.rms_norm_eps),context);
        });
        streams=independent_hc_merge(streams,attended.to(streams.scalar_type()).contiguous(),a.post);
        auto f=mix(ffn_hc);
        auto row=rms_norm_cuda(f.branch,ffn_norm,c.rms_norm_eps);
        auto projected=execution.profiler.measure("hy4.ffn",[&] {return ffn.forward(execution,row);}).reshape(row.sizes());
        auto normalized=rms_norm_cuda(projected.contiguous(),ffn_output_norm,c.rms_norm_eps);
        return independent_hc_merge(streams,normalized.to(streams.scalar_type()).contiguous(),f.post);
    } catch (const mfq::cuda::Error& error) {
        throw std::runtime_error("Hy4 CUDA layer "+std::to_string(layer)+": "+error.what());
    }

    Tensor forward(CudaExecutionContext&,Tensor,Tensor,int64_t,const MfqOptional<Tensor>&,const RopeCache&,
        const MfqOptional<Tensor>&,const MfqOptional<Tensor>&) override {
        throw std::logic_error("Hy4 needs position-aware MLA and Engram cache context");
    }
};

RouteWeights route_weights(const TransformerWeightLoader& loader,const std::string& prefix) {
    return {loader.dense(prefix+"function"),loader.dense(prefix+"scale"),loader.dense(prefix+"base")};
}

}  // namespace

namespace hy4 {
BatchCache take_cache(Hy4CausalLm& model) {
    BatchCache result;
    result.layers.reserve(model.blocks.size());
    for (const auto& owned:model.blocks) {
        auto& block=static_cast<Hy4Block&>(*owned);
        result.layers.push_back({std::exchange(block.kv,{}),std::exchange(block.records,{}),
            std::exchange(block.pending_values,{}),std::exchange(block.pending_logits,{}),
            std::exchange(block.sequences,{}),std::exchange(block.last_selection,{}),
            std::exchange(block.batch_selections,{}),std::exchange(block.batch_draft_kv,{})});
    }
    auto& shared=*model.shared;
    if (!shared.batch_hashes.empty()) result.hashes=std::exchange(shared.batch_hashes,{});
    else if (shared.engram && model.cache_pos>0) result.hashes.push_back(shared.engram->hash->snapshot());
    return result;
}

void bind_cache(Hy4CausalLm& model,BatchCache&& cache,int64_t position,bool decoding) {
    auto& shared=*model.shared;
    shared.active_rows.clear();
    if (cache.layers.empty()) {
        if (position) throw std::logic_error("Hy4 batch lost its prefill cache");
        shared.batch_hashes.clear();
        model.reset(1);
        return;
    }
    if (cache.layers.size()!=model.blocks.size()) throw std::logic_error("Hy4 batch cache topology changed");
    for (size_t i=0;i<cache.layers.size();++i) {
        auto& block=static_cast<Hy4Block&>(*model.blocks[i]);
        auto& incoming=cache.layers[i];
        block.kv=std::move(incoming.kv);
        block.records=std::move(incoming.records);
        block.pending_values=std::move(incoming.pending_values);
        block.pending_logits=std::move(incoming.pending_logits);
        block.sequences=std::move(incoming.sequences);
        block.last_selection=std::move(incoming.last_selection);
        block.batch_selections=std::move(incoming.batch_selections);
        block.batch_draft_kv=std::move(incoming.batch_draft_kv);
    }
    if (decoding) shared.batch_hashes=std::move(cache.hashes);
    else {
        shared.batch_hashes.clear();
        if (shared.engram && position) shared.engram->hash->restore(cache.hashes.at(0));
    }
    model.cache_pos=position;
    model.decode_position_delta=0;
}

static void reserve_cache(Tensor& target,const Tensor& source,int64_t slots,int64_t length,bool host=false) {
    const auto batch=target.defined()?target.size(0):0;
    const auto previous=target.defined()?target.size(1):0;
    if (batch>=slots && previous>=length) return;
    auto options=source.options();
    if (host) options=options.pinned_memory(true);
    auto next=tb::zeros({std::max(batch,slots),std::max(previous,length),source.size(2)},options);
    if (batch) next.narrow(0,0,batch).narrow(1,0,previous).copy_(target);
    target=std::move(next);
}

void reserve_slots(Hy4CausalLm& model,BatchCache& cache,const BatchCache& prototype,
                   int64_t slots,int64_t positions) {
    const auto& config=model.config;
    const auto limit=model.max_position_embeddings();
    const auto group=config.group_size;
    cache.layers.resize(model.blocks.size());
    if (model.shared->engram) cache.hashes.resize(slots);
    ResidentAllocationScope resident;
    for (size_t i=0;i<cache.layers.size();++i) {
        const auto& block=static_cast<const Hy4Block&>(*model.blocks[i]);
        const auto& example=prototype.layers.at(i);
        auto& layer=cache.layers[i];
        if (block.expose_selection) layer.batch_selections.resize(slots);
        if (block.reuse_draft_kv && model.shared->offload) layer.batch_draft_kv.resize(slots);
        const auto ring=config.window+int64_t(mfq::engine::mtp::kMaximumBlockwiseDraftDepth);
        const auto length=std::min(limit,block.sliding?ring:(positions+1023)/1024*1024);
        if (block.sliding || !model.shared->offload) reserve_cache(layer.kv,example.kv,slots,length);
        else layer.sequences.resize(slots);
        if (!block.indexed) continue;
        const auto groups=std::min((limit+group-1)/group,((positions+group-1)/group+255)/256*256);
        reserve_cache(layer.records,example.records,slots,groups,model.shared->host_indexer);
        reserve_cache(layer.pending_values,example.pending_values,slots,group);
        reserve_cache(layer.pending_logits,example.pending_logits,slots,group);
    }
}

void copy_to_slot(const Hy4CausalLm& model,BatchCache& cache,BatchCache& from,int64_t slot,int64_t length) {
    for (size_t i=0;i<cache.layers.size();++i) {
        const auto& block=static_cast<const Hy4Block&>(*model.blocks[i]);
        auto& source=from.layers.at(i);
        auto& target=cache.layers[i];
        if (block.expose_selection) target.batch_selections.at(slot)=std::move(source.last_selection);
        if (block.sliding) {
            const auto start=std::max(int64_t(0),length-model.config.window+1);
            auto positions=tb::arange(start,length,source.kv.options().dtype(tb::kInt64));
            const auto source_slots=tb::remainder(positions,source.kv.size(1));
            const auto target_slots=tb::remainder(positions,target.kv.size(1));
            target.kv.narrow(0,slot,1).index_copy_(1,target_slots,source.kv.index_select(1,source_slots));
        } else if (model.shared->offload) target.sequences.at(slot)=std::move(source.sequences.at(0));
        else target.kv.narrow(0,slot,1).narrow(1,0,length).copy_(source.kv.narrow(1,0,length));
        if (!block.indexed) continue;
        const auto n=length/model.config.group_size;
        if (n) target.records.narrow(0,slot,1).narrow(1,0,n).copy_(source.records.narrow(1,0,n));
        target.pending_values.narrow(0,slot,1).copy_(source.pending_values);
        target.pending_logits.narrow(0,slot,1).copy_(source.pending_logits);
    }
    if (!cache.hashes.empty()) cache.hashes.at(slot)=from.hashes.at(0);
}

void release_slot(Hy4CausalLm& model,BatchCache& cache,int64_t slot,bool bound) {
    if (!model.shared->offload) return;
    for (size_t i=0;i<model.blocks.size();++i) {
        auto& block=static_cast<Hy4Block&>(*model.blocks[i]);
        if (block.sliding) continue;
        auto& sequences=bound?block.sequences:cache.layers.at(i).sequences;
        sequences.at(slot).reset();
        auto& draft_kv=bound?block.batch_draft_kv:cache.layers.at(i).batch_draft_kv;
        if (!draft_kv.empty()) draft_kv.at(slot)={};
    }
}

BatchCache save_pending_groups(const Hy4CausalLm& model,const Tensor& slots) {
    BatchCache result;
    result.layers.resize(model.blocks.size());
    for (size_t i=0;i<model.blocks.size();++i) {
        const auto& block=static_cast<const Hy4Block&>(*model.blocks[i]);
        if (!block.indexed) continue;
        result.layers[i].pending_values=block.pending_values.index_select(0,slots);
        result.layers[i].pending_logits=block.pending_logits.index_select(0,slots);
    }
    return result;
}

void restore_pending_groups(Hy4CausalLm& model,const BatchCache& cache,const Tensor& slots) {
    for (size_t i=0;i<cache.layers.size();++i) {
        auto& block=static_cast<Hy4Block&>(*model.blocks[i]);
        if (!block.indexed) continue;
        block.pending_values.index_copy_(0,slots,cache.layers[i].pending_values);
        block.pending_logits.index_copy_(0,slots,cache.layers[i].pending_logits);
    }
}

bool can_batch(const Hy4CausalLm& model) {return bool(model.shared);}

void batch_engram_rows(Hy4CausalLm& model,std::span<const int64_t> ids,std::span<const int64_t> positions,
                       std::span<const uint8_t> active,EngramRows& output,int64_t width,bool speculative) {
    auto& shared=*model.shared;
    shared.active_rows.assign(active.begin(),active.end());
    if (!shared.engram) return;
    if (width<=0 || ids.size()!=positions.size() || ids.size()!=active.size()*size_t(width) ||
        active.size()!=shared.batch_hashes.size()) throw std::logic_error("Hy4 batched Engram geometry changed");
    auto& engram=*shared.engram;
    EngramState::HashBatch batch;
    for (size_t row=0;row<active.size();++row) {
        const auto first=row*size_t(width);
        if (positions[first]>0) engram.hash->restore(shared.batch_hashes[row]);
        auto one=engram.hash->forward(ids.data()+first,1,width,positions[first]);
        if (!row) {
            batch.batch=int(active.size());batch.tokens=int(width);batch.layers=one.layers;batch.columns=one.columns;
            batch.values.resize(active.size()*one.values.size());
        }
        const auto columns=width*one.columns;
        for (int layer=0;layer<one.layers;++layer)
            std::copy_n(one.layer_data(layer),columns,batch.values.data()+(layer*active.size()+row)*columns);
        if (!speculative && active[row]) shared.batch_hashes[row]=engram.hash->snapshot();
    }
    engram.prepare(std::move(batch));
    output.resize(engram.batch.layers);
    const tb::Device device(tb::kCUDA,model.execution->layer_placement.primary_device());
    for (int layer=0;layer<engram.batch.layers;++layer) {
        auto [rows,valid]=engram.lookup(layer,device);
        auto& [fixed_rows,fixed_valid]=output[layer];
        if (!fixed_rows.defined()) {fixed_rows=std::move(rows);fixed_valid=std::move(valid);}
        else {fixed_rows.copy_(rows);fixed_valid.copy_(valid);}
    }
}

void commit_batch_engram(Hy4CausalLm& model,std::span<const int64_t> ids,std::span<const int64_t> positions,
                        std::span<const int64_t> kept,int64_t width) {
    auto& shared=*model.shared;
    if (!shared.engram) return;
    auto& hash=*shared.engram->hash;
    for (size_t row=0;row<kept.size();++row) {
        if (!kept[row]) continue;
        const auto first=row*size_t(width);
        hash.restore(shared.batch_hashes[row]);
        (void)hash.forward(ids.data()+first,1,kept[row],positions[first]);
        shared.batch_hashes[row]=hash.snapshot();
    }
}

void commit_batch_selections(Hy4CausalLm& model,std::span<const uint8_t> active) {
    for (const auto& owned:model.blocks) {
        auto& block=static_cast<Hy4Block&>(*owned);
        if (!block.expose_selection) continue;
        for (size_t row=0;row<active.size();++row)
            if (active[row]) block.batch_selections[row]=block.last_selection.narrow(0,int64_t(row),1).clone();
    }
}

void set_batch_verification(Hy4CausalLm& model,bool value) {model.shared->batch_verification=value;}

BatchVerification take_batch_verification(Hy4CausalLm& model) {
    BatchVerification result;
    result.reserve(model.blocks.size());
    for (const auto& owned:model.blocks) {
        auto& block=static_cast<Hy4Block&>(*owned);
        result.push_back(std::move(block.verification.value()));
        block.verification.reset();
    }
    return result;
}

void retain_batch_prefix(Hy4CausalLm& model,const BatchVerification& journals,const Tensor& lengths,
        std::span<const int64_t> positions,std::span<const int64_t> kept,int64_t width) {
    for (size_t i=0;i<model.blocks.size();++i) {
        auto& block=static_cast<Hy4Block&>(*model.blocks[i]);
        const auto& journal=journals.at(i);
        for (size_t row=0;row<kept.size();++row) {
            if (!kept[row]) continue;
            if (!block.sequences.empty() && kept[row]<width)
                block.sequences.at(row)->truncate(positions[row*size_t(width)]+kept[row]);
            if (block.expose_selection) {
                block.batch_selections[row]=journal.selections.narrow(0,int64_t(row),1).narrow(1,kept[row]-1,1).clone();
                if (journal.gathered_kv.defined()) {
                    const auto mapped=journal.gathered_indices.select(0,int64_t(row)).select(0,kept[row]-1);
                    const auto position=positions[row*size_t(width)]+kept[row];
                    auto rows=journal.gathered_kv.narrow(0,int64_t(row),1)
                        .index_select(1,tb::clamp_min(mapped,0).to(tb::kInt64));
                    auto slots=tb::arange(mapped.numel(),mapped.options()).reshape({1,-1});
                    block.batch_draft_kv[row]={position,std::move(rows),tb::where(mapped>=0,slots,-1)};
                }
            }
        }
        if (block.indexed) index_pool_retain_prefix(journal.values,journal.logits,journal.positions,lengths,
            journal.pending_values,journal.pending_logits,block.pending_values,block.pending_logits);
    }
}

void set_staged_engram(Hy4CausalLm& model,const EngramRows* values) {model.shared->staged_engram=values;}
} // namespace hy4

void Hy4Model::adapter_validate_load_options() const {
    if (execution->layer_placement.enabled() || execution->tensor_parallel.enabled() || execution->expert_parallel.enabled())
        throw std::invalid_argument("Hy4 shared MLA and Engram execution currently requires one CUDA device");
}

bool Hy4Model::adapter_prefetch_token_rows(Hy4CausalLm& model,const Tensor& ids) {
    return shared && shared->engram && shared->engram->prefetch(ids,model.cache_pos);
}

void Hy4Model::adapter_load_final_state(const mfq::ModelSource& source,Tensor& norm) {
    CausalResources::adapter_load_final_state(source,norm);
    TransformerWeightLoader loader{*execution,source};
    const auto route=route_weights(loader,"model.mhc.pre.");
    output_function=route.function;output_scale=route.scale;output_base=route.base;
    shared=std::make_shared<hy4::Shared>();shared->config=config;
    const auto device=norm.device();
    shared->rope=RopeCache(metadata.max_position_embeddings,metadata.rotary_dim,metadata.rope_base,0,-1,device);
    shared->sliding_rope=RopeCache(metadata.max_position_embeddings,metadata.rotary_dim,config.sliding_rope_base,0,-1,device);
    if (config.engram) shared->engram.emplace().load(source,*config.engram);
}

void Hy4Model::configure_kv_tiers() {
    const auto option=kv_memory_options();
    if (!option.ram_preallocated && option.placement!="auto_ssd")
        throw std::invalid_argument("kv_ssd_required: disabling RAM preallocation requires SSD KV");
    const auto layers=std::count_if(metadata.layer_types.begin(),metadata.layer_types.end(),
        [](const auto& type){return type!="sliding_attention";});
    const auto indexed=std::count(metadata.layer_types.begin(),metadata.layer_types.end(),"index_pool_attention");
    if (!layers) return;
    const size_t width=config.kv_rank+metadata.rotary_dim;
    const size_t context=metadata.max_position_embeddings;
    const auto kv_bytes=layers*context*width*sizeof(uint16_t);
    const auto index_blocks=(context+config.group_size-1)/config.group_size;
    const auto index_bytes=indexed*(index_blocks*(config.index_width+2)*sizeof(int16_t)+
        2*config.group_size*config.index_width*sizeof(float));
    const auto sliding_rows=std::min(context,size_t(config.window)+mfq::engine::mtp::kMaximumBlockwiseDraftDepth);
    const auto sliding_bytes=(metadata.layer_types.size()-layers)*sliding_rows*width*sizeof(uint16_t);
    const auto staging=std::max(size_t(4096),size_t(config.index_budget)*width*sizeof(uint16_t));
    const auto available_vram=default_context(execution->layer_placement.primary_device())->available_memory();
    const auto workspace=runtime_options::workspace_reserve_bytes()+sliding_bytes;
    const auto host=mfq::sample_host_memory().available_host_bytes();
    const auto available_ram=option.ram_limit_set?std::min(host,option.ram_limit):host;
    const auto* force_cpu=std::getenv("TYLOINFER_QSA_CPU_INDEXER");
    const bool cpu_requested=force_cpu && force_cpu[0]=='1';
    const bool index_fits=available_vram>=workspace && available_vram-workspace>=index_bytes;
    shared->host_indexer=cpu_requested || (!force_cpu && !index_fits && option.placement!="vram" &&
        !option.gpu_budget && available_ram>=index_bytes);
    if (shared->host_indexer) {
        int mapped=0;
        TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&mapped,cudaDevAttrCanMapHostMemory,
            execution->layer_placement.primary_device()));
        if (!mapped || available_ram<index_bytes)
            throw std::invalid_argument("kv_ram_insufficient: Hy4 cannot place the IndexPool cache in mapped RAM");
    }
    const auto vram_indexer=shared->host_indexer?size_t(0):index_bytes;
    const auto ram=available_ram-(shared->host_indexer?index_bytes:0);
    const auto fixed=workspace+vram_indexer;
    if (option.gpu_budget && option.gpu_budget<vram_indexer)
        throw std::invalid_argument("kv_budget_below_indexer: Hy4 VRAM KV budget must cover the device IndexPool cache");
    if (available_vram<fixed)
        throw std::invalid_argument("kv_vram_insufficient: Hy4 indexer and sliding attention require more VRAM");
    auto available=available_vram-fixed;
    if (option.gpu_limit) available=std::min(available,option.gpu_limit);
    const auto desired=option.gpu_budget?option.gpu_budget-vram_indexer:
        option.budget_policy && option.placement!="vram" && !option.dynamic
            ?size_t(std::ceil(kv_bytes*kv_resident_fraction(context))):kv_bytes;
    if (option.gpu_budget && option.placement!="ram" && desired>available)
        throw std::invalid_argument("kv_vram_insufficient: explicit KV VRAM budget exceeds available capacity after workspace");
    const auto quantization=cuda_kv_quantization();
    if (!std::getenv("TYLOINFER_QSA_KV_BUDGET_BYTES") &&
        option.placement!="ram" && kv_bytes<=std::min(desired,available) &&
        (!quantization.enabled() || option.placement=="vram")) return;
    if (option.placement=="vram")
        throw std::invalid_argument("kv_vram_insufficient: configured Hy4 KV does not fit in VRAM");
    if (option.dynamic)
        throw std::invalid_argument("kv_dynamic_unavailable: dynamic Hy4 KV requires a resident cache");
    const auto gpu=option.placement=="ram"?size_t(0):std::min(desired,available>staging?available-staging:0);
    shared->cold_bits=quantization.key_bits();
    const auto row_bytes=std::size_t(shared->storage_width())*(shared->cold_bits?sizeof(int32_t):sizeof(uint16_t));
    const auto encoded=layers*context*row_bytes;
    const auto needed=encoded-std::min(gpu,encoded)+staging;
    if (option.placement!="auto_ssd" && needed>ram)
        throw std::invalid_argument("kv_memory_insufficient: enable SSD KV or lower the context length");
    KvOffloadConfig tiers;
    tiers.buffer_bytes=staging;
    tiers.gpu_budget_bytes=std::min(gpu,encoded)+tiers.buffer_bytes;
    tiers.ram_budget_bytes=std::min(ram,needed);
    tiers.preallocate=true;
    tiers.ram_preallocated=option.ram_preallocated;
    tiers.ssd_enabled=option.placement=="auto_ssd";
    tiers.directory=option.directory;
    if (tiers.ram_preallocated && !tiers.ssd_enabled)
        tiers.ram_growth_limit=option.ram_limit_set?ram:SIZE_MAX;
    if (tiers.ssd_enabled && (!tiers.ram_preallocated || needed>ram)) {
        const auto spill=tiers.ram_preallocated?needed-ram:needed;
        auto directory=std::filesystem::path(tiers.directory);
        while (!directory.empty() && !std::filesystem::exists(directory)) directory=directory.parent_path();
        if (directory.empty() || std::filesystem::space(directory).available<spill)
            throw std::invalid_argument("kv_ssd_insufficient: Hy4 does not have space for its cold KV cache");
        std::cerr<<"runtime_kv_warning code=kv_ssd_overflow target_spill_bytes="<<spill<<'\n';
    }
    shared->offload=std::make_shared<KvOffloadStore>(std::move(tiers));
    execution->qsa_kv_store=shared->offload;
    if (shared->cold_bits)
        std::fprintf(stderr,"native_hy4_cold_kv bits=%d row_bytes=%zu\n",shared->cold_bits,row_bytes);
}

int Hy4Model::cold_kv_bits() const noexcept {return shared?shared->cold_bits:0;}

std::unique_ptr<Block> Hy4Model::adapter_load_block(const mfq::ModelSource& source,int index,int device,
                                                  const std::string& type) {
    auto result=std::make_unique<Hy4Block>();
    auto& b=*result;
    b.cuda_device=device;b.layer=index;b.heads=config.heads[index];b.state=shared;
    b.sliding=type=="sliding_attention";b.indexed=type=="index_pool_attention";
    if (!(b.sliding || b.indexed || type=="full_attention")) throw std::invalid_argument("unknown Hy4 layer type");
    TransformerWeightLoader loader{*execution,source};
    const auto root="model.block."+std::to_string(index)+".";
    const auto a=root+"attention.",m=root+"mlp.";
    b.attention_hc=route_weights(loader,a+"mhc.pre.");b.ffn_hc=route_weights(loader,m+"mhc.pre.");
    b.attention_norm=loader.dense(a+"norm.weight");b.attention_output_norm=loader.dense(a+"output_norm.weight");
    b.ffn_norm=loader.dense(m+"norm.weight");b.ffn_output_norm=loader.dense(m+"output_norm.weight");
    b.q_a=loader.linear(a+"query_a.weight");b.q_b=loader.linear(a+"query_b.weight");b.q_norm=loader.dense(a+"query_a_norm.weight");
    b.kv_a=loader.linear(a+"key_value_a.weight");b.kv_norm=loader.dense(a+"key_value_norm.weight");
    b.key_b=load_dense_native_gpu(*execution,source,a+"key_b.weight");
    b.value_b=load_dense_native_gpu(*execution,source,a+"value_b.weight");
    b.gate=loader.linear(a+"gate.weight");b.output=loader.linear(a+"output.weight");b.sinks=loader.dense(a+"sink");
    if (b.indexed) {
        const auto ix=a+"indexer.";
        b.index_q=loader.linear(ix+"query.weight");b.index_k=loader.linear(ix+"key.weight");
        b.index_score=loader.linear(ix+"score.weight");b.index_key_norm=loader.dense(ix+"key_norm.weight");
        b.index_bias=loader.dense(ix+"pool.position_bias");b.index_pool_norm=loader.dense(ix+"pool.norm.weight");
        b.index_pool_value_gate=loader.projections({ix+"pool.value.weight",ix+"pool.gate.weight"});
    }
    if (index<config.dense_layers) {
        mfq::models::ModelConfig geometry;
        geometry.hidden_size=metadata.hidden_size;geometry.intermediate_size=config.dense_width;
        b.ffn=mfq::models::load_dense_ffn<FFN>(loader,geometry,m);
    } else {
        b.ffn=load_moe_weights(*execution,source,m,MoeWeightLoadOptions{index});
        b.ffn.moe_top_k=int(config.expert_top_k);b.ffn.moe_normalize=config.routing.normalize;
        b.ffn.moe_use_sigmoid=config.routing.activation==mfq::models::RouterActivation::sigmoid;
        b.ffn.moe_delayed_softmax=false;b.ffn.moe_router_scale=config.routing.scale;
        b.ffn.moe_shared_ungated=true;b.ffn.swiglu_limit=config.swiglu_limit[index];
    }
    if (config.engram) {
        const auto& layers=config.engram->layers;
        const auto found=std::find(layers.begin(),layers.end(),index);
        if (found!=layers.end()) {
            const auto e=root+"associative_memory.";
            b.engram.slot=int(found-layers.begin());
            b.engram.projection=loader.linear(e+"projection.weight");
            b.engram.projection_bias=loader.dense(e+"projection.bias");b.engram.lookup_norm=loader.dense(e+"lookup_norm.weight");
            b.engram.gate=loader.dense(e+"gate.weight");b.engram.hc_scale=loader.dense(e+"hc_scale.weight");
            b.engram.gamma=loader.dense(e+"output_norm.weight");
            b.engram.unit=tb::ones({metadata.hidden_size},b.engram.gamma.options());
        }
    }
    return result;
}

Tensor Hy4Model::adapter_prepare_hidden(Tensor x,int64_t batch,int64_t tokens) const {
    return x.to(tb::kFloat16).unsqueeze(2).expand({batch,tokens,metadata.hc_mult,metadata.hidden_size}).contiguous();
}

Tensor Hy4Model::adapter_finalize_hidden(Tensor x,const Tensor& norm,int64_t,int64_t) const {
    auto reduced=independent_hc_prepare(x,output_function,output_scale,output_base,float(metadata.rms_norm_eps),
        float(metadata.hc_eps),float(config.hc_magnitude)).branch;
    return rms_norm_cuda(reduced,norm,metadata.rms_norm_eps);
}

void Hy4Model::adapter_commit_speculative() {
    if (shared->engram) shared->engram->hash->commit_speculative();
}

void Hy4Model::adapter_reset(int64_t batch) {
    CausalResources::adapter_reset(batch);
    if (shared) adapter_commit_speculative();
}

void Hy4Model::adapter_rollback_speculative(int64_t keep) {
    if (shared->engram) shared->engram->hash->rollback_speculative(keep);
}

void enable_hy4_draft_history(Hy4CausalLm& model,const std::vector<int64_t>& layers,
        const std::vector<int64_t>& reusable) {
    for (const auto i:layers) static_cast<Hy4Block&>(*model.blocks.at(i)).expose_selection=true;
    for (const auto i:reusable) static_cast<Hy4Block&>(*model.blocks.at(i)).reuse_draft_kv=true;
}

const RopeCache& hy4_full_rope(const Hy4CausalLm& model) {return model.shared->rope;}

void clear_hy4_decode_graph(Hy4CausalLm& model) {
    if (model.shared) model.shared->graphs.reset();
}

void prepare_hy4_expert_cache(Hy4CausalLm& model) {
    std::vector<std::array<std::shared_ptr<GgufRoutedWeight>,3>> layers;
    for (const auto& owned:model.blocks) {
        const auto& layer=static_cast<const Hy4Block&>(*owned);
        if (const auto& weights=layer.ffn.moe_gguf)
            layers.push_back({weights->gate,weights->up,weights->down});
    }
    prepare_gguf_expert_slots(model.execution->gguf_expert_cache,layers);
}

std::optional<Tensor> hy4_graph_hidden(Hy4CausalLm& model,const Tensor& ids,const Tensor& host_ids,
        int64_t confirmed,int64_t horizon,Tensor& logits,bool& initialized) {
    if (model.execution->profiler.enabled || !mfq_cuda_graph_capture_supported() ||
        model.speculative_start>=0 || model.cache_pos<=0 || model.decode_position_delta!=0 ||
        !ids.is_cuda() || ids.dim()!=2 || ids.size(0)!=1 ||
        ids.size(1)>1+mfq::engine::mtp::kMaximumBlockwiseDraftDepth ||
        (ids.size(1)!=1 && !confirmed)) return std::nullopt;
    auto& state=*model.shared;
    auto& graphs=state.graphs;
    const auto position=model.cache_pos,tokens=ids.size(1);
    const auto planned=decode_graph_bucket(std::max(position+tokens,horizon),model.max_position_embeddings());
    const auto stream=mfq_current_cuda_stream();
    if (graphs && (graphs->length!=planned || graphs->stream!=stream)) graphs.reset();
    std::optional<mfq::models::deepseek_v41::EngramHashSnapshot> previous_hash;
    bool submitted=false;
    try {
        std::vector<const void*> addresses;
        for (const auto& owned:model.blocks) {
            auto& block=static_cast<Hy4Block&>(*owned);
            block.reserve_decode(ids,position,planned);
            for (const auto* tensor:{&block.kv,&block.records,&block.pending_values,&block.pending_logits})
                addresses.push_back(tensor->defined()?tensor->data_ptr():nullptr);
        }
        if (!graphs || graphs->state!=addresses) {
            graphs=std::make_unique<hy4::DecodeGraphs>();
            graphs->length=planned;
            graphs->stream=stream;
            graphs->state=std::move(addresses);
        }
        auto& entry=graphs->cells[{tokens,confirmed}];
        if (!entry) {
            ResidentAllocationScope resident;
            auto cell=std::make_unique<hy4::DecodeGraphs::Cell>();
            cell->ids=tb::empty_like(ids);
            cell->positions=tb::empty({tokens},ids.options().dtype(tb::kInt64));
            cell->lengths=tb::empty({1},cell->positions.options());
            cell->pool=GraphPoolLease::acquire(default_context(ids.get_device()),stream);
            cell->window=std::make_unique<DecodeWindow>(stream);
            cell->window->set_profile_geometry(confirmed?"hy4_verify":"hy4_decode",int(tokens));
            entry=std::move(cell);
        }
        auto& graph=*entry;
        graph.ids.copy_(ids);
        graph.positions.copy_(tb::arange(position,position+tokens,graph.positions.options()));
        graph.lengths.fill_(position+tokens);
        if (state.engram) {
            previous_hash=state.engram->hash->snapshot();
            if (confirmed) state.engram->hash->begin_speculative();
            state.engram->prepare(host_ids.defined()?host_ids:ids.to(tb::kCPU),position);
            const auto layers=state.engram->batch.layers;
            if (graph.engram_rows.empty()) graph.engram_rows.resize(layers);
            for (int slot=0;slot<layers;++slot) {
                auto [rows,valid]=state.engram->lookup(slot,ids.device());
                auto& [fixed_rows,fixed_valid]=graph.engram_rows[slot];
                if (!fixed_rows.defined()) {
                    fixed_rows=std::move(rows);
                    fixed_valid=std::move(valid);
                } else {
                    fixed_rows.copy_(rows);
                    fixed_valid.copy_(valid);
                }
            }
        }
        if (!graph.window->valid()) {
            std::vector<std::pair<Tensor*,Tensor>> pending;
            std::vector<Tensor> previous_selections;
            for (const auto& owned:model.blocks) {
                auto& block=static_cast<Hy4Block&>(*owned);
                previous_selections.push_back(block.last_selection);
                for (auto* cell:block.graph_warmup_state()) if (cell->defined())
                    pending.emplace_back(cell,cell->clone());
            }
            bool recorded=false;
            const auto restore=[&] {
                for (const auto& [destination,value]:pending) destination->copy_(value);
                if (recorded) {
                    graph.journals.clear();
                    graph.selections.clear();
                }
                for (size_t i=0;i<model.blocks.size();++i) {
                    auto& block=static_cast<Hy4Block&>(*model.blocks[i]);
                    if (recorded) {
                        graph.journals.push_back(std::move(block.verification));
                        graph.selections.push_back(block.last_selection);
                    }
                    block.verification.reset();
                    block.last_selection=previous_selections[i];
                }
                model.speculative_start=-1;
                model.speculative_confirmed=0;
                model.cache_pos=position;
            };
            {
                struct Recording {
                    hy4::Shared& state;
                    Recording(hy4::Shared& owner,const hy4::EngramRows& rows):state(owner) {state.staged_engram=&rows;}
                    ~Recording() {state.staged_engram=nullptr;}
                } recording(state,graph.engram_rows);
                graph.window->capture([&] {
                    cudaStreamCaptureStatus status;
                    TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stream,&status));
                    graph.hidden=model.hidden_forward(graph.ids,graph.positions,graph.lengths,nullptr,
                        graph.positions,nullptr,confirmed,{planned,0});
                    graph.logits=model.logits_from_hidden(graph.hidden);
                    recorded=status==cudaStreamCaptureStatusActive;
                },restore,[&]{graph.hidden={};graph.logits={};});
            }
            initialized=true;
            std::fprintf(stderr,"runtime_hy4_decode_graph planned=%lld rows=%lld verify=%d nodes=%zu segments=%zu\n",
                static_cast<long long>(planned),static_cast<long long>(tokens),int(confirmed>0),
                graph.window->nodes(),graph.window->segments());
        }
        submitted=true;
        if (!graph.window->enqueue_if_device_only()) graph.window->run();
        for (size_t i=0;i<model.blocks.size();++i) {
            auto& block=static_cast<Hy4Block&>(*model.blocks[i]);
            if (block.expose_selection) block.last_selection=graph.selections[i];
            if (confirmed) {
                block.verification=graph.journals[i];
                block.verification->start=position;
            }
        }
        if (confirmed) {
            model.speculative_start=position;
            model.speculative_confirmed=confirmed;
        }
        model.cache_pos=position+tokens;
        logits=graph.logits;
        return graph.hidden;
    } catch (...) {
        graphs.reset();
        if (!submitted && previous_hash) state.engram->hash->restore(*previous_hash);
        try {throw;}
        catch (const MfqResourceExhausted&) {
            if (!submitted) return std::nullopt;
            throw;
        }
    }
}

std::optional<Tensor> hy4_decode_graph_logits(Hy4CausalLm& model,const Tensor& ids,
        const Tensor& host_ids,int64_t horizon) {
    Tensor logits;
    bool initialized=false;
    if (!hy4_graph_hidden(model,ids,host_ids,0,horizon,logits,initialized)) return std::nullopt;
    return logits.reshape({ids.size(0),logits.size(-1)});
}

int64_t hy4_draft_selection_width(const Hy4CausalLm& model,int64_t layer,int64_t slot) {
    const auto& block=static_cast<const Hy4Block&>(*model.blocks.at(layer));
    const auto& selected=slot<0?block.last_selection:block.batch_selections.at(slot);
    if (!selected.defined()) throw std::logic_error("DFly2 target selection is unavailable");
    return selected.size(2);
}

std::pair<Tensor,Tensor> hy4_draft_history(const Hy4CausalLm& model,int64_t kv_layer,int64_t index_layer,
                                        int64_t position,int window,int64_t slot) {
    const auto& source=static_cast<const Hy4Block&>(*model.blocks.at(kv_layer));
    const auto& index=static_cast<const Hy4Block&>(*model.blocks.at(index_layer));
    if (!window && kv_layer==index_layer && slot>=0 && size_t(slot)<source.batch_draft_kv.size()) {
        const auto& prepared=source.batch_draft_kv[slot];
        if (prepared.position==position) return {prepared.rows,prepared.indices};
    }
    const auto& selected=slot<0?index.last_selection:index.batch_selections.at(slot);
    if (!selected.defined() || selected.size(0)!=1 || position<1)
        throw std::logic_error("DFly2 target history is unavailable");
    auto lengths=tb::full({1},position,selected.options().dtype(tb::kInt64));
    auto indices=index_pool_draft_history(selected,lengths,window).squeeze(0);
    if (source.sequences.empty()) {
        auto history=source.kv.narrow(0,std::max<int64_t>(slot,0),1).index_select(1,tb::clamp_min(indices.reshape({-1}),0).to(tb::kInt64));
        auto slots=tb::arange(indices.size(1),indices.options()).reshape({1,-1});
        return {std::move(history),tb::where(indices>=0,slots,-1)};
    }
    auto host=indices.to(tb::kCPU).contiguous();
    auto* data=host.data_ptr<int32_t>();
    std::vector<int64_t> rows;
    rows.reserve(size_t(host.numel()));
    for (int64_t i=0;i<host.numel();++i) {
        if (data[i]<0) continue;
        rows.push_back(data[i]);
        data[i]=int32_t(rows.size()-1);
    }
    auto& sequence=*source.sequences.at(std::max<int64_t>(slot,0));
    auto workspace=sequence.gather_workspace(int64_t(std::bit_ceil(rows.size())));
    sequence.gather_into(rows,*workspace);
    auto history=workspace->rows.narrow(0,0,int64_t(rows.size())).unsqueeze(0);
    return {model.shared->decode_kv(history),host.to(indices.device())};
}

TextSessionStateKind CudaSessionCodec<Hy4Model>::kind(const Model& model) {
    return model.shared && !model.blocks.empty()
        ?TextSessionStateKind::HybridAttention:TextSessionStateKind::Unsupported;
}

bool CudaSessionCodec<Hy4Model>::supports_paged(const Model& model) {
    return kind(model)==TextSessionStateKind::HybridAttention;
}

TextSessionState CudaSessionCodec<Hy4Model>::capture(const Model& model,const std::vector<int64_t>& tokens) {
    if (!supports_paged(model) || model.cache_pos<=0 || size_t(model.cache_pos)!=tokens.size())
        throw CudaSessionStateError("Hy4 checkpoint boundary is unavailable");
    TextSessionState result;
    result.cache_pos=model.cache_pos;
    result.tokens=tokens;
    auto& layers=result.payload.emplace<std::vector<HybridBlockSessionState>>();
    layers.reserve(model.blocks.size());
    for (const auto& block:model.blocks) {
        const auto& source=static_cast<const Hy4Block&>(*block);
        HybridBlockSessionState layer;
        layer.kind=HybridBlockSessionStateKind::LatentAttention;
        layer.latent_device=source.cuda_device;
        if (!source.sliding && model.shared->offload) {
            if (source.sequences.size()!=1 || source.sequences.front()->position()!=model.cache_pos)
                throw CudaSessionStateError("Hy4 checkpoint requires one active offloaded sequence");
            auto rows=source.sequences.front()->range_host(0,model.cache_pos);
            layer.full_attention={rows.unsqueeze(0).unsqueeze(0),{},model.cache_pos,false};
        } else {
            if (!source.kv.defined() || source.kv.size(0)!=1)
                throw CudaSessionStateError("Hy4 checkpoint requires one active sequence");
            const auto capacity=source.kv.size(1);
            const auto saved=source.sliding?capacity:model.cache_pos;
            layer.full_attention={source.kv.narrow(1,0,saved).unsqueeze(1).clone(),{},capacity,source.sliding};
        }
        if (source.indexed) {
            layer.group_size=model.config.group_size;
            layer.index_pool=source.records.narrow(1,0,model.cache_pos/layer.group_size).clone();
            layer.pending_values=source.pending_values.clone();
            layer.pending_logits=source.pending_logits.clone();
        }
        for (const auto* tensor:{&layer.full_attention.k,&layer.index_pool,&layer.pending_values,&layer.pending_logits})
            result.bytes+=session_tensor_bytes(*tensor);
        layers.push_back(std::move(layer));
    }
    return result;
}

void CudaSessionCodec<Hy4Model>::restore(Model& model,const TextSessionState& state) {
    const auto* layers=std::get_if<std::vector<HybridBlockSessionState>>(&state.payload);
    if (!supports_paged(model) || !layers || layers->size()!=model.blocks.size() || state.cache_pos<=0 ||
        size_t(state.cache_pos)!=state.tokens.size() || state.cache_pos>model.max_position_embeddings())
        throw CudaSessionStateError("Hy4 checkpoint topology changed");
    const auto& config=model.config;
    for (size_t i=0;i<layers->size();++i) {
        const auto& layer=layers->at(i);
        const auto& saved=layer.full_attention;
        auto& target=static_cast<Hy4Block&>(*model.blocks[i]);
        const tb::Device device(tb::kCUDA,target.cuda_device);
        const auto count=target.sliding?saved.capacity:state.cache_pos;
        const bool streamed=!target.sliding && model.shared->offload;
        const auto width=streamed?model.shared->storage_width():config.kv_rank+model.metadata.rotary_dim;
        const auto dtype=streamed?model.shared->storage_dtype():tb::kFloat16;
        if (layer.kind!=HybridBlockSessionStateKind::LatentAttention || saved.ring!=target.sliding ||
            !saved.k.defined() || layer.latent_device!=target.cuda_device ||
            saved.k.scalar_type()!=dtype || saved.k.sizes().vec()!=std::vector<int64_t>{1,1,count,width} ||
            layer.group_size!=(target.indexed?config.group_size:0))
            throw CudaSessionStateError("Hy4 latent KV checkpoint geometry changed");
        if (streamed) {
            target.sequences.clear();
            auto sequence=std::make_unique<KvOffloadSequence>(
                model.shared->offload,width,config.group_size,dtype);
            sequence->append_host(saved.k.reshape({count,width}).to(tb::kCPU),device);
            target.sequences.push_back(std::move(sequence));
        } else {
            const auto previous=target.kv.defined()?target.kv.size(1):0;
            const auto capacity=target.sliding?saved.capacity:std::max(saved.capacity,previous);
            restore_session_prefix_tensor(target.kv,saved.k.squeeze(1).to(device),1,capacity);
        }
        target.last_selection={};
        if (!target.indexed) continue;
        const auto groups=state.cache_pos/layer.group_size;
        if (groups && (!layer.index_pool.defined() || layer.index_pool.scalar_type()!=tb::kInt16 ||
            layer.index_pool.sizes().vec()!=std::vector<int64_t>{1,groups,config.index_width+2}))
            throw CudaSessionStateError("Hy4 index prefix geometry changed");
        for (const auto* tensor:{&layer.pending_values,&layer.pending_logits})
            if (!tensor->defined() || tensor->scalar_type()!=tb::kFloat32 || tensor->sizes().vec()!=
                std::vector<int64_t>{1,config.group_size,config.index_width})
                throw CudaSessionStateError("Hy4 pending index prefix geometry changed");
        if (groups) {
            const auto previous=target.records.defined()?target.records.size(1):0;
            if (model.shared->host_indexer) {
                target.records=target.allocate_growing(target.records,saved.k,std::max(groups,previous),
                    config.index_width+2,tb::kInt16,false,0,true);
                target.records.narrow(1,0,groups).copy_(layer.index_pool.to(tb::kCPU));
            } else {
                restore_session_prefix_tensor(target.records,layer.index_pool.to(device),1,std::max(groups,previous));
            }
        }
        restore_session_tensor(target.pending_values,layer.pending_values);
        restore_session_tensor(target.pending_logits,layer.pending_logits);
    }
    if (model.shared->engram) {
        const auto& e=*config.engram;
        std::vector<int64_t> history(e.ngram-1,e.remap[e.pad]);
        const auto count=std::min(history.size(),state.tokens.size());
        for (size_t i=0;i<count;++i) {
            const auto token=state.tokens[state.tokens.size()-count+i];
            if (token<0 || token>=int64_t(e.remap.size())) throw CudaSessionStateError("invalid Hy4 prefix token");
            history[history.size()-count+i]=e.remap[token];
        }
        model.shared->engram->hash->restore({1,state.cache_pos,std::move(history)});
    }
    model.cache_pos=state.cache_pos;
}

}  // namespace mfq::cuda

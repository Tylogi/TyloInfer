#include "ops.h"
#include "batch_state.h"

#include "core/decode_graph.h"
#include "core/ffn.h"
#include "core/mtp.h"
#include "core/kv_cache.h"
#include "mfq_cuda_attention_ops.h"
#include "mfq_cuda_norm_ops.h"
#include "mfq_cuda_sampling_ops.h"
#include "runtime/decode_window.h"
#include "runtime/execution_options.h"
#include "runtime/graph_pool_lease.h"
#include "runtime/routed_cache_resources.h"
#include "storage/session_state.h"
#include "storage/transformer_loader.h"
#include "storage/weight_loader.h"
#include "app/server/kernels/cuda/loop_mixing.h"

#include <array>
#include <cmath>
#include <map>

namespace mfq::cuda {
namespace tb=mfq_tensor_backend;
using Tensor=tb::Tensor;

struct InklingWeights {
    struct Controller {QuantLinear down;QuantLinearGroup gain_forget;};
    QuantLinear k,v,output,key_taps,value_taps;
    QuantLinearGroup query_projections;
    std::optional<QuantLinearGroup> sliding_qkv;
    Tensor q_norm,k_norm,v_norm,ffn_norm,query_clip,relative_clip,relative_bias;
    Tensor attention_query,ffn_query,loop_query,unit;
    std::array<Tensor,4> convolution;
    std::array<Controller,4> controls;
    FFN ffn;
};

namespace {

Tensor clip_norm(Tensor value,const Tensor& multiplier,float limit) {
    if (auto normalized=normalize_loop_attention_projection(value,multiplier,limit,true);normalized.defined())
        return normalized;
    if (multiplier.defined()) value=value*multiplier;
    auto norm=tb::sqrt(tb::sum(value*value,-1,true))+1e-6f;
    return value*tb::clamp_max(limit/norm,1.0);
}

Tensor depth_mix(const std::vector<Tensor>& bank,const Tensor& hidden,const Tensor& query) {
    const auto batch=hidden.size(0),tokens=hidden.size(1),width=hidden.size(2),heads=query.numel()/width;
    return residual_depth_mix(bank,hidden,query.reshape({heads,width}),1e-6f).reshape({batch,tokens,heads,width});
}

struct InklingBlock final : Block {
    std::shared_ptr<const InklingWeights> weights;
    Tensor output_query;
    int64_t visit=0,physical=0,q_heads=0,kv_heads=0;
    int64_t head_dim=0,context_limit=0,window=0,loop_first=0,loop_end=0,residual_block=0;
    int64_t relative_dim=0,log_floor=0,taps=0;
    float qk_clip=0,relative_clip=0,log_alpha=0;
    double eps=0;
    bool sliding=false,repeated=false;
    KVCache cache;
    std::array<Tensor,4> history;
    using Verification=inkling::ConvolutionVerification;
    std::optional<Verification> verification;
    bool batch_verifying=false;

    void reset(int64_t batch) override {
        verification.reset();
        batch_verifying=false;
        for (auto& state:history) {
            if (state.defined() && state.size(0)==batch) state.zero_();
            else state={};
        }
        if (cache.batch_size()!=batch) cache=KVCache();
    }

    bool supports_speculation() const noexcept override {return true;}
    void commit_speculative() override {verification.reset();}
    void rollback_speculative(int64_t keep) override {
        if (!verification) return;
        auto saved=std::move(*verification);
        verification.reset();
        const auto accepted=keep-saved.start;
        for (size_t slot=0;slot<history.size();++slot) {
            const auto& input=saved.inputs[slot];
            if (!input.defined() || accepted<0 || accepted>input.size(1))
                throw std::logic_error("invalid Inkling verification boundary");
            const auto retained=std::min(accepted,taps-1);
            if (retained<taps-1)
                history[slot].narrow(1,0,taps-1-retained).copy_(saved.previous[slot].narrow(1,retained,taps-1-retained));
            if (retained)
                history[slot].narrow(1,taps-1-retained,retained).copy_(input.narrow(1,accepted-retained,retained));
        }
    }

    std::vector<Tensor*> graph_warmup_state() override {
        return {&history[0],&history[1],&history[2],&history[3]};
    }

    Tensor control(CudaExecutionContext& execution,Tensor current,const Tensor& message,int branch) const {
        const auto& w=*weights;
        const auto& c=w.controls[branch];
        auto joined=rms_norm_concat_pair_cuda(current,message,w.unit,1e-6);
        const auto outputs=c.gain_forget.forward(execution,c.down.forward(execution,joined));
        return blend_loop_message(current,message,outputs[0],outputs[1]);
    }

    Tensor conv(Tensor input,int slot,const Tensor& dynamic={}) {
        input=input.to(tb::kFloat32).contiguous();
        if (!history[slot].defined())
            history[slot]=tb::zeros({input.size(0),taps-1,input.size(2)},input.options());
        if (verification) {
            verification->previous[slot]=history[slot].clone();
            verification->inputs[slot]=input;
        }
        return residual_short_convolution(history[slot],input,weights->convolution[slot],dynamic);
    }

    void prepare_kv(const Context& context,const Tensor& key,int64_t tokens) {
        const auto current=context.cache_position;
        const auto planned=std::max(current+tokens,context.planned_kv_length);
        const auto capacity=sliding?std::min(context_limit,window+tokens-1):
            std::min(context_limit,(planned+1023)/1024*1024);
        if (cache.defined() && cache.k.size(2)>=capacity) return;
        ResidentAllocationScope persistent_cache;
        KVCache replacement(key.size(0),kv_heads,capacity,head_dim,sliding,key.device());
        if (cache.defined() && current>0) {
            const auto old_capacity=cache.k.size(2);
            if (sliding) {
                const auto start=std::max(int64_t(0),current-window+1);
                auto positions=tb::arange(start,current,key.options().dtype(tb::kInt64));
                auto old_slots=tb::remainder(positions,old_capacity);
                auto new_slots=tb::remainder(positions,capacity);
                replacement.k.index_copy_(2,new_slots,cache.k.index_select(2,old_slots));
                replacement.v.index_copy_(2,new_slots,cache.v.index_select(2,old_slots));
            } else {
                replacement.k.narrow(2,0,current).copy_(cache.k.narrow(2,0,current));
                replacement.v.narrow(2,0,current).copy_(cache.v.narrow(2,0,current));
            }
        }
        cache=std::move(replacement);
    }

    Tensor attention(CudaExecutionContext& execution,const std::array<Tensor,3>& rows,const Context& context) {
        const auto& w=*weights;
        const auto b=rows[0].size(0),t=rows[0].size(1);
        std::array<Tensor,3> x;
        x[0]=rms_norm_cuda(rows[0].contiguous(),w.q_norm,eps);
        x[1]=sliding && !repeated?x[0]:rms_norm_cuda(rows[1].contiguous(),w.k_norm,eps);
        x[2]=sliding && !repeated?x[0]:rms_norm_cuda(rows[2].contiguous(),w.v_norm,eps);
        const auto shaped=[&](const QuantLinear& projection,const Tensor& source,int64_t heads,int64_t width) {
            return projection.forward(execution,source).to(tb::kFloat32).reshape({b,t,heads,width});
        };
        const auto dynamic=[&](const QuantLinear& project,const Tensor& input) {
            return sliding?Tensor{}:shaped(project,input,kv_heads,taps);
        };
        const bool grouped_kv=sliding && !repeated && w.sliding_qkv.has_value();
        auto projected=(grouped_kv?*w.sliding_qkv:w.query_projections).forward(execution,x[0]);
        auto query=clip_norm(projected[0].to(tb::kFloat32).reshape({b,t,q_heads,head_dim}),w.query_clip,qk_clip);
        auto position=clip_norm(projected[1].to(tb::kFloat32).reshape({b,t,q_heads,relative_dim}),
            w.relative_clip,relative_clip);
        auto key=conv(grouped_kv?std::move(projected[4]):w.k.forward(execution,x[1]),0,
            dynamic(w.key_taps,x[1])).reshape({b,t,kv_heads,head_dim});
        auto value=conv(grouped_kv?std::move(projected[5]):w.v.forward(execution,x[2]),1,
            dynamic(w.value_taps,x[2])).reshape({b,t,kv_heads,head_dim});
        key=clip_norm(key,{},qk_clip);
        value=normalize_loop_attention_projection(value,{},float(eps),false);
        if (!sliding && log_alpha>0) {
            auto scale=1.0+tb::log(tb::clamp_min((context.positions.to(tb::kFloat32)+1.0)/log_floor,1.0))*log_alpha;
            scale=scale.reshape({-1,t,1,1});
            query=query*scale;
            position=position*scale;
        }
        prepare_kv(context,key,t);
        const auto physical_positions=context.cache_positions.has_value()?context.cache_positions.value():context.positions;
        const auto end=context.cache_position+t;
        (void)cache.append(execution.config,key.transpose(1,2),value.transpose(1,2),physical_positions,
            context.cache_position,end);
        const auto lengths=context.sequence_lengths.has_value()?context.sequence_lengths.value():
            tb::full({b},end,physical_positions.options().dtype(tb::kInt64));
        const auto planned=std::min(context_limit,std::max(end,context.planned_kv_length));
        auto bias=tb::matmul(position,w.relative_bias.transpose(0,1)).transpose(1,2).contiguous();
        auto sinks=projected[2].to(tb::kFloat32).transpose(1,2).contiguous();
        auto attended=attention_cache_relative_swa_cuda(query.to(tb::kFloat16).transpose(1,2).contiguous(),
            cache.k,cache.v,lengths,bias,sinks,1/std::sqrt(double(head_dim)),std::min(window,planned),planned);
        attended=attended.transpose(1,2).reshape({b,t,q_heads*head_dim}).contiguous();
        return conv(w.output.forward(execution,attended*tb::sigmoid(projected[3])),2);
    }

    Tensor forward_context(CudaExecutionContext& execution,Tensor hidden,const Context& context,
                           const RopeCache&) override {
        if (context.confirmed_prefix>0 || batch_verifying) {
            verification.emplace();
            verification->start=context.cache_position;
        }
        auto* recording=DecodeWindow::recording();
        const auto mark=[&](const char* phase) {
            if (recording) recording->mark(int(visit),phase);
        };
        mark("inkling_block");
        auto& depth=*context.residual_depth;
        const auto& w=*weights;
        if (visit==loop_first) {depth.prelude=hidden;depth.prelude_blocks=depth.blocks;}
        if (repeated && physical==loop_first) {
            depth.previous_pass.assign(depth.blocks.begin()+depth.prelude_blocks.size()+1,depth.blocks.end());
            depth.previous_output=hidden;
            hidden=depth.prelude;
            depth.blocks=depth.prelude_blocks;
        }
        Tensor messages;
        if (repeated) messages=depth_mix(depth.previous_pass,depth.previous_output,w.loop_query);
        auto mixed=w.attention_query.defined()?depth_mix(depth.blocks,hidden,w.attention_query):hidden.unsqueeze(2);
        std::array<Tensor,3> inputs;
        for (int i=0;i<3;++i) {
            inputs[i]=mixed.select(2,sliding || !w.attention_query.defined()?0:i);
            if (repeated) inputs[i]=control(execution,inputs[i],messages.select(2,i),i);
        }
        mark("inkling_attention");
        auto current=attention(execution,inputs,context);
        mark("inkling_ffn_input");
        if (physical%residual_block==0) depth.blocks.push_back(hidden);
        else current=current+hidden;
        auto ffn=depth_mix(depth.blocks,current,w.ffn_query).squeeze(2);
        if (repeated) ffn=control(execution,ffn,messages.select(2,3),3);
        auto normalized=rms_norm_cuda(ffn.contiguous(),w.ffn_norm,eps);
        mark("inkling_experts");
        const DecodeWindow::OperatorSiteScope site(recording,this);
        auto routed=w.ffn.forward(execution,normalized).reshape(current.sizes());
        mark("inkling_output");
        auto output=current+conv(std::move(routed),3);
        return output_query.defined()?depth_mix(depth.blocks,output,output_query).squeeze(2):output;
    }

    Tensor forward(CudaExecutionContext&,Tensor,Tensor,int64_t,const MfqOptional<Tensor>&,const RopeCache&,
        const MfqOptional<Tensor>&,const MfqOptional<Tensor>&) override {
        throw std::logic_error("Inkling requires context-aware residual depth state");
    }
};

}  // namespace

namespace inkling {
BatchCache take_cache(InklingCausalLm& model) {
    BatchCache result;
    result.layers.reserve(model.blocks.size());
    for (const auto& block:model.blocks) {
        auto& layer=static_cast<InklingBlock&>(*block);
        result.layers.push_back({std::exchange(layer.cache,KVCache{}),
            std::exchange(layer.history,std::array<Tensor,4>{})});
    }
    return result;
}

void bind_cache(InklingCausalLm& model,BatchCache&& cache,int64_t position,bool) {
    if (cache.layers.empty()) {
        if (position) throw std::logic_error("Inkling batch lost its prefill cache");
        model.reset(1);
        return;
    }
    if (cache.layers.size()!=model.blocks.size()) throw std::logic_error("Inkling batch cache topology changed");
    for (size_t i=0;i<cache.layers.size();++i) {
        auto& block=static_cast<InklingBlock&>(*model.blocks[i]);
        block.cache=std::move(cache.layers[i].attention);
        block.history=std::move(cache.layers[i].convolution);
    }
    model.cache_pos=position;
    model.decode_position_delta=0;
}

void reserve_slots(InklingCausalLm& model,BatchCache& cache,const BatchCache& prototype,
                   int64_t slots,int64_t positions,int64_t draft_depth,std::span<const int64_t> live_lengths) {
    cache.layers.resize(model.blocks.size());
    ResidentAllocationScope resident;
    for (size_t i=0;i<cache.layers.size();++i) {
        const auto& block=static_cast<const InklingBlock&>(*model.blocks[i]);
        const auto& example=prototype.layers.at(i);
        auto& layer=cache.layers[i];
        const auto prev_batch=layer.attention.batch_size();
        const auto previous=layer.attention.defined()?layer.attention.k.size(2):0;
        const auto wanted=block.sliding?std::min(block.window+draft_depth,block.context_limit):
            std::min(block.context_limit,(positions+1023)/1024*1024);
        const auto capacity=std::max(previous,wanted);
        if (prev_batch>=slots && previous>=capacity) continue;
        const auto rows=std::max(prev_batch,slots);
        KVCache replacement(rows,block.kv_heads,capacity,block.head_dim,block.sliding,example.attention.k.device());
        if (previous && block.sliding && previous!=capacity) {
            for (int64_t slot=0;slot<prev_batch;++slot) {
                const auto end=live_lengths[slot];
                if (!end) continue;
                auto pos=tb::arange(std::max(int64_t(0),end-block.window+1),end,
                    replacement.k.options().dtype(tb::kInt64));
                auto src=tb::remainder(pos,previous),dst=tb::remainder(pos,capacity);
                replacement.k.narrow(0,slot,1).index_copy_(2,dst,layer.attention.k.narrow(0,slot,1).index_select(2,src));
                replacement.v.narrow(0,slot,1).index_copy_(2,dst,layer.attention.v.narrow(0,slot,1).index_select(2,src));
            }
        } else if (previous) {
            replacement.k.narrow(0,0,prev_batch).narrow(2,0,previous).copy_(layer.attention.k);
            replacement.v.narrow(0,0,prev_batch).narrow(2,0,previous).copy_(layer.attention.v);
        }
        layer.attention=std::move(replacement);
        if (prev_batch>=rows) continue;
        for (size_t channel=0;channel<layer.convolution.size();++channel) {
            const auto& value=example.convolution[channel];
            auto next=tb::zeros({rows,value.size(1),value.size(2)},value.options());
            if (prev_batch) next.narrow(0,0,prev_batch).copy_(layer.convolution[channel]);
            layer.convolution[channel]=std::move(next);
        }
    }
}

void copy_to_slot(const InklingCausalLm& model,BatchCache& cache,const BatchCache& from,int64_t slot,int64_t length) {
    for (size_t i=0;i<cache.layers.size();++i) {
        const auto& block=static_cast<const InklingBlock&>(*model.blocks[i]);
        const auto& source=from.layers.at(i);
        auto& target=cache.layers[i];
        if (block.sliding) {
            const auto start=std::max(int64_t(0),length-block.window+1);
            auto positions=tb::arange(start,length,source.attention.k.options().dtype(tb::kInt64));
            const auto source_slots=tb::remainder(positions,source.attention.k.size(2));
            const auto target_slots=tb::remainder(positions,target.attention.k.size(2));
            target.attention.k.narrow(0,slot,1).index_copy_(2,target_slots,source.attention.k.index_select(2,source_slots));
            target.attention.v.narrow(0,slot,1).index_copy_(2,target_slots,source.attention.v.index_select(2,source_slots));
        } else {
            target.attention.k.narrow(0,slot,1).narrow(2,0,length).copy_(source.attention.k.narrow(2,0,length));
            target.attention.v.narrow(0,slot,1).narrow(2,0,length).copy_(source.attention.v.narrow(2,0,length));
        }
        for (size_t channel=0;channel<target.convolution.size();++channel)
            target.convolution[channel].narrow(0,slot,1).copy_(source.convolution[channel]);
    }
}

BatchCache save_convolution(const InklingCausalLm& model,const Tensor& slots) {
    BatchCache result;
    for (const auto& layer:model.blocks) {
        const auto& source=static_cast<const InklingBlock&>(*layer);
        auto& target=result.layers.emplace_back();
        for (size_t channel=0;channel<target.convolution.size();++channel)
            target.convolution[channel]=source.history[channel].index_select(0,slots);
    }
    return result;
}

void restore_convolution(InklingCausalLm& model,const BatchCache& cache,const Tensor& slots) {
    for (size_t i=0;i<cache.layers.size();++i) {
        auto& target=static_cast<InklingBlock&>(*model.blocks[i]);
        const auto& source=cache.layers[i];
        for (size_t channel=0;channel<target.history.size();++channel)
            target.history[channel].index_copy_(0,slots,source.convolution[channel]);
    }
}

void set_batch_verification(InklingCausalLm& model,bool enabled) {
    for (const auto& block:model.blocks) static_cast<InklingBlock&>(*block).batch_verifying=enabled;
}

BatchVerification take_batch_verification(InklingCausalLm& model) {
    BatchVerification result;
    result.reserve(model.blocks.size());
    for (const auto& block:model.blocks) {
        auto& layer=static_cast<InklingBlock&>(*block);
        if (!layer.verification) throw std::logic_error("Inkling batch lost its convolution history");
        result.push_back(std::move(*layer.verification));
        layer.verification.reset();
    }
    return result;
}

void retain_batch_prefix(InklingCausalLm& model,const BatchVerification& checkpoint,const Tensor& kept) {
    for (size_t i=0;i<model.blocks.size();++i) {
        auto& layer=static_cast<InklingBlock&>(*model.blocks[i]);
        const auto& saved=checkpoint.at(i);
        for (size_t j=0;j<layer.history.size();++j)
            retain_convolution_prefix(layer.history[j],saved.previous[j],saved.inputs[j],kept);
    }
}

} // namespace inkling

struct InklingTargetGraphs {
    struct Cell {
        Tensor ids,positions,lengths,hidden,raw,logits;
        std::vector<InklingBlock::Verification> journals;
        std::shared_ptr<GraphPoolLease> pool;
        std::unique_ptr<DecodeWindow> window;
    };
    int64_t planned=0;
    cudaStream_t stream=nullptr;
    std::vector<const void*> state;
    std::map<std::tuple<int64_t,int64_t,bool>,std::unique_ptr<Cell>> cells;
};

void clear_inkling_graphs(InklingCausalLm& model) {model.decode_graphs.reset();}

std::optional<Tensor> inkling_graph_hidden(InklingCausalLm& model,const Tensor& ids,Tensor* raw,
        int64_t confirmed,int64_t horizon,Tensor& logits,bool& initialized) {
    if (model.execution->profiler.enabled || !mfq_cuda_graph_capture_supported() || model.speculative_start>=0 ||
        ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)>1+mfq::engine::mtp::kMaximumBlockwiseDraftDepth || model.cache_pos<=0)
        return std::nullopt;
    const auto position=model.cache_pos,tokens=ids.size(1);
    const auto planned=decode_graph_bucket(std::max(horizon,position+tokens),model.max_position_embeddings());
    const auto stream=mfq_current_cuda_stream();
    auto& graphs=model.decode_graphs;
    if (graphs && (graphs->stream!=stream || graphs->planned!=planned)) graphs.reset();
    bool submitted=false;
    try {
        std::vector<const void*> addresses;
        Block::Context geometry;
        geometry.cache_position=position;
        geometry.planned_kv_length=planned;
        const auto ring_rows=1+mfq::engine::mtp::kMaximumBlockwiseDraftDepth;
        for (const auto& owned:model.blocks) {
            auto& layer=static_cast<InklingBlock&>(*owned);
            layer.prepare_kv(geometry,ids,ring_rows);
            addresses.push_back(layer.cache.k.data_ptr());
            addresses.push_back(layer.cache.v.data_ptr());
            for (const auto& history:layer.history) addresses.push_back(history.data_ptr());
        }
        if (!graphs || graphs->state!=addresses) {
            graphs=std::make_shared<InklingTargetGraphs>();
            graphs->state=std::move(addresses);
            graphs->stream=stream;
            graphs->planned=planned;
        }
        auto& entry=graphs->cells[{tokens,confirmed,raw!=nullptr}];
        if (!entry) {
            const auto context=default_context(ids.get_device());
            const auto reserve=runtime_options::workspace_reserve_bytes();
            const bool returned=has_routed_expert_cache(*model.execution) && context->available_memory()<reserve;
            if (returned) reserve_routed_capture_workspace(*model.execution,reserve);
            auto cell=std::make_unique<InklingTargetGraphs::Cell>();
            {
                ResidentAllocationScope resident;
                cell->ids=tb::empty_like(ids);
                cell->positions=tb::empty({tokens},ids.options());
                cell->lengths=tb::empty({1},ids.options());
            }
            cell->pool=GraphPoolLease::acquire(context,stream);
            cell->window=std::make_unique<DecodeWindow>(stream,runtime_options::layer_timings());
            cell->window->set_profile_geometry(raw?"inkling_verify":"inkling_decode",int(tokens));
            cell->ids.copy_(ids);
            cell->positions.copy_(tb::arange(position,position+tokens,ids.options()));
            cell->lengths.fill_(position+tokens);
            const auto slots=tb::zeros({1},ids.options());
            const auto state=inkling::save_convolution(model,slots);
            bool recorded=false;
            const auto restore=[&] {
                if (confirmed && model.speculative_start>=0) {
                    cell->journals.clear();
                    for (const auto& owned:model.blocks) {
                        auto& layer=static_cast<InklingBlock&>(*owned);
                        if (recorded && layer.verification) cell->journals.push_back(std::move(*layer.verification));
                        layer.verification.reset();
                    }
                }
                model.speculative_start=-1;
                model.speculative_confirmed=0;
                model.cache_pos=position;
                inkling::restore_convolution(model,state,slots);
            };
            cell->window->capture([&] {
                    cudaStreamCaptureStatus status;
                    TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stream,&status));
                    recorded=status==cudaStreamCaptureStatusActive;
                    cell->hidden=model.hidden_forward(cell->ids,cell->positions,cell->lengths,nullptr,cell->positions,
                        raw?&cell->raw:nullptr,confirmed,{planned,0});
                    cell->logits=model.logits_from_hidden(cell->hidden);
                },restore,[&]{cell->hidden={};cell->raw={};cell->logits={};});
            if (confirmed && cell->journals.size()!=model.blocks.size())
                throw std::logic_error("Inkling verification graph lost a recurrent history");
            if (returned) restore_routed_decode_workspace_after_capture(*model.execution);
            entry=std::move(cell);
            initialized=true;
        }
        auto& cell=*entry;
        cell.ids.copy_(ids);
        cell.positions.copy_(tb::arange(position,position+tokens,ids.options()));
        cell.lengths.fill_(position+tokens);
        submitted=true;
        if (!cell.window->enqueue_if_device_only()) cell.window->run();
        if (confirmed) {
            for (size_t i=0;i<model.blocks.size();++i) {
                auto& layer=static_cast<InklingBlock&>(*model.blocks[i]);
                layer.verification=cell.journals[i];
                layer.verification->start=position;
            }
            model.speculative_start=position;
            model.speculative_confirmed=confirmed;
        }
        model.cache_pos=position+tokens;
        if (raw) *raw=cell.raw;
        logits=cell.logits;
        return cell.hidden;
    } catch (const MfqResourceExhausted&) {
        graphs.reset();
        if (submitted) throw;
        return std::nullopt;
    }
}

void place_inkling_experts(InklingCausalLm& model,int64_t sequences) {
    std::vector<std::shared_ptr<GgufRoutedWeight>> banks;
    for (const auto& weights:model.weight_layers) {
        if (!weights || !weights->ffn.moe_gguf) return;
        banks.push_back(weights->ffn.moe_gguf->gate_up);
        banks.push_back(weights->ffn.moe_gguf->down);
    }
    std::size_t kv_bytes=0;
    for (const auto& owned:model.blocks) {
        const auto& block=static_cast<const InklingBlock&>(*owned);
        const auto rows=block.sliding?std::min(model.max_position_embeddings(),
            block.window+int64_t(mfq::engine::mtp::kMaximumBlockwiseDraftDepth)):model.max_position_embeddings();
        kv_bytes+=std::size_t(rows)*block.kv_heads*block.head_dim*2*sizeof(uint16_t);
    }
    (void)place_gguf_experts_resident(*model.execution,banks,kv_bytes*std::max<int64_t>(1,sequences));
}

void InklingModel::adapter_validate_load_options() const {
    if (execution->layer_placement.enabled() || execution->tensor_parallel.enabled() || execution->expert_parallel.enabled())
        throw std::invalid_argument("Inkling residual-depth execution currently requires one CUDA device");
}

void InklingModel::adapter_load_final_state(const mfq::ModelSource& source,Tensor& norm) {
    CausalResources::adapter_load_final_state(source,norm);
    embedding_norm=load_dense_gpu(*execution,source,"model.token_embedding_norm.weight");
    output_query=load_dense_gpu(*execution,source,"model.residual.output_query.weight");
}

std::unique_ptr<Block> InklingModel::adapter_load_block(const mfq::ModelSource& source,int layer,int device,
                                                       const std::string& type) {
    const auto physical=config.plan->physical_layer(layer);
    const bool sliding=type=="sliding_attention";
    if (!sliding && type!="full_attention") throw std::invalid_argument("unsupported Inkling attention layer");
    auto& cached=weight_layers[physical];
    const auto query_heads=config.query_heads[physical],kv_heads=config.kv_heads[physical];
    if (!cached) {
        auto weights=std::make_shared<InklingWeights>();
        auto& w=*weights;
        const auto root="model.block."+std::to_string(physical)+".";
        const auto attn=root+"attention.";
        const auto mlp=root+"mlp.";
        TransformerWeightLoader loader{*execution,source};
        w.query_projections=loader.projections({attn+"query.weight",attn+"relative.query.weight",
            attn+"sink_projection.weight",attn+"gate.weight"});
        w.k=loader.linear(attn+"key.weight");w.v=loader.linear(attn+"value.weight");
        if (sliding && w.query_projections.layers.size()==w.query_projections.outs.size()) {
            auto projections=w.query_projections.layers;
            projections.push_back(w.k);
            projections.push_back(w.v);
            w.sliding_qkv=make_quant_group(*execution,std::move(projections),false);
        }
        w.output=loader.linear(attn+"output.weight");
        w.q_norm=loader.dense(attn+(sliding?"norm.weight":"query_input_norm.weight"));
        w.k_norm=sliding?w.q_norm:loader.dense(attn+"key_input_norm.weight");
        w.v_norm=sliding?w.q_norm:loader.dense(attn+"value_input_norm.weight");
        if (!sliding) {
            w.key_taps=loader.linear(attn+"key_conv_projection.weight");
            w.value_taps=loader.linear(attn+"value_conv_projection.weight");
        }
        w.ffn_norm=loader.dense(mlp+"norm.weight");w.query_clip=loader.dense(attn+"query_clip.weight");
        w.relative_clip=loader.dense(attn+"relative.clip.weight");w.relative_bias=loader.dense(attn+"relative.bias.weight");
        w.convolution={loader.dense(attn+"key_conv.weight"),loader.dense(attn+"value_conv.weight"),
                       loader.dense(attn+"output_conv.weight"),loader.dense(mlp+"output_conv.weight")};
        if (physical) w.attention_query=loader.dense(attn+"residual_query.weight");
        w.ffn_query=loader.dense(mlp+"residual_query.weight");
        if (physical>=config.loop_start && physical<config.loop_end) {
            w.loop_query=loader.dense(root+"loop_msg_query.weight");
            w.unit=tb::ones({metadata.hidden_size},w.ffn_norm.options());
            const std::array<std::string,4> names{"q","k","v","ffn"};
            for (size_t i=0;i<names.size();++i) {
                const auto ctrl=root+"loop_"+names[i]+"_ctrl_";
                w.controls[i]={loader.linear(ctrl+"down.weight"),loader.projections({ctrl+"gain.weight",ctrl+"forget.weight"})};
            }
        }
        MoeWeightLoadOptions moe_options{int(physical)};
        moe_options.has_shared_expert=false;
        w.ffn=load_moe_weights(*execution,source,mlp,moe_options);
        w.ffn.moe_use_sigmoid=true;w.ffn.moe_normalize=true;w.ffn.moe_delayed_softmax=false;
        w.ffn.moe_router_scale=config.routed_scale;w.ffn.moe_top_k=int(config.top_k[physical]);
        w.ffn.situ_beta=config.situ_beta;w.ffn.situ_linear_beta=config.situ_linear_beta;
        if (!w.ffn.moe_gguf || !w.ffn.moe_gguf->gate_up ||
            w.ffn.moe_gguf->gate_up->out!=2*config.intermediate[physical] ||
            w.ffn.moe_gguf->gate_up->experts!=metadata.num_experts ||
            w.ffn.moe_gguf->gate_up->input!=metadata.hidden_size)
            throw std::invalid_argument("Inkling checkpoint has incompatible expert geometry");
        cached=std::move(weights);
    }
    auto result=std::make_unique<InklingBlock>();
    result->weights=cached;result->cuda_device=device;result->visit=layer;result->physical=physical;
    result->q_heads=query_heads;result->kv_heads=kv_heads;result->head_dim=metadata.head_dim;
    result->context_limit=metadata.max_position_embeddings;result->sliding=sliding;
    result->window=sliding?config.window:metadata.max_position_embeddings;
    result->loop_first=config.loop_start;result->loop_end=config.loop_end;result->residual_block=config.residual_block;
    result->relative_dim=config.relative_dim;result->relative_clip=config.relative_clip;result->qk_clip=config.qk_clip;
    result->log_floor=config.log_scale_floor;result->log_alpha=config.log_scale_alpha;
    result->taps=config.taps;result->eps=metadata.rms_norm_eps;
    result->repeated=layer>=config.loop_end && physical>=config.loop_start && physical<config.loop_end;
    if (layer+1==metadata.num_hidden_layers) result->output_query=output_query;
    return result;
}

Tensor InklingModel::adapter_embed(Tensor value) const {
    return rms_norm_cuda(value.contiguous(),embedding_norm,metadata.rms_norm_eps);
}
Tensor InklingModel::adapter_logits(const QuantLinear& head,Tensor hidden) const {
    return CausalResources::adapter_logits(head,std::move(hidden)).narrow(-1,0,metadata.vocab_size);
}
Tensor InklingModel::adapter_last_logits(const QuantLinear& head,Tensor hidden) const {
    return CausalResources::adapter_last_logits(head,std::move(hidden)).narrow(-1,0,metadata.vocab_size);
}
Tensor InklingModel::adapter_next_token(const QuantLinear& head,Tensor hidden) const {
    return sample_greedy_cuda(adapter_last_logits(head,std::move(hidden)).contiguous());
}

void InklingModel::adapter_begin_forward(bool capture) {
    draft_features.assign(capture?draft_taps.size():0,{});
    next_draft_tap=0;
    draft_visit=0;
}

void InklingModel::adapter_capture_hidden(const Tensor& hidden) {
    if (next_draft_tap>=draft_features.size()) return;
    if (++draft_visit==draft_taps[next_draft_tap]) draft_features[next_draft_tap++]=hidden;
}

Tensor InklingModel::adapter_raw_hidden(const Tensor& hidden,const Tensor& finalized) const {
    if (draft_features.empty()) return hidden;
    if (next_draft_tap!=draft_features.size()) throw std::logic_error("Inkling did not capture all draft features");
    auto features=draft_features;
    if (draft_taps.back()==metadata.num_hidden_layers) features.back()=finalized;
    return tb::cat(features,-1).contiguous();
}

TextSessionStateKind CudaSessionCodec<InklingModel>::kind(const Model& model) {
    return model.blocks.empty()?TextSessionStateKind::Unsupported:TextSessionStateKind::HybridAttention;
}

bool CudaSessionCodec<InklingModel>::supports_paged(const Model& model) {
    return kind(model)==TextSessionStateKind::HybridAttention;
}

TextSessionState CudaSessionCodec<InklingModel>::capture(const Model& model,const std::vector<int64_t>& tokens) {
    if (model.cache_pos<=0 || size_t(model.cache_pos)!=tokens.size())
        throw CudaSessionStateError("Inkling checkpoint does not match the prompt");
    TextSessionState result;
    result.cache_pos=model.cache_pos;
    result.tokens=tokens;
    auto& layers=result.payload.emplace<std::vector<HybridBlockSessionState>>();
    layers.reserve(model.blocks.size());
    for (const auto& block:model.blocks) {
        const auto& layer=static_cast<const InklingBlock&>(*block);
        if (!layer.cache.defined() || layer.cache.batch_size()!=1)
            throw CudaSessionStateError("Inkling checkpoint has no single-request cache");
        HybridBlockSessionState saved;
        saved.kind=HybridBlockSessionStateKind::AttentionConvolution;
        if (layer.sliding) {
            auto keys=capture_session_ring_tensor(layer.cache.k,2,model.cache_pos,layer.window);
            auto values=capture_session_ring_tensor(layer.cache.v,2,model.cache_pos,layer.window);
            const auto capacity=keys.size(2);
            saved.full_attention={std::move(keys),std::move(values),capacity,true};
        } else {
            const auto length=model.cache_pos;
            saved.full_attention={layer.cache.k.narrow(2,0,length).clone(),
                layer.cache.v.narrow(2,0,length).clone(),layer.cache.k.size(2),false};
        }
        saved.convolution_state=tb::cat(std::vector<Tensor>(layer.history.begin(),layer.history.end()),2);
        result.bytes+=session_tensor_bytes(saved.full_attention.k)+session_tensor_bytes(saved.full_attention.v)+
            session_tensor_bytes(saved.convolution_state);
        layers.push_back(std::move(saved));
    }
    return result;
}

void CudaSessionCodec<InklingModel>::restore(Model& model,const TextSessionState& state) {
    const auto* layers=std::get_if<std::vector<HybridBlockSessionState>>(&state.payload);
    if (!layers || layers->size()!=model.blocks.size() || state.cache_pos<=0 ||
        size_t(state.cache_pos)!=state.tokens.size())
        throw CudaSessionStateError("Inkling checkpoint topology changed");
    for (size_t i=0;i<layers->size();++i) {
        const auto& saved=layers->at(i);
        const auto& kv=saved.full_attention;
        const auto& conv=saved.convolution_state;
        auto& layer=static_cast<InklingBlock&>(*model.blocks[i]);
        const auto kv_width=layer.kv_heads*layer.head_dim;
        const auto hidden=model.hidden_size();
        if (saved.kind!=HybridBlockSessionStateKind::AttentionConvolution || kv.ring!=layer.sliding ||
            !session_kv_geometry_matches(kv.k,kv.v) || kv.k.size(0)!=1 || kv.k.size(1)!=layer.kv_heads ||
            kv.k.size(3)!=layer.head_dim || kv.k.size(2)!=(kv.ring?kv.capacity:state.cache_pos) ||
            kv.k.scalar_type()!=tb::kFloat16 || kv.k.get_device()!=layer.cuda_device ||
            !conv.defined() || conv.sizes().vec()!=std::vector<int64_t>{1,layer.taps-1,2*(kv_width+hidden)} ||
            conv.scalar_type()!=tb::kFloat32 || conv.get_device()!=layer.cuda_device)
            throw CudaSessionStateError("Inkling checkpoint layer geometry changed");
        restore_session_prefix_tensor(layer.cache.k,kv.k,2,kv.capacity);
        restore_session_prefix_tensor(layer.cache.v,kv.v,2,kv.capacity);
        layer.cache.ring=kv.ring;
        const std::array<int64_t,4> widths{kv_width,kv_width,hidden,hidden};
        int64_t offset=0;
        for (size_t j=0;j<widths.size();++j) {
            restore_session_tensor(layer.history[j],conv.narrow(2,offset,widths[j]).contiguous());
            offset+=widths[j];
        }
    }
    model.cache_pos=state.cache_pos;
}

}  // namespace mfq::cuda

#include "ops.h"

#include "core/ffn.h"
#include "core/mtp.h"
#include "mfq_cuda_attention_ops.h"
#include "mfq_cuda_norm_ops.h"
#include "models/hy4_flash/dfly2_config.h"
#include "runtime/decode_window.h"
#include "runtime/graph_pool_lease.h"
#include "storage/transformer_loader.h"
#include "storage/weight_loader.h"
#include "tyloinfer/hf_model_source.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace mfq::cuda {
namespace {

namespace tb=mfq_tensor_backend;
using Tensor=tb::Tensor;
using Dfly2Config=mfq::models::hy4_flash::Dfly2Config;
using Dfly2LayerConfig=mfq::models::hy4_flash::Dfly2LayerConfig;
using DraftHistory=std::pair<Tensor,Tensor>;

Tensor project(CudaExecutionContext& execution,const QuantLinear& weight,Tensor input,const char* name) {
    if (input.size(-1)!=weight.neuron_len())
        throw std::invalid_argument(std::string(name)+" expects width "+std::to_string(weight.neuron_len())+
            ", got "+std::to_string(input.size(-1)));
    if (weight.is_dense() && input.numel()>input.size(-1))
        return tb::matmul(input.to(weight.dense.scalar_type()),weight.dense.transpose(0,1)).to(input.scalar_type());
    return weight.forward(execution,std::move(input));
}

struct DraftLayer {
    Dfly2LayerConfig config;
    QuantLinear query_a,query_b,kv_a,gate,output;
    Tensor input_norm,query_norm,kv_norm,output_norm,ffn_norm,key_b,value_b,sinks;
    FFN ffn;

    Tensor forward(Hy4CausalLm& target,Tensor hidden,const Tensor& positions,const DraftHistory& selected_history) const {
        const auto& c=target.config.metadata;
        const auto& geometry=target.config;
        auto& execution=*target.execution;
        const auto batch=hidden.size(0),heads=config.heads,tokens=hidden.size(1),nope=c.head_dim-c.rotary_dim;
        const auto& rope_cache=hy4_full_rope(target);
        const auto rotate=[&](const Tensor& x) {
            return dsv4_tail_rope_cuda(x.transpose(1,2),positions,rope_cache.cos,rope_cache.sin,c.rotary_dim);
        };
        auto input=rms_norm_cuda(hidden.contiguous(),input_norm,c.rms_norm_eps);
        auto low_query=rms_norm_cuda(project(execution,query_a,input,"query A").contiguous(),query_norm,c.rms_norm_eps);
        auto query=project(execution,query_b,low_query,"query B").to(tb::kFloat32).reshape({batch,tokens,heads,c.head_dim});
        auto q_nope=query.narrow(-1,0,nope).transpose(1,2).to(key_b.scalar_type());
        if (q_nope.size(-1)!=key_b.size(1))
            throw std::invalid_argument("DFly2 MLA absorbed query width differs from the key projection");
        auto q=tb::cat({tb::matmul(q_nope,key_b).to(tb::kFloat32),
            rotate(query.narrow(-1,nope,c.rotary_dim).contiguous())},-1).contiguous();
        auto packed=project(execution,kv_a,input,"KV A").to(tb::kFloat32);
        auto key=rms_norm_cuda(packed.narrow(-1,0,geometry.kv_rank).contiguous(),kv_norm,c.rms_norm_eps);
        auto rope=rotate(packed.narrow(-1,geometry.kv_rank,c.rotary_dim).contiguous().unsqueeze(2)).squeeze(1);
        auto suffix=tb::cat({key,rope},-1).to(tb::kFloat16);

        const auto& [history,indices]=selected_history;
        auto local=tb::arange(tokens,indices.options());
        auto causal=tb::where(local.unsqueeze(1)>=local.unsqueeze(0),local+history.size(1),-1)
            .unsqueeze(0).expand({batch,tokens,tokens});
        auto selected=tb::cat({indices.unsqueeze(1).expand({batch,tokens,indices.size(1)}),causal},2).contiguous();
        auto kv=tb::cat({history,suffix},1).contiguous();
        auto attended=attention_mla_latent_selected_cuda(q,kv,selected,sinks,1/std::sqrt(double(c.head_dim)));
        auto expanded=tb::matmul(attended.transpose(1,2).to(value_b.scalar_type()),value_b.transpose(1,2))
            .transpose(1,2).reshape({batch,tokens,heads*geometry.value_head_dim});
        auto projected=project(execution,output,expanded*tb::sigmoid(project(execution,gate,input,"attention gate")),
            "attention output");
        if (output_norm.defined()) projected=rms_norm_cuda(projected.contiguous(),output_norm,c.rms_norm_eps);
        hidden=hidden+projected.to(hidden.scalar_type());
        auto feed=ffn.forward(execution,rms_norm_cuda(hidden.contiguous(),ffn_norm,c.rms_norm_eps));
        return hidden+feed.to(hidden.scalar_type()).reshape(hidden.sizes());
    }
};

struct Dfly2Weights {
    Dfly2Config config;
    std::vector<DraftLayer> layers;
    QuantLinear fuse;
    FFN correction;
    Tensor final_norm,anchor_norm,anchor_hidden_norm,fusion_gate,correction_hidden_norm,correction_embedding_norm;
    double epsilon=0;
};

class Dfly2 final : public MtpModule {
    std::shared_ptr<const Dfly2Weights> weights_;
public:
    explicit Dfly2(std::shared_ptr<const Dfly2Weights> weights):weights_(std::move(weights)) {}
    std::unique_ptr<MtpModule> fork_session() const override {return std::make_unique<Dfly2>(weights_);}

    void reset(int64_t batch) override {
        if (batch!=1) throw std::invalid_argument("DFly2 requires a single active sequence");
        last_hidden_={};
        active_target_.reset();
        position_=0;
    }
    Tensor forward(const MtpTarget&,Tensor,Tensor) override {
        throw std::logic_error("DFly2 requires block drafting");
    }
    int64_t cache_position() const noexcept override {return position_;}
    void trim_cache_to(int64_t position) override {
        if (position!=position_) throw std::logic_error("DFly2 retains only the committed target boundary");
    }
    bool teacher_forced_prompt_prime() const noexcept override {return false;}
    bool target_bootstrap_decode() const noexcept override {return false;}
    bool preserve_output_dtype() const noexcept override {return true;}
    bool blockwise_drafting() const noexcept override {return true;}
    bool retains_partial_target_prefix() const noexcept override {return true;}
    bool take_graph_initialization() noexcept override {return std::exchange(graph_initialized_,false);}
    uint64_t graph_replay_count() const noexcept override {return graph_replays_;}
    void prepare_request(int64_t end,int maximum_depth) override {
        context_horizon_=end;
        maximum_request_depth_=std::clamp(maximum_depth,1,maximum_draft_depth());
    }
    int64_t prompt_context_window() const noexcept override {return 1;}
    int maximum_draft_depth() const noexcept override {return mfq::engine::mtp::kMaximumBlockwiseDraftDepth;}

    void seed_target_context(Tensor hidden,int64_t tokens) override {
        if (position_!=0 || !hidden.defined() || hidden.dim()!=3 || tokens<hidden.size(1))
            throw std::invalid_argument("invalid DFly2 prompt context");
        append_at(std::move(hidden),tokens);
    }
    void append_target_context(Tensor hidden,int64_t start) override {
        if (start!=position_ || !hidden.defined() || hidden.dim()!=3)
            throw std::invalid_argument("DFly2 target context has a discontinuity");
        const auto end=start+hidden.size(1);
        append_at(std::move(hidden),end);
    }
    bool supports_session_state() const noexcept override {return true;}
    MtpSessionState capture_session_state(int64_t position,const Tensor& hidden) const override {
        if (position!=position_ || position<1 || !valid_hidden(hidden) || hidden.size(1)!=1)
            throw std::invalid_argument("DFly2 session is outside its committed context");
        MtpSessionState result;
        result.cache_pos=position;
        result.last_target_hidden=hidden.clone();
        result.bytes=result.last_target_hidden.nbytes();
        return result;
    }
    void restore_session_state(const MtpSessionState& state) override {
        if (state.cache_pos<1 || !valid_hidden(state.last_target_hidden) || state.last_target_hidden.size(1)!=1)
            throw std::invalid_argument("DFly2 session has incompatible target hidden state");
        reset(1);
        append_at(state.last_target_hidden,state.cache_pos);
    }

    Tensor draft_block(const MtpTarget& target,Tensor anchor,int requested) override try {
        auto* language=std::get<Hy4CausalLm*>(target.model);
        if (requested<1 || requested>maximum_draft_depth() || !last_hidden_.defined() ||
            (target.batch_row<0 && position_!=language->cache_pos) || position_+requested>=language->max_position_embeddings())
            throw std::invalid_argument("DFly2 cannot draft at this target position");
        active_target_=target;
        const auto* setting=std::getenv("TYLOINFER_DFLASH2_GRAPH");
        const bool graphed=(!setting || setting[0]!='0') && mfq_cuda_graph_capture_supported();
        const auto stream=mfq_current_cuda_stream();
        const auto capacity=graphed?graph_capacity(*language,target.batch_row):0;
        int rows=requested;
        if (graphed) for (int depth=requested;depth<=maximum_request_depth_;++depth) {
            const auto& graph=graphs_[depth];
            if (graph && graph->window->valid() && graph->capacity==capacity && graph->stream==stream &&
                position_+depth<language->max_position_embeddings()) {
                rows=depth;
                break;
            }
        }
        auto positions=tb::arange(position_,position_+rows+1,anchor.options().dtype(tb::kInt64));
        auto mask=tb::full({1,rows},weights_->config.mask_token_id,anchor.options());
        auto embeddings=target.embed(tb::cat({anchor,mask},1)).to(tb::kBFloat16);
        if (graphed) return graph_block(*language,embeddings,positions,rows,capacity,target.batch_row).narrow(1,0,requested);
        return block_hidden(*language,embeddings,last_hidden_,positions,nullptr,target.batch_row);
    } catch (const std::exception& error) {
        throw std::runtime_error("DFly2 draft block: "+std::string(error.what()));
    }

    std::vector<Tensor> draft_blocks(const MtpTarget& target,std::span<const DraftLane> lanes) const override {
        const auto* setting=std::getenv("TYLOINFER_DFLASH2_GRAPH");
        if (lanes.size()<2 || !mfq_cuda_graph_capture_supported() || (setting && setting[0]=='0'))
            return MtpModule::draft_blocks(target,lanes);
        auto& model=*std::get<Hy4CausalLm*>(target.model);
        const auto batch=int64_t(lanes.size()),requested=int64_t(lanes.front().depth)+1;
        auto rows=requested;
        if (batch_graph_ && batch_graph_->embeddings.size(0)==batch) {
            const auto previous=batch_graph_->embeddings.size(1);
            if (previous>rows && std::all_of(lanes.begin(),lanes.end(),[&](const auto& lane) {
                return static_cast<const Dfly2&>(*lane.session).position_+previous<=model.max_position_embeddings();
            })) rows=previous;
        }
        const auto stream=mfq_current_cuda_stream();
        int64_t capacity=0;
        std::vector<Tensor> anchors,hidden;
        std::vector<int64_t> offsets;
        anchors.reserve(batch);hidden.reserve(batch);offsets.reserve(batch*rows);
        for (const auto& lane:lanes) {
            auto& session=static_cast<Dfly2&>(*lane.session);
            if (lane.depth+1!=requested || !session.last_hidden_.defined() ||
                session.position_+rows>model.max_position_embeddings())
                return MtpModule::draft_blocks(target,lanes);
            anchors.push_back(lane.anchor);
            hidden.push_back(session.last_hidden_);
            for (int64_t i=0;i<rows;++i) offsets.push_back(session.position_+i);
            capacity=std::max(capacity,session.graph_capacity(model,lane.slot));
        }
        const auto options=anchors.front().options();
        auto ids=tb::cat({tb::cat(anchors,0),tb::full({batch,rows-1},weights_->config.mask_token_id,options)},1);
        auto embeddings=target.embed(std::move(ids)).to(tb::kBFloat16);
        auto anchor_states=tb::cat(hidden,0);
        auto positions=tb::tensor(offsets,options.dtype(tb::kInt64)).reshape({batch,rows});
        auto& graph=prepare_graph(batch_graph_,model,embeddings,anchor_states,positions,capacity);
        for (size_t i=0;i<weights_->layers.size();++i) {
            auto& [fixed_rows,fixed_indices]=graph.history[i];
            fixed_indices.fill_(-1);
            for (int64_t row=0;row<batch;++row) {
                auto& session=static_cast<Dfly2&>(*lanes[row].session);
                const auto [values,indices]=session.history_for(model,i,lanes[row].slot);
                fixed_rows.narrow(0,row,1).narrow(1,0,values.size(1)).copy_(values);
                fixed_indices.narrow(0,row,1).narrow(1,0,indices.size(1)).copy_(indices);
            }
        }
        const bool initialized=!graph.window->valid();
        replay_graph(graph,model,-1);
        std::vector<Tensor> result;
        result.reserve(batch);
        auto binding=target;
        for (int64_t row=0;row<batch;++row) {
            auto& session=static_cast<Dfly2&>(*lanes[row].session);
            binding.batch_row=lanes[row].slot;
            session.active_target_=std::as_const(binding);
            session.graph_initialized_|=initialized;
            ++session.graph_replays_;
            result.push_back(graph.output.narrow(0,row,1).narrow(1,0,requested-1));
        }
        return result;
    }
    Tensor draft_next(Tensor hidden,Tensor previous) override {
        auto projected=draft_next_hidden(std::move(hidden),std::move(previous));
        return active_target_->logits(std::move(projected));
    }

    Tensor draft_next_hidden(Tensor hidden,Tensor previous) override try {
        if (!active_target_) throw std::logic_error("DFly2 draft has no target");
        auto& target=*active_target_;
        if (weights_->config.hidden_correction) {
            auto& execution=*std::get<Hy4CausalLm*>(target.model)->execution;
            auto embedding=target.embed(previous).to(hidden.scalar_type());
            auto projected=tb::cat({rms_norm_cuda(hidden,weights_->correction_hidden_norm,weights_->epsilon),
                rms_norm_cuda(embedding,weights_->correction_embedding_norm,weights_->epsilon)},-1);
            auto delta=weights_->correction.forward(execution,projected).reshape(hidden.sizes());
            hidden=hidden+delta.to(hidden.scalar_type());
        }
        return hidden;
    } catch (const std::exception& error) {
        throw std::runtime_error("DFly2 draft logits: "+std::string(error.what()));
    }
private:
    struct Graph {
        int64_t capacity=0;
        cudaStream_t stream=nullptr;
        Tensor embeddings,hidden,positions,output;
        std::vector<DraftHistory> history;
        std::shared_ptr<GraphPoolLease> pool;
        std::unique_ptr<DecodeWindow> window;
    };

    DraftHistory history_for(const Hy4CausalLm& model,size_t layer,int64_t slot) const {
        const auto& config_layer=weights_->layers[layer].config;
        return hy4_draft_history(model,config_layer.target_kv_layer,config_layer.target_index_layer,
            position_,int(weights_->config.recent_window),slot);
    }

    Tensor block_hidden(Hy4CausalLm& model,Tensor embeddings,const Tensor& hidden,const Tensor& positions,
                        const std::vector<DraftHistory>* prepared,int64_t slot) const {
        if (weights_->config.fusion=="slot0") {
            auto anchor=embeddings.narrow(1,0,1).contiguous();
            auto normalized=tb::cat({rms_norm_cuda(anchor,weights_->anchor_norm,weights_->epsilon),
                rms_norm_cuda(hidden,weights_->anchor_hidden_norm,weights_->epsilon)},-1);
            auto delta=weights_->fuse.forward(*model.execution,normalized);
            embeddings=tb::cat({(anchor+weights_->fusion_gate*delta).to(tb::kBFloat16),
                embeddings.narrow(1,1,embeddings.size(1)-1)},1);
        }
        for (size_t i=0;i<weights_->layers.size();++i) {
            auto& layer=weights_->layers[i];
            try {
                const auto history=prepared?prepared->at(i):history_for(model,i,slot);
                embeddings=layer.forward(model,std::move(embeddings),positions,history);
            } catch (const std::exception& error) {
                throw std::runtime_error("target KV layer "+std::to_string(layer.config.target_kv_layer)+": "+error.what());
            }
        }
        return rms_norm_cuda(embeddings.narrow(1,1,embeddings.size(1)-1).contiguous(),weights_->final_norm,weights_->epsilon);
    }

    int64_t graph_capacity(const Hy4CausalLm& model,int64_t slot) const {
        auto selected=std::min(weights_->config.target_top_k,context_horizon_);
        for (const auto& layer:weights_->layers) selected=std::max(selected,
            hy4_draft_selection_width(model,layer.config.target_index_layer,slot));
        return int64_t(std::bit_ceil(uint64_t(selected+weights_->config.recent_window)));
    }

    Graph& prepare_graph(std::unique_ptr<Graph>& item,Hy4CausalLm& model,const Tensor& embeddings,
                         const Tensor& hidden,const Tensor& positions,int64_t capacity) const {
        const auto stream=mfq_current_cuda_stream();
        if (!item || item->capacity!=capacity || item->stream!=stream || item->embeddings.sizes()!=embeddings.sizes()) {
            ResidentAllocationScope resident;
            item.reset();
            auto graph=std::make_unique<Graph>();
            graph->capacity=capacity;
            graph->stream=stream;
            graph->embeddings=tb::empty(embeddings.sizes(),embeddings.options());
            graph->hidden=tb::empty_like(hidden);
            graph->positions=tb::empty(positions.sizes(),positions.options());
            const auto batch=embeddings.size(0),width=model.config.kv_rank+model.metadata.rotary_dim;
            for (size_t i=0;i<weights_->layers.size();++i)
                graph->history.emplace_back(
                    tb::zeros({batch,capacity,width},embeddings.options().dtype(tb::kFloat16)),
                    tb::empty({batch,capacity},positions.options().dtype(tb::kInt32)));
            graph->pool=GraphPoolLease::acquire(default_context(embeddings.get_device()),stream);
            graph->window=std::make_unique<DecodeWindow>(stream);
            item=std::move(graph);
        }
        auto& graph=*item;
        graph.embeddings.copy_(embeddings);
        graph.hidden.copy_(hidden);
        graph.positions.copy_(positions);
        return graph;
    }

    Tensor graph_block(Hy4CausalLm& model,const Tensor& embeddings,const Tensor& positions,int requested,int64_t capacity,int64_t slot) {
        auto& graph=prepare_graph(graphs_[requested],model,embeddings,last_hidden_,positions,capacity);
        for (size_t i=0;i<weights_->layers.size();++i) {
            const auto [rows,indices]=history_for(model,i,slot);
            auto& [fixed_rows,fixed_indices]=graph.history[i];
            fixed_rows.narrow(1,0,rows.size(1)).copy_(rows);
            fixed_indices.fill_(-1);
            fixed_indices.narrow(1,0,indices.size(1)).copy_(indices);
        }
        graph_initialized_|=!graph.window->valid();
        replay_graph(graph,model,slot);
        ++graph_replays_;
        return graph.output;
    }

    void replay_graph(Graph& graph,Hy4CausalLm& model,int64_t slot) const {
        if (!graph.window->valid()) {
            graph.window->capture([&] {
                graph.output=block_hidden(model,graph.embeddings,graph.hidden,graph.positions,&graph.history,slot);
            },[]{},[&]{graph.output={};});
        }
        if (!graph.window->enqueue_if_device_only()) graph.window->run();
    }

    bool valid_hidden(const Tensor& hidden) const {
        return hidden.defined() && hidden.dim()==3 && hidden.size(0)==1 &&
            hidden.size(1)>0 && hidden.size(2)==weights_->config.hidden_size;
    }
    void append_at(Tensor hidden,int64_t end) {
        if (!valid_hidden(hidden)) throw std::invalid_argument("DFly2 target hidden shape changed");
        last_hidden_=hidden.narrow(1,hidden.size(1)-1,1).to(tb::kBFloat16).clone();
        position_=end;
    }
    Tensor last_hidden_;
    int64_t position_=0;
    std::optional<MtpTarget> active_target_;
    bool graph_initialized_=false;
    uint64_t graph_replays_=0;
    int64_t context_horizon_=0;
    int maximum_request_depth_=1;
    mutable std::unique_ptr<Graph> batch_graph_;
    std::array<std::unique_ptr<Graph>,mfq::engine::mtp::kMaximumBlockwiseDraftDepth+1> graphs_;
};

FFN draft_ffn(const TransformerWeightLoader& loader,const std::string& prefix) {
    FFN weights;
    weights.down=loader.linear(prefix+"down_proj.weight");
    weights.gate_up=loader.gate_up({prefix+"gate_proj.weight",prefix+"up_proj.weight"},weights.down);
    return weights;
}

}  // namespace

std::unique_ptr<::MtpModule> load_hy4_dfly2(Hy4CausalLm& model,const std::filesystem::path& path) {
    const std::string prefix="model.dfly_layers.";
    mfq::HfModelSource source(std::filesystem::is_regular_file(path)?path.parent_path():path,prefix);
    auto draft=std::make_shared<Dfly2Weights>();
    draft->config=Dfly2Config::from_sources(*model.source,source);
    const auto& config=draft->config;
    if ((config.fusion!="none" && config.fusion!="slot0") || config.recent_window>std::numeric_limits<int>::max())
        throw std::invalid_argument("unsupported Hy4 DFly2 fusion or local window");
    const auto& target=model.config;
    draft->epsilon=target.metadata.rms_norm_eps;
    auto& execution=*model.execution;
    TransformerWeightLoader loader{execution,source};
    draft->final_norm=loader.dense(prefix+"final_norm.weight");
    if (config.fusion=="slot0") {
        const auto base=prefix+"mtp_fusion.";
        draft->anchor_norm=loader.dense(base+"enorm.weight");
        draft->anchor_hidden_norm=loader.dense(base+"hnorm.weight");
        draft->fusion_gate=loader.dense(base+"gate");
        draft->fuse=loader.linear(base+"eh_proj.weight");
    }
    if (config.hidden_correction) {
        const auto base=prefix+"hidden_correction.";
        draft->correction_hidden_norm=loader.dense(base+"hidden_norm.weight");
        draft->correction_embedding_norm=loader.dense(base+"embed_norm.weight");
        draft->correction=draft_ffn(loader,base);
    }
    std::vector<int64_t> index_layers;
    const auto nope=target.metadata.head_dim-target.metadata.rotary_dim;
    for (size_t i=0;i<config.layers.size();++i) {
        const auto root=prefix+std::to_string(i)+".",attn=root+"self_attn.";
        auto& layer=draft->layers.emplace_back();
        layer.config=config.layers[i];
        layer.input_norm=loader.dense(root+"input_layernorm.weight");
        layer.ffn_norm=loader.dense(root+"post_attention_layernorm.weight");
        layer.query_norm=loader.dense(attn+"q_a_layernorm.weight");
        layer.kv_norm=loader.dense(attn+"kv_a_layernorm.weight");
        layer.sinks=loader.dense(attn+"learnable_sink_param");
        if (loader.has(attn+"post_proj_norm.weight"))
            layer.output_norm=loader.dense(attn+"post_proj_norm.weight");
        layer.query_a=loader.linear(attn+"q_a_proj.weight");
        layer.query_b=loader.linear(attn+"q_b_proj.weight");
        layer.kv_a=loader.linear(attn+"kv_a_proj_with_mqa.weight");
        layer.gate=loader.linear(attn+"linear_gate.weight");
        layer.output=loader.linear(attn+"o_proj.weight");
        auto expanded=load_dense_native_gpu(execution,source,attn+"kv_b_proj.weight")
            .reshape({layer.config.heads,nope+target.value_head_dim,target.kv_rank});
        layer.key_b=expanded.narrow(1,0,nope).contiguous();
        layer.value_b=expanded.narrow(1,nope,target.value_head_dim).contiguous();
        layer.ffn=draft_ffn(loader,root+"mlp.");
        index_layers.push_back(layer.config.target_index_layer);
    }
    enable_hy4_draft_history(model,index_layers);
    return std::make_unique<Dfly2>(std::move(draft));
}

}  // namespace mfq::cuda

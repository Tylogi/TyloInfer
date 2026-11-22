#include "ops.h"

#include "core/ffn.h"
#include "core/mtp.h"
#include "kernels/dflash2.h"
#include "mfq_cuda_norm_ops.h"
#include "runtime/decode_window.h"
#include "runtime/graph_pool_lease.h"
#include "storage/transformer_loader.h"
#include "tyloinfer/gguf_model_source.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <nlohmann/json.hpp>

namespace mfq::cuda {
namespace {
namespace tb=mfq_tensor_backend;
using Tensor=tb::Tensor;

struct DsConfig {
    int hidden,heads,kv_heads,width,window,rank,block,maximum;
    float epsilon,theta;
    std::vector<int64_t> taps;
    std::vector<bool> sliding;

    DsConfig(const mfq::GgufModelSource& source,const InklingCausalLm& target) {
        const auto metadata=nlohmann::json::parse(source.metadata().at("gguf_metadata"));
        const auto read=[&](const char* name)->const nlohmann::json& {return metadata.at(std::string("dflash.")+name);};
        hidden=read("embedding_length"); heads=read("attention.head_count"); kv_heads=read("attention.head_count_kv");
        width=read("attention.key_length"); window=read("attention.sliding_window"); block=read("block_size");
        rank=read("dspark.markov_rank"); epsilon=read("attention.layer_norm_rms_epsilon"); theta=read("rope.freq_base");
        taps=read("target_layers").get<std::vector<int64_t>>();
        sliding=read("attention.sliding_window_pattern").get<std::vector<bool>>();
        const auto vocabulary=read("vocab_size").get<int64_t>();
        const auto layers=read("block_count").get<int64_t>();
        const auto context=std::min(target.max_position_embeddings(),read("context_length").get<int64_t>());
        if (source.architecture()!="dflash" || hidden!=target.hidden_size() || hidden<=0 || width<=0 || width>512 || width%2 ||
            heads<1 || heads>256 || kv_heads<1 || heads%kv_heads || read("attention.value_length")!=width ||
            read("rope.dimension_count")!=width || !std::isfinite(epsilon) || epsilon<=0 || !std::isfinite(theta) || theta<=0 ||
            block<2 || block>1+mfq::engine::mtp::kMaximumBlockwiseDraftDepth || window<1 || window>=INT_MAX ||
            rank<=0 || rank>hidden || context<2 || context>INT_MAX-1024 ||
            vocabulary!=target.lm_head.out() || read("vocab_size_unpadded")!=target.vocab_size() ||
            read("final_logit_softcapping")!=target.metadata.final_logit_softcapping ||
            taps.empty() || taps.front()<1 || taps.back()>target.num_hidden_layers() ||
            !std::is_sorted(taps.begin(),taps.end()) || std::adjacent_find(taps.begin(),taps.end())!=taps.end() ||
            layers<1 || size_t(layers)!=sliding.size() || read("has_confidence_head")!=false || read("sample_from_anchor")!=false)
            throw std::invalid_argument("Inkling DSpark checkpoint does not match the target geometry");
        maximum=int(context);
    }
};

struct Layer {
    Tensor feature_mix,feature_norm,attn_norm,q_norm,k_norm,context_k_norm,ffn_norm;
    QuantLinearGroup qkv,context_kv;
    QuantLinear output;
    FFN ffn;
    bool sliding=false;
};

struct Weights {
    explicit Weights(DsConfig dimensions):config(std::move(dimensions)) {}
    DsConfig config;
    std::vector<Layer> layers;
    Tensor tail_mix,tail_norm,mask,final_norm,joint_bias,correction_bias;
    QuantLinear markov,joint,correction;
};

class InklingDspark final : public MtpModule {
    std::shared_ptr<const Weights> weights_;
    const DsConfig& config;
    CudaExecutionContext& execution;
public:
    InklingDspark(std::shared_ptr<const Weights> weights,CudaExecutionContext& owner)
        :weights_(std::move(weights)),config(weights_->config),execution(owner) {}

    std::unique_ptr<MtpModule> fork_session() const override {
        return std::make_unique<InklingDspark>(weights_,execution);
    }

    void reset(int64_t batch) override {
        if (batch!=1) throw std::invalid_argument("Inkling DSpark requires one sequence per predictor state");
        position_=0;
        active_target_=nullptr;
    }
    int64_t cache_position() const noexcept override {return position_;}
    void trim_cache_to(int64_t position) override {
        if (position!=position_) throw std::logic_error("Inkling DSpark only stores committed target history");
    }
    bool teacher_forced_prompt_prime() const noexcept override {return false;}
    bool target_bootstrap_decode() const noexcept override {return false;}
    bool preserve_output_dtype() const noexcept override {return true;}
    bool blockwise_drafting() const noexcept override {return true;}
    bool project_blockwise_prompt_chunks() const noexcept override {return true;}
    bool dspark_policy() const noexcept override {return true;}
    bool retains_partial_target_prefix() const noexcept override {return true;}
    int preferred_context_update_batch_size() const noexcept override {return config.block;}
    bool take_graph_initialization() noexcept override {return std::exchange(graph_initialized_,false);}
    uint64_t graph_replay_count() const noexcept override {return replays_;}
    int maximum_draft_depth() const noexcept override {return config.block-1;}
    void prepare_request(int64_t,int depth) override {depth_=std::clamp(depth,1,maximum_draft_depth());}
    Tensor forward(const MtpTarget&,Tensor,Tensor) override {throw std::logic_error("Inkling DSpark drafts in blocks");}

    void append_target_context(Tensor features,int64_t start) override {
        if (start!=position_ || !valid_features(features) || features.size(1)>config.maximum-start)
            throw std::invalid_argument("Inkling DSpark target context is discontinuous");
        const auto length=features.size(1),end=start+length;
        reserve(end,features.options());
        const auto stream=mfq_current_cuda_stream();
        for (size_t i=0;i<weights_->layers.size();++i) {
            const auto& layer=weights_->layers[i]; auto& history=history_[i];
            const auto n=std::min(length,history.k.size(0));
            auto input=mixed(features.narrow(1,length-n,n),layer.feature_mix,layer.feature_norm);
            auto projections=layer.context_kv.forward(execution,input);
            auto at=tb::arange(end-n,end,features.options().dtype(tb::kInt64));
            auto table=rotary(at,features.options());
            auto key=projections[0].to(tb::kFloat32).contiguous();
            auto value=projections[1].to(tb::kFloat32).contiguous();
            dflash2::store_context(key.data_ptr<float>(),value.data_ptr<float>(),layer.context_k_norm.data_ptr<float>(),
                table.data_ptr<float>(),at.data_ptr<int64_t>(),history.k.data_ptr(),history.v.data_ptr(),
                int(n),config.kv_heads,config.width,int(history.k.size(0)),config.epsilon,stream,true);
        }
        auto last=mixed(features.narrow(1,length-1,1),weights_->tail_mix,weights_->tail_norm);
        last_features_.copy_(last);
        position_=end;
    }

    bool supports_session_state() const noexcept override {return true;}
    MtpSessionState capture_session_state(int64_t position,const Tensor& hidden) const override {
        if (position!=position_ || position<1 || !valid_features(hidden) || hidden.size(1)!=1)
            throw std::invalid_argument("Inkling DSpark checkpoint is outside its committed context");
        MtpSessionState result;
        result.cache_pos=position;
        result.last_target_hidden=hidden.clone();
        result.bytes=result.last_target_hidden.nbytes();
        for (size_t i=0;i<history_.size();++i) {
            const auto& history=history_[i];
            const auto rows=std::min(position,history.k.size(0));
            FullBlockSessionState saved{history.k.narrow(0,0,rows).unsqueeze(0).transpose(1,2).clone(),
                history.v.narrow(0,0,rows).unsqueeze(0).transpose(1,2).clone(),history.k.size(0),
                weights_->layers[i].sliding && position>=history.k.size(0)};
            result.bytes+=saved.k.nbytes()+saved.v.nbytes();
            result.blocks.push_back(std::move(saved));
        }
        return result;
    }

    void restore_session_state(const MtpSessionState& state) override {
        if (state.cache_pos<1 || state.cache_pos>config.maximum || state.blocks.size()!=weights_->layers.size() ||
            !valid_features(state.last_target_hidden) || state.last_target_hidden.size(1)!=1)
            throw std::invalid_argument("Inkling DSpark checkpoint geometry changed");
        reset(1);
        const auto device=weights_->mask.device();
        reserve(state.cache_pos,weights_->mask.options());
        for (size_t i=0;i<weights_->layers.size();++i) {
            const auto& saved=state.blocks[i]; auto& history=history_[i];
            const auto rows=std::min(state.cache_pos,history.k.size(0));
            const bool wrapped=weights_->layers[i].sliding && state.cache_pos>=history.k.size(0);
            if (!session_kv_geometry_matches(saved.k,saved.v) || saved.k.sizes().vec()!=
                std::vector<int64_t>{1,config.kv_heads,rows,config.width} || saved.k.scalar_type()!=tb::kBFloat16 ||
                saved.ring!=wrapped || (wrapped && saved.capacity!=rows))
                throw std::invalid_argument("Inkling DSpark checkpoint KV geometry changed");
            history.k.narrow(0,0,rows).copy_(saved.k.to(device).transpose(1,2).squeeze(0));
            history.v.narrow(0,0,rows).copy_(saved.v.to(device).transpose(1,2).squeeze(0));
        }
        last_features_.copy_(mixed(state.last_target_hidden.to(device),weights_->tail_mix,weights_->tail_norm));
        position_=state.cache_pos;
    }

    Tensor draft_block(const MtpTarget& bindings,Tensor anchor,int requested) override {
        auto* target=std::get<InklingCausalLm*>(bindings.model);
        if (requested<1 || requested>maximum_draft_depth() ||
            position_<=0 || position_+requested>=config.maximum)
            throw std::invalid_argument("Inkling DSpark draft position exceeds its context");
        active_target_=target;
        auto embedding=bindings.embed(std::move(anchor)).to(tb::kFloat32);
        auto at=tb::arange(position_,position_+requested+1,embedding.options().dtype(tb::kInt64));
        markov_state_=tb::zeros({1,1,config.rank},embedding.options());
        const auto* setting=std::getenv("TYLOINFER_DFLASH2_GRAPH");
        if (!mfq_cuda_graph_capture_supported() || (setting && setting[0]=='0')) return block(embedding,at);
        const auto rows=std::min<int64_t>(std::max(depth_,requested),config.maximum-position_-1);
        if (rows!=requested) at=tb::arange(position_,position_+rows+1,at.options());
        const auto stream=mfq_current_cuda_stream();
        auto& item=graphs_[int(rows)];
        if (!item || item->stream!=stream) {
            ResidentAllocationScope resident;
            auto graph=std::make_unique<DraftGraph>();
            graph->stream=stream;
            graph->embedding=tb::empty_like(embedding);
            graph->positions=tb::empty_like(at);
            graph->pool=GraphPoolLease::acquire(default_context(embedding.get_device()),stream);
            graph->window=std::make_unique<DecodeWindow>(stream);
            item=std::move(graph);
        }
        auto& graph=*item;
        graph.embedding.copy_(embedding); graph.positions.copy_(at);
        if (!graph.window->valid()) {
            graph.window->capture([&]{graph.result=block(graph.embedding,graph.positions);},[]{},[&]{graph.result={};});
            graph_initialized_=true;
        }
        if (!graph.window->enqueue_if_device_only()) graph.window->run();
        ++replays_;
        return graph.result.narrow(1,0,requested);
    }

    Tensor draft_next(Tensor hidden,Tensor previous) override {
        auto projected=draft_next_hidden(std::move(hidden),std::move(previous));
        return active_target_->logits_from_hidden(std::move(projected));
    }

    Tensor draft_next_hidden(Tensor hidden,Tensor previous) override {
        if (!active_target_) throw std::logic_error("Inkling DSpark has no active draft");
        auto token=quant_embedding_lookup(weights_->markov,previous).reshape({1,1,config.rank}).to(tb::kFloat32);
        auto recurrent_input=tb::cat({markov_state_,token,hidden,last_features_},-1);
        auto gates=weights_->joint.forward(execution,recurrent_input).reshape({1,1,3*config.rank})+weights_->joint_bias;
        auto candidate=tb::tanh(gates.narrow(-1,config.rank,config.rank));
        markov_state_=candidate+tb::sigmoid(gates.narrow(-1,0,config.rank))*(markov_state_-candidate);
        auto delta=weights_->correction.forward(execution,tb::tanh(gates.narrow(-1,2*config.rank,config.rank).contiguous()));
        return hidden+delta.reshape(hidden.sizes())+weights_->correction_bias;
    }

private:
    struct History {Tensor k,v;};
    struct DraftGraph {
        cudaStream_t stream=nullptr;
        Tensor embedding,positions,result;
        std::shared_ptr<GraphPoolLease> pool;
        std::unique_ptr<DecodeWindow> window;
    };
    std::vector<History> history_;
    std::map<int,std::unique_ptr<DraftGraph>> graphs_;
    Tensor last_features_,markov_state_;
    InklingCausalLm* active_target_=nullptr;
    int64_t position_=0;
    int depth_=1;
    bool graph_initialized_=false;
    uint64_t replays_=0;

    bool valid_features(const Tensor& features) const {
        return features.defined() && features.dim()==3 && features.size(0)==1 && features.size(1)>0 &&
            features.size(2)==int64_t(config.taps.size())*config.hidden;
    }
    void reserve(int64_t end,tb::TensorOptions options) {
        ResidentAllocationScope resident;
        const auto& layers=weights_->layers;
        history_.resize(layers.size());
        if (!last_features_.defined()) last_features_=tb::empty({1,1,config.hidden},options.dtype(tb::kFloat32));
        for (size_t i=0;i<layers.size();++i) {
            auto& history=history_[i];
            const auto capacity=layers[i].sliding?std::min(config.maximum,config.window+1):
                std::min<int64_t>(config.maximum,(end+1023)/1024*1024);
            if (history.k.defined() && history.k.size(0)>=capacity) continue;
            graphs_.clear();
            History next{tb::empty({capacity,config.kv_heads,config.width},options.dtype(tb::kBFloat16)),
                tb::empty({capacity,config.kv_heads,config.width},options.dtype(tb::kBFloat16))};
            if (history.k.defined() && position_) {
                next.k.narrow(0,0,position_).copy_(history.k.narrow(0,0,position_));
                next.v.narrow(0,0,position_).copy_(history.v.narrow(0,0,position_));
            }
            history=std::move(next);
        }
    }
    Tensor mixed(const Tensor& features,const Tensor& coefficients,const Tensor& norm) const {
        auto rows=features.to(tb::kFloat32).reshape({1,features.size(1),int64_t(config.taps.size()),config.hidden});
        return rms_norm_cuda(tb::sum(rows*coefficients,2,false).contiguous(),norm,config.epsilon);
    }
    Tensor rotary(const Tensor& positions,tb::TensorOptions options) const {
        auto table=tb::empty({positions.numel(),config.width},options.dtype(tb::kFloat32));
        dflash2::rotary_table(positions.data_ptr<int64_t>(),table.data_ptr<float>(),int(positions.numel()),config.width,config.theta,mfq_current_cuda_stream());
        return table;
    }
    Tensor block(const Tensor& anchor,const Tensor& positions) {
        const auto stream=mfq_current_cuda_stream();
        const auto rows=int(positions.numel());
        auto hidden=tb::cat({anchor,weights_->mask.reshape({1,1,config.hidden}).expand({1,rows-1,config.hidden})},1);
        const auto table=rotary(positions,hidden.options());
        const auto& layers=weights_->layers;
        for (size_t i=0;i<layers.size();++i) {
            const auto& layer=layers[i];const auto& history=history_[i];
            auto qkv=layer.qkv.forward(execution,rms_norm_cuda(hidden.contiguous(),layer.attn_norm,config.epsilon));
            const auto normalized=[&](Tensor values,const Tensor& norm,int heads) {
                values=values.to(tb::kFloat32).contiguous();
                auto result=tb::empty_like(values);
                dflash2::norm_rope_tabled(values.data_ptr<float>(),norm.data_ptr<float>(),table.data_ptr<float>(),
                    result.data_ptr<float>(),rows,heads,config.width,config.epsilon,stream);
                return result;
            };
            auto q=normalized(qkv[0],layer.q_norm,config.heads),k=normalized(qkv[1],layer.k_norm,config.kv_heads);
            auto v=qkv[2].to(tb::kFloat32).contiguous();
            const auto window=int(history.k.size(0));
            const auto parts=(window-1+rows+127)/128;
            auto partial=tb::empty({int64_t(parts)*rows*config.heads*(config.width+2)},hidden.options());
            auto attended=tb::empty({1,rows,config.heads*config.width},hidden.options());
            dflash2::attention(q.data_ptr<float>(),history.k.data_ptr(),history.v.data_ptr(),k.data_ptr<float>(),v.data_ptr<float>(),
                partial.data_ptr<float>(),attended.data_ptr<float>(),rows,config.heads,config.kv_heads,config.width,
                window,position_,stream,positions.data_ptr<int64_t>(),true);
            hidden=hidden+layer.output.forward(execution,attended).reshape(hidden.sizes());
            auto norm=rms_norm_cuda(hidden.contiguous(),layer.ffn_norm,config.epsilon);
            hidden=hidden+layer.ffn.forward(execution,norm).reshape(hidden.sizes());
        }
        return rms_norm_cuda(hidden.narrow(1,1,rows-1).contiguous(),weights_->final_norm,config.epsilon);
    }
};

}  // namespace

std::unique_ptr<::MtpModule> load_inkling_dspark(InklingCausalLm& model,const std::filesystem::path& path) {
    mfq::GgufModelSource source(path);
    auto weights=std::make_shared<Weights>(DsConfig(source,model));
    const auto& config=weights->config;
    TransformerWeightLoader loader{*model.execution,source};
    const auto name=[&](const std::string& value){return source.canonical_tensor_name(value);};
    const auto norm=[&](const std::string& value,int64_t size) {
        auto tensor=loader.dense(name(value));
        if (tensor.numel()!=size) throw std::invalid_argument("Inkling DSpark tensor shape differs: "+value);
        return tensor.to(tb::kFloat32).reshape({size}).contiguous();
    };
    const auto linear=[&](const std::string& value,int64_t out,int64_t input) {
        auto weight=loader.linear(name(value));
        if (weight.out()!=out || weight.neuron_len()!=input)
            throw std::invalid_argument("Inkling DSpark projection geometry differs: "+value);
        return weight;
    };
    const auto taps=int64_t(config.taps.size());
    const auto learned_mix=norm("dspark_fuse.weight",(config.sliding.size()+1)*taps);
    auto mixing=learned_mix.to(tb::kCPU).reshape({-1,taps});
    const auto mixing_device=learned_mix.device();
    const auto mix_row=[&](int64_t row) {
        auto coefficients=mixing.select(0,row).clone();
        auto* values=coefficients.data_ptr<float>();
        const auto largest=*std::max_element(values,values+taps);
        float total=0;
        for (int64_t i=0;i<taps;++i) total+=(values[i]=std::exp(values[i]-largest));
        if (!std::isfinite(total) || total<=0) throw std::invalid_argument("invalid Inkling DSpark feature mixing");
        for (int64_t i=0;i<taps;++i) values[i]/=total;
        return coefficients.to(mixing_device).reshape({1,1,taps,1});
    };
    weights->mask=norm("mask_embd.weight",config.hidden);
    weights->final_norm=norm("output_norm.weight",config.hidden);
    weights->tail_norm=norm("dspark_norm.weight",config.hidden);
    weights->tail_mix=mix_row(int64_t(config.sliding.size()));
    weights->joint_bias=norm("joint_proj.bias",3*config.rank);
    weights->correction_bias=norm("dspark_hidden_out.bias",config.hidden);
    weights->markov=linear("markov_w1.weight",model.lm_head.out(),config.rank);
    weights->joint=linear("joint_proj.weight",3*config.rank,2*(config.hidden+config.rank));
    weights->correction=linear("dspark_hidden_out.weight",config.hidden,config.rank);
    for (size_t i=0;i<config.sliding.size();++i) {
        const auto prefix="blk."+std::to_string(i)+".";
        auto& layer=weights->layers.emplace_back();
        layer.sliding=config.sliding[i]; layer.feature_mix=mix_row(int64_t(i));
        layer.feature_norm=norm(prefix+"dflare_norm.weight",config.hidden);
        layer.attn_norm=norm(prefix+"attn_norm.weight",config.hidden);
        layer.q_norm=norm(prefix+"attn_q_norm.weight",config.width);
        layer.k_norm=norm(prefix+"attn_k_norm.weight",config.width);
        layer.context_k_norm=norm(prefix+"attn_k_ctx_norm.weight",config.width);
        layer.ffn_norm=norm(prefix+"ffn_norm.weight",config.hidden);
        layer.qkv=loader.projections({name(prefix+"attn_q.weight"),name(prefix+"attn_k.weight"),name(prefix+"attn_v.weight")});
        layer.context_kv=loader.projections({name(prefix+"attn_k_ctx.weight"),name(prefix+"attn_v_ctx.weight")});
        layer.output=linear(prefix+"attn_output.weight",config.hidden,config.heads*config.width);
        layer.ffn.down=loader.linear(name(prefix+"ffn_down.weight"));
        layer.ffn.gate_up=loader.gate_up({name(prefix+"ffn_gate.weight"),name(prefix+"ffn_up.weight")},layer.ffn.down);
    }
    model.draft_taps=config.taps;
    return std::make_unique<InklingDspark>(std::move(weights),*model.execution);
}

}  // namespace mfq::cuda

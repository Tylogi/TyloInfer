#pragma once

#include "core/causal_model.h"
#include "models/common/causal_model.h"
#include "models/inkling2/config.h"
#include <filesystem>

struct MtpModule;

namespace mfq::cuda {

struct InklingWeights;
struct InklingTargetGraphs;
std::unique_ptr<::MtpModule> load_inkling_dspark(InklingCausalLm&,const std::filesystem::path&);
std::optional<mfq_tensor_backend::Tensor> inkling_graph_hidden(InklingCausalLm&,
    const mfq_tensor_backend::Tensor& ids,mfq_tensor_backend::Tensor* raw,int64_t confirmed,
    int64_t horizon,mfq_tensor_backend::Tensor& logits,bool& initialized);
void place_inkling_experts(InklingCausalLm&,int64_t sequences);
void clear_inkling_graphs(InklingCausalLm&);

struct InklingModel : CausalResources {
    template<class Backend> using CausalModel=mfq::models::inkling2::CausalLm<Backend>;
    mfq::models::inkling2::Config config;
    mfq_tensor_backend::Tensor embedding_norm, output_query;
    std::vector<std::shared_ptr<const InklingWeights>> weight_layers;
    std::shared_ptr<InklingTargetGraphs> decode_graphs;
    std::vector<int64_t> draft_taps;
    std::vector<mfq_tensor_backend::Tensor> draft_features;
    size_t next_draft_tap=0;
    int64_t draft_visit=0;

    void adapter_validate_load_options() const;
    void adapter_load_final_state(const mfq::ModelSource&, mfq_tensor_backend::Tensor&);
    std::unique_ptr<Block> adapter_load_block(const mfq::ModelSource&,int,int,const std::string&);
    mfq_tensor_backend::Tensor adapter_embed(mfq_tensor_backend::Tensor) const;
    mfq_tensor_backend::Tensor adapter_logits(const QuantLinear&,mfq_tensor_backend::Tensor) const;
    mfq_tensor_backend::Tensor adapter_last_logits(const QuantLinear&,mfq_tensor_backend::Tensor) const;
    mfq_tensor_backend::Tensor adapter_next_token(const QuantLinear&,mfq_tensor_backend::Tensor) const;
    void adapter_begin_forward(bool capture);
    void adapter_capture_hidden(const mfq_tensor_backend::Tensor&);
    mfq_tensor_backend::Tensor adapter_raw_hidden(const mfq_tensor_backend::Tensor&,
        const mfq_tensor_backend::Tensor&) const;
};

template<> struct CudaSessionCodec<InklingModel> {
    using Model=InklingCausalLm;
    static TextSessionStateKind kind(const Model&);
    static bool supports_paged(const Model&);
    static TextSessionState capture(const Model&,const std::vector<int64_t>&);
    static void restore(Model&,const TextSessionState&);
};

}  // namespace mfq::cuda

namespace mfq::models::inkling2 {

template<class Backend> struct CausalLm : CausalModelBase<Backend,CausalLm<Backend>> {
    static bool accepts_backbone(std::string_view implementation) {return implementation=="inkling2_loop";}
    bool adapter_supports_speculation() const noexcept {return !this->draft_taps.empty();}
    bool adapter_allows_speculative_position_override() const noexcept {return true;}
    template<class Graph,class Source>
    void adapter_load_config(std::string_view payload,const Graph&,const Source&) {
        this->config=Config::from_json(payload);
        this->metadata=this->config.metadata;
        this->weight_layers.resize(this->config.plan->weight_layers());
    }
    void adapter_set_max_position_embeddings(int64_t value) {this->config.metadata.max_position_embeddings=value;}
};

}  // namespace mfq::models::inkling2

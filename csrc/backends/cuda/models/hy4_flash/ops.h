#pragma once

#include "core/causal_model.h"
#include "models/common/causal_model.h"
#include "models/hy4_flash/config.h"
#include "storage/session_state.h"

namespace mfq::cuda::hy4 { struct Shared; }
struct MtpModule;

namespace mfq::cuda {

struct Hy4Model : CausalResources {
    template<class Backend> using CausalModel=mfq::models::hy4_flash::CausalLm<Backend>;
    mfq::models::hy4_flash::Config config;
    std::shared_ptr<hy4::Shared> shared;
    mfq_tensor_backend::Tensor output_function,output_scale,output_base;

    void adapter_validate_load_options() const;
    void adapter_load_final_state(const mfq::ModelSource&,mfq_tensor_backend::Tensor&);
    void configure_kv_tiers();
    int cold_kv_bits() const noexcept;
    bool adapter_prefetch_token_rows(Hy4CausalLm&,const mfq_tensor_backend::Tensor&);
    std::unique_ptr<Block> adapter_load_block(const mfq::ModelSource&,int,int,const std::string&);
    mfq_tensor_backend::Tensor adapter_prepare_hidden(mfq_tensor_backend::Tensor,int64_t,int64_t) const;
    mfq_tensor_backend::Tensor adapter_finalize_hidden(mfq_tensor_backend::Tensor,
        const mfq_tensor_backend::Tensor&,int64_t,int64_t) const;
    mfq_tensor_backend::Tensor adapter_raw_hidden(const mfq_tensor_backend::Tensor&,
        const mfq_tensor_backend::Tensor& finalized) const {return finalized;}
    void adapter_reset(int64_t batch);
    void adapter_commit_speculative();
    void adapter_rollback_speculative(int64_t keep);
};

void enable_hy4_draft_history(Hy4CausalLm&,const std::vector<int64_t>& layers,const std::vector<int64_t>& reusable);
const RopeCache& hy4_full_rope(const Hy4CausalLm&);
int64_t hy4_draft_selection_width(const Hy4CausalLm&,int64_t layer,int64_t slot=-1);
std::pair<mfq_tensor_backend::Tensor,mfq_tensor_backend::Tensor> hy4_draft_history(
    const Hy4CausalLm&,int64_t kv_layer,int64_t index_layer,int64_t position,int window,int64_t slot=-1);
std::unique_ptr<::MtpModule> load_hy4_dfly2(Hy4CausalLm&,const std::filesystem::path&);
std::optional<mfq_tensor_backend::Tensor> hy4_decode_graph_logits(Hy4CausalLm&,
    const mfq_tensor_backend::Tensor& ids,const mfq_tensor_backend::Tensor& host_ids,int64_t horizon);
std::optional<mfq_tensor_backend::Tensor> hy4_graph_hidden(Hy4CausalLm&,
    const mfq_tensor_backend::Tensor& ids,const mfq_tensor_backend::Tensor& host_ids,
    int64_t confirmed,int64_t horizon,mfq_tensor_backend::Tensor& logits,bool& initialized);
void clear_hy4_decode_graph(Hy4CausalLm&);
void prepare_hy4_expert_cache(Hy4CausalLm&,int verification_rows);

template<> struct CudaSessionCodec<Hy4Model> {
    using Model=Hy4CausalLm;
    static TextSessionStateKind kind(const Model&);
    static bool supports_paged(const Model&);
    static TextSessionState capture(const Model&,const std::vector<int64_t>&);
    static void restore(Model&,const TextSessionState&);
};

}  // namespace mfq::cuda

namespace mfq::models::hy4_flash {

template<class Backend> struct CausalLm : CausalModelBase<Backend,CausalLm<Backend>> {
    static bool accepts_backbone(std::string_view name) {return name=="hy_v4_flash";}
    bool adapter_supports_speculation() const noexcept {return true;}
    bool adapter_allows_speculative_position_override() const noexcept {return true;}
    template<class Graph,class Source>
    void adapter_load_config(std::string_view payload,const Graph&,const Source&) {
        this->config=Config::from_json(payload);
        this->metadata=this->config.metadata;
    }
    void adapter_set_max_position_embeddings(int64_t value) {this->config.metadata.max_position_embeddings=value;}
};

}  // namespace mfq::models::hy4_flash

#include "config.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mfq::models::hy4_flash {

Config Config::from_json(std::string_view payload) {
    const auto json=nlohmann::json::parse(payload);
    Config result;
    auto& metadata=result.metadata;
    metadata.model_type=json.at("model_type").get<std::string>();
    if (metadata.model_type!="hy_v4_flash") throw std::invalid_argument("unsupported Hy4 config");
    metadata.vocab_size=json.at("vocab_size").get<int64_t>();
    metadata.hidden_size=json.at("hidden_size").get<int64_t>();
    metadata.num_hidden_layers=json.at("num_hidden_layers").get<int64_t>();
    metadata.max_position_embeddings=json.at("max_position_embeddings").get<int64_t>();
    metadata.num_key_value_heads=json.at("num_key_value_heads").get<int64_t>();
    metadata.head_dim=json.at("head_dim").get<int64_t>();
    metadata.num_experts=json.at("num_experts").get<int64_t>();
    metadata.layer_types=json.at("layer_types").get<std::vector<std::string>>();
    metadata.rotary_dim=json.at("rotary_dim").get<int64_t>();
    metadata.rope_base=json.at("rope_theta").get<double>();
    metadata.rope_interleaved=true;
    metadata.rms_norm_eps=json.at("rms_norm_eps").get<double>();
    metadata.tie_word_embeddings=json.value("tie_word_embeddings",false);
    const auto& hc=json.at("hyper_connection");
    metadata.hc_mult=hc.at("streams").get<int64_t>();
    metadata.hc_eps=hc.at("epsilon").get<double>();
    result.hc_magnitude=hc.at("magnitude").get<double>();
    result.q_rank=json.at("q_lora_rank").get<int64_t>();
    result.kv_rank=json.at("kv_lora_rank").get<int64_t>();
    result.value_head_dim=json.at("value_head_dim").get<int64_t>();
    result.window=json.at("sliding_window").get<int64_t>();
    result.sliding_rope_base=json.at("sliding_rope_theta").get<double>();
    result.dense_width=json.at("intermediate_size").get<int64_t>();
    result.dense_layers=json.at("dense_layers").get<int64_t>();
    result.latent_width=json.at("routed_latent_size").get<int64_t>();
    result.expert_width=json.at("moe_intermediate_size").get<int64_t>();
    result.expert_top_k=json.at("num_experts_per_tok").get<int64_t>();
    result.routing.scale=json.at("routed_scaling_factor").get<double>();
    result.routing.normalize=json.at("routed_weight_normalization").get<bool>();
    const auto activation=json.at("router_activation").get<int>();
    if (activation!=1 && activation!=2) throw std::invalid_argument("unsupported Hy4 routing activation");
    result.routing.activation=activation==2?RouterActivation::sigmoid:RouterActivation::softmax;
    const auto& index=json.at("index_pool");
    result.group_size=index.at("group_size").get<int64_t>();
    result.index_budget=index.at("token_budget").get<int64_t>();
    result.keep_tail=index.at("select_tail").get<bool>();
    result.index_heads=index.at("num_heads").get<int64_t>();
    result.index_width=index.at("key_head_dim").get<int64_t>();
    if (metadata.vocab_size<=0 || metadata.hidden_size<=0 || metadata.num_hidden_layers<=0 ||
        metadata.num_hidden_layers!=int64_t(metadata.layer_types.size()) ||
        metadata.max_position_embeddings<=0 || metadata.num_key_value_heads!=1 ||
        metadata.rotary_dim<=0 || metadata.rotary_dim%2 || metadata.head_dim<=metadata.rotary_dim ||
        metadata.hc_mult<=0 || result.q_rank<=0 || result.kv_rank<=0 || result.value_head_dim<=0 ||
        result.window<=0 || result.group_size<2 || result.index_budget%result.group_size ||
        result.index_budget/result.group_size<=int64_t(result.keep_tail) || result.index_heads<=0 ||
        result.index_width<metadata.rotary_dim || result.dense_width<=0 || result.dense_layers<0 ||
        result.dense_layers>metadata.num_hidden_layers || result.latent_width<0 ||
        result.expert_width<=0 || result.expert_top_k<=0 || result.expert_top_k>metadata.num_experts)
        throw std::invalid_argument("invalid Hy4 attention or expert geometry");
    for (double v:{metadata.rope_base,metadata.rms_norm_eps,metadata.hc_eps,result.hc_magnitude,
        result.sliding_rope_base,result.routing.scale})
        if (!(v>0) || !std::isfinite(v)) throw std::invalid_argument("invalid Hy4 scales");
    const auto& heads=json.at("num_attention_heads");
    result.heads=heads.is_array()?heads.get<std::vector<int64_t>>() :
        std::vector<int64_t>(metadata.num_hidden_layers,heads.get<int64_t>());
    const auto& limits=json.at("swiglu_clamp_exp");
    result.swiglu_limit=limits.is_array()?limits.get<std::vector<double>>() :
        std::vector<double>(metadata.num_hidden_layers,limits.get<double>());
    if (result.heads.size()<size_t(metadata.num_hidden_layers) ||
        result.swiglu_limit.size()<size_t(metadata.num_hidden_layers) ||
        std::any_of(result.heads.begin(),result.heads.end(),[](auto v){return v<=0;}) ||
        std::any_of(result.swiglu_limit.begin(),result.swiglu_limit.end(),[](auto v){return v<0 || !std::isfinite(v);}))
        throw std::invalid_argument("invalid Hy4 per-layer dimensions");
    metadata.num_attention_heads=result.heads.front();
    if (const auto it=json.find("engram");it!=json.end()) {
        auto& e=result.engram.emplace();
        e.ngram=it->at("ngram_size").get<int64_t>();
        e.heads=it->at("heads_per_ngram").get<int64_t>();
        e.row_width=it->at("row_width").get<int64_t>();
        e.pad=it->at("pad_token").get<int64_t>();
        e.layers=it->at("layers").get<std::vector<int64_t>>();
        e.remap=it->at("token_map").get<std::vector<int64_t>>();
        e.multipliers=it->at("multipliers").get<std::vector<uint64_t>>();
        e.offsets=it->at("head_offsets").get<std::vector<uint64_t>>();
        e.vocab_sizes=it->at("head_vocab_sizes").get<std::vector<uint64_t>>();
        if (e.ngram<2 || e.ngram>32 || e.heads<=0 || e.heads>4096 || e.row_width<=0 || e.layers.empty() ||
            e.pad<0 || e.pad>=int64_t(e.remap.size()) || e.remap[e.pad]<0 ||
            e.multipliers.size()!=e.layers.size()*size_t(e.ngram) ||
            e.offsets.size()!=e.layers.size()*size_t((e.ngram-1)*e.heads) || e.offsets.size()!=e.vocab_sizes.size())
            throw std::invalid_argument("invalid Hy4 Engram metadata");
        for (size_t i=0;i<e.layers.size();++i)
            if (e.layers[i]<0 || e.layers[i]>=metadata.num_hidden_layers || (i && e.layers[i]<=e.layers[i-1]))
                throw std::invalid_argument("invalid Hy4 Engram layer order");
        for (size_t i=0;i<e.offsets.size();++i)
            if (!e.vocab_sizes[i] || e.offsets[i]>std::numeric_limits<int32_t>::max() ||
                e.vocab_sizes[i]>std::numeric_limits<int32_t>::max()-e.offsets[i])
                throw std::invalid_argument("invalid Hy4 Engram bucket range");
    }
    return result;
}

}  // namespace mfq::models::hy4_flash

#include "config.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mfq::models::inkling2 {

Config Config::from_json(std::string_view payload) {
    const auto json=nlohmann::json::parse(payload);
    Config result;
    auto& metadata=result.metadata;
    const auto& loop=json.at("loop");
    const auto& relative=json.at("relative_attention");
    metadata.model_type=json.at("model_type").get<std::string>();
    if (metadata.model_type!="inkling2_loop") throw std::invalid_argument("unsupported Inkling config");
    metadata.no_rope=true;
    metadata.vocab_size=json.at("visible_vocab_size").get<int64_t>();
    metadata.hidden_size=json.at("hidden_size").get<int64_t>();
    metadata.num_hidden_layers=json.at("num_hidden_layers").get<int64_t>();
    metadata.max_position_embeddings=json.at("max_position_embeddings").get<int64_t>();
    metadata.head_dim=json.at("head_dim").get<int64_t>();
    metadata.num_experts=json.at("num_experts").get<int64_t>();
    metadata.rms_norm_eps=json.at("rms_norm_eps").get<double>();
    metadata.layer_types=json.at("layer_types").get<std::vector<std::string>>();
    metadata.final_logit_softcapping=json.value("final_logit_softcapping",0.0);
    result.padded_vocab=json.at("vocab_size").get<int64_t>();
    result.loop_start=loop.at("start").get<int64_t>();
    result.loop_end=loop.at("end").get<int64_t>();
    result.residual_block=loop.at("residual_block_size").get<int64_t>();
    result.plan.emplace(json.at("num_weight_layers").get<int64_t>(),result.loop_start,result.loop_end,
        loop.at("count").get<int64_t>(),LoopLayerNumbering::physical);
    result.window=json.at("sliding_window").get<int64_t>();
    result.relative_extent=relative.at("extent").get<int64_t>();
    result.relative_dim=relative.at("key_dim").get<int64_t>();
    result.log_scale_floor=relative.at("log_scale_floor").get<int64_t>();
    result.qk_clip=relative.at("query_key_clip").get<float>();
    result.relative_clip=relative.at("clip").get<float>();
    result.log_scale_alpha=relative.at("log_scale_alpha").get<float>();
    result.taps=json.at("short_convolution_width").get<int64_t>();
    result.situ_beta=json.at("situ_beta").get<float>();
    result.situ_linear_beta=json.at("situ_linear_beta").get<float>();
    if (json.contains("routed_scaling_factor") && json["routed_scaling_factor"].is_number())
        result.routed_scale=json["routed_scaling_factor"].get<float>();
    const auto layer_values=[&](const char* field) {
        const auto& value=json.at(field);
        auto out=value.is_array()?value.get<std::vector<int64_t>>() :
            std::vector<int64_t>(result.plan->weight_layers(),value.get<int64_t>());
        if (out.size()!=size_t(result.plan->weight_layers()) ||
            std::any_of(out.begin(),out.end(),[](auto dim) {return dim<=0;}))
            throw std::invalid_argument(std::string("invalid Inkling layer dimensions in ")+field);
        return out;
    };
    result.query_heads=layer_values("num_attention_heads");
    result.kv_heads=layer_values("num_key_value_heads");
    result.intermediate=layer_values("moe_intermediate_size");
    result.top_k=layer_values("num_experts_per_tok");
    metadata.num_attention_heads=result.query_heads.front();
    metadata.num_key_value_heads=result.kv_heads.front();
    if (metadata.vocab_size<=0 || result.padded_vocab<metadata.vocab_size || metadata.hidden_size<=0 ||
        metadata.head_dim<=0 || metadata.max_position_embeddings<=0 || metadata.num_experts<=0 ||
        metadata.num_hidden_layers!=result.plan->execution_layers() ||
        metadata.layer_types.size()!=size_t(metadata.num_hidden_layers) || result.residual_block<=0 ||
        result.loop_start<=0 || result.loop_start%result.residual_block || result.loop_end%result.residual_block ||
        result.window<=0 || result.relative_extent<=0 || result.relative_dim<=0 || result.log_scale_floor<=0 ||
        result.taps<2 || result.taps>64)
        throw std::invalid_argument("invalid Inkling topology");
    for (auto value:{metadata.rms_norm_eps,double(result.qk_clip),double(result.relative_clip),
        double(result.situ_beta),double(result.situ_linear_beta),double(result.routed_scale)})
        if (!(value>0) || !std::isfinite(value)) throw std::invalid_argument("invalid Inkling scales");
    if (result.log_scale_alpha<0 || !std::isfinite(result.log_scale_alpha) ||
        metadata.final_logit_softcapping<0 || !std::isfinite(metadata.final_logit_softcapping))
        throw std::invalid_argument("invalid Inkling attention or output scales");
    for (int64_t i=0;i<result.plan->weight_layers();++i)
        if (result.query_heads[i]%result.kv_heads[i] || result.top_k[i]>metadata.num_experts)
            throw std::invalid_argument("invalid Inkling attention or routing dimensions");
    return result;
}

}  // namespace mfq::models::inkling2

#pragma once

#include "tyloinfer/model_source.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <unordered_set>

namespace mfq {

inline constexpr const char* kDecisionConfigAsset = "__mfq_asset__/decision_config.json";
inline constexpr const char* kDecisionReadoutAsset = "__mfq_asset__/readout.safetensors";

enum class DecisionAttention { causal, bidirectional, prefix_bidirectional };

struct DecisionModelSpec {
    std::vector<std::string> codes;
    std::vector<std::int64_t> token_ids;
    DecisionAttention attention = DecisionAttention::causal;
    double temperature = 1;
    std::optional<std::size_t> max_tokens;
    bool vocabulary_readout = false;
    std::array<double,3> primitive_temperatures{1,1,1};
    double temperature_for(const std::string& primitive) const {
        return temperature*primitive_temperatures[primitive=="noul"?2:primitive=="score"?1:0];
    }
};

inline DecisionModelSpec parse_decision_model(const std::string& text) {
    using json = nlohmann::json;
    const auto config = json::parse(text);
    if (!config.is_object() || config.value("format_version", 0) != 1 ||
        config.value("format_id", std::string{}) != "d3-code-readout-v1" ||
        config.value("prompt", std::string("d3")) != "d3" ||
        config.value("pooling", std::string("last")) != "last")
        throw std::invalid_argument("unsupported decision model contract");
    DecisionModelSpec result;
    result.codes = config.at("codes").get<std::vector<std::string>>();
    result.token_ids = config.at("token_ids").get<std::vector<std::int64_t>>();
    if (result.codes.size() != 255 || result.token_ids.size() != result.codes.size())
        throw std::invalid_argument("D3 requires 255 answer codes and token IDs");
    std::unordered_set<std::string> codes;
    std::unordered_set<std::int64_t> tokens;
    for (std::size_t i = 0; i < result.codes.size(); ++i) {
        if (result.codes[i].empty() || result.codes[i].size() > 16 || result.token_ids[i] < 0 ||
            !codes.insert(result.codes[i]).second || !tokens.insert(result.token_ids[i]).second)
            throw std::invalid_argument("invalid or duplicate D3 answer code");
    }
    result.temperature = config.value("temperature", 1.0);
    if (!std::isfinite(result.temperature) || result.temperature <= 0)
        throw std::invalid_argument("decision temperature must be finite and positive");
    const auto mode = config.value("attention_mode", std::string("causal"));
    if (mode == "noncausal_full_attention") result.attention = DecisionAttention::bidirectional;
    else if (mode == "prefix_bidir") result.attention = DecisionAttention::prefix_bidirectional;
    else if (mode != "causal") throw std::invalid_argument("unsupported decision attention mode");
    for (const auto* name : {"max_length", "validated_length"}) {
        const auto found = config.find(name);
        if (found == config.end() || found->is_null()) continue;
        if (!found->is_number_integer() || found->get<std::int64_t>() <= 0)
            throw std::invalid_argument("decision input length must be positive");
        const auto limit = found->get<std::size_t>();
        result.max_tokens = result.max_tokens ? std::min(*result.max_tokens, limit) : limit;
    }
    return result;
}

inline std::optional<DecisionModelSpec> decision_model_spec(const ModelSource& source) {
    if (!source.has_asset(kDecisionConfigAsset)) {
        const auto found=source.metadata().find("gguf_metadata");
        if(found==source.metadata().end())return {};
        const auto meta=nlohmann::json::parse(found->second);
        const auto architecture=meta.value("general.architecture",std::string{});
        const auto prefix=architecture+".decision.";
        if(meta.value(prefix+"type",std::string{})!="vllm-d3")return {};
        DecisionModelSpec result;result.vocabulary_readout=true;
        const auto mode=meta.value(prefix+"attention_mode",std::string{});
        if(mode=="prefix_bidir")result.attention=DecisionAttention::prefix_bidirectional;
        else if(mode=="noncausal_full_attention" || (mode.empty() && !meta.value(architecture+".attention.causal",true)))
            result.attention=DecisionAttention::bidirectional;
        else if(!mode.empty() && mode!="causal")throw std::invalid_argument("unsupported GGUF decision attention mode");
        for(std::size_t i=0;i<3;++i) {
            result.primitive_temperatures[i]=meta.value(prefix+"temperature."+std::array<const char*,3>{"choice","score","noul"}[i],1.0);
            if(!std::isfinite(result.primitive_temperatures[i]) || result.primitive_temperatures[i]<=0)
                throw std::invalid_argument("invalid GGUF decision temperature");
        }
        return result;
    }
    const auto bytes = source.read_asset(kDecisionConfigAsset);
    if (bytes.empty() || bytes.size() > (1u << 20))
        throw std::invalid_argument("decision configuration size is invalid");
    return parse_decision_model(std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}

} // namespace mfq

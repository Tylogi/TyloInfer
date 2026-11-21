#pragma once

#include "tyloinfer/model_source.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
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
    bool question_header = false;
    std::array<double,3> primitive_temperatures{1,1,1};
    std::vector<std::size_t> temperature_limits;
    std::vector<double> question_temperatures;
    double temperature_for(const std::string& primitive, std::size_t questions) const {
        const auto bucket = std::lower_bound(temperature_limits.begin(), temperature_limits.end(), questions);
        const auto base = question_temperatures.empty() ? temperature
            : question_temperatures.at(bucket - temperature_limits.begin());
        return base * primitive_temperatures[primitive=="noul"?2:primitive=="score"?1:0];
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
    if (const auto layout = config.find("prompt_layout"); layout != config.end() && !layout->is_null()) {
        if (*layout != "header-v1") throw std::invalid_argument("unsupported decision prompt layout");
        result.question_header = true;
    }
    if (const auto table = config.find("temperature_by_questions"); table != config.end() && !table->is_null()) {
        const auto& limits = table->at("max_questions");
        const auto& temperatures = table->at("temperature");
        if (!table->is_object() || table->size() != 2 || !limits.is_array() || !temperatures.is_array() ||
                temperatures.size() != limits.size() + 1)
            throw std::invalid_argument("invalid decision question temperature table");
        for (const auto& limit : limits) {
            if (!limit.is_number_integer() || limit.get<std::int64_t>() <= 0 ||
                    (!result.temperature_limits.empty() && limit.get<std::size_t>() <= result.temperature_limits.back()))
                throw std::invalid_argument("decision question temperature limits must increase");
            result.temperature_limits.push_back(limit.get<std::size_t>());
        }
        for (const auto& temperature : temperatures) {
            const double value = temperature.get<double>();
            if (!std::isfinite(value) || value <= 0)
                throw std::invalid_argument("decision question temperatures must be finite and positive");
            result.question_temperatures.push_back(value);
        }
    }
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

struct DecisionReadout {
    std::string dtype;
    std::span<const std::byte> values;
};

inline DecisionReadout parse_decision_readout(std::span<const std::byte> bytes, std::int64_t hidden) {
    if (hidden <= 0 || bytes.size() < 8 || bytes.size() > (256u << 20))
        throw std::invalid_argument("decision readout size is invalid");
    std::uint64_t header_size = 0;
    for (unsigned i = 0; i < 8; ++i)
        header_size |= std::uint64_t(std::to_integer<unsigned char>(bytes[i])) << (8 * i);
    if (!header_size || header_size > (1u << 20) || header_size > bytes.size() - 8)
        throw std::invalid_argument("decision readout header is invalid");
    const auto header = nlohmann::json::parse(reinterpret_cast<const char*>(bytes.data() + 8),
        reinterpret_cast<const char*>(bytes.data() + 8 + header_size));
    const auto& weight = header.at("weight");
    if (weight.at("shape").get<std::vector<std::int64_t>>() != std::vector<std::int64_t>{255, hidden})
        throw std::invalid_argument("decision readout must have shape [255, hidden_size]");
    auto dtype = weight.at("dtype").get<std::string>();
    if (dtype != "F32" && dtype != "BF16" && dtype != "F16")
        throw std::invalid_argument("unsupported decision readout dtype");
    const auto offsets = weight.at("data_offsets").get<std::vector<std::uint64_t>>();
    const auto data = bytes.subspan(8 + header_size);
    const std::size_t row_bytes = 255 * (dtype == "F32" ? 4 : 2);
    if (offsets.size() != 2 || offsets[0] > offsets[1] || offsets[1] > data.size() ||
            std::uint64_t(hidden) > data.size() / row_bytes ||
            offsets[1] - offsets[0] != std::uint64_t(hidden) * row_bytes)
        throw std::invalid_argument("decision readout tensor range is invalid");
    return {std::move(dtype), data.subspan(offsets[0], offsets[1] - offsets[0])};
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

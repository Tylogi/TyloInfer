#include "engine/models/source/decision.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace tyloi::models {
namespace {

using Json = nlohmann::json;
constexpr std::size_t code_count = 255;

double temperature(double value) {
    if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument("decision temperature must be finite and positive");
    return value;
}

DecisionAttention attention(std::string_view mode) {
    if (mode == "causal") return DecisionAttention::causal;
    if (mode == "noncausal_full_attention") return DecisionAttention::bidirectional;
    if (mode == "prefix_bidir") return DecisionAttention::prefix_bidirectional;
    throw std::invalid_argument("unsupported decision attention mode");
}

DecisionSpec trained_spec(const std::vector<std::byte>& bytes) {
    if (bytes.empty() || bytes.size() > decision_config_limit)
        throw std::invalid_argument("decision configuration size is invalid");
    const auto* begin = reinterpret_cast<const char*>(bytes.data());
    const auto config = Json::parse(begin, begin + bytes.size());
    if (!config.is_object() || !config.contains("format_version") ||
        !config["format_version"].is_number_integer() || config["format_version"] != 1 ||
        config.value("format_id", std::string{}) != "d3-code-readout-v1" ||
        config.value("prompt", std::string("d3")) != "d3" ||
        config.value("pooling", std::string("last")) != "last")
        throw std::invalid_argument("unsupported decision model contract");

    DecisionSpec spec;
    spec.codes = config.at("codes").get<std::vector<std::string>>();
    const auto& ids = config.at("token_ids");
    if (spec.codes.size() != code_count || !ids.is_array() || ids.size() != code_count)
        throw std::invalid_argument("D3 requires 255 answer codes and token IDs");
    std::unordered_set<std::string> codes;
    std::unordered_set<std::int64_t> tokens;
    for (std::size_t i = 0; i < code_count; ++i) {
        const auto& code = spec.codes[i];
        if (!ids[i].is_number_integer() || code.empty() || code.size() > 16 || !codes.insert(code).second)
            throw std::invalid_argument("invalid or duplicate D3 answer code");
        const auto token = ids[i].get<std::int64_t>();
        if (token < 0 || !tokens.insert(token).second)
            throw std::invalid_argument("invalid or duplicate D3 answer token ID");
        spec.token_ids.push_back(token);
    }
    spec.temperature = temperature(config.value("temperature", 1.0));
    spec.attention = attention(config.value("attention_mode", std::string("causal")));
    for (const auto* field : {"max_length", "validated_length"}) {
        const auto found = config.find(field);
        if (found == config.end() || found->is_null()) continue;
        if (!found->is_number_integer() || found->get<std::int64_t>() <= 0)
            throw std::invalid_argument("decision input length must be positive");
        const auto limit = found->get<std::size_t>();
        spec.max_tokens = spec.max_tokens ? std::min(*spec.max_tokens, limit) : limit;
    }
    return spec;
}

} // namespace

std::optional<DecisionSpec> decision_spec(const ModelSource& source) {
    if (source.has_asset(decision_config_asset))
        return trained_spec(source.read_asset(decision_config_asset));
    const auto found = source.metadata().find("gguf_metadata");
    if (found == source.metadata().end()) return {};
    const auto meta = Json::parse(found->second);
    const auto architecture = meta.value("general.architecture", std::string{});
    const auto prefix = architecture + ".decision.";
    if (meta.value(prefix + "type", std::string{}) != "vllm-d3") return {};

    DecisionSpec spec;
    spec.vocabulary_readout = true;
    auto mode = meta.value(prefix + "attention_mode", std::string{});
    if (mode.empty()) mode = meta.value(architecture + ".attention.causal", true)
        ? "causal" : "noncausal_full_attention";
    spec.attention = attention(mode);
    constexpr std::array primitives{"choice", "score", "noul"};
    for (std::size_t i = 0; i < primitives.size(); ++i)
        spec.primitive_temperatures[i] = temperature(meta.value(prefix + "temperature." + primitives[i], 1.0));
    return spec;
}

DecisionMatrix read_decision_matrix(const ModelSource& source, std::int64_t hidden) {
    if (!source.has_asset(decision_readout_asset))
        throw std::invalid_argument("trained decision model is missing readout.safetensors");
    auto bytes = source.read_asset(decision_readout_asset);
    constexpr auto byte_limit = decision_readout_limit;
    if (hidden <= 0 || bytes.size() < 8 || bytes.size() > byte_limit)
        throw std::invalid_argument("decision readout size is invalid");
    std::uint64_t header_size = 0;
    for (unsigned i = 0; i < 8; ++i)
        header_size |= std::uint64_t(std::to_integer<unsigned char>(bytes[i])) << (8 * i);
    if (!header_size || header_size > (1u << 20) || header_size > bytes.size() - 8)
        throw std::invalid_argument("decision readout header is invalid");
    const auto* header = reinterpret_cast<const char*>(bytes.data() + 8);
    const auto document = Json::parse(header, header + header_size);
    const auto& weight = document.at("weight");
    const auto& shape = weight.at("shape");
    if (!shape.is_array() || shape.size() != 2 || !shape[0].is_number_integer() ||
        !shape[1].is_number_integer() || shape[0] != code_count || shape[1] != hidden)
        throw std::invalid_argument("decision readout must have shape [255, hidden_size]");
    const auto dtype = weight.at("dtype").get<std::string>();
    if (dtype != "F16" && dtype != "BF16" && dtype != "F32")
        throw std::invalid_argument("unsupported decision readout dtype");
    const std::size_t element = dtype == "F32" ? 4 : 2;
    const auto& range = weight.at("data_offsets");
    if (!range.is_array() || range.size() != 2 || !range[0].is_number_integer() || !range[1].is_number_integer())
        throw std::invalid_argument("decision readout tensor range is invalid");
    const auto begin = range[0].get<std::uint64_t>(), end = range[1].get<std::uint64_t>();
    const auto payload = 8 + header_size;
    if (begin > end || end > bytes.size() - payload ||
        std::uint64_t(hidden) > byte_limit / (code_count * element) ||
        end - begin != code_count * std::uint64_t(hidden) * element)
        throw std::invalid_argument("decision readout tensor range is invalid");

    const std::uint32_t exponent = dtype == "F16" ? 0x7c00 : dtype == "BF16" ? 0x7f80 : 0x7f800000;
    for (auto offset = payload + begin; offset < payload + end; offset += element) {
        std::uint32_t bits = 0;
        for (std::size_t i = 0; i < element; ++i)
            bits |= std::uint32_t(std::to_integer<unsigned char>(bytes[offset + i])) << (8 * i);
        if ((bits & exponent) == exponent)
            throw std::invalid_argument("decision readout contains nonfinite weights");
    }
    const auto kind = dtype == "F32" ? kFloat32 : dtype == "BF16" ? kBFloat16 : kFloat16;
    return {std::move(bytes), static_cast<std::size_t>(payload + begin), kind};
}

} // namespace tyloi::models

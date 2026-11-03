#include "context.h"
#include "tyloinfer/model_source.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace mfq::server {
namespace {
using json = nlohmann::json;

constexpr std::int64_t qsa_buffer = std::int64_t(64) << 20;

std::optional<std::int64_t> whole(const json& value) {
    if (!value.is_number_integer() && !value.is_number_unsigned()) return {};
    const auto number = value.get<std::int64_t>();
    if (number <= 0 || number > 2147483647) return {};
    return number;
}

// declared_context_size: the first positive length key in the config or its text/language sub-object.
std::optional<std::int64_t> declared(const json& config) {
    for (const auto* source : {&config,
            config.contains("text_config") && config["text_config"].is_object() ? &config["text_config"] : nullptr,
            config.contains("language_config") && config["language_config"].is_object() ? &config["language_config"] : nullptr}) {
        if (!source) continue;
        for (const char* key : {"max_position_embeddings", "max_seq_len", "max_seq_length", "seq_length", "n_positions"})
            if (source->contains(key)) {
                const auto value = whole((*source)[key]);
                if (value) return value;
            }
    }
    return {};
}

std::optional<std::int64_t> integer(const json& text, const char* name, std::int64_t fallback = -1) {
    if (!text.contains(name)) return fallback < 0 ? std::optional<std::int64_t>{} : std::optional<std::int64_t>{fallback};
    return whole(text[name]);
}
} // namespace

YarnLimits yarn_limits(const json& config) {
    YarnLimits result;
    const json empty = json::object();
    const json& text = config.contains("text_config") && config["text_config"].is_object()
        ? config["text_config"]
        : config.contains("language_config") && config["language_config"].is_object() ? config["language_config"] : empty;
    result.native = declared(config);
    std::string model_type = text.value("model_type", config.value("model_type", ""));
    std::replace(model_type.begin(), model_type.end(), '-', '_');
    static const std::array<const char*, 7> yarn_models{"qwen3_5", "qwen3_5_text", "qwen3_5_moe",
        "qwen3_6", "qwen3_8", "qwen4_exp", "qwen4_exp_text"};
    const bool supported = std::find(yarn_models.begin(), yarn_models.end(), model_type) != yarn_models.end();
    const json& rope_all = text.contains("rope_parameters") && text["rope_parameters"].is_object()
        ? text["rope_parameters"] : text.contains("rope_scaling") && text["rope_scaling"].is_object() ? text["rope_scaling"] : empty;
    const json& rope = rope_all.contains("full_attention") && rope_all["full_attention"].is_object()
        ? rope_all["full_attention"] : rope_all;
    if (rope.empty()) return result;
    const std::string rope_type = rope.contains("rope_type") ? rope["rope_type"].get<std::string>()
        : rope.value("type", std::string("default"));
    if (rope_type == "yarn" && rope.contains("original_max_position_embeddings")) {
        if (const auto original = whole(rope["original_max_position_embeddings"])) result.native = original;
    }
    const double factor = rope.contains("factor") && rope["factor"].is_number() ? rope["factor"].get<double>() : 4.0;
    if (!supported || (rope_type != "default" && rope_type != "yarn") || !result.native ||
        !std::isfinite(factor) || factor <= 1 || *result.native * factor > 2147483647.0)
        return result;
    result.factor = factor;
    result.extended = static_cast<std::int64_t>(std::floor(*result.native * factor));
    return result;
}

QsaNeed qsa_requirement(const ModelSource& source, const json& config, std::int64_t context,
        const std::optional<std::int64_t>& budget, const std::optional<std::int64_t>& maximum_context, bool cuda) {
    QsaNeed need;
    const json empty = json::object();
    const json& text = config.contains("text_config") && config["text_config"].is_object() ? config["text_config"] : config;
    if (!text.is_object() || (text.value("model_type", "") != "qwen4_exp" && text.value("model_type", "") != "qwen4_exp_text")) {
        need.reason = "KV offload currently supports QSA only";
        return need;
    }
    std::int64_t maximum = integer(text, "max_position_embeddings").value_or(0);
    const YarnLimits limits = yarn_limits(config);
    if (maximum_context) {
        if (limits.extended && *maximum_context > *limits.extended) {
            need.reason = "QSA target context exceeds the YaRN extension limit";
            return need;
        }
        if (!limits.extended && limits.native && *maximum_context > *limits.native) {
            need.reason = "QSA target context exceeds the YaRN extension limit";
            return need;
        }
        maximum = *maximum_context;
    } else if (limits.native) maximum = *limits.native;
    const auto layers = integer(text, "num_hidden_layers"), width = integer(text, "indexer_head_dim"),
        ratio = integer(text, "indexer_compress_ratio");
    const auto interval = integer(text, "full_attention_interval", 4);
    if (!layers || !width || !ratio || !interval || context < 512 || context > maximum) {
        need.reason = "QSA target context exceeds the model capacity";
        return need;
    }
    std::size_t full = 0;
    if (text.contains("layer_types") && text["layer_types"].is_array() && text["layer_types"].size() == static_cast<std::size_t>(*layers)) {
        for (const auto& layer : text["layer_types"]) if (layer == "full_attention") ++full;
    } else full = static_cast<std::size_t>((*layers + *interval - 1) / *interval);
    bool predictor = false;
    for (const auto& tensor : source.tensors())
        if (tensor.name.rfind("predictor.block.", 0) == 0 || tensor.name.rfind("mtp.", 0) == 0) { predictor = true; break; }
    if (predictor) full += static_cast<std::size_t>(integer(text, "mtp_num_hidden_layers", 0).value_or(0));
    if (!full) {
        need.reason = "the model has no QSA layers";
        return need;
    }
    if (budget) {
        const auto heads = integer(text, "num_key_value_heads"), dimension = integer(text, "head_dim");
        if (!heads || !dimension) {
            need.reason = "invalid QSA head geometry";
            return need;
        }
        const std::int64_t buffer = std::max<std::int64_t>({4096, *heads * *dimension * 4 * *ratio * 4, *width * 16 * 4 * 2});
        if (std::min(qsa_buffer, *budget / 4) < buffer) {
            need.reason = "QSA KV budget is below the microblock I/O floor (budget_bytes must be at least " +
                std::to_string(buffer * 4) + ")";
            return need;
        }
        if (cuda) {
            const std::int64_t selected = std::min(context, integer(text, "indexer_budget", 2048).value_or(2048) + *ratio - 1);
            const std::int64_t staging = selected * *heads * *dimension * 4 * 4;
            if (std::min(qsa_buffer, *budget / 4) < staging) {
                need.reason = "CUDA QSA KV budget is below the staging floor (budget_bytes must be at least " +
                    std::to_string(staging * 4) + ")";
                return need;
            }
        }
        need.buffer_bytes = std::min(qsa_buffer, *budget / 4);
    } else need.buffer_bytes = qsa_buffer;
    need.required_index_bytes = cuda
        ? static_cast<std::int64_t>(full) * *width * (context + context / *ratio) * 2
        : static_cast<std::int64_t>(full) * (*width * ((*ratio + 5) * 2 + (context / *ratio) * 4) + 2);
    need.supported = true;
    return need;
}
} // namespace mfq::server

#include "analysis.h"
#include "api/errors.h"
#include "instance.h"

#include "tyloinfer/model_source.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <vector>

namespace mfq::server {
namespace {

using json = nlohmann::json;

/** Signals a config the cache model cannot describe; the caller answers with null. */
struct Unsupported {};

constexpr std::array<const char*, 7> qwen4_kinds{"qwen4_exp", "qwen4_exp_text", "qwen3_5", "qwen3_5_text",
    "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next"};
constexpr std::array<const char*, 7> dense_kinds{"qwen2", "qwen3", "qwen3_moe", "llama", "mistral", "minicpm", "minicpmo"};

bool one_of(const std::string& value, std::initializer_list<const char*> names) {
    return std::any_of(names.begin(), names.end(), [&](const char* name) { return value == name; });
}

bool one_of(const std::string& value, const std::array<const char*, 7>& names) {
    return std::any_of(names.begin(), names.end(), [&](const char* name) { return value == name; });
}

json text_config(const json& config) {
    const auto found = config.find("text_config");
    return found == config.end() ? config : *found;
}

/** Reads a non-negative integer, rejecting booleans and fractional values. */
std::int64_t integer(const json& text, const char* key) {
    const auto found = text.find(key);
    if (found == text.end() || !found->is_number_integer()) throw Unsupported{};
    return found->get<std::int64_t>();
}

std::int64_t integer(const json& text, const char* key, std::int64_t fallback) {
    const auto found = text.find(key);
    if (found == text.end()) return fallback;
    if (!found->is_number_integer()) throw Unsupported{};
    return found->get<std::int64_t>();
}

bool truthy(const json& value) {
    if (value.is_null() || value.is_boolean()) return value.is_boolean() && value.get<bool>();
    if (value.is_number_integer()) return value.get<std::int64_t>() != 0;
    if (value.is_number()) return value.get<double>() != 0;
    if (value.is_string()) return !value.get_ref<const std::string&>().empty();
    return !value.empty();
}

json fixed_component(const std::string& group, const std::string& name, std::int64_t layers, std::int64_t bytes) {
    if (!layers || !bytes) return nullptr;
    return {{"group", group}, {"name", name}, {"layers", layers}, {"bytes", bytes}};
}

json cache_component(const std::string& group, const std::string& name, std::int64_t layers, std::int64_t bytes_per_row) {
    return {{"group", group}, {"name", name}, {"subgroup", nullptr}, {"layers", layers},
        {"bytes_per_row", bytes_per_row}, {"tokens_per_row", 1}, {"allocation", "exact"}, {"minimum_rows", 0},
        {"row_rounding", "ceil"}, {"active_after", 0}, {"max_read_rows_per_token", nullptr},
        {"head_dimension", nullptr}, {"kv_heads", nullptr}};
}

json cache_profile(const json& config, bool include_recurrent, std::int64_t predictor_layers, bool cuda) {
    const auto text = text_config(config);
    if (!text.is_object()) return nullptr;
    std::string kind;
    if (const auto found = text.find("model_type"); found != text.end() && found->is_string()) kind = found->get<std::string>();
    else if (const auto found = config.find("model_type"); found != config.end() && found->is_string()) kind = found->get<std::string>();
    const auto maximum = integer(text, "max_position_embeddings");
    const auto layers = integer(text, "num_hidden_layers");
    if (std::min(maximum, layers) <= 0) return nullptr;
    json fixed = json::array(), components = json::array();
    const auto state = [&](const std::string& group, const std::string& name, std::int64_t count, std::int64_t bytes) {
        if (auto entry = fixed_component(group, name, count, bytes)) fixed.push_back(entry);
    };
    if (kind == "deepseek_v4") {
        const auto dim = integer(text, "head_dim"), index = integer(text, "index_head_dim"), window = integer(text, "sliding_window");
        const auto stored = text.find("compress_ratios");
        if (std::min({dim, index, window}) <= 0 || stored == text.end() || !stored->is_array()
            || stored->size() < static_cast<std::size_t>(layers)) return nullptr;
        std::vector<int> ratios;
        for (std::size_t position = 0; position < static_cast<std::size_t>(layers); ++position)
            ratios.push_back((*stored)[position].is_number_integer() ? (*stored)[position].get<int>() : -1);
        state("MLA", "sliding_window", layers, layers * window * dim * 2);
        for (const auto ratio : {4, 128}) {
            const std::int64_t count = std::count(ratios.begin(), ratios.end(), ratio);
            const auto width = ratio == 4 ? dim + index : dim;
            state("MLA", "compression_state_" + std::to_string(ratio), count, count * ratio * width * (ratio == 4 ? 12 : 4));
            if (!count) continue;
            auto compressed = cache_component("MLA", "compressed_kv", count, count * dim * 2);
            compressed["tokens_per_row"] = ratio;
            components.push_back(std::move(compressed));
            if (ratio != 4) continue;
            auto indexer = cache_component("MLA", "indexer_key", count, count * index * 2);
            indexer["subgroup"] = "indexer";
            indexer["tokens_per_row"] = ratio;
            components.push_back(std::move(indexer));
        }
        for (const auto ratio : ratios)
            if (ratio != 0 && ratio != 4 && ratio != 128) return nullptr;
    } else if (std::any_of(qwen4_kinds.begin(), qwen4_kinds.end(), [&](const char* name) { return kind == name; })) {
        const bool qwen4 = kind == "qwen4_exp" || kind == "qwen4_exp_text";
        json types = json::array();
        if (const auto found = text.find("layer_types"); found != text.end() && found->is_array()) types = *found;
        else {
            const auto interval = integer(text, "full_attention_interval", 4);
            if (interval <= 0) return nullptr;
            for (std::int64_t layer = 0; layer < layers; ++layer)
                types.push_back((layer + 1) % interval == 0 ? "full_attention" : "linear_attention");
        }
        if (types.size() != static_cast<std::size_t>(layers)) return nullptr;
        std::int64_t linear = 0, full = predictor_layers;
        for (const auto& value : types) {
            if (value == "linear_attention") ++linear;
            else if (value == "full_attention") ++full;
            else return nullptr;
        }
        if (linear && include_recurrent) {
            const auto keys = integer(text, "linear_num_key_heads"), key_dim = integer(text, "linear_key_head_dim");
            const auto heads = integer(text, "linear_num_value_heads"), value_dim = integer(text, "linear_value_head_dim");
            const auto kernel = integer(text, "linear_conv_kernel_dim");
            if (std::min({keys, key_dim, heads, value_dim, kernel}) <= 0 || key_dim != value_dim) return nullptr;
            const auto channels = 2 * keys * key_dim + heads * value_dim;
            const auto recurrent = linear * 4 * heads * value_dim * value_dim;
            const auto convolution = linear * 4 * (kernel - 1) * channels;
            state("GDN", "recurrent_state", linear, recurrent);
            state("GDN", "convolution_state", linear, convolution);
            // Non-Qwen4 GDN keeps two state copies, so the second one is a fixed template.
            state("GDN", "reset_template", linear, qwen4 ? 0 : recurrent + convolution);
        }
        if (full) {
            const auto heads = integer(text, "num_key_value_heads"), dim = integer(text, "head_dim");
            if (std::min(heads, dim) <= 0) return nullptr;
            auto raw = cache_component(qwen4 ? "QSA" : "GQA", "raw_kv", full, full * heads * dim * 4);
            raw["allocation"] = "power_of_two";
            raw["minimum_rows"] = 16;
            if (qwen4 && text.contains("indexer_budget")) raw["max_read_rows_per_token"] = integer(text, "indexer_budget");
            raw["head_dimension"] = dim;
            raw["kv_heads"] = heads;
            components.push_back(std::move(raw));
        }
        if (qwen4) {
            const auto index = integer(text, "indexer_head_dim"), ratio = integer(text, "indexer_compress_ratio");
            if (std::min(index, ratio) <= 0) return nullptr;
            if (full && cuda) {
                auto key = cache_component("QSA", "indexer_key", full, full * index * 2);
                key["subgroup"] = "indexer";
                components.push_back(std::move(key));
                auto pooled = cache_component("QSA", "indexer_pooled", full, full * index * 2);
                pooled["subgroup"] = "indexer";
                pooled["tokens_per_row"] = ratio;
                components.push_back(std::move(pooled));
            } else if (full) {
                state("QSA", "indexer_tail", full, full * ((ratio + 5) * index * 2 + 2));
                auto pooled = cache_component("QSA", "indexer_pooled", full, full * index * 4);
                pooled["subgroup"] = "indexer";
                pooled["tokens_per_row"] = ratio;
                pooled["allocation"] = "power_of_two";
                pooled["minimum_rows"] = 16;
                pooled["row_rounding"] = "floor";
                components.push_back(std::move(pooled));
            }
            json ple_layers = json::array();
            if (const auto found = text.find("ple_layer_ids"); found != text.end()) {
                if (!found->is_array()) return nullptr;
                ple_layers = *found;
            }
            if (!ple_layers.empty() && include_recurrent) {
                const auto kernel = integer(text, "ple_conv_kernel_size"), ngram = integer(text, "ngram_size");
                const auto hc = integer(text, "hc_count"), hidden = integer(text, "hidden_size");
                if (std::min({kernel, ngram, hc, hidden}) <= 0) return nullptr;
                const auto count = static_cast<std::int64_t>(ple_layers.size());
                state("PLE", "convolution_state", count, count * 4 * (kernel - 1) * ngram * hc * hidden);
            }
        }
    } else if (std::any_of(dense_kinds.begin(), dense_kinds.end(), [&](const char* name) { return kind == name; })) {
        const auto heads = integer(text, "num_key_value_heads", integer(text, "num_attention_heads"));
        const auto dim = integer(text, "head_dim", integer(text, "hidden_size") / integer(text, "num_attention_heads"));
        if (std::min(heads, dim) <= 0) return nullptr;
        if (const auto found = text.find("sliding_window"); found != text.end() && truthy(*found)) return nullptr;
        const auto query_heads = integer(text, "num_attention_heads");
        auto raw = cache_component(query_heads > heads ? "GQA" : "MHA", "raw_kv", layers, layers * heads * dim * 4);
        raw["allocation"] = "power_of_two";
        raw["minimum_rows"] = 16;
        raw["head_dimension"] = dim;
        raw["kv_heads"] = heads;
        components.push_back(std::move(raw));
    } else return nullptr;
    std::int64_t fixed_bytes = 0;
    for (const auto& entry : fixed) fixed_bytes += entry.at("bytes").get<std::int64_t>();
    return {{"max_context", maximum}, {"fixed_bytes", fixed_bytes},
        {"fixed_components", std::move(fixed)}, {"components", std::move(components)}};
}

std::int64_t predictor_layers(const json& config, bool has_predictor) {
    if (!has_predictor) return 0;
    const auto text = text_config(config);
    if (!text.is_object()) return 0;
    const auto found = text.find("mtp_num_hidden_layers");
    if (found == text.end()) return 1;
    if (!found->is_number_integer() || found->get<std::int64_t>() < 0) throw Unsupported{};
    return found->get<std::int64_t>();
}

[[noreturn]] void unavailable(const char* code, const char* message) {
    throw ApiError(422, code, message);
}

std::shared_ptr<const mfq::ModelSource> open(const ModelCatalog::Artifact& artifact, const char* code) {
    try {
        const auto source = mfq::open_model_source(artifact.path);
        if (!source) throw std::runtime_error("no model source");
        return source;
    } catch (const std::exception&) { unavailable(code, "checkpoint metadata cannot be read"); }
}

json source_config(const mfq::ModelSource& source, const char* code) {
    if (!source.has_asset(mfq::kModelConfigAsset)) return json::object();
    try {
        const auto config = json::parse(source.model_config_json());
        return config.is_object() ? config : json::object();
    } catch (const std::exception&) { unavailable(code, "checkpoint metadata cannot be read"); }
}

bool has_predictor(const mfq::ModelSource& source) {
    for (const auto& tensor : source.tensors())
        if (tensor.name.rfind("predictor.block.", 0) == 0 || tensor.name.rfind("mtp.", 0) == 0) return true;
    return false;
}

// shortcut: analysis reads logical shapes from the model source. A container that stores no
// shape metadata decodes the geometry from the tensor payload header, but expert byte counts
// stay evenly split instead of using the per-row storage the Python inspector reports.
struct Tensor {
    std::string name, category, projection, format;
    std::vector<std::int64_t> shape;
    std::optional<std::int64_t> layer;
    std::int64_t parameters = 0, stored_bytes = 0;
};

struct Expert {
    std::int64_t layer = 0, expert = 0, parameters = 0;
    double stored_bytes = 0;
    std::string projection, format, granularity;
};

bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::uint64_t little(const std::vector<std::byte>& bytes, std::size_t offset, std::size_t width) {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < width; ++index)
        value |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(bytes[offset + index])) << (8 * index);
    return value;
}

json logical_shapes(const mfq::ModelSource& source) {
    const auto& metadata = source.metadata();
    for (const auto* key : {"tq_shapes", "tensor_shapes"}) {
        const auto found = metadata.find(key);
        if (found == metadata.end()) continue;
        try {
            auto value = json::parse(found->second);
            if (value.is_string()) value = json::parse(value.get<std::string>());
            return value.is_object() ? value : json::object();
        } catch (const std::exception&) { return json::object(); }
    }
    return json::object();
}

/** Recovers a tensor shape from the payload header of containers that store none of their own. */
std::vector<std::int64_t> payload_shape(const mfq::ModelSource& source, const mfq::TensorMetadata& tensor,
        const json& shapes) {
    const auto logical = [&] {
        const auto found = shapes.find(tensor.name);
        if (found == shapes.end() || !found->is_array()) return std::vector<std::int64_t>{};
        std::vector<std::int64_t> shape;
        for (const auto& axis : *found) {
            if (!axis.is_number_integer() || axis.get<std::int64_t>() <= 0) return std::vector<std::int64_t>{};
            shape.push_back(axis.get<std::int64_t>());
        }
        return shape;
    }();
    std::vector<std::byte> header;
    try { header = source.read(tensor.name, 0, std::min<std::uint64_t>(tensor.nbytes, 132)); } catch (...) { return {}; }
    if (header.size() < 8) return {};
    const auto magic = std::string(reinterpret_cast<const char*>(header.data()), header.size());
    if (magic.rfind("RTQEXP01", 0) == 0 && header.size() >= 32)
        return {static_cast<std::int64_t>(little(header, 12, 4)), static_cast<std::int64_t>(little(header, 16, 4)),
            static_cast<std::int64_t>(little(header, 20, 4))};
    if ((magic.rfind("RTRELL06", 0) == 0 || magic.rfind("RTRELL05", 0) == 0) && header.size() >= 32) {
        if (!logical.empty()) return logical;
        return {static_cast<std::int64_t>(little(header, 12, 4)), static_cast<std::int64_t>(little(header, 16, 4))};
    }
    if (magic.rfind("MXT1", 0) == 0 && header.size() >= 24) {
        if (!logical.empty()) return logical;
        return {static_cast<std::int64_t>(little(header, 8, 8)), static_cast<std::int64_t>(little(header, 16, 8))};
    }
    static const std::map<std::string, std::size_t> dense{{"BF16", 2}, {"F16", 2}, {"F32", 4}, {"F8_E4M3", 1}};
    const auto kind = dense.find(tensor.dtype);
    if (kind == dense.end() || header.size() < 4) return {};
    const auto rank = little(header, 0, 4);
    if (rank == 0 || rank > 16 || header.size() < 4 + 8 * rank) return {};
    std::vector<std::int64_t> shape;
    std::uint64_t elements = 1;
    for (std::size_t index = 0; index < rank; ++index) {
        const auto axis = little(header, 4 + 8 * index, 8);
        if (axis == 0 || axis > 9223372036854775807ULL) return {};
        elements *= axis;
        shape.push_back(static_cast<std::int64_t>(axis));
    }
    if (4 + 8 * rank + elements * kind->second != tensor.nbytes) return {};
    return shape;
}

std::string remove_suffix(std::string value, const std::string& suffix) {
    if (ends_with(value, suffix)) value.resize(value.size() - suffix.size());
    return value;
}

bool always_streamed(const std::string& name) {
    static const std::regex pattern(R"((?:\.associative_memory\.embedding\.weight|\.engram\.embed\.weight|\.position_embedding\.ngram\.shard\.[0-9]+\.(?:weight|scales|biases))$)");
    return std::regex_search(name, pattern);
}

/** Mirrors the Python classifier that turns storage names into analysis categories. */
void classify(Tensor& tensor) {
    static const std::regex layer(R"(^model\.block\.([0-9]+)\.)");
    static const std::regex expert(R"(\.experts\.(?:([0-9]+)\.)?(gate_up|gate|up|down)(?:_proj)?(?:\.weight)?$)");
    static const std::regex shard(R"(\.shard\.[0-9]+$)");
    const auto& name = tensor.name;
    std::smatch match;
    const bool prefixed = std::regex_search(name, match, layer);
    if (prefixed) tensor.layer = std::stoll(match[1].str());
    const auto suffix = prefixed ? name.substr(match[0].length()) : name;
    if (name.rfind("predictor.", 0) == 0) tensor.category = "predictor";
    else if (name.rfind("model.", 0) != 0) tensor.category = name.substr(0, name.find('.'));
    else if (always_streamed(name)) tensor.category = "ple";
    else if (name.find(".mhc.") != std::string::npos) tensor.category = "mhc";
    else if (name.find(".mlp.experts.") != std::string::npos) tensor.category = "routed_experts";
    else if (name.find(".mlp.shared_expert.") != std::string::npos) tensor.category = "shared_expert";
    else if (name.find(".linear_attention.") != std::string::npos) tensor.category = "gdn";
    else if (name.find(".attention.") != std::string::npos) tensor.category = "attention";
    else if (name.find(".position_embedding.") != std::string::npos || name.find(".associative_memory.") != std::string::npos
        || name.find(".engram.") != std::string::npos) tensor.category = "ple_projection";
    else if (name.find(".mlp.router.") != std::string::npos) tensor.category = "router";
    else if (name.find(".mlp.") != std::string::npos) tensor.category = "ffn";
    else if (suffix.find("embedding") != std::string::npos || suffix.find("embed_tokens") != std::string::npos) tensor.category = "embedding";
    else if (suffix.find("output.weight") != std::string::npos || suffix.find("lm_head") != std::string::npos) tensor.category = "lm_head";
    else tensor.category = "other";
    if (tensor.category == "routed_experts") {
        tensor.projection = std::regex_search(name, match, expert) ? match[2].str() : suffix;
        return;
    }
    auto projection = suffix;
    if (ends_with(projection, ".weight")) projection.resize(projection.size() - 7);
    else if (ends_with(projection, ".bias")) projection.resize(projection.size() - 5);
    for (const auto* prefix : {"linear_attention.", "attention.", "mlp.shared_expert.", "mlp.", "model."})
        if (projection.rfind(prefix, 0) == 0) { projection = projection.substr(std::string(prefix).size()); break; }
    tensor.projection = std::regex_replace(projection, shard, "");
}

std::optional<std::int64_t> expert_index(const std::string& name) {
    static const std::regex pattern(R"(\.experts\.([0-9]+)\.(?:gate_up|gate|up|down)(?:_proj)?(?:\.weight)?$)");
    std::smatch match;
    if (!std::regex_search(name, match, pattern)) return {};
    return std::stoll(match[1].str());
}

std::int64_t product(const std::vector<std::int64_t>& shape) {
    std::int64_t result = 1;
    for (const auto axis : shape) result *= axis;
    return result;
}

std::optional<std::int64_t> config_integer(const json& text, std::initializer_list<const char*> keys) {
    if (!text.is_object()) return {};
    for (const auto* key : keys) {
        const auto found = text.find(key);
        if (found != text.end() && found->is_number_integer() && found->get<std::int64_t>() > 0) return found->get<std::int64_t>();
    }
    return {};
}

std::optional<double> config_number(const json& text, std::initializer_list<const char*> keys) {
    if (!text.is_object()) return {};
    for (const auto* key : keys) {
        const auto found = text.find(key);
        if (found != text.end() && found->is_number() && !found->is_boolean() && found->get<double>() > 0
            && std::isfinite(found->get<double>())) return found->get<double>();
    }
    return {};
}

std::optional<std::string> config_string(const json& text, std::initializer_list<const char*> keys) {
    if (!text.is_object()) return {};
    for (const auto* key : keys) {
        const auto found = text.find(key);
        if (found == text.end() || !found->is_string()) continue;
        const auto& value = found->get_ref<const std::string&>();
        const auto begin = value.find_first_not_of(" \t\n");
        if (begin == std::string::npos) continue;
        return value.substr(begin, value.find_last_not_of(" \t\n") - begin + 1);
    }
    return {};
}

template <typename T> json optional_json(const std::optional<T>& value) {
    return value ? json(*value) : json(nullptr);
}

json or_else(const std::optional<std::string>& value, const std::optional<std::string>& fallback) {
    return value ? json(*value) : optional_json(fallback);
}

struct Normalization {
    std::string component, position;
    std::optional<std::string> kind;
    std::optional<std::int64_t> dimension, groups;
    std::optional<double> epsilon;
    std::vector<std::int64_t> layers;
};

json advanced(const json& text, const std::vector<Tensor>& tensors) {
    const auto width = [&](const std::string& category) -> std::optional<std::int64_t> {
        for (const auto& tensor : tensors) {
            const auto projection = remove_suffix(tensor.projection, "_proj");
            if (tensor.category != category || !one_of(projection, {"gate", "up", "gate_up"}) || tensor.shape.size() < 2) continue;
            const auto rows = tensor.shape[tensor.shape.size() - 2];
            if (projection != "gate_up" || rows % 2 == 0) return rows / (projection == "gate_up" ? 2 : 1);
        }
        return {};
    };
    std::vector<const Tensor*> shared;
    for (const auto& tensor : tensors) if (tensor.category == "shared_expert") shared.push_back(&tensor);
    std::optional<std::int64_t> shared_count;
    if (shared.empty()) shared_count = 0;
    else {
        shared_count = config_integer(text, {"n_shared_experts", "num_shared_experts"});
        if (!shared_count) {
            shared_count = 1;
            for (const auto* tensor : shared) if (tensor->shape.size() >= 3) { shared_count = tensor->shape[tensor->shape.size() - 3]; break; }
        }
    }
    auto shared_width = config_integer(text, {"shared_expert_intermediate_size"});
    if (!shared_width) shared_width = width("shared_expert");
    if (shared_width && shared_count && *shared_count > 0)
        shared_width = *shared_width % *shared_count == 0 ? std::optional(*shared_width / *shared_count) : std::nullopt;
    const auto hc_count = config_integer(text, {"hc_count", "hc_mult"});
    const auto hidden = config_integer(text, {"hidden_size"});
    std::string architecture;
    if (const auto found = text.find("model_type"); found != text.end() && found->is_string()) architecture = found->get<std::string>();
    const bool qwen4 = architecture == "qwen4_exp" || architecture == "qwen4_exp_text";
    const bool qwen35 = one_of(architecture, {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next"});
    const auto any_category = [&](std::initializer_list<const char*> names) {
        return std::any_of(tensors.begin(), tensors.end(),
            [&](const Tensor& tensor) { return one_of(tensor.category, names); });
    };
    const auto contains = [&](const char* needle) {
        return std::any_of(tensors.begin(), tensors.end(),
            [&](const Tensor& tensor) { return tensor.name.find(needle) != std::string::npos; });
    };
    const bool has_gdn = any_category({"gdn"}), has_ple = any_category({"ple", "ple_projection"});
    const bool has_predictor = any_category({"predictor"}), has_qsa = contains(".attention.indexer.");
    const auto attention_heads = config_integer(text, {"num_attention_heads"}), head_dim = config_integer(text, {"head_dim"});
    bool has_full_gate = false;
    for (const auto& tensor : tensors) {
        if (tensor.category != "attention") continue;
        if (tensor.projection == "gate") { has_full_gate = true; break; }
        if (tensor.projection == "query" && tensor.shape.size() >= 2 && attention_heads && head_dim
            && tensor.shape[tensor.shape.size() - 2] == 2 * *attention_heads * *head_dim) { has_full_gate = true; break; }
    }
    const auto stride = config_integer(text, {"indexer_compress_ratio"}), budget = config_integer(text, {"indexer_budget"});
    json ratios = json::array();
    if (const auto found = text.find("compress_ratios"); found != text.end() && found->is_array()) ratios = *found;
    json rope = json::object();
    if (const auto found = text.find("rope_parameters"); found != text.end() && found->is_object()) rope = *found;
    const auto norm_eps = config_number(text, {"rms_norm_eps", "norm_eps", "rms_eps"});
    std::vector<std::int64_t> sparse_layers, gdn_layers;
    std::set<std::string> tensor_names;
    for (const auto& tensor : tensors) {
        tensor_names.insert(tensor.name);
        if (tensor.layer && tensor.name.find(".attention.indexer.") != std::string::npos) sparse_layers.push_back(*tensor.layer);
        if (tensor.layer && tensor.category == "gdn") gdn_layers.push_back(*tensor.layer);
    }
    const bool known_rms = qwen4 || qwen35 || architecture.rfind("deepseek", 0) == 0
        || one_of(architecture, {"qwen2", "qwen2_moe", "qwen3", "qwen3_moe", "llama", "mistral", "mixtral"});
    const auto layer_in = [](const std::vector<std::int64_t>& layers, const std::optional<std::int64_t>& layer) {
        return layer && std::find(layers.begin(), layers.end(), *layer) != layers.end();
    };
    std::vector<Normalization> norms;
    const auto norm = [&](const std::string& component, const std::string& position, const std::optional<std::string>& kind,
            const std::optional<std::int64_t>& dimension, const std::optional<double>& epsilon,
            const std::optional<std::int64_t>& groups, const std::optional<std::int64_t>& layer) {
        auto found = std::find_if(norms.begin(), norms.end(), [&](const Normalization& item) {
            return item.component == component && item.position == position && item.kind == kind
                && item.dimension == dimension && item.epsilon == epsilon && item.groups == groups;
        });
        if (found == norms.end())
            found = norms.insert(norms.end(), Normalization{component, position, kind, dimension, groups, epsilon, {}});
        if (layer && std::find(found->layers.begin(), found->layers.end(), *layer) == found->layers.end()) found->layers.push_back(*layer);
    };
    static const std::regex norm_weight(R"((?:^|\.)(?:\w*norm)\.weight$)");
    static const std::regex head_norm(R"(\.attention\.(indexer\.)?(query|key)_norm\.weight$)");
    for (const auto& tensor : tensors) {
        if (!std::regex_search(tensor.name, norm_weight) || tensor.shape.size() != 1) continue;
        const auto& name = tensor.name;
        std::string component = "other", position = remove_suffix(name, ".weight");
        std::optional<std::string> kind = known_rms ? std::optional<std::string>("RMSNorm") : std::nullopt;
        std::optional<std::int64_t> groups, dimension = static_cast<std::int64_t>(tensor.shape[0]);
        auto epsilon = norm_eps;
        const auto grouped = [&] { return qwen4 && hc_count && hidden && dimension == *hc_count * *hidden; };
        const auto group_norm = [&] { kind = "GroupedRMSNorm"; groups = hc_count; dimension = hidden; };
        if (name.find(".mhc.") != std::string::npos) {
            component = "residual";
            position = name.find(".mlp.") != std::string::npos ? "ffn_input"
                : name.find(".attention.") != std::string::npos ? "attention_input" : "final_readout";
            if (grouped()) group_norm();
        } else if (tensor.category == "gdn") { component = "gdn"; position = "output"; }
        else if (tensor.category == "attention") {
            component = name.find(".indexer.") != std::string::npos ? "indexer"
                : layer_in(sparse_layers, tensor.layer) && qwen4 ? "qsa" : "attention";
            const auto tail = name.substr(name.find(".attention.") + 11);
            auto place = tail.rfind("indexer.", 0) == 0 ? tail.substr(8) : tail;
            place = remove_suffix(remove_suffix(place, ".weight"), "_norm");
            position = place == "norm" ? "input" : place;
            if (component == "attention" && position == "input" && layer_in(gdn_layers, tensor.layer)) component = "gdn";
            if (component == "indexer" && !qwen4) {
                kind = tensor_names.count(remove_suffix(name, ".weight") + ".bias") ? std::optional<std::string>("LayerNorm") : std::nullopt;
                epsilon = kind == "LayerNorm" ? config_number(text, {"layer_norm_eps"}) : std::nullopt;
            }
        } else if (tensor.category == "ple_projection") {
            component = "ple";
            const auto last = name.rfind('.');
            const auto previous = last == std::string::npos ? std::string::npos : name.rfind('.', last - 1);
            position = previous == std::string::npos ? name : remove_suffix(name.substr(previous + 1, last - previous - 1), "_norm");
            if (grouped()) group_norm();
        } else if (tensor.category == "ffn") { component = "ffn"; position = "input"; }
        else if (name == "model.output_norm.weight") { component = "output"; position = "output"; }
        if (qwen4 || qwen35) {
            std::smatch match;
            if (std::regex_search(name, match, head_norm)) {
                const bool indexer = match[1].matched, query = match[2].str() == "query";
                kind = "GroupedRMSNorm";
                groups = config_integer(text, indexer && query ? std::initializer_list<const char*>{"indexer_n_heads", "index_n_heads"}
                    : indexer ? std::initializer_list<const char*>{"indexer_kv_heads"}
                    : query ? std::initializer_list<const char*>{"num_attention_heads"}
                    : std::initializer_list<const char*>{"num_key_value_heads"});
            } else if (ends_with(name, ".linear_attention.norm.weight")) {
                kind = "GroupedRMSNorm";
                groups = config_integer(text, {"linear_num_value_heads"});
            }
        }
        if (tensor.category == "predictor") {
            component = "predictor";
            if (name.find(".mhc.") == std::string::npos)
                position = remove_suffix(name.rfind("predictor.", 0) == 0 ? name.substr(10) : name, ".weight");
        }
        if (component != "other") norm(component, position, kind, dimension, epsilon, groups, tensor.layer);
    }
    if (has_gdn && (qwen4 || qwen35)) {
        std::sort(gdn_layers.begin(), gdn_layers.end());
        gdn_layers.erase(std::unique(gdn_layers.begin(), gdn_layers.end()), gdn_layers.end());
        for (const auto layer : gdn_layers)
            for (const auto* position : {"query", "key"})
                norm("gdn", position, "L2Norm", config_integer(text, {"linear_key_head_dim"}), 1e-6,
                    config_integer(text, {"linear_num_key_heads"}), layer);
    }
    for (auto& entry : norms) std::sort(entry.layers.begin(), entry.layers.end());
    std::optional<bool> topk_norm;
    if (const auto found = text.find("norm_topk_prob"); found != text.end()) { if (found->is_boolean()) topk_norm = found->get<bool>(); }
    else if (qwen4) topk_norm = true;
    json normalizations = json::array();
    for (const auto& entry : norms)
        normalizations.push_back({{"component", entry.component}, {"position", entry.position},
            {"kind", optional_json(entry.kind)}, {"dimension", optional_json(entry.dimension)},
            {"epsilon", optional_json(entry.epsilon)}, {"groups", optional_json(entry.groups)}, {"layers", entry.layers}});
    json compression_strides = json::array();
    for (const auto& value : ratios)
        if (value.is_number_integer() && value.get<std::int64_t>() > 0) compression_strides.push_back(value);
    std::vector<std::int64_t> ple_layer_ids;
    for (const auto& tensor : tensors)
        if ((tensor.category == "ple" || tensor.category == "ple_projection") && tensor.layer
            && std::find(ple_layer_ids.begin(), ple_layer_ids.end(), *tensor.layer) == ple_layer_ids.end()) ple_layer_ids.push_back(*tensor.layer);
    std::sort(ple_layer_ids.begin(), ple_layer_ids.end());
    std::optional<std::int64_t> predictor_count = has_predictor ? std::optional<std::int64_t>{} : std::optional<std::int64_t>{0};
    if (has_predictor) predictor_count = config_integer(text, {"mtp_num_hidden_layers"});
    std::optional<std::int64_t> active_blocks = budget && stride && *budget % *stride == 0
        ? std::optional(*budget / *stride) : config_integer(text, {"index_topk"});
    const bool gdn_family = has_gdn && (qwen4 || qwen35);
    const auto hidden_act = config_string(text, {"hidden_act"});
    const auto scoring = config_string(text, {"scoring_func", "expert_gating_func"});
    const auto gate_type = config_string(text, {"output_gate_type", "hidden_act"});
    const auto ffn_hidden = config_integer(text, {"intermediate_size"});
    const auto expert_hidden = config_integer(text, {"moe_intermediate_size"});
    const auto rope_theta = config_number(rope, {"rope_theta"});
    const auto partial = config_number(text, {"partial_rotary_factor"});
    return {
        {"ffn_hidden_size", optional_json(ffn_hidden ? ffn_hidden : width("ffn"))},
        {"expert_hidden_size", optional_json(expert_hidden ? expert_hidden : width("routed_experts"))},
        {"shared_expert_count", optional_json(shared_count)}, {"shared_expert_hidden_size", optional_json(shared_width)},
        {"hc_count", hc_count && *hc_count > 1 ? json(*hc_count) : json(nullptr)},
        {"hc_dim", hc_count && *hc_count > 1 ? optional_json(hidden) : json(nullptr)},
        {"hc_lowrank", optional_json(config_integer(text, {"hc_lowrank", "hc_rank"}))},
        {"ffn_activation", optional_json(hidden_act)},
        {"router_activation", qwen4 ? json("softmax") : scoring ? json(*scoring)
            : qwen35 ? json("softmax") : json(nullptr)},
        {"router_normalization", topk_norm ? json(*topk_norm ? "l1_topk" : "none") : json(nullptr)},
        {"gdn_conv_activation", gdn_family ? or_else(hidden_act, "silu") : json(nullptr)},
        {"gdn_conv_type", gdn_family ? json("causal_depthwise_1d") : json(nullptr)},
        {"gdn_conv_stride", gdn_family ? json(1) : json(nullptr)},
        {"gdn_conv_dilation", gdn_family ? json(1) : json(nullptr)},
        {"gdn_gate_activation", has_gdn && qwen4 ? or_else(gate_type, std::optional<std::string>("silu"))
            : has_gdn && qwen35 ? json("silu") : json(nullptr)},
        {"qsa_gate_activation", qwen4 && has_qsa ? json("sigmoid") : json(nullptr)},
        {"full_attention_gate_activation", qwen35 && has_full_gate ? json("sigmoid") : json(nullptr)},
        {"residual_gate_activation", hc_count && *hc_count > 1 && (qwen4 || architecture.rfind("deepseek_v4", 0) == 0)
            ? json("sigmoid") : json(nullptr)},
        {"indexer_query_heads", optional_json(config_integer(text, {"indexer_n_heads", "index_n_heads"}))},
        {"indexer_kv_heads", optional_json(config_integer(text, {"indexer_kv_heads"}))},
        {"indexer_head_dim", optional_json(config_integer(text, {"indexer_head_dim", "index_head_dim"}))},
        {"compression_stride", optional_json(stride)}, {"compression_strides", std::move(compression_strides)},
        {"active_tokens", optional_json(budget)}, {"active_blocks", optional_json(active_blocks)},
        {"linear_key_head_dim", optional_json(config_integer(text, {"linear_key_head_dim"}))},
        {"linear_value_head_dim", optional_json(config_integer(text, {"linear_value_head_dim"}))},
        {"linear_conv_kernel", optional_json(config_integer(text, {"linear_conv_kernel_dim"}))},
        {"ple_ngram", optional_json(config_integer(text, {"ngram_size"}))},
        {"ple_heads", optional_json(config_integer(text, {"heads_per_ngram"}))},
        {"ple_conv_kernel", optional_json(config_integer(text, {"ple_conv_kernel_size"}))},
        {"ple_conv_type", has_ple && qwen4 ? json("causal_depthwise_1d") : json(nullptr)},
        {"ple_conv_stride", has_ple && qwen4 ? json(1) : json(nullptr)},
        {"ple_conv_dilation", has_ple && qwen4 ? optional_json(config_integer(text, {"ngram_size"})) : json(nullptr)},
        {"ple_conv_activation", has_ple && qwen4 ? json("silu") : json(nullptr)},
        {"ple_layer_ids", ple_layer_ids},
        {"predictor_layers", optional_json(predictor_count)},
        {"max_context", optional_json(config_integer(text, {"max_position_embeddings"}))},
        {"vocab_size", optional_json(config_integer(text, {"vocab_size"}))},
        {"rope_theta", optional_json(rope_theta ? rope_theta : config_number(text, {"rope_theta"}))},
        {"rope_partial_factor", optional_json(partial ? partial : config_number(rope, {"partial_rotary_factor"}))},
        {"rms_norm_eps", optional_json(norm_eps)}, {"normalizations", std::move(normalizations)},
    };
}

} // namespace

json model_cache_profile_json(const ModelCatalog::Artifact& artifact, bool include_recurrent, bool cuda) {
    constexpr const char* code = "model_cache_profile_unavailable";
    if (!artifact.complete) throw ApiError(422, code, "checkpoint cache metadata cannot be read");
    const auto source = open(artifact, code);
    const auto config = source_config(*source, code);
    try {
        return cache_profile(config, include_recurrent, predictor_layers(config, has_predictor(*source)), cuda);
    } catch (const Unsupported&) {
        return nullptr;
    }
}

json checkpoint_analysis_json(const ModelCatalog::Artifact& artifact, bool cuda) {
    constexpr const char* code = "checkpoint_analysis_unavailable";
    if (!artifact.complete) throw ApiError(422, code, "checkpoint is incomplete or its metadata cannot be read");
    const auto source = open(artifact, code);
    const auto config = source_config(*source, code);
    std::vector<Tensor> tensors;
    std::vector<Expert> experts;
    json warnings = json::array();
    const auto shapes = logical_shapes(*source);
    for (const auto& record : source->tensors()) {
        auto format = record.stored_dtype.empty() ? record.dtype : record.stored_dtype;
        if (format.rfind("GGUF_", 0) == 0) format = format.substr(5);
        if (one_of(format, {"I32", "I64", "U32", "U64", "U8", "I8", "BOOL"})) continue;
        const auto usable = record.shape && !record.shape->empty()
            && std::all_of(record.shape->begin(), record.shape->end(), [](std::int64_t axis) { return axis > 0; });
        auto shape = usable ? *record.shape : payload_shape(*source, record, shapes);
        if (shape.empty()) {
            warnings.push_back(record.name + ": logical tensor shape is unavailable");
            continue;
        }
        Tensor tensor;
        tensor.name = record.name;
        tensor.shape = std::move(shape);
        tensor.format = format;
        tensor.parameters = product(tensor.shape);
        tensor.stored_bytes = static_cast<std::int64_t>(record.nbytes);
        classify(tensor);
        tensors.push_back(tensor);
        if (tensor.category != "routed_experts" || !tensor.layer) continue;
        const auto index = expert_index(tensor.name);
        std::int64_t count = 0, first = 0, inner = 0;
        const auto rows = tensor.shape[tensor.shape.size() - 2];
        if (tensor.shape.size() == 3) { count = tensor.shape[0]; first = 0; inner = tensor.shape[tensor.shape.size() - 1]; }
        else if (tensor.shape.size() == 2 && index) { count = 1; first = *index; inner = tensor.shape[tensor.shape.size() - 1]; }
        else continue;
        const std::vector<std::string> parts = tensor.projection == "gate_up" ? std::vector<std::string>{"gate", "up"}
            : std::vector<std::string>{tensor.projection};
        if (rows % static_cast<std::int64_t>(parts.size())) {
            warnings.push_back(tensor.name + ": fused projection has an invalid row count");
            continue;
        }
        const auto parameters = rows * inner / static_cast<std::int64_t>(parts.size());
        const auto bytes = static_cast<double>(tensor.stored_bytes) / static_cast<double>(count * parts.size());
        for (std::int64_t position = 0; position < count; ++position)
            for (const auto& part : parts)
                experts.push_back(Expert{*tensor.layer, first + position, parameters, bytes, part, format, "tensor"});
    }
    json text = json::object();
    if (config.is_object()) {
        const auto found = config.find("text_config");
        text = found != config.end() && found->is_object() ? *found : config;
    }
    auto layer_count = config_integer(text, {"num_hidden_layers", "n_layer"}).value_or(0);
    for (const auto& tensor : tensors) if (tensor.layer) layer_count = std::max(layer_count, *tensor.layer + 1);
    json types = json::array();
    if (const auto found = text.find("layer_types"); found != text.end() && found->is_array()) types = *found;
    const auto query_heads = config_integer(text, {"num_attention_heads"}), kv_heads = config_integer(text, {"num_key_value_heads"});
    const std::string full_attention = query_heads && kv_heads && *query_heads > *kv_heads && *query_heads % *kv_heads == 0 ? "GQA"
        : query_heads && *query_heads == *kv_heads ? "MHA" : "Attention";
    std::map<std::string, std::vector<std::int64_t>> distribution;
    json layers = json::array();
    for (std::int64_t layer = 0; layer < layer_count; ++layer) {
        std::vector<const Tensor*> weights;
        for (const auto& tensor : tensors) if (tensor.layer && *tensor.layer == layer) weights.push_back(&tensor);
        const auto mentions = [&](const char* needle) {
            return std::any_of(weights.begin(), weights.end(),
                [&](const Tensor* tensor) { return tensor->name.find(needle) != std::string::npos; });
        };
        std::string kind = "unknown";
        if (mentions(".linear_attention.")) kind = "GDN";
        else if (mentions(".attention.indexer.")) kind = "QSA";
        else if (mentions(".attention.")) kind = config_integer(text, {"q_lora_rank", "kv_lora_rank"}) ? "MLA" : full_attention;
        else if (layer < static_cast<std::int64_t>(types.size()))
            kind = types[layer].is_string() ? types[layer].get<std::string>() : types[layer].dump();
        std::int64_t parameters = 0, stored = 0;
        for (const auto* tensor : weights) { parameters += tensor->parameters; stored += tensor->stored_bytes; }
        distribution[kind].push_back(layer);
        layers.push_back({{"layer", layer}, {"attention_type", kind}, {"parameters", parameters}, {"stored_bytes", stored},
            {"average_bpw", parameters ? json(8.0 * static_cast<double>(stored) / static_cast<double>(parameters)) : json(nullptr)}});
    }
    std::map<std::pair<std::string, std::string>, std::vector<const Tensor*>> groups;
    for (const auto& tensor : tensors) {
        const std::vector<std::string> parts = tensor.category == "routed_experts" && tensor.projection == "gate_up"
            ? std::vector<std::string>{"gate", "up"} : std::vector<std::string>{tensor.projection};
        for (const auto& part : parts) groups[{tensor.category, part}].push_back(&tensor);
    }
    json projections = json::array();
    for (const auto& [key, items] : groups) {
        const auto& projection = key.second;
        std::set<std::int64_t> ids;
        for (const auto* tensor : items) if (tensor->layer) ids.insert(*tensor->layer);
        std::vector<const Expert*> cells;
        for (const auto& expert : experts)
            if (expert.projection == projection && ids.count(expert.layer)) cells.push_back(&expert);
        const bool routed = key.first == "routed_experts" && !cells.empty();
        std::int64_t parameters = 0;
        double stored = 0;
        std::vector<std::string> formats;
        if (routed) {
            for (const auto* cell : cells) { parameters += cell->parameters; stored += cell->stored_bytes; formats.push_back(cell->format); }
        } else for (const auto* tensor : items) {
            const std::int64_t divisor = tensor->projection == "gate_up" ? 2 : 1;
            parameters += tensor->parameters / divisor;
            stored += static_cast<double>(tensor->stored_bytes) / static_cast<double>(divisor);
            formats.push_back(tensor->format);
        }
        std::sort(formats.begin(), formats.end());
        formats.erase(std::unique(formats.begin(), formats.end()), formats.end());
        projections.push_back({{"name", projection}, {"category", key.first}, {"parameters", parameters},
            {"stored_bytes", stored}, {"average_bpw", parameters ? json(8.0 * stored / static_cast<double>(parameters)) : json(nullptr)},
            {"tensor_count", static_cast<std::int64_t>(items.size())}, {"formats", formats}});
    }
    std::int64_t parameters = 0, stored = 0;
    double bpw_parameters = 0, bpw_stored = 0;
    std::int64_t expert_count = 0;
    for (const auto& tensor : tensors) {
        parameters += tensor.parameters;
        stored += tensor.stored_bytes;
        if (tensor.category != "predictor" && tensor.category != "vision") {
            bpw_parameters += tensor.parameters;
            bpw_stored += tensor.stored_bytes;
        }
    }
    for (const auto& expert : experts) expert_count = std::max(expert_count, expert.expert + 1);
    expert_count = std::max(config_integer(text, {"num_experts", "n_routed_experts"}).value_or(0), expert_count);
    if (config.empty()) warnings.push_back("checkpoint has no model config; head counts and architecture details are unavailable");
    json tensor_rows = json::array(), expert_rows = json::array(), graph = json::object();
    for (const auto& tensor : tensors)
        tensor_rows.push_back({{"name", tensor.name}, {"shape", tensor.shape}, {"category", tensor.category},
            {"projection", tensor.projection}, {"layer", optional_json(tensor.layer)}, {"parameters", tensor.parameters},
            {"stored_bytes", tensor.stored_bytes}, {"bpw", 8.0 * static_cast<double>(tensor.stored_bytes) / static_cast<double>(tensor.parameters)},
            {"format", tensor.format}});
    for (const auto& expert : experts)
        expert_rows.push_back({{"layer", expert.layer}, {"expert", expert.expert}, {"projection", expert.projection},
            {"parameters", expert.parameters}, {"stored_bytes", expert.stored_bytes},
            {"bpw", 8.0 * expert.stored_bytes / static_cast<double>(expert.parameters)},
            {"format", expert.format}, {"granularity", expert.granularity}});
    std::string architecture;
    if (const auto found = text.find("model_type"); found != text.end() && found->is_string()) architecture = found->get<std::string>();
    if (architecture.empty()) architecture = artifact.architecture;
    graph["hc_count"] = optional_json(config_integer(text, {"hc_count", "hc_mult"}));
    graph["has_ple"] = std::any_of(tensors.begin(), tensors.end(), [](const Tensor& tensor) { return tensor.category == "ple"; });
    graph["has_shared_expert"] = std::any_of(tensors.begin(), tensors.end(),
        [](const Tensor& tensor) { return tensor.category == "shared_expert"; });
    json profile_value = nullptr;
    try {
        profile_value = cache_profile(config, false, 0, cuda);
    } catch (const Unsupported&) {}
    return {
        {"model_id", artifact.id}, {"name", artifact.name}, {"architecture", architecture}, {"format", artifact.format},
        {"complete", artifact.complete}, {"layer_count", layer_count},
        {"hidden_size", optional_json(config_integer(text, {"hidden_size"}))},
        {"attention_heads", optional_json(query_heads)}, {"kv_heads", optional_json(kv_heads)},
        {"head_dim", optional_json(config_integer(text, {"head_dim"}))},
        {"linear_key_heads", optional_json(config_integer(text, {"linear_num_key_heads"}))},
        {"linear_value_heads", optional_json(config_integer(text, {"linear_num_value_heads"}))},
        {"expert_count", expert_count}, {"experts_per_token", optional_json(config_integer(text, {"num_experts_per_tok"}))},
        {"parameters", parameters}, {"stored_bytes", stored},
        {"average_bpw", bpw_parameters ? json(8.0 * bpw_stored / bpw_parameters) : json(nullptr)},
        {"attention_distribution", distribution}, {"projections", std::move(projections)},
        {"layers", std::move(layers)}, {"experts", std::move(expert_rows)}, {"tensors", std::move(tensor_rows)},
        {"graph", std::move(graph)}, {"advanced", advanced(text, tensors)}, {"cache_profile", std::move(profile_value)},
        {"warnings", std::move(warnings)},
    };
}

} // namespace mfq::server

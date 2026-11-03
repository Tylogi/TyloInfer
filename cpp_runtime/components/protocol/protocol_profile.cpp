#include "protocol.h"

#include "nlohmann/json.hpp"
#include "chat.h"
#include "mtp_metrics.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <aclapi.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mfq::protocol {
template <typename Value>
static void merge_optional(std::optional<Value> & target,
                           const std::optional<Value> & source) {
    if (source) target = source;
}
static void merge_runtime_profile(MfqRuntimeProfile & target,
                                  const MfqRuntimeProfile & source) {
#define TYLOINFER_MERGE(section, field) \
    merge_optional(target.section.field, source.section.field)
    TYLOINFER_MERGE(chat, max_tokens);
    TYLOINFER_MERGE(chat, temperature);
    TYLOINFER_MERGE(chat, top_k);
    TYLOINFER_MERGE(chat, top_p);
    TYLOINFER_MERGE(chat, presence_penalty);
    TYLOINFER_MERGE(chat, frequency_penalty);
    TYLOINFER_MERGE(chat, repetition_penalty);
    TYLOINFER_MERGE(chat, enable_thinking);
    TYLOINFER_MERGE(chat, enable_vision);
    TYLOINFER_MERGE(chat, enable_mtp);
    TYLOINFER_MERGE(chat, mtp_max_draft_tokens);
    TYLOINFER_MERGE(duplex, system_prompt);
    TYLOINFER_MERGE(duplex, decode_mode);
    TYLOINFER_MERGE(duplex, temperature);
    TYLOINFER_MERGE(duplex, top_k);
    TYLOINFER_MERGE(duplex, top_p);
    TYLOINFER_MERGE(duplex, text_repetition_penalty);
    TYLOINFER_MERGE(duplex, text_repetition_window_size);
    TYLOINFER_MERGE(duplex, length_penalty);
    TYLOINFER_MERGE(duplex, listen_prob_scale);
    TYLOINFER_MERGE(duplex, force_listen_count);
    TYLOINFER_MERGE(duplex, max_new_speak_tokens_per_chunk);
    TYLOINFER_MERGE(tts, temperature);
    TYLOINFER_MERGE(tts, repetition_penalty);
    TYLOINFER_MERGE(tts, token2wav_steps);
#undef TYLOINFER_MERGE
    if (source.source != "generic-defaults") target.source = source.source;
}
static std::string normalized_identity(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return std::isalnum(c) ? static_cast<char>(std::tolower(c)) : '_';
    });
    return value;
}
static bool identity_matches(const std::vector<std::string> & identities,
                             const std::string & needle) {
    return std::any_of(identities.begin(), identities.end(), [&](const auto & value) {
        return normalized_identity(value).find(needle) != std::string::npos;
    });
}
static MfqRuntimeProfile architecture_runtime_profile(
        const std::vector<std::string> & identities) {
    MfqRuntimeProfile result;
    if (identity_matches(identities, "minicpmo")) {
        result.chat.temperature = 0.7;
        result.chat.top_k = 100;
        result.chat.top_p = 0.8;
        result.chat.repetition_penalty = 1.02;
        result.chat.enable_thinking = false;
        result.duplex.system_prompt = "Streaming Omni Conversation.";
        result.duplex.decode_mode = "sampling";
        result.duplex.temperature = 0.7;
        result.duplex.top_k = 100;
        result.duplex.top_p = 0.8;
        result.duplex.text_repetition_penalty = 1.05;
        result.duplex.text_repetition_window_size = 512;
        result.duplex.length_penalty = 1.0;
        result.duplex.listen_prob_scale = 1.0;
        result.duplex.force_listen_count = 0;
        result.duplex.max_new_speak_tokens_per_chunk = 20;
        result.tts.temperature = 0.8;
        result.tts.repetition_penalty = 1.05;
        result.tts.token2wav_steps = 10;
        result.source = "architecture-registry:minicpmo";
    } else if (identity_matches(identities, "deepseek_v41")) {
        result.chat.temperature = 1.0;
        result.chat.top_p = 0.95;
        result.chat.mtp_max_draft_tokens = 5;
        result.source = "architecture-registry:deepseek_v41";
    } else if (identity_matches(identities, "deepseek_v4")) {
        result.chat.temperature = 1.0;
        result.chat.top_p = 0.8;
        result.chat.repetition_penalty = 1.05;
        result.chat.presence_penalty = 0.0;
        result.chat.mtp_max_draft_tokens = 5;
        result.source = "architecture-registry:deepseek_v4";
    }
    return result;
}
struct MfqModelCapabilityRegistration {
    std::array<const char *, 3> aliases{};
    MfqModelCapabilityProfile profile;
};
static const std::array<MfqModelCapabilityRegistration, 12>
    kModelCapabilityRegistry{{
        {{{"minicpmo", nullptr, nullptr}},
         {"minicpmo", true, true, true, true, true, true, false}},
        {{{"minicpmtts", nullptr, nullptr}},
         {"minicpmo_tts", false, false, false, false, true, false, false}},
        {{{"deepseek_v41", "deepseek_v41_text", nullptr}},
         {"deepseek_v41", true, false, false, false, false, false, true}},
        {{{"deepseek_v41_vision", nullptr, nullptr}},
         {"deepseek_v41", true, true, false, false, false, false, true}},
        {{{"deepseek_v4", nullptr, nullptr}},
         {"deepseek_v4", true, false, false, false, false, false, true}},
        {{{"deepseek_v4_vision", nullptr, nullptr}},
         {"deepseek_v4", true, true, false, false, false, false, true}},
        {{{"glm_moe_dsa", nullptr, nullptr}}, {"glm_dsa"}},
        {{{"gemma4", "gemma4_text", nullptr}}, {"gemma4"}},
        {{{"qwen3_5", "qwen35", nullptr}},
         {"qwen3_5", true, true, true, false, false, false, true}},
        {{{"qwen3_5_text", nullptr, nullptr}},
         {"qwen3_5", true, false, false, false, false, false, true}},
        {{{"qwen4_exp", "qwen4_exp_text", nullptr}},
         {"qwen4_exp", true, true, true, false, false, false, true}},
        {{{"glm5_next", "glm5_next_text", nullptr}},
         {"glm5_next", true, true, true, false, false, false, true}},
    }};
MfqModelCapabilityProfile architecture_capability_profile(
        const std::string & model_type) {
    const std::string identity = normalized_identity(model_type);
    for (const auto & registration : kModelCapabilityRegistry) {
        for (const char * alias : registration.aliases) {
            if (alias != nullptr && identity == alias) {
                return registration.profile;
            }
        }
    }
    MfqModelCapabilityProfile result;
    if (!identity.empty()) result.family = identity;
    return result;
}
json model_capability_profile_json(
        const MfqModelCapabilityProfile & profile) {
    return {
        {"architecture_family", profile.family},
        {"source", profile.source.empty()
            ? "architecture-registry:" + profile.family
            : profile.source},
        {"features", {
            {"text", profile.text},
            {"image_input", profile.image_input},
            {"video_input", profile.video_input},
            {"audio_input", profile.audio_input},
            {"audio_output", profile.audio_output},
            {"full_duplex", profile.full_duplex},
            {"mtp", profile.mtp},
        }},
    };
}
static MfqRuntimeProfile exact_model_runtime_profile(
        const std::vector<std::string> & identities) {
    MfqRuntimeProfile result;
    if (identity_matches(identities, "minicpm_o_4_5")) {
        result = architecture_runtime_profile({"minicpmo"});
        result.source = "model-registry:minicpm-o-4_5";
    } else if (identity_matches(identities, "deepseek_v4_flash_0731")) {
        result = architecture_runtime_profile({"deepseek_v4"});
        result.source = "model-registry:deepseek-v4-flash-0731";
    }
    return result;
}
static double profile_number(const json & section, const char * name) {
    if (!section[name].is_number()) {
        throw std::runtime_error(std::string("runtime profile ") + name + " must be numeric");
    }
    const double value = section[name].get<double>();
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string("runtime profile ") + name + " must be finite");
    }
    return value;
}
static int32_t profile_integer(const json & section, const char * name) {
    if (!section[name].is_number_integer()) {
        throw std::runtime_error(std::string("runtime profile ") + name + " must be an integer");
    }
    const auto value = section[name].get<int64_t>();
    if (value < std::numeric_limits<int32_t>::min() ||
        value > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(std::string("runtime profile ") + name + " is out of range");
    }
    return static_cast<int32_t>(value);
}
static MfqRuntimeProfile parse_runtime_profile(const std::string & text,
                                               const std::string & source) {
    const json root = json::parse(text);
    if (!root.is_object()) throw std::runtime_error("runtime profile must be a JSON object");
    if (root.contains("schema") && root["schema"] != "mfq.runtime.sampling") {
        throw std::runtime_error("unsupported runtime profile schema");
    }
    if (root.contains("version") &&
        (!root["version"].is_number_integer() || root["version"] != 1)) {
        throw std::runtime_error("unsupported runtime profile version");
    }
    MfqRuntimeProfile result;
    result.source = source;
    if (root.contains("chat")) {
        const auto & value = root["chat"];
        if (!value.is_object()) throw std::runtime_error("runtime profile chat must be an object");
#define TYLOINFER_CHAT_NUMBER(field) if (value.contains(#field)) result.chat.field = profile_number(value, #field)
        if (value.contains("max_tokens")) result.chat.max_tokens = profile_integer(value, "max_tokens");
        TYLOINFER_CHAT_NUMBER(temperature);
        if (value.contains("top_k")) result.chat.top_k = profile_integer(value, "top_k");
        TYLOINFER_CHAT_NUMBER(top_p);
        TYLOINFER_CHAT_NUMBER(presence_penalty);
        TYLOINFER_CHAT_NUMBER(frequency_penalty);
        TYLOINFER_CHAT_NUMBER(repetition_penalty);
        if (value.contains("enable_thinking")) {
            if (!value["enable_thinking"].is_boolean()) {
                throw std::runtime_error(
                    "runtime profile chat.enable_thinking must be boolean");
            }
            result.chat.enable_thinking =
                value["enable_thinking"].get<bool>();
        }
        if (value.contains("enable_vision")) {
            if (!value["enable_vision"].is_boolean()) {
                throw std::runtime_error(
                    "runtime profile chat.enable_vision must be boolean");
            }
            result.chat.enable_vision = value["enable_vision"].get<bool>();
        }
        if (value.contains("enable_mtp")) {
            if (!value["enable_mtp"].is_boolean()) {
                throw std::runtime_error(
                    "runtime profile chat.enable_mtp must be boolean");
            }
            result.chat.enable_mtp = value["enable_mtp"].get<bool>();
        }
        if (value.contains("mtp_max_draft_tokens")) {
            result.chat.mtp_max_draft_tokens =
                profile_integer(value, "mtp_max_draft_tokens");
        }
#undef TYLOINFER_CHAT_NUMBER
    }
    if (root.contains("duplex")) {
        const auto & value = root["duplex"];
        if (!value.is_object()) throw std::runtime_error("runtime profile duplex must be an object");
        if (value.contains("system_prompt")) {
            if (!value["system_prompt"].is_string()) {
                throw std::runtime_error(
                    "runtime profile duplex.system_prompt must be a string");
            }
            result.duplex.system_prompt =
                value["system_prompt"].get<std::string>();
        }
        if (value.contains("decode_mode")) {
            if (!value["decode_mode"].is_string()) throw std::runtime_error("runtime profile decode_mode must be a string");
            result.duplex.decode_mode = value["decode_mode"].get<std::string>();
            if (*result.duplex.decode_mode != "sampling" && *result.duplex.decode_mode != "greedy") {
                throw std::runtime_error("runtime profile decode_mode is invalid");
            }
        }
#define TYLOINFER_DUPLEX_NUMBER(field) if (value.contains(#field)) result.duplex.field = profile_number(value, #field)
#define TYLOINFER_DUPLEX_INTEGER(field) if (value.contains(#field)) result.duplex.field = profile_integer(value, #field)
        TYLOINFER_DUPLEX_NUMBER(temperature);
        TYLOINFER_DUPLEX_INTEGER(top_k);
        TYLOINFER_DUPLEX_NUMBER(top_p);
        TYLOINFER_DUPLEX_NUMBER(text_repetition_penalty);
        TYLOINFER_DUPLEX_INTEGER(text_repetition_window_size);
        TYLOINFER_DUPLEX_NUMBER(length_penalty);
        TYLOINFER_DUPLEX_NUMBER(listen_prob_scale);
        TYLOINFER_DUPLEX_INTEGER(force_listen_count);
        TYLOINFER_DUPLEX_INTEGER(max_new_speak_tokens_per_chunk);
#undef TYLOINFER_DUPLEX_NUMBER
#undef TYLOINFER_DUPLEX_INTEGER
    }
    if (root.contains("tts")) {
        const auto & value = root["tts"];
        if (!value.is_object()) throw std::runtime_error("runtime profile tts must be an object");
        if (value.contains("temperature")) result.tts.temperature = profile_number(value, "temperature");
        if (value.contains("repetition_penalty")) result.tts.repetition_penalty = profile_number(value, "repetition_penalty");
        if (value.contains("token2wav_steps")) result.tts.token2wav_steps = profile_integer(value, "token2wav_steps");
    }
    const auto bounded = [](const std::optional<double> & value,
                            double low, double high,
                            const char * name) {
        if (value && (*value < low || *value > high)) {
            throw std::runtime_error(
                std::string("runtime profile ") + name + " is out of range");
        }
    };
    const auto positive = [](const auto & value, const char * name) {
        if (value && *value <= 0) {
            throw std::runtime_error(
                std::string("runtime profile ") + name + " must be positive");
        }
    };
    bounded(result.chat.temperature, 0.0, 10.0, "chat.temperature");
    bounded(result.chat.top_p, 0.0, 1.0, "chat.top_p");
    bounded(result.duplex.temperature, 0.0, 10.0, "duplex.temperature");
    bounded(result.duplex.top_p, 0.0, 1.0, "duplex.top_p");
    bounded(result.tts.temperature, 0.0, 10.0, "tts.temperature");
    if (result.chat.top_k && *result.chat.top_k < 0) {
        throw std::runtime_error("runtime profile chat.top_k must be non-negative");
    }
    if (result.chat.mtp_max_draft_tokens &&
        (*result.chat.mtp_max_draft_tokens < 1 ||
         *result.chat.mtp_max_draft_tokens > 7)) {
        throw std::runtime_error(
            "runtime profile chat.mtp_max_draft_tokens must be in [1, 7]");
    }
    if (result.duplex.top_k && *result.duplex.top_k < 0) {
        throw std::runtime_error("runtime profile duplex.top_k must be non-negative");
    }
    positive(result.chat.max_tokens, "chat.max_tokens");
    positive(result.chat.repetition_penalty, "chat.repetition_penalty");
    positive(result.duplex.text_repetition_penalty,
             "duplex.text_repetition_penalty");
    positive(result.duplex.text_repetition_window_size,
             "duplex.text_repetition_window_size");
    positive(result.duplex.length_penalty, "duplex.length_penalty");
    positive(result.duplex.max_new_speak_tokens_per_chunk,
             "duplex.max_new_speak_tokens_per_chunk");
    positive(result.tts.repetition_penalty, "tts.repetition_penalty");
    positive(result.tts.token2wav_steps, "tts.token2wav_steps");
    if (result.duplex.listen_prob_scale &&
        *result.duplex.listen_prob_scale < 0.0) {
        throw std::runtime_error(
            "runtime profile duplex.listen_prob_scale must be non-negative");
    }
    if (result.duplex.force_listen_count &&
        (*result.duplex.force_listen_count < 0 ||
         *result.duplex.force_listen_count > 60)) {
        throw std::runtime_error(
            "runtime profile duplex.force_listen_count is out of range");
    }
    return result;
}
static std::string read_profile_file(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open runtime profile: " + path.string());
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}
static std::vector<std::filesystem::path> profile_sidecar_paths(
        const std::filesystem::path & mfq_path) {
    std::vector<std::filesystem::path> result;
    static const std::regex split_pattern(R"(^(.*)-[0-9]{5}-of-[0-9]{5}\.[mM][fF][qQ]$)");
    std::smatch match;
    const auto filename = mfq_path.filename().string();
    if (std::regex_match(filename, match, split_pattern)) {
        result.push_back(mfq_path.parent_path() / (match[1].str() + ".runtime.json"));
    } else {
        auto family = mfq_path;
        family.replace_extension(".runtime.json");
        result.push_back(std::move(family));
    }
    auto exact = std::filesystem::path(mfq_path.string() + ".runtime.json");
    if (exact != result.front()) result.push_back(std::move(exact));
    return result;
}
json duplex_profile_json(const MfqDuplexSamplingProfile & value) {
    json result = json::object();
#define TYLOINFER_SET(field) if (value.field) result[#field] = *value.field
    TYLOINFER_SET(system_prompt);
    TYLOINFER_SET(decode_mode);
    TYLOINFER_SET(temperature);
    TYLOINFER_SET(top_k);
    TYLOINFER_SET(top_p);
    TYLOINFER_SET(text_repetition_penalty);
    TYLOINFER_SET(text_repetition_window_size);
    TYLOINFER_SET(length_penalty);
    TYLOINFER_SET(listen_prob_scale);
    TYLOINFER_SET(force_listen_count);
    TYLOINFER_SET(max_new_speak_tokens_per_chunk);
#undef TYLOINFER_SET
    return result;
}
json tts_profile_json(const MfqTtsSamplingProfile & value) {
    json result = json::object();
    if (value.temperature) result["temperature"] = *value.temperature;
    if (value.repetition_penalty) result["repetition_penalty"] = *value.repetition_penalty;
    if (value.token2wav_steps) result["token2wav_steps"] = *value.token2wav_steps;
    return result;
}
json chat_template_capabilities_json(
        const mfq::engine::ChatTemplateCapabilities & capabilities) {
    return {
        {"thinking", {
            {"supported", capabilities.thinking},
        }},
        {"reasoning_effort", {
            {"supported", !capabilities.reasoning_effort_values.empty()},
            {"values", capabilities.reasoning_effort_values},
        }},
    };
}
} // namespace mfq::protocol

MfqRuntimeProfile resolve_mfq_runtime_profile(
        const std::string & mfq_path,
        const std::string & model_architecture,
        const std::string & model_type,
        const std::string & model_name,
        const std::string & embedded_profile_json,
        const std::string & model_config_json,
        const std::string & explicit_profile_path) {
    using namespace mfq::protocol;
    std::vector<std::string> identities{
        model_architecture, model_type, model_name,
    };
    if (!model_config_json.empty()) {
        const json config = json::parse(model_config_json);
        if (!config.is_object()) {
            throw std::runtime_error("embedded model config must be a JSON object");
        }
        for (const char * key : {"model_type", "_name_or_path", "name_or_path"}) {
            if (config.contains(key) && config[key].is_string()) {
                identities.push_back(config[key].get<std::string>());
            }
        }
        if (config.contains("architectures") && config["architectures"].is_array()) {
            for (const auto & value : config["architectures"]) {
                if (value.is_string()) identities.push_back(value.get<std::string>());
            }
        }
    }

    // Low to high priority. Every merge is field-wise.
    MfqRuntimeProfile result = architecture_runtime_profile(identities);
    merge_runtime_profile(result, exact_model_runtime_profile(identities));
    if (!embedded_profile_json.empty()) {
        merge_runtime_profile(result, parse_runtime_profile(
            embedded_profile_json, "embedded-mfq"));
    }
    if (!mfq_path.empty()) {
        for (const auto & sidecar : profile_sidecar_paths(mfq_path)) {
            std::error_code error;
            if (std::filesystem::is_regular_file(sidecar, error) && !error) {
                merge_runtime_profile(result, parse_runtime_profile(
                    read_profile_file(sidecar), "sidecar:" + sidecar.filename().string()));
            }
        }
    }
    if (!explicit_profile_path.empty()) {
        const std::filesystem::path path(explicit_profile_path);
        merge_runtime_profile(result, parse_runtime_profile(
            read_profile_file(path), "runtime-explicit:" + path.filename().string()));
    }
    return result;
}


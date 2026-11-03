#include "policies.h"
#include "api/errors.h"
#include "storage/storage.h"
#include "storage/sqlite.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>

namespace mfq::server {
namespace {
using json = nlohmann::json;

constexpr const char* context_key = "runtime_context_policy";
constexpr const char* inference_key = "runtime_inference_policy";
constexpr const char* memory_key = "runtime_memory_policy";

// Every rule below restates one Field constraint or model_validator from the Python models;
// the message names the offending field path so a client can find it without a schema.
ApiError rejected(const std::string& field, const std::string& rule) {
    return ApiError(422, "invalid_policy", field + " " + rule);
}

std::string joined(std::initializer_list<const char*> choices) {
    std::string text;
    for (const auto* choice : choices) text += text.empty() ? choice : std::string(", ") + choice;
    return text;
}

void reject_unknown(const json& object, const std::string& path, std::initializer_list<const char*> allowed) {
    if (!object.is_object()) throw rejected(path, "must be an object");
    for (const auto& item : object.items())
        if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end())
            throw rejected(path + "." + item.key(), "is not a supported field");
}

json checked_integer(const json& value, const std::string& field, std::int64_t minimum, std::int64_t maximum) {
    if (!value.is_number_integer() || value.get<std::int64_t>() < minimum || value.get<std::int64_t>() > maximum)
        throw rejected(field, "must be an integer between " + std::to_string(minimum) + " and " + std::to_string(maximum));
    return value;
}

// Absent fields take the model default; explicit null is only valid for Optional fields.
std::int64_t integer_field(const json& object, const char* field, std::int64_t fallback,
        std::int64_t minimum, std::int64_t maximum) {
    if (!object.contains(field)) return fallback;
    return checked_integer(object[field], field, minimum, maximum).get<std::int64_t>();
}

json optional_integer_field(const json& object, const char* field, std::int64_t minimum, std::int64_t maximum) {
    if (!object.contains(field) || object[field].is_null()) return nullptr;
    return checked_integer(object[field], field, minimum, maximum);
}

bool boolean_field(const json& object, const char* field, bool fallback) {
    if (!object.contains(field)) return fallback;
    if (!object[field].is_boolean()) throw rejected(field, "must be a boolean");
    return object[field].get<bool>();
}

std::string choice_field(const json& object, const char* field, const char* fallback,
        std::initializer_list<const char*> choices) {
    if (!object.contains(field)) return fallback;
    const auto& value = object[field];
    if (value.is_string() && std::find(choices.begin(), choices.end(), value.get<std::string>()) != choices.end())
        return value.get<std::string>();
    throw rejected(field, "must be one of " + joined(choices));
}

// KvQuantizationSettings.bits mixes integers and halves, so whole values stay whole in the dump.
json bits_field(const json& object, const char* field) {
    if (!object.contains(field)) return 4;
    constexpr std::array<double, 7> allowed{2, 2.5, 3, 3.5, 4, 6, 8};
    const auto& value = object[field];
    if (!value.is_number()) throw rejected(field, "must be one of " + joined({"2", "2.5", "3", "3.5", "4", "6", "8"}));
    const auto number = value.get<double>();
    if (std::find(allowed.begin(), allowed.end(), number) == allowed.end())
        throw rejected(field, "must be one of 2, 2.5, 3, 3.5, 4, 6, 8");
    return number == std::floor(number) ? json(static_cast<std::int64_t>(number)) : json(number);
}

json directory_field(const json& object, const char* field, std::size_t minimum, std::size_t maximum) {
    if (!object.contains(field) || object[field].is_null()) return nullptr;
    const auto& value = object[field];
    if (!value.is_string()) throw rejected(field, "must be a string");
    const auto& text = value.get_ref<const std::string&>();
    const auto length = std::count_if(text.begin(), text.end(), [](unsigned char c) { return (c & 0xc0) != 0x80; });
    if (length < minimum || length > maximum)
        throw rejected(field, "must be between " + std::to_string(minimum) + " and " + std::to_string(maximum) + " characters");
    return value;
}

json qsa_settings(const json& value, const std::string& path) {
    reject_unknown(value, path, {"enabled", "budget_bytes", "ram_budget_bytes"});
    json settings;
    settings["enabled"] = boolean_field(value, "enabled", false);
    settings["budget_bytes"] = integer_field(value, "budget_bytes", 2LL << 30, 1, 1LL << 50);
    settings["ram_budget_bytes"] = optional_integer_field(value, "ram_budget_bytes", 0, 1LL << 50);
    return settings;
}

json kv_quantization(const json& value, const std::string& path) {
    reject_unknown(value, path, {"enabled", "bits", "algorithm"});
    json settings;
    settings["enabled"] = boolean_field(value, "enabled", false);
    settings["bits"] = bits_field(value, "bits");
    settings["algorithm"] = choice_field(value, "algorithm", "turboquant", {"turboquant"});
    return settings;
}

json kv_memory(const json& value, const std::string& path) {
    reject_unknown(value, path, {"placement", "allocation", "gpu_budget_bytes", "ram_preallocated", "ram_budget_bytes"});
    json settings;
    settings["placement"] = choice_field(value, "placement", "auto", {"auto", "auto_ssd", "vram", "ram"});
    settings["allocation"] = choice_field(value, "allocation", "preallocated", {"preallocated", "dynamic"});
    settings["gpu_budget_bytes"] = optional_integer_field(value, "gpu_budget_bytes", 1, 1LL << 50);
    settings["ram_preallocated"] = boolean_field(value, "ram_preallocated", true);
    settings["ram_budget_bytes"] = optional_integer_field(value, "ram_budget_bytes", 0, 1LL << 50);
    if (!settings["ram_preallocated"].get<bool>() && settings["placement"] != "auto_ssd")
        throw rejected(path + ".ram_preallocated",
            "kv_ssd_required: disabling RAM preallocation requires SSD KV");
    return settings;
}

template<typename Build>
json settings_map(const json& policy, const char* field, Build build) {
    if (!policy.contains(field)) return json::object();
    const auto& values = policy[field];
    if (!values.is_object()) throw rejected(field, "must be an object of model names to settings");
    json result = json::object();
    for (const auto& item : values.items()) result[item.key()] = build(item.value(), std::string(field) + "." + item.key());
    return result;
}

json validate_context(const json& body) {
    reject_unknown(body, "runtime_context_policy", {"fallback_context_size", "max_context_size", "model_overrides",
        "model_yarn_enabled", "model_qsa_kv_offload", "model_kv_quantization", "model_kv_memory"});
    json policy;
    policy["fallback_context_size"] = integer_field(body, "fallback_context_size", 32768, 512, 2147483647);
    policy["max_context_size"] = optional_integer_field(body, "max_context_size", 512, 2147483647);
    policy["model_overrides"] = settings_map(body, "model_overrides", [](const json& value, const std::string& path) {
        return checked_integer(value, path, 512, 2147483647);
    });
    policy["model_yarn_enabled"] = settings_map(body, "model_yarn_enabled", [](const json& value, const std::string& path) {
        if (!value.is_boolean()) throw rejected(path, "must be a boolean");
        return value;
    });
    policy["model_qsa_kv_offload"] = settings_map(body, "model_qsa_kv_offload", qsa_settings);
    policy["model_kv_quantization"] = settings_map(body, "model_kv_quantization", kv_quantization);
    policy["model_kv_memory"] = settings_map(body, "model_kv_memory", kv_memory);
    return policy;
}

json validate_inference(const json& body) {
    reject_unknown(body, "runtime_inference_policy", {"mtp_enabled"});
    json policy;
    policy["mtp_enabled"] = boolean_field(body, "mtp_enabled", true);
    return policy;
}

json validate_memory(const json& body) {
    reject_unknown(body, "runtime_memory_policy", {"total_limit_bytes", "model_limit_bytes", "prefix_limit_bytes",
        "prefix_disk_limit_bytes", "prefix_ram_limit_bytes", "prefix_directory"});
    json policy;
    policy["total_limit_bytes"] = optional_integer_field(body, "total_limit_bytes", 1, std::numeric_limits<std::int64_t>::max());
    policy["model_limit_bytes"] = optional_integer_field(body, "model_limit_bytes", 1, std::numeric_limits<std::int64_t>::max());
    policy["prefix_limit_bytes"] = optional_integer_field(body, "prefix_limit_bytes", 0, std::numeric_limits<std::int64_t>::max());
    policy["prefix_disk_limit_bytes"] = optional_integer_field(body, "prefix_disk_limit_bytes", 0, std::numeric_limits<std::int64_t>::max());
    policy["prefix_ram_limit_bytes"] = optional_integer_field(body, "prefix_ram_limit_bytes", 0, std::numeric_limits<std::int64_t>::max());
    policy["prefix_directory"] = directory_field(body, "prefix_directory", 1, 4096);
    return policy;
}

json read_policy(sqlite3* database, const char* key) {
    sqlite::Statement query(database, "SELECT value FROM schema_meta WHERE key=?");
    query.bind(1, json(key));
    if (!query.step()) return nullptr;
    try { return json::parse(query.column(0).get<std::string>()); }
    catch (const json::exception&) { throw ApiError(500, "invalid_saved_policy", std::string(key) + " could not be read"); }
}

void write_policy(sqlite3* database, const char* key, const json& policy) {
    const auto encoded = policy.dump();
    try {
        sqlite::Statement write(database, "INSERT INTO schema_meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value");
        write.bind(1, json(key));
        write.bind(2, json(encoded));
        write.step();
    } catch (const sqlite::Error&) {
        throw ApiError(500, "policy_save_failed", std::string(key) + " could not be saved");
    }
}

// A PUT replaces the top-level fields the body names and keeps the rest, as the Python handlers do.
json merge(const json& saved, const json& body) {
    json merged = saved.is_object() ? saved : json::object();
    for (const auto& item : body.items()) merged[item.key()] = item.value();
    return merged;
}
} // namespace

RuntimePolicies::RuntimePolicies(Storage& storage) : storage_(storage) {}

json RuntimePolicies::context_policy() const {
    auto access = storage_.access();
    const auto saved = read_policy(access.database, context_key);
    return saved.is_null() ? validate_context(json::object()) : saved;
}

json RuntimePolicies::put_context_policy(const json& body) {
    if (!body.is_object()) throw ApiError(400, "invalid_request", "request must be a JSON object");
    auto access = storage_.access();
    const auto policy = validate_context(merge(read_policy(access.database, context_key), body));
    write_policy(access.database, context_key, policy);
    return policy;
}

json RuntimePolicies::inference_policy() const {
    auto access = storage_.access();
    const auto saved = read_policy(access.database, inference_key);
    return saved.is_null() ? validate_inference(json::object()) : saved;
}

json RuntimePolicies::put_inference_policy(const json& body) {
    if (!body.is_object()) throw ApiError(400, "invalid_request", "request must be a JSON object");
    auto access = storage_.access();
    const auto policy = validate_inference(body);
    write_policy(access.database, inference_key, policy);
    return policy;
}

json RuntimePolicies::memory_policy() const {
    auto access = storage_.access();
    const auto saved = read_policy(access.database, memory_key);
    return saved.is_null() ? validate_memory(json::object()) : saved;
}

json RuntimePolicies::put_memory_policy(const json& body) {
    if (!body.is_object()) throw ApiError(400, "invalid_request", "request must be a JSON object");
    auto access = storage_.access();
    const auto policy = validate_memory(merge(read_policy(access.database, memory_key), body));
    write_policy(access.database, memory_key, policy);
    return policy;
}

} // namespace mfq::server

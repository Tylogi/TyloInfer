#include "handlers.h"
#include "cache/prefix.h"
#include "http.h"
#include "instance_json.h"
#include "models/errors.h"
#include "models/instances.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <optional>

namespace mfq::server {
namespace {
using json = nlohmann::json;

void fields(const json& body, std::initializer_list<const char*> allowed) {
    if (!body.is_object()) throw ApiError(400, "invalid_request", "request must be a JSON object");
    for (const auto& item : body.items())
        if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end())
            throw ApiError(400, "invalid_request", "unknown cache operation parameter: " + item.key());
}

std::optional<std::string> instance_of(const json& body) {
    if (!body.contains("instance_id") || body["instance_id"].is_null()) return {};
    if (!body["instance_id"].is_string()) throw ApiError(400, "invalid_request", "instance_id must be a UUID");
    return normalize_uuid(body["instance_id"].get<std::string>());
}

std::string choice(const httplib::Request& request, const char* name, const char* fallback,
        std::initializer_list<const char*> allowed) {
    if (!request.has_param(name)) return fallback;
    const auto value = request.get_param_value(name);
    if (std::find(allowed.begin(), allowed.end(), value) != allowed.end()) return value;
    throw ApiError(400, "invalid_request", std::string(name) + " must be one of the supported values");
}

// The engine session metrics ride along on cache answers, as the worker transport does.
json metrics_json(const InstanceInfo& info) {
    json flat = json::object();
    for (const auto& [name, value] : info.execution.session) flat[name] = value;
    return flat;
}
} // namespace

void cache_entries(const Services& services, const httplib::Request& request, httplib::Response& response) {
    json filters{{"offset", 0}, {"limit", 100}};
    if (request.has_param("namespace")) {
        const auto space = request.get_param_value("namespace");
        if (space.size() != 64 || !std::all_of(space.begin(), space.end(),
                [](unsigned char c) { return std::isxdigit(c) != 0; }))
            throw ApiError(400, "invalid_request", "namespace must be a 64-character hex digest");
        filters["namespace"] = space;
    }
    for (const char* name : {"offset", "limit"})
        if (request.has_param(name)) {
            const auto text = request.get_param_value(name);
            std::int64_t value = 0;
            if (text.empty() || !std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isdigit(c) != 0; }) ||
                (value = std::stoll(text)) < 0)
                throw ApiError(400, "invalid_request", std::string(name) + " must be a non-negative integer");
            if (std::string(name) == "limit" && (value < 1 || value > 200))
                throw ApiError(400, "invalid_request", "limit must be between 1 and 200");
            filters[name] = value;
        }
    if (request.has_param("query")) filters["query"] = request.get_param_value("query");
    filters["search_in"] = choice(request, "search_in", "all", {"all", "text", "id"});
    filters["text_filter"] = choice(request, "text_filter", "all", {"all", "saved", "missing"});
    filters["chain_filter"] = choice(request, "chain_filter", "all", {"all", "complete", "incomplete"});
    filters["sort"] = choice(request, "sort", "recent", {"recent", "oldest", "length", "size"});
    set_json(response, services.cache.entries(filters));
}

void cache_block_text(const Services& services, const httplib::Request& request, httplib::Response& response) {
    std::int64_t offset = 0, limit = 8192;
    for (const char* name : {"offset", "limit"})
        if (request.has_param(name)) {
            const auto text = request.get_param_value(name);
            if (text.empty() || !std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isdigit(c) != 0; }))
                throw ApiError(400, "invalid_request", std::string(name) + " must be a non-negative integer");
            (name == std::string("offset") ? offset : limit) = std::stoll(text);
        }
    if (limit < 1 || limit > 8192) throw ApiError(400, "invalid_request", "limit must be between 1 and 8192");
    set_json(response, services.cache.block_text(request.matches[1], request.matches[2], offset, limit));
}

void cache_purge(const Services& services, const httplib::Request& request, httplib::Response& response) {
    auto& instances = services.instances;
    auto& cache = services.cache;
    const auto body = parse_json(request);
    fields(body, {"namespace"});
    std::optional<std::string> space;
    if (body.contains("namespace")) {
        if (!body["namespace"].is_string()) throw ApiError(400, "invalid_request", "namespace must be a hex digest string");
        const auto value = body["namespace"].get<std::string>();
        if (value.size() != 64 || !std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isxdigit(c) != 0; }))
            throw ApiError(400, "invalid_request", "namespace must be a 64-character hex digest");
        space = value;
    }
    // Python refuses to purge while inference or cache writes are in flight; the native gate is queued requests.
    if (instances.busy({})) throw ApiError(409, "runtime_busy",
        "wait for inference and cache writes to finish before clearing cache");
    const auto result = cache.purge(space);
    // The engine keeps an index of the purged files; refresh it so the freed space is reused.
    try { instances.cache_control({}, engine::SessionCommand{engine::SessionCommand::Kind::refresh}); }
    catch (const ModelError&) {}
    set_json(response, result);
}

void cache_clear(const Services& services, const httplib::Request& request, httplib::Response& response) {
    auto& instances = services.instances;
    const auto body = parse_json(request);
    fields(body, {"instance_id"});
    const auto target = instance_of(body);
    const auto items = instances.list();
    if (!target && items.empty()) throw ApiError(503, "model_not_loaded", "no runtime is available");
    json result{{"status", "ok"}, {"released_snapshots", instances.cache_control(target,
        engine::SessionCommand{engine::SessionCommand::Kind::clear})}};
    for (const auto& item : items)
        if (!target || item.id == *target) result.merge_patch(metrics_json(item));
    set_json(response, result);
}

void cache_trim(const Services& services, const httplib::Request& request, httplib::Response& response) {
    auto& instances = services.instances;
    const auto body = parse_json(request);
    fields(body, {"target_bytes", "instance_id"});
    const auto target = instance_of(body);
    std::int64_t bytes = 0;
    if (body.contains("target_bytes")) {
        if (!body["target_bytes"].is_number_integer() || body["target_bytes"].get<std::int64_t>() < 0)
            throw ApiError(400, "invalid_request", "target_bytes must be a non-negative integer");
        bytes = body["target_bytes"];
    }
    const auto items = instances.list();
    if (!target && items.empty()) throw ApiError(503, "model_not_loaded", "no runtime is available");
    engine::SessionCommand command{engine::SessionCommand::Kind::trim, {}, {},
        static_cast<std::uint64_t>(bytes)};
    json result{{"status", "ok"}, {"released_bytes", instances.cache_control(target, command)},
        {"target_bytes", bytes}};
    for (const auto& item : items)
        if (!target || item.id == *target) result.merge_patch(metrics_json(item));
    set_json(response, result);
}

} // namespace mfq::server

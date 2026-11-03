#include "runtime_config.h"
#include "http.h"
#include "instance_json.h"
#include "jobs/jobs.h"
#include "models.h"
#include "models/instances.h"
#include "models/policies.h"
#include "models/profiles.h"
#include "protocol.h"

#include <algorithm>
#include <chrono>
#include <initializer_list>
#include <optional>

namespace mfq::server {
namespace {
using json = nlohmann::json;

void fields(const json& body, std::initializer_list<const char*> allowed) {
    if (!body.is_object()) throw ApiError(400, "invalid_request", "request must be a JSON object");
    for (const auto& item : body.items())
        if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end())
            throw ApiError(400, "invalid_request", "unknown model operation parameter: " + item.key());
}
} // namespace

void register_runtime_config(httplib::Server& http, RuntimePolicies& policies, RuntimeProfiles& profiles,
        ModelInstances& instances, Jobs& jobs, const std::string& host, const int& port) {
    http.Get("/api/runtime/context-policy", [&](const httplib::Request&, httplib::Response& response) {
        set_json(response, policies.context_policy());
    });
    http.Put("/api/runtime/context-policy", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, policies.put_context_policy(parse_json(request)));
    });
    http.Get("/api/runtime/inference-policy", [&](const httplib::Request&, httplib::Response& response) {
        set_json(response, policies.inference_policy());
    });
    http.Put("/api/runtime/inference-policy", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, policies.put_inference_policy(parse_json(request)));
    });
    // shortcut: the engine has no memory budget seam yet, so the job only validates and persists;
    // wire instance planning into it when that work package lands.
    http.Get("/api/runtime/memory-policy", [&](const httplib::Request&, httplib::Response& response) {
        set_json(response, policies.memory_policy());
    });
    http.Put("/api/runtime/memory-policy", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request);
        policies.check_memory(body); // field errors reject the request before the job is accepted
        const auto operation = jobs.submit("runtime.memory.configure", body,
            [&policies, body](const Jobs::Control&) { return policies.put_memory_policy(body); });
        set_json(response, {{"operation_id", operation}, {"status", "accepted"}}, 202);
    });

    http.Post("/api/runtime/profiles", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, profiles.create(parse_json(request)), 201);
    });
    http.Get("/api/runtime/profiles", [&](const httplib::Request&, httplib::Response& response) {
        set_json(response, profiles.list());
    });
    http.Get("/api/runtime/profiles/([^/]+)", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, profiles.get(normalize_uuid(request.matches[1])));
    });
    http.Put("/api/runtime/profiles/([^/]+)", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, profiles.update(normalize_uuid(request.matches[1]), parse_json(request)));
    });
    http.Delete("/api/runtime/profiles/([^/]+)", [&](const httplib::Request& request, httplib::Response& response) {
        profiles.remove(normalize_uuid(request.matches[1]));
        response.status = 204;
    });
    // The profile load request is the stored load document; it finishes in the same model.load job as
    // POST /api/models/load, after the drift gate.
    http.Post("/api/runtime/profiles/([^/]+)/load", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request);
        fields(body, {"allow_drift"});
        if (body.contains("allow_drift") && !body["allow_drift"].is_boolean())
            throw ApiError(400, "invalid_request", "allow_drift must be a boolean");
        const auto profile = profiles.load_request(normalize_uuid(request.matches[1]), body.value("allow_drift", false));
        const auto& load = profile.request;
        LoadOptions options;
        options.context_size = load.contains("context_size") && !load["context_size"].is_null()
            ? load["context_size"].get<std::int64_t>()
            : policy_context_size(profile.artifact, policies.context_policy());
        LoadPolicy policy;
        policy.pin = load.value("pin", false);
        if (load.contains("idle_ttl_seconds") && !load["idle_ttl_seconds"].is_null())
            policy.idle_ttl = std::chrono::seconds(load["idle_ttl_seconds"].get<std::int64_t>());
        if (policy.pin && policy.idle_ttl)
            throw ApiError(400, "invalid_request", "pinned instances cannot have an idle timeout");
        policy.mtp_enabled = policies.inference_policy().at("mtp_enabled").get<bool>();
        const auto operation = submit_model_load(jobs, instances, profile.artifact.id, options, policy, load);
        response.set_header("Location", "/api/jobs/" + operation);
        set_json(response, {{"operation_id", operation}, {"status", "accepted"}}, 202);
    });

    http.Get("/api/runtime/capabilities", [&](const httplib::Request& request, httplib::Response& response) {
        const auto items = instances.list();
        auto chosen = items.end();
        if (const auto id = query_uuid(request, "instance_id")) {
            chosen = std::find_if(items.begin(), items.end(), [&](const auto& item) { return item.id == *id; });
            if (chosen == items.end())
                throw ApiError(404, "runtime_instance_not_found", "runtime instance was not found: " + *id);
        } else {
            // The native server has no shared runtime worker to ask for capabilities of its own, so an
            // idle server reports the same model_not_loaded the Python runtime pool raises.
            chosen = std::find_if(items.begin(), items.end(),
                [](const auto& item) { return item.state == InstanceInfo::State::loaded; });
            if (chosen == items.end()) throw ApiError(503, "model_not_loaded", "no runtime is available");
        }
        const auto& engine = chosen->capabilities;
        set_json(response, {{"model", chosen->model}, {"model_type", engine.model_type},
            {"model_capabilities", mfq::protocol::model_capability_profile_json(engine.capabilities)},
            {"vision_available", engine.capabilities.image_input || engine.capabilities.video_input},
            {"mtp_available", engine.capabilities.mtp}, {"duplex_available", engine.capabilities.full_duplex}});
    });

    http.Patch("/api/runtime/instances/([^/]+)", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request);
        fields(body, {"pinned", "idle_ttl_seconds"});
        if (body.empty()) throw ApiError(400, "invalid_request", "at least one runtime policy field is required");
        std::optional<bool> pinned;
        std::optional<std::optional<std::chrono::seconds>> idle_ttl;
        if (body.contains("pinned")) {
            if (!body["pinned"].is_boolean()) throw ApiError(400, "invalid_request", "pinned must be a boolean");
            pinned = body["pinned"].get<bool>();
        }
        if (body.contains("idle_ttl_seconds")) {
            if (body["idle_ttl_seconds"].is_null()) idle_ttl = std::optional<std::chrono::seconds>{};
            else {
                if (!body["idle_ttl_seconds"].is_number_integer() || body["idle_ttl_seconds"] < 0)
                    throw ApiError(400, "invalid_request", "idle_ttl_seconds must be an integer of at least 0");
                idle_ttl = std::chrono::seconds(body["idle_ttl_seconds"].get<std::int64_t>());
            }
        }
        set_json(response, instance_json(instances.update_policy(normalize_uuid(request.matches[1]), pinned, idle_ttl)));
    });

    // The native listener serves both protocols on the port it bound at startup, so there is nothing
    // to reconfigure at runtime; the Python managed listener does not exist here.
    http.Get("/api/runtime/listener", [&](const httplib::Request&, httplib::Response& response) {
        set_json(response, {{"host", host}, {"port", port}, {"anthropic_port", nullptr}, {"configurable", false}});
    });
    http.Put("/api/runtime/listener", [&](const httplib::Request& request, httplib::Response&) {
        const auto body = parse_json(request);
        fields(body, {"port", "protocol"});
        if (!body.contains("port") || !body["port"].is_number_integer() || body["port"] < 1 || body["port"] > 65535)
            throw ApiError(400, "invalid_request", "port must be an integer between 1 and 65535");
        const auto protocol = body.value("protocol", std::string("openai"));
        if (protocol != "openai" && protocol != "anthropic")
            throw ApiError(400, "invalid_request", "protocol must be openai or anthropic");
        throw ApiError(409, "listener_not_managed", "server listener is managed externally");
    });
}
} // namespace mfq::server

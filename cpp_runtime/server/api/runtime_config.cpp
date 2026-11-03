#include "handlers.h"
#include "jobs/jobs.h"
#include "http.h"
#include "instance_json.h"
#include "models.h"
#include "models/instances.h"
#include "models/policies.h"
#include "models/profiles.h"

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

void context_policy_get(const Services& services, const httplib::Request&, httplib::Response& response) {
    set_json(response, services.policies.context_policy());
}

void context_policy_put(const Services& services, const httplib::Request& request, httplib::Response& response) {
    set_json(response, services.policies.put_context_policy(parse_json(request)));
}

void inference_policy_get(const Services& services, const httplib::Request&, httplib::Response& response) {
    set_json(response, services.policies.inference_policy());
}

void inference_policy_put(const Services& services, const httplib::Request& request, httplib::Response& response) {
    set_json(response, services.policies.put_inference_policy(parse_json(request)));
}

// shortcut: the engine has no memory budget seam yet, so the job only validates and persists;
// wire instance planning into it when that work package lands.
void memory_policy_get(const Services& services, const httplib::Request&, httplib::Response& response) {
    set_json(response, services.policies.memory_policy());
}

void memory_policy_put(const Services& services, const httplib::Request& request, httplib::Response& response) {
    auto& policies = services.policies;
    auto& jobs = services.jobs;
    const auto body = parse_json(request);
    policies.check_memory(body); // field errors reject the request before the job is accepted
    const auto operation = jobs.submit("runtime.memory.configure", body,
        [&policies, body](const Jobs::Control&) { return policies.put_memory_policy(body); });
    accept(response, operation);
}

void profiles_create(const Services& services, const httplib::Request& request, httplib::Response& response) {
    set_json(response, services.profiles.create(parse_json(request)), 201);
}

void profiles_list(const Services& services, const httplib::Request&, httplib::Response& response) {
    set_json(response, services.profiles.list());
}

void profile_get(const Services& services, const httplib::Request& request, httplib::Response& response) {
    set_json(response, services.profiles.get(normalize_uuid(request.matches[1])));
}

void profile_update(const Services& services, const httplib::Request& request, httplib::Response& response) {
    set_json(response, services.profiles.update(normalize_uuid(request.matches[1]), parse_json(request)));
}

void profile_delete(const Services& services, const httplib::Request& request, httplib::Response& response) {
    services.profiles.remove(normalize_uuid(request.matches[1]));
    response.status = 204;
}

// The profile load request is the stored load document; it finishes in the same model.load job as
// POST /api/models/load, after the drift gate.
void profile_load(const Services& services, const httplib::Request& request, httplib::Response& response) {
    auto& policies = services.policies;
    auto& profiles = services.profiles;
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
    const auto operation = submit_model_load(services.jobs, services.instances, profile.artifact.id, options, policy, load);
    accept(response, operation);
}

void runtime_instance_patch(const Services& services, const httplib::Request& request, httplib::Response& response) {
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
    set_json(response, instance_json(services.instances.update_policy(normalize_uuid(request.matches[1]), pinned, idle_ttl)));
}

} // namespace mfq::server

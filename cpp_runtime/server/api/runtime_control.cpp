#include "handlers.h"
#include "http.h"
#include "instance_json.h"
#include "models/catalog.h"
#include "models/context.h"
#include "models/errors.h"
#include "models/instances.h"
#include "models/policies.h"
#include "models/components.h"
#include "jobs/jobs.h"
#include "tyloinfer/model_source.h"

#include <algorithm>
#include <optional>
#include <vector>

namespace mfq::server {
namespace {
using json = nlohmann::json;

void fields(const json& body, const char* who) {
    if (!body.is_object()) throw ApiError(400, "invalid_request", "request must be a JSON object");
    const std::vector<const char*> allowed{"apply_global_cap", "context_size", "instance_id", "yarn_enabled",
        "qsa_kv_offload", "kv_quantization", "kv_memory"};
    for (const auto& item : body.items())
        if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end())
            throw ApiError(400, "invalid_request", std::string("unknown ") + who + " parameter: " + item.key());
}

// RuntimeReloadRequest: the context knobs the engine seam supports plus the KV settings it does not yet.
std::optional<std::int64_t> requested_context(const json& body) {
    if (!body.contains("context_size") || body["context_size"].is_null()) return {};
    if (!body["context_size"].is_number_integer() || body["context_size"].get<std::int64_t>() < 512 ||
        body["context_size"].get<std::int64_t>() > 2147483647)
        throw ApiError(400, "invalid_request", "context_size must be an integer between 512 and 2147483647");
    return body["context_size"].get<std::int64_t>();
}

void reject_unsupported_kv(const json& body) {
    if (body.contains("yarn_enabled") && body["yarn_enabled"].is_boolean() && body["yarn_enabled"].get<bool>())
        throw ApiError(400, "yarn_unsupported", "YaRN extension is not supported by this model/backend");
    if (body.contains("qsa_kv_offload") && body["qsa_kv_offload"].is_object() &&
        body["qsa_kv_offload"].value("enabled", false))
        throw ApiError(501, "qsa_kv_offload_unsupported", "live QSA KV reconfiguration is not available on the native engine");
    if (body.contains("kv_quantization") && body["kv_quantization"].is_object() &&
        body["kv_quantization"].value("enabled", false))
        throw ApiError(501, "kv_quantization_unsupported", "live TurboQuant reconfiguration is not available on the native engine");
    if (body.contains("kv_memory") && body["kv_memory"].is_object() && !body["kv_memory"].empty()) {
        const json& memory = body["kv_memory"];
        const json neutral{{"placement", "auto"}, {"allocation", "preallocated"},
            {"gpu_budget_bytes", nullptr}, {"ram_preallocated", true}, {"ram_budget_bytes", nullptr}};
        if (memory != neutral)
            throw ApiError(501, "kv_memory_unsupported", "live KV placement reconfiguration is not available on the native engine");
    }
}

const ModelCatalog::Artifact& artifact_of(ModelCatalog& catalog, const InstanceInfo& info) {
    for (const auto& artifact : catalog.list(false))
        if (artifact.id == info.artifact_id) return artifact;
    throw ModelError(ModelError::Kind::not_found, "model_artifact_not_found", "the model artifact is no longer registered");
}

std::int64_t default_context(const ModelCatalog::Artifact& artifact, const json& policy) {
    const auto& cap = policy.at("max_context_size");
    if (artifact.context_capacity) {
        if (!cap.is_null()) return std::min(*artifact.context_capacity, cap.get<std::int64_t>());
        return *artifact.context_capacity;
    }
    return policy.at("fallback_context_size").get<std::int64_t>();
}

// shortcut: the native engine reloads a context size only; yarn and KV reconfiguration are rejected
// honestly until the engine grows those seams, and the reload never persists into model_overrides.
json reload_target(ModelInstances& instances, ModelCatalog& catalog, RuntimePolicies& policies, const json& body) {
    fields(body, "reload");
    reject_unsupported_kv(body);
    std::optional<std::string> instance;
    if (body.contains("instance_id") && !body["instance_id"].is_null()) {
        if (!body["instance_id"].is_string()) throw ApiError(400, "invalid_request", "instance_id must be a UUID");
        instance = normalize_uuid(body["instance_id"].get<std::string>());
    }
    const auto requested = requested_context(body);
    const auto items = instances.list();
    const auto chosen = instance ? std::vector<InstanceInfo>{[&] {
        const auto found = std::find_if(items.begin(), items.end(), [&](const auto& item) { return item.id == *instance; });
        if (found == items.end()) throw ModelError(ModelError::Kind::not_found, "runtime_instance_not_found",
            "runtime instance was not found: " + *instance);
        return *found;
    }() } : items;
    if (chosen.empty()) throw ApiError(503, "model_not_loaded", "no runtime is available");
    json models = json::array();
    std::int64_t cap = 0;
    for (const auto& prior : chosen) {
        std::int64_t target = requested.value_or(default_context(artifact_of(catalog, prior), policies.context_policy()));
        if (prior.capabilities.max_context) target = std::min(target, prior.capabilities.max_context);
        const auto reloaded = instances.reload(prior.id, target);
        if (reloaded.empty() || reloaded.front().capabilities.max_context != target)
            throw ApiError(400, "context_reload_mismatch",
                "requested context " + std::to_string(target) + ", but the engine reported another size");
        cap = std::max(cap, reloaded.front().capabilities.max_context);
        if (instance) return json{{"max_context", reloaded.front().capabilities.max_context},
            {"instance_id", prior.id}, {"model", prior.model}};
        models.push_back({{"model", prior.model}, {"max_context", reloaded.front().capabilities.max_context}});
    }
    return json{{"models_reloaded", chosen.size()}, {"max_context", cap}, {"models", std::move(models)}};
}
} // namespace

void runtime_context(const Services& services, const httplib::Request& request, httplib::Response& response) {
    const auto body = parse_json(request);
    fields(body, "reload");
    const auto operation = services.jobs.submit("runtime.context.configure", body,
        [&services, body](const Jobs::Control&) {
            return reload_target(services.instances, services.catalog, services.policies, body);
        });
    response.set_header("Location", "/api/jobs/" + operation);
    set_json(response, {{"operation_id", operation}, {"status", "accepted"}}, 202);
}


void voice_status(const Services& services, const httplib::Request&, httplib::Response& response) {
    set_json(response, services.voice.status());
}

void voice_install(const Services& services, const httplib::Request&, httplib::Response& response) {
    auto& voice = services.voice;
    const auto operation = services.jobs.submit("component.voice_output.install", json::object(),
        [&voice](const Jobs::Control& control) {
            return voice.install([&](double value, const std::string& message) { control.progress(message, value); },
                [&] { return control.cancelled.load(); });
        });
    set_json(response, {{"operation_id", operation}, {"status", "accepted"}}, 202);
}

void voice_activate(const Services&, const httplib::Request&, httplib::Response&) {
    // shortcut: activation flips the realtime duplex path on the Python runtime pool; the native
    // engine has no such seam yet, so this answers Python's not-managed error.
    throw ApiError(501, "model_management_unavailable", "managed model runtime is not available");
}

void runtime_yarn(const Services& services, const httplib::Request& request, httplib::Response& response) {
    auto& catalog = services.catalog;
    auto& policies = services.policies;
    const auto items = services.instances.list();
    const auto found = std::find_if(items.begin(), items.end(), [&](const auto& item) {
        return item.id == normalize_uuid(request.matches[1]); });
    if (found == items.end()) throw ModelError(ModelError::Kind::not_found, "model_not_loaded", "model is not loaded");
    const auto& artifact = artifact_of(catalog, *found);
    json limits = json::object();
    try {
        auto source = mfq::open_model_source(artifact.path);
        if (source->has_asset(mfq::kModelConfigAsset)) limits = json::parse(source->model_config_json());
    } catch (const std::exception&) { limits = json::object(); }
    const YarnLimits yarn = yarn_limits(limits.is_object() ? limits : json::object());
    const bool supported = yarn.extended.has_value();
    const auto& policy = policies.context_policy();
    const bool enabled = supported && policy.at("model_yarn_enabled").contains(artifact.name) &&
        policy.at("model_yarn_enabled")[artifact.name].get<bool>();
    const std::int64_t current = found->capabilities.max_context;
    set_json(response, json{{"supported", supported}, {"native_context", yarn.native.has_value() ? json(*yarn.native) : json(nullptr)},
        {"maximum_context", supported ? json(*yarn.extended) : json(yarn.native.has_value() ? json(*yarn.native) : json(nullptr))},
        {"maximum_factor", supported ? json(*yarn.factor) : json(nullptr)}, {"enabled", enabled},
        {"effective_factor", yarn.native && *yarn.native > 0
            ? json(std::max(1.0, static_cast<double>(current) / static_cast<double>(*yarn.native))) : json(1.0)}});
}

void qsa_kv_get(const Services& services, const httplib::Request& request, httplib::Response& response) {
    auto& catalog = services.catalog;
    std::optional<std::int64_t> target;
    if (request.has_param("target_context")) {
        const auto text = request.get_param_value("target_context");
        std::int64_t value = 0;
        if (text.empty() || !std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isdigit(c) != 0; }) ||
            (value = std::stoll(text)) < 512 || value > 2147483647)
            throw ApiError(400, "invalid_request", "target_context must be an integer between 512 and 2147483647");
        target = value;
    }
    const auto items = services.instances.list();
    const auto found = std::find_if(items.begin(), items.end(), [&](const auto& item) {
        return item.id == normalize_uuid(request.matches[1]); });
    if (found == items.end()) throw ModelError(ModelError::Kind::not_found, "runtime_instance_not_found",
        "runtime instance was not found: " + normalize_uuid(request.matches[1]));
    const auto& artifact = artifact_of(catalog, *found);
    json config = json::object();
    std::shared_ptr<const mfq::ModelSource> source;
    try {
        source = mfq::open_model_source(artifact.path);
        if (source->has_asset(mfq::kModelConfigAsset)) config = json::parse(source->model_config_json());
        if (!config.is_object()) config = json::object();
    } catch (const std::exception&) { config = json::object(); }
    if (!source) throw ApiError(503, "model_source_unavailable", "the model source cannot be opened");
    const std::int64_t aim = target.value_or(found->capabilities.max_context);
    const auto& policy = services.policies.context_policy().at("model_qsa_kv_offload");
    json settings{{"enabled", false}, {"budget_bytes", std::int64_t(2) << 30}, {"ram_budget_bytes", nullptr}};
    if (policy.contains(artifact.name) && policy[artifact.name].is_object()) settings = policy[artifact.name];
    settings["target_context"] = aim;
    const bool cuda = services.instances.backend() == "cuda";
    const QsaNeed need = qsa_requirement(*source, config, aim,
        settings["enabled"].get<bool>() ? std::optional<std::int64_t>{settings["budget_bytes"].get<std::int64_t>()} : std::nullopt,
        {}, cuda);
    set_json(response, json{{"supported", need.supported}, {"architecture", artifact.architecture},
        {"policy", std::move(settings)},
        {"required_index_bytes", need.supported ? json(need.required_index_bytes) : json(nullptr)},
        {"buffer_bytes", need.buffer_bytes}, {"target_context", aim},
        {"reason", need.supported ? json(nullptr) : json(need.reason)}});
}

void qsa_kv_put(const Services& services, const httplib::Request& request, httplib::Response& response) {
    auto& catalog = services.catalog;
    auto& policies = services.policies;
    auto& jobs = services.jobs;
    const auto body = parse_json(request);
    if (!body.is_object()) throw ApiError(400, "invalid_request", "request must be a JSON object");
    if (!body.contains("instance_id") || !body["instance_id"].is_string())
        throw ApiError(400, "invalid_request", "instance_id is required");
    const std::string id = normalize_uuid(body["instance_id"].get<std::string>());
    const bool enabled = body.value("enabled", false);
    const std::int64_t budget = body.contains("budget_bytes") && body["budget_bytes"].is_number_integer()
        && body["budget_bytes"].get<std::int64_t>() >= 1 && body["budget_bytes"].get<std::int64_t>() <= (std::int64_t(1) << 50)
        ? body["budget_bytes"].get<std::int64_t>() : throw ApiError(400, "invalid_request",
            "budget_bytes must be an integer between 1 and 2^50");
    const auto items = services.instances.list();
    const auto found = std::find_if(items.begin(), items.end(), [&](const auto& item) { return item.id == id; });
    if (found == items.end()) throw ModelError(ModelError::Kind::not_found, "runtime_instance_not_found",
        "runtime instance was not found: " + id);
    const auto& artifact = artifact_of(catalog, *found);
    // shortcut: the check runs synchronously as Python's pre-check does; the engine has no live QSA
    // reconfiguration seam, so the job persists the setting and it applies at the next model load.
    auto source = mfq::open_model_source(artifact.path);
    json config = json::object();
    if (source->has_asset(mfq::kModelConfigAsset)) { try { config = json::parse(source->model_config_json()); } catch (...) {} }
    const QsaNeed need = qsa_requirement(*source, config.is_object() ? config : json::object(),
        found->capabilities.max_context, enabled ? std::optional<std::int64_t>{budget} : std::nullopt, {},
        services.instances.backend() == "cuda");
    if (!need.supported) throw ApiError(400, "qsa_kv_offload_unsupported", need.reason);
    const auto operation = jobs.submit("runtime.qsa-kv.configure", body,
        [&policies, body, id, artifact, enabled, budget](const Jobs::Control&) {
        const auto saved = policies.context_policy();
        json model_policy = saved.at("model_qsa_kv_offload");
        model_policy[artifact.name] = {{"enabled", enabled}, {"budget_bytes", budget},
            {"ram_budget_bytes", body.contains("ram_budget_bytes") ? body["ram_budget_bytes"] : json(nullptr)}};
        return policies.put_context_policy(json{{"model_qsa_kv_offload", std::move(model_policy)}});
    });
    set_json(response, {{"operation_id", operation}, {"status", "accepted"}}, 202);
}

} // namespace mfq::server

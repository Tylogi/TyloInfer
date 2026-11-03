#include "routes.h"
#include "api/handlers.h"
#include "instance.h"
#include "media/media.h"

#include <regex>
#include <stdexcept>
#include <utility>

namespace mfq::server {
namespace {

void health(const Services&, const httplib::Request&, httplib::Response& response) {
    set_json(response, {{"status", "ok"}, {"service", "tyloinfer-server"}, {"protocol_version", "1.0"}});
}

void identity(const Services& services, const httplib::Request&, httplib::Response& response) {
    set_json(response, services.instance.identity());
}

} // namespace

// The whole API surface, in dispatch order (first matching row wins). Rows are grouped by the
// file that implements their handlers; adding a route means one row here plus one function there.
const std::vector<Route>& route_table() {
    static const std::vector<Route> table = {
        // server.cpp health glue (no auth: outside /api and /v1)
        {"GET", "/health", Scope::inference, health},
        {"GET", "/health/identity", Scope::inference, identity},

        // api/auth.cpp
        {"GET", "/api/auth/policy", Scope::admin, auth_policy_get},
        {"PUT", "/api/auth/policy", Scope::admin, auth_policy_put},
        {"GET", "/api/auth/keys", Scope::admin, auth_keys},
        {"POST", "/api/auth/keys", Scope::admin, auth_key_create},
        {"POST", "/api/auth/keys/([^/]+)/revoke", Scope::admin, auth_key_revoke},
        {"POST", "/api/auth/keys/([^/]+)/rotate", Scope::admin, auth_key_rotate},

        // api/models.cpp (catalog, directories, load/unload, analysis, deletion)
        {"GET", "/api/models", Scope::inference, models_list},
        {"GET", "/api/models/directories", Scope::models, models_browse},
        {"POST", "/api/models/directories/open", Scope::models, models_open},
        {"POST", "/api/models/directories/register", Scope::models, models_register},
        {"GET", "/api/models/([^/]+)/directory", Scope::models, model_directory},
        {"GET", "/api/models/([^/]+)/capabilities", Scope::inference, model_capabilities},
        {"POST", "/api/models/load", Scope::models, models_load},
        {"POST", "/api/models/unload", Scope::models, models_unload},

        {"GET", "/api/models/([^/]+)/analysis", Scope::inference, model_analysis},
        {"GET", "/api/models/([^/]+)/cache-profile", Scope::inference, model_cache_profile},
        {"GET", "/api/models/([^/]+)/deletion", Scope::inference, model_deletion},
        {"POST", "/api/models/([^/]+)/delete", Scope::models, model_delete},

        // api/runtime.cpp
        {"GET", "/api/runtime/status", Scope::inference, runtime_status},
        {"GET", "/api/runtime/instances", Scope::inference, runtime_instances},
        {"GET", "/api/runtime/instances/([^/]+)", Scope::inference, runtime_instance},
        {"GET", "/api/runtime/resources", Scope::inference, runtime_resources},
        {"GET", "/api/runtime/model-aliases", Scope::inference, runtime_aliases_get},
        {"PUT", "/api/runtime/model-aliases", Scope::admin, runtime_aliases_put},

        // api/runtime_config.cpp
        {"GET", "/api/runtime/context-policy", Scope::inference, context_policy_get},
        {"PUT", "/api/runtime/context-policy", Scope::admin, context_policy_put},
        {"GET", "/api/runtime/inference-policy", Scope::inference, inference_policy_get},
        {"PUT", "/api/runtime/inference-policy", Scope::admin, inference_policy_put},
        {"GET", "/api/runtime/memory-policy", Scope::inference, memory_policy_get},
        {"PUT", "/api/runtime/memory-policy", Scope::admin, memory_policy_put},
        {"POST", "/api/runtime/profiles", Scope::models, profiles_create},
        {"GET", "/api/runtime/profiles", Scope::inference, profiles_list},
        {"GET", "/api/runtime/profiles/([^/]+)", Scope::inference, profile_get},
        {"PUT", "/api/runtime/profiles/([^/]+)", Scope::models, profile_update},
        {"DELETE", "/api/runtime/profiles/([^/]+)", Scope::models, profile_delete},
        {"POST", "/api/runtime/profiles/([^/]+)/load", Scope::models, profile_load},
        {"GET", "/api/runtime/capabilities", Scope::inference, runtime_capabilities},
        {"PATCH", "/api/runtime/instances/([^/]+)", Scope::models, runtime_instance_patch},
        {"GET", "/api/runtime/listener", Scope::inference, listener_get},
        {"PUT", "/api/runtime/listener", Scope::admin, listener_put},

        // api/runtime_cache.cpp
        {"GET", "/api/runtime/cache/entries", Scope::admin, cache_entries},
        {"GET", R"(/api/runtime/cache/entries/([0-9a-f]{64})/([0-9a-f]{64})/text)", Scope::admin, cache_block_text},
        {"POST", "/api/runtime/cache/purge", Scope::admin, cache_purge},
        {"POST", "/api/runtime/cache/clear", Scope::models, cache_clear},
        {"POST", "/api/runtime/cache/trim", Scope::models, cache_trim},

        // api/runtime_control.cpp
        {"POST", "/api/runtime/context", Scope::models, runtime_context},
        {"POST", "/api/runtime/reload", Scope::models, runtime_reload},
        {"GET", "/api/components/voice-output", Scope::inference, voice_status},
        {"POST", "/api/components/voice-output/install", Scope::inference, voice_install},
        {"POST", "/api/components/voice-output/activate", Scope::inference, voice_activate},
        {"GET", "/api/runtime/yarn/([0-9a-fA-F-]{36})", Scope::inference, runtime_yarn},
        {"GET", "/api/runtime/qsa-kv/([0-9a-fA-F-]{36})", Scope::inference, qsa_kv_get},
        {"PUT", "/api/runtime/qsa-kv", Scope::models, qsa_kv_put},

        // api/jobs.cpp
        {"GET", "/api/jobs", Scope::jobs, jobs},
        {"DELETE", "/api/jobs/completed", Scope::jobs, jobs_clear},
        {"DELETE", "/api/jobs/([0-9a-f-]{36})", Scope::jobs, job_delete},
        {"GET", "/api/jobs/([0-9a-f-]{36})", Scope::jobs, job_get},
        {"POST", "/api/jobs/([0-9a-f-]{36})/cancel", Scope::jobs, job_cancel},
        {"GET", "/api/jobs/([0-9a-f-]{36})/events", Scope::jobs, job_events},
        {"GET", "/api/jobs/([0-9a-f-]{36})/events/stream", Scope::jobs, job_events_stream},

        // api/hub.cpp
        {"GET", "/api/hub/download-directory", Scope::models, hub_directory_get},
        {"PUT", "/api/hub/download-directory", Scope::models, hub_directory_put},
        {"GET", "/api/hub/models", Scope::inference, hub_search},
        {"GET", "/api/hub/official", Scope::inference, hub_official},
        {"GET", R"(/api/hub/models/([^/]+)/([^/]+)/([^/]+)/readme)", Scope::inference, hub_readme},
        {"GET", "/api/hub/models/([^/]+)/([^/]+)/([^/]+)", Scope::inference, hub_repository},
        {"POST", "/api/hub/resolve", Scope::models, hub_resolve},
        {"POST", "/api/hub/download-reuse", Scope::models, hub_reuse},
        {"POST", "/api/hub/download-sources/probe", Scope::models, hub_probe},
        {"GET", "/api/jobs/kinds", Scope::jobs, jobs_kinds},
        {"POST", "/api/jobs", Scope::jobs, jobs_submit},
        {"POST", "/api/jobs/([0-9a-f-]{36})/retry", Scope::jobs, job_retry},

        // api/anthropic.cpp
        {"POST", "/v1/messages/count_tokens", Scope::inference, anthropic_count_tokens},
        {"POST", "/v1/messages", Scope::inference, anthropic_messages},

        // api/inference.cpp
        {"GET", "/v1", Scope::inference, compat_root},
        {"GET", "/v1/models", Scope::inference, compat_models},
        {"GET", "/v1/models/(.+)", Scope::inference, compat_model},
        {"POST", "/api/inference/requests/([0-9a-f-]{36})/cancel", Scope::inference, inference_cancel},
        {"POST", "/v1/chat/completions", Scope::inference, chat_completions},

        // api/sessions.cpp
        {"POST", "/api/sessions", Scope::inference, sessions_create},
        {"GET", "/api/sessions", Scope::inference, sessions_list},
        {"POST", "/api/sessions/import", Scope::inference, sessions_import},
        {"GET", "/api/sessions/([0-9a-fA-F-]{36})", Scope::inference, session_get},
        {"PATCH", "/api/sessions/([0-9a-fA-F-]{36})", Scope::inference, session_update},
        {"DELETE", "/api/sessions/([0-9a-fA-F-]{36})", Scope::inference, session_delete},
        {"GET", "/api/sessions/([0-9a-fA-F-]{36})/messages", Scope::inference, session_messages},
        {"POST", "/api/sessions/([0-9a-fA-F-]{36})/messages", Scope::inference, session_message_add},
        {"POST", "/api/sessions/([0-9a-fA-F-]{36})/fork", Scope::inference, session_fork},
        {"POST", "/api/sessions/([0-9a-fA-F-]{36})/rewind", Scope::inference, session_rewind},
        {"GET", "/api/sessions/([0-9a-fA-F-]{36})/export", Scope::inference, session_export},
        {"GET", "/api/sessions/([0-9a-fA-F-]{36})/responses", Scope::inference, session_responses},
        {"POST", "/api/sessions/([0-9a-fA-F-]{36})/responses/cancel", Scope::inference, session_cancel},
        {"POST", "/api/sessions/([0-9a-fA-F-]{36})/responses", Scope::inference, session_respond},
        {"GET", "/api/presets", Scope::inference, presets_list},
        {"POST", "/api/presets", Scope::inference, presets_create},
        {"PUT", "/api/presets/([0-9a-fA-F-]{36})", Scope::inference, preset_update},
        {"DELETE", "/api/presets/([0-9a-fA-F-]{36})", Scope::inference, preset_delete},

        // api/media.cpp
        {"POST", "/api/media", Scope::inference, media_upload, Media::max_bytes},
        {"GET", "/api/media/([0-9a-fA-F-]{36})", Scope::inference, media_get},
    };
    return table;
}

const Route* find_route(const std::string& method, const std::string& path) {
    // shortcut: patterns compile lazily once per process; the server registers no routes at runtime.
    static const auto compiled = [] {
        std::vector<std::pair<std::regex, const Route*>> entries;
        for (const auto& row : route_table()) entries.emplace_back(std::regex(row.path), &row);
        return entries;
    }();
    const std::string& effective = method == "HEAD" ? std::string{"GET"} : method;
    for (const auto& [pattern, row] : compiled)
        if (effective == row->method && std::regex_match(path, pattern)) return row;
    return nullptr;
}

void register_routes(httplib::Server& http, const Services& services) {
    for (const auto& row : route_table()) {
        const auto handler = [&services, fn = row.handler](const httplib::Request& request,
                httplib::Response& response) { fn(services, request, response); };
        const std::string method = row.method;
        if (method == "GET") http.Get(row.path, handler);
        else if (method == "POST") http.Post(row.path, handler);
        else if (method == "PUT") http.Put(row.path, handler);
        else if (method == "PATCH") http.Patch(row.path, handler);
        else if (method == "DELETE") http.Delete(row.path, handler);
        else if (method == "OPTIONS") http.Options(row.path, handler);
        else throw std::logic_error("unsupported route method: " + method);
    }
}

} // namespace mfq::server

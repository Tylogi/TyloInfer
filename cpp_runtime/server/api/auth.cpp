#include "auth.h"
#include "http.h"
#include "instance.h"

namespace mfq::server {
Scope required_scope(const std::string& method, const std::string& path) {
    const auto starts = [&](const char* prefix) { return path.rfind(prefix, 0) == 0; };
    const bool read = method == "GET" || method == "HEAD";
    if (path == "/api/hub/download-directory") return Scope::models;
    if (starts("/api/runtime/cache/entries") || path == "/api/runtime/cache/purge") return Scope::admin;
    if (!read && (path == "/api/runtime/listener" || path == "/api/runtime/memory-policy" ||
        path == "/api/runtime/model-aliases" || path == "/api/runtime/context-policy")) return Scope::admin;
    if (starts("/api/auth/") || starts("/api/cluster/")) return Scope::admin;
    if (starts("/api/models/directories")) return Scope::models;
    if (starts("/api/models/") && path.size() >= 10 && path.compare(path.size() - 10, 10, "/directory") == 0) return Scope::models;
    if (starts("/api/models") || starts("/api/runtime") || starts("/api/hub"))
        return read ? Scope::inference : Scope::models;
    for (const auto* prefix : {"/api/jobs", "/api/artifacts", "/api/datasets", "/api/evaluations", "/api/quantization"})
        if (starts(prefix)) return Scope::jobs;
    return Scope::inference;
}

void register_auth(httplib::Server& http, Authentication& auth) {
    http.Get("/api/auth/policy", [&](const httplib::Request&, httplib::Response& response) {
        set_json(response, auth.status());
    });
    http.Put("/api/auth/policy", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, auth.configure(parse_json(request)));
    });
    http.Get("/api/auth/keys", [&](const httplib::Request&, httplib::Response& response) {
        set_json(response, auth.list_keys());
    });
    http.Post("/api/auth/keys", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, auth.create_key(parse_json(request)), 201);
    });
    http.Post("/api/auth/keys/([^/]+)/revoke", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, auth.revoke_key(normalize_uuid(request.matches[1])));
    });
    http.Post("/api/auth/keys/([^/]+)/rotate", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, auth.rotate_key(request.matches[1]));
    });
}
}

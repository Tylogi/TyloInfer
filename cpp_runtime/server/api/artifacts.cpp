#include "artifacts.h"
#include "http.h"
#include "models/catalog.h"
#include "models/instances.h"
#include "models/errors.h"
#include "models/analysis.h"
#include "models/deletion.h"
#include "storage/sqlite.h"
#include "storage/storage.h"

#include <algorithm>
#include <cstdint>

namespace mfq::server {
namespace {
using json = nlohmann::json;

void fields(const json& request, std::initializer_list<const char*> allowed) {
    if (!request.is_object()) throw ApiError(400, "invalid_request", "request must be a JSON object");
    for (const auto& item : request.items())
        if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end())
            throw ApiError(400, "invalid_request", "unknown model operation parameter: " + item.key());
}

std::string selector(const json& request, const char* key) {
    if (!request.contains(key) || !request[key].is_string())
        throw ApiError(400, "invalid_request", std::string(key) + " must be a string");
    return request[key].get<std::string>();
}

bool boolean_query(const httplib::Request& request, const char* name) {
    const auto value = request.get_param_value(name);
    if (request.has_param(name) && value != "true" && value != "false" && value != "1" && value != "0")
        throw ApiError(400, "invalid_request", std::string(name) + " must be a boolean");
    return value == "true" || value == "1";
}

void valid_token(const std::string& token) {
    if (token.size() != 64 || !std::all_of(token.begin(), token.end(),
            [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }))
        throw ApiError(400, "invalid_request", "token must be a SHA-256 hexadecimal digest");
}
} // namespace

void register_artifacts(httplib::Server& http, ModelCatalog& catalog, ModelInstances& instances,
        Storage& storage, const std::string& backend) {
    const bool cuda = backend == "cuda";
    http.Get("/api/artifacts/lineage", [&](const httplib::Request& request, httplib::Response& response) {
        std::string query = "SELECT id,artifact_uri,artifact_name,producer_job_id,producer_kind,source_uris_json,"
            "parameters_json,metadata_json,created_at FROM artifact_lineage";
        if (request.has_param("artifact_uri")) query += " WHERE artifact_uri=?";
        query += " ORDER BY created_at DESC,id DESC LIMIT ?";
        std::int64_t limit = 100;
        if (request.has_param("limit")) {
            const auto text = request.get_param_value("limit");
            if (text.empty() || !std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isdigit(c) != 0; }) ||
                (limit = std::stoll(text)) < 1 || limit > 500)
                throw ApiError(400, "invalid_request", "limit must be an integer between 1 and 500");
        }
        // The native server has no lineage producers yet (evaluations land later); reads still answer.
        const auto access = storage.access();
        sqlite::Statement select(access.database, query.c_str());
        if (request.has_param("artifact_uri")) select.bind(1, json(request.get_param_value("artifact_uri")));
        select.bind(request.has_param("artifact_uri") ? 2 : 1, json(limit));
        json data = json::array();
        while (select.step()) data.push_back(json{{"id", select.column(0)}, {"artifact_uri", select.column(1)},
            {"artifact_name", select.column(2)}, {"producer_job_id", select.column(3)}, {"producer_kind", select.column(4)},
            {"source_uris", json::parse(select.column(5).get<std::string>())},
            {"parameters", json::parse(select.column(6).get<std::string>())},
            {"metadata", json::parse(select.column(7).get<std::string>())}, {"created_at", select.column(8)}});
        set_json(response, {{"data", std::move(data)}});
    });
    http.Post("/api/artifacts/remove", [&](const httplib::Request&, httplib::Response&) {
        // shortcut: workspace artifact removal needs the workspace tool seam; nothing registers one natively yet.
        throw ApiError(501, "workspace_management_unavailable", "workspace artifact removal is not configured");
    });
    http.Get("/api/models/([^/]+)/analysis", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, checkpoint_analysis_json(catalog.resolve(request.matches[1]), cuda));
    });
    http.Get("/api/models/([^/]+)/cache-profile", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, model_cache_profile_json(catalog.resolve(request.matches[1]),
            boolean_query(request, "include_recurrent"), cuda));
    });
    http.Get("/api/models/([^/]+)/deletion", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, deletion_plan_json(model_deletion_plan(catalog.resolve(request.matches[1]))));
    });
    http.Post("/api/models/([^/]+)/delete", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request);
        fields(body, {"token"});
        const auto token = selector(body, "token");
        valid_token(token);
        const auto artifact = catalog.resolve(request.matches[1]);
        if (instances.busy_for(artifact.id, artifact.name))
            throw ModelError(ModelError::Kind::conflict, "model_delete_busy",
                "Unload the model and wait for loading to finish before deleting it");
        const auto result = delete_model_files(artifact, token);
        catalog.list(true);
        set_json(response, result);
    });
}
}

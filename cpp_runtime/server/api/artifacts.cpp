#include "handlers.h"
#include "http.h"
#include "models/catalog.h"
#include "models/instances.h"
#include "models/errors.h"
#include "models/analysis.h"
#include "models/deletion.h"

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


void model_analysis(const Services& services, const httplib::Request& request, httplib::Response& response) {
    set_json(response, checkpoint_analysis_json(services.catalog.resolve(request.matches[1]), services.backend == "cuda"));
}

void model_cache_profile(const Services& services, const httplib::Request& request, httplib::Response& response) {
    set_json(response, model_cache_profile_json(services.catalog.resolve(request.matches[1]),
        boolean_query(request, "include_recurrent"), services.backend == "cuda"));
}

void model_deletion(const Services& services, const httplib::Request& request, httplib::Response& response) {
    set_json(response, deletion_plan_json(model_deletion_plan(services.catalog.resolve(request.matches[1]))));
}

void model_delete(const Services& services, const httplib::Request& request, httplib::Response& response) {
    auto& catalog = services.catalog;
    auto& instances = services.instances;
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
}

}

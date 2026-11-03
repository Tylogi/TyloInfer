#include "profiles.h"
#include "api/errors.h"
#include "errors.h"
#include "instance.h"
#include "storage/sqlite.h"
#include "storage/storage.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

namespace mfq::server {
namespace {
using json = nlohmann::json;

constexpr const char* select_one = "SELECT id,name,load_json,sampling_json,artifact_id,artifact_modified_at,created_at,"
    "updated_at FROM runtime_profiles WHERE id=?";
constexpr const char* select_all = "SELECT id,name,load_json,sampling_json,artifact_id,artifact_modified_at,created_at,"
    "updated_at FROM runtime_profiles ORDER BY name COLLATE NOCASE, id";
constexpr const char* insert_one = "INSERT INTO runtime_profiles(id,name,load_json,sampling_json,artifact_id,"
    "artifact_modified_at,created_at,updated_at) VALUES(?,?,?,?,?,?,strftime('%Y-%m-%dT%H:%M:%f+00:00','now'),"
    "strftime('%Y-%m-%dT%H:%M:%f+00:00','now'))";
constexpr const char* update_one = "UPDATE runtime_profiles SET name=?,load_json=?,sampling_json=?,artifact_id=?,"
    "artifact_modified_at=?,updated_at=strftime('%Y-%m-%dT%H:%M:%f+00:00','now') WHERE id=?";

// The message names the offending field. Python answers the same bodies with a 422 pydantic error;
// the native server keeps the 400 invalid_request shape every other native route uses.
ApiError rejected(const std::string& field, const std::string& rule) {
    return ApiError(400, "invalid_request", field + " " + rule);
}

ApiError missing(const std::string& id) {
    return ApiError(404, "runtime_profile_not_found", "runtime profile was not found: " + id);
}

std::size_t characters(const std::string& text) {
    return std::count_if(text.begin(), text.end(), [](unsigned char c) { return (c & 0xc0) != 0x80; });
}

std::string profile_name(const json& body) {
    if (!body.is_object()) throw rejected("request", "must be a JSON object");
    if (!body.contains("name") || !body["name"].is_string()) throw rejected("name", "must be a string");
    const auto& name = body["name"].get_ref<const std::string&>();
    const auto length = characters(name);
    if (length < 1 || length > 64) throw rejected("name", "must be 1 to 64 characters");
    return name;
}

// Only the fields the native load path consumes are checked. The rest of the Python ModelLoadRequest
// (sampling, cache, KV and context settings) is stored and echoed for the Python server to consume.
json checked_load(const json& body) {
    if (!body.contains("load") || !body["load"].is_object()) throw rejected("load", "must be a JSON object");
    const auto load = body["load"];
    if (!load.contains("model") || !load["model"].is_string()) throw rejected("load.model", "must be a string");
    if (characters(load["model"].get<std::string>()) > 255) throw rejected("load.model", "must be 1 to 255 characters");
    const auto range = [&](const char* field, std::int64_t minimum, std::int64_t maximum) {
        if (!load.contains(field) || load[field].is_null()) return;
        if (!load[field].is_number_integer() || load[field] < minimum || load[field] > maximum)
            throw rejected(std::string("load.") + field,
                "must be null or an integer between " + std::to_string(minimum) + " and " + std::to_string(maximum));
    };
    range("context_size", 512, 2147483647);
    // shortcut: the native reaper has no zero-second idle timeout; accept 0 when it supports one.
    range("idle_ttl_seconds", 1, 2147483647);
    if (load.contains("pin") && !load["pin"].is_boolean()) throw rejected("load.pin", "must be a boolean");
    if (load.contains("sampling_defaults") && !load["sampling_defaults"].is_null() && !load["sampling_defaults"].is_object())
        throw rejected("load.sampling_defaults", "must be a JSON object or null");
    return load;
}

json sampling_column(const json& load) {
    const auto found = load.find("sampling_defaults");
    return found == load.end() || found->is_null() ? json(nullptr) : json(found->dump());
}

struct Row {
    std::string id, name, artifact_id, artifact_modified_at, created_at, updated_at;
    json load;
};

Row read_row(sqlite::Statement& query) {
    Row row;
    row.id = query.column(0).get<std::string>();
    row.name = query.column(1).get<std::string>();
    row.load = json::parse(query.column(2).get<std::string>());
    // Python keeps sampling_defaults in its own column; the resource always shows it inside load.
    const auto sampling = query.column(3);
    row.load["sampling_defaults"] = sampling.is_null() ? json(nullptr) : json::parse(sampling.get<std::string>());
    row.artifact_id = query.column(4).get<std::string>();
    row.artifact_modified_at = query.column(5).get<std::string>();
    row.created_at = query.column(6).get<std::string>();
    row.updated_at = query.column(7).get<std::string>();
    return row;
}

// _runtime_profile_drift: an artifact that no longer resolves is drift too, so the client learns why
// the profile cannot load instead of getting a bare flag.
std::pair<bool, json> drift(ModelCatalog& catalog, const Row& row) {
    try {
        const auto artifact = catalog.resolve(row.load.at("model").get<std::string>());
        const auto changed = artifact.id != row.artifact_id || artifact.modified_at != row.artifact_modified_at;
        return {changed, changed ? json("the selected model artifact changed after this profile was saved") : json(nullptr)};
    } catch (const ModelError& error) {
        return {true, std::string(error.what())};
    }
}

json resource(ModelCatalog& catalog, const Row& row) {
    const auto state = drift(catalog, row);
    return {{"id", row.id}, {"name", row.name}, {"load", row.load}, {"artifact_id", row.artifact_id},
        {"artifact_modified_at", row.artifact_modified_at}, {"drifted", state.first}, {"drift_reason", state.second},
        {"created_at", row.created_at}, {"updated_at", row.updated_at}};
}

Row require_row(sqlite3* database, const std::string& id) {
    sqlite::Statement query(database, select_one);
    query.bind(1, json(id));
    if (!query.step()) throw missing(id);
    return read_row(query);
}
} // namespace

RuntimeProfiles::RuntimeProfiles(Storage& storage, ModelCatalog& catalog) : storage_(storage), catalog_(catalog) {}

json RuntimeProfiles::create(const json& body) {
    const auto name = profile_name(body);
    const auto load = checked_load(body);
    const auto artifact = catalog_.resolve(load.at("model").get<std::string>());
    const auto id = normalize_uuid({});
    auto access = storage_.access();
    try {
        sqlite::Statement insert(access.database, insert_one);
        insert.bind(1, json(id));
        insert.bind(2, json(name));
        insert.bind(3, json(load.dump()));
        insert.bind(4, sampling_column(load));
        insert.bind(5, json(artifact.id));
        insert.bind(6, json(artifact.modified_at));
        insert.step();
    } catch (const sqlite::Error& error) {
        if ((error.code & 255) == SQLITE_CONSTRAINT)
            throw ApiError(409, "runtime_profile_conflict", "runtime profile already exists: " + name);
        throw;
    }
    return resource(catalog_, require_row(access.database, id));
}

json RuntimeProfiles::list() {
    auto access = storage_.access();
    sqlite::Statement query(access.database, select_all);
    auto data = json::array();
    while (query.step()) {
        const auto row = read_row(query);
        data.push_back(resource(catalog_, row));
    }
    return {{"data", std::move(data)}};
}

json RuntimeProfiles::get(const std::string& id) {
    auto access = storage_.access();
    return resource(catalog_, require_row(access.database, id));
}

json RuntimeProfiles::update(const std::string& id, const json& body) {
    const auto name = profile_name(body);
    const auto load = checked_load(body);
    const auto artifact = catalog_.resolve(load.at("model").get<std::string>());
    auto access = storage_.access();
    int changed = 0;
    try {
        sqlite::Statement write(access.database, update_one);
        write.bind(1, json(name));
        write.bind(2, json(load.dump()));
        write.bind(3, sampling_column(load));
        write.bind(4, json(artifact.id));
        write.bind(5, json(artifact.modified_at));
        write.bind(6, json(id));
        write.step();
        changed = sqlite3_changes(access.database);
    } catch (const sqlite::Error& error) {
        if ((error.code & 255) == SQLITE_CONSTRAINT)
            throw ApiError(409, "runtime_profile_conflict", "runtime profile already exists: " + name);
        throw;
    }
    if (changed != 1) throw missing(id);
    return resource(catalog_, require_row(access.database, id));
}

void RuntimeProfiles::remove(const std::string& id) {
    auto access = storage_.access();
    sqlite::Statement write(access.database, "DELETE FROM runtime_profiles WHERE id=?");
    write.bind(1, json(id));
    write.step();
    if (sqlite3_changes(access.database) != 1) throw missing(id);
}

RuntimeProfiles::Load RuntimeProfiles::load_request(const std::string& id, bool allow_drift) {
    auto access = storage_.access();
    const auto row = require_row(access.database, id);
    const auto state = drift(catalog_, row);
    if (state.first && !allow_drift) throw ApiError(409, "runtime_profile_drift", state.second.get<std::string>());
    return {catalog_.resolve(row.load.at("model").get<std::string>()), row.load};
}

} // namespace mfq::server

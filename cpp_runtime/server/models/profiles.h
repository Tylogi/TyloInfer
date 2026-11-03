#pragma once

#include "catalog.h"

#include <nlohmann/json.hpp>
#include <string>

namespace mfq::server {
class Storage;

// Named model-load requests in the runtime_profiles table. Every read re-resolves the artifact
// through the catalog, so a changed or missing artifact surfaces as drifted=true plus a reason,
// exactly as the Python service reports it.
class RuntimeProfiles {
public:
    RuntimeProfiles(Storage& storage, ModelCatalog& catalog);
    nlohmann::json create(const nlohmann::json& body);
    nlohmann::json list();
    nlohmann::json get(const std::string& id);
    nlohmann::json update(const std::string& id, const nlohmann::json& body);
    void remove(const std::string& id);
    // The load route's input: the resolved artifact for the context default plus the stored request.
    // Drift is rejected here unless the caller allows it.
    struct Load { ModelCatalog::Artifact artifact; nlohmann::json request; };
    Load load_request(const std::string& id, bool allow_drift);

private:
    Storage& storage_;
    ModelCatalog& catalog_;
};

} // namespace mfq::server

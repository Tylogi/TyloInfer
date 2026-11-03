#pragma once

#include "models/catalog.h"
#include "models/instances.h"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>

namespace httplib { class Server; }
namespace mfq::server {
class ModelCatalog;
class ModelInstances;
class Jobs;
class RuntimePolicies;
void register_models(httplib::Server& http, ModelCatalog& catalog, ModelInstances& instances, Jobs& jobs,
    RuntimePolicies& policies);

// The context policy default for a load request that omits context_size; shared by both load routes.
std::int64_t policy_context_size(const ModelCatalog::Artifact& artifact, const nlohmann::json& policy);
// Submits the model.load job both load routes finish with, and returns the operation ID.
std::string submit_model_load(Jobs& jobs, ModelInstances& instances, const std::string& artifact_id,
    const LoadOptions& options, const LoadPolicy& policy, const nlohmann::json& payload);
}

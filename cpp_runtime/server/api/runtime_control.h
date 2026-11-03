#pragma once
namespace httplib { class Server; }
namespace mfq::server {
class Jobs;
class ModelCatalog;
class ModelInstances;
class RuntimePolicies;
class VoiceOutput;
void register_runtime_control(httplib::Server& http, ModelInstances& instances,
    ModelCatalog& catalog, RuntimePolicies& policies, Jobs& jobs, VoiceOutput& voice);
}

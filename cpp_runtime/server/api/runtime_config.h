#pragma once
#include <string>

namespace httplib { class Server; }
namespace mfq::server {
class RuntimePolicies;
class RuntimeProfiles;
class ModelInstances;
class Jobs;
// `port` is the bound listener port, which the server fills in before it starts accepting requests.
void register_runtime_config(httplib::Server& http, RuntimePolicies& policies, RuntimeProfiles& profiles,
    ModelInstances& instances, Jobs& jobs, const std::string& host, const int& port);
}

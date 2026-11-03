#pragma once
namespace httplib { class Server; }
namespace mfq::server {
class RuntimePolicies;
void register_runtime_config(httplib::Server& http, RuntimePolicies& policies);
}

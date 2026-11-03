#pragma once
#include <string>
namespace httplib { class Server; }
namespace mfq::server {
class ModelCatalog;
class ModelInstances;
class Jobs;
void register_models(httplib::Server& http, ModelCatalog& catalog, ModelInstances& instances, Jobs& jobs,
    const std::string& backend);
}

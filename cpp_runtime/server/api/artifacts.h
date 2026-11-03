#pragma once
#include <string>
namespace httplib { class Server; }
namespace mfq::server {
class ModelCatalog;
class ModelInstances;
void register_artifacts(httplib::Server& http, ModelCatalog& catalog, ModelInstances& instances, const std::string& backend);
}

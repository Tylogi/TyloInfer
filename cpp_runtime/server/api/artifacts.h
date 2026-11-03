#pragma once
#include <string>
namespace httplib { class Server; }
namespace mfq::server {
class ModelCatalog;
class ModelInstances;
class Storage;
void register_artifacts(httplib::Server& http, ModelCatalog& catalog, ModelInstances& instances,
    Storage& storage, const std::string& backend);
}

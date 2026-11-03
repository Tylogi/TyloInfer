#pragma once
#include <string>
namespace httplib { class Server; }
namespace mfq::server {
class Hub;
class Jobs;
class ModelCatalog;
void register_hub(httplib::Server& http, Hub& hub, ModelCatalog& catalog, Jobs& jobs, const std::string& backend);
}

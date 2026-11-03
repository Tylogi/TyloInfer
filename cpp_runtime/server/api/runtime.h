#pragma once
namespace httplib { class Server; }
namespace mfq::server {
class ModelInstances;
void register_runtime(httplib::Server& http, ModelInstances& instances);
}

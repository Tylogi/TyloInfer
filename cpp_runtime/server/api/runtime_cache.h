#pragma once
namespace httplib { class Server; }
namespace mfq::server {
class ModelInstances;
class PrefixCache;
void register_runtime_cache(httplib::Server& http, ModelInstances& instances, PrefixCache& cache);
}

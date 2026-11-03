#include "runtime_config.h"
#include "http.h"
#include "models/policies.h"

namespace mfq::server {
void register_runtime_config(httplib::Server& http, RuntimePolicies& policies) {
    http.Get("/api/runtime/context-policy", [&](const httplib::Request&, httplib::Response& response) {
        set_json(response, policies.context_policy());
    });
    http.Put("/api/runtime/context-policy", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, policies.put_context_policy(parse_json(request)));
    });
    http.Get("/api/runtime/inference-policy", [&](const httplib::Request&, httplib::Response& response) {
        set_json(response, policies.inference_policy());
    });
    http.Put("/api/runtime/inference-policy", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, policies.put_inference_policy(parse_json(request)));
    });
    // shortcut: the native engine has no memory budget seam yet, so the memory policy is only
    // persisted and echoed; wire it into instance planning when that work package lands.
    http.Get("/api/runtime/memory-policy", [&](const httplib::Request&, httplib::Response& response) {
        set_json(response, policies.memory_policy());
    });
    http.Put("/api/runtime/memory-policy", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, policies.put_memory_policy(parse_json(request)));
    });
}
} // namespace mfq::server

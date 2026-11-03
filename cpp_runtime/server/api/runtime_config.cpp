#include "runtime_config.h"
#include "http.h"
#include "jobs/jobs.h"
#include "models/policies.h"

namespace mfq::server {
void register_runtime_config(httplib::Server& http, RuntimePolicies& policies, Jobs& jobs) {
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
    // shortcut: the engine has no memory budget seam yet, so the job only validates and persists;
    // wire instance planning into it when that work package lands.
    http.Get("/api/runtime/memory-policy", [&](const httplib::Request&, httplib::Response& response) {
        set_json(response, policies.memory_policy());
    });
    http.Put("/api/runtime/memory-policy", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request);
        policies.check_memory(body); // field errors reject the request before the job is accepted
        const auto operation = jobs.submit("runtime.memory.configure", body,
            [&policies, body](const Jobs::Control&) { return policies.put_memory_policy(body); });
        set_json(response, {{"operation_id", operation}, {"status", "accepted"}}, 202);
    });
}
} // namespace mfq::server

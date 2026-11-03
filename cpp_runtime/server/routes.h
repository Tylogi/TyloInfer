#pragma once

// The single seam of the HTTP surface: route_table() in routes.cpp is the one place
// that lists every API route (method, path, auth scope, body limit) in dispatch order.
// handlers live in api/*.cpp; server.cpp only assembles Services and calls register_routes.

#include "api/http.h"
#include "auth/auth.h"
#include "httplib.h"

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace mfq::server {

class InstanceLock;
class Storage;
class ModelCatalog;
class ModelInstances;
class Jobs;
class Sessions;
class Media;
class Hub;
class RuntimePolicies;
class RuntimeProfiles;
class PrefixCache;
class VoiceOutput;

// Every long-lived service the route handlers may touch, assembled once in run().
struct Services {
    InstanceLock& instance;
    Authentication& auth;
    Storage& storage;
    ModelCatalog& catalog;
    ModelInstances& instances;
    Jobs& jobs;
    Sessions& sessions;
    Media& media;
    Hub& hub;
    RuntimePolicies& policies;
    RuntimeProfiles& profiles;
    PrefixCache& cache;
    VoiceOutput& voice;
    std::string host;
    int port;  // the bound listener port; run() fills it in after binding
    std::string backend;
    std::shared_ptr<std::atomic<std::size_t>> streams;
};

using RouteHandler = void (*)(const Services&, const httplib::Request&, httplib::Response&);

struct Route {
    const char* method;
    const char* path;
    Scope scope;
    std::size_t body_limit;  // Content-Length ceiling enforced before dispatch
    RouteHandler handler;
    Route(const char* method, const char* path, Scope scope, RouteHandler handler,
        std::size_t body_limit = json_payload_bytes)
        : method(method), path(path), scope(scope), body_limit(body_limit), handler(handler) {}
};

// Every API route in registration order; first matching row wins, as with httplib dispatch.
const std::vector<Route>& route_table();
// First table row matching the method (HEAD falls back to GET) and path; null for unknown paths.
const Route* find_route(const std::string& method, const std::string& path);
// Registers every table row on the server; the only registration call run() makes.
void register_routes(httplib::Server& http, const Services& services);

} // namespace mfq::server

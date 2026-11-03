#include "handlers.h"
#include "http.h"
#include "instance.h"

namespace mfq::server {

void auth_policy_get(const Services& services, const httplib::Request&, httplib::Response& response) {
    set_json(response, services.auth.status());
}

void auth_policy_put(const Services& services, const httplib::Request& request, httplib::Response& response) {
    set_json(response, services.auth.configure(parse_json(request)));
}

void auth_keys(const Services& services, const httplib::Request&, httplib::Response& response) {
    set_json(response, services.auth.list_keys());
}

void auth_key_create(const Services& services, const httplib::Request& request, httplib::Response& response) {
    set_json(response, services.auth.create_key(parse_json(request)), 201);
}

void auth_key_revoke(const Services& services, const httplib::Request& request, httplib::Response& response) {
    set_json(response, services.auth.revoke_key(normalize_uuid(request.matches[1])));
}

void auth_key_rotate(const Services& services, const httplib::Request& request, httplib::Response& response) {
    set_json(response, services.auth.rotate_key(request.matches[1]));
}

}

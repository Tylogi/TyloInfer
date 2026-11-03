#include "media.h"
#include "http.h"
#include "instance.h"

namespace mfq::server {
void register_media(httplib::Server& http, Media& media) {
    http.Post("/api/media", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, media.put(request.body, request.get_header_value("Content-Type"),
            request.get_header_value("X-Content-SHA256")), 201);
    });
    http.Get("/api/media/([0-9a-fA-F-]{36})", [&](const httplib::Request& request, httplib::Response& response) {
        const auto id = normalize_uuid(request.matches[1]);
        const auto resource = media.get(id);
        const auto mime = resource.at("media").at("mime_type").get<std::string>();
        response.set_header("Cache-Control", "private, immutable, max-age=31536000");
        response.set_header("ETag", "\"" + resource.at("media").at("sha256").get<std::string>() + "\"");
        response.set_file_content(path_utf8(media.path(id)), mime);
    });
}
} // namespace mfq::server

#include "hub/hub.h"
#include "hub/fetch.h"
#include "models/catalog.h"
#include "instance.h"
#include "server_test_process.h"
#include "mfq_paged_prefix_cache.h"
#include "httplib.h"
#include <nlohmann/json.hpp>
#include <atomic>
#include <future>
#include <iostream>
#include <sstream>
#include <regex>

using namespace mfq::server;
using namespace std::chrono_literals;
namespace fs = std::filesystem;
using Json = nlohmann::json;
namespace {
#define CHECK(value) do { if (!(value)) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #value); } while (false)
struct Directory {
    fs::path root;
    explicit Directory(const fs::path& parent = fs::temp_directory_path()) : root(parent / ("mfq-hub-" + normalize_uuid({}))) { fs::create_directories(root); }
    ~Directory() { std::error_code ignored; fs::remove_all(root, ignored); }
};
void write(const fs::path& path, const std::string& value) { fs::create_directories(path.parent_path()); std::ofstream out(path, std::ios::binary); out << value; CHECK(out.good()); }
template<class Action> void error(const char* code, Action action) {
    try { action(); } catch (const Hub::Error& failure) { CHECK(std::string(failure.code) == code); return; }
    throw std::runtime_error(std::string("expected error: ") + code);
}
template<class Action> void wait(Action action) {
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (!action()) { CHECK(std::chrono::steady_clock::now() < deadline); std::this_thread::sleep_for(10ms); }
}
std::string digest(const std::string& value) { return mfq::cache::block_hash_hex(mfq::cache::sha256(value)); }
struct Environment {
    std::string name; std::optional<std::string> saved;
    static void set(const std::string& key, const std::optional<std::string>& value) {
#ifdef _WIN32
        _putenv_s(key.c_str(), value ? value->c_str() : "");
#else
        if (value) ::setenv(key.c_str(), value->c_str(), 1); else ::unsetenv(key.c_str());
#endif
    }
    Environment(std::string name, std::string value) : name(std::move(name)) {
        if (const auto* previous = std::getenv(this->name.c_str())) saved = previous;
        set(this->name, value);
    }
    ~Environment() { set(name, saved); }
};
struct Origin {
    httplib::Server http;
    int port = 0;
    std::thread thread;
    std::map<std::string, std::string> files;
    // Model cards and the official Tylogi repository are separate fixtures so downloads stay unaffected.
    std::map<std::string, std::string> cards, tylogi;
    std::vector<std::pair<std::string, std::int64_t>> hits{{"owner/alpha-tq-7b", 30}, {"owner/beta-gguf", 90}, {"owner/gamma-tq-13b", 10}};
    std::atomic<int> mode{0}, ranges{0}, requests{0}, active{0};
    std::string revision = std::string(40, 'a');
    // Modes: normal, disconnect, ignore Range, wrong Range, changed ETag, bad bytes, HTTP error, slow.
    Origin() {
        tylogi["model.tq"] = std::string(4096, 'q');
        files["config.json"] = R"({"model_type":"qwen3_5","vocab_size":8,"max_position_embeddings":8192})";
        const std::string header = R"({"model.embed_tokens.weight":{"dtype":"BF16","shape":[262144],"data_offsets":[0,524288]}})";
        const std::uint64_t size = header.size();
        files["model.safetensors"] = std::string(reinterpret_cast<const char*>(&size), sizeof(size)) + header + std::string(524288, 'x');
        files["notes/README.md"] = "fixture notes";
        files["skip.txt"] = "excluded";
        http.Get(R"(/api/models/owner/model/revision/[^/]+)", [&](const auto&, auto& response) {
            auto list = Json::array(); for (const auto& [name, bytes] : files)
                list.push_back({{"rfilename", name}, {"size", bytes.size()}, {"lfs", {{"sha256", digest(bytes)}, {"size", bytes.size()}}}});
            response.set_content(Json{{"sha", revision}, {"siblings", list}}.dump(), "application/json");
        });
        // The two vendor routes below keep ModelScope's own versioned prefix; our routes use /api.
        http.Get("/api/v1/models/owner/model/repo/files", [&](const auto&, auto& response) {
            auto list = Json::array(); for (const auto& [name, bytes] : files)
                list.push_back({{"Path", name}, {"Size", bytes.size()}, {"Sha256", digest(bytes)}, {"Type", "blob"}});
            response.set_content(Json{{"Code", 200}, {"Data", {{"Files", list}}}}.dump(), "application/json");
        });
        http.Get(R"(/owner/model/resolve/[^/]+/(.+))", [&](const auto& request, auto& response) { serve(request, response, request.matches[1]); });
        http.Get("/api/v1/models/owner/model/repo", [&](const auto& request, auto& response) { serve(request, response, request.get_param_value("FilePath")); });
        http.Get("/echo", [](const auto& request, auto& response) { response.set_content(request.get_header_value("Authorization"), "text/plain"); });
        http.Get("/api/models", [&](const auto& request, auto& response) {
            if (request.has_param("author")) {
                response.set_content(Json::array({{{"id", "Tylogi/tylogi-flash-next-tq-7b"}, {"downloads", 7}, {"likes", 3},
                    {"lastModified", "2024-05-01T00:00:00.000Z"}, {"trendingScore", 1}}}).dump(), "application/json");
                return;
            }
            // The first page advertises a Link cursor; the second page ends the search.
            const bool second = request.get_param_value("cursor") == "next";
            const std::size_t from = second ? 2 : 0, to = second ? hits.size() : 2;
            auto list = Json::array();
            for (std::size_t index = from; index < to; ++index)
                list.push_back({{"id", hits[index].first}, {"downloads", hits[index].second}, {"likes", 1},
                    {"lastModified", "2024-01-0" + std::to_string(index + 1) + "T00:00:00.000Z"}, {"trendingScore", 2}});
            if (to < hits.size()) response.set_header("Link", "<" + url() + "/api/models?cursor=next>; rel=\"next\"");
            response.set_content(list.dump(), "application/json");
        });
        http.Get("/openapi/v1/models", [&](const auto& request, auto& response) {
            const auto author = request.has_param("owner");
            const Json model = author ? Json{{"id", "Tylogi/tylogi-flash-next-tq-7b"}, {"downloads", 4}, {"likes", 1},
                {"last_modified", "2024-04-01 00:00:00"}} :
                Json{{"id", "owner/ms-tq-7b"}, {"downloads", 5}, {"likes", 2}, {"storage_size", 1024}, {"last_modified", "2024-02-01 00:00:00"}};
            response.set_content(Json{{"success", true}, {"data", {{"total_count", 1}, {"models", Json::array({model})}}}}.dump(), "application/json");
        });
        http.Get(R"(/api/models/Tylogi/([^/]+)/revision/[^/]+)", [&](const auto&, auto& response) {
            auto list = Json::array();
            for (const auto& [name, bytes] : tylogi)
                list.push_back({{"rfilename", name}, {"size", bytes.size()}, {"lfs", {{"sha256", digest(bytes)}, {"size", bytes.size()}}}});
            response.set_content(Json{{"sha", std::string(40, 'c')}, {"siblings", list}}.dump(), "application/json");
        });
        // Same ModelScope mock paths as above; they are not our route prefix.
        http.Get(R"(/api/v1/models/Tylogi/([^/]+)/repo/files)", [&](const auto&, auto& response) {
            auto list = Json::array();
            for (const auto& [name, bytes] : tylogi) list.push_back({{"Path", name}, {"Size", bytes.size()}, {"Sha256", digest(bytes)}, {"Type", "blob"}});
            response.set_content(Json{{"Code", 200}, {"Data", {{"Files", list}}}}.dump(), "application/json");
        });
        http.Get(R"(/Tylogi/([^/]+)/resolve/[^/]+/(.+))", [&](const auto& request, auto& response) { serve(request, response, request.matches[2], &tylogi); });
        http.Get(R"(/api/v1/models/Tylogi/([^/]+)/repo)", [&](const auto& request, auto& response) { serve(request, response, request.get_param_value("FilePath"), &tylogi); });
        http.Get("/loop", [](const auto&, auto& response) { response.set_redirect("/loop"); });
        http.set_post_routing_handler([&](const auto& request, auto& response) {
            if (mode == 3 && request.has_header("Range")) { response.headers.erase("Content-Range"); response.set_header("Content-Range", "bytes 0-1/2"); }
        });
    }
    void start() {
        http.new_task_queue = [] { return new httplib::ThreadPool(4); };
        port = http.bind_to_any_port("127.0.0.1"); CHECK(port > 0);
        thread = std::thread([&] { CHECK(http.listen_after_bind()); }); wait([&] { return http.is_running(); });
    }
    ~Origin() { http.stop(); if (thread.joinable()) thread.join(); }
    std::string url() const { return "http://127.0.0.1:" + std::to_string(port); }
    Hub::Network network() const { Hub::Network value; value.hf.url = value.ms.url = url(); return value; }
    void serve(const httplib::Request& request, httplib::Response& response, const std::string& name,
            const std::map<std::string, std::string>* source = nullptr) {
        ++requests;
        const std::map<std::string, std::string>* pool = source;
        if (!pool) pool = files.count(name) ? &files : cards.count(name) ? &cards : nullptr;
        if (!pool || !pool->count(name)) { response.status = 404; return; }
        const bool weight = name == "model.safetensors";
        const int behavior = weight ? mode.load() : 0;
        if (request.has_header("Range")) ++ranges;
        response.set_header("ETag", behavior == 4 ? "\"v2\"" : "\"v1\"");
        if (behavior == 6) { response.status = 503; response.set_content("not weights", "text/plain"); return; }
        if (behavior == 2) response.status = 200;
        auto bytes = pool->at(name); if (behavior == 5) bytes.back() ^= 1;
        ++active;
        response.set_content_provider(bytes.size(), "application/octet-stream",
            [bytes, behavior](std::size_t offset, std::size_t length, httplib::DataSink& sink) {
                if (behavior == 1 && offset >= 65536) return false;
                if (behavior == 7) std::this_thread::sleep_for(100ms);
                const auto count = std::min<std::size_t>(length, 16384);
                return sink.write(bytes.data() + offset, count);
            }, [&](bool) { --active; });
    }
};
Hub::Request request(const std::string& destination, Hub::Provider provider = Hub::Provider::hf) {
    Hub::Request value; value.source = {provider, "owner/model", {}}; value.destination = "models/" + destination;
    value.include = {"*.json", "*.safetensors", "notes/*"}; value.exclude = {"skip.*"}; value.workers = 2; return value;
}
void modules() {
    Directory temporary; Origin origin; origin.start(); Hub hub(temporary.root, origin.network());
    const auto repo = hub.info({Hub::Provider::hf, "owner/model", {}});
    CHECK(repo.source.revision == origin.revision && repo.files.size() == 4);
    const auto first = hub.download(request("first"));
    CHECK(first.files == 3 && server_test::read_file(first.path / "model.safetensors") == origin.files["model.safetensors"]);
    CHECK(!fs::exists(first.path / "skip.txt"));
    ModelCatalog catalog({temporary.root / "models"}); CHECK(catalog.list(true).size() == 1 && catalog.list(false).front().complete);
    error("download_destination_exists", [&] { hub.download(request("first")); });
    const auto ms = hub.download(request("ms", Hub::Provider::ms)); CHECK(ms.files == 3);
    for (const auto& destination : {"../escape", "/tmp/escape", "x/../escape", "x\\escape", "x/.tyloinfer-downloads/z", "x//z", "NUL.txt"}) {
        auto invalid = request("bad"); invalid.destination = destination;
        error("invalid_download_path", [&] { hub.download(invalid); });
    }
    auto invalid = request("bad"); invalid.workers = 17; error("invalid_download_workers", [&] { hub.download(invalid); });
    invalid = request("empty"); invalid.include = {"*.none"}; error("empty_download", [&] { hub.download(invalid); });
    invalid = request("filters"); invalid.include = {"[ab]*"}; error("invalid_download_filter", [&] { hub.download(invalid); });
    invalid = request("space"); invalid.expected_bytes = INT64_MAX; error("download_disk_full", [&] { hub.download(invalid); });
    origin.files["../outside"] = "escape";
    error("invalid_download_path", [&] { hub.info({Hub::Provider::hf, "owner/model", {}}); }); origin.files.erase("../outside");
    origin.files["config.json/subfile"] = "overlap";
    invalid = request("overlap"); invalid.include = {};
    error("invalid_download_path", [&] { hub.download(invalid); }); origin.files.erase("config.json/subfile");
    fs::create_directories(temporary.root / "external"); hub.configure(temporary.root / "external");
    { Hub restored(temporary.root, origin.network()); CHECK(restored.directory().actual == temporary.root / "external"); CHECK(restored.download(request("configured")).path == temporary.root / "external/configured"); }
    fs::rename(temporary.root / "download-directory.json", temporary.root / "saved.json");
    fs::create_directory(temporary.root / "download-directory.json");
    error("download_settings_save_failed", [&] { hub.configure({}); });
    CHECK(hub.directory().actual == temporary.root / "external");
    fs::remove(temporary.root / "download-directory.json"); fs::rename(temporary.root / "saved.json", temporary.root / "download-directory.json");
    hub.configure({});
    fs::create_directory_symlink(temporary.root / "external", temporary.root / "models/link");
    error("unsafe_download_path", [&] { hub.download(request("link/escape")); });
    for (int behavior : {0, 2, 3, 4}) {
        const auto name = "resume-" + std::to_string(behavior); auto req = request(name);
        origin.mode = 1;
        error("download_connection_failed", [&] { hub.download(req); });
        CHECK(!fs::exists(temporary.root / "models" / name));
        const int ranges = origin.ranges;
        origin.mode = behavior;
        if (behavior == 3 || behavior == 4) {
            error(behavior == 3 ? "invalid_download_range" : "download_changed", [&] { hub.download(req); }); origin.mode = 0;
        }
        Hub restarted(temporary.root, origin.network()); const auto done = restarted.download(req);
        CHECK(origin.ranges > ranges && server_test::read_file(done.path / "model.safetensors") == origin.files["model.safetensors"]);
    }
    origin.mode = 5; error("download_checksum_mismatch", [&] { hub.download(request("checksum")); });
    CHECK(!fs::exists(temporary.root / "models/checksum")); origin.mode = 0; hub.download(request("checksum"));
    origin.mode = 6; error("hub_http_error", [&] { hub.download(request("http-error")); }); origin.mode = 0;
    std::atomic<bool> cancelled{false}; origin.mode = 7;
    error("download_cancelled", [&] { hub.download(request("cancel"), {[&] { return cancelled.load(); },
        [&](auto bytes, auto, auto) { if (bytes > 32768) cancelled = true; }}); });
    CHECK(!fs::exists(temporary.root / "models/cancel")); wait([&] { return origin.active == 0; });
    origin.mode = 0; hub.download(request("cancel"));
    origin.mode = 1; error("download_connection_failed", [&] { hub.download(request("changed")); });
    origin.revision.assign(40, 'b'); origin.mode = 0;
    error("download_changed", [&] { hub.download(request("changed")); }); origin.revision.assign(40, 'a');
    const auto stage = temporary.root / "models/.tyloinfer-downloads" / digest("disk");
    fs::create_directories(stage / "parts/1.partial");
    error("download_write_failed", [&] { hub.download(request("disk")); });
    const auto hidden = temporary.root / "models/.tyloinfer-downloads/fake";
    write(hidden / "config.json", origin.files["config.json"]); write(hidden / "model.safetensors", origin.files["model.safetensors"]);
    for (const auto& artifact : catalog.list(true)) CHECK(path_utf8(artifact.path).find(".tyloinfer-downloads") == std::string::npos);
    for (const auto& child : catalog.browse(ModelCatalog::DirectorySource{{}, temporary.root / "models"}).children) CHECK(child.name != ".tyloinfer-downloads");
    auto unsafe = origin.network(); unsafe.hf.url = "file:///tmp"; error("invalid_hub_endpoint", [&] { Hub rejected(temporary.root, unsafe); });
}
void network(const fs::path& openssl) {
    Directory temporary(fs::current_path()); Origin target, origin;
    target.start();
    origin.http.Get("/redirect", [&](const auto&, auto& response) { response.set_redirect(target.url() + "/echo"); });
    origin.http.Get("/local", [&](const auto&, auto& response) { response.set_redirect("/echo"); }); origin.start();
    hub::Fetch fetch{{origin.url(), "fixture-token"}, {}, true};
    CHECK(hub::get(fetch, "/local", {}) == "Bearer fixture-token");
    CHECK(hub::get(fetch, "/redirect", {}).empty());
    error("download_connection_failed", [&] { hub::get(fetch, "/loop", {}); });
    Origin proxy; proxy.http.Get(".*", [](const auto&, auto& response) { response.set_content("proxy", "text/plain"); }); proxy.start();
    {
        Environment bypass("no_proxy", ""), upper("NO_PROXY", ""), route("http_proxy", proxy.url());
        fetch.direct = false; CHECK(hub::get(fetch, "/echo", {}) == "proxy");
        fetch.direct = true; CHECK(hub::get(fetch, "/echo", {}) == "Bearer fixture-token");
    }
    const auto cert = temporary.root / "cert.pem", key = temporary.root / "key.pem";
    const auto config = temporary.root / "openssl.cnf";
    write(config, "[req]\ndistinguished_name=dn\n[dn]\n");
    server_test::Process generate(openssl, {"req", "-config", path_utf8(config), "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=localhost",
        "-addext", "subjectAltName=DNS:localhost", "-keyout", path_utf8(key), "-out", path_utf8(cert)}, temporary.root / "cert.log");
    if (generate.wait() != 0) throw std::runtime_error("test certificate generation failed: " + generate.output());
    int port;
    { httplib::Server reserve; port = reserve.bind_to_any_port("127.0.0.1"); CHECK(port > 0);
      std::thread listener([&] { reserve.listen_after_bind(); }); wait([&] { return reserve.is_running(); }); reserve.stop(); listener.join(); }
    write(temporary.root / "ok", "HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\nok");
    write(temporary.root / "redirect", "HTTP/1.0 302 Found\r\nLocation: " + origin.url() + "/echo\r\nContent-Length: 0\r\n\r\n");
    const auto prefix = "/" + path_utf8(temporary.root.filename()) + "/";
    server_test::Process tls(openssl, {"s_server", "-quiet", "-HTTP", "-http_server_binmode", "-accept", "127.0.0.1:" + std::to_string(port), "-cert", path_utf8(cert), "-key", path_utf8(key)}, temporary.root / "tls.log");
    fetch = {{"https://localhost:" + std::to_string(port), {}}, cert, true};
    wait([&] { try { return hub::get(fetch, prefix + "ok", {}) == "ok"; } catch (const Hub::Error&) { if (!tls.running()) throw std::runtime_error(tls.output()); return false; } });
    error("download_connection_failed", [&] { hub::get(fetch, prefix + "redirect", {}); });
    fetch.ca.clear(); error("download_connection_failed", [&] { hub::get(fetch, prefix + "ok", {}); });
    fetch.ca = cert; fetch.remote.url = "https://127.0.0.1:" + std::to_string(port);
    error("download_connection_failed", [&] { hub::get(fetch, prefix + "ok", {}); });
}
struct Server {
    std::unique_ptr<server_test::Process> process;
    std::unique_ptr<httplib::Client> client;
    Server(const fs::path& executable, const fs::path& root, const Origin& origin) {
        fs::create_directories(root);
        process = std::make_unique<server_test::Process>(executable,
            std::vector<std::string>{"--port", "0", "--data-dir", path_utf8(root), "--exit-on-stdin-close"}, root / "server.log",
            std::map<std::string,std::string>{{"HF_ENDPOINT", origin.url()}, {"MODELSCOPE_ENDPOINT", origin.url()}, {"TYLOINFER_SERVER_API_KEY", "hub-test-root"}, {"TYLOINFER_SERVER_MODEL_DIRS", ""}});
        int port = 0;
        wait([&] {
            if (!process->running()) throw std::runtime_error(process->output());
            std::istringstream lines(process->output()); std::string line;
            while (std::getline(lines, line)) { const auto event = Json::parse(line, nullptr, false); if (event.is_object() && event.value("event", "") == "listening") port = event.at("port"); }
            return port > 0;
        });
        client = std::make_unique<httplib::Client>("127.0.0.1", port); client->set_default_headers({{"Authorization", "Bearer hub-test-root"}});
    }
    ~Server() { process->close_input(); try { process->wait(); } catch (...) {} }
    Json post(const std::string& path, const Json& body, int status = 202) {
        const auto result = client->Post(path, body.dump(), "application/json");
        if (!result || result->status != status) throw std::runtime_error(path + ": " + (result ? result->body : "connection failed"));
        return Json::parse(result->body);
    }
    Json get(const std::string& path) {
        const auto result = client->Get(path);
        if (!result || result->status != 200) throw std::runtime_error(path + ": " + (result ? std::to_string(result->status) + " " + result->body : "connection failed"));
        return Json::parse(result->body);
    }
    Json complete(const std::string& id) {
        Json job; wait([&] { job = get("/api/jobs/" + id); const auto state = job["status"]; return state == "succeeded" || state == "failed" || state == "cancelled" || state == "interrupted"; }); return job;
    }
};
void routes(const fs::path& executable) {
    Directory temporary; Origin origin; origin.start(); std::string interrupted;
    const Json payload{{"repo_id", "owner/model"}, {"destination", "models/api"}, {"include", {"*.json", "*.safetensors"}}, {"max_workers", 2}};
    {
        Server server(executable, temporary.root, origin);
        CHECK(server.get("/api/hub/download-directory")["directory"].is_null());
        const auto selected = temporary.root / "selected"; fs::create_directory(selected);
        const auto folder = server.get("/api/models/directories?path=" + path_utf8(selected));
        auto configured = server.client->Put("/api/hub/download-directory", Json{{"directory_id", folder.at("current_id")}}.dump(), "application/json");
        CHECK(configured && configured->status == 200 && Json::parse(configured->body)["actual_directory"] == path_utf8(selected));
        configured = server.client->Put("/api/hub/download-directory", R"({"directory_id":null})", "application/json");
        CHECK(configured && configured->status == 200 && Json::parse(configured->body)["directory"].is_null());
        const auto unsupported = server.client->Put("/api/hub/download-directory", R"({"evict_prefix_cache":true})", "application/json");
        CHECK(unsupported && unsupported->status == 501);
        for (const auto* scope : {"inference", "jobs", "models"}) {
            const auto key = server.post("/api/auth/keys", {{"name", scope}, {"scopes", {scope}}}, 201);
            server.client->set_default_headers({{"Authorization", "Bearer " + key.at("token").get<std::string>()}});
            auto result = server.client->Get("/api/jobs/kinds"); CHECK(result && result->status == (std::string(scope) == "jobs" ? 200 : 403));
            result = server.client->Get("/api/hub/download-directory"); CHECK(result && result->status == (std::string(scope) == "models" ? 200 : 403));
            result = server.client->Get("/api/hub/models?provider=huggingface&limit=1");
            CHECK(result && result->status == (std::string(scope) == "inference" ? 200 : 403));
            server.client->set_default_headers({{"Authorization", "Bearer hub-test-root"}});
        }
        CHECK(server.get("/api/jobs/kinds")["data"].size() == 2);
        auto info = server.get("/api/hub/models/huggingface/owner/model"); CHECK(info["revision"] == origin.revision && info["variants"].size() == 1 && info["downloads"].is_null());
        CHECK(server.post("/api/hub/resolve", {{"reference", "https://modelscope.cn/models/owner/model"}}, 200)["provider"] == "modelscope");
        server.post("/api/hub/resolve", {{"reference", "https://evil.invalid/owner/model"}}, 400);
        // Hub search, the merged browser, the official catalog, model cards and source probes.
        const auto found = server.get("/api/hub/models?provider=huggingface&query=model&limit=5");
        CHECK(found["data"].size() == 2 && found["next_cursor"].is_null() && found["data"][0]["repo_id"] == "owner/alpha-tq-7b");
        CHECK(found["data"][0]["formats"] == Json::array({"tq"}) && found["data"][0]["downloads"] == 30 && found["data"][0]["total_bytes"] == 0);
        CHECK(found["data"][0]["author"] == "owner" && found["data"][0]["source_url"] == "https://huggingface.co/owner/alpha-tq-7b");
        CHECK(found["data"][0]["updated_at"] == "2024-01-01T00:00:00.000Z" && found["data"][0]["trending_score"] == 2);
        const auto scoped = server.get("/api/hub/models?provider=modelscope&query=model&limit=5");
        CHECK(scoped["data"][0]["repo_id"] == "owner/ms-tq-7b" && scoped["data"][0]["total_bytes"] == 1024);
        const auto browsed = server.get("/api/hub/models?provider=huggingface&browse=true&formats=tq,gguf&sort=downloads&limit=1");
        CHECK(browsed["data"].size() == 1 && browsed["data"][0]["repo_id"] == "owner/beta-gguf" && !browsed["next_cursor"].is_null());
        const auto onward = server.get("/api/hub/models?provider=huggingface&browse=true&formats=tq,gguf&sort=downloads&limit=1&cursor=" +
            browsed["next_cursor"].get<std::string>());
        CHECK(onward["data"].size() == 1 && onward["data"][0]["repo_id"] == "owner/alpha-tq-7b");
        const auto expired = server.client->Get("/api/hub/models?provider=huggingface&browse=true&cursor=aaaaaaaaaaaaaaaa");
        CHECK(expired && expired->status == 400 && Json::parse(expired->body)["error"]["code"] == "search_cursor_expired");
        for (const auto* bad : {"provider=nope", "limit=0", "sort=trending", "formats=tq,bogus", "provider=modelscope&direction=asc&browse=true"}) {
            const auto rejected = server.client->Get("/api/hub/models?" + std::string(bad));
            CHECK(rejected && rejected->status == 400);
        }
        Json catalog;
        wait([&] { catalog = server.get("/api/hub/official"); return !catalog["refreshing"].get<bool>(); });
        CHECK(catalog["data"].size() == 1 && catalog["system"]["memory_pools"].size() == 1 && catalog["system"]["cpu_cores"] >= 1);
        CHECK(catalog["system"]["backend"] == "cpu" && catalog["system"]["physical_memory_bytes"].is_number());
        const auto& official = catalog["data"][0];
        CHECK(official["name"] == "tylogi-flash-next" && official["sources"].size() == 2 && official["variants"].size() == 1);
        CHECK(official["id"].get<std::string>().rfind("tylogi-", 0) == 0 && official["configuration"]["status"] == "unknown");
        CHECK(official["selected_source"]["provider"] == "huggingface" && official["revision"] == std::string(40, 'c'));
        CHECK(official["downloads"] == 7 && official["updated_at"] == "2024-05-01T00:00:00.000Z");
        CHECK(official["description_zh"].get<std::string>().find("PLE") != std::string::npos);
        CHECK(official["sources"][1]["provider"] == "modelscope" && official["sources"][1]["available"] == true);
        CHECK(server.get("/api/hub/official?refresh=true")["refreshing"] == true);
        const auto bad_refresh = server.client->Get("/api/hub/official?refresh=maybe"); CHECK(bad_refresh && bad_refresh->status == 400);
        origin.cards["README.md"] = "# fixture card";
        CHECK(server.get("/api/hub/models/huggingface/owner/model/readme")["content"] == "# fixture card");
        CHECK(server.get("/api/hub/models/modelscope/owner/model/readme?revision=master")["content"] == "# fixture card");
        const auto absent = server.client->Get("/api/hub/models/huggingface/owner/missing/readme");
        CHECK(absent && absent->status == 200 && Json::parse(absent->body)["content"].is_null());
        const auto bad_card = server.client->Get("/api/hub/models/gitlab/owner/model/readme"); CHECK(bad_card && bad_card->status == 400);
        const auto probed = server.post("/api/hub/download-sources/probe",
            Json{{"sources", Json::array({Json{{"provider", "huggingface"}, {"repo_id", "owner/model"}}})}}, 200);
        CHECK(probed["data"].size() == 1 && probed["data"][0]["repo_id"] == "owner/model");
        CHECK(probed["data"][0]["route"] == "direct" && probed["data"][0]["bytes_per_second"] > 0);
        server.post("/api/hub/download-sources/probe", Json{{"sources", Json::array()}}, 400);
        server.post("/api/hub/download-sources/probe", Json{{"sources", Json::array({Json{{"provider", "gitlab"}, {"repo_id", "owner/model"}}})}}, 400);
        server.post("/api/jobs", {{"kind", "unknown"}, {"payload", payload}}, 501);
        auto invalid = payload; invalid["max_workers"] = 17;
        server.post("/api/jobs", {{"kind", "download.huggingface"}, {"payload", invalid}}, 400);
        const auto started = server.post("/api/jobs", {{"kind", "download.huggingface"}, {"payload", payload}});
        const auto id = started.at("id").get<std::string>(); const auto job = server.complete(id);
        CHECK(job["status"] == "succeeded" && job["progress"] == 1 && job["result"]["registration"] == "registered");
        CHECK(job["progress_data"]["downloaded_bytes"] == job["progress_data"]["total_bytes"] && job["progress_data"]["files_completed"] == 2);
        CHECK(server.get("/api/models?refresh=true")["data"].size() == 1);
        const auto events = server.get("/api/jobs/" + id + "/events");
        bool progress = false; for (const auto& event : events["data"])
            if (event["type"] == "progress" && event["data"].contains("downloaded_bytes")) { progress = true; CHECK(!event["progress"].is_null()); }
        CHECK(progress); server.post("/api/jobs/" + id + "/retry", {}, 409);
        origin.mode = 7; auto cancel = payload; cancel["destination"] = "models/cancel-api";
        const auto cancel_id = server.post("/api/jobs", {{"kind", "download.huggingface"}, {"payload", cancel}}).at("id").get<std::string>();
        // Confirm that some weights arrived so a retry after cancellation must send Range.
        wait([&] { const auto job = server.get("/api/jobs/" + cancel_id); return job["progress_data"].is_object() && job["progress_data"].value("downloaded_bytes", 0) > 32768; });
        server.post("/api/jobs/" + cancel_id + "/cancel", {}, 200);
        CHECK(server.complete(cancel_id)["status"] == "cancelled"); CHECK(!fs::exists(temporary.root / "models/cancel-api"));
        origin.mode = 0;
        const auto hf_ranges = origin.ranges.load();
        const auto retry = server.post("/api/jobs/" + cancel_id + "/retry", {}); CHECK(retry["id"] != cancel_id);
        CHECK(server.complete(retry["id"])["status"] == "succeeded");
        CHECK(origin.ranges > hf_ranges);
        origin.mode = 7; cancel["destination"] = "models/restart";
        interrupted = server.post("/api/jobs", {{"kind", "download.modelscope"}, {"payload", cancel}}).at("id");
        wait([&] { const auto job = server.get("/api/jobs/" + interrupted); return job["progress_data"].is_object() && job["progress_data"].value("downloaded_bytes", 0) > 32768; });
#ifndef _WIN32
        server.process->signal(SIGKILL); CHECK(server.process->wait() == -SIGKILL);
#else
        server.post("/api/jobs/" + interrupted + "/cancel", {}, 200);
        CHECK(server.complete(interrupted)["status"] == "cancelled");
#endif
    }
    origin.mode = 0;
    {
        Server server(executable, temporary.root, origin);
#ifndef _WIN32
        CHECK(server.get("/api/jobs/" + interrupted)["status"] == "interrupted");
#endif
        const auto ms_ranges = origin.ranges.load();
        const auto retry = server.post("/api/jobs/" + interrupted + "/retry", {});
        CHECK(server.complete(retry["id"])["status"] == "succeeded");
        CHECK(origin.ranges > ms_ranges);
        CHECK(server.get("/api/models?refresh=true")["data"].size() == 3);
        for (const auto* provider : {"huggingface", "modelscope"}) {
            const auto info = server.get(std::string("/api/hub/models/") + provider + "/owner/model");
            const auto label = info["variants"][0]["label"].get<std::string>();
            auto download = payload;
            download["destination"] = std::string("models/") + provider + "/owner/model/" +
                std::regex_replace(label, std::regex("[^A-Za-z0-9_.-]+"), "-");
            const auto job = server.post("/api/jobs", {{"kind", std::string("download.") + provider}, {"payload", download}});
            CHECK(server.complete(job["id"])["status"] == "succeeded");
        }
        CHECK(server.get("/api/models?refresh=true")["data"].size() == 5);
        write(temporary.root / "models" / "reuse" / "model.tq", origin.tylogi["model.tq"]);
        const auto reuse = server.post("/api/hub/download-reuse",
            Json{{"provider", "huggingface"}, {"repo_id", "Tylogi/tylogi-flash-next-tq-7b"}, {"destination", "models/reuse"}}, 200);
        CHECK(reuse["reusable_files"].size() == 1 && reuse["reusable_files"]["model.tq"] == origin.tylogi["model.tq"].size());
        const auto missing = server.post("/api/hub/download-reuse",
            Json{{"provider", "huggingface"}, {"repo_id", "Tylogi/tylogi-flash-next-tq-7b"}, {"destination", "models/absent"}}, 200);
        CHECK(missing["reusable_files"].empty());
        server.post("/api/hub/download-reuse", Json{{"provider", "huggingface"}, {"repo_id", "Tylogi/tylogi-flash-next-tq-7b"}, {"destination", ""}}, 400);
        server.post("/api/hub/download-reuse", Json{{"provider", "huggingface"}, {"repo_id", "owner/model"}, {"destination", "models/reuse"}, {"bogus", 1}}, 400);
        server.client->set_default_headers({}); const auto denied = server.client->Post("/api/jobs", payload.dump(), "application/json"); CHECK(denied && denied->status == 401);
    }
}
}
int main(int argc, char** argv) {
    try {
        CHECK(argc == 3); Environment bypass("no_proxy", "*"); Environment upper("NO_PROXY", "*");
        modules(); network(fs::u8path(argv[2])); routes(fs::u8path(argv[1]));
        std::cout << "Hub downloads, resume, TLS/proxy and HTTP job lifecycle passed\n"; return 0;
    } catch (const std::exception& failure) { std::cerr << failure.what() << '\n'; return 1; }
}

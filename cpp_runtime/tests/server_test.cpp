#include "httplib.h"
#include "nlohmann/json.hpp"
#include "schema.h"
#include "server_test_process.h"
#include "mfq_paged_prefix_cache.h"

#include <sqlite3.h>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <random>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sddl.h>
#include <sys/locking.h>
#include <sys/stat.h>
#else
#include <sys/file.h>
#endif

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;
using server_test::Process;
using server_test::read_file;
int checks = 0;
#define CHECK(value) do { ++checks; if (!(value)) throw std::runtime_error( \
    std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #value); } while (false)

fs::path executable;

void write_file(const fs::path& path, const std::string& content) {
    std::ofstream stream(path, std::ios::binary);
    stream << content;
    CHECK(stream.good());
}

template <typename T> void write_scalar(std::ofstream& stream, T value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_model_string(std::ofstream& stream, const std::string& value) {
    write_scalar(stream, static_cast<std::uint32_t>(value.size()));
    stream.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void write_mfq(const fs::path& path) {
    const std::string config = R"({"model_type":"qwen3_5","max_position_embeddings":8192})";
    std::ofstream stream(path, std::ios::binary);
    stream.write("MFQ1", 4);
    write_scalar<std::uint32_t>(stream, 2);
    write_model_string(stream, "qwen3_5");
    write_scalar<std::uint32_t>(stream, 0);
    write_scalar<std::uint32_t>(stream, 2);
    write_model_string(stream, "model.token_embedding.weight");
    write_model_string(stream, "BF16");
    write_scalar<std::uint64_t>(stream, 14);
    write_model_string(stream, "__mfq_asset__/model_config.json");
    write_model_string(stream, "BLOB");
    write_scalar<std::uint64_t>(stream, config.size());
    write_scalar<std::uint32_t>(stream, 1);
    write_scalar<std::int64_t>(stream, 1);
    write_scalar<std::uint16_t>(stream, 0);
    stream << config;
    CHECK(stream.good());
}

void write_hf(const fs::path& path) {
    fs::create_directories(path);
    write_file(path / "config.json", R"({"model_type":"qwen3_5","text_config":{"max_position_embeddings":4096}})");
    const std::string header = R"({"model.embed_tokens.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}})";
    std::ofstream stream(path / "model.safetensors", std::ios::binary);
    write_scalar<std::uint64_t>(stream, header.size());
    stream << header;
    write_scalar<std::uint16_t>(stream, 0);
    CHECK(stream.good());
}

/** Writes a Safetensors shard from {name,dtype,shape,bytes} entries with sequential offsets. */
void write_safetensors(const fs::path& path, const json& entries) {
    json header = json::object();
    std::size_t offset = 0;
    for (const auto& entry : entries) {
        const auto bytes = entry.at("bytes").get<std::size_t>();
        header[entry.at("name").get<std::string>()] = {{"dtype", entry.at("dtype")}, {"shape", entry.at("shape")},
            {"data_offsets", {offset, offset + bytes}}};
        offset += bytes;
    }
    const auto text = header.dump();
    std::ofstream stream(path, std::ios::binary);
    write_scalar<std::uint64_t>(stream, text.size());
    stream << text;
    const auto payload = std::string(offset, '\0');
    stream.write(payload.data(), static_cast<std::streamsize>(offset));
    CHECK(stream.good());
}

class Database {
public:
    explicit Database(const fs::path& path) {
        if (sqlite3_open(path.u8string().c_str(), &database_) != SQLITE_OK) {
            sqlite3_close(database_);
            throw std::runtime_error("cannot open test database");
        }
        sqlite3_busy_timeout(database_, 5000);
    }
    ~Database() { sqlite3_close(database_); }
    void execute(const std::string& sql) {
        if (sqlite3_exec(database_, sql.c_str(), nullptr, nullptr, nullptr) != SQLITE_OK)
            throw std::runtime_error(sqlite3_errmsg(database_));
    }
    std::string scalar(const std::string& sql) {
        sqlite3_stmt* statement = nullptr;
        CHECK(sqlite3_prepare_v2(database_, sql.c_str(), -1, &statement, nullptr) == SQLITE_OK);
        const auto result = sqlite3_step(statement);
        std::string value;
        if (result == SQLITE_ROW && sqlite3_column_text(statement, 0))
            value = reinterpret_cast<const char*>(sqlite3_column_text(statement, 0));
        sqlite3_finalize(statement);
        CHECK(result == SQLITE_ROW);
        return value;
    }
private:
    sqlite3* database_ = nullptr;
};

struct Response {
    int status;
    httplib::Headers headers;
    std::string body;
    json document() const { return json::parse(body); }
    std::string header(const char* name) const {
        const auto found = headers.find(name);
        return found == headers.end() ? "" : found->second;
    }
};

struct Server {
    Process& process;
    std::string host;
    int port;
    Response raw(const std::string& method, const std::string& path, httplib::Headers headers = {}, const std::string& body = {}) {
        httplib::Client client(host == "0.0.0.0" ? "127.0.0.1" : host == "::" ? "::1" : host, port);
        client.set_connection_timeout(2);
        client.set_read_timeout(2);
        client.set_path_encode(false);
        httplib::Request request;
        request.method = method;
        request.path = path;
        request.headers = std::move(headers);
        request.body = body;
        const auto response = client.send(request);
        if (!response) throw std::runtime_error("HTTP request failed: " + process.output());
        return {response->status, response->headers, response->body};
    }
    Response get(const std::string& path, httplib::Headers headers = {}) { return raw("GET", path, std::move(headers)); }
    Response send(const std::string& method, const std::string& path, const json& body, httplib::Headers headers = {}) {
        headers.emplace("Content-Type", "application/json");
        return raw(method, path, std::move(headers), body.dump());
    }
    json identity() { return get("/health/identity").document(); }
    void stop() { process.close_input(); CHECK(process.wait() == 0); }
};

class Fixture {
public:
    Fixture() {
        static unsigned sequence = 0;
        root = fs::temp_directory_path() / ("tyloinfer-server-native-test-" + std::to_string(std::random_device{}()) + "-" + std::to_string(++sequence));
        CHECK(fs::create_directory(root));
        data = root / "data";
    }
    ~Fixture() {
        for (auto& child : children_) {
            child.first->close_input();
            if (!child.second && child.first->running()) {
#ifndef _WIN32
                child.first->signal(SIGTERM);
#endif
            }
            if (child.second) try { child.first->wait(); } catch (...) {}
        }
        children_.clear();
        std::error_code ignored;
        fs::permissions(data, fs::perms::owner_all, fs::perm_options::add, ignored);
        fs::remove_all(root, ignored);
    }
    Process& launch(const fs::path& directory, std::vector<std::string> options = {},
            std::map<std::string, std::string> environment = {}, bool owned = false) {
        std::vector<std::string> arguments{"--port", "0", "--data-dir", directory.u8string()};
        if (owned) arguments.push_back("--exit-on-stdin-close");
        arguments.insert(arguments.end(), options.begin(), options.end());
        auto child = std::make_unique<Process>(executable, arguments, root / ("child-" + std::to_string(children_.size()) + ".log"), environment);
        auto& result = *child;
        children_.emplace_back(std::move(child), owned);
        return result;
    }
    Server start_at(const fs::path& directory, std::vector<std::string> options = {},
            std::map<std::string, std::string> environment = {}, bool owned = true) {
        auto& child = launch(directory, std::move(options), std::move(environment), owned);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < deadline) {
            if (!child.running()) throw std::runtime_error("server exited during startup: " + child.output());
            const auto log = child.output();
            const auto end = log.find('\n');
            if (end != std::string::npos) {
                const auto event = json::parse(log.substr(0, end));
                Server server{child, event.at("host"), event.at("port")};
                CHECK(server.get("/health").status == 200);
                return server;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        throw std::runtime_error("server startup timed out: " + child.output());
    }
    Server start(std::vector<std::string> options = {}, std::map<std::string, std::string> environment = {}, bool owned = true) {
        return start_at(data, std::move(options), std::move(environment), owned);
    }
    fs::path root, data;
private:
    std::vector<std::pair<std::unique_ptr<Process>, bool>> children_;
};

httplib::Headers bearer(const std::string& token) { return {{"Authorization", "Bearer " + token}}; }
const httplib::Headers admin = bearer("example-root");
const std::map<std::string, std::string> root_environment{{"TYLOINFER_SERVER_API_KEY", "example-root"}};

class PartialRequest {
public:
    explicit PartialRequest(int port) {
        socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
        CHECK(socket_ != INVALID_SOCKET);
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<std::uint16_t>(port));
        inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
        CHECK(::connect(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
        const std::string request = "GET /health HTTP/1.1\r\n";
        CHECK(::send(socket_, request.data(), static_cast<int>(request.size()), 0) == static_cast<int>(request.size()));
    }
    ~PartialRequest() {
#ifdef _WIN32
        ::closesocket(socket_);
#else
        ::close(socket_);
#endif
    }
    PartialRequest(const PartialRequest&) = delete;
    PartialRequest& operator=(const PartialRequest&) = delete;
private:
    socket_t socket_ = INVALID_SOCKET;
};

void prefix_cache() {
    Fixture f;
    auto server = f.start({}, root_environment);
    const auto wait_job = [&](const json& accepted) {
        for (int attempt = 0; attempt < 500; ++attempt) {
            const auto job = server.get("/api/jobs/" + accepted.at("operation_id").get<std::string>(), admin).document();
            if (job.at("status") != "queued" && job.at("status") != "running") return job;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        throw std::runtime_error("memory policy job did not finish");
    };
    // Synthesize the engine's on-disk layout: one namespace, a two-block complete chain, one token sidecar.
    const fs::path root = fs::temp_directory_path() / ("tyloinfer-prefix-test-" + std::to_string(std::random_device{}()));
    const std::string space(64, '1'), parent(64, '2'), child(64, '3');
    const auto bytes_of = [](const std::string& hex) {
        std::string raw(hex.size() / 2, '\0');
        for (std::size_t index = 0; index < raw.size(); ++index)
            raw[index] = static_cast<char>(std::stoul(hex.substr(index * 2, 2), nullptr, 16));
        return raw;
    };
    const auto block_file = [&](const std::string& digest, const std::string& parent_digest, std::uint32_t tokens, std::uint32_t size) {
        const fs::path path = root / space / digest.substr(0, 2) / (digest + ".tqkv");
        fs::create_directories(path.parent_path());
        std::string raw(156, '\0');
        std::memcpy(raw.data(), "MFQKVB1\0", 8);
        std::uint32_t version = 1;
        std::memcpy(raw.data() + 8, &version, 4);
        std::memcpy(raw.data() + 12, bytes_of(space).data(), 32);
        std::memcpy(raw.data() + 44, bytes_of(digest).data(), 32);
        std::memcpy(raw.data() + 76, bytes_of(parent_digest).data(), 32);
        std::memcpy(raw.data() + 108, &size, 4);
        std::memcpy(raw.data() + 112, &tokens, 4);
        const std::uint64_t payload = 8;
        std::memcpy(raw.data() + 116, &payload, 8);
        std::ofstream(path, std::ios::binary) << raw << std::string(8, '\x7f');
        return path;
    };
    const fs::path parent_block = block_file(parent, std::string(64, '0'), 2, 4);
    const fs::path child_block = block_file(child, parent, 2, 4);
    {
        const std::int64_t tokens[2] = {72, 101};
        const std::string payload(reinterpret_cast<const char*>(tokens), sizeof tokens);
        std::string raw(108, '\0');
        std::memcpy(raw.data(), "MFQTXT1\0", 8);
        std::memcpy(raw.data() + 8, bytes_of(space).data(), 32);
        std::memcpy(raw.data() + 40, bytes_of(child).data(), 32);
        const std::uint32_t count = 2;
        std::memcpy(raw.data() + 72, &count, 4);
        const auto checksum = mfq::cache::sha256(payload);
        std::memcpy(raw.data() + 76, checksum.data(), 32);
        std::ofstream(fs::path(child_block) += ".tokens", std::ios::binary) << raw << payload;
        std::ofstream(root / space / "identity.txt") << "mfq-cache-identity-v1\nnamespace=" << space
            << "\narchitecture=test\ncodec=test\ncontext=4096\nmodel_path=/nonexistent/model\nsource=0:test-model:0:0\n";
    }
    const auto accepted = server.send("PUT", "/api/runtime/memory-policy", {{"prefix_directory", root.string()}}, admin);
    CHECK(accepted.status == 202 && wait_job(accepted.document()).at("status") == "succeeded");

    const auto summary = server.get("/api/runtime/cache/entries", admin).document();
    CHECK(summary.at("total_blocks") == 2 && summary.at("data").size() == 1 && summary.at("can_clear").get<bool>());
    const auto& group = summary.at("data").at(0);
    CHECK(group.at("id") == space && group.at("blocks") == 2 && group.at("prefixes") == 1 &&
        group.at("text_blocks") == 1 && group.at("max_prefix_tokens") == 4 && group.at("incomplete_prefixes") == 0 &&
        group.at("architecture") == "test" && group.at("context_size") == 4096);
    const auto detail = server.get("/api/runtime/cache/entries?namespace=" + space + "&chain_filter=complete&sort=length", admin).document();
    CHECK(detail.at("matched_blocks") == 2 && detail.at("blocks").size() == 2);
    CHECK(detail.at("blocks").at(0).at("id") == child && detail.at("blocks").at(0).at("prefix_tokens") == 4 &&
        detail.at("blocks").at(1).at("id") == parent);
    const auto missing_text = server.get("/api/runtime/cache/entries?namespace=" + space + "&text_filter=missing", admin).document();
    CHECK(missing_text.at("matched_blocks") == 1 && missing_text.at("blocks").at(0).at("id") == parent);
    // The text decoder needs the model that wrote the cache; the synthetic identity has none.
    const auto text = server.get("/api/runtime/cache/entries/" + space + "/" + child + "/text", admin).document();
    CHECK(text.at("available") == false && text.at("reason") == "tokenizer_unavailable");
    CHECK(server.get("/api/runtime/cache/entries?namespace=zz", admin).status == 400);

    CHECK(server.send("POST", "/api/runtime/cache/clear", json::object(), admin).status == 503);
    CHECK(server.send("POST", "/api/runtime/cache/trim", {{"target_bytes", 0}}, admin).status == 503);
    CHECK(server.send("POST", "/api/runtime/cache/trim", {{"target_bytes", -1}}, admin).status == 400);
    const auto purged = server.send("POST", "/api/runtime/cache/purge", {{"namespace", space}}, admin);
    CHECK(purged.status == 200 && purged.document().at("removed_blocks") == 2);
    CHECK(server.get("/api/runtime/cache/entries", admin).document().at("total_blocks") == 0);
    CHECK(server.send("POST", "/api/runtime/cache/purge", {{"namespace", space}}, admin).status == 404);
    fs::remove_all(root);
    server.stop();
}

void media_uploads() {
    Fixture f;
    auto server = f.start({}, root_environment);
    const std::string picture{"tyloinfer-media-check"};
    const std::string digest{"2b03e8c6dff53e6694f0f92589500c61062a1789d8e5d5b65714f64cf2c000a7"};
    auto upload = [&](const char* mime, const std::string& body, const std::string& sha, bool authorized = true) {
        httplib::Headers headers{{"Content-Type", mime}};
        if (!sha.empty()) headers.emplace("X-Content-SHA256", sha);
        if (authorized) headers.insert(admin.begin(), admin.end());
        return server.raw("POST", "/api/media", headers, body);
    };
    CHECK(upload("image/png", picture, digest, false).status == 401);
    const auto created = upload("image/png", picture, digest);
    CHECK(created.status == 201);
    const auto resource = created.document();
    CHECK(resource.at("media").at("sha256") == digest && resource.at("media").at("mime_type") == "image/png" &&
        resource.at("media").at("byte_size") == picture.size() &&
        resource.at("media").at("id").get<std::string>().size() == 36 && !resource.at("created_at").is_null());
    const auto id = resource.at("media").at("id").get<std::string>();
    CHECK(upload("application/octet-stream", picture, digest).document().at("media").at("id") == id);
    const auto fetched = server.get("/api/media/" + id, admin);
    CHECK(fetched.status == 200 && fetched.body == picture);
    CHECK(fetched.header("Content-Type") == "image/png" && fetched.header("ETag") == "\"" + digest + "\"" &&
        fetched.header("Cache-Control") == "private, immutable, max-age=31536000" &&
        fetched.header("X-Content-Type-Options") == "nosniff");
    CHECK(server.get("/api/media/" + id).status == 401);
    CHECK(server.get("/api/media/00000000-0000-4000-8000-000000000000", admin).status == 404);
    CHECK(server.get("/api/media/not-a-media-id", admin).status == 404);
    CHECK(upload("application/x-shockwave-flash", picture, digest).status == 415);
    CHECK(upload("image/png", "", digest).status == 400 && upload("image/png", "", digest).document()["error"]["code"] == "empty_media");
    CHECK(upload("image/png", picture, std::string(64, '0')).status == 422);
    CHECK(upload("image/png", picture, "not-a-digest").status == 400);
    CHECK(upload("image/png", picture, "").status == 400);
    CHECK(upload("image/png", picture, digest, false).status == 401);
    const auto oversized = server.send("POST", "/api/sessions", json{{"title", std::string(2 * 1024 * 1024, 'x')}});
    CHECK(oversized.status == 413 && oversized.document()["error"]["code"] == "request_too_large");
    const std::string large(2 * 1024 * 1024, 'm');
    // The payload cap only rises for uploads, so this exceeds the JSON cap while staying under the media limit.
    const auto uploaded = upload("image/png", large, "67c082ebaf8f24d2ad6f6e1ca6e37cbe7ff17f3dcde39ecd6a4ed311c6210fb7");
    CHECK(uploaded.status == 201 && uploaded.document().at("media").at("byte_size") == large.size());
    const auto issued = server.send("POST", "/api/auth/keys", {{"name", "jobs-only"}, {"scopes", {"jobs"}}}, admin);
    CHECK(issued.status == 201);
    CHECK(server.raw("POST", "/api/media", {{"Content-Type", "image/png"}, {"X-Content-SHA256", digest},
        {"Authorization", "Bearer " + issued.document().at("token").get<std::string>()}}, picture).status == 403);
    server.stop();
    auto restarted = f.start({}, root_environment);
    CHECK(restarted.get("/api/media/" + id, admin).body == picture);
}

void cli_and_lifetime() {
    Fixture f;
    write_file(f.root / "file", "not a directory");
    for (const auto& arguments : std::vector<std::vector<std::string>>{
        {"--unknown"}, {"--port"}, {"--port", "-1"}, {"--port", "+80"}, {"--port", "65536"},
        {"--port", "80junk"}, {"--port", "999999999999999999999999999999"},
        {"--host", "0.0.0.0"}, {"--host", "localhost"}, {"--host", "::"},
        {"--api-key-env", "9INVALID"}, {"--studio-instance-id", "00000000-0000-4000-8000-000000000001"},
        {"--studio-instance-id", "invalid", "--exit-on-stdin-close"},
        {"--web-root", (f.root / "missing").u8string()}, {"--data-dir", (f.root / "file").u8string()},
        {"--model-dir", (f.root / "file").u8string()}}) {
        auto& child = f.launch(f.data, arguments);
        CHECK(child.wait() != 0);
        CHECK(child.output().find("TyloInfer server not started:") != std::string::npos);
    }
    CHECK(!fs::exists(f.data));
    auto& version = f.launch(f.data, {"--version"});
    CHECK(version.wait() == 0);
    auto& help = f.launch(f.data, {"--help"});
    CHECK(help.wait() == 0);
    CHECK(help.output().find("Usage: tyloinfer-server") != std::string::npos);
    auto server = f.start({"--studio-instance-id", "ABCDEF01-0000-4000-8000-000000000001"});
    const auto identity = server.identity();
    CHECK(identity == server.identity());
    CHECK(identity.at("service") == "tyloinfer-server");
    CHECK(identity.at("protocol_version") == "1.0");
    CHECK(identity.at("studio_protocol_version") == "1.0");
    CHECK(identity.at("instance_id") == "abcdef01-0000-4000-8000-000000000001");
    CHECK(identity.at("pid") == server.process.pid());
    CHECK(identity.at("launch_mode") == "studio");
    const auto version_output = version.output();
    CHECK(identity.at("version") == version_output.substr(0, version_output.find_first_of("\r\n")));
    CHECK(identity.at("workspace_id").get<std::string>().size() == 64);
    CHECK(identity.dump().find(f.data.u8string()) == std::string::npos);
    CHECK(server.get("/health").document() == json({{"status", "ok"}, {"service", "tyloinfer-server"}, {"protocol_version", "1.0"}}));
    CHECK(server.get("/health").header("Content-Type").find("application/json") != std::string::npos);
    CHECK(server.get("/health").header("X-Content-Type-Options") == "nosniff");
    CHECK(server.get("/health", {{"X-TyloInfer-Instance-ID", "old-instance"}}).status == 409);
    CHECK(server.get("/health", {{"X-TyloInfer-Instance-ID", identity.at("instance_id")}}).status == 200);
    server.process.write_input("parent still alive\n");
    CHECK(server.get("/health").status == 200);
    server.stop();
    auto replacement = f.start();
    CHECK(replacement.identity().at("workspace_id") == identity.at("workspace_id"));
    CHECK(replacement.identity().at("instance_id") != identity.at("instance_id"));
    CHECK(replacement.identity().at("launch_mode") == "studio");
    replacement.stop();
    auto unicode = f.start_at(f.root / fs::u8path("数据 space"));
    CHECK(unicode.identity().at("workspace_id") != identity.at("workspace_id"));
    unicode.stop();
    {
        Fixture cli_fixture;
        auto cli = cli_fixture.start({}, {}, false);
        CHECK(cli.identity().at("launch_mode") == "cli");
        cli.process.close_input();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        CHECK(cli.get("/health").status == 200);
    }
    const auto probe = ::socket(AF_INET6, SOCK_STREAM, 0);
    sockaddr_in6 address{};
    address.sin6_family = AF_INET6;
    inet_pton(AF_INET6, "::1", &address.sin6_addr);
    const bool ipv6_available = probe != INVALID_SOCKET && ::bind(probe, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0;
#ifdef _WIN32
    if (probe != INVALID_SOCKET) ::closesocket(probe);
#else
    if (probe != INVALID_SOCKET) ::close(probe);
#endif
    if (ipv6_available) {
        auto ipv6 = f.start({"--host", "::1"});
        ipv6.stop();
    } else std::cout << "SKIP IPv6 loopback unavailable\n";
#ifndef _WIN32
    for (const auto signal : {SIGTERM, SIGINT, SIGKILL}) {
        auto child = f.start();
        child.process.signal(signal);
        CHECK(child.process.wait() == (signal == SIGKILL ? -SIGKILL : 0));
        auto next = f.start();
        next.stop();
    }
#endif
}

void locks_and_ports() {
    Fixture f;
    auto server = f.start();
    auto& duplicate = f.launch(f.data);
    CHECK(duplicate.wait() != 0);
    CHECK(duplicate.output().find("already managed") != std::string::npos);
    auto& conflict = f.launch(f.root / "other", {"--port", std::to_string(server.port)});
    CHECK(conflict.wait() != 0);
    CHECK(conflict.output().find("cannot bind listener") != std::string::npos);
    auto other = f.start_at(f.root / "other");
    other.stop();
#ifdef _WIN32
    const auto* local = _wgetenv(L"LOCALAPPDATA");
    const auto lock_path = (local ? fs::path(local) : fs::path(_wgetenv(L"USERPROFILE")) / "AppData" / "Local") /
        "MFQ" / "run" / (server.identity().at("workspace_id").get<std::string>() + ".lock");
    int lock = -1;
    CHECK(_wsopen_s(&lock, lock_path.c_str(), _O_RDWR | _O_BINARY | _O_NOINHERIT, _SH_DENYNO, _S_IREAD | _S_IWRITE) == 0);
    CHECK(_locking(lock, _LK_NBLCK, 1) != 0);
    server.stop();
    CHECK(_locking(lock, _LK_NBLCK, 1) == 0);
    auto& externally_locked = f.launch(f.data);
    CHECK(externally_locked.wait() != 0);
    CHECK(externally_locked.output().find("already managed") != std::string::npos);
    CHECK(_locking(lock, _LK_UNLCK, 1) == 0);
    _close(lock);
#else
    fs::create_directory_symlink(f.data, f.root / "alias");
    auto& alias = f.launch(f.root / "alias");
    CHECK(alias.wait() != 0);
    CHECK(alias.output().find("already managed") != std::string::npos);
    const auto lock_path = fs::path("/tmp") / ("tyloinfer-server-locks-" + std::to_string(getuid())) /
        (server.identity().at("workspace_id").get<std::string>() + ".lock");
    auto record = read_file(lock_path);
    CHECK(!record.empty());
    CHECK(json::parse(record.substr(1)) == server.identity());
    const int lock = ::open(lock_path.c_str(), O_RDWR | O_CLOEXEC);
    CHECK(lock >= 0);
    CHECK(::flock(lock, LOCK_EX | LOCK_NB) != 0);
    server.stop();
    CHECK(::flock(lock, LOCK_EX | LOCK_NB) == 0);
    auto& externally_locked = f.launch(f.data);
    CHECK(externally_locked.wait() != 0);
    CHECK(externally_locked.output().find("already managed") != std::string::npos);
    ::close(lock);
    auto removed = f.start();
    fs::remove_all(f.data);
    fs::create_directory(f.data);
    auto& deleted = f.launch(f.data);
    CHECK(deleted.wait() != 0);
    CHECK(deleted.output().find("already managed") != std::string::npos);
    removed.stop();
#endif
    auto retry = f.start();
    retry.stop();
}

void shutdown_with_pending_work() {
    {
        Fixture f;
        auto server = f.start({}, root_environment);
        Database database(f.data / "server.sqlite3");
        database.execute("BEGIN IMMEDIATE");
        auto pending = std::async(std::launch::async, [&] {
            return server.send("POST", "/api/auth/keys", {{"name", "during-shutdown"}, {"scopes", {"jobs"}}}, admin);
        });
        CHECK(pending.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout);
        CHECK(server.get("/health").status == 200);
        server.process.close_input();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK(server.process.running());
        database.execute("COMMIT");
        const auto created = pending.get();
        CHECK(created.status == 201);
        CHECK(server.process.wait() == 0);
        CHECK(database.scalar("PRAGMA integrity_check") == "ok");
        auto restarted = f.start({}, root_environment);
        CHECK(restarted.get("/api/jobs", bearer(created.document().at("token"))).status == 200);
    }
    {
        Fixture f;
        auto server = f.start();
        std::vector<std::unique_ptr<PartialRequest>> connections;
        for (int i = 0; i < 12; ++i) connections.push_back(std::make_unique<PartialRequest>(server.port));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        server.process.close_input();
        CHECK(server.process.wait() == 0);
        auto restarted = f.start();
        CHECK(restarted.get("/health").status == 200);
    }
}

void static_files() {
    Fixture f;
    const auto web = f.root / "web";
    fs::create_directories(web / "assets");
    fs::create_directory(web / "guide");
    fs::create_directory(web / "api");
    const std::string index = "<!doctype html><title>MFQ Studio</title>";
    write_file(web / "index.html", index);
    write_file(web / "assets" / "app.js", "console.log('ok');");
    write_file(web / "guide" / "index.html", "guide");
    write_file(web / "api" / "secret", "not an API");
    write_file(web / "health", "not health");
    auto server = f.start({"--web-root", web.u8string() + "/"});
    CHECK(std::string(mfq::server::schema_sql) == read_file(fs::u8path(TYLOINFER_SERVER_SCHEMA_FILE)));
    CHECK(server.get("/openapi.json").status == 404);
    for (const auto* path : {"/", "/chat", "/settings/profile", "/future/page?session=1"}) {
        const auto response = server.get(path, {{"Accept", "text/html"}});
        CHECK(response.status == 200);
        CHECK(response.body == index);
        CHECK(response.header("Content-Type").find("text/html") != std::string::npos);
        const auto csp = response.header("Content-Security-Policy");
        CHECK(csp.find("connect-src 'self' https://api.github.com ws: wss:;") != std::string::npos);
        CHECK(csp.find("connect-src *") == std::string::npos);
        CHECK(csp.find("script-src 'self';") != std::string::npos);
        const auto head = server.raw("HEAD", path, {{"Accept", "text/html"}});
        CHECK(head.status == 200);
        CHECK(head.body.empty());
        CHECK(std::stoul(head.header("Content-Length")) == index.size());
    }
    CHECK(server.get("/assets/app.js").body == "console.log('ok');");
    CHECK(server.get("/guide/").body == "guide");
    // API responses carry the strict policy without the web UI's release-check origin.
    const auto api_csp = server.get("/health").header("Content-Security-Policy");
    CHECK(api_csp.find("connect-src 'self' ws: wss:;") != std::string::npos);
    CHECK(api_csp.find("api.github.com") == std::string::npos);
    CHECK(server.get("/health").document().at("service") == "tyloinfer-server");
    for (const auto* path : {"/api/secret", "/v1/missing", "/health/missing", "/docs/missing",
            "/assets/missing", "/missing.js/", "/%2e%2e/secret.txt", "/%5csecret", "/%00secret"})
        CHECK(server.get(path, {{"Accept", "text/html"}}).status == 404);
    CHECK(server.get("/chat", {{"Accept", "*/*"}}).status == 404);
    CHECK(server.get("/chat", {{"Accept", "application/json"}}).status == 404);
    CHECK(server.get("/chat", {{"Accept", "text/html"}, {"Sec-Fetch-Dest", "script"}}).status == 404);
    CHECK(server.raw("POST", "/chat", {{"Accept", "text/html"}}).status != 200);
#ifndef _WIN32
    write_file(f.root / "secret.txt", "private content");
    fs::create_symlink(f.root / "secret.txt", web / "outside.txt");
    CHECK(server.get("/outside.txt").status == 403);
    fs::remove(web / "index.html");
    fs::create_symlink(f.root / "secret.txt", web / "index.html");
    CHECK(server.get("/chat", {{"Accept", "text/html"}}).status == 403);
    server.stop();
    auto& invalid = f.launch(f.data, {"--web-root", web.u8string()});
    CHECK(invalid.wait() != 0);
#else
    server.stop();
#endif
}

void policy_and_protocols() {
    Fixture f;
    auto server = f.start();
    CHECK(server.get("/api/auth/policy").document() == json({{"enabled", false}, {"has_key", false}}));
    CHECK(server.send("PUT", "/api/auth/policy", {{"enabled", true}}).status == 400);
    CHECK(server.send("PUT", "/api/auth/policy", {{"enabled", true}, {"api_key", "example-root"}}).status == 200);
    CHECK(server.get("/api/auth/policy").status == 401);
    for (const auto& key : std::vector<json>{"", "has space", "中文", std::string(513, 'x'), 123})
        CHECK(server.send("PUT", "/api/auth/policy", {{"enabled", true}, {"api_key", key}}, admin).status == 400);
    CHECK(server.send("PUT", "/api/auth/policy", {{"enabled", 1}}, admin).status == 400);
    CHECK(server.send("PUT", "/api/auth/policy", {{"enabled", true}, {"unknown", 1}}, admin).status == 400);
    CHECK(server.raw("PUT", "/api/auth/policy", admin, "{}").status == 415);
    auto content_headers = admin;
    content_headers.emplace("Content-Type", "application/json");
    CHECK(server.raw("PUT", "/api/auth/policy", content_headers, "{").status == 400);
    for (const auto* origin : {"tauri://localhost", "http://tauri.localhost", "https://tauri.localhost", "http://127.0.0.1:5174"}) {
        const auto denied = server.get("/api/auth/policy", {{"Origin", origin}});
        CHECK(denied.status == 401);
        CHECK(denied.header("Access-Control-Allow-Origin") == origin);
        CHECK(denied.header("WWW-Authenticate") == "Bearer");
        const auto preflight = server.raw("OPTIONS", "/api/auth/policy", {{"Origin", origin},
            {"Access-Control-Request-Method", "PUT"}, {"Access-Control-Request-Headers", "Authorization, Content-Type, X-TyloInfer-Instance-ID"}});
        CHECK(preflight.status == 204);
        CHECK(preflight.header("Access-Control-Allow-Origin") == origin);
    }
    CHECK(server.get("/api/auth/policy", {{"Origin", "https://untrusted.example"}}).header("Access-Control-Allow-Origin").empty());
    CHECK(server.raw("OPTIONS", "/api/auth/policy", {{"Origin", "tauri://localhost"},
        {"Access-Control-Request-Method", "PUT"}, {"Access-Control-Request-Headers", "X-Untrusted"}}).status == 403);
    CHECK(server.raw("POST", "/v1/messages", {{"X-Api-Key", "wrong"}}).document().at("error").at("type") == "authentication_error");
    const auto invalid_message = server.raw("POST", "/v1/messages", {{"X-Api-Key", "example-root"}});
    CHECK(invalid_message.status == 415);
    CHECK(invalid_message.document().at("error").at("type") == "invalid_request_error");
    const httplib::Headers upgrade{{"Upgrade", "websocket"}, {"Connection", "Upgrade"},
        {"Sec-WebSocket-Version", "13"}, {"Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ=="}};
    CHECK(server.get("/api/realtime", upgrade).status == 401);
    CHECK(server.get("/api/realtime?access_token=example-root", upgrade).status == 404);
    CHECK(server.get("/api/realtime?access_token=example-root&server_instance=old", upgrade).status == 409);
#ifndef _WIN32
    const auto previous = read_file(f.data / "auth.json");
    fs::rename(f.data / "auth.json", f.data / "auth.saved");
    fs::create_directory(f.data / "auth.json");
    const auto failed = server.send("PUT", "/api/auth/policy", {{"enabled", true}, {"api_key", "failed-key"}}, admin);
    fs::remove(f.data / "auth.json");
    fs::rename(f.data / "auth.saved", f.data / "auth.json");
    CHECK(failed.status == 500);
    CHECK(read_file(f.data / "auth.json") == previous);
    CHECK(server.get("/api/auth/policy", admin).status == 200);
    CHECK(server.get("/api/auth/policy", bearer("failed-key")).status == 401);
    CHECK((fs::status(f.data / "auth.json").permissions() & fs::perms::mask) == (fs::perms::owner_read | fs::perms::owner_write));
#endif
    CHECK(server.send("PUT", "/api/auth/policy", {{"enabled", true}, {"api_key", "replacement"}}, admin).status == 200);
    CHECK(server.get("/api/auth/policy", admin).status == 401);
    server.stop();
    const auto saved = read_file(f.data / "auth.json");
    CHECK(saved.find("example-root") == std::string::npos);
    CHECK(saved.find("replacement") == std::string::npos);
    CHECK(json::parse(saved).at("key_hash").get<std::string>().size() == 64);
    auto network_saved = f.start({"--host", "0.0.0.0"});
    CHECK(network_saved.get("/api/auth/policy").status == 401);
    CHECK(network_saved.get("/api/auth/policy", bearer("replacement")).status == 200);
    CHECK(network_saved.send("PUT", "/api/auth/policy", {{"enabled", false}}, bearer("replacement")).status == 400);
    network_saved.stop();
    auto restarted = f.start({}, root_environment);
    CHECK(restarted.get("/api/auth/policy", admin).status == 401);
    CHECK(restarted.send("PUT", "/api/auth/policy", {{"enabled", false}}, bearer("replacement")).status == 200);
    restarted.stop();
    auto disabled = f.start();
    CHECK(disabled.get("/api/auth/policy").document() == json({{"enabled", false}, {"has_key", true}}));
    disabled.stop();
    auto& network = f.launch(f.data, {"--host", "0.0.0.0"}, root_environment);
    CHECK(network.wait() != 0);
    CHECK(network.output().find("authentication is required") != std::string::npos);
}

void keys_and_network() {
    Fixture f;
    auto server = f.start({"--host", "0.0.0.0", "--api-key-env", "TYLOINFER_TEST_ROOT"}, {{"TYLOINFER_TEST_ROOT", "example-root"}});
    CHECK(server.get("/api/auth/policy").status == 401);
    CHECK(server.send("PUT", "/api/auth/policy", {{"enabled", false}}, admin).status == 400);
    auto issue = [&](const json& input) {
        const auto response = server.send("POST", "/api/auth/keys", input, admin);
        CHECK(response.status == 201);
        return response.document();
    };
    auto key = issue({{"name", "jobs"}, {"scopes", {"jobs"}}});
    const auto token = key.at("token").get<std::string>(), id = key.at("key").at("id").get<std::string>();
    CHECK(token.rfind("mfq_", 0) == 0);
    CHECK(server.get("/api/jobs", bearer(token)).status == 200);
    CHECK(server.get("/api/sessions", bearer(token)).status == 403);
    CHECK(server.get("/api/auth/keys", bearer(token)).status == 403);
    CHECK(server.send("POST", "/api/auth/keys", {{"name", "jobs"}, {"scopes", {"jobs"}}}, admin).status == 409);
    for (const auto* role : {"viewer", "operator", "administrator"}) {
        const auto issued = issue({{"name", role}, {"role", role}});
        const auto credentials = bearer(issued.at("token"));
        CHECK(server.get("/api/runtime/status", credentials).status == 200);
        CHECK(server.get("/api/runtime/resources", credentials).status == 200);
        CHECK(server.get("/api/models/directories", credentials).status == (std::string(role) == "viewer" ? 403 : 200));
        CHECK(server.get("/api/models/00000000000000000000000000000000/directory", credentials).status == (std::string(role) == "viewer" ? 403 : 404));
        CHECK(server.get("/api/models", credentials).status == 200);
        CHECK(server.raw("HEAD", "/api/models", credentials).status == 200);
        CHECK(server.raw("HEAD", "/api/runtime/listener", credentials).status == 200);
        CHECK(server.get("/api/auth/keys", credentials).status == (std::string(role) == "administrator" ? 200 : 403));
    }
    for (const auto& input : std::vector<json>{json::object(), {{"name", "x"}}, {{"name", "x"}, {"scopes", {"root"}}},
        {{"name", "x"}, {"scopes", {"jobs", "jobs"}}}, {{"name", "x"}, {"role", "root"}},
        {{"name", "x"}, {"role", "viewer"}, {"unknown", 1}},
        {{"name", "x"}, {"role", "viewer"}, {"expires_at", "2025-02-29T00:00:00Z"}},
        {{"name", "x"}, {"role", "viewer"}, {"expires_at", "2099-01-01"}},
        {{"name", "x"}, {"role", "viewer"}, {"expires_at", "2099-01-01T00:00:00+25:00"}}})
        CHECK(server.send("POST", "/api/auth/keys", input, admin).status == 400);
    for (const auto expired : {true, false}) {
        const auto issued = issue({{"name", expired ? "expired" : "future"}, {"role", "viewer"},
            {"expires_at", expired ? "2024-02-29T00:00:00+00:00" : "2099-02-28T23:59:59.999999-08:00"}});
        CHECK(server.get("/v1/models", bearer(issued.at("token"))).status == (expired ? 401 : 200));
    }
    issue({{"name", "'; DROP TABLE sessions; --"}, {"scopes", {"inference"}}});
    const auto rotated = server.raw("POST", "/api/auth/keys/" + id + "/rotate", admin);
    CHECK(rotated.status == 200);
    const auto replacement = rotated.document().at("token").get<std::string>();
    CHECK(replacement != token);
    CHECK(server.get("/api/jobs", bearer(token)).status == 401);
    CHECK(server.get("/api/jobs", bearer(replacement)).status == 200);
    const auto listed = server.get("/api/auth/keys", admin).body;
    CHECK(listed.find("key_hash") == std::string::npos);
    CHECK(listed.find(replacement) == std::string::npos);
    for (const auto& file : fs::directory_iterator(f.data)) if (file.path().filename().u8string().rfind("server.sqlite3", 0) == 0) {
        CHECK(read_file(file.path()).find(token) == std::string::npos);
        CHECK(read_file(file.path()).find(replacement) == std::string::npos);
    }
    server.stop();
    auto restarted = f.start({}, root_environment);
    CHECK(restarted.get("/api/jobs", bearer(replacement)).status == 200);
    CHECK(restarted.raw("POST", "/api/auth/keys/" + id + "/revoke", admin).status == 200);
    CHECK(restarted.get("/api/jobs", bearer(replacement)).status == 401);
    CHECK(restarted.raw("POST", "/api/auth/keys/00000000-0000-4000-8000-000000000000/revoke", admin).status == 404);
    CHECK(restarted.raw("POST", "/api/auth/keys/invalid/rotate", admin).status == 400);
    Database database(f.data / "server.sqlite3");
    CHECK(database.scalar("SELECT count(*) FROM sessions") == "0");
}

void runtime_observation() {
    Fixture f;
    auto server = f.start({}, root_environment);
    for (const auto* path : {"/api/runtime/status", "/api/runtime/metrics", "/api/runtime/logs",
            "/api/runtime/resources", "/api/runtime/serving", "/api/runtime/models",
            "/api/runtime/instances/00000000-0000-4000-8000-000000000001"})
        CHECK(server.get(path).status == 401);
    const auto idle = server.get("/api/runtime/status", admin).document();
    CHECK(idle == json({{"source", "native-instances"}, {"status", "ok"}, {"runtime_state", "idle"}}));
    CHECK(server.get("/api/runtime/status?instance_id=", admin).status == 400);
    CHECK(server.get("/api/runtime/status?instance_id=not-a-uuid", admin).status == 400);
    CHECK(server.get("/api/runtime/status?instance_id=00000000-0000-4000-8000-000000000001", admin).status == 404);
    CHECK(server.get("/api/runtime/instances/00000000-0000-4000-8000-000000000001", admin).status == 404);
    CHECK(server.get("/api/runtime/instances/not-a-uuid", admin).status == 400);
    CHECK(server.get("/api/runtime/models", admin).document() == json({{"data", json::array()}}));
    CHECK(server.get("/api/runtime/metrics", admin).document() == json({{"data", json::array()}}));
    CHECK(server.get("/api/runtime/logs", admin).document() == json({{"data", json::array()}}));
    for (const auto* query : {"?limit=0", "?limit=2001", "?limit=two", "?instance_id=nope"}) {
        CHECK(server.get(std::string("/api/runtime/metrics") + query, admin).status == 400);
        CHECK(server.get(std::string("/api/runtime/logs") + query, admin).status == 400);
    }
    CHECK(server.get("/api/runtime/metrics?since=yesterday", admin).status == 400);
    for (const auto* query : {"?after=-1", "?level=trace", "?level="})
        CHECK(server.get(std::string("/api/runtime/logs") + query, admin).status == 400);
    CHECK(server.get("/api/runtime/metrics?limit=1&since=2024-01-02T03:04:05.500+00:00", admin).status == 200);
    CHECK(server.get("/api/runtime/logs?limit=2000&after=0&level=warning", admin).document() ==
        json({{"data", json::array()}}));
    const auto serving = server.get("/api/runtime/serving", admin).document();
    CHECK(serving.at("scope") == "session");
    CHECK(serving.at("started_at").get<double>() > 0);
    CHECK(serving.at("activity").empty());
    CHECK(serving.at("totals") == json({{"completed_requests", 0}, {"prompt_tokens", 0}, {"prefill_tokens", 0},
        {"completion_tokens", 0}, {"cached_tokens", 0}, {"prefill_tps", nullptr}, {"decode_tps", nullptr},
        {"ttft_ms", nullptr}, {"cache_hit_rate", nullptr}}));
    CHECK(server.get("/api/runtime/serving?scope=alltime&model=any", admin).document().at("scope") == "alltime");
    CHECK(server.get("/api/runtime/serving?scope=week", admin).status == 400);
    CHECK(server.get("/api/runtime/serving?model=", admin).status == 400);
    const auto resources = server.get("/api/runtime/resources", admin).document();
    CHECK(resources.at("cpu_name").is_null());
    CHECK(resources.at("cpu_cores").get<int>() >= 1);
    CHECK(resources.at("gpus").empty() && resources.at("disks").empty() && resources.at("weights").empty());
    CHECK(resources.at("interval_seconds").is_null() && resources.at("cpu_utilization_percent").is_null());
    CHECK(resources.at("memory_total_bytes").get<std::int64_t>() > 0);
    CHECK(resources.at("memory_available_bytes").get<std::int64_t>() > 0);
    CHECK(resources.at("memory_used_bytes").get<std::int64_t>() >= 0);
    CHECK(resources.at("memory_bandwidth_bytes_per_second").is_null() &&
        resources.at("memory_bandwidth_limit_bytes_per_second").is_null() &&
        resources.at("memory_bandwidth_utilization_percent").is_null());
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    const auto second = server.get("/api/runtime/resources", admin).document();
    const auto interval = second.at("interval_seconds").get<double>();
    const auto utilization = second.at("cpu_utilization_percent").get<double>();
    CHECK(interval >= 1.0 && interval <= 10.0);
    CHECK(utilization >= 0.0 && utilization <= 100.0);
    server.stop();
}

void runtime_policies() {
    Fixture f;
    auto server = f.start({}, root_environment);
    const std::vector<const char*> policies{"/api/runtime/context-policy", "/api/runtime/inference-policy",
        "/api/runtime/memory-policy"};
    for (const auto* path : policies) CHECK(server.get(path).status == 401);
    const json context_defaults{{"fallback_context_size", 32768}, {"max_context_size", nullptr},
        {"model_overrides", json::object()}, {"model_yarn_enabled", json::object()},
        {"model_qsa_kv_offload", json::object()}, {"model_kv_quantization", json::object()},
        {"model_kv_memory", json::object()}};
    const json inference_defaults{{"mtp_enabled", true}};
    const json memory_defaults{{"total_limit_bytes", nullptr}, {"model_limit_bytes", nullptr},
        {"prefix_limit_bytes", nullptr}, {"prefix_disk_limit_bytes", nullptr}, {"prefix_ram_limit_bytes", nullptr},
        {"prefix_directory", nullptr}};
    CHECK(server.get("/api/runtime/context-policy", admin).document() == context_defaults);
    CHECK(server.get("/api/runtime/inference-policy", admin).document() == inference_defaults);
    CHECK(server.get("/api/runtime/memory-policy", admin).document() == memory_defaults);

    const json context{{"fallback_context_size", 8192}, {"max_context_size", 65536},
        {"model_overrides", {{"chat", 4096}}}, {"model_yarn_enabled", {{"chat", true}}},
        {"model_qsa_kv_offload", {{"chat", {{"enabled", true}, {"budget_bytes", 1073741824LL}, {"ram_budget_bytes", 0}}}}},
        {"model_kv_quantization", {{"chat", {{"enabled", true}, {"bits", 2.5}, {"algorithm", "turboquant"}}}}},
        {"model_kv_memory", {{"chat", {{"placement", "auto_ssd"}, {"allocation", "dynamic"}, {"gpu_budget_bytes", 2048},
            {"ram_preallocated", false}, {"ram_budget_bytes", 4096}}}}}};
    CHECK(server.send("PUT", "/api/runtime/context-policy", context, admin).document() == context);
    CHECK(server.get("/api/runtime/context-policy", admin).document() == context);
    const json context_partial{{"fallback_context_size", 512}, {"max_context_size", 65536},
        {"model_overrides", {{"chat", 4096}}}, {"model_yarn_enabled", {{"chat", true}}},
        {"model_qsa_kv_offload", {{"chat", {{"enabled", true}, {"budget_bytes", 1073741824LL}, {"ram_budget_bytes", 0}}}}},
        {"model_kv_quantization", {{"chat", {{"enabled", true}, {"bits", 2.5}, {"algorithm", "turboquant"}}}}},
        {"model_kv_memory", {{"chat", {{"placement", "auto_ssd"}, {"allocation", "dynamic"}, {"gpu_budget_bytes", 2048},
            {"ram_preallocated", false}, {"ram_budget_bytes", 4096}}}}}};
    CHECK(server.send("PUT", "/api/runtime/context-policy", {{"fallback_context_size", 512}}, admin).document() ==
        context_partial);
    for (const auto& body : std::vector<json>{{{"fallback_context_size", 511}}, {{"fallback_context_size", 2147483648LL}},
            {{"max_context_size", 0}}, {{"model_overrides", {{"chat", "4096"}}}}, {{"model_yarn_enabled", {{"chat", 1}}}},
            {{"model_qsa_kv_offload", {{"chat", {{"budget_bytes", 0}}}}}},
            {{"model_kv_quantization", {{"chat", {{"bits", 5}}}}}},
            {{"model_kv_memory", {{"chat", {{"placement", "disk"}}}}}}, {{"unexpected", true}}}) {
        const auto rejected = server.send("PUT", "/api/runtime/context-policy", body, admin);
        CHECK(rejected.status == 422 && rejected.document()["error"]["code"] == "invalid_policy");
    }
    // KvMemorySettings.ram_preallocated=false is only legal with SSD KV placement, as in the Python model validator.
    const auto kv_rule = server.send("PUT", "/api/runtime/context-policy",
        {{"model_kv_memory", {{"chat", {{"ram_preallocated", false}}}}}}, admin).document();
    CHECK(kv_rule["error"]["code"] == "invalid_policy" &&
        kv_rule["error"]["message"].get<std::string>().find("model_kv_memory.chat") != std::string::npos &&
        kv_rule["error"]["message"].get<std::string>().find("kv_ssd_required") != std::string::npos);
    CHECK(server.get("/api/runtime/context-policy", admin).document() == context_partial);

    CHECK(server.send("PUT", "/api/runtime/inference-policy", {{"mtp_enabled", false}}, admin).document() ==
        json({{"mtp_enabled", false}}));
    // The inference policy has no other fields, so a PUT replaces it instead of merging.
    CHECK(server.send("PUT", "/api/runtime/inference-policy", json::object(), admin).document() == inference_defaults);
    for (const auto& body : std::vector<json>{{{"mtp_enabled", "yes"}}, {{"mtp_enabled", 1}}, {{"mtp", true}}})
        CHECK(server.send("PUT", "/api/runtime/inference-policy", body, admin).status == 422);

    // The memory policy applies as a job, as in Python: field errors reject the PUT synchronously,
    // directory problems fail the accepted job.
    const auto wait_job = [&](const json& accepted) {
        for (int attempt = 0; attempt < 500; ++attempt) {
            const auto job = server.get("/api/jobs/" + accepted.at("operation_id").get<std::string>(), admin).document();
            if (job.at("status") != "queued" && job.at("status") != "running") return job;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        throw std::runtime_error("memory policy job did not finish");
    };
    const json memory{{"total_limit_bytes", 17179869184LL}, {"model_limit_bytes", 8589934592LL},
        {"prefix_limit_bytes", 0}, {"prefix_disk_limit_bytes", 1024}, {"prefix_ram_limit_bytes", nullptr},
        {"prefix_directory", "/tmp/tyloinfer-prefix"}};
    const auto accepted = server.send("PUT", "/api/runtime/memory-policy", memory, admin);
    CHECK(accepted.status == 202 && accepted.document().at("status") == "accepted");
    CHECK(wait_job(accepted.document()).at("status") == "succeeded");
    json memory_saved = memory;
    memory_saved["prefix_directory"] = fs::canonical("/tmp/tyloinfer-prefix").string();
    CHECK(server.get("/api/runtime/memory-policy", admin).document() == memory_saved);
    CHECK(wait_job(server.send("PUT", "/api/runtime/memory-policy",
        {{"prefix_directory", "/tmp/tyloinfer-other"}}, admin).document()).at("status") == "succeeded");
    json memory_partial = memory_saved;
    memory_partial["prefix_directory"] = fs::canonical("/tmp/tyloinfer-other").string();
    CHECK(server.get("/api/runtime/memory-policy", admin).document() == memory_partial);
    for (const auto& body : std::vector<json>{{{"total_limit_bytes", 0}}, {{"prefix_limit_bytes", -1}},
            {{"prefix_directory", ""}}, {{"prefix_directory", 5}}})
        CHECK(server.send("PUT", "/api/runtime/memory-policy", body, admin).status == 422);
    // A relative prefix directory passes the field rules and fails inside the job.
    const auto invalid = server.send("PUT", "/api/runtime/memory-policy", {{"prefix_directory", "relative/cache"}}, admin);
    CHECK(invalid.status == 202);
    const auto failed = wait_job(invalid.document());
    CHECK(failed.at("status") == "failed" && failed.at("error").at("code") == "invalid_cache_directory");

    const auto issued = server.send("POST", "/api/auth/keys", {{"name", "models-only"}, {"scopes", {"models"}}}, admin);
    CHECK(issued.status == 201);
    const auto scoped = bearer(issued.document().at("token").get<std::string>());
    for (const auto* path : policies) {
        CHECK(server.get(path, scoped).status == 403);
        CHECK(server.send("PUT", path, json::object(), scoped).status == 403);
    }
    server.stop();
    auto restarted = f.start({}, root_environment);
    CHECK(restarted.get("/api/runtime/context-policy", admin).document() == context_partial);
    CHECK(restarted.get("/api/runtime/inference-policy", admin).document() == inference_defaults);
    CHECK(restarted.get("/api/runtime/memory-policy", admin).document() == memory_partial);
    restarted.stop();
}

// Profiles, capabilities, instance policy patches and the listener against the real server binary;
// no model can load in this fixture, so the instance-backed paths live in server_inference_test.cpp.
void runtime_profiles() {
    Fixture f;
    auto server = f.start({}, root_environment);
    const auto root = f.data / "models";
    write_mfq(root / "beta.MFQ");
    const auto artifacts = server.get("/api/models?refresh=true", admin).document().at("data");
    CHECK(artifacts.size() == 1);
    const auto artifact_id = artifacts.at(0).at("id").get<std::string>();
    const auto modified_at = artifacts.at(0).at("modified_at").get<std::string>();
    const json load{{"model", artifact_id}, {"context_size", 4096}};
    const std::string absent = "00000000-0000-4000-8000-000000000001";
    const std::string instance_path = "/api/runtime/instances/" + absent;

    for (const auto* path : {"/api/runtime/profiles", "/api/runtime/capabilities", "/api/runtime/listener"})
        CHECK(server.get(path).status == 401);
    CHECK(server.send("POST", "/api/runtime/profiles", {{"name", "chat"}, {"load", load}}).status == 401);
    CHECK(server.send("PUT", "/api/runtime/listener", {{"port", 8080}}).status == 401);
    CHECK(server.send("PATCH", instance_path, {{"pinned", true}}).status == 401);

    // One native listener serves both protocols on the port chosen at startup.
    const auto listener = server.get("/api/runtime/listener", admin).document();
    CHECK(listener.at("host") == "127.0.0.1" && listener.at("port").get<int>() > 0);
    CHECK(listener.at("anthropic_port").is_null() && listener.at("configurable") == false);
    const auto managed = server.send("PUT", "/api/runtime/listener", {{"port", 8080}, {"protocol", "anthropic"}}, admin);
    CHECK(managed.status == 409 && managed.document()["error"]["code"] == "listener_not_managed" &&
        managed.document()["error"]["message"] == "server listener is managed externally");
    for (const auto& body : std::vector<json>{json::object(), {{"port", 0}}, {{"port", 70000}}, {{"port", "8080"}},
            {{"port", 8080}, {"protocol", "grpc"}}, {{"port", 8080}, {"unexpected", true}}})
        CHECK(server.send("PUT", "/api/runtime/listener", body, admin).status == 400);

    // With no loaded instance the native server has no runtime to describe, as the Python pool reports.
    const auto no_runtime = server.get("/api/runtime/capabilities", admin);
    CHECK(no_runtime.status == 503 && no_runtime.document()["error"]["code"] == "model_not_loaded");
    CHECK(server.get("/api/runtime/capabilities?instance_id=" + absent, admin).status == 404);
    CHECK(server.get("/api/runtime/capabilities?instance_id=nope", admin).status == 400);

    for (const auto& body : std::vector<json>{json::object(), {{"pinned", "yes"}}, {{"pinned", nullptr}},
            {{"idle_ttl_seconds", -1}}, {{"idle_ttl_seconds", "60"}}, {{"unexpected", true}}})
        CHECK(server.send("PATCH", instance_path, body, admin).status == 400);
    CHECK(server.send("PATCH", "/api/runtime/instances/not-a-uuid", {{"pinned", true}}, admin).status == 400);
    const auto absent_instance = server.send("PATCH", instance_path, {{"pinned", true}}, admin);
    CHECK(absent_instance.status == 404 && absent_instance.document()["error"]["code"] == "runtime_instance_not_found");

    CHECK(server.get("/api/runtime/profiles", admin).document() == json({{"data", json::array()}}));
    CHECK(server.get("/api/runtime/profiles/" + absent, admin).status == 404);
    CHECK(server.get("/api/runtime/profiles/not-a-uuid", admin).status == 400);
    const auto created = server.send("POST", "/api/runtime/profiles", {{"name", "chat"}, {"load", load}}, admin);
    CHECK(created.status == 201);
    const auto profile = created.document();
    const auto id = profile.at("id").get<std::string>();
    CHECK(id.size() == 36 && profile.at("name") == "chat" && profile.at("artifact_id") == artifact_id &&
        profile.at("artifact_modified_at") == modified_at && profile.at("drifted") == false &&
        profile.at("drift_reason").is_null() && profile.at("created_at") == profile.at("updated_at"));
    // sampling_defaults always shows inside load, as the Python resource does.
    CHECK(profile.at("load") == json({{"model", artifact_id}, {"context_size", 4096}, {"sampling_defaults", nullptr}}));
    CHECK(server.get("/api/runtime/profiles/" + id, admin).document() == profile);
    CHECK(server.get("/api/runtime/profiles", admin).document().at("data") == json::array({profile}));

    const json edited{{"name", "chat-lab"}, {"load", {{"model", artifact_id}, {"context_size", 2048}, {"pin", true}}}};
    const auto renamed = server.send("PUT", "/api/runtime/profiles/" + id, edited, admin).document();
    CHECK(renamed.at("id") == id && renamed.at("name") == "chat-lab" &&
        renamed.at("load").at("context_size") == 2048 && renamed.at("load").at("pin") == true &&
        renamed.at("updated_at").get<std::string>() >= renamed.at("created_at").get<std::string>());
    const auto conflict = server.send("POST", "/api/runtime/profiles", {{"name", "chat-lab"}, {"load", load}}, admin);
    CHECK(conflict.status == 409 && conflict.document()["error"]["code"] == "runtime_profile_conflict");
    for (const auto& body : std::vector<json>{json::object(), {{"name", ""}, {"load", load}},
            {{"name", std::string(65, 'n')}, {"load", load}}, {{"name", "unnamed"}},
            {{"name", "bad-load"}, {"load", json::array()}}, {{"name", "bad-load"}, {"load", {{"context_size", 4096}}}},
            {{"name", "bad-load"}, {"load", {{"model", artifact_id}, {"context_size", 511}}}},
            {{"name", "bad-load"}, {"load", {{"model", artifact_id}, {"idle_ttl_seconds", 0}}}},
            {{"name", "bad-load"}, {"load", {{"model", artifact_id}, {"pin", "yes"}}}},
            {{"name", "bad-load"}, {"load", {{"model", artifact_id}, {"sampling_defaults", 4}}}}})
        CHECK(server.send("POST", "/api/runtime/profiles", body, admin).status == 400);
    const auto unlisted = server.send("POST", "/api/runtime/profiles",
        {{"name", "unlisted"}, {"load", {{"model", "00000000000000000000000000000000"}}}}, admin);
    CHECK(unlisted.status == 404 && unlisted.document()["error"]["code"] == "model_artifact_not_found");

    // Rewriting the artifact changes what the profile pinned, so reads report drift and load refuses.
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    write_mfq(root / "beta.MFQ");
    CHECK(server.get("/api/models?refresh=true", admin).document().at("data").at(0).at("modified_at") != modified_at);
    const auto drifted = server.get("/api/runtime/profiles/" + id, admin).document();
    CHECK(drifted.at("drifted") == true &&
        drifted.at("drift_reason") == "the selected model artifact changed after this profile was saved");
    CHECK(server.get("/api/runtime/profiles", admin).document().at("data").at(0).at("drifted") == true);
    const auto blocked = server.send("POST", "/api/runtime/profiles/" + id + "/load", json::object(), admin);
    CHECK(blocked.status == 409 && blocked.document()["error"]["code"] == "runtime_profile_drift" &&
        blocked.document()["error"]["message"].get<std::string>().find("changed after this profile was saved") != std::string::npos);
    for (const auto& body : std::vector<json>{{{"allow_drift", "yes"}}, {{"unexpected", true}}})
        CHECK(server.send("POST", "/api/runtime/profiles/" + id + "/load", body, admin).status == 400);
    CHECK(server.send("POST", "/api/runtime/profiles/" + absent + "/load", json::object(), admin).status == 404);
    const auto accepted = server.send("POST", "/api/runtime/profiles/" + id + "/load", {{"allow_drift", true}}, admin);
    CHECK(accepted.status == 202 && accepted.document().at("status") == "accepted");
    const auto job_id = accepted.document().at("operation_id").get<std::string>();
    CHECK(accepted.header("Location") == "/api/jobs/" + job_id);
    json job;
    for (int attempt = 0; attempt < 500; ++attempt) {
        job = server.get("/api/jobs/" + job_id, admin).document();
        if (job.at("status") != "queued" && job.at("status") != "running") break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    // The job carries the stored load document; this server build has no engine, so it fails honestly.
    CHECK(job.at("kind") == "model.load" && job.at("status") == "failed" &&
        job.at("error").at("code") == "backend_unavailable" &&
        job.at("payload") == json({{"model", artifact_id}, {"context_size", 2048}, {"pin", true},
            {"sampling_defaults", nullptr}}));

    // The Python scope table puts these writes under `models` and their reads under `inference`.
    const auto issued = server.send("POST", "/api/auth/keys", {{"name", "models-only"}, {"scopes", {"models"}}}, admin);
    CHECK(issued.status == 201);
    const auto model_key = bearer(issued.document().at("token").get<std::string>());
    CHECK(server.get("/api/runtime/profiles", model_key).status == 403);
    CHECK(server.get("/api/runtime/listener", model_key).status == 403);
    CHECK(server.send("PATCH", instance_path, {{"pinned", true}}, model_key).status == 404);
    CHECK(server.send("POST", "/api/runtime/profiles", {{"name", "scoped"}, {"load", load}}, model_key).status == 201);
    CHECK(server.send("PUT", "/api/runtime/listener", {{"port", 8080}}, model_key).status == 403);

    CHECK(server.raw("DELETE", "/api/runtime/profiles/" + id, admin).status == 204);
    CHECK(server.get("/api/runtime/profiles/" + id, admin).status == 404);
    CHECK(server.raw("DELETE", "/api/runtime/profiles/" + id, admin).status == 404);
    CHECK(server.send("PUT", "/api/runtime/profiles/" + id, edited, admin).status == 404);
    const auto kept = server.send("POST", "/api/runtime/profiles", {{"name", "kept"}, {"load", load}}, admin);
    CHECK(kept.status == 201);
    const auto kept_id = kept.document().at("id").get<std::string>();
    server.stop();
    auto restarted = f.start({}, root_environment);
    CHECK(restarted.get("/api/runtime/profiles", admin).document().at("data").size() == 2);
    CHECK(restarted.get("/api/runtime/profiles/" + kept_id, admin).document().at("name") == "kept");
    restarted.stop();
}

void model_catalog() {
    Fixture f;
    auto server = f.start({}, root_environment);
    CHECK(server.get("/api/models").status == 401);
    CHECK(server.get("/api/models", admin).document().at("data").empty());
    const auto roots = server.get("/api/models/directories", admin).document();
    CHECK(roots.at("current_path").is_null());
    CHECK(roots.at("data").size() == 1);
    const auto root_id = roots.at("data")[0].at("id").get<std::string>();
    CHECK(root_id.size() == 32);
    const auto root = f.data / "models";
    const auto browse = server.get("/api/models/directories?directory_id=" + root_id, admin).document();
    CHECK(browse.at("current_path") == fs::canonical(root).u8string());
    CHECK(!browse.at("can_open_directory").get<bool>());
    CHECK(!browse.at("parent_id").is_null());
    CHECK(server.get("/api/models?refresh=invalid", admin).status == 400);
    for (const auto* query : {"directory_id=invalid", "directory_id=", "path=", "path=%00bad", "directory_id=00000000000000000000000000000000&path=/"})
        CHECK(server.get(std::string("/api/models/directories?") + query, admin).status == 400);
    CHECK(server.get("/api/models/directories?directory_id=00000000000000000000000000000000", admin).status == 404);
    write_hf(root / "alpha");
    write_mfq(root / "beta.MFQ");
    write_file(root / "broken.mfq", "broken");
    write_file(root / "split-00002-of-00003.mfq", "missing shards");
    CHECK(server.get("/api/models", admin).document().at("data").empty());
    const auto artifacts = server.get("/api/models?refresh=true", admin).document().at("data");
    CHECK(artifacts.size() == 4);
    CHECK(artifacts[0].at("name") == "alpha");
    CHECK(artifacts[0].at("format") == "hf");
    CHECK(artifacts[0].at("complete") == true);
    CHECK(artifacts[0].at("loadable") == false);
    CHECK(artifacts[0].at("error").is_null());
    CHECK(artifacts[0].at("support_error").is_string());
    CHECK(artifacts[0].at("architecture") == "qwen3_5-hf-full-mfq");
    CHECK(artifacts[0].at("tensor_count") == 1);
    CHECK(artifacts[0].at("dtypes") == json({"BF16"}));
    CHECK(artifacts[0].at("context_capacity") == 4096);
    CHECK(artifacts[0].at("total_bytes") == fs::file_size(root / "alpha" / "model.safetensors"));
    CHECK(artifacts[1].at("name") == "beta");
    CHECK(artifacts[1].at("complete") == true);
    CHECK(artifacts[1].at("tensor_count") == 1);
    CHECK(artifacts[1].at("record_count") == 2);
    CHECK(artifacts[1].at("context_capacity") == 8192);
    CHECK(artifacts[2].at("complete") == false);
    CHECK(artifacts[3].at("shard_count") == 3);
    CHECK(artifacts[3].at("missing_shards") == 2);
    CHECK(artifacts.dump().find(f.root.u8string()) == std::string::npos);
    const auto hf_id = artifacts[0].at("id").get<std::string>();
    const auto mfq_id = artifacts[1].at("id").get<std::string>();
    CHECK(server.get("/api/models/" + hf_id + "/directory", admin).document().at("current_path") == (root / "alpha").u8string());
    CHECK(server.get("/api/models/" + mfq_id + "/directory", admin).document().at("current_path") == root.u8string());
    CHECK(server.get("/api/models/invalid/directory", admin).status == 400);
    const auto files = server.get("/api/models/directories?directory_id=" + root_id, admin).document();
    CHECK(files.at("model_file_count") == 4);
    CHECK(files.at("files").size() == 3);
    CHECK(files.at("data").size() == 1);

    const auto external = f.root / fs::u8path("外部 models");
    write_hf(external / "gamma");
    const auto selected = server.get("/api/models/directories?path=" + httplib::encode_uri_component(external.u8string()), admin).document();
    const auto source = json{{"directory_id", selected.at("current_id")}};
    const auto registered = server.send("POST", "/api/models/directories/register", source, admin);
    CHECK(registered.status == 200);
    CHECK(registered.document().at("data").size() == 1);
    const auto gamma_id = registered.document().at("data")[0].at("id");
    const auto index = root / ".tq-files.json";
    const auto saved = read_file(index);
    CHECK(json::parse(saved).at("version") == 1);
    CHECK(json::parse(saved).at("files") == json({(external / "gamma").u8string()}));
    CHECK(server.send("POST", "/api/models/directories/register", source, admin).document().at("data").size() == 1);
    CHECK(server.get("/api/models?refresh=1", admin).document().at("data").size() == 5);
    for (const auto& input : std::vector<json>{json::object(), json::array(), {{"path", ""}}, {{"path", 42}},
            {{"path", external.u8string()}, {"directory_id", root_id}}, {{"path", external.u8string()}, {"unexpected", true}},
            {{"path", external.u8string()}, {"directory_id", ""}}})
        CHECK(server.send("POST", "/api/models/directories/register", input, admin).status == 400);
    CHECK(server.send("POST", "/api/models/directories/register", {{"directory_id", "00000000000000000000000000000000"}}, admin).status == 404);
    fs::create_directory(f.root / "empty");
    CHECK(server.send("POST", "/api/models/directories/register", {{"path", (f.root / "empty").u8string()}}, admin).status == 400);
    write_hf(f.root / "duplicate" / "alpha");
    CHECK(server.send("POST", "/api/models/directories/register", {{"path", (f.root / "duplicate").u8string()}}, admin).status == 409);
    CHECK(read_file(index) == saved);
    CHECK(server.get("/api/models", admin).document().at("data").size() == 5);
    auto deep = f.root / "deep";
    for (int i = 0; i < 34; ++i) deep /= "d";
    fs::create_directories(deep);
    write_mfq(deep / "deep.mfq");
    CHECK(server.send("POST", "/api/models/directories/register", {{"path", (f.root / "deep").u8string()}}, admin).status == 413);
    CHECK(read_file(index) == saved);
    json large{{"version", 1}, {"files", json::array()}};
    constexpr std::size_t limit = 4 * 1024 * 1024;
    std::size_t size = large.dump().size();
    const auto prefix = f.root.generic_u8string() + "/missing-";
    for (unsigned i = 0; ; ++i) {
        const auto value = prefix + std::to_string(i) + std::string(512, 'x');
        const auto growth = json(value).dump().size() + (large["files"].empty() ? 0 : 1);
        if (size + growth > limit - 1024) break;
        large["files"].push_back(value);
        size += growth;
    }
    large["files"].push_back(prefix + std::string(limit - 1 - size - json(prefix).dump().size() - 1, 'x'));
    const auto large_saved = large.dump();
    CHECK(large_saved.size() == limit - 1);
    CHECK(large["files"].size() < 10000);
    write_file(index, large_saved);
    CHECK(server.send("POST", "/api/models/directories/register", source, admin).status == 413);
    CHECK(read_file(index) == large_saved);
    write_file(index, saved);
    write_file(index, "{invalid");
    CHECK(server.get("/api/models?refresh=true", admin).status == 500);
    CHECK(read_file(index) == "{invalid");
    write_file(index, saved);
    server.stop();
    auto restarted = f.start({}, root_environment);
    const auto recovered = restarted.get("/api/models", admin).document().at("data");
    CHECK(recovered.size() == 5);
    CHECK(recovered[0].at("id") == hf_id);
    CHECK(recovered[1].at("id") == mfq_id);
    CHECK(recovered[3].at("id") == gamma_id);
    CHECK(restarted.get("/api/models/directories?directory_id=" + root_id, admin).status == 404);
    fs::remove(external / "gamma" / "model.safetensors");
    CHECK(restarted.get("/api/models?refresh=true", admin).document().at("data").size() == 4);
}

/** Verifies checkpoint analysis and cache profiles match the Python response shape. */
void model_inspection() {
    Fixture f;
    const auto root = f.data / "models";
    fs::create_directories(root / "alpha");
    write_file(root / "alpha" / "config.json", R"({"model_type":"qwen3","hidden_size":8,"num_attention_heads":2,
        "num_key_value_heads":1,"head_dim":4,"num_hidden_layers":2,"max_position_embeddings":128,"intermediate_size":16})");
    write_safetensors(root / "alpha" / "model.safetensors", json::array({
        json{{"name", "model.embed_tokens.weight"}, {"dtype", "BF16"}, {"shape", {4, 8}}, {"bytes", 64}},
        json{{"name", "model.block.0.attention.q_proj.weight"}, {"dtype", "BF16"}, {"shape", {8, 8}}, {"bytes", 128}},
        json{{"name", "model.block.0.mlp.down_proj.weight"}, {"dtype", "BF16"}, {"shape", {8, 16}}, {"bytes", 256}},
        json{{"name", "model.block.1.attention.q_proj.weight"}, {"dtype", "BF16"}, {"shape", {8, 8}}, {"bytes", 128}},
        json{{"name", "model.block.1.mlp.down_proj.weight"}, {"dtype", "BF16"}, {"shape", {8, 16}}, {"bytes", 256}}}));
    auto server = f.start({}, root_environment);
    write_mfq(root / "beta.mfq");
    const auto artifacts = server.get("/api/models?refresh=true", admin).document().at("data");
    CHECK(artifacts.size() == 2);
    std::string id, tq_id;
    for (const auto& item : artifacts) (item.at("name") == "alpha" ? id : tq_id) = item.at("id").get<std::string>();
    CHECK(!id.empty() && !tq_id.empty());
    const auto analysis = server.get("/api/models/" + id + "/analysis", admin);
    CHECK(analysis.status == 200);
    const auto body = analysis.document();
    CHECK(body.at("model_id") == id);
    CHECK(body.at("name") == "alpha");
    CHECK(body.at("format") == "hf");
    CHECK(body.at("architecture") == "qwen3");
    CHECK(body.at("complete") == true);
    CHECK(body.at("layer_count") == 2);
    CHECK(body.at("hidden_size") == 8);
    CHECK(body.at("attention_heads") == 2);
    CHECK(body.at("kv_heads") == 1);
    CHECK(body.at("head_dim") == 4);
    CHECK(body.at("linear_key_heads").is_null());
    CHECK(body.at("expert_count") == 0);
    CHECK(body.at("experts_per_token").is_null());
    CHECK(body.at("parameters") == 416);
    std::int64_t tensor_bytes = 0;
    for (const auto& tensor : body.at("tensors")) tensor_bytes += tensor.at("stored_bytes").get<std::int64_t>();
    CHECK(body.at("stored_bytes") == tensor_bytes);
    CHECK(body.at("average_bpw") == 8.0 * static_cast<double>(tensor_bytes) / 416.0);
    CHECK(body.at("tensors").size() == 5);
    CHECK(body.at("experts").empty());
    CHECK(body.at("layers").size() == 2);
    CHECK(body.at("layers")[0].at("attention_type") == "GQA");
    CHECK(body.at("layers")[0].at("parameters") == 64 + 128);
    CHECK(body.at("attention_distribution").at("GQA") == json({0, 1}));
    CHECK(body.at("projections").size() == 3);
    CHECK(body.at("projections")[0].at("category") == "attention");
    CHECK(body.at("projections")[0].at("name") == "q_proj");
    CHECK(body.at("projections")[0].at("tensor_count") == 2);
    CHECK(body.at("projections")[0].at("formats") == json({"BF16"}));
    CHECK(body.at("advanced").at("ffn_hidden_size") == 16);
    CHECK(body.at("advanced").at("max_context") == 128);
    CHECK(body.at("advanced").at("predictor_layers") == 0);
    CHECK(body.at("advanced").at("normalizations").empty());
    CHECK(body.at("graph").at("has_ple") == false);
    CHECK(body.at("graph").at("has_shared_expert") == false);
    CHECK(body.at("warnings").empty());
    const auto profile = body.at("cache_profile");
    CHECK(profile.at("max_context") == 128);
    CHECK(profile.at("fixed_bytes") == 0);
    CHECK(profile.at("fixed_components").empty());
    CHECK(profile.at("components").size() == 1);
    CHECK(profile.at("components")[0] == json({{"group", "GQA"}, {"name", "raw_kv"}, {"subgroup", nullptr},
        {"layers", 2}, {"bytes_per_row", 32}, {"tokens_per_row", 1}, {"allocation", "power_of_two"}, {"minimum_rows", 16},
        {"row_rounding", "ceil"}, {"active_after", 0}, {"max_read_rows_per_token", nullptr}, {"head_dimension", 4},
        {"kv_heads", 1}}));
    CHECK(server.get("/api/models/" + id + "/cache-profile", admin).document() == profile);
    CHECK(server.get("/api/models/" + id + "/cache-profile?include_recurrent=true", admin).document() == profile);
    CHECK(server.get("/api/models/" + id + "/cache-profile?include_recurrent=maybe", admin).status == 400);
    CHECK(server.get("/api/models/invalid/analysis", admin).status == 404);
    CHECK(server.get("/api/models/00000000000000000000000000000000/analysis", admin).status == 404);
    CHECK(server.get("/api/models/00000000000000000000000000000000/cache-profile", admin).status == 404);
    // An unreadable checkpoint stays listed but answers both inspection routes with 422.
    write_file(root / "broken.mfq", "broken");
    const auto listed = server.get("/api/models?refresh=true", admin).document().at("data");
    CHECK(listed.size() == 3);
    const auto broken_id = listed[2].at("id").get<std::string>();
    CHECK(listed[2].at("complete") == false);
    CHECK(server.get("/api/models/" + broken_id + "/analysis", admin).status == 422);
    CHECK(server.get("/api/models/" + broken_id + "/cache-profile", admin).status == 422);
    // A container without stored shapes still analyses its tensors from the payload headers.
    const auto container = server.get("/api/models/" + tq_id + "/analysis", admin).document();
    CHECK(container.at("format") == "tq");
    CHECK(container.at("architecture") == "qwen3_5");
    CHECK(container.at("tensors").size() == 1);
    CHECK(container.at("tensors")[0].at("shape") == json({1}));
    CHECK(container.at("tensors")[0].at("category") == "embedding");
    CHECK(container.at("warnings").empty());
    CHECK(container.at("layer_count") == 0);
    CHECK(container.at("cache_profile").is_null());
    // An architecture without a cache model answers with a null profile instead of an error.
    write_file(root / "alpha" / "config.json", R"({"model_type":"unsupported_kind"})");
    const auto unsupported = server.get("/api/models/" + id + "/cache-profile", admin);
    CHECK(unsupported.status == 200);
    CHECK(unsupported.document().is_null());
}

/** Covers deletion planning, token binding, safe path rejection, and what survives a delete. */
void model_deletion() {
    Fixture f;
    const auto root = f.data / "models";
    const auto alpha = root / "alpha";
    fs::create_directories(alpha);
    write_file(alpha / "config.json", R"({"model_type":"qwen3"})");
    write_safetensors(alpha / "part-0.safetensors", json::array({
        json{{"name", "model.embed_tokens.weight"}, {"dtype", "BF16"}, {"shape", {4, 8}}, {"bytes", 64}}}));
    write_safetensors(alpha / "part-1.safetensors", json::array({
        json{{"name", "model.lm_head.weight"}, {"dtype", "BF16"}, {"shape", {8, 8}}, {"bytes", 128}}}));
    write_file(alpha / "model.safetensors.index.json",
        R"({"metadata":{},"weight_map":{"model.embed_tokens.weight":"part-0.safetensors","model.lm_head.weight":"part-1.safetensors"}})");
    write_file(alpha / "tokenizer.json", "{}");
    // A weight map that leaves the checkpoint directory must be refused before anything is removed.
    const auto gamma = root / "gamma";
    fs::create_directories(gamma);
    write_file(gamma / "config.json", R"({"model_type":"qwen3"})");
    write_safetensors(gamma / "part-0.safetensors", json::array({
        json{{"name", "model.embed_tokens.weight"}, {"dtype", "BF16"}, {"shape", {4, 8}}, {"bytes", 64}}}));
    write_file(gamma / "model.safetensors.index.json", R"({"weight_map":{"model.embed_tokens.weight":"../escape.safetensors"}})");
    write_file(root / "beta-00001-of-00002.tq", "first shard");
    write_file(root / "beta-00002-of-00002.tq", "second shard");
    write_file(root / "beta.mfq.notes.json", "{}");
    auto server = f.start({}, root_environment);
    const auto listed = [&] { return server.get("/api/models?refresh=true", admin).document().at("data"); };
    CHECK(listed().size() == 3);
    std::string hf_id, tq_id;
    for (const auto& item : listed()) {
        if (item.at("format") == "hf") { if (item.at("name") == "alpha") hf_id = item.at("id").get<std::string>(); }
        else tq_id = item.at("id").get<std::string>();
    }
    CHECK(!hf_id.empty() && !tq_id.empty());
    const auto plan = server.get("/api/models/" + hf_id + "/deletion", admin);
    CHECK(plan.status == 200);
    const auto planned = plan.document();
    const auto token = planned.at("token").get<std::string>();
    CHECK(planned.at("model_id") == hf_id);
    CHECK(planned.at("name") == "alpha");
    CHECK(token.size() == 64);
    CHECK(planned.at("files") == json({fs::canonical(alpha / "config.json").u8string(),
        fs::canonical(alpha / "model.safetensors.index.json").u8string(),
        fs::canonical(alpha / "part-0.safetensors").u8string(), fs::canonical(alpha / "part-1.safetensors").u8string()}));
    std::uint64_t total = 0;
    for (const auto* name : {"config.json", "model.safetensors.index.json", "part-0.safetensors", "part-1.safetensors"})
        total += fs::file_size(alpha / name);
    CHECK(planned.at("total_bytes") == total);
    CHECK(server.get("/api/models/invalid/deletion", admin).status == 404);
    CHECK(server.get("/api/models/00000000000000000000000000000000/deletion", admin).status == 404);
    CHECK(server.send("POST", "/api/models/00000000000000000000000000000000/delete", {{"token", token}}, admin).status == 404);
    for (const auto& body : {json{{"token", "short"}}, json{{"token", std::string(64, 'A')}}, json{{"unexpected", 1}}, json::object()})
        CHECK(server.send("POST", "/api/models/" + hf_id + "/delete", body, admin).status == 400);
    CHECK(server.send("POST", "/api/models/" + hf_id + "/delete", {{"token", std::string(64, '0')}}, admin).status == 409);
    CHECK(fs::exists(alpha / "config.json"));
    for (const auto* escaped : {"../escape.safetensors", "/tmp/escape.safetensors", "nested/part-0.bin"}) {
        write_file(gamma / "model.safetensors.index.json",
            std::string(R"({"weight_map":{"model.embed_tokens.weight":")") + escaped + R"(}})");
        std::string current;
        for (const auto& item : listed()) if (item.at("name") == "gamma") current = item.at("id").get<std::string>();
        CHECK(!current.empty());
        const auto refused = server.get("/api/models/" + current + "/deletion", admin);
        CHECK(refused.status == 409);
        CHECK(refused.document().at("error").at("code") == "model_delete_unsafe");
        CHECK(server.send("POST", "/api/models/" + current + "/delete", {{"token", token}}, admin).status == 409);
    }
    CHECK(fs::exists(gamma / "part-0.safetensors") && fs::exists(alpha / "part-0.safetensors"));
    const auto deleted = server.send("POST", "/api/models/" + hf_id + "/delete", {{"token", token}}, admin);
    CHECK(deleted.status == 200);
    CHECK(deleted.document().at("files") == 4);
    CHECK(deleted.document().at("total_bytes") == total);
    CHECK(!fs::exists(alpha / "config.json") && !fs::exists(alpha / "model.safetensors.index.json"));
    CHECK(!fs::exists(alpha / "part-0.safetensors") && !fs::exists(alpha / "part-1.safetensors"));
    CHECK(fs::exists(alpha / "tokenizer.json"));
    CHECK(server.get("/api/models/" + hf_id + "/deletion", admin).status == 404);
    const auto shards = server.get("/api/models/" + tq_id + "/deletion", admin).document();
    CHECK(shards.at("files") == json({fs::canonical(root / "beta-00001-of-00002.tq").u8string(),
        fs::canonical(root / "beta-00002-of-00002.tq").u8string()}));
    CHECK(shards.at("total_bytes") == 23);
    const auto removed = server.send("POST", "/api/models/" + tq_id + "/delete",
        {{"token", shards.at("token").get<std::string>()}}, admin);
    CHECK(removed.status == 200);
    CHECK(removed.document().at("files") == 2);
    CHECK(!fs::exists(root / "beta-00001-of-00002.tq") && !fs::exists(root / "beta-00002-of-00002.tq"));
    CHECK(fs::exists(root / "beta.mfq.notes.json") && fs::exists(gamma / "part-0.safetensors"));
    CHECK(listed().size() == 1);
    CHECK(listed()[0].at("name") == "gamma");
}

/** Verifies child access failures are isolated while unavailable selections remain explicit errors. */
void directory_browsing() {
    Fixture f;
    const auto root = f.root / "browse", busy = root / "busy", visible = root / "visible";
    fs::create_directories(busy);
    fs::create_directory(visible);
    write_hf(root / "hf");
    write_mfq(root / "visible.mfq");
    auto server = f.start();
    const auto browse = [&](const fs::path& path) {
        return server.get("/api/models/directories?path=" + httplib::encode_uri_component(path.u8string()));
    };
    const auto initial = browse(root);
    CHECK(initial.status == 200);
    const auto initial_document = initial.document();
    std::string busy_id, visible_id;
    for (const auto& entry : initial_document.at("data")) {
        if (entry.at("name") == "busy") busy_id = entry.at("id");
        if (entry.at("name") == "visible") visible_id = entry.at("id");
    }
    CHECK(!busy_id.empty() && !visible_id.empty());
    {
#ifdef _WIN32
        // Simulate access denial by changing only the temporary subdirectory's ACL, then restore it on scope exit.
        struct Unavailable {
            fs::path path;
            std::vector<BYTE> saved;
            SECURITY_INFORMATION restore = DACL_SECURITY_INFORMATION;
            explicit Unavailable(const fs::path& path) : path(path) {
                DWORD bytes = 0;
                GetFileSecurityW(path.c_str(), DACL_SECURITY_INFORMATION, nullptr, 0, &bytes);
                CHECK(GetLastError() == ERROR_INSUFFICIENT_BUFFER);
                saved.resize(bytes);
                CHECK(GetFileSecurityW(path.c_str(), DACL_SECURITY_INFORMATION, saved.data(), bytes, &bytes));
                SECURITY_DESCRIPTOR_CONTROL control = 0;
                DWORD revision = 0;
                CHECK(GetSecurityDescriptorControl(saved.data(), &control, &revision));
                restore |= (control & SE_DACL_PROTECTED) ? PROTECTED_DACL_SECURITY_INFORMATION : UNPROTECTED_DACL_SECURITY_INFORMATION;
                PSECURITY_DESCRIPTOR denied = nullptr;
                CHECK(ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(D;;FA;;;WD)", SDDL_REVISION_1, &denied, nullptr));
                const auto applied = SetFileSecurityW(path.c_str(), DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION, denied);
                LocalFree(denied);
                CHECK(applied);
            }
            ~Unavailable() { SetFileSecurityW(path.c_str(), restore, saved.data()); }
        } unavailable{busy};
        std::error_code error;
        (void)fs::canonical(busy, error);
        CHECK(error);
#else
        struct Unavailable {
            fs::path path;
            ~Unavailable() { std::error_code ignored; fs::permissions(path, fs::perms::owner_all, ignored); }
        } unavailable{busy};
        fs::permissions(busy, fs::perms::none);
        std::error_code error;
        fs::directory_iterator probe(busy, error);
        if (!error) {
            std::cout << "SKIP unavailable directory: process can bypass permissions\n";
        } else
#endif
        {
            const auto listed = browse(root);
            CHECK(listed.status == 200);
            const auto document = listed.document();
            CHECK(document.at("model_file_count") == 2);
            CHECK(document.at("files").size() == 1);
            CHECK(document.at("data").size() == 2);
            CHECK(document.at("data")[0].at("name") == "hf");
            CHECK(document.at("data")[1].at("name") == "visible");
            const auto selected = browse(busy);
            CHECK(selected.status == 404);
            CHECK(selected.document().at("error").at("code") == "model_directory_not_found");
            CHECK(server.get("/api/models/directories?directory_id=" + busy_id).status == 404);
        }
    }
    CHECK(browse(busy).status == 200);
#ifndef _WIN32
    fs::create_directory_symlink(root / "missing", root / "dangling");
    CHECK(browse(root).status == 200);
#endif
    CHECK(fs::remove(visible));
    for (const auto& path : {visible, root / "missing", root / "visible.mfq"}) {
        const auto missing = browse(path);
        CHECK(missing.status == 404);
        CHECK(missing.document().at("error").at("code") == "model_directory_not_found");
    }
    CHECK(server.get("/api/models/directories?directory_id=" + visible_id).status == 404);
    server.stop();
}

void model_roots() {
    Fixture f;
    const auto first = f.root / "first", second = f.root / "second", environment = f.root / "environment";
    write_hf(first / "one");
    write_hf(second / "two");
    write_hf(environment / "ignored");
    auto explicit_roots = f.start({"--model-dir", first.u8string(), "--model-dir", second.u8string()}, {{"TYLOINFER_SERVER_MODEL_DIRS", environment.u8string()}});
    CHECK(explicit_roots.get("/api/models").document().at("data").size() == 2);
    CHECK(explicit_roots.get("/api/models/directories").document().at("data").size() == 2);
    explicit_roots.stop();
    auto from_environment = f.start({}, {{"TYLOINFER_SERVER_MODEL_DIRS", environment.u8string()}});
    CHECK(from_environment.get("/api/models").document().at("data")[0].at("name") == "ignored");
    from_environment.stop();
#ifndef _WIN32
    fs::create_directory_symlink(first, f.root / "alias");
    auto alias = f.start({"--model-dir", first.u8string(), "--model-dir", (f.root / "alias").u8string()});
    CHECK(alias.get("/api/models/directories").document().at("data").size() == 1);
    alias.stop();
#endif
    write_mfq(environment / "ignored.mfq");
    auto dual_format = f.start({"--model-dir", environment.u8string()});
    const auto artifacts = dual_format.get("/api/models").document().at("data");
    CHECK(artifacts.size() == 1);
    CHECK(artifacts[0].at("format") == "tq");
}

void databases_and_migration() {
    Fixture f;
    fs::create_directory(f.data);
    const auto database = f.data / "server.sqlite3";
    {
        Database db(database);
        db.execute("CREATE TABLE schema_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL); INSERT INTO schema_meta VALUES('schema_version','99');");
    }
    auto before = read_file(database);
    auto& unsupported = f.launch(f.data);
    CHECK(unsupported.wait() != 0);
    CHECK(unsupported.output().find("unsupported database schema") != std::string::npos);
    CHECK(read_file(database) == before);
    write_file(database, "corrupt data must not be overwritten");
    before = read_file(database);
    CHECK(f.launch(f.data).wait() != 0);
    CHECK(read_file(database) == before);
    fs::remove(database);
    {
        Database db(database);
        db.execute("CREATE TABLE existing_data(value TEXT); INSERT INTO existing_data VALUES('must survive');");
    }
    before = read_file(database);
    CHECK(f.launch(f.data).wait() != 0);
    CHECK(read_file(database) == before);
    fs::remove(database);
    for (const auto& policy : std::vector<json>{{{"enabled", true}, {"key_hash", ""}},
            {{"enabled", "yes"}, {"key_hash", std::string(64, 'a')}}, {{"enabled", true}, {"key_hash", std::string(64, 'A')}}}) {
        write_file(f.data / "auth.json", policy.dump());
        CHECK(f.launch(f.data, {}, root_environment).wait() != 0);
        CHECK(read_file(f.data / "auth.json") == policy.dump());
        CHECK(!fs::exists(database));
    }
    fs::remove(f.data / "auth.json");
    const auto explicit_database = f.root / "custom.sqlite3";
    auto explicit_server = f.start({"--db", explicit_database.u8string()});
    CHECK(fs::exists(explicit_database));
    CHECK(!fs::exists(database));
    explicit_server.stop();
    {
        Database db(explicit_database);
        db.execute("CREATE TRIGGER reject_version_rewrite BEFORE INSERT ON schema_meta "
            "WHEN NEW.key='schema_version' BEGIN SELECT RAISE(ABORT,'current version must not be rewritten'); END;");
        auto current = f.start({"--db", explicit_database.u8string()});
        CHECK(current.get("/api/auth/keys").status == 200);
        current.stop();
        db.execute("DROP TRIGGER reject_version_rewrite;");
        db.execute("ALTER TABLE api_keys DROP COLUMN role;");
        auto& incomplete = f.launch(f.data, {"--db", explicit_database.u8string()});
        CHECK(incomplete.wait() != 0);
        CHECK(db.scalar("SELECT value FROM schema_meta WHERE key='schema_version'") == "19");
        CHECK(db.scalar("SELECT count(*) FROM pragma_table_info('api_keys') WHERE name='role'") == "0");
    }
    for (const auto version : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}) {
        const auto data = f.root / ("version-" + std::to_string(version));
        fs::create_directory(data);
        const auto legacy = data / "tyloinfer-server.sqlite3";
        {
            Database db(legacy);
            db.execute(mfq::server::schema_sql);
            db.execute("INSERT INTO schema_meta VALUES('schema_version','" + std::to_string(version) + "');"
                "ALTER TABLE api_keys DROP COLUMN role; ALTER TABLE responses DROP COLUMN settings_json;"
                "ALTER TABLE responses DROP COLUMN performance_json; ALTER TABLE responses DROP COLUMN output_message_id;"
                "DROP TABLE generation_presets;"
                "CREATE TABLE generation_presets(id TEXT PRIMARY KEY,name TEXT NOT NULL UNIQUE,model TEXT NOT NULL,mode TEXT NOT NULL,"
                    "settings_json TEXT NOT NULL,context_size INTEGER CHECK(context_size>=512),created_at TEXT NOT NULL,updated_at TEXT NOT NULL);"
                "ALTER TABLE jobs DROP COLUMN progress_data_json;"
                "ALTER TABLE jobs DROP COLUMN archived_at;"
                "INSERT INTO generation_presets(id,name,model,mode,settings_json,created_at,updated_at) VALUES('preset','old','model','chat','{}','2020','2020');"
                "INSERT INTO sessions(id,model,mode,state,revision,created_at,updated_at,metadata_json) VALUES('session','model','chat','open',7,'2020','2020','{\"keep\":1}');"
                "INSERT INTO messages(id,role,parts_json,created_at) VALUES('message','user','[]','2020');"
                "INSERT INTO session_messages VALUES('session',1,'message');"
                "INSERT INTO responses(id,session_id,request_id,request_fingerprint,status,input_message_id,output_json,created_at) "
                    "VALUES('response','session','request','fingerprint','completed','message','{\"text\":\"history\"}','2020');"
                "INSERT INTO jobs(id,kind,status,payload_json,progress,cancel_requested,created_at,updated_at) VALUES('job','download','completed','{}',1,0,'2020','2020');"
                "INSERT INTO api_keys VALUES('00000000-0000-4000-8000-000000000001','legacy',"
                "'762c08fc17a1cc5f00d248f8b50f2f2f4d17ff2934ac31e64deacb3f5bb3f2ec','old','[\"admin\"]',NULL,NULL,'2020-01-01T00:00:00+00:00',NULL);"
                "CREATE TABLE preserve_me(value TEXT); INSERT INTO preserve_me VALUES('existing data');");
        }
        auto server = f.start_at(data, {}, root_environment);
        CHECK(!fs::exists(data / "server.sqlite3"));
        CHECK(server.get("/api/auth/keys", bearer("old-key")).status == 200);
        server.stop();
        Database db(legacy);
        CHECK(db.scalar("SELECT value FROM schema_meta WHERE key='schema_version'") == "19");
        CHECK(db.scalar("SELECT name||'|'||metadata_json FROM generation_presets WHERE id='preset'") == "old|{}");
        db.execute("INSERT INTO generation_presets(id,name,settings_json,metadata_json,created_at,updated_at) VALUES('nullable','new','{}','{}','2020','2020');");
        CHECK(db.scalar("SELECT model IS NULL AND mode IS NULL FROM generation_presets WHERE id='nullable'") == "1");
        CHECK(db.scalar("SELECT revision||'|'||metadata_json FROM sessions WHERE id='session'") == "7|{\"keep\":1}");
        CHECK(db.scalar("SELECT message_id FROM session_messages WHERE session_id='session' AND ordinal=1") == "message");
        CHECK(db.scalar("SELECT output_json FROM responses WHERE id='response'") == "{\"text\":\"history\"}");
        CHECK(db.scalar("SELECT performance_json IS NULL AND output_message_id IS NULL AND settings_json IS NULL FROM responses WHERE id='response'") == "1");
        CHECK(db.scalar("SELECT status||'|'||progress_data_json FROM jobs WHERE id='job'") == "completed|{}");
        CHECK(db.scalar("SELECT value FROM preserve_me") == "existing data");
        if (version == 18) {
            db.execute("UPDATE schema_meta SET value='18' WHERE key='schema_version'; ALTER TABLE api_keys DROP COLUMN role;"
                "CREATE TABLE generation_presets_nullable(dummy TEXT);");
            CHECK(f.launch(data).wait() != 0);
            CHECK(db.scalar("SELECT value FROM schema_meta WHERE key='schema_version'") == "18");
            CHECK(db.scalar("SELECT count(*) FROM pragma_table_info('api_keys') WHERE name='role'") == "0");
            CHECK(db.scalar("SELECT name FROM generation_presets WHERE id='preset'") == "old");
        }
    }
}

} // namespace

int run_tests(const fs::path& server, const std::string& selection) {
    executable = fs::absolute(server);
    using Tests = std::vector<std::pair<const char*, std::function<void()>>>;
    const Tests tests = selection == "--models" ? Tests{{"model catalog", model_catalog}, {"directory browsing", directory_browsing},
        {"model roots", model_roots}, {"model inspection", model_inspection}, {"model deletion", model_deletion}} :
        selection == "--policies" ? Tests{{"runtime policies", runtime_policies}, {"runtime profiles", runtime_profiles}} :
        Tests{
        {"CLI and lifetime", cli_and_lifetime}, {"locks and ports", locks_and_ports}, {"static files", static_files},
        {"policy and protocols", policy_and_protocols}, {"keys and network", keys_and_network},
        {"media uploads", media_uploads},
        {"prefix cache", prefix_cache},
        {"runtime observation", runtime_observation},
        {"runtime policies", runtime_policies},
        {"runtime profiles", runtime_profiles},
        {"shutdown with pending work", shutdown_with_pending_work}, {"database migrations", databases_and_migration}};
    int failures = 0;
    for (const auto& test : tests) {
        try { test.second(); std::cout << "PASS " << test.first << '\n'; }
        catch (const std::exception& error) { ++failures; std::cerr << "FAIL " << test.first << ": " << error.what() << '\n'; }
    }
    std::cout << checks << " checks, " << failures << " failed scenarios\n";
    return failures ? 1 : 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    const std::wstring selection = argc == 3 ? std::wstring(argv[2]) : std::wstring();
    if (argc != 2 && !(selection == L"--models" || selection == L"--policies")) return 2;
    return run_tests(argv[1], argc == 3 ? (selection == L"--policies" ? "--policies" : "--models") : "");
}
#else
int main(int argc, char** argv) {
    const std::string selection = argc == 3 ? argv[2] : std::string();
    if (argc != 2 && !(selection == "--models" || selection == "--policies")) return 2;
    return run_tests(fs::u8path(argv[1]), selection);
}
#endif

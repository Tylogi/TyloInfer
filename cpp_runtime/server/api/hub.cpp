#include "hub.h"
#include "hub/hub.h"
#include "http.h"
#include "instance.h"
#include "jobs/jobs.h"
#include "models/catalog.h"
#include "models/errors.h"
#include "mfq_paged_prefix_cache.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <map>
#include <regex>
#include <sstream>
#include <thread>
#include <tuple>
#include <utility>
#ifndef _WIN32
#include <sys/utsname.h>
#include <unistd.h>
#endif

namespace mfq::server {
namespace {
using Json = nlohmann::json;
std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}
void fields(const Json& body, std::initializer_list<const char*> allowed) {
    if (!body.is_object()) throw ApiError(400, "invalid_request", "expected a JSON object");
    for (const auto& item : body.items()) if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end())
        throw ApiError(400, "invalid_request", "unknown parameter: " + item.key());
}
std::string text(const Json& body, const char* key, bool required = false) {
    if (!body.contains(key) || body[key].is_null()) {
        if (!required) return {};
    } else if (body[key].is_string() && !body[key].get_ref<const std::string&>().empty() && body[key].get_ref<const std::string&>().size() <= 4096 &&
            body[key].get_ref<const std::string&>().find('\0') == std::string::npos) return body[key].get<std::string>();
    throw ApiError(400, "invalid_request", std::string(key) + " must be a nonempty string");
}
Hub::Provider provider(const std::string& name) {
    if (name == "huggingface") return Hub::Provider::hf;
    if (name == "modelscope") return Hub::Provider::ms;
    throw ApiError(400, "invalid_provider", "provider must be huggingface or modelscope");
}
const char* provider(Hub::Provider name) { return name == Hub::Provider::hf ? "huggingface" : "modelscope"; }
Json directory(const Hub::Directory& value) {
    return {{"directory", value.selected.empty() ? Json(nullptr) : Json(path_utf8(value.selected))},
        {"actual_directory", path_utf8(value.actual)}, {"default_directory", path_utf8(value.fallback)},
        {"disk_total_bytes", value.space ? Json(value.space->capacity) : Json(nullptr)},
        {"disk_free_bytes", value.space ? Json(value.space->free) : Json(nullptr)},
        {"disk_available_bytes", value.space ? Json(value.space->available) : Json(nullptr)}, {"evict_prefix_cache", false}};
}
Json variant_groups(const char* source, const std::string& repo, const std::vector<Hub::File>& files) {
    std::map<std::string, Json> groups;
    static const std::regex shard(R"((.*)-[0-9]{5}-of-[0-9]{5}(\.tq|\.mfq|\.gguf))");
    for (const auto& file : files) {
        const auto path = std::filesystem::u8path(file.name);
        const auto extension = path.extension();
        if (extension != ".tq" && extension != ".mfq" && extension != ".safetensors" && extension != ".gguf") continue;
        const auto format = extension == ".gguf" ? "gguf" : extension == ".safetensors" ? "hf" : "tq";
        std::string key = file.name;
        if (extension == ".safetensors") key = path_utf8(path.parent_path() / "checkpoint");
        else { std::smatch match; if (std::regex_match(file.name, match, shard)) key = match[1].str() + match[2].str(); }
        auto& group = groups[key];
        // HF directory names become model names; include the source so repositories do not all download as checkpoint.
        const auto label = extension == ".safetensors" ? std::string(source) + "/" + repo + "/" + key : key;
        if (group.is_null()) group = {{"id", key}, {"label", label}, {"format", format}, {"files", Json::array()}, {"byte_size", 0},
            {"configuration", {{"status", "unknown"}, {"recommendation", "unknown"}, {"reasons", Json::array()}}}};
        group["files"].push_back(file.name); group["byte_size"] = group["byte_size"].get<std::uint64_t>() + file.size;
    }
    Json variants = Json::array();
    for (auto& [name, value] : groups) variants.push_back(std::move(value));
    return variants;
}
Json repository(const Hub::Repo& repo) {
    Json files = Json::array();
    std::uint64_t total = 0;
    for (const auto& file : repo.files) {
        total += file.size;
        files.push_back({{"name", file.name}, {"byte_size", file.size}, {"sha256", file.sha256.empty() ? Json(nullptr) : Json(file.sha256)}});
    }
    return {{"provider", provider(repo.source.provider)}, {"repo_id", repo.source.repo}, {"revision", repo.source.revision},
        {"source_url", repo.url}, {"downloads", nullptr}, {"likes", nullptr}, {"total_bytes", total},
        {"files", files}, {"variants", variant_groups(provider(repo.source.provider), repo.source.repo, repo.files)},
        {"tags", Json::array()}, {"architectures", Json::array()},
        {"modalities", Json::array()}, {"gated", nullptr}, {"runtime_compatible", nullptr}};
}
// Query parameters follow the Python hub routes: an absent value keeps its documented default.
std::string parameter(const httplib::Request& request, const char* name, const std::string& fallback, std::size_t limit) {
    if (!request.has_param(name)) return fallback;
    const auto value = request.get_param_value(name);
    if (value.size() > limit) throw ApiError(400, "invalid_request", std::string(name) + " is longer than " + std::to_string(limit));
    return value;
}
std::string choice(const httplib::Request& request, const char* name, const std::string& fallback, std::initializer_list<const char*> values) {
    const auto value = parameter(request, name, fallback, 64);
    if (std::find(values.begin(), values.end(), value) == values.end())
        throw ApiError(400, "invalid_request", std::string(name) + " must be one of the supported values");
    return value;
}
bool flag(const httplib::Request& request, const char* name, bool fallback) {
    if (!request.has_param(name)) return fallback;
    const auto value = request.get_param_value(name);
    if (value != "true" && value != "false" && value != "1" && value != "0")
        throw ApiError(400, "invalid_request", std::string(name) + " must be a boolean");
    return value == "true" || value == "1";
}
std::size_t quantity(const httplib::Request& request, const char* name, std::size_t fallback, std::size_t minimum, std::size_t maximum) {
    const auto value = parameter(request, name, std::to_string(fallback), 8);
    std::size_t result = 0;
    if (!value.empty() && value.find_first_not_of("0123456789") == std::string::npos) {
        try { result = static_cast<std::size_t>(std::stoull(value)); } catch (...) {}
    } else throw ApiError(400, "invalid_request", std::string(name) + " must be an integer");
    if (result < minimum || result > maximum)
        throw ApiError(400, "invalid_request", std::string(name) + " must be between " + std::to_string(minimum) + " and " + std::to_string(maximum));
    return result;
}
const char* repository_url(Hub::Provider source) {
    return source == Hub::Provider::hf ? "https://huggingface.co/" : "https://modelscope.cn/models/";
}
Json summary(const Hub::Summary& item) {
    return {{"provider", provider(item.provider)}, {"repo_id", item.repo}, {"source_url", std::string(repository_url(item.provider)) + item.repo},
        {"author", item.author.empty() ? Json(nullptr) : Json(item.author)},
        {"description", item.description.empty() ? Json(nullptr) : Json(item.description)},
        {"downloads", item.downloads}, {"likes", item.likes}, {"total_bytes", item.bytes},
        {"updated_at", item.updated.empty() ? Json(nullptr) : Json(item.updated)},
        {"formats", item.formats}, {"trending_score", item.trending}};
}
Json searched(const Hub::Page& value) {
    Json data = Json::array();
    for (const auto& item : value.items) data.push_back(summary(item));
    return {{"data", std::move(data)}, {"next_cursor", value.cursor.empty() ? Json(nullptr) : Json(value.cursor)}};
}
Json system_profile(const std::string& backend) {
    std::int64_t physical = 0, available = 0;
    const char* platform = "unknown";
    const char* machine = "unknown";
#ifdef _WIN32
    platform = "Windows";
#else
    utsname names{};
    if (uname(&names) == 0) { platform = names.sysname; machine = names.machine; }
    const auto pages = sysconf(_SC_PHYS_PAGES), block = sysconf(_SC_PAGESIZE);
    if (pages > 0 && block > 0) physical = static_cast<std::int64_t>(pages) * static_cast<std::int64_t>(block);
#ifdef _SC_AVPHYS_PAGES
    const long free_pages = sysconf(_SC_AVPHYS_PAGES);
    if (free_pages > 0 && block > 0) available = static_cast<std::int64_t>(free_pages) * static_cast<std::int64_t>(block);
#endif
#endif
    const auto normalized = backend == "cuda" ? "cuda" : backend == "hip" ? "rocm" : backend.empty() ? "cpu" : "unknown";
    const auto capacity = physical ? Json(physical) : Json(nullptr);
    return {{"platform", platform}, {"machine", machine}, {"backend", normalized}, {"cpu_name", nullptr},
        {"cpu_cores", static_cast<int>(std::max(1u, std::thread::hardware_concurrency()))},
        {"gpu_names", Json::array()}, {"gpu_cores", nullptr}, {"physical_memory_bytes", capacity},
        {"available_memory_bytes", available ? Json(available) : Json(nullptr)},
        {"runtime_memory_budget_bytes", nullptr},
        {"memory_pools", Json::array({{{"kind", "ram"}, {"device", nullptr}, {"capacity_bytes", capacity},
            {"bandwidth_bytes_per_second", nullptr}}})}};
}
// The official catalog keeps the Python field set; residency estimates stay null natively.
std::string official_label(const std::string& name) {
    const auto lower = lowercase(name);
    for (std::size_t at = 0; at < lower.size(); ++at) {
        if (lower[at] != '-') continue;
        for (const auto* tag : {"ewq", "ew", "mfq", "tq"}) {
            const auto length = std::strlen(tag);
            if (lower.compare(at + 1, length, tag) != 0) continue;
            const auto after = at + 1 + length;
            if (after == lower.size() || lower[after] == '-') return name.substr(0, at);
        }
    }
    return name;
}
Json official_model(const std::vector<const Hub::OfficialModel*>& group, const Json& system) {
    const auto& chosen = *group.front();
    const auto name = chosen.summary.repo.substr(chosen.summary.repo.find('/') + 1);
    const auto display = official_label(name);
    const auto modalities = chosen.summary.modalities.empty() ? std::vector<std::string>{"text"} : chosen.summary.modalities;
    auto capabilities = Json::array();
    if (std::any_of(modalities.begin(), modalities.end(), [](const auto& item) { return item == "image" || item == "video"; })) capabilities.push_back("Vision");
    if (std::find(modalities.begin(), modalities.end(), "audio") != modalities.end()) capabilities.push_back("Audio");
    const auto flash_next = lowercase(display).find("flash-next") != std::string::npos;
    const auto description = flash_next ?
        "High-performance, fast compact MoE model with large PLE tables. PLE tables stream row-wise from SSD without full-table memory residency." :
        chosen.summary.description.empty() ? display + " model published by Tylogi in TQ format." : chosen.summary.description;
    const auto description_zh = flash_next ?
        "高性能、快速的中小型 MoE 模型，带有大规模的 PLE 表，可高效卸载至 SSD，按行读取且无需整表常驻内存。" :
        display + " 模型，提供 TQ 格式的精度版本。";
    Json sources = Json::array();
    for (const auto* item : group)
        sources.push_back({{"provider", provider(item->summary.provider)}, {"repo_id", item->summary.repo},
            {"revision", item->source.revision.empty() ? Json(nullptr) : Json(item->source.revision)},
            {"url", std::string(repository_url(item->summary.provider)) + item->summary.repo},
            {"available", !item->files.empty()}});
    auto reasons = Json::array({"Configuration requirements could not be determined."});
    if (chosen.files.empty()) reasons.push_back("The catalog is available offline; repository metadata could not be refreshed.");
    const auto digest = mfq::cache::block_hash_hex(mfq::cache::sha256(lowercase(name)));
    return {{"id", "tylogi-" + digest.substr(0, 16)},
        {"name", display}, {"family", display}, {"architecture", "unknown"}, {"attention_profile", nullptr},
        {"description", description}, {"description_zh", description_zh}, {"parameter_label", nullptr},
        {"active_parameter_label", nullptr}, {"parameter_breakdown", nullptr}, {"mtp_supported", nullptr},
        {"cache_profile", nullptr}, {"modalities", modalities}, {"capabilities", capabilities},
        {"precision_options", Json::array()}, {"license", nullptr}, {"supports_ssd_streaming", false},
        {"sources", sources}, {"selected_source", sources.front()},
        {"revision", chosen.source.revision.empty() ? "main" : chosen.source.revision},
        {"downloads", chosen.summary.downloads}, {"likes", chosen.summary.likes},
        {"updated_at", chosen.summary.updated.empty() ? Json(nullptr) : Json(chosen.summary.updated)},
        {"variants", variant_groups(provider(chosen.summary.provider), chosen.summary.repo, chosen.files)},
        {"published_at", nullptr},
        {"configuration", {{"status", "unknown"}, {"recommendation", "unknown"}, {"required_memory_bytes", nullptr},
            {"available_memory_bytes", system.at("physical_memory_bytes")}, {"reasons", reasons}}}};
}
bool weight_file(const std::string& name) {
    static const std::vector<std::string> suffixes{".tq", ".mfq", ".safetensors", ".gguf", ".bin"};
    return std::find(suffixes.begin(), suffixes.end(), lowercase(std::filesystem::path(name).extension().string())) != suffixes.end();
}
Hub::Request download(const std::string& kind, Json& body, const Hub& hub) {
    if (kind != "download.huggingface" && kind != "download.modelscope")
        throw ApiError(501, "job_kind_not_supported", "this job kind is not available in the native server");
    fields(body, {"repo_id", "revision", "destination", "download_directory", "include", "exclude", "max_workers", "direct", "expected_bytes", "repo_type"});
    Hub::Request result;
    result.source = {provider(kind.substr(9)), text(body, "repo_id", true), text(body, "revision")};
    result.destination = text(body, "destination", true);
    if (body.contains("repo_type") && body["repo_type"] != "model") throw ApiError(501, "repo_type_not_supported", "only model repositories are supported");
    auto selected = text(body, "download_directory");
    if (selected.empty()) selected = path_utf8(hub.directory().actual);
    result.directory = std::filesystem::u8path(selected); body["download_directory"] = selected;
    for (auto [key, target, limit] : {std::tuple{"include", &result.include, 4096u}, {"exclude", &result.exclude, 64u}}) {
        if (!body.contains(key)) continue;
        if (!body[key].is_array() || body[key].size() > limit) throw ApiError(400, "invalid_request", "invalid file filters");
        for (const auto& value : body[key]) {
            if (!value.is_string() || value.get_ref<const std::string&>().empty() || value.get_ref<const std::string&>().size() > 1024)
                throw ApiError(400, "invalid_request", "file patterns must be nonempty strings");
            target->push_back(value.get<std::string>());
        }
    }
    if (body.contains("max_workers")) {
        if (!body["max_workers"].is_number_integer() || body["max_workers"] < 1 || body["max_workers"] > 16)
            throw ApiError(400, "invalid_request", "max_workers must be between 1 and 16");
        result.workers = body["max_workers"].get<std::size_t>();
    }
    if (body.contains("direct")) {
        if (!body["direct"].is_boolean()) throw ApiError(400, "invalid_request", "direct must be boolean");
        result.direct = body["direct"].get<bool>();
    }
    if (body.contains("expected_bytes") && !body["expected_bytes"].is_null()) {
        if (!body["expected_bytes"].is_number_integer() || body["expected_bytes"] < 0 || body["expected_bytes"] > INT64_MAX)
            throw ApiError(400, "invalid_request", "expected_bytes must be a nonnegative 64-bit integer");
        result.expected_bytes = body["expected_bytes"].get<std::uint64_t>();
    }
    return result;
}
std::string submit(Hub& hub, ModelCatalog& catalog, Jobs& jobs, const std::string& kind, Json body) {
    const auto request = download(kind, body, hub);
    return jobs.submit(kind, body, [&, request](const Jobs::Control& control) {
        control.progress("resolving");
        Hub::Result result;
        try {
            result = hub.download(request, {[&] { return control.cancelled.load(); },
                [&](std::uint64_t bytes, std::uint64_t total, std::size_t files) {
                    control.progress("downloading", total ? static_cast<double>(bytes) / total : 0,
                        {{"downloaded_bytes", bytes}, {"total_bytes", total}, {"files_completed", files}});
                }});
        } catch (const Hub::Error& error) {
            if (error.kind == Hub::Error::cancelled) throw Jobs::Cancelled{};
            throw;
        }
        auto models = Json::array();
        try { for (const auto& artifact : catalog.register_directory({{}, result.path})) models.push_back(artifact.id); }
        catch (const ModelError& error) { if (std::string(error.code) != "model_registration_failed") throw; }
        return Json{{"path", path_utf8(result.path)}, {"repo_id", result.source.repo}, {"revision", result.source.revision},
            {"total_bytes", result.bytes}, {"files_completed", result.files}, {"models", models},
            {"registration", models.empty() ? "unsupported" : "registered"}};
    });
}
}
void register_hub(httplib::Server& http, Hub& hub, ModelCatalog& catalog, Jobs& jobs, const std::string& backend) {
    http.Get("/api/v1/hub/download-directory", [&](const httplib::Request&, httplib::Response& response) { set_json(response, directory(hub.directory())); });
    http.Put("/api/v1/hub/download-directory", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request); fields(body, {"directory_id", "evict_prefix_cache"});
        if (body.contains("evict_prefix_cache")) {
            if (!body["evict_prefix_cache"].is_boolean()) throw ApiError(400, "invalid_request", "evict_prefix_cache must be boolean");
            if (body["evict_prefix_cache"].get<bool>()) throw ApiError(501, "cache_reclaim_not_supported", "download cache reclamation is not available");
        }
        if (!body.contains("directory_id")) { set_json(response, directory(hub.directory())); return; }
        const auto id = text(body, "directory_id");
        set_json(response, directory(hub.configure(id.empty() ? std::optional<std::filesystem::path>{} : catalog.browse(ModelCatalog::DirectorySource{id, {}}).path)));
    });
    http.Get("/api/v1/hub/models", [&](const httplib::Request& request, httplib::Response& response) {
        const auto selected = provider(parameter(request, "provider", "", 32));
        const auto query = parameter(request, "query", {}, 255);
        const auto format = choice(request, "format", "all", {"all", "tq", "gguf", "mlx"});
        const auto sort = choice(request, "sort", "downloads", {"relevance", "downloads", "likes", "updated"});
        const auto direction = choice(request, "direction", "desc", {"asc", "desc"});
        const auto limit = quantity(request, "limit", 20, 1, 100);
        const auto cursor = parameter(request, "cursor", {}, 64);
        if (!cursor.empty() && (cursor.size() < 16 || !std::regex_match(cursor, std::regex("[A-Za-z0-9_-]+"))))
            throw ApiError(400, "invalid_request", "cursor is not a search cursor");
        std::vector<std::string> formats;
        if (request.has_param("formats")) {
            std::istringstream list(parameter(request, "formats", "", 64));
            std::string item;
            while (std::getline(list, item, ',')) {
                if (item != "tq" && item != "gguf" && item != "mlx") throw ApiError(400, "invalid_request", "formats must list tq, gguf or mlx");
                if (std::find(formats.begin(), formats.end(), item) == formats.end()) formats.push_back(item);
            }
            if (formats.empty()) throw ApiError(400, "invalid_request", "formats must list at least one format");
        }
        // The browser path is the one Studio uses; the plain search keeps the legacy single-page shape.
        const bool browse = flag(request, "browse", false) || !formats.empty() || !cursor.empty() || sort == "relevance" || direction == "asc";
        if (!browse) {
            auto legacy = hub.search(selected, query, format, sort, direction, limit);
            legacy.cursor.clear(); // the single-page search has no continuation, unlike the browser
            set_json(response, searched(legacy));
            return;
        }
        if (selected == Hub::Provider::ms && direction == "asc")
            throw ApiError(400, "invalid_hub_search", "ModelScope does not support ascending search order.");
        if (formats.empty()) formats = format == "all" ? std::vector<std::string>{"tq", "gguf", "mlx"} : std::vector<std::string>{format};
        set_json(response, searched(hub.browse(selected, query, formats, sort, direction, limit, cursor)));
    });
    http.Get("/api/v1/hub/official", [&](const httplib::Request& request, httplib::Response& response) {
        const auto system = system_profile(backend);
        const auto models = hub.official(flag(request, "refresh", false));
        std::map<std::string, std::vector<const Hub::OfficialModel*>> groups;
        for (const auto& model : models)
            groups[lowercase(model.summary.repo.substr(model.summary.repo.find('/') + 1))].push_back(&model);
        auto data = Json::array();
        for (const auto& [name, group] : groups) data.push_back(official_model(group, system));
        std::sort(data.begin(), data.end(), [](const auto& left, const auto& right) {
            const auto older = [](const Json& model) { return model["updated_at"].is_string() ? model["updated_at"].get<std::string>() : std::string(); };
            return older(left) > older(right);
        });
        set_json(response, {{"system", system}, {"data", std::move(data)}, {"refreshing", hub.refreshing()}});
    });
    http.Get(R"(/api/v1/hub/models/([^/]+)/([^/]+)/([^/]+)/readme)", [&](const httplib::Request& request, httplib::Response& response) {
        const auto revision = parameter(request, "revision", {}, 255);
        std::string content;
        try {
            content = hub.readme(provider(request.matches[1]), request.matches[2].str() + "/" + request.matches[3].str(), revision);
        } catch (const Hub::Error& error) {
            if (error.kind != Hub::Error::invalid) throw ApiError(502, "model_card_unavailable", "Could not retrieve the model card.");
            throw;
        }
        set_json(response, {{"content", content.empty() ? Json(nullptr) : Json(content)}});
    });
    http.Get("/api/v1/hub/models/([^/]+)/([^/]+)/([^/]+)", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, repository(hub.info({provider(request.matches[1]), request.matches[2].str() + "/" + request.matches[3].str(), request.get_param_value("revision")},
            {request.is_connection_closed, {}})));
    });
    http.Post("/api/v1/hub/resolve", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request); fields(body, {"reference", "fallback_provider"});
        auto name = text(body, "reference", true), fallback = text(body, "fallback_provider");
        auto selected = provider(fallback.empty() ? "huggingface" : fallback);
        static const std::regex url(R"(https://(huggingface\.co|modelscope\.cn)/(?:models/)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/?)");
        std::smatch match;
        if (std::regex_match(name, match, url)) { selected = match[1] == "huggingface.co" ? Hub::Provider::hf : Hub::Provider::ms; name = match[2]; }
        set_json(response, repository(hub.info({selected, name, {}}, {request.is_connection_closed, {}})));
    });
    http.Post("/api/v1/hub/download-reuse", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request);
        fields(body, {"provider", "repo_id", "revision", "destination", "download_directory"});
        const Hub::Source source{provider(text(body, "provider", true)), text(body, "repo_id", true), text(body, "revision")};
        const auto destination = text(body, "destination", true);
        const auto selected = text(body, "download_directory");
        Json reusable = Json::object();
        for (const auto& [name, bytes] : hub.reusable(source, destination,
                selected.empty() ? std::optional<std::filesystem::path>{} : std::filesystem::u8path(selected)))
            reusable[name] = bytes;
        set_json(response, {{"reusable_files", std::move(reusable)}});
    });
    http.Post("/api/v1/hub/download-sources/probe", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request);
        fields(body, {"sources"});
        if (!body.contains("sources") || !body["sources"].is_array() || body["sources"].empty() || body["sources"].size() > 2)
            throw ApiError(400, "invalid_request", "sources must contain one or two entries");
        auto data = Json::array();
        for (const auto& item : body["sources"]) {
            if (!item.is_object()) throw ApiError(400, "invalid_request", "sources must be objects");
            fields(item, {"provider", "repo_id", "revision"});
            const Hub::Source source{provider(text(item, "provider", true)), text(item, "repo_id", true), text(item, "revision")};
            Json entry{{"provider", provider(source.provider)}, {"repo_id", source.repo}, {"bytes_per_second", 0.0}};
            try {
                const auto repo = hub.info(source);
                for (const auto& file : repo.files) {
                    if (file.size < 262144 || !weight_file(file.name)) continue;
                    const auto route = hub.probe(source.provider, repo.source, file.name);
                    entry["bytes_per_second"] = route.bytes_per_second;
                    entry["route"] = route.proxy ? "proxy" : "direct";
                    break;
                }
            } catch (const Hub::Error&) {}
            data.push_back(std::move(entry));
        }
        set_json(response, {{"data", std::move(data)}});
    });
    http.Get("/api/v1/jobs/kinds", [](const httplib::Request&, httplib::Response& response) {
        set_json(response, {{"data", Json::array({{{"kind", "download.huggingface"}}, {{"kind", "download.modelscope"}}})}});
    });
    http.Post("/api/v1/jobs", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request); fields(body, {"kind", "payload"});
        if (!body.contains("payload")) throw ApiError(400, "invalid_request", "payload is required");
        set_json(response, jobs.get(submit(hub, catalog, jobs, text(body, "kind", true), body.at("payload"))), 202);
    });
    http.Post("/api/v1/jobs/([0-9a-f-]{36})/retry", [&](const httplib::Request& request, httplib::Response& response) {
        const auto previous = jobs.get(normalize_uuid(request.matches[1]));
        if (!Jobs::terminal(previous) || previous.at("status") == "succeeded") throw ApiError(409, "job_state_conflict", "only failed, cancelled or interrupted downloads can be retried");
        set_json(response, jobs.get(submit(hub, catalog, jobs, previous.at("kind"), previous.at("payload"))), 202);
    });
}
}

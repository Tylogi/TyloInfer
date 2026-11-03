#include "hub.h"
#include "fetch.h"
#include "instance.h"
#include "storage/files.h"
#include "mfq_paged_prefix_cache.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <set>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <stdio.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>
#endif

namespace mfq::server {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using Error = Hub::Error;

fs::path relative(const std::string& name, bool pattern = false) {
    if (name.empty() || name.size() > 1024 || name.front() == '/' || name.back() == '/' || name.find("//") != std::string::npos ||
        name.find_first_of("\\:<>|\"") != std::string::npos || (!pattern && name.find_first_of("*?") != std::string::npos) ||
        std::any_of(name.begin(), name.end(), [](unsigned char ch) { return ch < 32 || ch == 127; }))
        throw Error(Error::invalid, "invalid_download_path", "download paths must be relative file names");
    const auto path = fs::u8path(name);
    static const std::regex device(R"((CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?)", std::regex::icase);
    for (const auto& part : path)
        if (part.empty() || part == "." || part == ".." || part == ".tyloinfer-downloads" ||
            part.native().back() == '.' || part.native().back() == ' ' || std::regex_match(path_utf8(part), device))
            throw Error(Error::invalid, "invalid_download_path", "download path contains a reserved component");
    return path;
}
void safe(const fs::path& root, const fs::path& path) {
    auto current = root;
    for (const auto& part : path) {
        current /= part;
        if (fs::is_symlink(fs::symlink_status(current)))
            throw Error(Error::invalid, "unsafe_download_path", "download path contains a symbolic link");
    }
}
Json read(const fs::path& path) {
    if (fs::file_size(path) > 16 * 1024 * 1024) throw Error(Error::io, "download_state_invalid", "download state is too large");
    std::ifstream input(path);
    return Json::parse(input);
}
fs::path absolute(const std::string& value) {
    if (value.empty() || value.size() > 4096 || value.find('\0') != std::string::npos || !fs::u8path(value).is_absolute())
        throw Error(Error::invalid, "invalid_download_directory", "download directory must be an absolute path");
    return fs::weakly_canonical(fs::u8path(value));
}
const std::regex repository(R"([A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*)");
void source(Hub::Source& value) {
    if (value.repo.size() > 256 || !std::regex_match(value.repo, repository))
        throw Error(Error::invalid, "invalid_repository", "repository must be owner/name");
    if (value.revision.empty()) value.revision = value.provider == Hub::Provider::hf ? "main" : "master";
    if (value.revision.size() > 255 || std::any_of(value.revision.begin(), value.revision.end(), [](unsigned char ch) { return ch < 32 || ch == 127; }))
        throw Error(Error::invalid, "invalid_revision", "invalid repository revision");
}
// The resolve endpoint for one repository file; downloads and speed probes share it with the Python hub helper.
std::string content_path(Hub::Provider provider, const std::string& repo, const std::string& revision, const std::string& file) {
    const auto name = hub::escape(repo, true), tag = hub::escape(revision), path = hub::escape(file, true);
    return provider == Hub::Provider::hf ? "/" + name + "/resolve/" + tag + "/" + path :
        "/api/v1/models/" + name + "/repo?Revision=" + tag + "&FilePath=" + path;
}
bool matches(const std::string& pattern, const std::string& name) {
    std::size_t p = 0, n = 0, star = std::string::npos, retry = 0;
    while (n < name.size()) {
        if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == name[n])) { ++p; ++n; }
        else if (p < pattern.size() && pattern[p] == '*') { star = p++; retry = n; }
        else if (star != std::string::npos) { p = star + 1; n = ++retry; }
        else return false;
    }
    while (p < pattern.size() && pattern[p] == '*') ++p;
    return p == pattern.size();
}
bool matches(const std::vector<std::string>& patterns, const std::string& name) {
    return std::any_of(patterns.begin(), patterns.end(), [&](const auto& pattern) { return matches(pattern, name); });
}
void patterns(const std::vector<std::string>& values, std::size_t limit) {
    if (values.size() > limit) throw Error(Error::invalid, "invalid_download_filter", "too many file patterns");
    for (const auto& value : values) {
        relative(value, true);
        if (value.find_first_of("[]") != std::string::npos)
            throw Error(Error::invalid, "invalid_download_filter", "file patterns support only * and ? wildcards");
    }
}
void publish(const fs::path& from, const fs::path& to) {
#ifdef _WIN32
    const bool ok = MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_WRITE_THROUGH);
#elif defined(__APPLE__)
    const bool ok = ::renamex_np(from.c_str(), to.c_str(), RENAME_EXCL) == 0;
#else
    const bool ok = ::syscall(SYS_renameat2, AT_FDCWD, from.c_str(), AT_FDCWD, to.c_str(), 1) == 0;
#endif
    if (!ok) throw Error(Error::io, "download_publish_failed", "could not publish download without replacing an existing directory");
}
}

void Hub::Control::checkpoint() const {
    if (cancelled && cancelled()) throw Error(Error::cancelled, "download_cancelled", "download cancelled");
}
Hub::Hub(fs::path workspace, Network network)
    : workspace_(resolved_path(workspace)), fallback_(workspace_ / "models"), roots_{fallback_}, network_(std::move(network)) {
    static const std::regex endpoint(R"(https?://(?:\[[0-9a-fA-F:]+\]|[A-Za-z0-9.-]+)(?::[0-9]+)?)");
    for (auto* remote : {&network_.hf, &network_.ms}) {
        while (!remote->url.empty() && remote->url.back() == '/') remote->url.pop_back();
        if (!std::regex_match(remote->url, endpoint)) throw Error(Error::invalid, "invalid_hub_endpoint", "hub endpoint must be an HTTP(S) origin");
        if (std::any_of(remote->token.begin(), remote->token.end(), [](unsigned char ch) { return ch < 32 || ch == 127; }))
            throw Error(Error::invalid, "invalid_hub_token", "hub token contains invalid characters");
    }
    const auto config = workspace_ / "download-directory.json";
    if (fs::exists(config)) {
        try {
            const auto saved = read(config);
            for (const auto& item : saved.at("directories")) roots_.push_back(absolute(item.get<std::string>()));
            if (!saved.at("selected").is_null()) selected_ = absolute(saved.at("selected").get<std::string>());
            if (!selected_.empty() && std::find(roots_.begin(), roots_.end(), selected_) == roots_.end()) throw std::runtime_error("unregistered directory");
        } catch (...) { throw Error(Error::io, "download_settings_invalid", "saved download directories are invalid; existing file was preserved"); }
    }
}
Hub::Remote Hub::remote(Provider provider) const { return provider == Provider::hf ? network_.hf : network_.ms; }
Hub::Repo Hub::info(Source request, const Control& control, bool direct) const {
    source(request); control.checkpoint();
    const auto endpoint = remote(request.provider);
    const auto repo = hub::escape(request.repo, true), revision = hub::escape(request.revision);
    const auto path = request.provider == Provider::hf ? "/api/models/" + repo + "/revision/" + revision + "?blobs=true" :
        "/api/v1/models/" + repo + "/repo/files?Revision=" + revision + "&Recursive=True";
    try {
        const auto document = Json::parse(hub::get({endpoint, network_.ca, direct}, path, control));
        Repo result{request, endpoint.url + (request.provider == Provider::ms ? "/models/" : "/") + repo, {}};
        Json files;
        if (request.provider == Provider::hf) {
            result.source.revision = document.at("sha").get<std::string>();
            if ((result.source.revision.size() != 40 && result.source.revision.size() != 64) ||
                result.source.revision.find_first_not_of("0123456789abcdef") != std::string::npos) throw std::runtime_error("invalid commit");
            source(result.source);
            files = document.at("siblings");
        } else {
            if (document.value("Code", 200) != 200 || !document.value("Success", document.value("success", true)))
                throw Error(Error::network, "hub_metadata_failed", "repository rejected the file listing");
            const auto& data = document.contains("Data") ? document.at("Data") : document.at("data");
            files = data.contains("Files") ? data.at("Files") : data.at("files");
        }
        if (!files.is_array() || files.size() > 100000) throw std::runtime_error("invalid file listing");
        std::set<std::string> names;
        std::uint64_t total = 0;
        for (const auto& item : files) {
            control.checkpoint();
            if (item.value("Type", item.value("type", "blob")) == "tree") continue;
            File file;
            Json size;
            if (request.provider == Provider::hf) {
                file.name = item.at("rfilename").get<std::string>();
                size = item.contains("size") ? item.at("size") : item.at("lfs").at("size");
                if (item.contains("lfs") && !item["lfs"].is_null()) file.sha256 = item.at("lfs").value("sha256", "");
            } else {
                file.name = item.contains("Path") ? item.at("Path").get<std::string>() : item.at("path").get<std::string>();
                size = item.contains("Size") ? item.at("Size") : item.at("size");
                file.sha256 = item.value("Sha256", item.value("sha256", ""));
            }
            relative(file.name);
            if (!names.insert(file.name).second || !size.is_number_integer() || size < 0 || size > INT64_MAX) throw std::runtime_error("invalid file size or name");
            file.size = size.get<std::uint64_t>();
            if (file.size > INT64_MAX - total) throw std::runtime_error("repository size exceeds limit");
            total += file.size;
            if (!file.sha256.empty() && (file.sha256.size() != 64 || !std::all_of(file.sha256.begin(), file.sha256.end(), [](char ch) {
                return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
            }))) throw std::runtime_error("invalid checksum");
            if (request.provider == Provider::ms && file.sha256.empty()) throw std::runtime_error("missing checksum");
            result.files.push_back(std::move(file));
        }
        std::sort(result.files.begin(), result.files.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
        return result;
    } catch (const Json::exception&) { throw Error(Error::integrity, "invalid_hub_metadata", "repository returned invalid metadata"); }
    catch (const Error&) { throw; }
    catch (const std::runtime_error&) { throw Error(Error::integrity, "invalid_hub_metadata", "repository returned invalid file metadata"); }
}
Hub::Directory Hub::directory() const {
    std::lock_guard lock(mutex_);
    Directory result{selected_, selected_.empty() ? fallback_ : selected_, fallback_, {}};
    std::error_code error; const auto space = fs::space(result.actual, error);
    if (!error) result.space = space;
    return result;
}
Hub::Directory Hub::configure(const std::optional<fs::path>& path) {
    {
        std::lock_guard lock(mutex_);
        auto selected = path ? absolute(path_utf8(*path)) : fs::path{};
        if (!selected.empty() && !fs::is_directory(selected)) throw Error(Error::invalid, "invalid_download_directory", "download directory is unavailable");
        auto roots = roots_;
        if (!selected.empty() && std::find(roots.begin(), roots.end(), selected) == roots.end()) roots.push_back(selected);
        auto directories = Json::array(); for (const auto& root : roots) directories.push_back(path_utf8(root));
        try { write_json(workspace_ / "download-directory.json", {{"selected", selected.empty() ? Json(nullptr) : Json(path_utf8(selected))},
            {"directories", directories}, {"evict_prefix_cache", false}}); }
        catch (...) { throw Error(Error::io, "download_settings_save_failed", "could not save download directory"); }
        selected_ = std::move(selected); roots_ = std::move(roots);
    }
    return directory();
}
Hub::Result Hub::download(const Request& request, const Control& control) try {
    std::unique_lock transfer(transfer_, std::try_to_lock);
    if (!transfer) throw Error(Error::conflict, "download_busy", "another download is running");
    if (request.workers < 1 || request.workers > 16) throw Error(Error::invalid, "invalid_download_workers", "max_workers must be between 1 and 16");
    patterns(request.include, 4096); patterns(request.exclude, 64);
    auto name = request.destination;
    if (name.rfind("models/", 0) == 0) name.erase(0, 7);
    const auto destination = relative(name);
    fs::path root;
    {
        std::lock_guard lock(mutex_);
        root = request.directory ? absolute(path_utf8(*request.directory)) : selected_.empty() ? fallback_ : selected_;
        if (std::find(roots_.begin(), roots_.end(), root) == roots_.end()) throw Error(Error::invalid, "invalid_download_directory", "download directory has not been selected on this server");
    }
    if (root == fallback_) fs::create_directories(root);
    if (!fs::is_directory(root) || fs::is_symlink(root)) throw Error(Error::invalid, "invalid_download_directory", "download directory is unavailable");
    safe(root, destination);
    if (fs::exists(root / destination)) throw Error(Error::conflict, "download_destination_exists", "download destination already exists");
    auto repo = info(request.source, control, request.direct);
    repo.files.erase(std::remove_if(repo.files.begin(), repo.files.end(), [&](const auto& file) {
        return (!request.include.empty() && !matches(request.include, file.name)) || matches(request.exclude, file.name);
    }), repo.files.end());
    if (repo.files.empty()) throw Error(Error::invalid, "empty_download", "file selection is empty");
    std::uint64_t total = 0;
    auto entries = Json::array();
    std::set<std::string> names;
    for (const auto& file : repo.files) {
        auto parent = fs::u8path(file.name).parent_path();
        for (; !parent.empty(); parent = parent.parent_path())
            if (names.count(path_utf8(parent))) throw Error(Error::invalid, "invalid_download_path", "file and directory names overlap");
        names.insert(file.name);
        if (file.size > INT64_MAX - total) throw Error(Error::invalid, "download_too_large", "download exceeds the supported size");
        total += file.size; entries.push_back({{"name", file.name}, {"size", file.size}, {"sha256", file.sha256}});
    }
    const auto key = mfq::cache::block_hash_hex(mfq::cache::sha256(name));
    const auto stage = root / ".tyloinfer-downloads" / key, content = stage / "files", parts = stage / "parts";
    safe(root, content.lexically_relative(root)); safe(root, parts.lexically_relative(root));
    fs::create_directories(content); fs::create_directories(parts);
    const auto manifest = stage / "source.json";
    safe(stage, "source.json");
    const Json identity{{"endpoint", remote(repo.source.provider).url}, {"repo", repo.source.repo}, {"revision", repo.source.revision}, {"destination", name}, {"files", entries}};
    if (fs::exists(manifest)) {
        if (read(manifest) != identity) throw Error(Error::conflict, "download_changed", "staged download belongs to a different revision or file selection");
    } else write_json(manifest, identity);
    std::uint64_t remaining = std::max(total, request.expected_bytes.value_or(total));
    for (std::size_t index = 0; index < repo.files.size(); ++index) {
        const auto& file = repo.files[index];
        const auto path = content / fs::u8path(file.name);
        safe(stage, path.lexically_relative(stage));
        const auto part = parts / std::to_string(index);
        auto partial = part; partial += ".partial";
        auto meta = partial; meta += ".json";
        safe(stage, part.lexically_relative(stage));
        safe(stage, partial.lexically_relative(stage)); safe(stage, meta.lexically_relative(stage));
        fs::create_directories(path.parent_path());
        if (fs::is_regular_file(path)) remaining -= std::min(file.size, fs::file_size(path));
        else if (fs::is_regular_file(part)) remaining -= std::min(file.size, fs::file_size(part));
        else if (fs::is_regular_file(partial)) remaining -= std::min(file.size, fs::file_size(partial));
    }
    if (fs::space(root).available < remaining) throw Error(Error::io, "download_disk_full", "not enough free space for the selected files");
    std::mutex progress_mutex;
    std::vector<std::uint64_t> downloaded(repo.files.size());
    std::size_t completed = 0;
    auto reported = std::chrono::steady_clock::time_point{};
    auto report = [&](std::size_t index, std::uint64_t bytes, bool done) {
        std::lock_guard lock(progress_mutex);
        downloaded[index] = bytes; completed += done;
        const auto now = std::chrono::steady_clock::now();
        if (control.progress && (done || now - reported >= std::chrono::milliseconds(100))) {
            std::uint64_t sum = 0; for (auto value : downloaded) sum += value;
            control.progress(sum, total, completed); reported = now;
        }
    };
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false}; std::exception_ptr failure;
    Control active{[&] { return failed.load() || (control.cancelled && control.cancelled()); }, {}};
    auto work = [&] {
        try {
            for (;;) {
                active.checkpoint(); const auto index = next.fetch_add(1); if (index >= repo.files.size()) break;
                const auto& file = repo.files[index];
                const auto url = content_path(repo.source.provider, repo.source.repo, repo.source.revision, file.name);
                const auto ready = content / fs::u8path(file.name);
                hub::save({remote(repo.source.provider), network_.ca, request.direct}, url, file, fs::exists(ready) ? ready : parts / std::to_string(index), active,
                    [&](std::uint64_t bytes) { report(index, bytes, false); });
                report(index, file.size, true);
            }
        } catch (...) {
            std::lock_guard lock(progress_mutex);
            if (!failed.exchange(true)) failure = std::current_exception();
        }
    };
    std::vector<std::thread> workers;
    try { for (std::size_t i = 1; i < std::min(request.workers, repo.files.size()); ++i) workers.emplace_back(work); }
    catch (...) { failed = true; for (auto& worker : workers) worker.join(); throw; }
    work(); for (auto& worker : workers) worker.join();
    if (failure) std::rethrow_exception(failure);
    for (std::size_t index = 0; index < repo.files.size(); ++index) {
        control.checkpoint();
        const auto path = content / fs::u8path(repo.files[index].name);
        if (!fs::exists(path)) fs::rename(parts / std::to_string(index), path);
        else if (fs::exists(parts / std::to_string(index))) throw Error(Error::invalid, "invalid_download_path", "selected file names collide on this filesystem");
    }
    control.checkpoint(); safe(root, destination);
    if (fs::exists(root / destination)) throw Error(Error::conflict, "download_destination_exists", "download destination already exists");
    fs::create_directories((root / destination).parent_path());
    publish(content, root / destination);
    std::error_code ignored; fs::remove_all(stage, ignored);
    return {root / destination, repo.source, total, repo.files.size()};
} catch (const fs::filesystem_error&) {
    throw Error(Error::io, "download_io_failed", "download directory could not be read or written");
} catch (const Json::exception&) {
    throw Error(Error::integrity, "download_state_invalid", "saved download state is invalid; existing files were preserved");
}
namespace {
Hub::Control stopping_control(const std::atomic<bool>& stopping) {
    return Hub::Control{[&] { return stopping.load(); }, {}};
}
std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}
std::string text(const Json& value, const char* key) {
    return value.contains(key) && value.at(key).is_string() ? value.at(key).get<std::string>() : std::string();
}
std::int64_t number(const Json& value, const char* key) {
    if (!value.contains(key) || !value.at(key).is_number_integer()) return 0;
    return std::max<std::int64_t>(0, value.at(key).get<std::int64_t>());
}
double real(const Json& value, const char* key) {
    return value.contains(key) && value.at(key).is_number() ? value.at(key).get<double>() : 0;
}
std::vector<std::string> strings(const Json& value, const char* key) {
    std::vector<std::string> result;
    if (!value.contains(key) || !value.at(key).is_array()) return result;
    for (const auto& entry : value.at(key)) if (entry.is_string()) result.push_back(entry.get<std::string>());
    return result;
}
bool rejected(const Json& document) {
    return document.contains("success") && document.at("success").is_boolean() && !document.at("success").get<bool>();
}
bool has(const std::vector<std::string>& values, const std::string& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}
// Byte limits must not split a UTF-8 sequence; invalid text would break JSON serialization.
std::string clipped(std::string value, std::size_t limit) {
    if (value.size() <= limit) return value;
    value.resize(limit);
    while (!value.empty() && (static_cast<unsigned char>(value.back()) & 0xC0) == 0x80) value.pop_back();
    if (!value.empty() && (static_cast<unsigned char>(value.back()) & 0xC0) >= 0xC0) value.pop_back();
    return value;
}
bool tq_token(const std::string& name) {
    for (std::size_t at = name.find("tq"); at != std::string::npos; at = name.find("tq", at + 1)) {
        const bool before = at == 0 || name[at - 1] == '-' || name[at - 1] == '_' || name[at - 1] == '/';
        const auto after = at + 2;
        const bool tail = after == name.size() || name[after] == '-' || name[after] == '_' || (name[after] >= '0' && name[after] <= '9');
        if (before && tail) return true;
    }
    return false;
}
// Weight formats a hub entry advertises, mirroring the Python search classifier.
std::vector<std::string> formats(const std::string& repo, const std::vector<std::string>& tags) {
    std::set<std::string> labels;
    for (const auto& tag : tags) labels.insert(lowercase(tag.substr(tag.rfind(':') + 1)));
    const auto name = lowercase(repo);
    std::vector<std::string> result;
    if (labels.count("gguf") || name.find("gguf") != std::string::npos) result.push_back("gguf");
    if (labels.count("mlx") || name.find("mlx") != std::string::npos) result.push_back("mlx");
    if (!has(result, "gguf") && (labels.count("tq") || tq_token(name))) result.push_back("tq");
    if (result.empty() && labels.count("safetensors")) result.push_back("hf");
    return result;
}
std::vector<std::string> modalities(const std::vector<std::string>& tags) {
    std::string joined;
    for (const auto& tag : tags) joined += lowercase(tag) + ' ';
    std::vector<std::string> result;
    for (const auto* name : {"text", "image", "video", "audio"}) if (joined.find(name) != std::string::npos) result.push_back(name);
    return result;
}
const std::vector<std::string> media_pipelines{"text-to-image", "image-to-image", "text-to-video", "image-to-video", "unconditional-image-generation"};
bool media_model(const std::string& pipeline, const std::vector<std::string>& tags) {
    if (!pipeline.empty()) return has(media_pipelines, pipeline);
    for (const auto& tag : tags) if (has(media_pipelines, lowercase(tag.substr(tag.rfind(':') + 1)))) return true;
    return false;
}
// shortcut: word runs are ASCII-only, so non-ASCII names fall back to the weaker relevance tiers.
std::vector<std::string> words(const std::string& value) {
    std::vector<std::string> result; std::string current; int kind = 0;
    for (const auto raw : lowercase(value)) {
        const auto ch = static_cast<unsigned char>(raw);
        const int next = std::isalpha(ch) ? 1 : std::isdigit(ch) ? 2 : 0;
        if (next && next == kind) current += raw;
        else { if (!current.empty()) result.push_back(current); current = next ? std::string(1, raw) : std::string(); kind = next; }
    }
    if (!current.empty()) result.push_back(current);
    return result;
}
std::string joined(const std::vector<std::string>& values) {
    std::string result;
    for (const auto& value : values) result += value;
    return result;
}
std::string strip_formats(const std::string& name) {
    const auto lower = lowercase(name);
    for (std::size_t at = 1; at < lower.size(); ++at) {
        const auto before = lower[at - 1];
        if (before != '-' && before != '_' && before != '.') continue;
        for (const auto* suffix : {"gguf", "mlx", "tq"}) {
            const auto length = std::strlen(suffix);
            if (lower.compare(at, length, suffix) != 0) continue;
            const auto after = at + length;
            if (after == lower.size() || lower[after] == '-' || lower[after] == '_') return name.substr(0, at - 1);
        }
    }
    return name;
}
bool contains_all(const std::vector<std::string>& words, const std::vector<std::string>& requested) {
    return std::all_of(requested.begin(), requested.end(), [&](const auto& word) { return has(words, word); });
}
std::pair<int, int> relevance(const Hub::Summary& item, const std::string& query) {
    const auto name = item.repo.substr(item.repo.find('/') + 1);
    const auto found = words(name), requested = words(query);
    const auto joined_name = joined(found), requested_name = joined(requested);
    const auto base = joined(words(strip_formats(name)));
    const int tier = joined_name == requested_name ? 6 : base == requested_name ? 5 : joined_name.rfind(requested_name, 0) == 0 ? 4 :
        contains_all(found, requested) ? 3 : joined_name.find(requested_name) != std::string::npos ? 2 :
        contains_all(words(item.repo), requested) ? 1 : 0;
    int matched = 0;
    for (const auto& word : requested) if (has(found, word)) ++matched;
    return {tier, matched};
}
int order(std::int64_t left, std::int64_t right) { return left < right ? -1 : left > right ? 1 : 0; }
int order(double left, double right) { return left < right ? -1 : left > right ? 1 : 0; }
// Lexicographic comparison of the Python search key; descending searches pick the greatest entry.
int rank(const Hub::Summary& left, const Hub::Summary& right, const std::string& sort, const std::string& query) {
    const auto by_repo = lowercase(left.repo).compare(lowercase(right.repo));
    if (sort == "downloads") { if (const auto value = order(left.downloads, right.downloads)) return value; return by_repo; }
    if (sort == "likes") { if (const auto value = order(left.likes, right.likes)) return value; return by_repo; }
    if (sort == "updated") {
        // shortcut: each hub emits fixed-width timestamps, so lexical order matches chronological order.
        if (const auto value = left.updated.compare(right.updated)) return value;
        return by_repo;
    }
    if (query.empty()) {
        if (const auto trending = order(left.trending, right.trending)) return trending;
        if (const auto value = order(left.downloads, right.downloads)) return value;
        return by_repo;
    }
    const auto left_rank = relevance(left, query), right_rank = relevance(right, query);
    if (left_rank.first != right_rank.first) return left_rank.first < right_rank.first ? -1 : 1;
    if (left_rank.second != right_rank.second) return left_rank.second < right_rank.second ? -1 : 1;
    if (const auto downloads = order(left.downloads, right.downloads)) return downloads;
    const auto left_name = left.repo.substr(left.repo.find('/') + 1), right_name = right.repo.substr(right.repo.find('/') + 1);
    if (left_name.size() != right_name.size()) return left_name.size() > right_name.size() ? -1 : 1;
    return by_repo;
}
bool official_repo(const std::string& repo) {
    const auto slash = repo.find('/');
    if (slash == std::string::npos || !std::regex_match(repo, repository)) return false;
    if (lowercase(repo.substr(0, slash)) != "tylogi") return false;
    std::vector<std::string> parts; std::string current;
    for (const auto ch : lowercase(repo.substr(slash + 1))) {
        if (ch == '-' || ch == '_') { parts.push_back(current); current.clear(); } else current += ch;
    }
    parts.push_back(current);
    return has(parts, "tq");
}
bool parse_summary(const Json& item, Hub::Provider provider, Hub::Summary& summary) {
    if (!item.is_object()) return false;
    summary = Hub::Summary{};
    summary.provider = provider;
    summary.repo = text(item, "id");
    if (summary.repo.size() > 255 || !std::regex_match(summary.repo, repository)) return false;
    const auto tags = strings(item, "tags");
    const auto pipeline = text(item, "pipeline_tag").empty() ? text(item, "pipeline") : text(item, "pipeline_tag");
    summary.formats = formats(summary.repo, tags);
    summary.modalities = modalities(tags);
    summary.media = media_model(pipeline, tags);
    summary.author = summary.repo.substr(0, summary.repo.find('/'));
    summary.description = clipped(text(item, "description"), 2048);
    summary.downloads = number(item, "downloads");
    summary.likes = number(item, "likes");
    summary.bytes = provider == Hub::Provider::ms ? number(item, "storage_size") : 0;
    summary.updated = text(item, provider == Hub::Provider::hf ? "lastModified" : "last_modified");
    summary.trending = real(item, provider == Hub::Provider::hf ? "trendingScore" : "trending_score");
    return true;
}
// The next-page URL from a Link header, reduced to a path so the configured endpoint stays authoritative.
std::string link_path(const std::string& link) {
    std::size_t at = 0;
    while ((at = link.find('<', at)) != std::string::npos) {
        const auto end = link.find('>', at);
        if (end == std::string::npos) break;
        const auto boundary = link.find('<', end);
        const auto tail = link.substr(end + 1, (boundary == std::string::npos ? link.size() : boundary) - end - 1);
        if (tail.find("rel=\"next\"") != std::string::npos) {
            const auto url = link.substr(at + 1, end - at - 1);
            const auto scheme = url.find("://");
            const auto slash = url.find('/', scheme == std::string::npos ? 0 : scheme + 3);
            return slash == std::string::npos ? std::string() : url.substr(slash);
        }
        at = end;
    }
    return {};
}
std::string key(Hub::Provider provider, const std::string& repo) {
    return std::string(provider == Hub::Provider::hf ? "huggingface:" : "modelscope:") + repo;
}
}

Hub::~Hub() {
    stopping_ = true;
    if (refresher_.joinable()) refresher_.join();
}

Hub::Page Hub::search(Provider provider, const std::string& query, const std::string& format,
    const std::string& sort, const std::string& direction, std::size_t limit, const std::string& position) const {
    if (limit < 1 || limit > 100) throw Error(Error::invalid, "invalid_hub_search", "search limit is outside its range");
    const auto endpoint = remote(provider);
    const auto control = stopping_control(stopping_);
    const auto order_name = sort == "updated" ? (provider == Provider::hf ? "lastModified" : "last_modified") :
        sort == "relevance" ? (provider == Provider::hf ? (query.empty() ? "trendingScore" : "downloads") : "downloads") : sort;
    const auto search = format == "tq" ? (query.empty() ? "TQ" : query + " TQ") : query;
    Json values = Json::array();
    std::string next;
    if (provider == Provider::hf) {
        auto path = position;
        if (path.empty()) {
            path = "/api/models?search=" + hub::escape(search) + "&limit=" + std::to_string(limit) +
                "&direction=" + (direction == "asc" ? "1" : "-1") + "&sort=" + order_name;
            for (const auto* field : {"downloads", "likes", "lastModified", "tags", "trendingScore", "pipeline_tag"}) path += "&expand=" + std::string(field);
            if (format == "gguf" || format == "mlx") path += "&filter=" + format;
        }
        std::string link;
        const auto document = Json::parse(hub::get({endpoint, network_.ca, false}, path, control, &link));
        if (!document.is_array()) throw Error(Error::integrity, "invalid_hub_metadata", "model source returned an invalid search response");
        values = document;
        next = link_path(link);
    } else {
        const auto size = std::min<std::size_t>(limit, 50);
        std::size_t page = position.empty() ? 1 : static_cast<std::size_t>(std::stoul(position));
        for (;;) {
            auto path = "/openapi/v1/models?search=" + hub::escape(search) + "&page_size=" + std::to_string(size) +
                "&page_number=" + std::to_string(page) + "&sort=" + order_name;
            if (format == "gguf" || format == "mlx") path += "&filter.library=" + format;
            const auto document = Json::parse(hub::get({endpoint, network_.ca, false}, path, control));
            if (!document.is_object() || rejected(document))
                throw Error(Error::network, "hub_search_failed", "model source rejected the search");
            if (!document.contains("data") || !document.at("data").is_object() ||
                !document.at("data").contains("models") || !document.at("data").at("models").is_array())
                throw Error(Error::integrity, "invalid_hub_metadata", "model source returned an invalid search response");
            const auto& models = document.at("data").at("models");
            values.insert(values.end(), models.begin(), models.end());
            if (models.size() == size) next = std::to_string(page + 1);
            ++page;
            if (models.size() < size || values.size() >= limit) break;
        }
    }
    Page result;
    for (const auto& item : values) {
        Summary summary;
        if (parse_summary(item, provider, summary)) result.items.push_back(std::move(summary));
    }
    result.cursor = next;
    return result;
}

bool Hub::Search::matches(const Search& other) const {
    return provider == other.provider && query == other.query && sort == other.sort && direction == other.direction &&
        limit == other.limit && formats == other.formats;
}

Hub::Page Hub::browse(Provider provider, const std::string& query, const std::vector<std::string>& formats,
    const std::string& sort, const std::string& direction, std::size_t limit, const std::string& cursor) {
    Search wanted{provider, query, sort, direction, formats, limit};
    for (const auto& format : formats) wanted.streams.push_back(Stream{format, {}, false, {}});
    Search state = wanted;
    {
        // Only the cursor table is shared; provider paging below must not hold the lock.
        std::lock_guard lock(search_);
        const auto now = std::chrono::steady_clock::now();
        for (auto entry = cursors_.begin(); entry != cursors_.end();)
            entry = entry->second.expires < now ? cursors_.erase(entry) : std::next(entry);
        while (cursors_.size() > 128) cursors_.erase(cursors_.begin());
        if (!cursor.empty()) {
            auto found = cursors_.find(cursor);
            if (found == cursors_.end() || !found->second.matches(wanted))
                throw Error(Error::invalid, "search_cursor_expired", "The search has expired or its filters changed. Start a new search.");
            if (found->second.consumed) return {found->second.page, found->second.next};
            state = found->second;
        }
    }
    std::vector<Summary> items;
    while (items.size() < state.limit) {
        for (auto& stream : state.streams) {
            while (stream.buffer.empty() && !stream.exhausted) {
                auto page = search(state.provider, state.query, stream.format, state.sort, state.direction, 50, stream.position);
                if (!stream.position.empty() && page.cursor == stream.position)
                    throw Error(Error::network, "hub_search_failed", "model source returned a repeated page cursor");
                stream.position = page.cursor;
                stream.exhausted = page.cursor.empty();
                for (auto& item : page.items) {
                    if (!has(item.formats, stream.format) || item.media || state.seen.count(item.repo)) continue;
                    stream.buffer.push_back(std::move(item));
                }
                if (state.sort == "relevance")
                    std::sort(stream.buffer.begin(), stream.buffer.end(), [&](const auto& left, const auto& right) {
                        return rank(left, right, state.sort, state.query) > 0;
                    });
            }
        }
        Stream* best = nullptr;
        for (auto& stream : state.streams) {
            if (stream.buffer.empty()) continue;
            if (!best) { best = &stream; continue; }
            const auto value = rank(stream.buffer.front(), best->buffer.front(), state.sort, state.query);
            if ((state.direction == "desc" && value > 0) || (state.direction == "asc" && value < 0)) best = &stream;
        }
        if (!best) break;
        auto item = best->buffer.front();
        best->buffer.erase(best->buffer.begin());
        if (state.seen.insert(item.repo).second) items.push_back(std::move(item));
    }
    state.page = items;
    state.next = std::any_of(state.streams.begin(), state.streams.end(), [](const auto& stream) {
        return !stream.buffer.empty() || !stream.exhausted;
    }) ? normalize_uuid({}) : std::string();
    state.consumed = true;
    state.expires = std::chrono::steady_clock::now() + std::chrono::minutes(15);
    auto resumed = state;
    resumed.consumed = false; resumed.page.clear(); resumed.next.clear();
    std::lock_guard lock(search_);
    if (!cursor.empty()) cursors_[cursor] = state;
    if (!state.next.empty()) cursors_[state.next] = resumed;
    return {items, state.next};
}

std::string Hub::readme(Provider provider, const std::string& repo, const std::string& revision) const {
    Source candidate{provider, repo, revision};
    source(candidate);
    const auto path = provider == Provider::hf ?
        "/" + hub::escape(candidate.repo, true) + "/resolve/" + hub::escape(candidate.revision) + "/README.md" :
        "/api/v1/models/" + hub::escape(candidate.repo, true) + "/repo?Revision=" + hub::escape(candidate.revision) + "&FilePath=README.md";
    try { return hub::get({remote(provider), network_.ca, false}, path, stopping_control(stopping_)); }
    catch (const Error& error) { if (error.kind == Error::missing) return {}; throw; }
}

std::vector<Hub::Summary> Hub::discover(Provider provider) const {
    const auto endpoint = remote(provider);
    const auto control = stopping_control(stopping_);
    std::vector<Summary> result;
    if (provider == Provider::hf) {
        std::string path = "/api/models?author=Tylogi&limit=100";
        for (std::size_t page = 0; !path.empty(); ++page) {
            if (page >= 150) throw Error(Error::network, "hub_search_failed", "official model discovery did not complete");
            std::string link;
            const auto document = Json::parse(hub::get({endpoint, network_.ca, false}, path, control, &link));
            if (!document.is_array()) throw Error(Error::integrity, "invalid_hub_metadata", "official model discovery returned an invalid response");
            for (const auto& item : document) {
                if (item.contains("private") && item.at("private").is_boolean() && item.at("private").get<bool>()) continue;
                Summary summary;
                if (parse_summary(item, provider, summary) && official_repo(summary.repo)) result.push_back(std::move(summary));
            }
            path = link_path(link);
        }
        return result;
    }
    for (std::size_t page = 1; page <= 150; ++page) {
        const auto document = Json::parse(hub::get({endpoint, network_.ca, false},
            "/openapi/v1/models?owner=Tylogi&page_size=20&page_number=" + std::to_string(page), control));
        if (!document.is_object() || rejected(document))
            throw Error(Error::network, "hub_search_failed", "official model discovery failed");
        if (!document.contains("data") || !document.at("data").is_object() ||
            !document.at("data").contains("models") || !document.at("data").at("models").is_array())
            throw Error(Error::integrity, "invalid_hub_metadata", "official model discovery returned an invalid response");
        const auto& models = document.at("data").at("models");
        for (const auto& item : models) {
            Summary summary;
            if (parse_summary(item, provider, summary) && official_repo(summary.repo)) result.push_back(std::move(summary));
        }
        if (models.size() < 20) break;
    }
    return result;
}

void Hub::refresh_official() {
    Snapshot next;
    {
        std::lock_guard lock(official_);
        next = snapshot_;
    }
    for (const auto provider : {Provider::hf, Provider::ms}) {
        std::vector<Summary> found;
        try { found = discover(provider); } catch (const Error&) { continue; }
        // A successful discovery replaces this provider's entries; a failed one keeps the last usable snapshot.
        const auto prefix = provider == Provider::hf ? "huggingface:" : "modelscope:";
        for (auto entry = next.models.begin(); entry != next.models.end();)
            entry = entry->first.rfind(prefix, 0) == 0 ? next.models.erase(entry) : std::next(entry);
        for (const auto& summary : found) {
            OfficialModel model;
            model.summary = summary;
            model.source = {provider, summary.repo, {}};
            try {
                auto repo = info(model.source, stopping_control(stopping_));
                model.source.revision = repo.source.revision;
                model.files = std::move(repo.files);
            } catch (const Error&) {}
            next.models[key(provider, summary.repo)] = std::move(model);
        }
    }
    if (stopping_.load()) return;
    next.refreshed = std::chrono::steady_clock::now();
    std::lock_guard lock(official_);
    snapshot_ = std::move(next);
}

std::vector<Hub::OfficialModel> Hub::official(bool refresh) {
    bool stale;
    {
        std::lock_guard lock(official_);
        stale = snapshot_.refreshed == std::chrono::steady_clock::time_point{} ||
            std::chrono::steady_clock::now() - snapshot_.refreshed >= std::chrono::minutes(15);
    }
    if ((refresh || stale) && !refreshing_.exchange(true)) {
        if (refresher_.joinable()) refresher_.join();
        refresher_ = std::thread([this] {
            try { refresh_official(); } catch (const Error&) {}
            refreshing_ = false;
        });
    }
    std::lock_guard lock(official_);
    std::vector<OfficialModel> result;
    for (const auto& [name, model] : snapshot_.models) result.push_back(model);
    return result;
}

bool Hub::refreshing() const {
    std::lock_guard lock(official_);
    return refreshing_;
}

std::vector<std::pair<std::string, std::uint64_t>> Hub::reusable(const Source& reference, const std::string& destination,
    const std::optional<fs::path>& directory) const {
    auto name = destination;
    if (name.rfind("models/", 0) == 0) name.erase(0, 7);
    const auto target = relative(name);
    fs::path root;
    {
        std::lock_guard lock(mutex_);
        root = directory ? absolute(path_utf8(*directory)) : selected_.empty() ? fallback_ : selected_;
        if (std::find(roots_.begin(), roots_.end(), root) == roots_.end())
            throw Error(Error::invalid, "invalid_download_directory", "download directory has not been selected on this server");
    }
    safe(root, target);
    const auto base = root / target;
    std::vector<std::pair<std::string, std::uint64_t>> result;
    for (const auto& file : info(reference).files) {
        const auto extension = fs::u8path(file.name).extension();
        if (file.size == 0 || (extension != ".tq" && extension != ".mfq")) continue;
        std::error_code error;
        const auto size = fs::is_regular_file(base / fs::u8path(file.name), error) ? fs::file_size(base / fs::u8path(file.name), error) : 0;
        if (!error && size == file.size) result.emplace_back(file.name, file.size);
    }
    return result;
}

Hub::Route Hub::probe(Provider provider, const Source& reference, const std::string& file) const {
    auto candidate = reference;
    source(candidate);
    const auto path = content_path(provider, candidate.repo, candidate.revision, file);
    const auto endpoint = remote(provider);
    auto measure = [&](bool direct) {
        const auto started = std::chrono::steady_clock::now();
        const auto bytes = hub::range({endpoint, network_.ca, direct}, path, 1048576, stopping_control(stopping_));
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return bytes ? static_cast<double>(bytes) / std::max(0.001, elapsed) : 0.0;
    };
    const auto direct = measure(true);
    // shortcut: the proxy route is only measured when the environment configures a proxy at all.
    if (!std::getenv("http_proxy") && !std::getenv("HTTP_PROXY") && !std::getenv("https_proxy") &&
        !std::getenv("HTTPS_PROXY") && !std::getenv("all_proxy") && !std::getenv("ALL_PROXY")) return {direct, false};
    const auto proxied = measure(false);
    return proxied > direct ? Route{proxied, true} : Route{direct, false};
}
}

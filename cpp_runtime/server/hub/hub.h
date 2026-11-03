#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mfq::server {
class Hub {
public:
    struct Error : std::runtime_error {
        enum Kind { invalid, missing, conflict, network, integrity, io, cancelled, unavailable } kind;
        const char* code;
        Error(Kind kind, const char* code, const std::string& message) : std::runtime_error(message), kind(kind), code(code) {}
    };
    enum class Provider { hf, ms };
    struct Source { Provider provider; std::string repo, revision; };
    struct File { std::string name, sha256; std::uint64_t size = 0; };
    struct Repo { Source source; std::string url; std::vector<File> files; };
    struct Remote { std::string url, token; };
    struct Network {
        Remote hf{"https://huggingface.co", {}}, ms{"https://modelscope.cn", {}};
        std::filesystem::path ca;
    };
    struct Control {
        std::function<bool()> cancelled;
        std::function<void(std::uint64_t, std::uint64_t, std::size_t)> progress;
        void checkpoint() const;
    };
    struct Request {
        Source source;
        std::string destination;
        std::optional<std::filesystem::path> directory;
        std::vector<std::string> include, exclude;
        std::size_t workers = 8;
        bool direct = false;
        std::optional<std::uint64_t> expected_bytes;
    };
    struct Result { std::filesystem::path path; Source source; std::uint64_t bytes; std::size_t files; };
    struct Directory { std::filesystem::path selected, actual, fallback; std::optional<std::filesystem::space_info> space; };
    struct Summary {
        Provider provider = Provider::hf;
        std::string repo, author, description, updated, pipeline;
        std::int64_t downloads = 0, likes = 0, bytes = 0;
        double trending = 0;
        std::vector<std::string> formats, modalities;
        bool media = false;
    };
    // `cursor` is the continuation token for the merged browse, empty when the search is finished.
    struct Page { std::vector<Summary> items; std::string cursor; };
    struct OfficialModel { Summary summary; Source source; std::vector<File> files; };
    struct Route { double bytes_per_second = 0; bool proxy = false; };

    Hub(std::filesystem::path workspace, Network network);
    ~Hub();
    Repo info(Source source, const Control& control = {}, bool direct = false) const;
    Result download(const Request& request, const Control& control = {});
    Directory directory() const;
    Directory configure(const std::optional<std::filesystem::path>& path);
    // One provider page in hub order. `position` is a provider continuation: a path for Hugging Face, a page number for ModelScope.
    Page search(Provider provider, const std::string& query, const std::string& format,
        const std::string& sort, const std::string& direction, std::size_t limit,
        const std::string& position = {}) const;
    // Merged multi-format search with server-side cursors, matching the Python hub browser.
    Page browse(Provider provider, const std::string& query, const std::vector<std::string>& formats,
        const std::string& sort, const std::string& direction, std::size_t limit, const std::string& cursor);
    // Empty result means the repository has no model card.
    std::string readme(Provider provider, const std::string& repo, const std::string& revision) const;
    // Cached official catalog; a stale or forced request starts a background refresh and returns the last snapshot.
    std::vector<OfficialModel> official(bool refresh);
    bool refreshing() const;
    std::vector<std::pair<std::string, std::uint64_t>> reusable(const Source& source, const std::string& destination,
        const std::optional<std::filesystem::path>& directory) const;
    Route probe(Provider provider, const Source& source, const std::string& file) const;
private:
    struct Stream { std::string format, position; bool exhausted = false; std::vector<Summary> buffer; };
    struct Search {
        Provider provider = Provider::hf;
        std::string query, sort, direction;
        std::vector<std::string> formats;
        std::size_t limit = 0;
        std::vector<Stream> streams;
        std::set<std::string> seen;
        std::vector<Summary> page;
        std::string next;
        bool consumed = false;
        std::chrono::steady_clock::time_point expires{};
        bool matches(const Search& other) const;
    };
    struct Snapshot { std::map<std::string, OfficialModel> models; std::chrono::steady_clock::time_point refreshed{}; };
    std::vector<Summary> discover(Provider provider) const;
    void refresh_official();
    std::filesystem::path workspace_, fallback_, selected_;
    std::vector<std::filesystem::path> roots_;
    Network network_;
    mutable std::mutex mutex_;
    std::mutex transfer_;
    std::mutex search_;
    std::map<std::string, Search> cursors_;
    mutable std::mutex official_;
    Snapshot snapshot_;
    std::thread refresher_;
    std::atomic<bool> refreshing_{false};
    std::atomic<bool> stopping_{false};
    Remote remote(Provider provider) const;
};
}

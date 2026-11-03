#include "deletion.h"
#include "errors.h"
#include "instance.h"
#include "mfq_paged_prefix_cache.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <system_error>

#ifdef _WIN32
#include <sys/stat.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace mfq::server {
namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

[[noreturn]] void unsafe() {
    throw ModelError(ModelError::Kind::conflict, "model_delete_unsafe", "model files cannot be safely deleted");
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool within(const fs::path& root, const fs::path& path) {
    return std::mismatch(root.begin(), root.end(), path.begin(), path.end()).first == root.end();
}

struct ShardName { std::string base; std::size_t index = 0, count = 0; };

// MFQ shards use a fixed five-digit form; GGUF additionally splits raw parts.
std::optional<ShardName> parse_shard(const std::string& name) {
    static const std::regex pattern(R"(^(.+)-([0-9]{5})-of-([0-9]{5})(\.tq|\.mfq)$)", std::regex::icase);
    std::smatch match;
    if (!std::regex_match(name, match, pattern)) return {};
    const ShardName parsed{match[1].str() + lowercase(match[4].str()), std::stoul(match[2]), std::stoul(match[3])};
    return parsed.index >= 1 && parsed.index <= parsed.count ? parsed : std::optional<ShardName>{};
}

std::optional<ShardName> parse_gguf(const std::string& name) {
    static const std::regex part(R"(^(.+\.gguf)\.part-([0-9]{1,5})-of-([0-9]{1,5})$)", std::regex::icase);
    static const std::regex shard(R"(^(.+)-([0-9]{5})-of-([0-9]{5})\.gguf$)", std::regex::icase);
    std::smatch match;
    const bool raw = std::regex_match(name, match, part);
    if (!raw && !std::regex_match(name, match, shard)) return {};
    const ShardName parsed{raw ? match[1].str() : match[1].str() + ".gguf", std::stoul(match[2]), std::stoul(match[3])};
    return parsed.index >= 1 && parsed.index <= parsed.count ? parsed : std::optional<ShardName>{};
}

/** Every sibling shard that shares this artifact's base name and shard count. */
std::vector<fs::path> shard_group(const fs::path& source, bool gguf) {
    const auto name = path_utf8(source.filename());
    const auto parsed = gguf ? parse_gguf(name) : parse_shard(name);
    if (!parsed) return {source};
    std::vector<fs::path> files;
    std::error_code error;
    auto it = fs::directory_iterator(source.parent_path(), fs::directory_options::skip_permission_denied, error);
    for (; it != fs::directory_iterator(); it.increment(error)) {
        std::error_code item;
        if (!it->is_regular_file(item)) continue;
        const auto child = path_utf8(it->path().filename());
        const auto sibling = gguf ? parse_gguf(child) : parse_shard(child);
        if (sibling && sibling->base == parsed->base && sibling->count == parsed->count) files.push_back(it->path());
    }
    return files;
}

bool reserved_name(const std::string& value) {
    const auto relative = fs::u8path(value);
    if (relative.is_absolute() || relative.empty() || lowercase(relative.extension()) != ".safetensors") return true;
    return std::any_of(relative.begin(), relative.end(), [](const fs::path& part) { return part == ".."; });
}

std::vector<fs::path> hf_files(const fs::path& source) {
    std::vector<fs::path> files{source / "config.json"};
    std::error_code error;
    auto it = fs::directory_iterator(source, fs::directory_options::skip_permission_denied, error);
    for (; it != fs::directory_iterator(); it.increment(error)) {
        std::error_code item;
        if (it->is_regular_file(item) && lowercase(path_utf8(it->path().extension())) == ".safetensors") files.push_back(it->path());
    }
    const auto index = source / "model.safetensors.index.json";
    std::error_code index_error;
    if (!fs::is_regular_file(index, index_error)) return files;
    json document;
    try {
        std::ifstream input(index);
        document = json::parse(input);
    } catch (const std::exception&) { unsafe(); }
    if (!document.is_object()) unsafe();
    const auto weight_map = document.value("weight_map", json::object());
    if (!weight_map.is_object()) unsafe();
    for (const auto& entry : weight_map) {
        if (!entry.is_string() || reserved_name(entry.get<std::string>())) unsafe();
        const auto path = source / fs::u8path(entry.get<std::string>());
        std::error_code ignored;
        const auto status = fs::symlink_status(path, ignored);
        if (!ignored && (fs::is_regular_file(status) || fs::is_symlink(status))) files.push_back(path);
    }
    files.push_back(index);
    return files;
}

struct Identity {
    std::uint64_t device = 0, inode = 0, size = 0, modified_ns = 0, created_ns = 0;
    bool regular = false, symlink = false;
};

bool identity(const fs::path& path, Identity& result) {
#ifdef _WIN32
    struct _stat64 info;
    if (::_wstat64(path.c_str(), &info) != 0) return false;
    result.regular = (info.st_mode & _S_IFMT) == _S_IFREG;
    result.symlink = (info.st_mode & _S_IFMT) == _S_IFLNK;
    result.device = static_cast<std::uint64_t>(info.st_dev);
    result.inode = static_cast<std::uint64_t>(info.st_ino);
    result.size = static_cast<std::uint64_t>(info.st_size);
    result.modified_ns = static_cast<std::uint64_t>(info.st_mtime) * 1000000000ULL;
    result.created_ns = static_cast<std::uint64_t>(info.st_ctime) * 1000000000ULL;
#else
    struct stat info;
    if (::lstat(path_utf8(path).c_str(), &info) != 0) return false;
    result.regular = S_ISREG(info.st_mode);
    result.symlink = S_ISLNK(info.st_mode);
    result.device = static_cast<std::uint64_t>(info.st_dev);
    result.inode = static_cast<std::uint64_t>(info.st_ino);
    result.size = static_cast<std::uint64_t>(info.st_size);
    result.modified_ns = static_cast<std::uint64_t>(info.st_mtim.tv_sec) * 1000000000ULL +
        static_cast<std::uint64_t>(info.st_mtim.tv_nsec);
    result.created_ns = static_cast<std::uint64_t>(info.st_ctim.tv_sec) * 1000000000ULL +
        static_cast<std::uint64_t>(info.st_ctim.tv_nsec);
#endif
    return true;
}

} // namespace

DeletionPlan model_deletion_plan(const ModelCatalog::Artifact& artifact) {
    const auto& source = artifact.path;
    const auto root = artifact.format == "hf" ? source : source.parent_path();
    std::vector<fs::path> candidates;
    if (artifact.format == "hf") candidates = hf_files(source);
    else candidates = shard_group(source, artifact.format == "gguf");
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    std::error_code error;
    const auto root_path = fs::weakly_canonical(root, error);
    if (error) unsafe();
    json rows = json::array();
    DeletionPlan plan;
    plan.model_id = artifact.id;
    plan.name = artifact.name;
    for (const auto& path : candidates) {
        const auto parent = fs::weakly_canonical(path.parent_path(), error);
        if (error || !within(root_path, parent)) unsafe();
        Identity info;
        if (!identity(path, info) || !(info.regular || info.symlink)) unsafe();
        rows.push_back({path_utf8(path), info.device, info.inode, info.size, info.modified_ns, info.created_ns});
        plan.total_bytes += info.size;
        plan.files.push_back(path);
    }
    if (rows.empty())
        throw ModelError(ModelError::Kind::conflict, "model_delete_changed", "model files changed; refresh the list and try again");
    plan.token = mfq::cache::block_hash_hex(mfq::cache::sha256(rows.dump()));
    return plan;
}

json deletion_plan_json(const DeletionPlan& plan) {
    auto files = json::array();
    for (const auto& path : plan.files) files.push_back(path_utf8(path));
    return {{"model_id", plan.model_id}, {"name", plan.name}, {"files", std::move(files)},
        {"total_bytes", plan.total_bytes}, {"token", plan.token}};
}

json delete_model_files(const ModelCatalog::Artifact& artifact, const std::string& token) {
    const auto plan = model_deletion_plan(artifact);
    if (plan.token != token)
        throw ModelError(ModelError::Kind::conflict, "model_delete_changed", "model files changed; refresh the list and try again");
    // Index and config go last so a partial checkpoint is never loadable.
    static const std::map<std::string, int> order{{"model.safetensors.index.json", 1}, {"config.json", 2}};
    auto files = plan.files;
    std::stable_sort(files.begin(), files.end(), [](const fs::path& left, const fs::path& right) {
        const auto key = [](const fs::path& path) {
            const auto found = order.find(path_utf8(path.filename()));
            return found == order.end() ? 0 : found->second;
        };
        return key(left) < key(right);
    });
    for (const auto& path : files) {
        std::error_code error;
        fs::remove(path, error);
        if (error) throw ModelError(ModelError::Kind::conflict, "model_delete_failed",
            "could not delete all model files; check permissions and refresh the list");
    }
    return {{"files", static_cast<std::int64_t>(plan.files.size())}, {"total_bytes", plan.total_bytes}};
}

} // namespace mfq::server

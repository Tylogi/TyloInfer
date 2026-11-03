#include "prefix.h"
#include "api/errors.h"
#include "runtime_config.h"
#include "text_processor.h"
#include "tyloinfer/model_source.h"
#include "mfq_paged_prefix_cache.h"
#include "models/catalog.h"
#include "models/instances.h"
#include "models/policies.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

namespace mfq::server {
namespace {
using json = nlohmann::json;
namespace fs = std::filesystem;

constexpr std::size_t block_header = 156;   // MFQKVB1: magic8 u32 compat32 digest32 parent32 size32 tokens32 payload64 pad32
constexpr std::size_t text_header = 108;    // MFQTXT1: magic8 compat32 identity32 tokens32 checksum32
constexpr const char* genesis = "0000000000000000000000000000000000000000000000000000000000000000";

bool hex_of(std::string_view value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isxdigit(c) != 0; });
}

std::string hex(const std::uint8_t* data, std::size_t size) {
    static const char* digits = "0123456789abcdef";
    std::string text(size * 2, '0');
    for (std::size_t index = 0; index < size; ++index) {
        text[2 * index] = digits[data[index] >> 4];
        text[2 * index + 1] = digits[data[index] & 15];
    }
    return text;
}

std::string read_bytes(const fs::path& path, std::size_t size, std::uintmax_t& total) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    std::string buffer(size, '\0');
    file.read(buffer.data(), static_cast<std::streamsize>(size));
    std::error_code failure;
    total = fs::file_size(path, failure);
    return static_cast<std::size_t>(file.gcount()) == size && !failure ? buffer : std::string{};
}

std::int64_t file_seconds(const fs::path& path, std::error_code& failure) {
    const auto moment = fs::last_write_time(path, failure);
    if (failure) return 0;
    const auto system = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        moment - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
    return std::chrono::duration_cast<std::chrono::seconds>(system.time_since_epoch()).count();
}

std::string stamp(std::int64_t seconds) {
    std::tm utc{};
    const std::time_t moment = static_cast<std::time_t>(seconds);
#ifdef _WIN32
    gmtime_s(&utc, &moment);
#else
    gmtime_r(&moment, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S+00:00");
    return out.str();
}

struct Block {
    fs::path path;
    std::string space, digest, parent;
    std::uint32_t tokens = 0, block_size = 0;
    std::uint64_t payload = 0, bytes = 0;
    std::int64_t modified = 0;
    bool text_saved = false;
    std::uintmax_t text_bytes = 0;
};

struct TextRecord {
    std::vector<std::int64_t> tokens;
    std::uintmax_t bytes = 0;
    std::int64_t modified = 0;
};

TextRecord text_record(const fs::path& block, const std::string& space, const std::string& digest, bool read) {
    TextRecord record;
    const fs::path side = fs::path(block) += ".tokens";
    std::string raw;
    {
        std::ifstream file(side, std::ios::binary);
        if (!file) return record;
        raw.resize(text_header);
        file.read(raw.data(), static_cast<std::streamsize>(text_header));
        if (static_cast<std::size_t>(file.gcount()) != text_header) return record;
        std::error_code failure;
        const std::uintmax_t total = fs::file_size(side, failure);
        if (failure || !total || total > (64 << 20)) return record;
        const std::uint32_t count = *reinterpret_cast<const std::uint32_t*>(raw.data() + 72);
        if (raw.compare(0, 8, "MFQTXT1\0", 8) != 0 || hex(reinterpret_cast<const std::uint8_t*>(raw.data()) + 8, 32) != space ||
            hex(reinterpret_cast<const std::uint8_t*>(raw.data()) + 40, 32) != digest || count == 0 || count > 65536 ||
            total != text_header + static_cast<std::uintmax_t>(count) * 8)
            return record;
        record.bytes = total;
        record.modified = file_seconds(side, failure);
        if (!read) return record;
        std::string payload(static_cast<std::size_t>(count) * 8, '\0');
        file.read(payload.data(), static_cast<std::streamsize>(payload.size()));
        if (static_cast<std::size_t>(file.gcount()) != payload.size()) return record;
        const auto checksum = mfq::cache::sha256(payload);
        if (std::memcmp(checksum.data(), raw.data() + 76, 32) != 0) return record;
        record.tokens.resize(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            std::uint64_t value = 0;
            std::memcpy(&value, payload.data() + static_cast<std::size_t>(index) * 8, 8);
            if (value > 0xffffffffull) return record;
            record.tokens[index] = static_cast<std::int64_t>(value);
        }
    }
    return record;
}

std::vector<Block> scan_blocks(const fs::path& directory) {
    std::vector<Block> blocks;
    std::error_code ignored;
    if (!fs::is_directory(directory, ignored)) return blocks;
    for (const auto& space : fs::directory_iterator(directory, fs::directory_options::skip_permission_denied, ignored)) {
        const std::string namespace_ = space.path().filename().string();
        if (!hex_of(namespace_) || !space.is_directory()) continue;
        for (const auto& bucket : fs::directory_iterator(space.path(), fs::directory_options::skip_permission_denied, ignored)) {
            const std::string bucket_name = bucket.path().filename().string();
            if (bucket_name.size() != 2 || !std::all_of(bucket_name.begin(), bucket_name.end(),
                    [](unsigned char c) { return std::isxdigit(c) != 0; }) || !bucket.is_directory())
                continue;
            for (const auto& entry : fs::directory_iterator(bucket.path(), fs::directory_options::skip_permission_denied, ignored)) {
                const std::string name = entry.path().filename().string();
                const std::size_t dot = name.rfind('.');
                if (dot == std::string::npos || (name.compare(dot, 5, ".tqkv") != 0 && name.compare(dot, 6, ".mfqkv") != 0) ||
                    !entry.is_regular_file())
                    continue;
                std::uintmax_t total = 0;
                const std::string raw = read_bytes(entry.path(), block_header, total);
                if (raw.size() != block_header) continue;
                const auto* bytes = reinterpret_cast<const std::uint8_t*>(raw.data());
                if (raw.compare(0, 8, "MFQKVB1\0", 8) != 0 ||
                    *reinterpret_cast<const std::uint32_t*>(raw.data() + 8) != 1 ||
                    hex(bytes + 12, 32) != namespace_ || name.compare(0, dot, hex(bytes + 44, 32)) != 0 ||
                    bucket_name != hex(bytes + 44, 1))
                    continue;
                Block block;
                block.space = namespace_;
                block.path = entry.path();
                block.digest = hex(bytes + 44, 32);
                block.parent = hex(bytes + 76, 32);
                block.block_size = *reinterpret_cast<const std::uint32_t*>(raw.data() + 108);
                block.tokens = *reinterpret_cast<const std::uint32_t*>(raw.data() + 112);
                block.payload = *reinterpret_cast<const std::uint64_t*>(raw.data() + 116);
                if (block.tokens == 0 || block.tokens > block.block_size || block.block_size > 65536 ||
                    total != block_header + block.payload)
                    continue;
                block.bytes = total;
                std::error_code failure;
                block.modified = file_seconds(entry.path(), failure);
                if (const TextRecord side = text_record(block.path, namespace_, block.digest, false); side.bytes) {
                    block.text_saved = true;
                    block.text_bytes = side.bytes;
                }
                blocks.push_back(std::move(block));
            }
        }
    }
    return blocks;
}

// identity.txt names the model that wrote a namespace; model_path is private to the decode path only.
json identity(const fs::path& directory, const std::string& space) {
    json record{{"model_name", nullptr}, {"source_file", nullptr}, {"architecture", nullptr},
        {"codec", nullptr}, {"context_size", nullptr}, {"model_path", nullptr}};
    if (!hex_of(space)) return record;
    std::ifstream file(directory / space / "identity.txt", std::ios::binary);
    if (!file) return record;
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (text.size() > 32768) return record;
    std::istringstream lines(text);
    std::string line;
    std::getline(lines, line);
    if (line != "mfq-cache-identity-v1") return record;
    std::getline(lines, line);
    if (line != "namespace=" + space) return record;
    while (std::getline(lines, line)) {
        const std::size_t split = line.find('=');
        if (split == std::string::npos) continue;
        const std::string key = line.substr(0, split), value = line.substr(split + 1);
        if (key == "codec" || key == "architecture") record[key] = value.substr(0, 256);
        else if (key == "context" && !value.empty() &&
            std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isdigit(c) != 0; }))
            record["context_size"] = std::stoll(value);
        else if (key == "model_path") record["model_path"] = value;
        else if ((key == "shard" || key == "source") && value.rfind("0:", 0) == 0) {
            const std::size_t last = value.rfind(':');
            const std::string name = value.substr(2, last == std::string::npos ? std::string::npos : last - 2);
            record["model_name"] = name.substr(0, name.rfind(':') == std::string::npos ? std::string::npos : name.rfind(':'));
        }
    }
    return record;
}

struct Chain {
    std::int64_t tokens = 0;
    bool complete = false;
};

// Prefix length per block digest, following parent links back to the genesis block.
std::map<std::string, Chain> chain_lengths(const std::map<std::string, const Block*>& by_digest) {
    std::map<std::string, Chain> lengths;
    for (const auto& [digest, block] : by_digest) {
        std::vector<std::string> path;
        std::set<std::string> visited;
        std::string current = digest;
        while (lengths.find(current) == lengths.end() && by_digest.find(current) != by_digest.end() && visited.insert(current).second) {
            path.push_back(current);
            current = by_digest.at(current)->parent;
        }
        auto known = lengths.find(current);
        Chain count = known != lengths.end() ? known->second : Chain{0, current == genesis};
        for (auto item = path.rbegin(); item != path.rend(); ++item) {
            count.tokens += by_digest.at(*item)->tokens;
            lengths[*item] = count;
        }
    }
    return lengths;
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
    return value;
}

std::shared_ptr<engine::TextProcessor> decoder(ModelCatalog& catalog, const fs::path& directory,
    const std::string& space) {
    const auto meta = identity(directory, space);
    if (!meta.contains("model_path") || meta["model_path"].is_null()) return nullptr;
    std::error_code ignored;
    const fs::path model = fs::weakly_canonical(meta["model_path"].get<std::string>(), ignored);
    for (const auto& artifact : catalog.list(false)) {
        std::error_code skip;
        if (!fs::equivalent(fs::weakly_canonical(artifact.path, skip), model, skip)) continue;
        try {
            auto source = mfq::open_model_source(artifact.path);
            return engine::TextProcessor::load(*source, "", 0, "");
        } catch (const std::exception&) { return nullptr; }
    }
    return nullptr;
}

std::string decode(const std::vector<std::int64_t>& tokens, ModelCatalog& catalog,
    const fs::path& directory, const std::string& space) {
    const auto processor = decoder(catalog, directory, space);
    if (!processor) throw ApiError(503, "tokenizer_unavailable", "the model tokenizer is unavailable");
    return processor->decode_tokens(tokens);
}
} // namespace

PrefixCache::PrefixCache(Storage& storage, ModelCatalog& catalog, ModelInstances& instances)
    : storage_(storage), catalog_(catalog), instances_(instances) {}

fs::path PrefixCache::directory() const {
    RuntimePolicies policies(storage_);
    const auto saved = policies.memory_policy();
    if (saved.contains("prefix_directory") && saved["prefix_directory"].is_string())
        return fs::weakly_canonical(saved["prefix_directory"].get<std::string>());
    // The engine reads the same variable, so an unset policy shows the cache the engine writes.
    if (const char* configured = std::getenv("TYLOINFER_RUNTIME_PREFIX_CACHE_DIR"); configured && *configured)
        return fs::weakly_canonical(configured);
    return engine::resolve_runtime_config(1).prefix_cache.directory;
}

json PrefixCache::entries(const json& filters) const {
    const fs::path directory = this->directory();
    std::vector<Block> blocks = scan_blocks(directory);
    const std::string space = filters.value("namespace", "");
    const std::int64_t offset = filters.value("offset", 0), limit = filters.value("limit", 100);
    const std::string needle = lower(filters.value("query", ""));
    const std::string search_in = filters.value("search_in", "all");
    const std::string text_filter = filters.value("text_filter", "all");
    const std::string chain_filter = filters.value("chain_filter", "all");
    const std::string sort = filters.value("sort", "recent");

    std::map<std::string, std::vector<const Block*>> groups;
    std::map<std::string, const Block*> by_digest;
    std::uint64_t total_bytes = 0;
    for (const auto& block : blocks) {
        total_bytes += block.bytes + (block.text_saved ? block.text_bytes : 0);
        groups[block.space].push_back(&block);
        by_digest[block.digest] = &block;
    }
    const auto lengths = chain_lengths(by_digest);

    json data = json::array(), details = json::array();
    std::size_t matched = 0;
    bool text_unavailable = false;
    for (const auto& [identity_space, items] : groups) {
        if (!space.empty() && space != identity_space) continue;
        std::set<std::string> parents;
        for (const auto* item : items) parents.insert(item->parent);
        std::vector<const Block*> tips;
        for (const auto* item : items) if (!parents.count(item->digest)) tips.push_back(item);
        json meta = identity(directory, identity_space);
        meta.erase("model_path");
        std::int64_t newest = 0;
        std::uint64_t bytes = 0;
        std::size_t text_blocks = 0;
        std::uint32_t block_tokens = 0;
        std::int64_t max_prefix = 0;
        std::size_t incomplete = 0;
        for (const auto* item : items) {
            bytes += item->bytes + (item->text_saved ? item->text_bytes : 0);
            text_blocks += item->text_saved;
            block_tokens = std::max(block_tokens, item->block_size);
            newest = std::max(newest, item->modified);
        }
        for (const auto* tip : tips) {
            const auto& length = lengths.at(tip->digest);
            if (length.complete) max_prefix = std::max(max_prefix, length.tokens);
            else ++incomplete;
        }
        json item{{"id", identity_space}, {"bytes", bytes}, {"blocks", items.size()},
            {"prefixes", tips.size()}, {"text_blocks", text_blocks}, {"block_tokens", block_tokens},
            {"max_prefix_tokens", max_prefix}, {"incomplete_prefixes", incomplete}, {"last_used_at", stamp(newest)}};
        item.merge_patch(meta);
        data.push_back(std::move(item));
        if (space.empty()) continue;
        const auto reader = !needle.empty() && search_in != "id" ? decoder(catalog_, directory, identity_space) : nullptr;
        std::vector<const Block*> filtered;
        for (const auto* item : items) {
            if (text_filter != "all" && item->text_saved != (text_filter == "saved")) continue;
            if (chain_filter != "all" && lengths.at(item->digest).complete != (chain_filter == "complete")) continue;
            bool matches = needle.empty() || (search_in != "text" && item->digest.find(needle) != std::string::npos);
            if (!matches && search_in != "id" && item->text_saved) {
                if (!reader) text_unavailable = true;
                else if (const auto record = text_record(item->path, identity_space, item->digest, true); !record.tokens.empty()) {
                    const std::string text = lower(reader->decode_tokens(record.tokens));
                    matches = text.find(needle) != std::string::npos;
                }
            }
            if (matches) filtered.push_back(item);
        }
        const auto value = [&](const Block* item) {
            if (sort == "length") return std::make_pair(lengths.at(item->digest).tokens, item->digest);
            if (sort == "size") return std::make_pair(static_cast<std::int64_t>(item->bytes), item->digest);
            return std::make_pair(item->modified, item->digest);
        };
        std::sort(filtered.begin(), filtered.end(), [&](const Block* left, const Block* right) {
            const auto one = value(left), two = value(right);
            return sort == "oldest" ? std::make_pair(one.first, one.second) < std::make_pair(two.first, two.second) : one > two;
        });
        matched = filtered.size();
        for (std::size_t index = static_cast<std::size_t>(std::max<std::int64_t>(offset, 0));
            index < filtered.size() && index < static_cast<std::size_t>(std::max<std::int64_t>(offset, 0)) +
            static_cast<std::size_t>(std::min<std::int64_t>(limit, 200)); ++index) {
            const auto* item = filtered[index];
            const auto& length = lengths.at(item->digest);
            details.push_back(json{{"id", item->digest}, {"parent", item->parent}, {"tokens", item->tokens},
                {"prefix_tokens", length.tokens}, {"complete_chain", length.complete},
                {"payload_bytes", item->payload}, {"bytes", item->bytes + (item->text_saved ? item->text_bytes : 0)},
                {"text_available", item->text_saved}, {"last_used_at", stamp(item->modified)}});
        }
    }
    std::sort(data.begin(), data.end(), [](const json& left, const json& right) {
        return left.at("last_used_at").get<std::string>() > right.at("last_used_at").get<std::string>();
    });
    bool can_clear = true;
    for (const auto& item : instances_.list())
        if (item.execution.requests || item.state != InstanceInfo::State::loaded) can_clear = false;
    return json{{"directory", directory.string()}, {"total_bytes", total_bytes}, {"total_blocks", blocks.size()},
        {"data", std::move(data)}, {"blocks", std::move(details)}, {"offset", offset}, {"limit", limit},
        {"matched_blocks", matched}, {"search_text_unavailable", text_unavailable}, {"can_clear", can_clear}};
}

json PrefixCache::block_text(const std::string& space, const std::string& block, std::int64_t offset, std::int64_t limit) const {
    if (!hex_of(space) || !hex_of(block))
        throw ApiError(404, "cache_block_not_found", "prefix cache block was not found");
    const fs::path path = directory() / space / block.substr(0, 2) / (block + ".tqkv");
    const TextRecord record = text_record(path, space, block, true);
    if (record.tokens.empty()) return json{{"available", false}, {"reason", "block_not_found"}};
    const std::size_t from = static_cast<std::size_t>(std::max<std::int64_t>(offset, 0));
    if (from >= record.tokens.size()) return json{{"available", false}, {"reason", "offset_out_of_range"}};
    const std::size_t count = std::min<std::size_t>(limit, record.tokens.size() - from);
    const std::vector<std::int64_t> slice(record.tokens.begin() + static_cast<std::ptrdiff_t>(from),
        record.tokens.begin() + static_cast<std::ptrdiff_t>(from + count));
    try {
        return json{{"available", true}, {"text", decode(slice, catalog_, directory(), space)},
            {"total_tokens", record.tokens.size()}, {"offset", from},
            {"next_offset", from + count < record.tokens.size() ? json(from + count) : json(nullptr)}};
    } catch (const ApiError&) { return json{{"available", false}, {"reason", "tokenizer_unavailable"}}; }
}

json PrefixCache::purge(const std::optional<std::string>& space) {
    const fs::path directory = this->directory();
    std::vector<Block> blocks = scan_blocks(directory);
    std::size_t removed = 0, bytes = 0, failed = 0;
    bool found = !space.has_value();
    for (const auto& block : blocks) {
        if (space && block.space != *space) continue;
        found = true;
        std::error_code ignored;
        const std::uintmax_t size = fs::file_size(block.path, ignored);
        if (fs::remove(block.path, ignored)) {
            ++removed;
            bytes += size;
        } else ++failed;
        if (block.text_saved) {
            const std::uintmax_t text_size = fs::file_size(fs::path(block.path) += ".tokens", ignored);
            if (fs::remove(fs::path(block.path) += ".tokens", ignored)) bytes += text_size;
        }
    }
    if (!found) throw ApiError(404, "cache_not_found", "prefix cache group was not found");
    // Drop the group directory too when it is empty, as the Python purge does.
    if (space) {
        std::error_code ignored;
        fs::remove(directory / *space, ignored);
    }
    return json{{"status", "ok"}, {"removed_blocks", removed}, {"released_bytes", bytes}, {"failed_blocks", failed}};
}
} // namespace mfq::server

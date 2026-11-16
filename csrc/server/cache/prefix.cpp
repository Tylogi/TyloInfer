#include "prefix.h"
#include "api/errors.h"
#include "engine/config.h"
#include "engine/requests/processor.h"
#include "engine/models/source/model.h"
#include "engine/state/session/prefix_blocks.h"
#include "models/catalog.h"
#include "models/instances.h"
#include "models/policies.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

namespace tyloi::server {
namespace {
using json = nlohmann::json;
namespace fs = std::filesystem;

constexpr const char* genesis = "0000000000000000000000000000000000000000000000000000000000000000";

bool hex_of(std::string_view value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isxdigit(c) != 0; });
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

struct Chain {
    std::int64_t tokens = 0;
    bool complete = false;
};

// Prefix length per block digest, following parent links back to the genesis block.
std::map<std::string, Chain> chain_lengths(
    const std::map<std::string, const cache::PrefixBlockInfo*>& by_digest) {
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

json identity_json(const cache::PrefixIdentity& identity) {
    json record{{"model_name", nullptr}, {"source_file", nullptr}, {"architecture", nullptr},
        {"codec", nullptr}, {"context_size", nullptr}};
    if (identity.model_name) record["model_name"] = *identity.model_name;
    if (identity.source_file) record["source_file"] = *identity.source_file;
    if (identity.architecture) record["architecture"] = *identity.architecture;
    if (identity.codec) record["codec"] = *identity.codec;
    if (identity.context_size) record["context_size"] = *identity.context_size;
    return record;
}

std::shared_ptr<engine::TextProcessor> decoder(ModelCatalog& catalog, const cache::PrefixIdentity& identity) {
    if (!identity.model_path) return nullptr;
    std::error_code ignored;
    const fs::path model = fs::weakly_canonical(*identity.model_path, ignored);
    if (ignored) return nullptr;
    for (const auto& artifact : catalog.list(false)) {
        std::error_code skip;
        if (!fs::equivalent(fs::weakly_canonical(artifact.path, skip), model, skip)) continue;
        try {
            auto source = tyloi::open_model_source(artifact.path);
            return engine::TextProcessor::load(*source, "", 0, "");
        } catch (const std::exception&) { return nullptr; }
    }
    return nullptr;
}

std::string decode(const std::vector<std::int64_t>& tokens, ModelCatalog& catalog,
    const cache::PrefixIdentity& identity) {
    const auto processor = decoder(catalog, identity);
    if (!processor) throw ApiError(503, "tokenizer_unavailable", "the model tokenizer is unavailable");
    return processor->decode_tokens(tokens);
}
} // namespace

PrefixCache::PrefixCache(Storage& storage, ModelCatalog& catalog, EngineInstances& instances)
    : storage_(storage), catalog_(catalog), instances_(instances) {}

fs::path PrefixCache::directory() const {
    RuntimePolicies policies(storage_);
    const auto saved = policies.memory_policy();
    if (saved.contains("prefix_directory") && saved["prefix_directory"].is_string())
        return fs::weakly_canonical(saved["prefix_directory"].get<std::string>());
    // The engine reads the same variable, so an unset policy shows the cache the engine writes.
    if (const char* configured = std::getenv("TYLOINFER_RUNTIME_PREFIX_CACHE_DIR"); configured && *configured)
        return fs::weakly_canonical(configured);
    return engine::resolve_config(1).prefix_cache.directory;
}

json PrefixCache::entries(const json& filters) const {
    const fs::path directory = this->directory();
    const std::string space = filters.value("namespace", "");
    const std::int64_t offset = filters.value("offset", 0), limit = filters.value("limit", 100);
    const std::string needle = lower(filters.value("query", ""));
    const std::string search_in = filters.value("search_in", "all");
    const std::string text_filter = filters.value("text_filter", "all");
    const std::string chain_filter = filters.value("chain_filter", "all");
    const std::string sort = filters.value("sort", "recent");
    const auto inspection = cache::PrefixBlockStore::inspect_directory(directory);
    const auto& blocks = inspection.blocks;

    std::map<std::string, std::vector<const cache::PrefixBlockInfo*>> groups;
    std::map<std::string, const cache::PrefixBlockInfo*> by_digest;
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
        std::vector<const cache::PrefixBlockInfo*> tips;
        for (const auto* item : items) if (!parents.count(item->digest)) tips.push_back(item);
        const auto identity = inspection.identities.find(identity_space);
        json meta = identity == inspection.identities.end()
            ? identity_json({})
            : identity_json(identity->second);
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
        const auto reader = !needle.empty() && search_in != "id"
            ? decoder(catalog_, identity == inspection.identities.end()
                ? cache::PrefixIdentity{} : identity->second)
            : nullptr;
        std::vector<const cache::PrefixBlockInfo*> filtered;
        for (const auto* item : items) {
            if (text_filter != "all" && item->text_saved != (text_filter == "saved")) continue;
            if (chain_filter != "all" && lengths.at(item->digest).complete != (chain_filter == "complete")) continue;
            bool matches = needle.empty() || (search_in != "text" && item->digest.find(needle) != std::string::npos);
            if (!matches && search_in != "id" && item->text_saved) {
                if (!reader) text_unavailable = true;
                else if (const auto record = cache::PrefixBlockStore::read_text_tokens(
                    directory, identity_space, item->digest); !record.tokens.empty()) {
                    const std::string text = lower(reader->decode_tokens(record.tokens));
                    matches = text.find(needle) != std::string::npos;
                }
            }
            if (matches) filtered.push_back(item);
        }
        const auto value = [&](const cache::PrefixBlockInfo* item) {
            if (sort == "length") return std::make_pair(lengths.at(item->digest).tokens, item->digest);
            if (sort == "size") return std::make_pair(static_cast<std::int64_t>(item->bytes), item->digest);
            return std::make_pair(item->modified, item->digest);
        };
        std::sort(filtered.begin(), filtered.end(), [&](const cache::PrefixBlockInfo* left,
            const cache::PrefixBlockInfo* right) {
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
                {"payload_bytes", item->payload},
                {"bytes", item->bytes + (item->text_saved ? item->text_bytes : 0)},
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
    const auto record = cache::PrefixBlockStore::read_text_tokens(directory(), space, block);
    if (record.tokens.empty()) return json{{"available", false}, {"reason", "block_not_found"}};
    const std::size_t from = static_cast<std::size_t>(std::max<std::int64_t>(offset, 0));
    if (from >= record.tokens.size()) return json{{"available", false}, {"reason", "offset_out_of_range"}};
    const std::size_t count = std::min<std::size_t>(limit, record.tokens.size() - from);
    const std::vector<std::int64_t> slice(record.tokens.begin() + static_cast<std::ptrdiff_t>(from),
        record.tokens.begin() + static_cast<std::ptrdiff_t>(from + count));
    try {
        return json{{"available", true}, {"text", decode(slice, catalog_, record.identity)},
            {"total_tokens", record.tokens.size()}, {"offset", from},
            {"next_offset", from + count < record.tokens.size() ? json(from + count) : json(nullptr)}};
    } catch (const ApiError&) { return json{{"available", false}, {"reason", "tokenizer_unavailable"}}; }
}

json PrefixCache::purge(const std::optional<std::string>& space) {
    const auto result = instances_.reclaim_prefix_cache(directory(), space);
    if (!result.found) throw ApiError(404, "cache_not_found", "prefix cache group was not found");
    return json{{"status", "ok"}, {"removed_blocks", result.removed_blocks},
        {"released_bytes", result.released_bytes}, {"failed_blocks", result.failed_blocks}};
}
} // namespace tyloi::server

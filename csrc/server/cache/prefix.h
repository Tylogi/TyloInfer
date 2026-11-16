#pragma once

#include <nlohmann/json.hpp>
#include <filesystem>
#include <optional>
#include <string>

namespace tyloi::server {
class Storage;
class ModelCatalog;
class EngineInstances;

// Adapter for the engine's on-disk prefix cache. The engine owns inspection,
// token reads, and reclamation; this module preserves the Server API view.
class PrefixCache {
public:
    PrefixCache(Storage& storage, ModelCatalog& catalog, EngineInstances& instances);
    std::filesystem::path directory() const;
    // Filters: namespace, offset, limit, query, search_in, text_filter, chain_filter, sort.
    nlohmann::json entries(const nlohmann::json& filters) const;
    nlohmann::json block_text(const std::string& space, const std::string& block,
        std::int64_t offset, std::int64_t limit) const;
    nlohmann::json purge(const std::optional<std::string>& space);

private:
    Storage& storage_;
    ModelCatalog& catalog_;
    EngineInstances& instances_;
};
} // namespace tyloi::server

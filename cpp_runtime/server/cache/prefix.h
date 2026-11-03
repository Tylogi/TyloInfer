#pragma once

#include <nlohmann/json.hpp>
#include <filesystem>
#include <optional>
#include <string>

namespace mfq::server {
class Storage;
class ModelCatalog;
class ModelInstances;

// The engine's on-disk prefix cache (MFQKVB1 blocks, MFQTXT1 token sidecars, identity.txt).
// The same files the engine writes; this module only inspects, purges and decodes them,
// mirroring the Python prefix_disk_budget service so Studio sees the same view.
class PrefixCache {
public:
    PrefixCache(Storage& storage, ModelCatalog& catalog, ModelInstances& instances);
    std::filesystem::path directory() const;
    // Filters: namespace, offset, limit, query, search_in, text_filter, chain_filter, sort.
    nlohmann::json entries(const nlohmann::json& filters) const;
    nlohmann::json block_text(const std::string& space, const std::string& block,
        std::int64_t offset, std::int64_t limit) const;
    nlohmann::json purge(const std::optional<std::string>& space);

private:
    Storage& storage_;
    ModelCatalog& catalog_;
    ModelInstances& instances_;
};
} // namespace mfq::server

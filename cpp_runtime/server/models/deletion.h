#pragma once

#include "catalog.h"

#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace mfq::server {

/** The files a delete request would remove, bound to their exact on-disk state. */
struct DeletionPlan {
    std::string model_id, name, token;
    std::vector<std::filesystem::path> files;
    std::uint64_t total_bytes = 0;
};

/** Throws ModelError(conflict, "model_delete_unsafe" | "model_delete_changed") when the set is not safe. */
DeletionPlan model_deletion_plan(const ModelCatalog::Artifact& artifact);
nlohmann::json deletion_plan_json(const DeletionPlan& plan);
/** Removes the planned files only when the token still matches; returns {"files","total_bytes"}. */
nlohmann::json delete_model_files(const ModelCatalog::Artifact& artifact, const std::string& token);

} // namespace mfq::server

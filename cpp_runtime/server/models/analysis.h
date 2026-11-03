#pragma once

#include "catalog.h"

#include <nlohmann/json.hpp>

namespace mfq::server {

/** ModelCacheProfile JSON for the artifact, or null when its architecture has no cache model. */
nlohmann::json model_cache_profile_json(const ModelCatalog::Artifact& artifact, bool include_recurrent, bool cuda);

/** CheckpointAnalysis JSON built from the model source metadata. */
nlohmann::json checkpoint_analysis_json(const ModelCatalog::Artifact& artifact, bool cuda);

} // namespace mfq::server

#pragma once

#include "engine/models/source/model.h"

namespace tyloi::models {

std::shared_ptr<const ModelSource> open_residual_model_source(
    const std::filesystem::path& path);

} // namespace tyloi::models

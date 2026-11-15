#pragma once

#include "engine/models/common/mtp.h"

#include <memory>

namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda::glm5_next {

struct Model;

void load_components(
    Model& model, const tyloi::weights::Loader& weights, bool load_optional_components);

} // namespace tyloi::cuda::glm5_next

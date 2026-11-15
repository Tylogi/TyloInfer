#pragma once

#include "engine/metrics.h"
#include "engine/ops.h"
#include "engine/models/common/mtp.h"
#include "engine/models/common/vision/grid_encoder.h"
#include "engine/state/media.h"

#include <memory>
#include <optional>
#include <string_view>

namespace tyloi::weights { struct Loader; }
namespace tyloi::models { struct LoadOptions; }

namespace tyloi::models::qwen4_exp {

struct Model;
class State;

struct Components {
    std::optional<tyloi::models::grid::Vision> grid_vision;
    std::unique_ptr<tyloi::models::Mtp> mtp;

    static constexpr bool streamed_kv() noexcept { return true; }
    tyloi::engine::Metrics profile_metrics(const Model&, const State&) const;
    std::optional<tyloi::engine::Metrics> memory_metrics(const Model&, const State&) const;
    /** Prepare supported media using the request's cache and model weights. */
    tyloi::StepSequence<tyloi::models::Prepared> prepare(
        const Model& language, tyloi::cache::Media& images,
        Device device,
        const std::vector<int64_t>& tokens, const MfqMultimodalInput& media) const;
};

/** Load supported optional components using the model's configuration. */
void load_components(Components& result, Model& model, const tyloi::weights::Loader& weights,
    bool load_optional_components, const tyloi::models::LoadOptions& options);

} // namespace tyloi::models::qwen4_exp

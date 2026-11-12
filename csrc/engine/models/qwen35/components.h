#pragma once
#include "engine/models/source.h"
#include "engine/ops.h"

#include "engine/engine.h"
#include "engine/models/common/capabilities.h"
#include "engine/models/qwen35/model.h"
#include "engine/models/qwen35/mtp.h"
#include "engine/backends/cuda/native/profile.h"

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace tyloi { class Executor; }

namespace tyloi::weights {
class Cache;
struct Loader;
}

namespace tyloi::cuda {
class Executor;
}

struct LayerPlacementConfig;

namespace tyloi::cuda::qwen35 {

struct Components {
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    bool vision_available = false;
    std::unique_ptr<Mtp> mtp;

    static Components load(Model& model, const ::tyloi::weights::Loader& weights, ::tyloi::weights::Cache& cache,
        LayerPlacementConfig& placement, int& cpu_layers, Executor& executor,
        std::unique_ptr<tyloi::Executor>& predictor_execution,
        bool optional, bool cpu_offload, bool preallocate);

    CudaComponentState state() const noexcept {
        return cuda_component_state(
            graph, plan, vision_available, bool(mtp));
    }

    static constexpr bool streamed_kv() noexcept { return false; }

    tyloi::engine::Metrics metrics() const;

    void describe(tyloi::engine::EngineInfo& info) const;

    template <class T>
    tyloi::engine::ControlResult control(const T&) {
        throw std::invalid_argument("model has no duplex component");
    }
};

} // namespace tyloi::cuda::qwen35

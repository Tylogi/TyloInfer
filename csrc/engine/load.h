#pragma once

#include "engine/weights/placement.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>

namespace tyloi::engine { class Engine; }

namespace tyloi::engine {

struct LoadOptions {
    Device device{};
    std::string model_path, config_path, tokenizer_model;
    std::string moe_cache_profile_path;
    std::unordered_set<int> cpu_offload_layers;
    ParallelConfig tensor_parallel, expert_parallel;
    LayerPlacementConfig layer_placement;
    double moe_gpu_cache_gb = 0.0;
    int cpu_threads = 0;
    int64_t context_size = 0;
    bool yarn_context_extension = false;
    bool n_gpu_layers_set = false;
    bool cpu_threads_set = false;
    int n_gpu_layers = -1;
    std::string dflash2_draft_path;
    std::optional<bool> jev_full_vram;
};

struct EngineOptions : LoadOptions {
    int continuous_batching = 0;
    // Zero selects Auto; the resolved runtime limit is always positive.
    int64_t prefill_chunk_size = 0;
};

/** Load and publish an engine after model, cache and execution preparation succeeds. */
std::unique_ptr<Engine> load(EngineOptions options);
/** Replace dependencies while the engine's reload owner holds the scheduler quiescent. */
void reload(Engine& engine, EngineOptions options);

} // namespace tyloi::engine

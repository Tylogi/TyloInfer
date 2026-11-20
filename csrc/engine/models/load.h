#pragma once

#include "engine/ops/ops.h"
#include "engine/state/kv/layout.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>

struct LayerPlacementConfig;

namespace tyloi {
class ModelSource;
namespace weights { struct Loader; }

namespace models {
class CausalModel;

/** Return an explanation when the graph has no loader for the requested device. */
std::string unavailable_reason(const ModelSource& source, Device target);

struct LoadOptions {
    std::int64_t context = 0;
    int gpu_layers = -1;
    bool expert_parallel = false;
    bool recurrent_transposed = false;
    bool preallocate = false;
    bool vision_resident = false;
    bool mirror_scalars = true;
    tyloi::ops::YarnScaling yarn;
    std::unordered_set<int> cpu_offload_layers;
    std::size_t engram_cache_rows = 16'384;
    bool host_tokens = false;
    bool optional_components = true;
    tyloi::state::KvQuantization kv_quantization;
    std::string dflash2_draft_path;
    std::optional<bool> stream_layer_weights;
    bool prefetch_layer_weights = true;
};

/** Assemble the model into an owner retained by the caller. */
void load(std::unique_ptr<CausalModel>& owner,
    std::shared_ptr<const ModelSource> source, std::string_view definition,
    const weights::Loader& weights, LayerPlacementConfig& placement,
    LoadOptions options = {});

std::unique_ptr<CausalModel> load(
    std::shared_ptr<const ModelSource> source, std::string_view definition,
    const weights::Loader& weights, LayerPlacementConfig& placement,
    LoadOptions options = {});

} // namespace models
} // namespace tyloi

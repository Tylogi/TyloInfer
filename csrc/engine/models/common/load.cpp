#include "engine/models/source/model.h"
#include "engine/ops/ops.h"
#include "engine/models/common/load.h"
#include "engine/models/common/attention.h"
#include "engine/models/source/model_graph.h"
#include "engine/weights/moe/prepare/load.h"
#include "engine/weights/loader.h"
#include "engine/weights/prepare/quant.h"
#include "engine/weights/tq/routed.h"
#include "engine/backends/cuda/ops/ffn/experts/moe.h"

#include <cstdio>

std::vector<std::int64_t> TransformerWeightLoader::shape(const ::tyloi::DenseTensor& value) { return value.sizes().vec(); }
int64_t TransformerWeightLoader::elements(const ::tyloi::DenseTensor& value) { return value.numel(); }

tyloi::models::Moe load_moe_weights(const tyloi::weights::Loader& weights,
                     std::string_view prefix, const MoeWeightLoadOptions& options) {
    TransformerWeightLoader loader{weights};
    loader.keep_quantized_router=true;
    loader.cpu_offloaded = options.cpu_offloaded;
    loader.shared_gate_up_compatible_prefix = options.shared_gate_up_compatible_prefix;
    auto result=tyloi::models::load_moe_weights(
        loader, std::string(prefix), options.layer, options.router_bias_required,
        options.ungated_experts, options.ungated_shared);
    std::shared_ptr<const tyloi::TqBank> gate, up, down;
    if (result.gate_up.size() == 2)
        gate = std::dynamic_pointer_cast<const tyloi::TqBank>(result.gate_up[0].implementation());
    if (!result.gate_up.empty())
        up = std::dynamic_pointer_cast<const tyloi::TqBank>(
            result.gate_up.size() == 2 ? result.gate_up[1].implementation() : result.gate_up[0].implementation());
    down = std::dynamic_pointer_cast<const tyloi::TqBank>(result.down.implementation());
    if (gate || up || down) {
        TYLOINFER_RUNTIME_CHECK(up && down && (options.ungated_experts || gate),
            "TQ expert banks cannot be mixed with legacy routed formats");
        TYLOINFER_RUNTIME_CHECK(up->experts==down->experts &&
            up->out==down->input && up->input==down->out &&
            (!gate || (gate->experts==up->experts &&
                gate->out==up->out && gate->input==up->input)),
            "TQ Gate/Up/Down expert dimensions disagree");
    }
    return result;
}

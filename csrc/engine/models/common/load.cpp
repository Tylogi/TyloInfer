#include "engine/models/source/model.h"
#include "engine/ops/ops.h"
#include "engine/ops/ffn.h"
#include "engine/models/common/load.h"
#include "engine/models/common/important_neurons.h"
#include "engine/models/common/mlp.h"
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

void TransformerWeightLoader::important_neurons(int64_t hidden, int64_t intermediate,
    tyloi::ops::FfnWeights& ffn,
    const std::string& down, const std::string& gate, const std::string& up) const {
    tyloi::models::load_important_branch(weights, hidden, intermediate, ffn, ffn.is_moe,
        down, gate, up);
}
void TransformerWeightLoader::experts(tyloi::ops::FfnWeights& ffn,
                                     tyloi::models::ExpertProjection projection,
                                     const std::string& name, int layer) const {
    auto weight = expert(projection, name, layer);
    switch (projection) {
        case tyloi::models::ExpertProjection::gate: ffn.moe_gate = std::move(weight); break;
        case tyloi::models::ExpertProjection::up: ffn.moe_up = std::move(weight); break;
        case tyloi::models::ExpertProjection::gate_up: ffn.moe_gate_up = std::move(weight); break;
        case tyloi::models::ExpertProjection::down: ffn.moe_down = std::move(weight); break;
    }
}
tyloi::ops::FfnWeights load_moe_weights(const tyloi::weights::Loader& weights,
                     std::string_view prefix, const MoeWeightLoadOptions& options) {
    TransformerWeightLoader loader{weights};
    loader.keep_quantized_router=true;
    loader.cpu_offloaded = options.cpu_offloaded;
    loader.shared_gate_up_compatible_prefix = options.shared_gate_up_compatible_prefix;
    auto result=tyloi::models::load_moe_weights(
        loader, std::string(prefix), options.layer, options.router_bias_required,
        options.ungated_experts, options.ungated_shared);
    const auto gate = std::dynamic_pointer_cast<const tyloi::TqBank>(result.moe_gate.implementation());
    const auto up = std::dynamic_pointer_cast<const tyloi::TqBank>(result.moe_up.implementation());
    const auto down = std::dynamic_pointer_cast<const tyloi::TqBank>(result.moe_down.implementation());
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

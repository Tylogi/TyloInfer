#pragma once

#include "tyloinfer/loop_layer_plan.h"
#include "models/common/causal_metadata.h"

#include <optional>
#include <string_view>
#include <vector>

namespace mfq::models::inkling2 {

struct Config {
    CausalLmMetadata metadata;
    std::optional<LoopLayerPlan> plan;
    int64_t loop_start=0, loop_end=0, residual_block=0;
    int64_t window=0, relative_extent=0, relative_dim=0, log_scale_floor=0;
    int64_t taps=0, padded_vocab=0;
    std::vector<int64_t> query_heads, kv_heads, intermediate, top_k;
    float qk_clip=0, relative_clip=0, log_scale_alpha=0;
    float situ_beta=0, situ_linear_beta=0, routed_scale=1;

    static Config from_json(std::string_view payload);
};

}  // namespace mfq::models::inkling2

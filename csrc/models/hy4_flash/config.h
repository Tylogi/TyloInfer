#pragma once

#include "models/common/causal_metadata.h"
#include "models/common/moe.h"

#include <optional>
#include <string_view>
#include <vector>

namespace mfq::models::hy4_flash {

struct EngramConfig {
    int64_t ngram=0, heads=0, row_width=0, pad=0;
    std::vector<int64_t> layers, remap;
    std::vector<uint64_t> multipliers, offsets, vocab_sizes;
};

struct Config {
    CausalLmMetadata metadata;
    int64_t q_rank=0, kv_rank=0, value_head_dim=0, window=0, dense_width=0, dense_layers=0;
    int64_t group_size=0, index_budget=0, index_heads=0, index_width=0;
    int64_t latent_width=0, expert_width=0, expert_top_k=0;
    double sliding_rope_base=0, hc_magnitude=0;
    bool keep_tail=false;
    MoeRouting routing;
    std::vector<int64_t> heads;
    std::vector<double> swiglu_limit;
    std::optional<EngramConfig> engram;

    static Config from_json(std::string_view payload);
};

}  // namespace mfq::models::hy4_flash

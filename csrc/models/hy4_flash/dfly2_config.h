#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mfq { class ModelSource; }

namespace mfq::models::hy4_flash {

struct Dfly2LayerConfig {
    int64_t target_kv_layer;
    int64_t target_index_layer;
    int64_t heads;
};

struct Dfly2Config {
    int64_t hidden_size = 0;
    int64_t target_top_k = 0;
    int64_t recent_window = 0;
    int64_t mask_token_id = 0;
    int64_t intermediate_size = 0;
    bool hidden_correction = false;
    std::string fusion;
    std::vector<Dfly2LayerConfig> layers;

    static Dfly2Config from_sources(const ModelSource& target, const ModelSource& draft);
};

} // namespace mfq::models::hy4_flash

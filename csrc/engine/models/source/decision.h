#pragma once

#include "engine/models/source/model.h"
#include "engine/tensor/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace tyloi::models {

inline constexpr char decision_config_asset[] = "__mfq_asset__/decision_config.json";
inline constexpr char decision_readout_asset[] = "__mfq_asset__/readout.safetensors";
inline constexpr std::size_t decision_config_limit = 1u << 20;
inline constexpr std::size_t decision_readout_limit = 256u << 20;

inline std::size_t decision_asset_limit(std::string_view name) {
    if (name == decision_config_asset) return decision_config_limit;
    if (name == decision_readout_asset) return decision_readout_limit;
    return std::numeric_limits<std::size_t>::max();
}

enum class DecisionAttention { causal, bidirectional, prefix_bidirectional };

struct DecisionSpec {
    std::vector<std::string> codes;
    std::vector<std::int64_t> token_ids;
    DecisionAttention attention = DecisionAttention::causal;
    double temperature = 1.0;
    std::array<double, 3> primitive_temperatures{1.0, 1.0, 1.0};
    std::optional<std::size_t> max_tokens;
    bool vocabulary_readout = false;
};

std::optional<DecisionSpec> decision_spec(const ModelSource& source);
struct DecisionMatrix {
    std::vector<std::byte> bytes;
    std::size_t offset;
    ScalarType dtype;
};
DecisionMatrix read_decision_matrix(const ModelSource& source, std::int64_t hidden);

} // namespace tyloi::models

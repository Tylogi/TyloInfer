#pragma once

#include "engine/models/qwen35/config.h"
#include "engine/models/metadata.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/common/attention.h"
#include "engine/models/common/linear.h"
#include "engine/models/source.h"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tyloi::cuda::qwen35 {

using DenseTensor = ::tyloi::DenseTensor;
using Config = tyloi::models::qwen35::Config;

struct Full {
    ::tyloi::models::FullAttention attention;
    ::tyloi::Tensor ffn_norm;
    Ffn ffn;
};

struct Linear {
    ::tyloi::models::LinearAttention attention;
    ::tyloi::Tensor ffn_norm;
    Ffn ffn;
};

struct Layer {
    std::variant<Full, Linear> weights;
    int device = 0;
    bool cpu = false;
};

struct Model {
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    Config config;
    tyloi::models::CausalLmMetadata metadata;
    tyloi::ops::RopeTable rope, cpu_rope;
    std::unordered_map<int, tyloi::ops::RopeTable> device_ropes;
    ::tyloi::Tensor embed;
    ::tyloi::Tensor lm_head;
    ::tyloi::Tensor output_norm;
    std::vector<Layer> layers;
    ::tyloi::Tensor embedding(const tyloi::Tensor& ids, int primary = 0) const;
    ::tyloi::Tensor normalize(const tyloi::Tensor& hidden, std::int64_t batch, std::int64_t tokens) const;
    ::tyloi::Tensor logits(const tyloi::Tensor& hidden, bool capped = true) const;
    ::tyloi::Tensor softcap(const tyloi::Tensor& logits) const;
};

} // namespace tyloi::cuda::qwen35

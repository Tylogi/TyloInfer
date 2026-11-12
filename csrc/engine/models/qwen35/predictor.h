#pragma once

#include "engine/models/qwen35/config.h"
#include "engine/models/common/attention.h"
#include "engine/models/common/mlp.h"
#include "engine/models/input.h"

#include <span>
#include <vector>

namespace tyloi::cuda::qwen35 { class MtpState; }

namespace tyloi::models::qwen35 {

struct Predictor {
    struct Layer {
        ::tyloi::models::FullAttention attention;
        ::tyloi::Tensor ffn_norm;
        Mlp ffn;
    };

    tyloi::models::qwen35::Config config;
    ::tyloi::Tensor fusion;
    ::tyloi::Tensor hidden_norm, embedding_norm, output_norm;
    ::tyloi::Tensor embedding;
    ops::RopeTable rope;
    Device device;
    std::vector<Layer> layers;

    Tensor step(const Tensor& hidden, const Tensor& next_ids, const Tensor& positions,
                tyloi::cuda::qwen35::MtpState& state, const ForwardOptions& options = {}) const;
    Tensor forward(const Tensor& hidden, const Tensor& embeddings, const AttentionStep& step,
                   std::span<const state::KvView> kv, const ops::RopeTable& rope,
                   ops::Trace* trace = nullptr, int trace_layer = -1) const;
};

} // namespace tyloi::models::qwen35

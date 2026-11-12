#pragma once

#include "engine/models/qwen35/config.h"
#include "engine/models/qwen35/mtp.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/vision/grid_encoder.h"
#include "engine/models/metadata.h"
#include "engine/models/input.h"
#include "engine/models/common/ffn.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/common/attention.h"
#include "engine/models/common/linear.h"
#include "engine/models/source.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tyloi::cuda::qwen35 {

using Config = tyloi::models::qwen35::Config;

struct Full {
    ::tyloi::models::FullAttention attention;
    ::tyloi::Tensor ffn_norm;
    ::tyloi::models::Ffn ffn;
};

struct Linear {
    ::tyloi::models::LinearAttention attention;
    ::tyloi::Tensor ffn_norm;
    ::tyloi::models::Ffn ffn;
};

struct Layer {
    std::variant<Full, Linear> weights;
    tyloi::Device device;
};

class State;

struct Model : tyloi::models::CausalModel {
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    Config config;
    tyloi::Device device;
    tyloi::models::CausalLmMetadata metadata;
    tyloi::ops::RopeTable rope, cpu_rope;
    std::unordered_map<int, tyloi::ops::RopeTable> device_ropes;
    ::tyloi::Tensor embed;
    ::tyloi::Tensor lm_head;
    ::tyloi::Tensor output_norm;
    std::vector<Layer> layers;
    std::optional<tyloi::models::grid::Vision> vision;
    std::unique_ptr<qwen35::Mtp> mtp;
    bool mtp_supported = false;
    bool recurrent_transposed = false;
    tyloi::models::Info info() const override;
    MfqModelCapabilities capabilities() const override;
    tyloi::engine::Metrics metrics() const override;
    std::unique_ptr<tyloi::models::State> make_state() const override;
    tyloi::models::Mtp* predictor() const noexcept override { return mtp.get(); }
    tyloi::StepSequence<tyloi::models::Prepared> prepare(
        const std::vector<std::int64_t>& tokens, const MfqMultimodalInput& media,
        tyloi::models::State& state) const override;
    ::tyloi::Tensor forward(tyloi::models::Input input, tyloi::models::State& state, const tyloi::models::ForwardOptions& options = {}) const override;
    void replay(State& state, std::int64_t accepted = 0) const;
    bool supports_speculation() const noexcept override;
    ::tyloi::Tensor embedding(const tyloi::Tensor& ids, int primary = 0) const override;
    ::tyloi::Tensor normalize(const tyloi::Tensor& hidden, std::int64_t batch, std::int64_t tokens) const override;
    ::tyloi::Tensor logits(const tyloi::Tensor& hidden, bool capped = true) const override;
    ::tyloi::Tensor softcap(const tyloi::Tensor& logits) const override;
};

} // namespace tyloi::cuda::qwen35

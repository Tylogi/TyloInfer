#pragma once

#include "engine/models/qwen35/config.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/mtp.h"
#include "engine/models/common/vision/grid_encoder.h"
#include "engine/models/metadata.h"
#include "engine/models/input.h"
#include "engine/models/common/ffn.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/common/attention.h"
#include "engine/models/common/linear.h"
#include "engine/models/source/model.h"
#include "engine/models/source/decision.h"
#include "engine/weights/layer_stream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tyloi::models::qwen35 {


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

    template<class Visitor>
    void visit_tensors(Visitor visit) {
        const auto add = [&](Tensor& value) { if (value.defined()) visit(value); };
        const auto group = [&](std::vector<Tensor>& values) { for (auto& value : values) add(value); };
        const auto mlp_fields = [&](auto&& walk, Mlp& mlp) -> void {
            group(mlp.gate_up);
            add(mlp.down);
            if (mlp.important_neurons) walk(walk, *mlp.important_neurons);
        };
        std::visit([&](auto& block) {
            add(block.ffn_norm);
            auto& attention = block.attention;
            add(attention.attn_norm);
            if constexpr (std::is_same_v<std::decay_t<decltype(block)>, Full>) {
                if (attention.attention_rope.cos)
                    throw std::invalid_argument("streamed layers require the model RoPE table");
                add(attention.q_norm); add(attention.k_norm); add(attention.v_norm);
                add(attention.q_projection); add(attention.k_projection); add(attention.v_projection);
                add(attention.o);
                group(attention.qkv);
            } else {
                add(attention.conv_weight); add(attention.conv_bias); add(attention.dt_bias);
                add(attention.a_log); add(attention.linear_norm);
                add(attention.ab_proj); add(attention.zab_proj); add(attention.z_proj);
                add(attention.out_proj); add(attention.out_proj_dense);
                group(attention.in_proj); group(attention.qkv_proj); group(attention.ab_nint_proj);
            }
            mlp_fields(mlp_fields, std::get<Mlp>(block.ffn));
        }, weights);
    }
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
    std::optional<DecisionSpec> decision;
    Tensor decision_head;
    std::shared_ptr<tyloi::weights::LayerStream> layer_stream;
    std::size_t ram_embedding_bytes = 0, ram_head_bytes = 0;
    std::vector<Layer> layers;
    std::optional<tyloi::models::grid::Vision> vision;
    std::unique_ptr<tyloi::models::Mtp> mtp;
    bool recurrent_transposed = false;
    tyloi::state::KvQuantization kv_quantization;
    tyloi::models::Info info() const override;
    MfqModelCapabilities capabilities() const override;
    tyloi::engine::Metrics metrics() const override;
    std::unique_ptr<tyloi::models::State> make_state() const override;
    std::unique_ptr<tyloi::models::Batch> make_batch(const tyloi::models::BatchConfig&) const override;
    tyloi::models::Mtp* predictor() const noexcept override { return mtp.get(); }
    tyloi::StepSequence<tyloi::models::Prepared> prepare(
        const std::vector<std::int64_t>& tokens, const MfqMultimodalInput& media,
        tyloi::models::State& state) const override;
    ::tyloi::Tensor forward(tyloi::models::Input input, tyloi::models::State& state, const tyloi::models::ForwardOptions& options = {}) const override;
    StepSequence<models::Prefill> prefill(models::Input, models::State&,
        const models::ForwardOptions&, const models::PrefillOptions&) const override;
    StepSequence<models::Decoded> decode(models::Input, models::State&,
        const models::ForwardOptions&, const models::DecodeOptions&) const override;
    StepSequence<bool> replay(State& state, std::int64_t accepted, bool capture) const;
    bool supports_speculation() const noexcept override;
    ::tyloi::Tensor embedding(const tyloi::Tensor& ids, int primary = 0) const override;
    ::tyloi::Tensor normalize(const tyloi::Tensor& hidden, std::int64_t batch, std::int64_t tokens) const override;
    ::tyloi::Tensor logits(const tyloi::Tensor& hidden, bool capped = true, tyloi::models::LogitPrecision precision = tyloi::models::LogitPrecision::model) const override;
    const DecisionSpec* decisions() const noexcept override { return decision ? &*decision : nullptr; }
    Tensor decision_logits(const Tensor& hidden, const Tensor& labels) const override;
    ::tyloi::Tensor softcap(const tyloi::Tensor& logits) const override;
};

} // namespace tyloi::models::qwen35

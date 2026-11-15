#pragma once

#include "engine/models/plan.h"
#include "engine/ops.h"
#include "engine/models/causal_model.h"
#include "engine/models/metadata.h"
#include "engine/metrics.h"
#include "engine/models/common/prompt.h"
#include "engine/models/qwen4_exp/layer.h"
#include "engine/models/session/size.h"
#include "engine/models/qwen4_exp/state.h"
#include "engine/models/qwen4_exp/config.h"

#include <memory>
#include <cstdint>
#include <iosfwd>
#include <vector>

namespace tyloi::ops { class Residual; }
namespace tyloi::models { struct Mtp; struct LoadOptions; }
namespace tyloi::weights { struct Loader; }
struct LayerPlacementConfig;

namespace tyloi::models::qwen4_exp {
class State;

struct Components;

// The model record owns weights and load-time state.
struct Model : tyloi::models::CausalModel {
    using CausalConfig = tyloi::models::qwen4_exp::Config;
    static bool accepts_backbone(std::string_view backbone) { return backbone == "qwen4_exp"; }

    // ---- tyloi::models::CausalModel
    tyloi::models::Info info() const override;
    MfqModelCapabilities capabilities() const override;
    tyloi::engine::Metrics metrics() const override;
    tyloi::models::Mtp* predictor() const noexcept override;
    std::unique_ptr<tyloi::models::State> make_state() const override;
    tyloi::StepSequence<tyloi::models::Prepared> prepare(const std::vector<std::int64_t>& tokens,
        const MfqMultimodalInput& media, tyloi::models::State& state) const override;
    ::tyloi::Tensor embedding(const ::tyloi::Tensor& ids, int primary = 0) const override;
    ::tyloi::Tensor normalize(const ::tyloi::Tensor& hidden, std::int64_t batch,
        std::int64_t tokens) const override;
    ::tyloi::Tensor softcap(const ::tyloi::Tensor& logits) const override;
    ::tyloi::Tensor logits(const ::tyloi::Tensor& hidden, bool capped = true) const override;
    ::tyloi::Tensor forward(tyloi::models::Input input, tyloi::models::State& state,
                                 const tyloi::models::ForwardOptions& options = {}) const override;
    tyloi::models::PrefillPlan prefill_plan(const tyloi::models::State&,
        const tyloi::models::PrefillOptions&) const override;
    tyloi::StepSequence<tyloi::models::Prefill> prefill(tyloi::models::Input,
        tyloi::models::State&, const tyloi::models::ForwardOptions&,
        const tyloi::models::PrefillOptions&) const override;
    tyloi::StepSequence<tyloi::models::Decoded> decode(tyloi::models::Input,
        tyloi::models::State&, const tyloi::models::ForwardOptions&,
        const tyloi::models::DecodeOptions&) const override;
    tyloi::StepSequence<tyloi::models::Decoded> warmup(tyloi::models::State&,
        const tyloi::models::DecodeOptions&) const override;
    std::unique_ptr<tyloi::models::Batch> make_batch(const tyloi::models::BatchConfig&) const override;

    // ---- speculative contract used by the MTP sequence
    bool supports_speculation() const noexcept override { return true; }
    bool supports_suffix_speculation() const noexcept override { return true; }
    bool prefetch_token_rows(State& state, const ::tyloi::Tensor& ids);

    // ---- weights and load-time state
    tyloi::models::CausalLmMetadata metadata;
    std::unique_ptr<Components> parts;
    Device device;
    bool host_tokens = false;
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    ::tyloi::models::Plan plan;
    ::tyloi::Tensor embed;
    ::tyloi::Tensor lm_head;
    CausalConfig config;
    std::unique_ptr<tyloi::ops::Residual> final_mixer;
    std::vector<std::unique_ptr<qwen4_exp::Layer>> layers;

    // ---- helpers absorbed from the retired CausalLm plumbing
    Tensor device_ids(const Tensor& ids) const {
        return ops::contiguous(ops::copy(ops::cast(ids, kInt64), device));
    }
    Tensor embed_tokens(const Tensor& ids) const {
        ops::ProfileRange range("model.embed", ids.device());
        auto result = ops::embedding(ids, embed);
        range.finish();
        return result;
    }
    std::int64_t hidden_size() const noexcept { return metadata.hidden_size; }
    std::int64_t max_position_embeddings() const noexcept { return config.maximum; }
    std::int64_t num_hidden_layers() const noexcept { return config.layers; }
    bool tie_word_embeddings() const noexcept { return metadata.tie_word_embeddings; }
    void load_config(std::string_view payload);
    static Tensor repeat_hidden(const Tensor& hidden, std::int64_t streams) {
        return ops::repeat(ops::cast(hidden, kFloat16), {1, 1, streams});
    }
    Model();
    ~Model();
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;

};

void read(Model& model, std::shared_ptr<const tyloi::ModelSource> source,
          std::string_view definition, std::int64_t context_size);
void load(Model& model, const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
          const tyloi::models::LoadOptions& options, bool load_layers);

} // namespace tyloi::models::qwen4_exp

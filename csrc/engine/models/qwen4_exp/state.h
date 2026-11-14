#pragma once

#include "engine/models/state.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/session/size.h"
#include "engine/models/qwen4_exp/snapshot.h"
#include "engine/models/qwen4_exp/layer.h"
#include "engine/state/media.h"

#include <memory>
#include <utility>
#include <vector>

namespace tyloi::models { struct Mtp; }
namespace tyloi::cuda { class KvMemoryManager; class KvOffloadStore; }
namespace tyloi::cuda { struct Qwen4DecodeGraph; }
namespace tyloi::cuda::qwen4_exp {

struct Model;

// Per-request recurrent and attention storage for one shared layer.
struct LayerState {
    tyloi::ops::Gdn::State gdn;
    std::unique_ptr<tyloi::ops::Qsa::State> qsa;
    Ple::State ple;
    bool retain_graph_storage = false, retained_prefix = false;

    explicit LayerState(const Layer& layer);
    void reset(const Layer& layer);
    void commit(const Layer& layer);
    void rollback(const Layer& layer, std::int64_t keep);
    std::vector<::tyloi::Tensor*> graph_state(const Layer& layer);
};

tyloi::models::qwen4_exp::LayerSnapshot capture_layer_session(
    const Layer& layer, const LayerState& state, std::int64_t position, std::size_t& bytes);
void validate_layer_session(const Layer&, const LayerState&,
    const tyloi::models::qwen4_exp::LayerSnapshot&, std::int64_t position);
void stage_layer_session(const Layer&, LayerState&,
    const tyloi::models::qwen4_exp::LayerSnapshot&, std::int64_t position);
std::size_t layer_session_growth(const Layer&, const LayerState&,
    const tyloi::models::qwen4_exp::LayerSnapshot&, std::int64_t position);
void stage_session_tensor(::tyloi::Tensor&, const ::tyloi::Tensor&, tyloi::Device);
void add_session_growth(std::vector<std::pair<tyloi::Device, std::size_t>>&,
    tyloi::Device, std::size_t);

// One request owns its layer caches, decode graph, positions and journal.
class State : public tyloi::models::State {
public:
    explicit State(const Model& model, bool initialize_layers = true);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) = delete;
    State& operator=(State&&) = delete;
    ~State() override;
    bool belongs_to(const Model& model) const noexcept { return &model_ == &model; }

    void preallocate() override {}
    std::vector<std::pair<tyloi::Device, std::size_t>> preallocate_growth() const override { return {}; }
    void reset(std::int64_t batch) override;
    void release_graphs() override;
    std::int64_t position() const noexcept override { return cache_pos; }
    bool speculating() const noexcept override { return speculative_start >= 0; }
    std::int64_t confirmed_prefix() const noexcept override { return speculative_confirmed; }
    tyloi::engine::Metrics metrics() const override;
    void advance(std::int64_t tokens) noexcept override { cache_pos += tokens; }
    void set_position_delta(std::int64_t delta) noexcept override { decode_position_delta = delta; }
    std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        const tyloi::models::Input&, const tyloi::models::ForwardOptions&) const override;
    std::vector<std::pair<tyloi::Device, std::size_t>> graph_exit_growth() const override;
    std::vector<::tyloi::Tensor> storage() const override;
    std::vector<tyloi::Tensor*> graph_state() override;

    bool supports_text_session_state() const override;
    bool supports_paged_text_session_state() const override;
    TextSessionStateKind text_session_state_kind() const;
    TextSessionState capture_text_session_state(const std::vector<std::int64_t>& tokens) const;
    std::unique_ptr<tyloi::models::Restore> prepare_restore(
        const TextSessionState&, const tyloi::models::Mtp* predictor_model = nullptr) override;
    void publish_restore(State& candidate) noexcept;

    // Speculative suffix journal; the layer records hold the checkpoints.
    void begin_speculative_suffix(std::int64_t draft_tokens) override;
    void commit_speculative() override;
    void rollback_speculative(std::int64_t accepted) override;
    bool retains_partial_prefix(std::int64_t accepted) const noexcept override {
        return speculative_prefix_retention && accepted >= 0;
    }
    void begin_forward();
    void finish_forward(const ::tyloi::Tensor& full_positions, std::int64_t batch,
                        std::int64_t tokens);

    std::int64_t cache_pos = 0;
    std::int64_t decode_position_delta = 0;
    std::int64_t batch = 0;
    std::int64_t speculative_start = -1;
    std::int64_t speculative_confirmed = 0;
    bool speculative_suffix_forward = false;
    bool speculative_suffix_journal = false;
    ::tyloi::Tensor positions;
    tyloi::cache::Media images;
    std::vector<std::unique_ptr<LayerState>> layers;
    std::unique_ptr<Qwen4DecodeGraph> graph_registry;
    bool reuse_decode_graphs = false;
    std::uint64_t decode_graph_captures = 0;
    std::int64_t decode_reserve_until = 0;
    bool speculative_prefix_retention = false;

private:
    friend tyloi::models::SnapshotSize snapshot_size(const State&, const tyloi::models::Mtp*);
    const Model& model_;
};

tyloi::models::SnapshotSize snapshot_size(const State&, const tyloi::models::Mtp*);

} // namespace tyloi::cuda::qwen4_exp

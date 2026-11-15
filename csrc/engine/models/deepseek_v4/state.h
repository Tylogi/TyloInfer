#pragma once

#include "engine/models/session/snapshot.h"
#include "engine/models/state.h"
#include <cstdint>
#include <vector>

namespace tyloi::cuda::deepseek_v4 {
using DenseTensor = ::tyloi::DenseTensor;
struct Model;

struct PoolState {
    int64_t ratio = 0, head_dim = 0, cache_quant_mode = 0, capacity = 0;
    bool overlap = false;
    DenseTensor state_kv, state_gate, previous_kv, previous_gate, pool;
    void reset(int64_t batch, int64_t max_positions);
};

struct LayerState { DenseTensor local_cache; PoolState compressor, indexer_compressor; };

class State : public tyloi::models::State {
public:
    explicit State(const Model&);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;
    void reset(int64_t batch);
    std::vector<std::pair<tyloi::Device, std::size_t>> reset_growth(int64_t batch) const override;
    std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        const tyloi::models::Input&, const tyloi::models::ForwardOptions&) const override;
    int64_t position() const noexcept { return cache_pos; }
    void advance(int64_t tokens) noexcept { cache_pos += tokens; }
    void set_position_delta(int64_t delta) noexcept { decode_position_delta = delta; }
    void preallocate() {}
    // This family allocates its request cache during reset or forward.
    std::vector<std::pair<tyloi::Device, std::size_t>> preallocate_growth() const override { return {}; }
    std::vector<::tyloi::Tensor> storage() const;
    std::vector<::tyloi::Tensor*> graph_state() { return {}; }
    TextSessionStateKind text_session_state_kind() const;
    bool supports_text_session_state() const;
    bool supports_paged_text_session_state() const { return false; }
    TextSessionState capture_text_session_state(const std::vector<int64_t>&) const;
    void restore_text_session_state(const TextSessionState&);
    std::unique_ptr<tyloi::models::Restore> prepare_restore(
        const TextSessionState&, const tyloi::models::Mtp* predictor_model = nullptr) override;

    std::vector<LayerState> layers;
    int64_t cache_pos = 0, decode_position_delta = 0, batch = 0;

private:
    class SessionRestore;
    const Model* model_;
    struct Layout { int device; int64_t head_dim, max_positions; };
    std::vector<Layout> layout_;
};
} // namespace tyloi::cuda::deepseek_v4

#pragma once

#include "engine/state/recurrent.h"
#include "engine/state/kv_view.h"
#include "engine/models/state.h"
#include "engine/models/session/snapshot.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace tyloi::cuda::nemotron_h {

struct Model;
struct Mamba;

struct MambaState {
    Recurrent mixer;
    ::tyloi::DenseTensor speculative_xbc, speculative_dt;
};

using LayerState = std::variant<std::monostate, MambaState, tyloi::state::KvView>;

class State
#ifdef TYLOINFER_NATIVE_CUDA
    : public tyloi::models::State
#endif
{
public:
    explicit State(const Model&);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;

    void reset(std::int64_t batch);
    std::int64_t position() const noexcept { return cache_pos; }
    void advance(std::int64_t tokens) noexcept { cache_pos += tokens; }
    void set_position_delta(std::int64_t delta) noexcept { decode_position_delta = delta; }
    void preallocate();
    void commit_speculative();
#ifdef TYLOINFER_NATIVE_CUDA
    std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        const tyloi::models::Input& input, const tyloi::models::ForwardOptions& options) const override;
    void rollback_speculative(std::int64_t accepted) override;
#endif
    std::vector<::tyloi::Tensor> storage() const;
    std::vector<::tyloi::Tensor*> graph_state();
    TextSessionStateKind text_session_state_kind() const;
    bool supports_text_session_state() const;
    bool supports_paged_text_session_state() const;
    TextSessionState capture_text_session_state(const std::vector<std::int64_t>&) const;
    void restore_text_session_state(const TextSessionState&);

    struct MambaLayout {
        std::int64_t heads, head_dim, state_size, groups, conv_kernel;
#ifdef TYLOINFER_NATIVE_CUDA
        const Mamba* weights;
#endif
    };
    struct FullLayout {
        std::int64_t heads, head_dim, capacity;
        bool sliding;
        ::tyloi::ScalarType dtype;
        tyloi::state::KvQuantization quantization;
    };
    struct Layout {
        int device;
        std::variant<std::monostate, MambaLayout, FullLayout> mixer;
    };
    const std::vector<Layout>& layout() const noexcept { return layout_; }

    std::vector<LayerState> layers;
#ifndef TYLOINFER_NATIVE_CUDA
    std::unique_ptr<tyloi::models::MtpState> predictor;
#endif
    std::int64_t cache_pos = 0, decode_position_delta = 0, batch = 0;
    std::int64_t speculative_start = -1, speculative_confirmed = 0;

private:
    std::vector<Layout> layout_;
};

} // namespace tyloi::cuda::nemotron_h

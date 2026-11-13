#pragma once

#include "engine/models/common/attention_ops.h"
#include "engine/state/sequence.h"
#include "engine/models/state.h"
#include "engine/models/session/snapshot.h"
#include <cstdint>
#include <variant>
#include <vector>

namespace tyloi::cuda::glm5_next {

using DenseTensor = ::tyloi::DenseTensor;
struct Model;

struct KdaState {
    DenseTensor conv, recurrent, rollback_conv, rollback_recurrent;
    void reset() { conv = DenseTensor(); recurrent = DenseTensor(); commit(); }
    bool pending() const { return rollback_conv.defined(); }
    void commit() { rollback_conv = DenseTensor(); rollback_recurrent = DenseTensor(); }
    void rollback() {
        TYLOINFER_RUNTIME_CHECK(pending(), "GLM KDA has no speculative checkpoint");
        conv = rollback_conv;
        recurrent = rollback_recurrent;
        commit();
    }
};

struct MlaState {
    ::tyloi::state::Sequence latent, index;
    MlaState(int64_t maximum, int64_t width, int64_t index_width)
        : latent(maximum, width), index(maximum, 2 * index_width) {}
    void reset() { latent.reset(); index.reset(); }
    int64_t position() const { return latent.position(); }
    void truncate(int64_t keep) {
        TYLOINFER_RUNTIME_CHECK(keep >= 0 && keep <= latent.position() && keep <= index.position(),
                               "invalid Flash-Next MLA cache truncation");
        latent.truncate(keep);
        index.truncate(keep);
    }
};

using LayerState = std::variant<KdaState, MlaState>;

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
    void reset(int64_t batch);
    int64_t position() const noexcept { return cache_pos; }
    void advance(int64_t tokens) noexcept { cache_pos += tokens; }
    void set_position_delta(int64_t delta) noexcept { decode_position_delta = delta; }
    void preallocate() {}
#ifdef TYLOINFER_NATIVE_CUDA
    void commit_speculative() override;
    std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        const tyloi::models::Input&, const tyloi::models::ForwardOptions&) const override;
    std::vector<::tyloi::Tensor> storage() const override;
    std::vector<::tyloi::Tensor*> graph_state() override { return {}; }
    void rollback_speculative(std::int64_t accepted) override;
#else
    void commit_speculative();
    std::vector<DenseTensor> storage() const { return {}; }
    std::vector<DenseTensor*> graph_state() { return {}; }
#endif
    TextSessionStateKind text_session_state_kind() const { return TextSessionStateKind::Unsupported; }
    bool supports_text_session_state() const { return false; }
    bool supports_paged_text_session_state() const { return false; }
    TextSessionState capture_text_session_state(const std::vector<int64_t>&) const;
    void restore_text_session_state(const TextSessionState&);

    std::vector<LayerState> layers;
#ifndef TYLOINFER_NATIVE_CUDA
    std::unique_ptr<tyloi::models::MtpState> predictor;
#endif
    int64_t cache_pos = 0, decode_position_delta = 0, batch = 0;
    int64_t speculative_start = -1, speculative_confirmed = 0;
#ifdef TYLOINFER_NATIVE_CUDA
private:
    std::vector<int> devices;
    std::int64_t maximum_, kda_heads_, kda_width_, kda_kernel_;
#endif
};

} // namespace tyloi::cuda::glm5_next

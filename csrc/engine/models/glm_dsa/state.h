#pragma once

#include "engine/models/session/snapshot.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/models/state.h"
#endif
#include <cstdint>
#include <map>
#include <vector>

namespace tyloi::cuda::glm_dsa {
using DenseTensor = ::tyloi::DenseTensor;
struct Model;

struct LayerState { DenseTensor kv_cache, index_cache; };
struct Selection { DenseTensor topk_indices; int64_t dense_prefix_rows = 0; };

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
    // Plans ordinary forward allocations; other execution paths plan separately.
    std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        const tyloi::models::Input& input, const tyloi::models::ForwardOptions& options) const override;
    std::vector<::tyloi::Tensor> storage() const;
    std::vector<::tyloi::Tensor*> graph_state() { return {}; }
#else
    std::vector<DenseTensor> storage() const;
    std::vector<DenseTensor*> graph_state() { return {}; }
#endif
    TextSessionStateKind text_session_state_kind() const;
    bool supports_text_session_state() const;
    bool supports_paged_text_session_state() const { return false; }
    TextSessionState capture_text_session_state(const std::vector<int64_t>&) const;
    void restore_text_session_state(const TextSessionState&);

    std::vector<LayerState> layers;
    std::map<int, Selection> selections;
    int64_t cache_pos = 0, decode_position_delta = 0, batch = 0;

private:
    struct Layout {
        int device;
        bool full_indexer;
        int64_t capacity, index_head_dim, index_topk;
    };
    std::vector<Layout> layout_;
};
} // namespace tyloi::cuda::glm_dsa

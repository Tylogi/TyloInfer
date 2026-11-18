#pragma once

#include "engine/generation/sequence.h"
#include "engine/models/input.h"

#include <vector>

namespace tyloi::models::qwen35 {

class State;
struct Model;

struct ForwardRow {
    State* state = nullptr;
    ::tyloi::models::Input input;
};

// Requires prefilled, single-device CUDA state and stable GPU position and length tensors.
// Leaves cache_pos unchanged; the caller owns capture teardown and state restoration.
::tyloi::models::Decoded capture_forward(
    const Model& model, State& state, ::tyloi::models::Input input,
    const ::tyloi::models::ForwardOptions& options);

#if TYLOINFER_HAS_CUDA
bool replay_graph(State& state, std::int64_t retained);
#endif

::tyloi::StepSequence<std::vector<::tyloi::models::Decoded>> forward_batch(
    const Model& model, std::vector<ForwardRow> rows,
    const ::tyloi::models::ForwardOptions& options, bool advance = true);

} // namespace tyloi::models::qwen35

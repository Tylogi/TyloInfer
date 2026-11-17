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

::tyloi::StepSequence<std::vector<::tyloi::models::Decoded>> forward_batch(
    const Model& model, std::vector<ForwardRow> rows,
    const ::tyloi::models::ForwardOptions& options, bool advance = true);

} // namespace tyloi::models::qwen35

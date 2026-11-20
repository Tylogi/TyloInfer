#pragma once

#include "engine/models/input.h"
#include "engine/models/metadata.h"

namespace tyloi { class Executor; }
namespace tyloi::weights { class Cache; }
namespace tyloi::models { class CausalModel; class State; }
namespace tyloi::engine { class Budget; class InferenceOutput; }

namespace tyloi::engine {

/** Retire unmanaged graph storage before planning ordinary forward growth. */
void exit_unmanaged_graph(models::State& state, weights::Cache& weights, Budget& budget);

/** Select a whole prompt when the model can process it layer by layer. */
std::int64_t prefill_span(const models::CausalModel& model, const models::State& state,
    const weights::Cache& weights, const Budget& budget, std::int64_t tokens, std::int64_t compute,
    std::size_t reused, std::size_t stable, bool transformed);

/** Drive a model's prefill job or its ordinary forward path with allocation reservations. */
models::Prefill prefill(const models::CausalModel& model, models::State& state, Executor& executor,
    weights::Cache& weights, Budget& budget, const models::ForwardOptions& options,
    models::Input input, std::int64_t compute, InferenceOutput& output, bool raw_host = false);

} // namespace tyloi::engine

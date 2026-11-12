#pragma once

#include "engine/generation/step.h"
#include "engine/models/input.h"

namespace tyloi {
class Executor;
class ModelSource;
namespace weights { class Cache; }
namespace models { class CausalModel; class State; }
namespace engine {
class Budget;
namespace mtp { struct State; }
Generation generate_mtp(const models::CausalModel& model, models::State& state,
    ::tyloi::Executor& executor, ::tyloi::Executor& predictor_execution,
    weights::Cache& weights, Budget& budget, const ModelSource* source,
    mtp::State& mtp_state, InferenceRequest& request, InferenceOutput& output,
    std::int64_t prefill_chunk_size, const models::Prepared* prepared,
    std::size_t reused_tokens, Tensor restored_last_hidden, Tensor* session_last_hidden,
    std::size_t stable_prefix_tokens, Device device, models::ForwardOptions options, bool profile);
}
}

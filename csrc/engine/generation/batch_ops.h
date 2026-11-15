#pragma once

#include <memory>

namespace tyloi { class Executor; }
namespace tyloi::weights { class Cache; }
namespace tyloi::models { class CausalModel; struct ForwardOptions; }
namespace tyloi::engine {
class Batch;
class Budget;
struct Config;

/** Construct the shared scheduler around model-owned batch computation. */
std::unique_ptr<Batch> make_batch(models::CausalModel& model, Executor& executor,
    weights::Cache& weights, Budget& budget, const Config& config, const models::ForwardOptions& forward);
} // namespace tyloi::engine

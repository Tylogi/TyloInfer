#pragma once

#include "engine/models/batch.h"

namespace tyloi { class Executor; }
namespace tyloi::weights { class Cache; }
namespace tyloi::engine {
class Budget;

/** Reserve the combined per-device backing for a batch operation. */
void reserve_batch(Budget& budget, Device device, const models::BatchGrowth& values,
    std::size_t margin, bool primary = false);

/** Submit numerical outputs together with request state retained for later steps. */
void submit_batch(Executor& executor, std::span<const Tensor> outputs,
    std::span<models::BatchState* const> states);

/** Submit decoded batch outputs while retaining all published tensor fields. */
void submit_batch(Executor& executor, std::span<const models::Decoded> outputs,
    std::span<models::BatchState* const> states);

/** Drive model batch prefill events using the executor's working-set owner. */
std::vector<models::Decoded> prefill_batch(models::Batch& batch, Executor& executor, weights::Cache& weights,
    Budget& budget, Device device, std::vector<models::BatchState*> states, std::vector<Tensor> ids,
    bool verify = false);

/** Drive a batch graph plan through the shared capture coordinator. */
std::unique_ptr<models::Batch::Plan> plan_batch(models::Batch& batch, Executor& executor,
    weights::Cache& weights, Budget& budget, Device device, const std::vector<models::BatchRow>& rows, bool greedy);

/** Drive admitted graph capture and its resource events. */
std::optional<Tensor> capture_batch(models::Batch& batch, Executor& executor,
    weights::Cache& weights, Budget& budget, Device device, const std::vector<models::BatchRow>& rows,
    const Tensor& ids, const Tensor& host_ids, bool greedy, const models::Batch::Plan& plan);

}

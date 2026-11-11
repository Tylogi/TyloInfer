#pragma once

#include "engine/weights/moe/batch.h"
#include "engine/backends/cuda/executor/tq_moe.h"
#include "engine/backends/cuda/ops/ffn/routes.h"
#include "engine/weights/encoding/mfe.h"

namespace tyloi::weights {

struct ExpertBatch::Impl {
    std::vector<Tensor> gate_up_weights;
    Tensor down_weight, ids;
    TqMoeBanks tq;
    std::shared_ptr<const MfeTensor> gate, up, gate_up, down;
    std::shared_ptr<const MfeTensor> cpu_gate, cpu_up, cpu_gate_up, cpu_down;
    std::shared_ptr<const DenseTensor> dense_gate, dense_up, dense_down;
    MoeRoutePlan route;
    bool split = false, sharded = false, bundle_prefetched = false;

    void ready();
    std::shared_ptr<const MfeTensor> stage(const std::shared_ptr<const MfeTensor>& source,
                                           const char* label);
private:
    bool prefetch_bundle();
};

} // namespace tyloi::weights

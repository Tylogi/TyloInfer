#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/common/attention.h"
#include "engine/models/deepseek_v4/model.h"
#include "engine/models/deepseek_v4/state.h"
#include "engine/models/input.h"

extern bool g_dsv4_fused_hc;
extern bool g_dsv4_compare_hc_ops;

namespace tyloi::cuda::deepseek_v4 {
DenseTensor forward_layer(const Layer&, LayerState&, Executor&, DenseTensor input, DenseTensor token_ids,
                     const ::tyloi::models::AttentionStep&, const Executor::Mode&);
} // namespace tyloi::cuda::deepseek_v4

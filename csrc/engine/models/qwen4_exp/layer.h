#pragma once

// Read-only weights and configuration shared by one Qwen4 layer.

#include "engine/models/qwen4_exp/model.h"
#include "engine/models/common/moe.h"
#include "engine/ops/ffn.h"
#include "engine/ops/gdn.h"
#include "engine/ops/qsa.h"
#include "engine/ops/residual.h"
#include "engine/models/qwen4_exp/config.h"

#include <cstdint>
#include <memory>

namespace tyloi::models::qwen4_exp {
struct LayerState;

struct Layer final {
    // Placement, as the layer loop sets it for CPU-offloaded layers.
    ::tyloi::Device device;
    tyloi::models::qwen4_exp::Config config;
    tyloi::ops::Residual attention_gr, ffn_gr;
    tyloi::models::Moe ffn;
    std::unique_ptr<tyloi::ops::Gdn> gdn;
    std::unique_ptr<tyloi::ops::Qsa> qsa;
    std::unique_ptr<Ple> ple;
};

Tensor feed_forward(const Moe& weights, const Tensor& input,
                    std::int64_t projection_rows = 0);

Tensor forward(const Layer& weights, LayerState& request,
                    Tensor x, const Tensor& ids,
                    const Tensor& positions, const Tensor& full_positions,
                    int64_t confirmed = 0, const Tensor& host_ids = {});

Tensor prefill_ple(const Layer& weights, LayerState& request,
                        Tensor x, const Tensor& ids, const Tensor& host_ids);

Tensor prefill_attention_core(const Layer& weights, LayerState& request,
                                   const Tensor& input, const Tensor& positions,
                                   const Tensor& full_positions);

/** Prepare the attention and residual inputs used by layer-major prefill. */
tyloi::ops::Residual::Result prefill_attention(const Layer& weights, LayerState& request,
                                        Tensor x, const Tensor& ids,
                                        const Tensor& host_ids, const Tensor& positions,
                                        const Tensor& full_positions);

} // namespace tyloi::models::qwen4_exp

#pragma once

#include "engine/generation/mtp/ops.h"
#include "engine/models/deepseek_v41/generation.h"
#include "engine/weights/cache.h"

namespace tyloi::cuda::deepseek_v41 {
using Model = deepseek_v41::Model;

template<class Ops, class Chunk, class Tensor, class Timer, class Finish>
std::optional<typename Ops::PrefillResult> mtp_prefill(Model& model, Ops& ops, const Chunk& chunk,
                                                      Tensor& ids, Tensor& raw, Timer& timer, const Finish& finish_prefill) {
    if(!ops.transformed_prompt) {
        auto hidden = ced_prefill(model, ops, ids, true, &raw);
        finish_prefill();
        return typename Ops::PrefillResult{std::move(hidden),std::move(raw),timer.elapsed_ms()};
    }
    return std::nullopt;
}

} // namespace tyloi::cuda::deepseek_v41

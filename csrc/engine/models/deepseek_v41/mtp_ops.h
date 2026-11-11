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
#ifdef TYLOINFER_NATIVE_CUDA
        auto hidden = ced_prefill(model, ops, ids, true, &raw);
#else
        ::tyloi::Tensor raw_result;
        auto hidden = ced_prefill(model, ops, ids, true, &raw_result);
        if (raw_result) raw = std::move(raw_result.dense());
#endif
        finish_prefill();
#ifdef TYLOINFER_NATIVE_CUDA
        return typename Ops::PrefillResult{std::move(hidden),std::move(raw),timer.elapsed_ms()};
#else
        return typename Ops::PrefillResult{hidden ? std::move(hidden.dense()) : Tensor{},
                                           std::move(raw),timer.elapsed_ms()};
#endif
    }
    return std::nullopt;
}

} // namespace tyloi::cuda::deepseek_v41

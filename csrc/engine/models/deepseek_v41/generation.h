#pragma once

#include "engine/generation/ops.h"
#include "engine/budget.h"
#include "engine/models/deepseek_v41/ops.h"
#include "engine/backends/cuda/executor/decode_graph.h"
#include "engine/requests/output.h"
#include "engine/weights/cache.h"

#include <algorithm>
#include <utility>

namespace tyloi::cuda::deepseek_v41 {
using Model = deepseek_v41::Model;

template<class Ops>
::tyloi::Tensor ced_prefill(Model& model, Ops& ops, const ::tyloi::Tensor& ids,
                            bool capture_raw_hidden, ::tyloi::Tensor* raw_hidden = nullptr) {
    const auto old_batch = ops.state.batch;
    const auto old_position = ops.state.cache_pos;
    auto input = deepseek_v41_prefill_input(model, ops.state, ids, ops.prefill_compute_chunk);
    const auto batch = input.size(0), tokens = input.size(1);
    if (old_batch != batch || old_position == 0) {
#ifdef TYLOINFER_NATIVE_CUDA
        if (ops.budget) ops.budget->reserve(ops.state.allocation_growth());
#endif
        ops.state.allocate();
    }
    const auto split = model.config.causal_encoder_layers;
    const auto compute_chunk = std::min<int64_t>(ops.prefill_compute_chunk, 8192);
    CedEncoding encoded{input.dense(), {}, {}, {}, {}, batch, ops.state.cache_pos,
                        ops.state.cache_pos + tokens, tokens, compute_chunk};
    std::optional<tyloi::weights::Cache::Priority> priority;
    if (tokens > model.config.sliding_window)
        priority.emplace(ops.weights, 0, static_cast<int>(split));
    const auto stopped = [&] { return ops.output && ops.output->stopped(); };
    for (int64_t offset = 0; offset < tokens; offset += compute_chunk) {
        if (stopped()) { ops.state.reset(batch); return {}; }
        const auto count = std::min(compute_chunk, tokens - offset);
#ifdef TYLOINFER_NATIVE_CUDA
        if (ops.budget) ops.budget->reserve(ops.state.ced_growth(ops.state.cache_pos, count, 0, split));
#endif
        if (!deepseek_v41_prefill_encode_chunk(model, ops.state, encoded, offset, count,
                                               stopped, capture_raw_hidden)) return {};
    }
    if (priority) priority->finish();
    priority.reset();
    if (stopped()) { ops.state.reset(batch); return {}; }
    const auto replay_tokens = encoded.tail_ids.size(1);
#ifdef TYLOINFER_NATIVE_CUDA
    if (ops.budget) ops.budget->reserve(ops.state.ced_growth(
        encoded.end - replay_tokens, replay_tokens, split, model.num_hidden_layers()));
#endif
    return deepseek_v41_prefill_decode(model, ops.state, std::move(encoded), stopped, raw_hidden);
}

template<class Ops, class Finish>
#ifdef TYLOINFER_NATIVE_CUDA
std::tuple<::tyloi::Tensor, bool, bool> generation_prefill(
        Model& model, Ops& ops, const tyloi::engine::PrefillChunk&,
        ::tyloi::Tensor& ids, const Finish&) {
    ::tyloi::Tensor hidden;
#else
std::tuple<::tyloi::DenseTensor, bool, bool> generation_prefill(
        Model& model, Ops& ops, const tyloi::engine::PrefillChunk&,
        ::tyloi::DenseTensor& ids, const Finish&) {
    ::tyloi::DenseTensor hidden;
#endif
    if(!ops.prepared || !ops.prepared->transformed()) {
        auto result = ced_prefill(model, ops, ids, false);
#ifdef TYLOINFER_NATIVE_CUDA
        if(result) hidden=std::move(result);
#else
        if(result) hidden=result.dense();
#endif
    }
    return {std::move(hidden),false,false};
}

inline bool generation_prefill_begin(Model&, DecodeGraphCache& graph, bool streamed) {
    if(streamed)graph.invalidate();
    return streamed;
}

} // namespace tyloi::cuda::deepseek_v41

#include "engine/models/qwen4_exp/layer.h"
#include "engine/models/qwen4_exp/state.h"

#include <utility>

namespace tyloi::cuda::qwen4_exp {

DenseTensor forward(const Layer& weights, LayerState& request,
                    DenseTensor x, const DenseTensor& ids,
                    const DenseTensor& positions, const DenseTensor& full_positions,
                    int64_t confirmed, const DenseTensor& host_ids) {
    request.retained_prefix = confirmed > 0 && ids.size(1) <= 8;
    const auto device = weights.device;
    if (weights.ple) {
        ::tyloi::ops::ProfileRange range("prefill.ple", device);
        auto ple = weights.ple->forward(request.ple, x, ids, true, confirmed, host_ids,
                                        full_positions.size(-1) - ids.size(1));
        range.finish();
        x = x.is_cuda() ? tyloi::ops::add(x, ple).dense() : x + ple;
    }
    tyloi::ops::Residual::Result attention_mix;
    {
        ::tyloi::ops::ProfileRange range("prefill.gr_attention", device);
        attention_mix = weights.attention_gr.pre(x);
        range.finish();
    }
    DenseTensor attention_output;
    {
        ::tyloi::ops::ProfileRange range(weights.gdn ? "prefill.gdn" : "prefill.qsa", device);
        attention_output = weights.gdn
            ? weights.gdn->forward(request.gdn, attention_mix.mixed, true, confirmed,
                                   full_positions.size(-1) - ids.size(1)).dense()
            : weights.qsa->forward(*request.qsa, attention_mix.mixed, positions, full_positions, true).dense();
        range.finish();
    }
    tyloi::ops::Residual::Result ffn_mix;
    {
        ::tyloi::ops::ProfileRange range("prefill.gr_ffn_pre", device);
        ffn_mix = weights.ffn_gr.pre_after(attention_output, attention_mix.residual, attention_mix.gate);
        range.finish();
    }
    attention_output = {};
    DenseTensor ffn_output;
    {
        ::tyloi::ops::ProfileRange range("prefill.ffn", device);
        ffn_output = tyloi::ops::ffn(weights.ffn, ffn_mix.mixed).dense();
        range.finish();
    }
    ::tyloi::ops::ProfileRange range("prefill.gr_ffn_post", device);
    auto output = weights.ffn_gr.post(ffn_output, ffn_mix.residual, ffn_mix.gate).dense();
    range.finish();
    return output;
}

DenseTensor prefill_ple(const Layer& weights, LayerState& request,
                        DenseTensor x, const DenseTensor& ids, const DenseTensor& host_ids) {
    if (weights.ple) {
        ::tyloi::ops::ProfileRange range("prefill.ple", weights.device);
        auto ple = weights.ple->forward(request.ple, x, ids, true, 0, host_ids);
        range.finish();
        x = x.is_cuda() ? tyloi::ops::add(x, ple).dense() : x + ple;
    }
    return x;
}

DenseTensor prefill_attention_core(const Layer& weights, LayerState& request,
                                   const DenseTensor& input, const DenseTensor& positions,
                                   const DenseTensor& full_positions) {
    ::tyloi::ops::ProfileRange range(weights.gdn ? "prefill.gdn" : "prefill.qsa",
                                     weights.device);
    auto output = weights.gdn
        ? weights.gdn->forward(request.gdn, input, true, 0).dense()
        : weights.qsa->forward(*request.qsa, input, positions, full_positions, true).dense();
    range.finish();
    return output;
}

tyloi::ops::Residual::Result prefill_attention(const Layer& weights, LayerState& request,
                                        DenseTensor x, const DenseTensor& ids,
                                        const DenseTensor& host_ids, const DenseTensor& positions,
                                        const DenseTensor& full_positions) {
    const auto device = weights.device;
    x = prefill_ple(weights, request, std::move(x), ids, host_ids);
    tyloi::ops::Residual::Result mix;
    {
        ::tyloi::ops::ProfileRange range("prefill.gr_attention", device);
        mix = weights.attention_gr.pre(x);
        range.finish();
    }
    auto attention_output = prefill_attention_core(weights, request,
                                                    mix.mixed.dense(), positions, full_positions);
    ::tyloi::ops::ProfileRange range("prefill.gr_ffn_pre", device);
    auto output = weights.ffn_gr.pre_after(attention_output, mix.residual, mix.gate);
    range.finish();
    return output;
}

} // namespace tyloi::cuda::qwen4_exp

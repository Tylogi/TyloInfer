#include "engine/models/qwen4_exp/layer.h"
#include "engine/models/qwen4_exp/state.h"
#include "engine/weights/moe/batch.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::models::qwen4_exp {

Tensor feed_forward(const Moe& weights, const Tensor& input, std::int64_t projection_rows,
                    std::int64_t prefill_offset, std::int64_t prefill_rows) {
    if (!input || input.dim() < 2 || weights.gate_up.empty() ||
        !weights.router || input.size(-1) != weights.router.size(1))
        throw std::invalid_argument("invalid Qwen4 FFN input");
    std::vector<std::int64_t> shape;
    for (std::int64_t axis = 0; axis < input.dim(); ++axis) shape.push_back(input.size(axis));
    auto source = ops::contiguous(ops::cast(ops::reshape(input, {-1, input.size(-1)}),
        ops::Ffn::input_type(weights.gate_up.front(), input.scalar_type())));
    auto logits = ops::Ffn::project(source, weights.router, projection_rows);
    auto selected = ops::topk(logits, weights.top_k, weights.routing);
    tyloi::weights::ExpertBatch batch(weights.gate_up, weights.down, selected.ids,
        prefill_offset, prefill_rows ? prefill_rows : source.size(0), weights.activation, prefill_rows != 0);
    const ops::ExpertSpec spec{weights.activation, weights.swiglu_limit, weights.layer};
    auto output = ops::Ffn::routed_shared(source, selected.weights, batch,
        weights.shared.gate_up, weights.shared.down, source, weights.shared_gate,
        spec, projection_rows);
    return ops::reshape(output, shape);
}

Tensor forward(const Layer& weights, LayerState& request,
                    Tensor x, const Tensor& ids,
                    const Tensor& positions, const Tensor& full_positions,
                    int64_t confirmed, const Tensor& host_ids,
                    std::int64_t prefill_offset, std::int64_t prefill_rows) {
    request.retained_prefix = confirmed > 0 && ids.size(1) <= 8;
    const auto device = weights.device;
    if (weights.ple) {
        ::tyloi::ops::ProfileRange range("prefill.ple", device);
        auto ple = weights.ple->forward(request.ple, x, ids, true, confirmed, host_ids,
                                        full_positions.size(-1) - ids.size(1));
        range.finish();
        x = tyloi::ops::add(x, ple);
    }
    tyloi::ops::Residual::Result attention_mix;
    {
        ::tyloi::ops::ProfileRange range("prefill.gr_attention", device);
        attention_mix = weights.attention_gr.pre(x);
        range.finish();
    }
    Tensor attention_output;
    {
        ::tyloi::ops::ProfileRange range(weights.gdn ? "prefill.gdn" : "prefill.qsa", device);
        attention_output = weights.gdn
            ? weights.gdn->forward(request.gdn, attention_mix.mixed, true, confirmed,
                                   full_positions.size(-1) - ids.size(1))
            : weights.qsa->forward(*request.qsa, attention_mix.mixed, positions, full_positions, true);
        range.finish();
    }
    tyloi::ops::Residual::Result ffn_mix;
    {
        ::tyloi::ops::ProfileRange range("prefill.gr_ffn_pre", device);
        ffn_mix = weights.ffn_gr.pre_after(attention_output, attention_mix.residual, attention_mix.gate);
        range.finish();
    }
    attention_output = {};
    Tensor ffn_output;
    {
        ::tyloi::ops::ProfileRange range("prefill.ffn", device);
        ffn_output = feed_forward(weights.ffn, ffn_mix.mixed, 0, prefill_offset, prefill_rows);
        range.finish();
    }
    ::tyloi::ops::ProfileRange range("prefill.gr_ffn_post", device);
    auto output = weights.ffn_gr.post(ffn_output, ffn_mix.residual, ffn_mix.gate);
    range.finish();
    return output;
}

Tensor prefill_ple(const Layer& weights, LayerState& request,
                        Tensor x, const Tensor& ids, const Tensor& host_ids) {
    if (weights.ple) {
        ::tyloi::ops::ProfileRange range("prefill.ple", weights.device);
        auto ple = weights.ple->forward(request.ple, x, ids, true, 0, host_ids);
        range.finish();
        x = tyloi::ops::add(x, ple);
    }
    return x;
}

Tensor prefill_attention_core(const Layer& weights, LayerState& request,
                                   const Tensor& input, const Tensor& positions,
                                   const Tensor& full_positions) {
    ::tyloi::ops::ProfileRange range(weights.gdn ? "prefill.gdn" : "prefill.qsa",
                                     weights.device);
    auto output = weights.gdn
        ? weights.gdn->forward(request.gdn, input, true, 0)
        : weights.qsa->forward(*request.qsa, input, positions, full_positions, true);
    range.finish();
    return output;
}

tyloi::ops::Residual::Result prefill_attention(const Layer& weights, LayerState& request,
                                        Tensor x, const Tensor& ids,
                                        const Tensor& host_ids, const Tensor& positions,
                                        const Tensor& full_positions) {
    const auto device = weights.device;
    x = prefill_ple(weights, request, std::move(x), ids, host_ids);
    tyloi::ops::Residual::Result mix;
    {
        ::tyloi::ops::ProfileRange range("prefill.gr_attention", device);
        mix = weights.attention_gr.pre(x);
        range.finish();
    }
    auto attention_output = prefill_attention_core(weights, request,
                                                    mix.mixed, positions, full_positions);
    ::tyloi::ops::ProfileRange range("prefill.gr_ffn_pre", device);
    auto output = weights.ffn_gr.pre_after(attention_output, mix.residual, mix.gate);
    range.finish();
    return output;
}

} // namespace tyloi::models::qwen4_exp

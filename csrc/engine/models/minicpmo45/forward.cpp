#include "engine/models/minicpmo45/forward.h"
#include "engine/models/common/ffn.h"
#include "engine/models/common/check.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#endif

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::minicpmo45 {

namespace {
Tensor token_ids(int primary, const Tensor& ids) {
    tyloi::models::require_model(ids.defined() && (ids.dim() == 1 || ids.dim() == 2),
        "token IDs must have shape [tokens] or [batch,tokens]");
    auto shaped = ids.dim() == 1 ? tyloi::ops::reshape(ids, {1, ids.size(0)}) : ids;
    tyloi::models::require_model(shaped.size(0) > 0 && shaped.size(1) > 0, "token IDs must not be empty");
    return tyloi::ops::contiguous(tyloi::ops::copy(tyloi::ops::cast(shaped, tyloi::kInt64),
        tyloi::Device(tyloi::kCUDA, primary)));
}
}

Tensor Model::embedding(const Tensor& ids, int primary) const {
    auto normalized = token_ids(primary, ids);
    return tyloi::ops::measure("model.embed", normalized.device(), [&] {
        return tyloi::ops::embedding(normalized, embed);
    });
}
Tensor Model::normalize(const Tensor& hidden, int64_t batch, int64_t tokens) const {
    auto shaped = tyloi::ops::reshape(hidden, {batch, tokens, metadata.hidden_size});
    return tyloi::ops::measure("model.output_norm", shaped.device(), [&] {
        return tyloi::ops::rms_norm(tyloi::ops::cast(shaped, tyloi::kFloat32), output_norm,
                                    metadata.rms_norm_eps, 1.0);
    });
}
Tensor Model::softcap(const Tensor& value) const { return value; }
Tensor Model::logits(const Tensor& hidden, bool) const {
    return tyloi::ops::measure("model.lm_head", hidden.device(), [&] {
        return tyloi::ops::project(hidden, lm_head);
    });
}
tyloi::StepSequence<tyloi::models::Prepared> Model::prepare(
        const std::vector<std::int64_t>&, const MfqMultimodalInput&, tyloi::models::State&) const {
    throw std::invalid_argument("model has no multimodal component");
}
Tensor Model::forward(tyloi::models::Input input, tyloi::models::State& state_value,
                      const tyloi::models::ForwardOptions& mode) const {
    if (input.suffix) throw std::invalid_argument("MiniCPM TTS does not support speculative suffix forwarding");
    auto& state = dynamic_cast<State&>(state_value);
    auto ids = token_ids(mode.primary, input.ids);
    const auto batch = ids.size(0), tokens = ids.size(1);
    tyloi::models::require_model(input.confirmed == 0, "MiniCPM TTS has no speculative predictor");
    tyloi::models::require_model(state.layers.size() == layers.size(), "MiniCPM TTS layer state mismatch");
    tyloi::models::require_model(state.cache_pos <= metadata.max_position_embeddings - tokens,
        "MiniCPM TTS history exceeds context capacity");
    if (state.cache_pos == 0) state.reset(batch);
    tyloi::models::require_model(state.batch == batch, "MiniCPM TTS cache batch does not match inputs");
    auto hidden = input.embeddings ? input.embeddings : embedding(ids, mode.primary);
    tyloi::models::require_model(hidden.dim() == 3 && hidden.size(0) == batch && hidden.size(1) == tokens &&
        hidden.size(2) == metadata.hidden_size, "inputs_embeds must match [batch,tokens,hidden_size]");
    const tyloi::Device primary(tyloi::kCUDA, mode.primary);
    const auto copy_position = [&](const Tensor& value) {
        return tyloi::ops::contiguous(tyloi::ops::copy(tyloi::ops::cast(value, tyloi::kInt64), primary));
    };
    auto physical = input.cache_positions ? copy_position(input.cache_positions)
        : tyloi::ops::arange(state.cache_pos, state.cache_pos + tokens, primary);
    tyloi::models::require_model((physical.dim() == 1 && physical.numel() == tokens) ||
        (physical.dim() == 2 && physical.size(0) == batch && physical.size(1) == tokens),
        "cache_positions must have shape [tokens] or [batch,tokens]");
    auto positions = input.positions ? copy_position(input.positions)
        : (state.decode_position_delta ? tyloi::ops::add(physical, state.decode_position_delta) : physical);
    tyloi::models::require_model((positions.dim() == 1 && positions.numel() == tokens) ||
        (positions.dim() == 2 && positions.size(0) == batch && positions.size(1) == tokens),
        "position_ids must have shape [tokens] or [batch,tokens]");
    if (input.lengths) {
        tyloi::models::require_model(input.lengths.dim() == 1 && input.lengths.size(0) == batch,
            "sequence lengths must have shape [batch]");
        input.lengths = copy_position(input.lengths);
    } else if (state.cache_pos > 0 && tokens == 1) {
        input.lengths = tyloi::ops::full(std::array<int64_t, 1>{batch}, state.cache_pos + 1, primary);
    }
    hidden = tyloi::ops::contiguous(tyloi::ops::copy(hidden, primary));
    const auto trace = [&] {
        if (input.trace) input.trace->push_back(tyloi::ops::clone(tyloi::ops::contiguous(
            tyloi::ops::cast(tyloi::ops::copy(hidden, primary), tyloi::kFloat32))));
    };
    trace();
    for (size_t index = 0; index < layers.size(); ++index) {
        const auto& layer = layers[index];
        const auto& attention = layer.attention;
        const tyloi::Device device(layer.cpu ? tyloi::kCPU : tyloi::kCUDA, layer.cpu ? -1 : layer.device);
        const auto local = [&](const Tensor& value) {
            return tyloi::ops::contiguous(tyloi::ops::copy(value, device));
        };
        hidden = local(hidden);
        ::tyloi::models::AttentionStep step{local(positions), {}, {}, state.cache_pos, input.kv_length, input.attention_parts};
        if (input.lengths) step.sequence_lengths = local(input.lengths);
        if (input.cache_positions) step.cache_positions = local(physical);
        auto& cache = state.layers[index];
        if (!cache.defined()) {
            const auto capacity = mode.kv_capacity > 0
                ? std::max(state.cache_pos + tokens, mode.kv_capacity) : attention.max_position_embeddings;
            cache = tyloi::state::allocate_kv(batch, attention.kv_heads, capacity, attention.attention_head_dim,
                                               false, device, tyloi::kFloat16);
        }
        const auto* rope = layer.cpu ? &cpu_rope : &this->rope;
        if (!layer.cpu) {
            const auto found = device_ropes.find(layer.device);
            if (found != device_ropes.end()) rope = &found->second;
        }
        auto* layer_trace = mode.trace_layer == attention.layer ? mode.trace : nullptr;
        auto output = attention.forward(hidden, step, cache, *rope, layer_trace);
        hidden = tyloi::models::full_ffn(layer.ffn, layer.ffn_norm, output[0], output[1],
            attention.rms_norm_eps, attention.norm_weight_offset,
            attention.official_bf16, attention.attention_output_gate, layer_trace);
        trace();
    }
    if (!input.positions || input.advance) state.cache_pos += tokens;
    hidden = tyloi::ops::contiguous(tyloi::ops::copy(hidden, primary));
    if (input.raw_hidden) *input.raw_hidden = hidden;
    return normalize(hidden, batch, tokens);
}

} // namespace tyloi::cuda::minicpmo45

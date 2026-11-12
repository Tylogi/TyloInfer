#include "engine/models/qwen35/predictor.h"
#include "engine/models/common/ffn.h"
#include "engine/models/qwen35/mtp_state.h"

#include <array>
#include <stdexcept>

namespace tyloi::models::qwen35 {

Tensor Predictor::step(const Tensor& hidden, const Tensor& next_ids, const Tensor& positions,
                       tyloi::cuda::qwen35::MtpState& state, const ForwardOptions& options) const {
    hidden.require_dense();
    next_ids.require_dense();
    if (positions.defined()) positions.require_dense();
    if (hidden.dim() != 3 || next_ids.dim() != 2 ||
        hidden.size(0) != next_ids.size(0) || hidden.size(1) != next_ids.size(1) ||
        hidden.size(2) != config.hidden_size || hidden.size(0) <= 0 || hidden.size(1) <= 0)
        throw std::runtime_error("Qwen MTP inputs must be matching [B,T,H] hidden states and [B,T] next-token IDs");
    const auto batch = hidden.size(0), tokens = hidden.size(1);
    if (state.position() > config.max_position_embeddings - tokens)
        throw std::runtime_error("Qwen MTP history exceeds context capacity");
    ops::DeviceScope scope(device);
    const bool explicit_positions = positions.defined();
    auto pos = explicit_positions
        ? ops::contiguous(ops::cast(ops::copy(positions, device), kInt64))
        : ops::arange(state.position(), state.position() + tokens, device);
    if (!((pos.dim() == 1 && pos.numel() == tokens) ||
          (pos.dim() == 2 && pos.size(1) == tokens &&
           (pos.size(0) == batch || (pos.size(0) == 3 && !config.mrope_sections.empty())))))
        throw std::runtime_error("Qwen MTP positions must have shape [T], [B,T], or configured grid-MRoPE [3,T]");

    state.prepare(batch, tokens, options.kv_capacity);
    const auto kv = state.views();
    auto ids = ops::contiguous(ops::copy(ops::cast(next_ids, kInt64), device));
    auto embedded = ops::measure("model.embed", device, [&] { return ops::embedding(ids, embedding); });
    auto embedded_value = ops::cast(ops::contiguous(ops::cast(embedded, kFloat16)), hidden.scalar_type());

    AttentionStep attention;
    attention.positions = pos;
    attention.cache_position = state.position();
    if (tokens == 1 && state.position() > 0)
        attention.sequence_lengths = ops::full(std::array<std::int64_t, 1>{batch}, state.position() + 1, pos.device());
    if (explicit_positions)
        attention.cache_positions = ops::arange(state.position(), state.position() + tokens, pos.device());
    auto result = forward(hidden, embedded_value, attention, kv, rope, options.trace, options.trace_layer);
    state.advance(tokens);
    return result;
}

Tensor Predictor::forward(const Tensor& hidden, const Tensor& embeddings, const AttentionStep& step,
                          std::span<const state::KvView> kv, const ops::RopeTable& rope,
                          ops::Trace* trace, int trace_layer) const {
    if (!hidden || !embeddings || hidden.dim() != 3 || embeddings.dim() != 3 ||
        hidden.size(0) <= 0 || hidden.size(1) <= 0 || hidden.size(2) <= 0 ||
        hidden.size(2) != config.hidden_size ||
        embeddings.size(0) != hidden.size(0) || embeddings.size(1) != hidden.size(1) ||
        embeddings.size(2) != hidden.size(2))
        throw std::runtime_error("Qwen MTP inputs must be matching nonempty [B,T,H] tensors");
    if (kv.size() != layers.size())
        throw std::runtime_error("Qwen MTP cache does not match predictor layers");
    if (step.cache_position < 0 || hidden.size(1) > config.max_position_embeddings ||
        step.cache_position > config.max_position_embeddings - hidden.size(1))
        throw std::runtime_error("Qwen MTP history exceeds context capacity");

    auto trace_for = [&](int layer) { return trace && trace_layer == layer ? trace : nullptr; };
    auto record = [&](const char* name, const Tensor& value) {
        if (auto* output = trace_for(0))
            output->emplace_back(name, ops::clone(ops::contiguous(ops::cast(value, kFloat32))));
    };
    auto embedded_norm = ops::rms_norm(embeddings, embedding_norm, config.rms_norm_eps,
                                       config.legacy_tensor_layout.norm_weight_offset);
    auto hidden_value = ops::rms_norm(hidden, hidden_norm, config.rms_norm_eps,
                                      config.legacy_tensor_layout.norm_weight_offset);
    record("mtp.embedding_norm", embedded_norm);
    record("mtp.hidden_norm", hidden_value);
    std::array<Tensor, 2> fusion_inputs{embedded_norm, hidden_value};
    auto current = ops::project(ops::cat(fusion_inputs, -1), fusion);
    record("mtp.fusion", current);

    for (std::size_t index = 0; index < layers.size(); ++index) {
        const auto& layer = layers[index];
        const auto& attention = layer.attention;
        auto output = attention.forward(current, step, kv[index], rope, trace_for(attention.layer));
        current = full_ffn(layer.ffn, layer.ffn_norm, output[0], output[1],
                           attention.rms_norm_eps, attention.norm_weight_offset,
                           attention.official_bf16, attention.attention_output_gate,
                           trace_for(attention.layer));
    }
    auto output = ops::rms_norm(current, output_norm, config.rms_norm_eps,
                                config.legacy_tensor_layout.norm_weight_offset);
    record("mtp.output_norm", output);
    return output;
}

} // namespace tyloi::models::qwen35

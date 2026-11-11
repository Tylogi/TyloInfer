#include "engine/models/qwen35/predictor.h"
#include "engine/models/common/ffn.h"

#include <array>
#include <stdexcept>

namespace tyloi::models::qwen35 {

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

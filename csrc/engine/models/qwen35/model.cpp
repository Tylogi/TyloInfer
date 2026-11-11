#include "engine/models/qwen35/model.h"
#include "engine/models/qwen35/state.h"
#include "engine/ops.h"

#include <stdexcept>
#include <utility>

namespace tyloi::cuda::qwen35 {

tyloi::models::Info Model::info() const {
    auto result = tyloi::models::model_info(metadata);
    for (const auto& layer : layers)
        if (layer.device.is_cpu()) result.decode_graph = false;
    return result;
}

MfqModelCapabilities Model::capabilities() const {
    return {graph.architecture, graph.has_component("text"), vision.has_value(), false,
            false, false, false, false, "model-graph"};
}

tyloi::engine::Metrics Model::metrics() const {
    double bits = 0;
    bool full_attention = false;
    for (const auto& layer : layers) {
        if (const auto* full = std::get_if<Full>(&layer.weights)) {
            full_attention = true;
            bits = full->attention.kv_quantization.bits;
            break;
        }
    }
    return {{"kv_memory_supported", 0.0},
            {"kv_quantization_supported", double(full_attention)},
            {"kv_quantization_bits", bits},
            {"vision_declared", double(graph.has_component("vision"))},
            {"vision_supported", double(graph.has_component("vision"))},
            {"vision_available", double(vision.has_value())},
            {"mtp_declared", double(graph.has_component("predictor"))},
            {"mtp_supported", 0.0}, {"mtp_available", 0.0}};
}

std::unique_ptr<tyloi::models::State> Model::make_state() const {
    return std::make_unique<State>(*this, recurrent_transposed);
}

tyloi::StepSequence<tyloi::models::Prepared> Model::prepare(
    const std::vector<std::int64_t>& tokens, const MfqMultimodalInput& media,
    tyloi::models::State& value) const {
    if (!vision) throw std::invalid_argument("model has no multimodal component");
    auto* state = dynamic_cast<State*>(&value);
    if (!state || !state->belongs_to(*this))
        throw std::invalid_argument("Qwen3.5 state belongs to another model");
    return vision->prepare(*this, state->images, device, tokens, media);
}

::tyloi::Tensor Model::embedding(const tyloi::Tensor& ids, int primary) const {
    if (ids.dim() != 1 && ids.dim() != 2)
        throw std::runtime_error("token IDs must have shape [tokens] or [batch,tokens]");
    auto shaped = ids.dim() == 1 ? tyloi::Tensor(tyloi::ops::unsqueeze(ids, 0)) : tyloi::Tensor{};
    const auto& input = shaped ? shaped : ids;
    if (input.size(0) <= 0 || input.size(1) <= 0)
        throw std::runtime_error("token IDs must not be empty");
    auto normalized = tyloi::ops::contiguous(tyloi::ops::copy(
        tyloi::ops::cast(input, tyloi::ScalarType::int64),
        tyloi::Device{device.type, primary}));
    auto result = tyloi::ops::measure("model.embed", normalized.device(), [&] {
        return tyloi::ops::embedding(normalized, embed);
    });
    return tyloi::ops::contiguous(tyloi::ops::cast(result, tyloi::ScalarType::float16));
}

::tyloi::Tensor Model::normalize(const tyloi::Tensor& hidden, std::int64_t batch, std::int64_t tokens) const {
    auto shaped = tyloi::ops::reshape(hidden, {batch, tokens, metadata.hidden_size});
    return tyloi::ops::measure("model.output_norm", shaped.device(), [&] {
        return tyloi::ops::rms_norm(shaped, output_norm, metadata.rms_norm_eps,
                                    metadata.norm_weight_offset);
    });
}

::tyloi::Tensor Model::softcap(const tyloi::Tensor& logits) const {
    return tyloi::ops::softcap(logits, metadata.final_logit_softcapping);
}

::tyloi::Tensor Model::logits(const tyloi::Tensor& hidden, bool capped) const {
    auto result = tyloi::ops::measure("model.lm_head", hidden.device(), [&] {
        return tyloi::ops::project(hidden, lm_head);
    });
    return capped ? softcap(result) : result;
}

} // namespace tyloi::cuda::qwen35

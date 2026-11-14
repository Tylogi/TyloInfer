#include "engine/models/plan.h"
#include "engine/models/qwen4_exp/components.h"

#include "engine/models/load.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/source/model_config.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/qwen4_exp/predictor.h"
#include "engine/weights/loader.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::qwen4_exp {

Model::Model() = default;
Model::~Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

tyloi::models::Info Model::info() const {
    auto result = tyloi::models::Info{metadata.vocab_size, config.maximum, "qwen4_exp",
        std::size_t(metadata.hidden_size) * sizeof(float) * 8, true, true};
    result.streamed_kv = std::any_of(layers.begin(), layers.end(),
        [](const auto& layer) { return bool(layer->qsa); });
    result.graph = tyloi::models::GraphOwner::model;
    result.graph_weights = tyloi::models::GraphWeights::native;
    result.raw_row_bytes = std::size_t(metadata.hidden_size) * metadata.hc_mult * sizeof(std::uint16_t);
    return result;
}

MfqModelCapabilities Model::capabilities() const {
    const bool vision = parts && parts->grid_vision.has_value();
    const auto state = ::tyloi::models::component_state(graph, plan, vision, predictor() != nullptr);
    const auto caps = ::tyloi::models::runtime_capabilities(graph, plan, state, vision, false);
    return {graph.architecture, caps.text, caps.image_input, caps.video_input,
        caps.audio_input, caps.audio_output, caps.full_duplex, caps.mtp, "model-graph+cuda-adapters"};
}

tyloi::engine::Metrics Model::metrics() const {
    const bool vision = parts && parts->grid_vision.has_value();
    const auto state = ::tyloi::models::component_state(graph, plan, vision, predictor() != nullptr);
    return {{"kv_memory_supported", 0.0}, {"kv_quantization_supported", 0.0},
        {"kv_quantization_bits", 0.0}, {"vision_declared", double(state.vision_declared)},
        {"vision_supported", double(state.vision_supported)}, {"vision_available", double(state.vision_available)},
        {"mtp_declared", double(state.mtp_declared)}, {"mtp_supported", double(state.mtp_supported)},
        {"mtp_available", double(state.mtp_available)}};
}

tyloi::models::Mtp* Model::predictor() const noexcept { return parts ? parts->mtp.get() : nullptr; }

std::unique_ptr<tyloi::models::State> Model::make_state() const {
    auto state = std::make_unique<State>(*this);
    if (auto* mtp = predictor()) state->predictor = mtp->make_state();
    return state;
}

tyloi::StepSequence<tyloi::models::Prepared> Model::prepare(
        const std::vector<std::int64_t>& tokens, const MfqMultimodalInput& media,
        tyloi::models::State& state_value) const {
    auto* state = dynamic_cast<State*>(&state_value);
    if (!state || !state->belongs_to(*this))
        throw std::invalid_argument("Qwen4 state belongs to a different model");
    if (!parts) throw std::invalid_argument("model has no multimodal component");
    return parts->prepare(*this, state->images, primary_device, tokens, media);
}

tyloi::engine::Metrics Components::profile_metrics(const Model& language, const State& state) const {
    return qwen4_decode_graph_profile_metrics(language, state);
}

std::optional<tyloi::engine::Metrics> Components::memory_metrics(const Model& language, const State& state) const {
    return qwen4_cache_memory_metrics(language, state, mtp.get());
}

tyloi::StepSequence<tyloi::models::Prepared> Components::prepare(
        const Model& language, tyloi::cache::Media& images,
        int primary,
        const std::vector<int64_t>& tokens, const MfqMultimodalInput& media) const {
    if (grid_vision) return grid_vision->prepare(
        language, images, tyloi::Device(tyloi::kCUDA, primary), tokens, media);
    throw std::invalid_argument("model has no multimodal component");
}

void load_components(
        Components& result,
        tyloi::cuda::qwen4_exp::Model& model,
        const tyloi::weights::Loader& weights,
        bool load_optional_components, const tyloi::models::LoadOptions& load_options) {
    if (!load_optional_components) return;
    if (model.plan.vision == ::tyloi::models::VisionAdapter::grid_vit) {
        const auto document = nlohmann::json::parse(
            tyloi::models::config_json(*model.source));
        const auto& vision = document.at("vision_config");
        if (vision.value("hidden_act", std::string("gelu_pytorch_tanh")) != "gelu_pytorch_tanh" ||
                !vision.value("deepstack_visual_indexes", std::vector<int>{}).empty())
            throw std::runtime_error("unsupported Qwen4 grid-ViT activation or deepstack");
        tyloi::GridVisionConfig config;
        config.hidden_size = vision.at("hidden_size").get<int64_t>();
        config.intermediate_size = vision.at("intermediate_size").get<int64_t>();
        config.depth = vision.at("depth").get<int64_t>();
        config.num_heads = vision.at("num_heads").get<int64_t>();
        config.in_channels = vision.value("in_channels", int64_t{3});
        config.patch_size = vision.at("patch_size").get<int64_t>();
        config.temporal_patch_size = vision.at("temporal_patch_size").get<int64_t>();
        config.spatial_merge_size = vision.value("spatial_merge_size", int64_t{2});
        config.out_hidden_size = vision.at("out_hidden_size").get<int64_t>();
        config.num_position_embeddings = vision.at("num_position_embeddings").get<int64_t>();
        config.rope_theta = vision.value("rope_theta", 10000.0);
        if (vision.contains("rope_parameters"))
            config.rope_theta = vision.at("rope_parameters").value("rope_theta", config.rope_theta);
        config.layer_norm_eps = vision.value("layer_norm_eps", 1e-6);
        config.validate();
        const auto image = document.at("image_token_id").get<int64_t>();
        const auto video = document.at("video_token_id").get<int64_t>();
        if (config.out_hidden_size != model.config.hidden || model.config.sections.size() != 3 ||
                image < 0 || video < 0 || image >= model.config.vocab || video >= model.config.vocab || image == video)
            throw std::runtime_error("invalid Qwen4 grid-ViT text geometry or placeholder IDs");
        const auto* component = model.graph.component("vision");
        result.grid_vision.emplace(
            tyloi::models::grid::Vision::load(
                weights, weights.cache(), config, image, video,
                component->input_contract, component->position_policy, load_options.vision_resident));
    } else if (model.plan.vision != ::tyloi::models::VisionAdapter::none) {
        throw std::runtime_error("unsupported Qwen4 CUDA vision adapter");
    }
    if (model.plan.predictor != ::tyloi::models::PredictorAdapter::none &&
             model.plan.predictor !=
                 ::tyloi::models::PredictorAdapter::flash_next) {
        throw std::runtime_error(
            "unsupported Qwen4 CUDA component adapter");
    }
    if (model.plan.predictor ==
            ::tyloi::models::PredictorAdapter::none) return;
    TYLOINFER_RUNTIME_CHECK(
        model.supports_speculation(),
        "invalid Flash-Next predictor model");
    auto predictor = tyloi::cuda::qwen4_exp::Predictor::load_if_present(
        weights, load_options, model.config, model.embed, model.lm_head,
        model.metadata.final_logit_softcapping);
    if (predictor) {
        result.mtp =
            std::make_unique<tyloi::cuda::qwen4_exp::Predictor>(
                std::move(*predictor));
    }
}

} // namespace tyloi::cuda::qwen4_exp

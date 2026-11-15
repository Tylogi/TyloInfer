#include "engine/models/nemotron_h/components.h"

#include "engine/models/source/model_config.h"
#include "engine/models/nemotron_h/model.h"
#include "engine/models/nemotron_h/mtp.h"
#include "engine/models/nemotron_h/state.h"
#include "engine/models/load.h"

#include <stdexcept>
#include <utility>

namespace tyloi::cuda::nemotron_h {

Model::Model() = default;
Model::~Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

tyloi::models::Info Model::info() const {
    return tyloi::models::model_info(metadata);
}

MfqModelCapabilities Model::capabilities() const {
    const bool vision = parts && parts->radio_vision.has_value();
    return {graph.architecture, graph.has_component("text"), vision, false,
        false, false, false, predictor() != nullptr, "model"};
}

tyloi::engine::Metrics Model::metrics() const {
    const bool vision = parts && parts->radio_vision.has_value();
    const bool vision_declared = graph.has_component("vision");
    const bool mtp_declared = graph.has_component("predictor");
    return {{"kv_memory_supported", 0.0}, {"kv_quantization_supported", 0.0},
        {"kv_quantization_bits", 0.0}, {"vision_declared", double(vision_declared)},
        {"vision_supported", double(vision_declared)}, {"vision_available", double(vision)},
        {"mtp_declared", double(mtp_declared)}, {"mtp_supported", double(mtp_declared)},
        {"mtp_available", double(predictor() != nullptr)}};
}

tyloi::models::Mtp* Model::predictor() const noexcept { return parts ? parts->mtp.get() : nullptr; }

std::unique_ptr<tyloi::models::State> Model::make_state() const {
    auto state = std::make_unique<State>(*this);
    if (auto* predictor = this->predictor()) state->predictor = predictor->make_state();
    return state;
}

tyloi::StepSequence<tyloi::models::Prepared> Model::prepare(
        const std::vector<std::int64_t>& tokens, const MfqMultimodalInput& media,
        tyloi::models::State& state_value) const {
    const auto* state = dynamic_cast<const State*>(&state_value);
    if (!state || !state->belongs_to(*this))
        throw std::invalid_argument("Nemotron-H state belongs to a different model");
    if (!parts || !parts->radio_vision)
        throw std::invalid_argument("model has no multimodal component");
    return parts->radio_vision->prepare(*this, primary, tokens, media);
}

Components load_components(Model& model, const tyloi::weights::Loader& weights,
        bool load_optional_components
        ) {
    Components result;
    if (!load_optional_components) return result;
    if (const auto* component = model.graph.component("vision")) {
        const auto vision = tyloi::models::nemotron_h::RadioVisionConfig::from_json(
            tyloi::models::config_json(*model.source));
        result.radio_vision.emplace(CudaRadioPromptComponent::load(
            weights, vision, component->input_contract, component->position_policy));
    }
    if (!model.graph.has_component("predictor")) return result;
    TYLOINFER_RUNTIME_CHECK(supports_speculation(model), "invalid Nemotron-H predictor model");
    auto predictor = Mtp::load_if_present(
        weights, model.config, model.embed);
    TYLOINFER_RUNTIME_CHECK(predictor,
        "Nemotron-H graph declares the predictor component but the source has no MTP weights");
    result.mtp = std::make_unique<Mtp>(std::move(*predictor));
    return result;
}

void load(std::unique_ptr<tyloi::models::CausalModel>& owner, std::shared_ptr<const tyloi::ModelSource> source,
        std::string_view definition, const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
        const tyloi::models::LoadOptions& options) {
    if (!options.cpu_offload_layers.empty())
        throw std::invalid_argument("Nemotron-H does not support --cpu-offload-layers");
    owner = std::make_unique<Model>(read(std::move(source), definition, options.context));
    auto* model = static_cast<Model*>(owner.get());
    load(*model, weights, placement, options.gpu_layers, options.expert_parallel);
    model->parts = std::make_unique<Components>(load_components(*model, weights, options.optional_components));
}


} // namespace tyloi::cuda::nemotron_h

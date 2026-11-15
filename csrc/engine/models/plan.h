#pragma once

#include "engine/models/source/model_graph.h"

#include <string>
#include <string_view>
#include <utility>

namespace tyloi::models {

enum class VisionAdapter {
    none,
    grid_vit,
    minicpmo45,
    deepseek_v41,
    radio,
};

enum class PredictorAdapter {
    none,
    qwen35,
    flash_next,
    deepseek_v41_dspark,
    nemotron_h,
};

struct Plan {
    VisionAdapter vision = VisionAdapter::none;
    PredictorAdapter predictor = PredictorAdapter::none;
};

struct CapabilityMatrix {
    bool text = true;
    bool image_input = false;
    bool video_input = false;
    bool audio_input = false;
    bool audio_output = false;
    bool full_duplex = false;
    bool mtp = false;
};

struct ComponentState {
    bool vision_declared = false;
    bool vision_supported = false;
    bool vision_available = false;
    bool vision_enabled = false;
    bool mtp_declared = false;
    bool mtp_supported = false;
    bool mtp_available = false;
    bool mtp_enabled = false;
};

inline Plan model_plan(
        const ModelGraph& graph) noexcept {
    Plan result;
    const auto* vision = graph.component("vision");
    if ((graph.backbone == "qwen3_5" || graph.backbone == "qwen4_exp") && vision != nullptr &&
        vision->tensor_root == "vision" &&
        vision->implementation == "grid_vit" &&
        vision->input_contract == kGridVisionInputContract &&
        vision->position_policy == kGridMropePositionPolicy) {
        result.vision = VisionAdapter::grid_vit;
    }
    const auto* audio = graph.component("audio_input");
    if(graph.backbone=="deepseek_v41" && vision && vision->tensor_root=="vision" &&
            vision->implementation=="deepseek_v41_vision" &&
            vision->input_contract=="deepseek_v41_vision.v1" &&
            vision->position_policy=="deepseek_v41_positions")result.vision=VisionAdapter::deepseek_v41;
    const auto* tts = graph.component("audio_output");
    // The MiniCPM adapter is one composite implementation.  Do
    // not advertise a partially declared graph as supported until those
    // weighted components have independent adapters.
    if (graph.backbone == "minicpmo45" &&
        vision != nullptr && audio != nullptr && tts != nullptr &&
        vision->implementation == "minicpmo45_vision" &&
        audio->implementation == "minicpmo45_audio" &&
        tts->implementation == "minicpmo45_tts") {
        result.vision = VisionAdapter::minicpmo45;
    }
    // Cropped-position-embedding towers are selected by their declared data
    // contract alone: the graph names the implementation and the patch
    // contract, and RADIO's graph carries no multimodal position policy.
    if (vision != nullptr && vision->tensor_root == "vision" &&
        vision->implementation == "radio" &&
        vision->input_contract == kRadioCpeInputContract &&
        vision->position_policy.empty()) {
        result.vision = VisionAdapter::radio;
    }
    const auto* predictor = graph.component("predictor");
    if (graph.backbone == "qwen3_5" && predictor != nullptr &&
        predictor->implementation == "next_token_prediction" &&
        predictor->tensor_root == "predictor") {
        result.predictor = PredictorAdapter::qwen35;
    }
    if ((graph.backbone=="qwen4_exp" || graph.backbone=="glm5_next") && predictor!=nullptr &&
        predictor->implementation=="next_token_prediction" && predictor->tensor_root=="predictor") {
        result.predictor=PredictorAdapter::flash_next;
    }
    if (graph.backbone == "deepseek_v41" && predictor != nullptr &&
        predictor->implementation == "deepseek_v41_dspark" &&
        predictor->tensor_root == "predictor") {
        result.predictor = PredictorAdapter::deepseek_v41_dspark;
    }
    if (graph.backbone == "nemotron_h" && predictor != nullptr &&
        predictor->implementation == "nemotron_h_mtp" &&
        predictor->tensor_root == "predictor") {
        result.predictor = PredictorAdapter::nemotron_h;
    }
    return result;
}

inline constexpr CapabilityMatrix capabilities(
        const Plan& plan) noexcept {
    CapabilityMatrix result;
    result.image_input =
        plan.vision == VisionAdapter::grid_vit ||
        plan.vision == VisionAdapter::minicpmo45 ||
        plan.vision == VisionAdapter::deepseek_v41 ||
        plan.vision == VisionAdapter::radio;
    result.video_input = plan.vision == VisionAdapter::minicpmo45;
    result.audio_input = plan.vision == VisionAdapter::minicpmo45;
    result.audio_output = plan.vision == VisionAdapter::minicpmo45;
    result.full_duplex = plan.vision == VisionAdapter::minicpmo45;
    result.mtp = plan.predictor != PredictorAdapter::none;
    return result;
}

inline ComponentState component_state(
        const ModelGraph& graph,
        const Plan& plan,
        bool vision_loaded,
        bool mtp_loaded,
        bool enable_vision = true,
        bool enable_mtp = true) noexcept {
    ComponentState result;
    result.vision_declared = graph.has_component("vision");
    result.vision_supported = result.vision_declared &&
        plan.vision != VisionAdapter::none;
    result.vision_available = result.vision_supported && vision_loaded;
    result.vision_enabled = result.vision_available && enable_vision;
    result.mtp_declared = graph.has_component("predictor");
    result.mtp_supported = result.mtp_declared &&
        plan.predictor != PredictorAdapter::none;
    result.mtp_available = result.mtp_supported && mtp_loaded;
    result.mtp_enabled = result.mtp_available && enable_mtp;
    return result;
}

inline CapabilityMatrix runtime_capabilities(
        const ModelGraph& graph,
        const Plan& plan,
        const ComponentState& components,
        bool multimodal_available,
        bool duplex_available) noexcept {
    auto result = capabilities(plan);
    result.text = result.text && graph.has_component("text");
    result.image_input = result.image_input &&
        components.vision_available && multimodal_available;
    result.video_input = result.video_input &&
        components.vision_available && multimodal_available;
    result.audio_input = result.audio_input && multimodal_available &&
        graph.has_component("audio_input");
    result.audio_output = result.audio_output && duplex_available &&
        graph.has_component("audio_output");
    result.full_duplex = result.full_duplex && duplex_available &&
        graph.has_component("duplex");
    result.mtp = result.mtp && components.mtp_available;
    return result;
}

} // namespace tyloi::models

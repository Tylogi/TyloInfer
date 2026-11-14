#include "engine/models/plan.h"
#include "components.h"
#include "duplex.h"
#include "text/load.h"

#include "engine/backends/cuda/executor/sampling.h"
#include "engine/backends/cuda/executor/executor.h"
#include "backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::cuda::minicpmo45 {

std::unique_ptr<Components> load_components(text::Model& model,
        Executor& executor, const tyloi::weights::Loader& weights,
        bool load_optional_components, int primary) {
    if (!load_optional_components || model.plan.vision == ::tyloi::models::VisionAdapter::none)
        return nullptr;
    if (model.plan.vision != ::tyloi::models::VisionAdapter::minicpmo45)
        throw std::runtime_error("unsupported MiniCPM-o CUDA vision adapter");
    model.primary = primary;
    text::load_media(model, weights);
    return std::make_unique<Components>(model, executor, weights, primary);
}

struct Components::State {
    text::Model& language;
    Executor& executor;
    text::State prepare_state;
    std::optional<MiniCPMO45DuplexSession> duplex_session;

    State(text::Model& model, Executor& execution)
        : language(model), executor(execution), prepare_state(model) {}
};

Components::Components(text::Model& language, Executor& executor,
                       const tyloi::weights::Loader&, int)
    : state_(std::make_unique<State>(language, executor)) {}

Components::~Components() = default;

tyloi::StepSequence<tyloi::models::Prepared> Components::prepare(
        const std::vector<int64_t>& prompt, const MfqMultimodalInput& media) {
    return state_->language.prepare(prompt, media, state_->prepare_state);
}

void Components::start(const MfqDuplexSessionParams& parameters) {
        auto policy = Executor::current_ffn();
        policy.independent |= state_->language.info().separate_projections;
        auto model_scope = state_->executor.enter(policy);
        if (parameters.special_ids.size() != 15) {
            throw std::invalid_argument(
                "MiniCPM-o duplex requires 15 special token IDs");
        }
        if ((!parameters.greedy &&
                (!std::isfinite(parameters.temperature) ||
                 parameters.temperature <= 0.0)) ||
                parameters.top_k < 0 ||
                parameters.top_k >
                    std::min<int64_t>(
                        state_->language.metadata.vocab_size, 1024) ||
                !std::isfinite(parameters.top_p) ||
                parameters.top_p <= 0.0 || parameters.top_p > 1.0 ||
                !std::isfinite(parameters.listen_probability_scale) ||
                parameters.listen_probability_scale < 0.0 ||
                !std::isfinite(parameters.repetition_penalty) ||
                parameters.repetition_penalty <= 0.0 ||
                parameters.repetition_window <= 0 ||
                !std::isfinite(parameters.length_penalty) ||
                parameters.length_penalty <= 0.0 ||
                !std::isfinite(parameters.tts_temperature) ||
                parameters.tts_temperature <= 0.0 ||
                !std::isfinite(parameters.tts_repetition_penalty) ||
                parameters.tts_repetition_penalty <= 0.0) {
            throw std::invalid_argument(
                "MiniCPM-o duplex sampling configuration is invalid");
        }
        const int64_t audio_bos = parameters.special_ids.back();
        if (audio_bos < 0 || audio_bos >= 152064 ||
                std::any_of(
                    parameters.special_ids.begin(),
                    parameters.special_ids.end() - 1,
                    [&](int64_t token) {
                        return token < 0 ||
                            token >= state_->language.metadata.vocab_size;
                    }) ||
                std::any_of(
                    parameters.forbidden_ids.begin(),
                    parameters.forbidden_ids.end(),
                    [&](int64_t token) {
                        return token < 0 ||
                            token >= state_->language.metadata.vocab_size;
                    })) {
            throw std::invalid_argument(
                "MiniCPM-o duplex token ID is out of range");
        }
        if ((parameters.reference_audio_frames == 0) !=
                parameters.reference_audio_features.empty() ||
                parameters.reference_audio_frames < 0 ||
                (!parameters.reference_audio_features.empty() &&
                 (parameters.reference_audio_frames < 3 ||
                  parameters.reference_audio_features.size() !=
                    static_cast<size_t>(
                        parameters.reference_audio_frames) * 80))) {
            throw std::invalid_argument(
                "MiniCPM-o reference Mel geometry is invalid");
        }

        MfqCudaGuard guard(state_->language.primary);
        ::tyloi::cuda::manual_seed(static_cast<int64_t>(parameters.seed));
        mfq_cuda_manual_seed_all(parameters.seed);
        auto special_ids = MiniCPMO45DuplexSpecialIds::from_tensor(
            ::tyloi::cuda::tensor(
                parameters.special_ids,
                ::tyloi::TensorOptions().dtype(::tyloi::kInt64)));
        if (state_->duplex_session) mfq_cuda_synchronize();
        state_->duplex_session.reset();
        const auto mode = state_->executor.forward_mode(state_->language.primary);
        state_->duplex_session.emplace(
            state_->language, tyloi::models::ForwardOptions{mode.kv_capacity, mode.trace, mode.trace_layer, mode.primary},
            special_ids, parameters.forbidden_ids, parameters.greedy);
        state_->duplex_session->temperature = parameters.temperature;
        state_->duplex_session->top_k = parameters.top_k;
        state_->duplex_session->top_p = parameters.top_p;
        state_->duplex_session->listen_probability_scale =
            parameters.listen_probability_scale;
        state_->duplex_session->repetition_penalty = parameters.repetition_penalty;
        state_->duplex_session->repetition_window = parameters.repetition_window;
        state_->duplex_session->length_penalty = parameters.length_penalty;
        state_->duplex_session->tts_temperature = parameters.tts_temperature;
        state_->duplex_session->tts_repetition_penalty =
            parameters.tts_repetition_penalty;

        const auto ids_tensor = [](const std::vector<int64_t> & values) {
            return values.empty()
                ? ::tyloi::DenseTensor()
                : ::tyloi::cuda::tensor(
                    values,
                    ::tyloi::TensorOptions().dtype(::tyloi::kInt64));
        };
        ::tyloi::DenseTensor reference_features;
        if (!parameters.reference_audio_features.empty()) {
            reference_features = ::tyloi::cuda::tensor(
                parameters.reference_audio_features,
                ::tyloi::TensorOptions().dtype(::tyloi::kFloat32))
                .reshape({1, 80, parameters.reference_audio_frames});
        }
        state_->duplex_session->prepare(
            ids_tensor(parameters.system_prefix),
            reference_features,
            ids_tensor(parameters.system_suffix));
        mfq_cuda_synchronize();
}

MfqDuplexStepResult Components::step(const MfqDuplexStepInput& input) {
        auto policy = Executor::current_ffn();
        policy.independent |= state_->language.info().separate_projections;
        auto model_scope = state_->executor.enter(policy);
        const bool has_audio = input.audio_frames > 0;
        const bool has_text = !input.text_tokens.empty();
        if (has_audio && input.audio_features.size() !=
                static_cast<size_t>(input.audio_frames) * 80) {
            throw std::invalid_argument(
                "MiniCPM-o duplex Mel geometry is invalid");
        }
        if (!has_audio && !has_text) {
            throw std::invalid_argument(
                "MiniCPM-o duplex step has no input");
        }
        if (input.max_new_speak_tokens < 2) {
            throw std::invalid_argument(
                "MiniCPM-o duplex generation requires at least two token slots");
        }

        MfqCudaGuard guard(state_->language.primary);
        if (!state_->duplex_session) {
            throw std::runtime_error(
                "MiniCPM-o duplex session is not prepared");
        }
        ::tyloi::DenseTensor audio_features;
        if (has_audio) {
            audio_features = ::tyloi::cuda::tensor(
                input.audio_features,
                ::tyloi::TensorOptions().dtype(::tyloi::kFloat32))
                .reshape({1, 80, input.audio_frames});
        }
        ::tyloi::DenseTensor text_ids;
        if (has_text) {
            text_ids = ::tyloi::cuda::tensor(
                input.text_tokens,
                ::tyloi::TensorOptions().dtype(::tyloi::kInt64))
                .reshape({1, static_cast<int64_t>(input.text_tokens.size())});
        }
        const auto started = std::chrono::steady_clock::now();
        auto result = state_->duplex_session->run_step(
            {}, {}, {}, {}, audio_features,
            input.audio_prefix_extra_frames,
            input.audio_suffix_extra_frames,
            text_ids,
            input.max_new_speak_tokens,
            input.force_listen,
            input.force_speak);

        MfqDuplexStepResult response;
        auto generated = result.generated_ids
#ifdef TYLOINFER_NATIVE_CUDA
            .dense()
#endif
            .to(::tyloi::kCPU, ::tyloi::kInt64).contiguous().reshape({-1});
        const auto * generated_data = generated.template data_ptr<int64_t>();
        response.generated_tokens.assign(
            generated_data, generated_data + generated.numel());
        auto codes = result.tts_codes
#ifdef TYLOINFER_NATIVE_CUDA
            .dense()
#endif
            .to(::tyloi::kCPU, ::tyloi::kInt32).contiguous().reshape({-1});
        const auto * code_data = codes.template data_ptr<int32_t>();
        response.audio_tokens.assign(
            code_data, code_data + codes.numel());
        response.is_listen = result.is_listen;
        response.end_of_turn = result.end_of_turn;
        response.tts_force_flush = result.tts_force_flush;
        response.audio_chunk_index = state_->duplex_session->audio_chunk_index;
        response.language_cache_position = state_->duplex_session->language_state.cache_pos;
        response.audio_cache_position = state_->duplex_session->audio_state.cache_length();
        response.tts_cache_position = state_->duplex_session->tts_state.position;
        response.inference_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
        return response;
}

void Components::stop() {
        MfqCudaGuard guard(state_->language.primary);
        mfq_cuda_synchronize();
        state_->duplex_session.reset();
        mfq_cuda_synchronize();
}
} // namespace tyloi::cuda::minicpmo45

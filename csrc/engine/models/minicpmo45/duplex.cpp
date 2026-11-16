#include "engine/models/minicpmo45/duplex.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/ops/sampling/sampler.h"
#include "engine/backends/cuda/native/tensor/shape.h"
#include "engine/state/kv/view.h"
#include "backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::cuda::minicpmo45 {
namespace {
using tyloi::models::ForwardOptions;
using Growth = std::vector<std::pair<Device, std::size_t>>;

void add_growth(Growth& growth, Device device, std::size_t bytes) {
    if (!bytes) return;
    auto it = std::find_if(growth.begin(), growth.end(), [device](const auto& v) { return v.first == device; });
    if (it == growth.end()) growth.emplace_back(device, bytes);
    else {
        if (it->second > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::overflow_error("MiniCPM duplex growth overflow");
        it->second += bytes;
    }
}

void validate(const text::Model& model, const MfqDuplexSessionParams& parameters) {
    if (parameters.special_ids.size() != 15)
        throw std::invalid_argument("MiniCPM-o duplex requires 15 special token IDs");
    if ((!parameters.greedy && (!std::isfinite(parameters.temperature) || parameters.temperature <= 0.0)) ||
        parameters.top_k < 0 || parameters.top_k > std::min<int64_t>(model.metadata.vocab_size, 1024) ||
        !std::isfinite(parameters.top_p) || parameters.top_p <= 0.0 || parameters.top_p > 1.0 ||
        !std::isfinite(parameters.listen_probability_scale) || parameters.listen_probability_scale < 0.0 ||
        !std::isfinite(parameters.repetition_penalty) || parameters.repetition_penalty <= 0.0 ||
        parameters.repetition_window <= 0 || !std::isfinite(parameters.length_penalty) ||
        parameters.length_penalty <= 0.0 || !std::isfinite(parameters.tts_temperature) ||
        parameters.tts_temperature <= 0.0 || !std::isfinite(parameters.tts_repetition_penalty) ||
        parameters.tts_repetition_penalty <= 0.0)
        throw std::invalid_argument("MiniCPM-o duplex sampling configuration is invalid");
    const auto audio_bos = parameters.special_ids.back();
    const auto invalid = [&](int64_t token) { return token < 0 || token >= model.metadata.vocab_size; };
    if (audio_bos < 0 || audio_bos >= 152064 ||
        std::any_of(parameters.special_ids.begin(), parameters.special_ids.end() - 1, invalid) ||
        std::any_of(parameters.forbidden_ids.begin(), parameters.forbidden_ids.end(), invalid))
        throw std::invalid_argument("MiniCPM-o duplex token ID is out of range");
    if ((parameters.reference_audio_frames == 0) != parameters.reference_audio_features.empty() ||
        parameters.reference_audio_frames < 0 || (!parameters.reference_audio_features.empty() &&
        (parameters.reference_audio_frames < 3 || parameters.reference_audio_features.size() !=
            std::size_t(parameters.reference_audio_frames) * 80)))
        throw std::invalid_argument("MiniCPM-o reference Mel geometry is invalid");
}

void validate(const MfqDuplexStepInput& input) {
    const bool has_audio = input.audio_frames > 0;
    if (has_audio && input.audio_features.size() != std::size_t(input.audio_frames) * 80)
        throw std::invalid_argument("MiniCPM-o duplex Mel geometry is invalid");
    if (!has_audio && input.text_tokens.empty())
        throw std::invalid_argument("MiniCPM-o duplex step has no input");
    if (input.max_new_speak_tokens < 2)
        throw std::invalid_argument("MiniCPM-o duplex generation requires at least two token slots");
}

} // namespace

Duplex::Duplex(const text::Model& model) : model_(model) {
    if (!model.vision || !model.resampler || !model.audio || !model.tts)
        throw std::invalid_argument("MiniCPM-o duplex components are unavailable");
}

Growth Duplex::growth(const MfqDuplexSessionParams& params, const ForwardOptions& mode) const {
        validate(model_, params);
        Growth result;
        const bool preallocate = cuda::options::kv_vram_preallocated();
        std::int64_t first = static_cast<std::int64_t>(params.system_prefix.size());
        if (!first && params.reference_audio_frames > 0)
            first = std::max<std::int64_t>(1, (std::int64_t(params.reference_audio_frames) + 1) / 10);
        if (!first) first = static_cast<std::int64_t>(params.system_suffix.size());
        for (const auto& layer : model_.layers) {
            const auto& attention = layer.attention;
            if (layer.cpu && !first) continue;
            if (!layer.cpu && !preallocate && !first) continue;
            const auto capacity = !layer.cpu && preallocate ?
                (attention.sliding ? attention.attention_window : attention.max_position_embeddings) :
                (attention.sliding ? attention.attention_window :
                 (mode.kv_capacity > 0 ? std::max(first, mode.kv_capacity) : attention.max_position_embeddings));
            add_growth(result, Device(layer.cpu ? kCPU : kCUDA, layer.cpu ? 0 : layer.device),
                tyloi::state::kv_bytes(1, attention.kv_heads, capacity, attention.attention_head_dim,
                    attention.sliding, kBFloat16, attention.kv_quantization));
        }
        return result;
    }

Growth Duplex::growth(const MfqDuplexStepInput& input, const ForwardOptions& mode) const {
        validate(input);
        if (!session_) throw std::runtime_error("MiniCPM-o duplex session is not prepared");
        Growth result;
        const auto& state = *session_;
        for (std::size_t i = 0; i < model_.layers.size(); ++i) {
            if (state.language_state.layers[i].defined()) continue;
            const auto& layer = model_.layers[i];
            const auto& attention = layer.attention;
            const auto end = state.language_state.cache_pos + 1; // unit_start is the first forward
            const auto capacity = attention.sliding ? attention.attention_window :
                (mode.kv_capacity > 0 ? std::max(end, mode.kv_capacity) : attention.max_position_embeddings);
            add_growth(result, Device(layer.cpu ? kCPU : kCUDA, layer.cpu ? 0 : layer.device),
                tyloi::state::kv_bytes(1, attention.kv_heads, capacity, attention.attention_head_dim,
                    attention.sliding, kBFloat16, attention.kv_quantization));
        }
        if (input.audio_frames > 0) {
            if (input.audio_prefix_extra_frames < 0 || input.audio_suffix_extra_frames < 0)
                throw std::invalid_argument("MiniCPM-o streaming audio extra frames must be non-negative");
            const auto full = (std::int64_t(input.audio_frames) - 1) / 2 + 1;
            const auto prefix = input.audio_prefix_extra_frames / 2 + input.audio_prefix_extra_frames % 2;
            const auto suffix = input.audio_suffix_extra_frames / 2 + input.audio_suffix_extra_frames % 2;
            if (prefix >= full || suffix >= full - prefix || full - prefix - suffix < 5)
                throw std::invalid_argument("MiniCPM-o streaming audio chunk is too short");
            const auto tokens = full - prefix - suffix;
            auto past = state.audio_state.cache_length();
            if (past + full >= model_.audio->position_embedding.size(0)) past = 0;
            if (past + tokens > model_.audio->position_embedding.size(0))
                throw std::invalid_argument("MiniCPM-o streaming Whisper position cache exceeds capacity");
            const auto incoming_dtype = cuda::native_ops_detail::promote(
                model_.audio->conv2_weight.scalar_type(), model_.audio->position_embedding.scalar_type());
            for (std::size_t i = 0; i < model_.audio->layers.size(); ++i) {
                const auto& attention = model_.audio->layers[i].attention;
                const auto& cache = state.audio_state.layers[i].key_cache;
                const auto dtype = cache && past ? cuda::native_ops_detail::promote(cache.scalar_type(), incoming_dtype)
                                                 : incoming_dtype;
                const auto rows = static_cast<std::size_t>(past + tokens);
                const auto width = static_cast<std::size_t>(attention.heads * attention.head_dim);
                if (width && rows > std::numeric_limits<std::size_t>::max() / width / scalar_size(dtype) / 2)
                    throw std::overflow_error("MiniCPM audio cache growth overflow");
                add_growth(result, model_.audio->conv2_weight.device(), rows * width * scalar_size(dtype) * 2);
            }
        }
        if (!input.force_listen) {
            const auto& tts = *model_.tts;
            const auto tokens = std::max<std::int64_t>(1, input.max_new_speak_tokens - 1);
            for (const auto& layer : tts.layers) {
                const auto& attention = layer.attention;
                if (state.tts_state.kv.at(attention.layer).defined()) continue;
                const auto capacity = mode.kv_capacity > 0 ? std::max(state.tts_state.position + tokens, mode.kv_capacity)
                                                         : attention.max_position_embeddings;
                add_growth(result, Device(kCUDA, tts.device),
                    tyloi::state::kv_bytes(1, attention.kv_heads, capacity, attention.attention_head_dim,
                        false, kFloat16, attention.kv_quantization));
            }
        }
        return result;
    }

void Duplex::start(const MfqDuplexSessionParams& parameters, const ForwardOptions& mode) {
        validate(model_, parameters);
        auto policy = Executor::current_ffn();
        policy.independent |= model_.info().separate_projections;
        auto scope = Executor::current().enter(policy);
        MfqCudaGuard guard(model_.primary);
        cuda::manual_seed(static_cast<int64_t>(parameters.seed));
        mfq_cuda_manual_seed_all(parameters.seed);
        auto ids = MiniCPMO45DuplexSpecialIds::from_tensor(cuda::tensor(parameters.special_ids,
            TensorOptions().dtype(kInt64)));
        if (session_) mfq_cuda_synchronize();
        session_.reset();
        session_.emplace(model_, mode, ids, parameters.forbidden_ids, parameters.greedy);
        auto& s = *session_;
        s.temperature = parameters.temperature; s.top_k = parameters.top_k; s.top_p = parameters.top_p;
        s.listen_probability_scale = parameters.listen_probability_scale;
        s.repetition_penalty = parameters.repetition_penalty; s.repetition_window = parameters.repetition_window;
        s.length_penalty = parameters.length_penalty; s.tts_temperature = parameters.tts_temperature;
        s.tts_repetition_penalty = parameters.tts_repetition_penalty;
        parameters_ = parameters;
        prepare();
        mfq_cuda_synchronize();
    }
MfqDuplexStepResult Duplex::step(const MfqDuplexStepInput& input, const ForwardOptions& mode) {
        validate(input);
        StepInput value;
        value.audio_prefix_extra_frames = input.audio_prefix_extra_frames;
        value.audio_suffix_extra_frames = input.audio_suffix_extra_frames;
        value.max_new_speak_tokens = input.max_new_speak_tokens;
        value.force_listen = input.force_listen; value.force_speak = input.force_speak;
        if (input.audio_frames > 0) value.audio_features = cuda::tensor(input.audio_features,
            TensorOptions().dtype(kFloat32)).reshape({1, 80, input.audio_frames});
        if (!input.text_tokens.empty()) value.text_ids = cuda::tensor(input.text_tokens,
            TensorOptions().dtype(kInt64)).reshape({1, static_cast<int64_t>(input.text_tokens.size())});
        return step(std::move(value), mode).result;
    }
Duplex::StepOutput Duplex::step(StepInput input, const ForwardOptions& mode) {
        auto policy = Executor::current_ffn();
        policy.independent |= model_.info().separate_projections;
        auto scope = Executor::current().enter(policy);
        MfqCudaGuard guard(model_.primary);
        if (!session_) throw std::runtime_error("MiniCPM-o duplex session is not prepared");
        session_->mode = mode;
        const auto started = std::chrono::steady_clock::now();
        auto result = session_->run_step(input.pixels, input.patch_mask, input.target_sizes, input.image_slice_counts,
            input.audio_features, input.audio_prefix_extra_frames, input.audio_suffix_extra_frames,
            input.text_ids, input.max_new_speak_tokens, input.force_listen, input.force_speak);
        StepOutput output;
        output.decision_logits = result.decision_logits; output.audio_embeddings = result.audio_embeddings;
        output.generated_ids = result.generated_ids; output.tts_codes = result.tts_codes;
        output.result.generated_tokens = ops::read_integers(result.generated_ids);
        const auto codes = ops::read_integers(result.tts_codes);
        output.result.audio_tokens.reserve(codes.size());
        for (auto code : codes) output.result.audio_tokens.push_back(static_cast<int32_t>(code));
        output.result.is_listen = result.is_listen; output.result.end_of_turn = result.end_of_turn;
        output.result.tts_force_flush = result.tts_force_flush;
        output.result.audio_chunk_index = session_->audio_chunk_index;
        output.result.language_cache_position = session_->language_state.cache_pos;
        output.result.audio_cache_position = session_->audio_state.cache_length();
        output.result.tts_cache_position = session_->tts_state.position;
        output.result.inference_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        return output;
    }
void Duplex::reset(const ForwardOptions& mode) {
        if (!session_) throw std::runtime_error("MiniCPM-o duplex session is not prepared");
        auto policy = Executor::current_ffn();
        policy.independent |= model_.info().separate_projections;
        auto scope = Executor::current().enter(policy);
        MfqCudaGuard guard(model_.primary);
        session_->mode = mode;
        mfq_cuda_synchronize();
        prepare();
        mfq_cuda_synchronize();
    }
void Duplex::stop() {
        MfqCudaGuard guard(model_.primary);
        mfq_cuda_synchronize();
        session_.reset(); parameters_.reset();
        mfq_cuda_synchronize();
    }
void Duplex::prepare() {
        const auto ids = [](const std::vector<int64_t>& values) -> Tensor {
            return values.empty() ? Tensor{} : Tensor(cuda::tensor(values, TensorOptions().dtype(kInt64)));
        };
        Tensor reference;
        if (!parameters_->reference_audio_features.empty())
            reference = cuda::tensor(parameters_->reference_audio_features, TensorOptions().dtype(kFloat32))
                .reshape({1, 80, parameters_->reference_audio_frames});
        session_->prepare(ids(parameters_->system_prefix), reference, ids(parameters_->system_suffix));
    }

} // namespace tyloi::cuda::minicpmo45

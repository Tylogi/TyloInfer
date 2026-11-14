#include "eval/minicpmo45.h"
#include "eval/tensors.h"
#include "engine/program.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/models/causal_model.h"
#include "engine/models/minicpmo45/media.h"
#include "engine/models/minicpmo45/text/model.h"
#include "engine/models/minicpmo45/text/state.h"
#include "engine/models/minicpmo45/vision.h"
#include "engine/models/minicpmo45/audio.h"
#include "engine/models/minicpmo45/tts.h"
#include "engine/ops.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

namespace tyloi::eval {
using tyloi::indexing::Slice;

namespace {
struct Work {
    cuda::Program& program;
    std::vector<Tensor> tensors;
    std::vector<std::unique_ptr<MiniCPMO45AudioEncoder::State>> audio;
    std::vector<std::unique_ptr<cuda::minicpmo45::Tts::State>> speech;
    std::vector<StepSequence<Tensor>> steps;
    std::vector<std::unique_ptr<std::mt19937>> rng;
    Tensor keep(Tensor value) {
        if (value.defined()) tensors.push_back(value);
        return value;
    }
    DenseTensor keep(DenseTensor value) {
        if (value.defined()) tensors.emplace_back(value);
        return value;
    }
    /** Retire only tensor handles after an existing synchronous observation. */
    void clear_completed_tensors() { tensors.clear(); }
    void retire() {
        program.drain();
        steps.clear();
        speech.clear();
        audio.clear();
        rng.clear();
        tensors.clear();
    }
    ~Work() noexcept {
        try { program.drain(); }
        catch (const std::exception& error) { std::cerr << "minicpmo_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "minicpmo_cleanup error=unknown\n"; }
        steps.clear();
        speech.clear();
        audio.clear();
        rng.clear();
        tensors.clear();
    }
};

Tensor finish_media(Work& work, StepSequence<Tensor> sequence) {
    work.steps.push_back(std::move(sequence));
    Tensor result;
    while (auto step = work.steps.back().next()) {
        if (step.layer || step.graph || step.workspace) throw std::logic_error("unsupported media resource event");
        if (step.value) result = std::move(*step.value);
    }
    if (!result) throw std::runtime_error("MiniCPM-o media encoding produced no result");
    return work.keep(std::move(result));
}

DenseTensor forward(Work& work, DenseTensor ids, DenseTensor embeddings = {}, DenseTensor mask = {}) {
    models::Input input;
    input.ids = work.keep(Tensor(std::move(ids)));
    if (embeddings.defined()) input.embeddings = work.keep(Tensor(std::move(embeddings)));
    auto& program = work.program;
    auto& state = dynamic_cast<cuda::minicpmo45::text::State&>(program.state());
    if (mask.defined()) state.attention_mask = work.keep(Tensor(std::move(mask)));
    else if (state.attention_mask) {
        const auto end = state.position() + input.ids.size(input.ids.dim() - 1);
        if (state.attention_mask.size(1) < end) {
            auto existing = state.attention_mask.dense();
            auto extra = work.keep(tyloi::cuda::ones({existing.size(0), end - existing.size(1)}, existing.options()));
            state.attention_mask = work.keep(tyloi::cuda::cat({existing, extra}, 1));
        }
    }
    const int primary = program.config().layer_placement.primary_device();
    const auto mode = program.executor().forward_mode(primary);
    const models::ForwardOptions options{mode.kv_capacity, mode.trace, mode.trace_layer, primary};
    return work.keep(program.model().forward(input, state, options)).dense();
}

DenseTensor model_logits(Work& work, DenseTensor hidden) {
    auto input = work.keep(Tensor(std::move(hidden)));
    return work.keep(work.program.model().logits(input)).dense();
}
} // namespace
static std::vector<tyloi::models::minicpmo45::MediaBound> parse_bounds(
        tyloi::DenseTensor bounds, const char* label) {
    if (!bounds.defined() || bounds.numel() == 0) return {};
    if (bounds.dim() != 2 || bounds.size(1) != 4)
        throw std::runtime_error(std::string("MiniCPM-o ") + label + " bounds must have shape [count,4]");
    bounds = bounds.to(tyloi::kCPU, tyloi::kInt64).contiguous();
    return tyloi::models::minicpmo45::media_bounds(
        std::span<const int64_t>(bounds.data_ptr<int64_t>(), bounds.numel()));
}

static tyloi::DenseTensor minicpmo45_eval_embeddings(
    Work& work, const tyloi::models::CausalModel& model, int primary, tyloi::DenseTensor input_ids,
    const tyloi::DenseTensor &image_embeddings, tyloi::DenseTensor image_bounds,
    const tyloi::DenseTensor &audio_embeddings = tyloi::DenseTensor(),
    tyloi::DenseTensor audio_bounds = tyloi::DenseTensor(),
    tyloi::DenseTensor audio_valid_lengths = tyloi::DenseTensor()) {
    if (input_ids.dim() == 1)
        input_ids = input_ids.unsqueeze(0);
    if (input_ids.dim() != 2 || input_ids.size(0) != 1) {
        throw std::runtime_error("MiniCPM-o eval batch input_ids must have shape [1,tokens]");
    }
    work.keep(input_ids);
    input_ids = work.keep(input_ids.to(tyloi::Device(tyloi::kCUDA, primary), tyloi::kInt64).contiguous());
    auto embeddings = work.keep(model.embedding(input_ids, primary)).dense();
    const auto bounds = parse_bounds(image_bounds, "image");
    const int64_t image_count = image_embeddings.defined() ? image_embeddings.size(0) : 0;
    if (bounds.size() != static_cast<size_t>(image_count)) {
        throw std::runtime_error("MiniCPM-o eval batch requires one image bound per frame");
    }
    for (const auto &bound : bounds) {
        if (bound.batch != 0 || bound.source >= image_embeddings.size(0) ||
            bound.end > input_ids.size(1) || bound.end - bound.begin != image_embeddings.size(1)) {
            throw std::runtime_error("MiniCPM-o eval image bound does not match resampler output");
        }
        auto source = work.keep(image_embeddings.index({bound.source}));
        auto converted = work.keep(source.to(embeddings.scalar_type()));
        embeddings.index({0, Slice(bound.begin, bound.end), Slice()}).copy_(converted);
    }
    const auto audios = parse_bounds(audio_bounds, "audio");
    if (!audios.empty()) {
        if (!audio_embeddings.defined() || !audio_valid_lengths.defined()) {
            throw std::runtime_error("MiniCPM-o eval batch audio bounds require audio embeddings");
        }
        audio_valid_lengths =
            audio_valid_lengths.to(tyloi::kCPU, tyloi::kInt64)
                .contiguous()
                .reshape({-1});
        if (audio_valid_lengths.numel() != audio_embeddings.size(0)) {
            throw std::runtime_error("MiniCPM-o eval batch audio length count is invalid");
        }
        const auto *valid_lengths = audio_valid_lengths.data_ptr<int64_t>();
        std::vector<bool> used_audio(static_cast<size_t>(audio_embeddings.size(0)), false);
        for (const auto &bound : audios) {
            const int64_t required = bound.end - bound.begin;
            if (bound.batch != 0 || bound.source < 0 || bound.source >= audio_embeddings.size(0) ||
                bound.end > input_ids.size(1)) {
                throw std::runtime_error(
                    "MiniCPM-o eval audio bound does not match Whisper output");
            }
            const int64_t available = valid_lengths[bound.source];
            if (available < 0 || available > audio_embeddings.size(1) || required != available) {
                throw std::runtime_error(
                    "MiniCPM-o eval audio bound length does not match its source");
            }
            auto source = work.keep(audio_embeddings.index({bound.source, Slice(0, available), Slice()}));
            auto converted = work.keep(source.to(embeddings.scalar_type()));
            embeddings.index({0, Slice(bound.begin, bound.end), Slice()}).copy_(converted);
            used_audio[static_cast<size_t>(bound.source)] = true;
        }
        if (std::find(used_audio.begin(), used_audio.end(), false) != used_audio.end()) {
            throw std::runtime_error("MiniCPM-o eval batch has unused Whisper segments");
        }
    }
    return embeddings;
}

static bool minicpmo45_contains_token(const std::vector<int64_t> &values, int64_t token) {
    return std::find(values.begin(), values.end(), token) != values.end();
}

static std::vector<std::pair<int64_t, int64_t>>
minicpmo45_prefill_segments(tyloi::DenseTensor boundaries, int64_t range_begin,
                            int64_t range_end, int64_t chunk_tokens) {
    if (range_begin < 0 || range_end <= range_begin || chunk_tokens < 0) {
        throw std::runtime_error("MiniCPM-o prefill segment range is invalid");
    }
    boundaries = boundaries.to(tyloi::kCPU, tyloi::kInt64)
                     .contiguous()
                     .reshape({-1});
    if (boundaries.numel() == 0) {
        throw std::runtime_error("MiniCPM-o segmented prefill requires boundaries");
    }
    const auto *values = boundaries.data_ptr<int64_t>();
    std::vector<int64_t> eligible;
    eligible.reserve(static_cast<size_t>(boundaries.numel()));
    bool found_begin = range_begin == 0;
    bool found_end = false;
    int64_t previous = 0;
    for (int64_t index = 0; index < boundaries.numel(); ++index) {
        const int64_t boundary = values[index];
        if (boundary <= previous) {
            throw std::runtime_error("MiniCPM-o prefill boundaries must be increasing");
        }
        previous = boundary;
        if (boundary == range_begin)
            found_begin = true;
        if (boundary > range_begin && boundary <= range_end) {
            eligible.push_back(boundary);
        }
        if (boundary == range_end)
            found_end = true;
    }
    if (!found_begin || !found_end) {
        throw std::runtime_error("MiniCPM-o segmented prefill omits a prompt span");
    }

    std::vector<std::pair<int64_t, int64_t>> segments;
    int64_t segment_begin = range_begin;
    for (const int64_t boundary : eligible) {
        if (chunk_tokens == 0 || boundary == range_end ||
            boundary - segment_begin >= chunk_tokens) {
            segments.emplace_back(segment_begin, boundary);
            segment_begin = boundary;
        }
    }
    if (segments.empty() || segments.back().second != range_end) {
        throw std::runtime_error("MiniCPM-o segmented prefill omits a prompt span");
    }
    return segments;
}

static std::vector<std::pair<int64_t, int64_t>>
minicpmo45_teacher_prefill_segments(tyloi::DenseTensor boundaries, int64_t range_end,
                                    int64_t text_begin, int64_t text_end, int64_t chunk_tokens) {
    if (range_end <= 0 || text_begin < 0 || text_end <= text_begin || text_end > range_end ||
        chunk_tokens < 0) {
        throw std::runtime_error("MiniCPM-o teacher-forcing prefill range is invalid");
    }
    const auto exact = minicpmo45_prefill_segments(boundaries, 0, range_end, 0);
    const auto text = std::make_pair(text_begin, text_end);
    if (std::find(exact.begin(), exact.end(), text) == exact.end()) {
        throw std::runtime_error("MiniCPM-o teacher-forcing splits omit the text span");
    }
    if (chunk_tokens == 0)
        return exact;

    std::vector<std::pair<int64_t, int64_t>> segments;
    const auto append = [&](const auto &parts) {
        segments.insert(segments.end(), parts.begin(), parts.end());
    };
    if (text_begin > 0) {
        append(minicpmo45_prefill_segments(boundaries, 0, text_begin, chunk_tokens));
    }
    segments.emplace_back(text);
    if (text_end < range_end) {
        append(minicpmo45_prefill_segments(boundaries, text_end, range_end, chunk_tokens));
    }
    return segments;
}

int minicpmo45_batch(cuda::Program& program,
                              int64_t context_size,
                              int64_t vision_batch_size) {
    tyloi::cuda::NoGradGuard no_grad;
    if (context_size <= 0)
        context_size = 8192;
    if (vision_batch_size <= 0)
        vision_batch_size = 16;
    auto scope = program.executor().enter(program.config().layer_placement.primary_device());
    program.load(true);
    auto& model = dynamic_cast<cuda::minicpmo45::text::Model&>(program.model());
    auto& state = dynamic_cast<cuda::minicpmo45::text::State&>(program.state());
    auto policy = cuda::Executor::current_ffn();
    policy.independent |= model.info().separate_projections;
    auto model_scope = program.executor().enter(policy);
    if (!model.vision || !model.resampler || !model.audio || !model.tts)
        throw std::runtime_error("MiniCPM-o media weights are unavailable");
    if (model.info().max_context < context_size)
        throw std::invalid_argument("MiniCPM-o eval context exceeds the loaded model context");
    const int primary = program.config().layer_placement.primary_device();
    const tyloi::Device device(tyloi::kCUDA, primary);
    Work work{program};
    std::cout << "minicpmo45_eval_batch=ready"
              << " context_size=" << context_size << " vision_batch_size=" << vision_batch_size
              << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty())
            continue;
        auto request = nlohmann::json::parse(line, nullptr, false);
        if (!request.is_object()) {
            throw std::runtime_error("MiniCPM-o eval request must be a JSON object");
        }
        if (request.value("shutdown", false))
            break;
        const std::string request_id = request.at("request_id").get<std::string>();
        const std::string input_prefix = request.at("input_prefix").get<std::string>();
        const std::string output_prefix = request.value("output_prefix", std::string{});
        const int64_t question_count = request.at("question_count").get<int64_t>();
        const int64_t prefix_length = request.at("common_prefix_length").get<int64_t>();
        const bool reuse_prefix_cache = request.value("reuse_prefix_cache", true);
        const bool require_segmented_prefill = request.value("require_segmented_prefill", false);
        const int64_t segmented_prefill_chunk_tokens =
            request.value("segmented_prefill_chunk_tokens", int64_t{0});
        const int64_t max_new_tokens = request.value("max_new_tokens", int64_t{128});
        const double repetition_penalty = request.value("repetition_penalty", 1.02);
        const bool capture_logits = request.value("capture_logits", false);
        const bool ignore_eos = request.value("ignore_eos", false);
        const bool generate_tts = request.value("generate_tts", false);
        const bool tts_teacher_forcing = request.value("tts_teacher_forcing", false);
        const int64_t tts_max_steps = request.value("tts_max_steps", int64_t{2048});
        const double tts_temperature = request.value("tts_temperature", 0.8);
        const double tts_top_p = request.value("tts_top_p", 0.85);
        const int64_t tts_top_k = request.value("tts_top_k", int64_t{25});
        const double tts_min_p = request.value("tts_min_p", 0.01);
        const double tts_repetition_penalty = request.value("tts_repetition_penalty", 1.05);
        const int64_t tts_minimum_steps = request.value("tts_minimum_steps", int64_t{0});
        const int64_t tts_min_tokens_to_keep = request.value("tts_min_tokens_to_keep", int64_t{3});
        const int64_t seed = request.value("seed", int64_t{0});
        const auto eos_token_ids = request.at("eos_token_ids").get<std::vector<int64_t>>();
        if (question_count <= 0 || prefix_length < 0 ||
            (reuse_prefix_cache && prefix_length <= 0) || max_new_tokens <= 0 ||
            repetition_penalty <= 0.0 || segmented_prefill_chunk_tokens < 0 ||
            (segmented_prefill_chunk_tokens > 0 && !require_segmented_prefill) ||
            eos_token_ids.empty()) {
            throw std::runtime_error("MiniCPM-o eval request has invalid generation settings");
        }
        if (generate_tts && (question_count != 1 || tts_max_steps <= 0 || tts_temperature <= 0.0 ||
                             tts_top_p <= 0.0 || tts_top_p > 1.0 || tts_top_k < 3 ||
                             tts_top_k > 6562 || tts_min_p < 0.0 || tts_min_p > 1.0 ||
                             tts_repetition_penalty <= 0.0 || tts_minimum_steps < 0 ||
                             tts_min_tokens_to_keep < 0 || tts_min_tokens_to_keep > tts_top_k)) {
            throw std::runtime_error(
                "MiniCPM-o TTS batch requires one question and positive steps");
        }
        if (tts_teacher_forcing && !generate_tts) {
            throw std::runtime_error("MiniCPM-o teacher forcing requires TTS generation");
        }
        const auto wall_started = std::chrono::steady_clock::now();
        auto pixels = read_tensor(input_prefix + ".pixel_values.pt", false);
        auto patch_mask = read_tensor(input_prefix + ".patch_mask.pt", false);
        auto target_sizes = read_tensor(input_prefix + ".target_sizes.pt", false);
        if (target_sizes.defined()) {
            target_sizes =
                target_sizes.to(tyloi::kCPU, tyloi::kInt64).contiguous();
        }
        auto audio_features = read_tensor(input_prefix + ".audio_features.pt", false);
        auto audio_lengths = read_tensor(input_prefix + ".audio_lengths.pt", false);
        if (pixels.defined() != patch_mask.defined() ||
            pixels.defined() != target_sizes.defined() ||
            (pixels.defined() &&
             (pixels.dim() != 4 || pixels.size(1) != 3 || patch_mask.dim() != 2 ||
              target_sizes.dim() != 2 || pixels.size(0) != patch_mask.size(0) ||
              pixels.size(0) != target_sizes.size(0)))) {
            throw std::runtime_error("MiniCPM-o eval tensors have incompatible shapes");
        }
        if (audio_features.defined() != audio_lengths.defined() ||
            (audio_features.defined() &&
             (audio_features.dim() != 3 || audio_features.size(1) != 80 ||
              audio_lengths.dim() != 1 || audio_lengths.size(0) != audio_features.size(0)))) {
            throw std::runtime_error("MiniCPM-o eval batch audio tensors have incompatible shapes");
        }

        work.retire();
        const auto vision_started = std::chrono::steady_clock::now();
        std::vector<tyloi::DenseTensor> vision_parts;
        std::vector<tyloi::DenseTensor> image_embedding_parts;
        tyloi::DenseTensor image_embeddings;
        if (pixels.defined() && pixels.size(0) > 0) {
            for (int64_t begin = 0; begin < pixels.size(0); begin += vision_batch_size) {
                const int64_t count = std::min<int64_t>(vision_batch_size, pixels.size(0) - begin);
                auto host_pixels = work.keep(pixels.narrow(0, begin, count));
                auto device_pixels = work.keep(host_pixels.to(device));
                auto batch_mask = work.keep(patch_mask.narrow(0, begin, count));
                auto batch_sizes = work.keep(target_sizes.narrow(0, begin, count));
                auto vision = finish_media(work, model.vision->forward_steps(
                    Tensor(device_pixels), Tensor(batch_mask), Tensor(batch_sizes)));
                auto image = work.keep(model.resampler->forward(vision, Tensor(batch_sizes)));
                vision_parts.push_back(vision.dense());
                image_embedding_parts.push_back(image.dense());
            }
            image_embeddings = work.keep(tyloi::cuda::cat(image_embedding_parts, 0).contiguous());
        }
        tyloi::DenseTensor audio_embeddings;
        tyloi::DenseTensor audio_valid_lengths;
        double audio_seconds = 0.0;
        if (audio_features.defined()) {
            const auto audio_started = std::chrono::steady_clock::now();
            auto host_audio_lengths =
                audio_lengths.to(tyloi::kCPU, tyloi::kInt64).contiguous();
            const auto *raw_lengths = host_audio_lengths.data_ptr<int64_t>();
            std::vector<int64_t> valid_length_values;
            for (int64_t i = 0; i < host_audio_lengths.numel(); ++i)
                valid_length_values.push_back(models::minicpmo45::pooled_audio_length(raw_lengths[i]));
            audio_valid_lengths = tyloi::cuda::tensor(
                valid_length_values,
                tyloi::TensorOptions().dtype(tyloi::kInt64));
            const int64_t max_valid_length =
                *std::max_element(valid_length_values.begin(), valid_length_values.end());
            std::vector<tyloi::DenseTensor> audio_parts;
            audio_parts.reserve(static_cast<size_t>(audio_features.size(0)));
            for (int64_t index = 0; index < audio_features.size(0); ++index) {
                auto host_part = work.keep(audio_features.index(
                    {Slice(index, index + 1), Slice(), Slice(0, raw_lengths[index])}));
                auto part_features = work.keep(host_part.to(device));
                auto part_length = work.keep(tyloi::cuda::tensor(
                    std::vector<int64_t>{raw_lengths[index]},
                    tyloi::TensorOptions().dtype(tyloi::kInt64)));
                work.audio.push_back(std::make_unique<MiniCPMO45AudioEncoder::State>(*model.audio));
                auto part = finish_media(work, model.audio->forward_steps(
                    Tensor(part_features), Tensor(part_length), *work.audio.back(), false)).dense();
                if (part.size(1) != valid_length_values[static_cast<size_t>(index)]) {
                    throw std::runtime_error("MiniCPM-o per-segment Whisper length mismatch");
                }
                if (part.size(1) < max_valid_length) {
                    part = work.keep(tyloi::cuda::constant_pad_nd(
                        part, {0, 0, 0, max_valid_length - part.size(1)}, 0.0));
                }
                audio_parts.push_back(part);
            }
            audio_embeddings = work.keep(tyloi::cuda::cat(audio_parts, 0).contiguous());
            work.retire();
            audio_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - audio_started)
                    .count();
        }
        work.retire();
        const double vision_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - vision_started)
                .count();
        pixels = tyloi::DenseTensor();
        patch_mask = tyloi::DenseTensor();
        vision_parts.clear();
        image_embedding_parts.clear();

        const auto question_prefix = [&](int64_t index) {
            std::ostringstream value;
            value << input_prefix << ".q" << std::setw(3) << std::setfill('0') << index;
            return value.str();
        };
        const auto first_prefix = question_prefix(0);
        auto first_host_ids = work.keep(read_tensor(first_prefix + ".input_ids.pt", true));
        auto first_ids = work.keep(first_host_ids.to(device, tyloi::kInt64).contiguous());
        if (first_ids.dim() == 1)
            first_ids = first_ids.unsqueeze(0);
        auto first_mask = work.keep(read_tensor(first_prefix + ".attention_mask.pt", true).to(device, tyloi::kBool).contiguous());
        if (first_mask.dim() == 1) first_mask = first_mask.unsqueeze(0);
        if (first_ids.dim() != 2 || first_ids.size(0) != 1 || first_mask.sizes() != first_ids.sizes())
            throw std::runtime_error("MiniCPM-o first prompt mask has incompatible shape");
        auto first_bounds = read_tensor(first_prefix + ".image_bounds.pt", false);
        auto first_audio_bounds = read_tensor(first_prefix + ".audio_bounds.pt", false);
        auto first_prefill_splits =
            read_tensor(first_prefix + ".prefill_splits.pt", require_segmented_prefill);
        auto first_embeddings =
            minicpmo45_eval_embeddings(work, model, primary, first_ids, image_embeddings, first_bounds,
                                       audio_embeddings, first_audio_bounds, audio_valid_lengths);
        if (prefix_length > first_ids.size(1)) {
            throw std::runtime_error("MiniCPM-o common prefix exceeds the first prompt");
        }
        double prefix_seconds = 0.0;
        int64_t prefix_prefill_calls = 0;
        TextSessionState prefix_snapshot;
        if (reuse_prefix_cache) {
            state.reset(1);
            state.attention_mask = {};
            auto prefix_ids = first_ids.narrow(1, 0, prefix_length);
            auto prefix_embeddings = first_embeddings.narrow(1, 0, prefix_length);
            auto prefix_mask = first_mask.narrow(1, 0, prefix_length);
            work.retire();
            const auto prefix_started = std::chrono::steady_clock::now();
            tyloi::DenseTensor prefix_hidden;
            if (require_segmented_prefill) {
                const auto segments = minicpmo45_prefill_segments(
                    first_prefill_splits, 0, prefix_length, segmented_prefill_chunk_tokens);
                for (const auto &segment : segments) {
                    const int64_t segment_begin = segment.first;
                    const int64_t segment_end = segment.second;
                    prefix_hidden = forward(work,
                        prefix_ids.narrow(1, segment_begin, segment_end - segment_begin),
                        prefix_embeddings.narrow(1, segment_begin, segment_end - segment_begin),
                        prefix_mask.narrow(1, 0, segment_end));
                    ++prefix_prefill_calls;
                }
            } else {
                prefix_hidden = forward(work, prefix_ids, prefix_embeddings, prefix_mask);
                prefix_prefill_calls = 1;
            }
            work.retire();
            prefix_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - prefix_started)
                    .count();
            if (state.position() != prefix_length) {
                throw std::runtime_error("MiniCPM-o prefix cache length mismatch");
            }
            auto host_prefix = first_host_ids.to(tyloi::kCPU, tyloi::kInt64).contiguous().reshape({-1});
            const auto* ids = host_prefix.data_ptr<int64_t>();
            prefix_snapshot = state.capture_text_session_state(std::vector<int64_t>(ids, ids + prefix_length));
            work.retire();
        }

        nlohmann::json results = nlohmann::json::array();
        for (int64_t question = 0; question < question_count; ++question) {
            const auto current_prefix = question_prefix(question);
            auto input_ids = first_ids;
            if (question != 0) {
                auto host_ids = work.keep(read_tensor(current_prefix + ".input_ids.pt", true));
                input_ids = work.keep(host_ids.to(device, tyloi::kInt64).contiguous());
            }
            auto image_bounds =
                question == 0 ? first_bounds
                              : read_tensor(current_prefix + ".image_bounds.pt", false);
            auto audio_bounds =
                question == 0 ? first_audio_bounds
                              : read_tensor(current_prefix + ".audio_bounds.pt", false);
            auto host_mask = work.keep(read_tensor(current_prefix + ".attention_mask.pt", true));
            auto attention_mask = work.keep(host_mask.to(device, tyloi::kBool).contiguous());
            auto tts_bound =
                read_tensor(current_prefix + ".tts_bound.pt", tts_teacher_forcing);
            auto prefill_splits =
                read_tensor(current_prefix + ".prefill_splits.pt",
                                       tts_teacher_forcing || require_segmented_prefill);
            if (input_ids.dim() == 1)
                input_ids = input_ids.unsqueeze(0);
            if (attention_mask.dim() == 1) {
                attention_mask = attention_mask.unsqueeze(0);
            }
            if (input_ids.size(0) != 1 || attention_mask.sizes() != input_ids.sizes() ||
                (reuse_prefix_cache && prefix_length >= input_ids.size(1)) ||
                input_ids.size(1) + max_new_tokens > context_size) {
                throw std::runtime_error("MiniCPM-o eval prompt exceeds the configured context");
            }
            if (reuse_prefix_cache &&
                (!input_ids.narrow(1, 0, prefix_length).eq(first_ids.narrow(1, 0, prefix_length)).all().item<bool>() ||
                 !attention_mask.narrow(1, 0, prefix_length).eq(first_mask.narrow(1, 0, prefix_length)).all().item<bool>()))
                throw std::runtime_error("MiniCPM-o shared prefix tokens or mask differ");
            auto input_embeddings =
                question == 0 ? first_embeddings
                              : minicpmo45_eval_embeddings(work, model, primary, input_ids, image_embeddings,
                                                           image_bounds, audio_embeddings,
                                                           audio_bounds, audio_valid_lengths);
            if (reuse_prefix_cache) {
                program.restore(prefix_snapshot);
            } else {
                state.reset(1);
                state.attention_mask = {};
            }
            const int64_t evaluated_prefix_length = reuse_prefix_cache ? prefix_length : 0;
            const int64_t suffix_length = input_ids.size(1) - evaluated_prefix_length;
            auto suffix_ids = input_ids.narrow(1, evaluated_prefix_length, suffix_length);
            auto suffix_embeddings =
                input_embeddings.narrow(1, evaluated_prefix_length, suffix_length);
            work.retire();
            const auto generation_started = std::chrono::steady_clock::now();
            tyloi::DenseTensor hidden;
            tyloi::DenseTensor teacher_text_hidden;
            tyloi::DenseTensor logits;
            int64_t suffix_prefill_calls = 0;
            if (tts_teacher_forcing) {
                if (reuse_prefix_cache || evaluated_prefix_length != 0) {
                    throw std::runtime_error(
                        "MiniCPM-o teacher forcing cannot reuse a prefix cache");
                }
                tts_bound = tts_bound.to(tyloi::kCPU, tyloi::kInt64)
                                .contiguous()
                                .reshape({-1});
                prefill_splits =
                    prefill_splits.to(tyloi::kCPU, tyloi::kInt64)
                        .contiguous()
                        .reshape({-1});
                if (tts_bound.numel() != 2 || prefill_splits.numel() < 2) {
                    throw std::runtime_error("MiniCPM-o teacher-forcing boundaries are invalid");
                }
                const int64_t text_begin = tts_bound[0].item<int64_t>();
                const int64_t text_end = tts_bound[1].item<int64_t>();
                const auto segments = minicpmo45_teacher_prefill_segments(
                    prefill_splits, input_ids.size(1), text_begin, text_end,
                    segmented_prefill_chunk_tokens);
                bool found_text = false;
                for (const auto &segment : segments) {
                    const int64_t segment_begin = segment.first;
                    const int64_t segment_end = segment.second;
                    auto segment_hidden = forward(work,
                        input_ids.narrow(1, segment_begin, segment_end - segment_begin),
                        input_embeddings.narrow(1, segment_begin, segment_end - segment_begin),
                        attention_mask.narrow(1, 0, segment_end));
                    if (segment_begin == text_begin && segment_end == text_end) {
                        teacher_text_hidden = segment_hidden;
                        found_text = true;
                    }
                    hidden = segment_hidden;
                    ++suffix_prefill_calls;
                }
                if (!found_text) {
                    throw std::runtime_error("MiniCPM-o teacher-forcing splits omit the text span");
                }
            } else if (require_segmented_prefill) {
                const auto segments =
                    minicpmo45_prefill_segments(prefill_splits, evaluated_prefix_length,
                                                input_ids.size(1), segmented_prefill_chunk_tokens);
                for (const auto &segment : segments) {
                    const int64_t segment_begin = segment.first;
                    const int64_t segment_end = segment.second;
                    hidden = forward(work,
                        input_ids.narrow(1, segment_begin, segment_end - segment_begin),
                        input_embeddings.narrow(1, segment_begin, segment_end - segment_begin),
                        attention_mask.narrow(1, 0, segment_end));
                    ++suffix_prefill_calls;
                }
                logits = model_logits(work, hidden.index({Slice(), -1, Slice()}));
            } else {
                hidden = forward(work, suffix_ids, suffix_embeddings, attention_mask);
                suffix_prefill_calls = 1;
                logits = model_logits(work, hidden.index({Slice(), -1, Slice()}));
            }
            work.retire();
            const double suffix_prefill_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - generation_started)
                    .count();
            const auto decode_started = std::chrono::steady_clock::now();
            double first_token_seconds = 0.0;
            auto counts = work.keep(tyloi::cuda::zeros({model.info().vocab_size},
                                                    tyloi::TensorOptions()
                                                        .device(device)
                                                        .dtype(tyloi::kInt32)));
            std::vector<int64_t> generated;
            std::vector<tyloi::DenseTensor> captured_logits;
            std::vector<tyloi::DenseTensor> generated_hidden;
            generated.reserve(static_cast<size_t>(max_new_tokens));
            if (capture_logits) {
                captured_logits.reserve(static_cast<size_t>(max_new_tokens));
            }
            tyloi::DenseTensor next;
            const auto token_limit = tts_teacher_forcing ? 0 : max_new_tokens;
            for (int64_t step = 0; step < token_limit; ++step) {
                if (capture_logits)
                    captured_logits.push_back(work.keep(logits.detach().to(tyloi::kCPU)));
                auto scores = work.keep(tyloi::Tensor(logits));
                if (!generated.empty() && repetition_penalty != 1.0)
                    tyloi::ops::apply_penalties(scores, tyloi::Tensor(counts), 0.0, 0.0, repetition_penalty);
                auto contiguous = work.keep(tyloi::ops::contiguous(scores));
                auto sample_input = work.keep(tyloi::ops::reshape(contiguous, {1, -1}));
                next = work.keep(tyloi::ops::sample_greedy(sample_input)).dense();
                const auto token = next.item<int64_t>();
                work.clear_completed_tensors();
                work.keep(logits);
                work.keep(counts);
                work.keep(next);
                if (step == 0)
                    first_token_seconds = std::chrono::duration<double>(
                                              std::chrono::steady_clock::now() - decode_started)
                                              .count();
                generated.push_back(token);
                if (!ignore_eos && minicpmo45_contains_token(eos_token_ids, token)) break;
                if (step + 1 == token_limit) break;
                auto count_handle = tyloi::Tensor(counts);
                auto next_ids = work.keep(next.contiguous());
                tyloi::ops::add_token_counts(count_handle, tyloi::Tensor(next_ids));
                auto next_hidden = forward(work, next.reshape({1, 1}));
                generated_hidden.push_back(next_hidden);
                logits = model_logits(work,
                    next_hidden.index({Slice(), -1, Slice()}));
            }
            tyloi::DenseTensor tts_codes;
            if (generate_tts) {
                tyloi::DenseTensor text_ids;
                tyloi::DenseTensor text_hidden;
                if (tts_teacher_forcing) {
                    const int64_t begin = tts_bound[0].item<int64_t>();
                    const int64_t end = tts_bound[1].item<int64_t>();
                    if (begin < evaluated_prefix_length || end <= begin ||
                        end > input_ids.size(1)) {
                        throw std::runtime_error("MiniCPM-o teacher-forcing span is invalid");
                    }
                    text_ids = input_ids.narrow(1, begin, end - begin);
                    text_hidden = teacher_text_hidden;
                } else {
                    if (generated_hidden.empty()) {
                        throw std::runtime_error(
                            "MiniCPM-o TTS generation produced no text condition");
                    }
                    constexpr int64_t kTtsTextEosToken = 151704;
                    const auto text_eos =
                        std::find(generated.begin(), generated.end(), kTtsTextEosToken);
                    const int64_t text_length = std::min<int64_t>(
                        static_cast<int64_t>(generated_hidden.size()),
                        static_cast<int64_t>(std::distance(generated.begin(), text_eos)));
                    if (text_length <= 0) {
                        throw std::runtime_error(
                            "MiniCPM-o TTS generation produced no text tokens");
                    }
                    text_ids = work.keep(tyloi::cuda::tensor(
                                   std::vector<int64_t>(generated.begin(),
                                                        generated.begin() + text_length),
                                   tyloi::TensorOptions()
                                       .device(device)
                                       .dtype(tyloi::kInt64))
                                   .reshape({1, text_length}));
                    auto combined = work.keep(tyloi::cuda::cat(generated_hidden, 1));
                    text_hidden = work.keep(combined.narrow(1, 0, text_length).contiguous());
                }
                work.keep(text_ids);
                work.keep(text_hidden);
                auto condition = work.keep(model.tts->condition(Tensor(text_ids), Tensor(text_hidden)));
                work.rng.push_back(std::make_unique<std::mt19937>(static_cast<std::mt19937::result_type>(seed)));
                work.speech.push_back(std::make_unique<cuda::minicpmo45::Tts::State>(*model.tts));
                const auto mode = program.executor().forward_mode(primary);
                const models::ForwardOptions options{mode.kv_capacity, mode.trace, mode.trace_layer, primary};
                auto codes = work.keep(model.tts->generate_official(*work.speech.back(), options, condition,
                    tts_max_steps, 6561, tts_minimum_steps, tts_temperature, tts_top_p, tts_top_k,
                    tts_repetition_penalty, nullptr, tts_min_p, work.rng.back().get(), tts_min_tokens_to_keep));
                tts_codes = work.keep(codes.dense().to(tyloi::kCPU, tyloi::kInt64).reshape({-1}));
                if (!output_prefix.empty()) {
                    std::ostringstream path;
                    path << output_prefix << ".q" << std::setw(3) << std::setfill('0') << question
                         << ".tts_codes.pt";
                    write_tensor(tts_codes, path.str());
                }
            }
            work.retire();
            const double generation_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - generation_started)
                    .count();
            const double decode_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_started)
                    .count();
            if (capture_logits && !output_prefix.empty()) {
                std::ostringstream path;
                path << output_prefix << ".q" << std::setw(3) << std::setfill('0') << question
                     << ".logits.pt";
                write_tensor(tyloi::cuda::cat(captured_logits, 0), path.str());
            }
            results.push_back({
                {"question_index", question},
                {"context_tokens", input_ids.size(1)},
                {"prefix_tokens", evaluated_prefix_length},
                {"suffix_tokens", suffix_length},
                {"suffix_prefill_calls", suffix_prefill_calls},
                {"generated_token_ids", generated},
                {"generation_seconds", generation_seconds},
                {"suffix_prefill_seconds", suffix_prefill_seconds},
                {"first_token_seconds", first_token_seconds},
                {"decode_seconds", decode_seconds},
                {"tts_code_count", tts_codes.defined() ? tts_codes.numel() : 0},
            });
            if (question == 0) {
                first_ids = tyloi::DenseTensor();
                first_bounds = tyloi::DenseTensor();
                first_audio_bounds = tyloi::DenseTensor();
                first_embeddings = tyloi::DenseTensor();
            }
        }
        work.retire();
        const auto stats = mfq_cuda_memory_stats(mfq_current_cuda_device());
        nlohmann::json response = {
            {"request_id", request_id},
            {"status", "ok"},
            {"frame_count", image_embeddings.defined() ? image_embeddings.size(0) : 0},
            {"audio_count", audio_embeddings.defined() ? audio_embeddings.size(0) : 0},
            {"vision_seconds", vision_seconds},
            {"audio_seconds", audio_seconds},
            {"prefix_seconds", prefix_seconds},
            {"prefix_prefill_calls", prefix_prefill_calls},
            {"segmented_prefill_chunk_tokens", segmented_prefill_chunk_tokens},
            {"wall_seconds",
             std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_started)
                 .count()},
            {"allocated_bytes", stats.allocated_bytes},
            {"reserved_bytes", stats.reserved_bytes},
            {"peak_allocated_bytes", stats.peak_allocated_bytes},
            {"peak_reserved_bytes", stats.peak_reserved_bytes},
            {"questions", results},
        };
        std::cout << "minicpmo45_eval_result=" << response.dump() << std::endl;
    }
    return 0;
}


} // namespace tyloi::eval

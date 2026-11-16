#include "minicpmo45.h"
#include "cli.h"
#include "engine/program.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/models/causal_model.h"
#include "engine/models/minicpmo45/duplex.h"
#include "engine/ops/ops.h"
#include "eval/tensors.h"
#include "eval/minicpmo45.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>

namespace {

using tyloi::indexing::Slice;

std::string minicpmo45_duplex_step_prefix(
        const std::string & prefix,
        int64_t step) {
    std::ostringstream stream;
    stream << prefix << ".step" << std::setfill('0')
           << std::setw(4) << step;
    return stream.str();
}

} // namespace

namespace tyloi::cuda::minicpmo45 {

bool parse_command_option(
        std::string_view option,
        tyloi::commands::ArgCursor& args,
        CommandOptions& result) {
    using tyloi::commands::integer;
    using tyloi::commands::usage_error;
    if (option == "--minicpmo-input-prefix") {
        result.input_prefix = args.value(option);
    } else if (option == "--minicpmo-output-prefix") {
        result.output_prefix = args.value(option);
    } else if (option == "--minicpmo-tts-steps") {
        result.tts_steps = integer<int64_t>(args.value(option), option);
        if (result.tts_steps < 0) {
            usage_error("--minicpmo-tts-steps must be non-negative");
        }
    } else if (option == "--minicpmo-duplex-input-prefix") {
        result.duplex_input_prefix = args.value(option);
    } else if (option == "--minicpmo-duplex-output-prefix") {
        result.duplex_output_prefix = args.value(option);
    } else if (option == "--minicpmo-duplex-steps") {
        result.duplex_steps = integer<int64_t>(args.value(option), option);
        if (result.duplex_steps < 0) {
            usage_error("--minicpmo-duplex-steps must be non-negative");
        }
    } else if (option == "--minicpmo-duplex-max-speak-tokens") {
        result.duplex_max_speak_tokens =
            integer<int64_t>(args.value(option), option);
        if (result.duplex_max_speak_tokens <= 0) {
            usage_error(
                "--minicpmo-duplex-max-speak-tokens must be positive");
        }
    } else if (option == "--minicpmo-duplex-seed") {
        result.duplex_seed = integer<int64_t>(args.value(option), option);
    } else if (option == "--minicpmo-duplex-greedy") {
        result.duplex_greedy = true;
    } else {
        return false;
    }
    return true;
}

void validate_command_options(const CommandOptions& options) {
    if (!options.output_prefix.empty() && options.input_prefix.empty()) {
        tyloi::commands::usage_error(
            "--minicpmo-output-prefix requires --minicpmo-input-prefix");
    }
    if (!options.duplex_output_prefix.empty() &&
            options.duplex_input_prefix.empty()) {
        tyloi::commands::usage_error(
            "--minicpmo-duplex-output-prefix requires "
            "--minicpmo-duplex-input-prefix");
    }
}

using tyloi::eval::read_tensor;
using tyloi::eval::write_tensor;
using tyloi::eval::read_scalar;

int run_composite(Program& program, const CommandOptions& options) {
    if (options.input_prefix.empty() || options.output_prefix.empty())
        throw std::runtime_error("MiniCPM-o composite graph requires input and output prefixes");
    tyloi::cuda::NoGradGuard no_grad;
    auto scope = program.executor().enter(program.config().layer_placement.primary_device());
    program.load(true);
    auto& model = dynamic_cast<text::Model&>(program.model());
    auto& state = dynamic_cast<text::State&>(program.state());
    auto policy = Executor::current_ffn();
    policy.independent |= model.info().separate_projections;
    auto model_scope = program.executor().enter(policy);
    if (!model.vision || !model.resampler || !model.audio || !model.tts)
        throw std::runtime_error("MiniCPM-o media weights are unavailable");
    const int primary = program.config().layer_placement.primary_device();
    const auto mode = program.executor().forward_mode(primary);
    const tyloi::models::ForwardOptions forward_options{mode.kv_capacity, mode.trace, mode.trace_layer, primary};
    struct Work {
        Program& program;
        std::vector<tyloi::Tensor> tensors, traces;
        std::unique_ptr<MiniCPMO45AudioEncoder::State> audio_state;
        std::unique_ptr<Tts::State> tts_state;
        tyloi::StepSequence<tyloi::Tensor> vision, audio;
        ~Work() noexcept {
            try { program.drain(); }
            catch (const std::exception& error) { std::cerr << "minicpmo_cleanup error=" << error.what() << '\n'; }
            catch (...) { std::cerr << "minicpmo_cleanup error=unknown\n"; }
        }
        tyloi::Tensor keep(tyloi::Tensor value) { tensors.push_back(value); return value; }
    } work{program};
    const auto& in = options.input_prefix;
    const auto& out = options.output_prefix;
    auto ids = read_tensor(in + ".input_ids.pt", true);
    if (ids.dim() == 1) ids = ids.unsqueeze(0);
    ids = ids.to(tyloi::Device(tyloi::kCUDA, primary), tyloi::kInt64).contiguous();
    const auto id_tensor = work.keep(ids);
    auto embeddings = work.keep(model.embedding(id_tensor, primary));
    auto pixels = work.keep(read_tensor(in + ".pixel_values.pt", false));
    auto patch_mask = work.keep(read_tensor(in + ".patch_mask.pt", false));
    auto target_sizes = work.keep(read_tensor(in + ".target_sizes.pt", false));
    auto image_bounds = work.keep(read_tensor(in + ".image_bounds.pt", false));
    auto audio_features = work.keep(read_tensor(in + ".audio_features.pt", false));
    auto audio_lengths = work.keep(read_tensor(in + ".audio_lengths.pt", false));
    auto audio_bounds = work.keep(read_tensor(in + ".audio_bounds.pt", false));
    auto mask = work.keep(read_tensor(in + ".attention_mask.pt", false));
    auto positions = work.keep(read_tensor(in + ".position_ids.pt", false));
    const auto bounds = [](const tyloi::Tensor& value) {
        if (!value) return std::vector<tyloi::models::minicpmo45::MediaBound>{};
        auto host = value.dense().to(tyloi::kCPU, tyloi::kInt64).contiguous();
        if (host.dim() != 2 || host.size(1) != 4)
            throw std::runtime_error("MiniCPM-o media bounds must have shape [count,4]");
        return tyloi::models::minicpmo45::media_bounds(
            std::span<const int64_t>(host.data_ptr<int64_t>(), host.numel()));
    };
    const auto images = bounds(image_bounds), audios = bounds(audio_bounds);
    const auto finish = [&](tyloi::StepSequence<tyloi::Tensor>& sequence) {
        tyloi::Tensor result;
        while (auto step = sequence.next()) {
            if (step.workspace || step.graph || step.layer)
                throw std::logic_error("MiniCPM-o media request requires unsupported resource handling");
            if (step.value) result = std::move(*step.value);
        }
        if (!result) throw std::runtime_error("MiniCPM-o media request produced no result");
        return work.keep(std::move(result));
    };
    const auto scatter = [&](const tyloi::Tensor& encoded,
                             const tyloi::models::minicpmo45::MediaBound& bound, int64_t length) {
        if (bound.batch >= ids.size(0) || bound.source >= encoded.size(0) ||
            bound.end > ids.size(1) || bound.end - bound.begin != length)
            throw std::runtime_error("MiniCPM-o media bound does not match encoded length");
        auto& target = embeddings.dense();
        target.index({bound.batch, Slice(bound.begin, bound.end), Slice()}).copy_(
            encoded.dense().index({bound.source, Slice(0, length), Slice()}).to(target.scalar_type()));
    };
    tyloi::Tensor vision_states, image_embeddings, audio_embeddings;
    if (!images.empty()) {
        if (!pixels || !patch_mask || !target_sizes)
            throw std::runtime_error("image bounds require image tensors");
        pixels = work.keep(tyloi::ops::copy(pixels, tyloi::Device(tyloi::kCUDA, primary)));
        work.vision = model.vision->forward_steps(pixels, patch_mask, target_sizes);
        vision_states = finish(work.vision);
        image_embeddings = work.keep(model.resampler->forward(vision_states, target_sizes));
        for (const auto& bound : images) scatter(image_embeddings, bound, image_embeddings.size(1));
    }
    if (!audios.empty()) {
        if (!audio_features || !audio_lengths) throw std::runtime_error("audio bounds require audio tensors");
        audio_features = work.keep(tyloi::ops::copy(audio_features, tyloi::Device(tyloi::kCUDA, primary)));
        work.audio_state = std::make_unique<MiniCPMO45AudioEncoder::State>(*model.audio);
        work.audio = model.audio->forward_steps(audio_features, audio_lengths, *work.audio_state, false);
        audio_embeddings = finish(work.audio);
        const auto lengths = MiniCPMO45AudioEncoder::pooled_lengths(audio_lengths);
        for (const auto& bound : audios) {
            if (bound.source < 0 || static_cast<size_t>(bound.source) >= lengths.size())
                throw std::runtime_error("audio bound source is out of range");
            scatter(audio_embeddings, bound, lengths[bound.source]);
        }
    }
    state.reset(ids.size(0));
    state.attention_mask = mask;
    tyloi::models::Input input;
    input.ids = id_tensor;
    input.embeddings = embeddings;
    input.positions = positions;
    input.advance = positions.defined();
    auto hidden = work.keep(model.forward(input, state, forward_options));
    auto logits = work.keep(model.logits(hidden));
    const auto save = [&](const tyloi::Tensor& value, const char* name) {
        if (value.defined()) write_tensor(value.dense(), out + name);
    };
    save(vision_states, ".vision_states.pt");
    save(image_embeddings, ".image_embeddings.pt");
    save(audio_embeddings, ".audio_embeddings.pt");
    save(embeddings, ".input_embeddings.pt");
    save(hidden, ".hidden_states.pt");
    save(logits, ".logits.pt");
    if (options.tts_steps > 0) {
        auto raw = work.keep(read_tensor(in + ".tts_inputs_embeds.pt", false));
        tyloi::Tensor condition;
        if (!raw) {
            auto bound = read_tensor(in + ".tts_bound.pt", true).to(tyloi::kCPU, tyloi::kInt64).contiguous();
            if (bound.dim() != 1 || bound.numel() != 2)
                throw std::runtime_error("MiniCPM-o tts_bound must contain [begin,end]");
            const auto begin = bound[0].item<int64_t>(), end = bound[1].item<int64_t>();
            if (begin < 0 || end <= begin || end > ids.size(-1) || ids.size(0) != 1)
                throw std::runtime_error("MiniCPM-o TTS text span is invalid");
            auto text_ids = work.keep(ids.index({0, Slice(begin, end)}));
            auto text_hidden = work.keep(hidden.dense().index({0, Slice(begin, end), Slice()}));
            condition = work.keep(model.tts->condition(text_ids, text_hidden));
        } else {
            condition = work.keep(tyloi::ops::copy(raw, tyloi::Device(tyloi::kCUDA, primary)));
        }
        work.tts_state = std::make_unique<Tts::State>(*model.tts);
        auto codes = work.keep(model.tts->generate_official(*work.tts_state, forward_options, condition,
            options.tts_steps, 6561, 50, 0.8, 0.85, 25, 1.05, &work.traces));
        save(condition, ".tts_inputs_embeds.pt");
        save(codes, ".tts_codes.pt");
        if (!work.traces.empty()) {
            std::vector<tyloi::DenseTensor> trace;
            for (const auto& value : work.traces) trace.push_back(value.dense());
            save(work.keep(tyloi::cuda::stack(trace, 1)), ".tts_logits.pt");
        }
    }
    program.drain();
    std::cout << "minicpmo45_composite=ok" << " batch=" << ids.size(0) << " tokens=" << ids.size(1)
              << " images=" << images.size() << " audios=" << audios.size()
              << " tts_steps=" << options.tts_steps << "\n";
    return 0;
}

int run_eval_batch(Program& program, std::int64_t context_size, std::int64_t vision_batch_size) {
    return tyloi::eval::minicpmo45_batch(program, context_size, vision_batch_size);
}

int run_duplex(Program& program, const CommandOptions& options) {
    const auto& in = options.duplex_input_prefix;
    const auto& out = options.duplex_output_prefix;
    if (in.empty() || out.empty() || options.duplex_steps <= 0 ||
        options.duplex_max_speak_tokens < 2 || options.duplex_seed < 0)
        throw std::runtime_error("MiniCPM-o duplex mode requires prefixes, positive steps, "
                                 "at least two generation slots, and a non-negative seed");
    tyloi::cuda::NoGradGuard no_grad;
    auto scope = program.executor().enter(program.config().layer_placement.primary_device());
    program.load(true);
    auto& model = dynamic_cast<text::Model&>(program.model());
    auto policy = Executor::current_ffn();
    policy.independent |= model.info().separate_projections;
    auto model_scope = program.executor().enter(policy);
    if (!model.vision || !model.resampler || !model.audio || !model.tts)
        throw std::runtime_error("MiniCPM-o media weights are unavailable");
    const int primary = program.config().layer_placement.primary_device();
    const auto mode = program.executor().forward_mode(primary);
    const tyloi::models::ForwardOptions forward_options{mode.kv_capacity, mode.trace, mode.trace_layer, primary};
    struct Work {
        Program& program;
        std::optional<MiniCPMO45DuplexSession> session;
        std::vector<tyloi::Tensor> tensors;
        MiniCPMO45DuplexStepResult output;
        ~Work() noexcept {
            try { program.drain(); }
            catch (const std::exception& error) { std::cerr << "minicpmo_cleanup error=" << error.what() << '\n'; }
            catch (...) { std::cerr << "minicpmo_cleanup error=unknown\n"; }
        }
        tyloi::Tensor keep(tyloi::Tensor value) { tensors.push_back(value); return value; }
    } work{program};
    const auto values = [](tyloi::DenseTensor value) {
        if (!value.defined()) return std::vector<int64_t>{};
        value = value.to(tyloi::kCPU, tyloi::kInt64).contiguous().reshape({-1});
        if (!value.numel()) return std::vector<int64_t>{};
        const auto* data = value.data_ptr<int64_t>();
        return std::vector<int64_t>(data, data + value.numel());
    };
    auto special_tensor = read_tensor(in + ".special_ids.pt", true);
    auto special = MiniCPMO45DuplexSpecialIds::from_tensor(special_tensor);
    auto forbidden = values(read_tensor(in + ".forbidden_ids.pt", false));
    auto system = read_tensor(in + ".system_ids.pt", false);
    if (system.defined() && system.numel() &&
        (system.dim() > 2 || (system.dim() == 2 && system.size(0) != 1)))
        throw std::runtime_error("MiniCPM-o duplex system IDs must contain one sequence");
    const auto token_valid = [&](int64_t token) { return token >= 0 && token < model.metadata.vocab_size; };
    if (special.audio_bos >= 152064 ||
        std::any_of(forbidden.begin(), forbidden.end(), [&](int64_t token) { return !token_valid(token); }))
        throw std::invalid_argument("MiniCPM-o duplex token ID is out of range");
    auto special_values = values(special_tensor);
    if (std::any_of(special_values.begin(), special_values.end() - 1,
                    [&](int64_t token) { return !token_valid(token); }))
        throw std::invalid_argument("MiniCPM-o duplex token ID is out of range");
    tyloi::cuda::manual_seed(options.duplex_seed);
    mfq_cuda_manual_seed_all(static_cast<std::uint64_t>(options.duplex_seed));
    work.session.emplace(model, forward_options, special, std::move(forbidden), options.duplex_greedy);
    auto system_tensor = work.keep(std::move(system));
    work.session->prepare(system_tensor);
    program.drain();
    int64_t listen_steps = 0, speak_steps = 0, chunk_index = 0;
    for (int64_t step = 0; step < options.duplex_steps; ++step) {
        const auto input = minicpmo45_duplex_step_prefix(in, step);
        const auto output = minicpmo45_duplex_step_prefix(out, step);
        if (read_scalar(input + ".reset_session.pt", 0) != 0) {
            work.session->prepare(system_tensor);
            program.drain();
            chunk_index = 0;
        }
        auto pixels = work.keep(read_tensor(input + ".pixel_values.pt", false));
        auto patch_mask = work.keep(read_tensor(input + ".patch_mask.pt", false));
        auto target_sizes = work.keep(read_tensor(input + ".target_sizes.pt", false));
        auto image_counts = work.keep(read_tensor(input + ".image_slice_counts.pt", false));
        auto audio = work.keep(read_tensor(input + ".audio_features.pt", false));
        auto text = work.keep(read_tensor(input + ".text_ids.pt", false));
        work.output = work.session->run_step(pixels, patch_mask, target_sizes, image_counts, audio,
            read_scalar(input + ".audio_prefix_extra_frames.pt", chunk_index == 0 ? 0 : 2),
            read_scalar(input + ".audio_suffix_extra_frames.pt", 2), text,
            options.duplex_max_speak_tokens, read_scalar(input + ".force_listen.pt", 0) != 0);
        const auto save = [&](const tyloi::Tensor& value, const char* suffix) {
            if (value.defined()) write_tensor(value.dense(), output + suffix);
        };
        save(work.output.decision_logits, ".decision_logits.pt");
        save(work.output.audio_embeddings, ".audio_embeddings.pt");
        save(work.output.generated_ids, ".generated_ids.pt");
        save(work.output.tts_codes, ".tts_codes.pt");
        auto state_tensor = work.keep(tyloi::cuda::tensor(std::vector<int64_t>{
            work.output.is_listen ? 1 : 0, work.output.end_of_turn ? 1 : 0,
            work.session->language_state.cache_pos, work.session->audio_state.cache_length(),
            work.session->tts_state.position, work.session->audio_chunk_index},
            tyloi::TensorOptions().dtype(tyloi::kInt64)));
        save(state_tensor, ".state.pt");
        if (work.output.is_listen) ++listen_steps; else ++speak_steps;
        chunk_index = work.session->audio_chunk_index;
        program.drain();
        work.tensors.clear();
        work.tensors.push_back(system_tensor);
    }
    program.drain();
    std::cout << "minicpmo45_duplex=ok" << " steps=" << options.duplex_steps
              << " listens=" << listen_steps << " speaks=" << speak_steps
              << " language_cache=" << work.session->language_state.cache_pos
              << " audio_cache=" << work.session->audio_state.cache_length()
              << " tts_cache=" << work.session->tts_state.position << " seed=" << options.duplex_seed << "\n";
    return 0;
}

} // namespace tyloi::cuda::minicpmo45

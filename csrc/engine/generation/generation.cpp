#include "engine/engine.h"
#include "engine/memory/budget.h"
#include "engine/generation/decode.h"
#include "engine/generation/prefill/prefill.h"
#include "engine/generation/prefill/activity.h"
#include "engine/generation/batch/batch.h"
#include "engine/generation/session.h"
#include "engine/generation/mtp/state.h"
#include "engine/generation/mtp/generation.h"
#include "engine/generation/sampling.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/mtp.h"
#include "engine/models/input.h"
#include "engine/models/session/snapshot.h"
#include "engine/requests/policy.h"
#include "engine/scheduler/request.h"
#include "engine/models/source/model.h"
#include "engine/state/session/sessions.h"
#include "engine/weights/cache/cache.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace tyloi::engine {

std::size_t save_session(models::State& state, state::Sessions& sessions,
        const std::string& session, const std::vector<std::int64_t>& history,
        const std::string& input_key, models::Mtp* mtp, const Tensor& last_hidden) {
    const auto position = state.position();
    if (position <= 0 || static_cast<std::size_t>(position) > history.size()) return 0;
    if (mtp && (!last_hidden || !last_hidden.defined() || position <= 1)) return 0;
    try {
        std::vector<std::int64_t> tokens(history.begin(), history.begin() + position);
        auto saved = state.capture_text_session_state(tokens);
        saved.input_key = input_key;
        if (mtp) {
            saved.mtp = mtp->capture_session_state(*state.predictor, position, last_hidden);
            saved.bytes += saved.mtp->bytes;
        }
        sessions.store(session, std::move(saved));
        return static_cast<std::size_t>(position);
    } catch (const std::exception& error) {
        std::cerr << "runtime_session_cache action=skip error=" << error.what() << '\n';
    }
    return 0;
}

namespace {

struct Plain {
    models::CausalModel& model;
    models::State& state;
    tyloi::Executor& executor;
    Budget& budget;
    Decode& graph;
    weights::Cache& weights;
    const models::Info& info;
    const Config& config;
    const models::ForwardOptions& forward_options;
    const ModelSource* source;
    InferenceRequest& input;
    InferenceOutput& output;
    const std::optional<models::Prepared>& prepared;
    int64_t chunk_size;
    Tensor full_ids;
    Sampler sampler;
    bool penalties;
    Tensor pending;
    std::int64_t pending_host = 0;
    bool graph_active = false;
    bool graph_prepared = false;

    Tensor project(const Tensor& hidden, bool capped) {
        auto last = ops::contiguous(ops::select(hidden, 1, -1));
        return model.logits(last, capped);
    }
    void reserve(const models::Input& input) {
        budget.reserve(state.growth(input, forward_options));
    }
    double prefill(const PrefillChunk& chunk) {
        models::Input model_input;
        model_input.ids = ops::slice(full_ids, 1, chunk.offset, chunk.offset + chunk.count);
        model_input.advance = true;
        if (prepared && prepared->transformed()) {
            if (prepared->embeddings) model_input.embeddings = ops::slice(prepared->embeddings, 1, chunk.offset, chunk.offset + chunk.count);
            if (prepared->positions) model_input.positions = ops::slice(prepared->positions, -1, chunk.offset, chunk.offset + chunk.count);
        }
        model_input.ids = ops::contiguous(model_input.ids);
        if (model_input.embeddings) model_input.embeddings = ops::contiguous(model_input.embeddings);
        if (model_input.positions) model_input.positions = ops::contiguous(model_input.positions);
        auto result = engine::prefill(model, state, executor, weights, budget, forward_options,
            std::move(model_input), chunk_size, output);
        auto hidden = std::move(result.hidden);
        if (prepared && prepared->transformed()) state.set_position_delta(prepared->decode_position_delta);
        if (!output.stopped() && chunk.offset + chunk.count == int64_t(input.prompt.size()))
            pending = sample_logits(sampler, project(hidden, true), graph.counts, input.token_constraint);
        const auto ms = result.elapsed_ms;
        if (config.decode_graph.trace) std::cerr << "runtime_prefill_chunk offset=" << chunk.offset
            << " tokens=" << chunk.count << " layer_major=" << (result.mode == models::Prefill::Mode::layered)
            << " short_windows=" << (result.mode == models::Prefill::Mode::windows) << " compute_chunk=" << chunk_size
            << " elapsed_ms=" << ms << " timing=before_cache_restore\n";
        return ms;
    }
    int64_t first_token() {
        const auto token = ops::item(pending);
        pending_host = token;
        weights.prepare(::tyloi::weights::Cache::Stage::decode, config.device, source);
        graph_active = config.decode_graph.enabled &&
            (!prepared || !prepared->transformed()) && !input.token_constraint && info.decode_graph &&
            info.graph == models::GraphOwner::executor &&
            graph.graph && input.sampling.max_tokens >= config.decode_graph.minimum_generation_tokens &&
            input.sampling.max_tokens <= graph.generated_capacity;
        return token;
    }
    void accept(int64_t) {
        if (penalties && !graph_active) ops::add_token_counts(graph.counts, ops::contiguous(pending));
    }
    int64_t advance() {
        weights.admit_routes(config.device);
        if (!graph_prepared && config.decode_graph.enabled && (!prepared || !prepared->transformed())) {
            models::Input model_input;
            model_input.ids = ops::reshape(pending, {1, 1});
            const std::vector<std::int64_t> host{pending_host};
            model_input.host_ids = ops::reshape(ops::tensor(host, Device(kCPU)), {1, 1});
            model_input.advance = true;
            budget.reserve(state.graph_exit_growth());
            reserve(model_input);
            models::DecodeOptions policy;
            policy.projection = sampler.greedy() && !penalties && !input.token_constraint
                ? models::DecodeOptions::Projection::greedy : models::DecodeOptions::Projection::logits;
            policy.routed = weights.decode() == weights::Cache::Decode::routed;
            policy.enabled = weights.decode() != weights::Cache::Decode::none;
            policy.reserve_until = std::min<std::int64_t>(info.max_context,
                std::int64_t(input.prompt.size()) + std::min<std::int64_t>(input.sampling.max_tokens, 256));
            if (auto result = engine::decode(model, state, executor, weights, budget, forward_options, std::move(model_input), policy)) {
                if (graph_active && penalties) ops::add_token_counts(graph.counts, ops::contiguous(pending));
                graph_active = false;
                pending = policy.projection == models::DecodeOptions::Projection::greedy
                    ? ops::copy(result->ids, config.device) : sample_logits(sampler,
                        ops::reshape(ops::contiguous(result->logits), {1, -1}), graph.counts, input.token_constraint);
                pending_host = ops::item(pending);
                return pending_host;
            }
        }
        if (!graph_active) {
            models::Input model_input;
            model_input.ids = ops::reshape(pending, {1, 1});
            model_input.advance = true;
            exit_unmanaged_graph(state, weights, budget);
            reserve(model_input);
            auto profile = config.profile_decode ? executor.profile("runtime.decode") : nullptr;
            auto hidden = model.forward(std::move(model_input), state, forward_options);
            executor.submit(state.storage());
            pending = sample_logits(sampler,
                project(hidden, !(sampler.greedy() && !penalties && !input.token_constraint)),
                graph.counts, input.token_constraint);
            if (profile) profile->finish({pending});
            pending_host = ops::item(pending);
            return pending_host;
        }
        auto scope = graph.graph->enter();
        if (!graph_prepared) {
            if (penalties) ops::add_token_counts(graph.counts, ops::contiguous(pending));
            graph.initialize_inputs(pending, state.position(), sampler.random_host());
            const auto requested = state.position() + input.sampling.max_tokens;
            const auto planned = decode_bucket(requested, info.max_context);
            const auto parts = ops::attention_parts(planned, graph.device);
            models::Input model_input;
            model_input.ids = graph.static_input; model_input.positions = graph.static_pos;
            model_input.cache_positions = graph.static_pos; model_input.lengths = graph.static_len;
            model_input.kv_length = planned; model_input.attention_parts = parts;
            reserve(model_input);
            const auto sample_static = [&] {
                auto hidden = model.forward(model_input, state, forward_options);
                auto logits = ops::reshape(ops::contiguous(project(hidden, !(sampler.greedy() && !penalties))), {1, -1});
                if (penalties) sampler.apply_penalties(logits, graph.counts);
                return sampler.greedy() ? ops::sample_greedy(logits) :
                    ops::sample_stochastic(logits, graph.random, sampler.params().temperature, sampler.params().top_k, sampler.params().top_p);
            };
            auto captured = state.storage();
            const bool hit = graph.matches(planned, sampler.params(), sampler.greedy(), captured);
            if (hit) ++graph.reuses;
            else {
                graph.invalidate();
                try {
                    auto recurrent = state.graph_state();
                    const auto saved_position = state.position();
                    graph.graph->prepare(recurrent);
                    if (!recurrent.empty() || info.double_warmup) {
                        const auto restore = [&] {
                            graph.graph->restore();
                            if (state.position() != saved_position)
                                throw std::runtime_error("static decode warmup changed cache position");
                        };
                        for (int pass = 0; pass < 2; ++pass) {
                            try { sample_static(); restore(); }
                            catch (...) { restore(); throw; }
                        }
                    }
                    graph.graph->begin();
                    auto next = sample_static();
                    if (penalties) ops::add_token_counts(graph.counts, ops::contiguous(next));
                    graph.capture_next(std::move(next));
                    captured = state.storage();
                    graph.graph->end(captured);
                    graph.set_key(planned, sampler.params(), sampler.greedy());
                    ++graph.captures;
                } catch (...) { graph.invalidate(); throw; }
            }
            if (config.decode_graph.trace) std::cerr << "runtime_cuda_graph action=" << (hit ? "reuse" : "capture")
                << " requested_len=" << requested << " planned_len=" << planned << " captures=" << graph.captures
                << " reuses=" << graph.reuses << std::endl;
            graph_prepared = true;
        }
        if (!sampler.greedy()) graph.upload_random(sampler.next_random());
        graph.graph->replay();
        const auto token = ops::item(graph.static_next);
        state.advance(1);
        return token;
    }
};

struct Native {
    using Prepared = models::Prepared;
    using Hidden = Tensor;
    models::CausalModel& model;
    models::State& state;
    tyloi::Executor& executor;
    Budget& budget;
    Decode& graph;
    weights::Cache& weights;
    const models::Info& info;
    const Config& config;
    const models::ForwardOptions& forward_options;
    const ModelSource* source;
    state::Sessions& cache;
    mtp::State& mtp_state;
    models::Mtp* mtp;
    tyloi::Executor* predictor_execution;
    std::size_t prefix_boundary = 0;
    std::optional<Prepared> prepared;

    void trace_expert_cache(const char* boundary, bool wait = true) {
        if (!config.trace_moe || std::uncaught_exceptions()) return;
        weights.trace(boundary, wait, source);
    }
    void reset() {
        exit_unmanaged_graph(state, weights, budget);
        budget.reserve(state.reset_growth(1));
        state.reset(1);
        if (auto* predictor = model.predictor()) {
            if (!state.predictor || !state.predictor->belongs_to(*predictor))
                throw std::invalid_argument("predictor state is unavailable or belongs to another model");
            budget.reserve(predictor->reset_growth(*state.predictor, 1));
            predictor->reset(*state.predictor, 1);
        }
    }
    void release() {
        trace_expert_cache("end");
        try { reset(); }
        catch (...) { graph.invalidate(); throw; }
        if (std::uncaught_exceptions()) graph.invalidate();
    }
    Generation plain(InferenceRequest& input, InferenceOutput& output, std::size_t reused, std::size_t stable) {
        const auto& prompt = input.prompt;
        const auto chunk_size = budget.prefill(prompt.size() - reused, config.generation.prefill_chunk_size,
            info.prefill_row_bytes, state.position() > 0, config.prefill_chunk_auto);
        const auto logical_chunk = prefill_span(model, state, weights, prompt.size(), chunk_size, reused,
            stable, prepared && prepared->transformed());
        auto full_ids = ops::reshape(ops::tensor(std::span<const int64_t>(prompt),
            config.device), {1, -1});
        graph.ensure_storage(info.vocab_size);
        Sampler sampler(input.sampling, full_ids.device());
        const bool penalties = sampler.has_penalties();
        if (penalties) { ops::fill(graph.counts, 0); ops::add_token_counts(graph.counts, full_ids); }
        Plain plain{model, state, executor, budget, graph, weights, info, config,
                    forward_options, source, input, output, prepared, chunk_size,
                    std::move(full_ids), std::move(sampler), penalties};
        double elapsed = 0.0;
        auto offset = static_cast<std::int64_t>(reused);
        const auto prompt_tokens = static_cast<std::int64_t>(prompt.size());
        const auto stable_tokens = static_cast<std::int64_t>(stable);
        while (offset < prompt_tokens && !output.stopped()) {
            const auto end = offset < stable_tokens ? std::min(prompt_tokens, stable_tokens) : prompt_tokens;
            const auto chunk = next_prefill_chunk(end, offset, logical_chunk);
            {
                PrefillActivity activity;
                elapsed += plain.prefill(chunk);
            }
            offset += chunk.count;
            co_yield PrefillProgress{{static_cast<std::size_t>(offset - reused), elapsed, 0.0, elapsed}};
        }
        if (output.stopped()) co_return;
        auto token = plain.first_token();
        while (!output.stopped()) {
            auto delta = output.append(std::vector<std::int64_t>{token});
            plain.accept(token);
            co_yield std::move(delta);
            if (!output.stopped()) token = plain.advance();
        }
    }
    Generation speculate(InferenceRequest& input, InferenceOutput& output, std::size_t reused,
                         const Tensor& restored, Hidden* last_hidden) {
        if (!predictor_execution) throw std::invalid_argument("predictor executor is unavailable");
        const auto chunk_size = budget.prefill(input.prompt.size() - reused, config.generation.prefill_chunk_size,
            info.prefill_row_bytes, state.position() > 0, config.prefill_chunk_auto);
        const auto logical_chunk = prefill_span(model, state, weights, input.prompt.size(), chunk_size, reused,
            prefix_boundary, prepared && prepared->transformed());
        return generate_mtp(model, state, executor, *predictor_execution, weights, budget, source,
            mtp_state, input, output, logical_chunk, chunk_size, prepared ? &*prepared : nullptr, reused, restored,
            last_hidden, prefix_boundary, config.device, forward_options, config.profile_mtp, config.decode_graph.enabled);
    }
};

Generation generate_session(Native& ops, InferenceRequest& request, InferenceOutput& output,
                            const GenerationPlan& plan) {
    auto& state = ops.state;
    auto& session_cache = ops.cache;
    auto* mtp = ops.mtp;
    const auto& prepared = ops.prepared;
    const auto& prompt = request.prompt;
    const auto& sampling = request.sampling;
    const auto& cache_plan = request.cache_plan;
    const std::string input_key = prepared ? prepared->cache_key : std::string{};
    const bool caching =
        state.supports_text_session_state() &&
        (!prepared || !prepared->transformed() || !input_key.empty()) &&
        plan.stable_prefix_tokens > 0 &&
        (!cache_plan.session_id.empty() || session_cache.persistent_prefix_enabled());
    const bool use_mtp = mtp && sampling.enable_mtp && sampling.max_tokens > 1;
    // Prefix restore leaves one prompt token to recover the next logits.
    // Recurrent models therefore need a checkpoint at that same boundary.
    const auto prefix_boundary=caching?std::min(plan.stable_prefix_tokens,prompt.size()-1):0;
    ops.prefix_boundary = prefix_boundary;
    state::SessionRestore restored;
    if (caching) {
        restored = session_cache.restore_best(state, ops.executor, ops.weights, ops.budget,
            use_mtp ? mtp : nullptr, cache_plan.session_id, prompt, plan.stable_prefix_tokens, input_key);
    }
    if (restored.tokens || use_mtp)
        ops.graph.invalidate();
    if (!restored.tokens) ops.reset();
    std::vector<int64_t> history = prompt;
    Tensor last_target_hidden;
    std::size_t last_snapshot = 0;
    bool initial_snapshot_attempted = false;
    const auto snapshot = [&] {
        const auto position = state.position();
        if (!caching || output.result.cancelled ||
            static_cast<std::size_t>(position) == last_snapshot)
            return;
        const auto saved = save_session(state, session_cache, cache_plan.session_id,
            history, input_key, use_mtp ? mtp : nullptr, last_target_hidden);
        if (saved) last_snapshot = saved;
    };
    auto sequence = use_mtp
        ? ops.speculate(request, output, restored.tokens, restored.mtp_last_target_hidden,
                        &last_target_hidden)
        : ops.plain(request, output, restored.tokens, prefix_boundary);
    bool traced_decode_boundary = false;
    while (auto step = sequence.next()) {
        auto& event = step.value;
        if (auto *progress = event ? std::get_if<PrefillProgress>(&*event) : nullptr;
            progress && !use_mtp && caching &&
            progress->timing.prompt_tokens + restored.tokens == prefix_boundary)
            snapshot();
        if (auto *progress = event ? std::get_if<PrefillProgress>(&*event) : nullptr;
            progress && progress->session_boundary && use_mtp && caching &&
            !initial_snapshot_attempted) {
            initial_snapshot_attempted = true;
            snapshot();
        }
        if (auto *delta = event ? std::get_if<OutputDelta>(&*event) : nullptr) {
            if (!traced_decode_boundary && !delta->token_ids.empty()) {
                ops.trace_expert_cache("decode_begin", false);
                traced_decode_boundary = true;
            }
            history.insert(history.end(), delta->token_ids.begin(), delta->token_ids.end());
        }
        co_yield std::move(step);
    }
    if(traced_decode_boundary) {
        ops.trace_expert_cache("decode_end", false);
    }
    // Flush parser-held text before a final session snapshot can copy or evict
    // a large cache. The scheduler's terminal finish remains idempotent and
    // still reports cleanup/cache completion after this final text event.
    if(caching) {
        auto tail=output.finish();
        if(!tail.diffs.empty())co_yield std::move(tail);
    }
    snapshot();
}

Generation generate_request(Native& ops, InferenceRequest& request, InferenceOutput& output) {
    const auto& info = ops.info;
    const auto& config = ops.config;
    const auto& prepared = ops.prepared;
    const auto& prompt = request.prompt;
    auto& sampling = request.sampling;
    const auto& cache_plan = request.cache_plan;
    if (config.generation.prefill_chunk_size <= 0)
        throw std::invalid_argument("prefill chunk size must be positive");
    if (prepared && prepared->token_ids != prompt)
        throw std::invalid_argument("prepared prompt token IDs disagree");
    if (prepared && prepared->image_mask && prepared->image_mask.defined())
        throw std::invalid_argument("model cannot apply an image participation mask");
    const auto occupied =
        static_cast<int64_t>(prompt.size()) +
        (prepared && prepared->transformed() ? prepared->decode_position_delta : 0);
    const auto plan = plan_generation(prompt,
        info.vocab_size,
        info.max_context,
        sampling.max_tokens,
        cache_plan.stable_prefix_tokens,
        std::max<int64_t>(prompt.size(), occupied));
    sampling.max_tokens = plan.generation_tokens;
    if (output.stopped())
        co_return;
    if (ops.mtp) {
        ops.mtp_state.stats = {};
        ops.mtp_state.stats.available = true;
    }
    ExecutionCleanup cleanup{output.cleanup_failure, [&] { ops.release(); }};
    ops.trace_expert_cache("begin");
    {
        auto sequence = generate_session(ops, request, output, plan);
        while (auto step = sequence.next()) co_yield std::move(step);
    }
    cleanup.finish();
}

Generation generate_prepared(Native ops, InferenceRequest& input, InferenceOutput& output) {
    auto& executor = ops.executor;
    std::optional<models::Prepared> prepared;
    if (input.vision && !output.stopped()) {
        auto preparation = ops.model.prepare(input.prompt, *input.vision, ops.state);
        while (!output.stopped()) {
            StepResult<models::Prepared> step;
            {
                auto timer = executor.timer();
                step = preparation.next();
                if (step.value)
                    timer->stop({step.value->embeddings, step.value->positions, step.value->image_mask});
                else timer->stop();
                output.metrics.multimodal_ms += timer->elapsed_ms();
            }
            if (step.value) {
                prepared = std::move(*step.value);
                break;
            }
            if (!step) throw std::runtime_error("media preparation returned no prompt");
            co_yield step.state;
        }
        if (output.stopped()) co_return;
        const auto elapsed = output.metrics.multimodal_ms;
        co_yield PrefillProgress{{0, 0.0, elapsed, elapsed}};
    }
    if (output.stopped()) co_return;
    ops.prepared = std::move(prepared);
    auto sequence = generate_request(ops, input, output);
    while (auto step = sequence.next()) {
        auto& event = step.value;
        if (auto *progress = event ? std::get_if<PrefillProgress>(&*event) : nullptr) {
            progress->timing.multimodal_ms = output.metrics.multimodal_ms;
            progress->timing.model_ms += progress->timing.multimodal_ms;
        }
        co_yield std::move(step);
    }
}

} // namespace

StepSequence<EventData> Engine::generate(const RequestId& id, Request& request) {
    if (request.batched) {
        if (!batch_) throw std::logic_error("batch generation is unavailable");
        return batch_->generate(id, request.input, *request.output);
    }
    return generate_prepared(
        Native{*model_, *request.state, *executor_, *budget_, *graph_, *weights_, model_info_,
               config_, forward_options_, source_.get(), *cache_,
               mtp_state_, model_->predictor(), predictor_execution_.get()},
        request.input, *request.output);
}

} // namespace tyloi::engine

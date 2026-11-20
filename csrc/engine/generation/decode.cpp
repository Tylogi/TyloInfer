#include "engine/generation/decode.h"
#include "engine/models/causal_model.h"
#include "engine/memory/budget.h"

#include <algorithm>
#include <array>
#include <exception>
#include <stdexcept>
#include <utility>

namespace tyloi::engine {
Capture::Capture(Executor& executor, weights::Cache& weights, Budget& budget, Device device)
    : executor_(executor), weights_(weights), budget_(budget), device_(device) {}
Capture::~Capture() = default;
void Capture::step(GraphRequest request) {
    budget_.reserve(request.growth);
    switch (request.phase) {
    case GraphPhase::admit: break;
    case GraphPhase::prepare: {
        const bool prepared = weights_.routing().fixed_views && executor_.prepare_routed_graph(device_);
        if (request.prepared) *request.prepared = prepared;
        break;
    }
    case GraphPhase::ready:
        weights_.retire_prefill(); weights_.finish_exchange(); drain(); break;
    case GraphPhase::warmup:
        if (warmup_) throw std::logic_error("graph warmup already started");
        warmup_ = std::make_unique<weights::Cache::Warmup>(weights_, true);
        break;
    case GraphPhase::target:
        if (!warmup_) throw std::logic_error("graph warmup not started");
        drain(); warmup_->release(device_, true); target_ = true;
        break;
    case GraphPhase::settle:
        if (!warmup_ || !target_) throw std::logic_error("graph warmup target not reached");
        drain(); warmup_->release(device_, false); warmup_.reset(); target_ = false;
        break;
    }
}
void Capture::finish() {
    if (warmup_) {
        drain();
        warmup_->release(device_, target_ ? false : true);
        warmup_.reset(); target_ = false;
    }
}
void Capture::drain() {
    std::exception_ptr failure;
    try { weights_.drain(); } catch (...) { failure = std::current_exception(); }
    try { executor_.drain(); } catch (...) { if (!failure) failure = std::current_exception(); }
    if (failure) std::rethrow_exception(failure);
}

namespace {
std::optional<models::Decoded> run_decode(StepSequence<models::Decoded> job,
        models::State& state, Executor& executor, weights::Cache& weights, Budget& budget, Device device,
        bool require_result) {
    Capture capture(executor, weights, budget, device);
    auto work = executor.work();
    std::optional<models::Decoded> result;
    bool observed = false;
    try {
        for (;;) {
            StepResult<models::Decoded> step;
            { auto scope = work->enter(); step = job.next(); }
            if (step.workspace || step.layer || (step.graph && step.value) ||
                (!step && (step.graph || step.value)))
                throw std::logic_error("invalid decode job event");
            if (!step) break;
            if (result) throw std::logic_error("decode job continued after its result");
            if (step.graph) { observed = true; capture.step(*step.graph); }
            else if (step.value) { observed = true; result = std::move(*step.value); }
        }
        if (observed && require_result && !result)
            throw std::logic_error("decode job ended without a result");
        if (result) {
            executor.submit(state.storage());
            const std::array<Tensor, 5> outputs{result->hidden, result->raw, result->logits, result->compact, result->ids};
            executor.submit(outputs);
        }
        capture.finish(); work->finish(); job = {};
        return result;
    } catch (...) {
        auto failure = std::current_exception();
        try { capture.drain(); } catch (...) {}
        job = {};
        if (observed) try { budget.reserve(state.graph_exit_growth()); state.release_graphs(); } catch (...) {}
        std::rethrow_exception(failure);
    }
}
}

std::optional<models::Decoded> decode(const models::CausalModel& model, models::State& state,
        Executor& executor, weights::Cache& weights, Budget& budget, const models::ForwardOptions& options,
        models::Input input, models::DecodeOptions decode_options) {
    const auto device = input.ids.device();
    auto* raw = input.raw_hidden;
    auto result = run_decode(model.decode(std::move(input), state, options, decode_options),
        state, executor, weights, budget, device, true);
    if (raw && result && result->raw) *raw = result->raw;
    return result;
}
void warmup(const models::CausalModel& model, models::State& state, Executor& executor,
        weights::Cache& weights, Budget& budget, Device device, models::DecodeOptions options) {
    (void)run_decode(model.warmup(state, options), state, executor, weights, budget, device, false);
}

std::int64_t decode_bucket(std::int64_t planned_len, std::int64_t context_capacity) {
    const std::int64_t quantum = planned_len <= 4096 ? 512 : (planned_len <= 16384 ? 1024 : 2048);
    return std::min(((planned_len + quantum - 1) / quantum) * quantum, context_capacity);
}

Decode::Decode(std::int64_t capacity, Device target, std::unique_ptr<Executor::Graph> execution_graph)
    : device(target), graph(std::move(execution_graph)), generated_capacity(std::max<std::int64_t>(capacity, 2048)) {}
void Decode::ensure_storage(std::int64_t vocabulary) {
    const auto i64 = TensorOptions().dtype(kInt64).device(device);
    if (!static_input) static_input = ops::empty({1, 1}, i64);
    if (!static_pos) static_pos = ops::empty({1}, i64);
    if (!static_len) static_len = ops::empty({1}, i64);
    if (!static_step) static_step = ops::empty({1}, i64);
    if (!generated) generated = ops::empty({generated_capacity}, i64);
    if (!random) random = ops::empty({1}, TensorOptions().dtype(kFloat32).device(device));
    if (!counts) counts = ops::empty({vocabulary}, TensorOptions().dtype(kInt32).device(device));
}
void Decode::initialize_inputs(const Tensor& pending, std::int64_t position, Tensor& random_host) {
    ops::copy_into(static_input, ops::reshape(pending, {1, 1}));
    auto first_generated = ops::slice(generated, 0, 0, 1);
    ops::copy_into(first_generated, ops::reshape(pending, {1}));
    ops::fill(static_pos, position);
    ops::fill(static_len, position + 1);
    ops::fill(static_step, 1);
    ops::fill(random_host, 0.5);
    upload_random(random_host);
    graph->wait();
}
void Decode::upload_random(const Tensor& random_host) { ops::copy_into(random, random_host); }
void Decode::capture_next(Tensor next) {
    static_next = std::move(next);
    ops::decode_commit(static_next, generated, static_step, static_input, static_pos, static_len);
}
bool Decode::matches(std::int64_t planned, const MfqSamplingParams& sampling, bool greedy,
                     std::span<const Tensor> owners) const {
    return valid_ && planned_len == planned && greedy_ == greedy &&
        (greedy || (temperature_ == sampling.temperature && top_k_ == sampling.top_k && top_p_ == sampling.top_p)) &&
        presence_ == sampling.presence_penalty && frequency_ == sampling.frequency_penalty &&
        repetition_ == sampling.repetition_penalty && graph->matches(owners);
}
void Decode::set_key(std::int64_t planned, const MfqSamplingParams& sampling, bool greedy) {
    planned_len = planned; greedy_ = greedy;
    temperature_ = sampling.temperature; top_k_ = sampling.top_k; top_p_ = sampling.top_p;
    presence_ = sampling.presence_penalty; frequency_ = sampling.frequency_penalty;
    repetition_ = sampling.repetition_penalty; valid_ = true;
}
void Decode::invalidate() {
    if (graph) graph->invalidate();
    static_next = {}; valid_ = false;
}
} // namespace tyloi::engine

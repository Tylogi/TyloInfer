#include "engine/engine.h"
#include "engine/generation/prefill/prefill.h"
#include "engine/generation/session.h"
#include "engine/models/source/decision.h"
#include "engine/ops/ops.h"
#include "engine/scheduler/request.h"
#include "engine/state/session/sessions.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>

namespace tyloi::models {
Tensor CausalModel::decision_logits(const Tensor& hidden, const Tensor& labels) const {
    auto projected = logits(hidden);
    return labels ? ops::index_select(projected, -1, labels) : projected;
}
} // namespace tyloi::models

namespace tyloi::engine {
namespace {

DecisionReadResult probabilities(const std::vector<float>& logits, const DecisionTokenRead& read,
        const models::DecisionSpec* spec, bool selected, std::size_t vocabulary) {
    const bool trained = spec && !spec->vocabulary_readout;
    const auto width = selected ? read.labels.size() : trained ? spec->token_ids.size() : vocabulary;
    if (logits.size() != width)
        throw std::runtime_error("decision readout returned an unexpected number of logits");
    const auto primitive = read.type == "score" ? 1 : read.type == "noul" ? 2 : 0;
    const auto temperature = spec ? spec->temperature * spec->primitive_temperatures[primitive] : 1.0;
    DecisionReadResult result;
    auto& weights = result.probabilities;
    weights.reserve(read.labels.size());
    for (std::size_t i = 0; i < read.labels.size(); ++i) {
        const auto token = read.labels[i];
        auto row = selected ? i : std::size_t(token);
        if (trained) {
            const auto found = std::find(spec->token_ids.begin(), spec->token_ids.end(), token);
            if (found == spec->token_ids.end()) throw std::invalid_argument("decision label is absent from the codebook");
            row = static_cast<std::size_t>(found - spec->token_ids.begin());
        }
        const auto value = logits[row];
        if (!std::isfinite(value)) throw std::runtime_error("decision label logit is not finite");
        weights.push_back(value);
    }
    const auto peak = *std::max_element(weights.begin(), weights.end());
    double sum = 0.0;
    for (auto& value : weights) {
        value = std::exp((value - peak) / temperature);
        sum += value;
    }
    for (auto& value : weights) value /= sum;
    result.confidence = 1.0 / sum;
    if (!spec && !selected) {
        const auto best = std::max_element(logits.begin(), logits.end());
        double total = 0.0;
        for (const auto value : logits) {
            if (!std::isfinite(value)) throw std::runtime_error("decision vocabulary logit is not finite");
            total += std::exp(double(value) - *best);
        }
        result.label_mass = std::min(1.0, std::exp(peak - *best) * sum / total);
        result.confidence *= *result.label_mass;
        result.argmax_is_label = std::count(read.labels.begin(), read.labels.end(), best - logits.begin()) != 0;
    }
    return result;
}

} // namespace

StepSequence<EventData> Engine::generate_decisions(Request& request) {
    auto& output = *request.output;
    if (output.stopped()) co_return;
    auto& state = *request.state;
    const auto& decisions = *request.decisions;
    const auto* spec = model_->decisions();
    const auto& first = decisions.reads.front().prompt;
    auto common = first.size();
    for (const auto& read : decisions.reads) {
        const auto limit = std::min(common, read.prompt.size());
        common = std::mismatch(first.begin(), first.begin() + limit, read.prompt.begin()).first - first.begin();
    }
    graph_->invalidate();
    const auto reset = [&] {
        exit_unmanaged_graph(state, *weights_, *budget_);
        budget_->reserve(state.reset_growth(1));
        state.reset(1);
    };
    ExecutionCleanup cleanup{output.cleanup_failure, reset};
    const bool caching = decisions.prefix_cache && cache_->persistent_prefix_enabled() &&
        state.supports_text_session_state();
    const std::string readout = spec ? (spec->vocabulary_readout ? "vocabulary" : "trained") : "base";
    for (const auto& read : decisions.reads) {
        if (output.stopped()) co_return;
        const auto total = read.prompt.size();
        const auto prefix = std::size_t(read.bidirectional_prefix);
        const auto boundary = !caching || prefix == total ? 0 : prefix ? prefix : std::min(common, total - 1);
        const auto key = "decision:" + readout + ":" + std::to_string(prefix) + ":" + decisions.cache_salt;
        auto reused = boundary ? cache_->restore_best(state, *executor_, *weights_, *budget_, nullptr,
            "", read.prompt, boundary, key).tokens : 0;
        if (prefix && reused != prefix) reused = 0;
        if (!reused) reset();
        Tensor hidden;
        double prefill_ms = 0.0;
        for (auto offset = reused; offset < total;) {
            const auto limit = offset < boundary ? boundary : total;
            const auto compute = budget_->prefill(limit - offset, config_.generation.prefill_chunk_size,
                model_info_.prefill_row_bytes, offset != 0, config_.prefill_chunk_auto);
            const auto count = prefix ? limit - offset : std::min<std::size_t>(limit - offset, compute);
            models::Input input;
            input.ids = ops::reshape(ops::tensor(std::span(read.prompt).subspan(offset, count), config_.device), {1, -1});
            input.advance = true;
            input.bidirectional_prefix = read.bidirectional_prefix;
            auto step = engine::prefill(*model_, state, *executor_, *weights_, *budget_, forward_options_,
                std::move(input), compute, output);
            if (output.stopped()) co_return;
            prefill_ms += step.elapsed_ms;
            offset += count;
            if (offset == boundary) save_session(state, *cache_, "", read.prompt, key, nullptr, {});
            if (offset == total)
                hidden = ops::contiguous(ops::select(step.hidden, 1, -1));
            else {
                co_yield StepState::advanced;
                if (output.stopped()) co_return;
            }
        }
        const bool selected = spec ? spec->vocabulary_readout : !decisions.full_vocabulary;
        const auto labels = selected ? ops::tensor(std::span(read.labels), hidden.device()) : Tensor{};
        auto result = probabilities(ops::read_floats(model_->decision_logits(hidden, labels)),
            read, spec, selected, model_info_.vocab_size);
        result.prompt_tokens = total;
        result.cached_tokens = reused;
        result.prefill_ms = prefill_ms;
        output.metrics.prefill_tokens += total;
        output.metrics.prefill_ms += prefill_ms;
        output.metrics.model_prefill_ms += prefill_ms;
        output.metrics.saw_prefill = true;
        co_yield EventData(std::move(result));
    }
    cleanup.finish();
    output.result.finish_reason = "stop";
}

} // namespace tyloi::engine

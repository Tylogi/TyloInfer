#include "engine/generation/prefill/prefill.h"

#include "engine/memory/budget.h"
#include "engine/executor.h"
#include "engine/generation/decode.h"
#include "engine/generation/prefill/budget.h"
#include "engine/models/causal_model.h"
#include "engine/requests/output.h"
#include "engine/weights/cache/cache.h"
#include "engine/weights/cache/workspace.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace tyloi::engine {

void exit_unmanaged_graph(models::State& state, weights::Cache& weights, Budget& budget) {
    const auto growth = state.graph_exit_growth();
    if (growth.empty()) return;
    budget.reserve(growth);
    weights.finish_exchange();
    state.release_graphs();
}

std::int64_t prefill_span(const models::CausalModel& model, const models::State& state,
        const weights::Cache& weights, const Budget& budget, std::int64_t tokens, std::int64_t compute,
        std::size_t reused, std::size_t stable, bool transformed) {
    const auto routing = weights.routing();
    models::PrefillOptions policy{tokens, compute, 0, 0, transformed,
        budget.retained_prefill(tokens, state.position() > 0), routing.fixed_views,
        routing.shared_slots, !routing.fixed_views || tokens <= 8};
    const auto plan = model.prefill_plan(state, policy);
    if (!reused && (!stable || stable == std::size_t(tokens)) && plan.layered)
        return plan.max_span > 0 ? std::min(tokens, plan.max_span) : tokens;
    return plan.max_span > 0 ? std::min(compute, plan.max_span) : compute;
}

models::Prefill prefill(const models::CausalModel& model, models::State& state, Executor& executor,
    weights::Cache& weights, Budget& budget, const models::ForwardOptions& options,
    models::Input input, std::int64_t compute, InferenceOutput& output, bool raw_host) {
    const auto device = input.ids.device();
    weights::Cache::Routes routes;
    Capture capture(executor, weights, budget, device);
    std::unique_ptr<weights::Loan> loan;
    auto work = executor.work();
    tyloi::StepSequence<models::Prefill> job;
    std::optional<weights::Cache::Priority> priority;
    models::Prefill result;
    bool observed = false;
    bool computed = false;
    auto timer = executor.timer();
    try {
        const auto tokens = input.ids.size(1);
        const int learning_rows = input.ids.size(0) == 1 && tokens > 8 && tokens <= INT_MAX
            ? static_cast<int>(tokens) : 0;
        routes.begin_prefill(weights, learning_rows);
        const bool transformed = bool(input.embeddings) || bool(input.positions) || bool(input.lengths) ||
            bool(input.cache_positions) || bool(input.image_mask) || bool(input.trace) ||
            input.suffix || input.confirmed != 0 || input.bidirectional_prefix != 0;
        const auto routing = weights.routing();
        const bool decode_windows = !routing.fixed_views || tokens <= 8;
        models::PrefillOptions policy{tokens, compute, 0, 0, transformed,
            budget.retained_prefill(tokens, state.position() > 0),
            routing.fixed_views, routing.shared_slots, decode_windows, raw_host};
        const auto plan = model.prefill_plan(state, policy);
        for (const auto& [device, bytes] : state.reset_growth(input.ids.size(0)))
            if (bytes) budget.reserve_kv(device, prefill_budget_add(bytes, std::size_t(64) << 20));
        if (!plan.layered)
            for (const auto& [device, bytes] : state.growth(input, options))
                if (bytes) budget.reserve_kv(device, prefill_budget_add(bytes, std::size_t(64) << 20));
        if (!transformed && tokens <= 128 && (decode_windows || input.raw_hidden)) {
            std::vector<Tensor> hidden_rows, raw_rows;
            for (std::int64_t offset = 0; offset < tokens && !output.stopped();) {
                const auto rows = state.position() == 0 ? std::int64_t(1) : std::min<std::int64_t>(4, tokens - offset);
                models::Input part;
                part.ids = ops::contiguous(ops::slice(input.ids, 1, offset, offset + rows));
                part.advance = true; part.confirmed = rows > 1 ? 1 : 0;
                Tensor raw;
                if (input.raw_hidden) part.raw_hidden = &raw;
                models::DecodeOptions decode_options;
                decode_options.enabled = routing.fixed_views && !routing.host_ids_required;
                decode_options.routed = routing.fixed_views;
                budget.reserve(state.graph_exit_growth());
                auto decoded = engine::decode(model, state, executor, weights, budget, options, part, decode_options);
                if (!decoded && offset == 0) break;
                Tensor hidden;
                if (decoded) hidden = ops::clone(decoded->hidden);
                else {
                    exit_unmanaged_graph(state, weights, budget);
                    budget.reserve(state.growth(part, options));
                    hidden = model.forward(std::move(part), state, options);
                }
                executor.submit(state.storage());
                hidden_rows.push_back(std::move(hidden));
                if (input.raw_hidden) raw_rows.push_back(raw_host ? ops::copy(raw, Device(kCPU)) : std::move(raw));
                if (rows > 1) state.commit_speculative();
                offset += rows;
            }
            if (!hidden_rows.empty()) {
                if (output.stopped()) { capture.drain(); return {}; }
                result.hidden = hidden_rows.size() == 1 ? std::move(hidden_rows.front()) : ops::cat(hidden_rows, 1);
                if (!raw_rows.empty()) result.raw = raw_rows.size() == 1 ? std::move(raw_rows.front()) : ops::cat(raw_rows, 1);
                result.mode = models::Prefill::Mode::windows;
                timer->stop({result.hidden, result.raw});
                result.elapsed_ms = timer->elapsed_ms();
                routes.finish_prefill(!output.stopped());
                return result;
            }
        }
        exit_unmanaged_graph(state, weights, budget);
        if (!plan.layered) budget.reserve(state.growth(input, options));
        job = model.prefill(input, state, options, policy);
        while (!output.stopped()) {
            StepResult<models::Prefill> step;
            { auto scope = work->enter(); step = job.next(); }
            if (step.layer) {
                if (step.state != StepState::advanced || step.value || step.workspace || step.graph)
                    throw std::logic_error("invalid prefill layer event");
                observed = true;
                work->step(*step.layer);
                continue;
            }
            if (step.graph) {
                if (step.state != StepState::advanced || step.value || step.workspace)
                    throw std::logic_error("invalid prefill graph event");
                observed = true;
                capture.step(*step.graph);
                continue;
            }
            if (step.workspace) {
                if (step.state != StepState::advanced || step.value)
                    throw std::logic_error("invalid prefill workspace event");
                observed = true;
                const auto request = *step.workspace;
                if (request.bytes) {
                    if (loan || request.output || request.extra)
                        throw std::logic_error("invalid prefill workspace acquisition");
                    loan = std::make_unique<weights::Loan>(weights);
                    if (request.tile) {
                        const auto admitted = budget.tiled_workspace(device, *loan, request);
                        *request.tile->compute = admitted;
                    } else loan->begin(request.bytes);
                } else {
                    if (!loan || !request.output || request.tile)
                        throw std::logic_error("invalid prefill workspace completion");
                    std::array<Tensor*, 2> outputs{request.output, request.extra};
                    loan->finish(std::span<Tensor* const>{outputs.data(), request.extra ? 2U : 1U});
                    loan.reset();
                }
                continue;
            }
            if (!step) {
                if (step.value) throw std::logic_error("invalid prefill completion event");
                if (loan) throw std::logic_error("prefill job ended while workspace is borrowed");
                if (observed || plan.layered) throw std::runtime_error("prefill job ended without a result");
                capture.finish();
                work->finish();
                auto* raw = input.raw_hidden;
                if (state.position() == 0) weights.finish_exchange();
                { auto scope = executor.enter(); result.hidden = model.forward(std::move(input), state, options); }
                if (raw) result.raw = *raw;
                executor.submit(state.storage());
                timer->stop({result.hidden, result.raw});
                result.elapsed_ms = timer->elapsed_ms();
                routes.finish_prefill(!output.stopped());
                return result;
            }
            observed = true;
            if (!step.value) continue;
            if (step.value->phase == models::Prefill::Phase::progress) {
                if (!step.value->progress) throw std::logic_error("prefill progress data is missing");
                const auto& report = *step.value->progress;
                executor.drain();
                const std::array<Device, 1> devices{report.device};
                const auto memory = executor.memory(devices).at(0);
                std::cout << "prefill_layer_major_memory tokens=" << report.tokens << " chunk=" << report.chunk
                          << " completed_layers=" << report.completed << " owned_bytes=" << memory.allocated
                          << " reserved_bytes=" << memory.reserved << std::endl;
                std::cout << "prefill_layer_major_progress tokens=" << report.tokens << " chunk=" << report.chunk
                          << " completed_layers=" << report.completed << " total_layers=" << report.total
                          << " elapsed_ms=" << std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - report.started).count() << std::endl;
                continue;
            }
            result = std::move(*step.value);
            if (result.phase == models::Prefill::Phase::computed) {
                executor.submit(state.storage());
                timer->stop({result.hidden, result.raw});
                computed = true;
            } else if (result.phase == models::Prefill::Phase::encode) {
                if (result.first < result.last) {
                    if (priority) { priority->finish(); priority.reset(); }
                    priority.emplace(weights, result.first, result.last);
                }
            } else if (result.phase == models::Prefill::Phase::decode ||
                       result.phase == models::Prefill::Phase::complete) {
                if (priority) { priority->finish(); priority.reset(); }
                if (result.phase == models::Prefill::Phase::complete) {
                    if (loan) throw std::logic_error("prefill completed while workspace is borrowed");
                    executor.submit(state.storage());
                    if (!computed) timer->stop({result.hidden, result.raw});
                    capture.finish();
                    work->finish();
                    result.elapsed_ms = timer->elapsed_ms();
                    routes.finish_prefill(!output.stopped());
                    return result;
                }
            } else throw std::logic_error("unknown prefill job phase");
        }
        capture.drain();
        job = {};
        if (observed) { budget.reserve(state.graph_exit_growth()); state.release_graphs(); }
        work.reset();
        loan.reset();
        return {};
    } catch (...) {
        auto failure = std::current_exception();
        try { capture.drain(); } catch (...) {}
        job = {};
        if (observed) try { budget.reserve(state.graph_exit_growth()); state.release_graphs(); } catch (...) {}
        work.reset();
        loan.reset();
        std::rethrow_exception(failure);
    }
}

} // namespace tyloi::engine

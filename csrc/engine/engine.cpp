#include "engine/engine.h"
#include "engine/scheduler/scheduler.h"
#include "engine/generation/batch.h"
#include "engine/generation/generation.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <utility>

#include "engine/resource_metrics.h"
#include "engine/models/prepare.h"
#include "engine/models/source.h"
#include "engine/scheduler/request.h"
#include "engine/budget.h"
#include "engine/requests/processor.h"
#include "engine/weights/memory.h"
#include "engine/weights/cache.h"
#include "engine/state/session/sessions.h"
#include <stdexcept>
#include <type_traits>

namespace tyloi::engine {

Engine::Engine() = default;
Engine::Engine(std::unique_ptr<models::CausalModel> model, std::unique_ptr<tyloi::Executor> executor,
        Config config, std::shared_ptr<const ModelSource> source,
        std::shared_ptr<weights::Memory> memory, std::unique_ptr<weights::Cache> weights,
        std::unique_ptr<models::State> model_state, std::unique_ptr<state::Sessions> sessions,
        std::unique_ptr<TextProcessor> text, models::ForwardOptions forward,
        std::unique_ptr<tyloi::Executor::Graph> graph, std::unique_ptr<Batch> batch) {
    rebind(std::move(model), std::move(executor), std::move(config), std::move(source),
        std::move(memory), std::move(weights), std::move(model_state), std::move(sessions),
        std::move(text), std::move(forward), std::move(graph), std::move(batch));
    start();
}

// Wiring that needs this Engine: caches, budget, expert space, reservations.
void Engine::install(std::unique_ptr<tyloi::Executor::Graph> graph) {
    // Keep the graph with its Executor even when metadata or scope setup fails.
    graph_.emplace(0, config_.device, std::move(graph));
    try {
        if (config_.continuous_batch.max_sequences && !batch_)
            throw std::invalid_argument("continuous batching is unavailable for this model");
        model_info_ = model_->info();
        graph_->generated_capacity = std::max<std::int64_t>(model_info_.max_context, 2048);
        info_ = {};
        auto& info = info_;
        info.max_requests = std::max<std::size_t>(1, config_.continuous_batch.max_sequences);
        info.max_context = model_info_.max_context;
        info.vocab_size = static_cast<std::int32_t>(model_info_.vocab_size);
        info.model_type = model_info_.model_type;
        info.stop_token_ids = config_.stop_token_ids;
        info.capabilities = model_->capabilities();
        const auto& caps = info.capabilities;
        info.duplex = caps.full_duplex;
        info.multimodal = caps.image_input || caps.video_input || caps.audio_input || caps.audio_output || caps.full_duplex;
        info.reload = true;
        if (text_) info.chat = text_->chat_template_capabilities();
        auto scope = executor_->enter();
        tyloi::weights::LoadScope memory_load(*ledger_);
        budget_.emplace(*executor_, *ledger_, *weights_, *cache_, config_.device);
        budget_->bind(false);
        if (batch_) batch_->bind(*budget_);
        if (const auto bytes = config_.resident_budget_bytes) {
            engine::SessionCommand command; command.kind = engine::SessionCommand::Kind::memory_budget; command.bytes = bytes;
            set_budget(command);
        }
        if (config_.preallocate_kv) {
            budget_->reserve(idle_state_->preallocate_growth());
            idle_state_->preallocate();
        }
        weights_->prepare(::tyloi::weights::Cache::Stage::loaded, config_.device, source_.get());
        drain();
        ready_ = true;
    } catch (...) {
        const auto failure = std::current_exception();
        try { unload_model(); }
        catch (const std::exception& error) { std::cerr << "runtime_load_cleanup error=" << error.what() << std::endl; std::terminate(); }
        catch (...) { std::cerr << "runtime_load_cleanup error=unknown" << std::endl; std::terminate(); }
        std::rethrow_exception(failure);
    }
}

void Engine::rebind(std::unique_ptr<models::CausalModel>&& model, std::unique_ptr<tyloi::Executor>&& executor,
        Config config, std::shared_ptr<const ModelSource>&& source,
        std::shared_ptr<weights::Memory>&& memory, std::unique_ptr<weights::Cache>&& weights,
        std::unique_ptr<models::State>&& model_state, std::unique_ptr<state::Sessions>&& sessions,
        std::unique_ptr<TextProcessor>&& text, models::ForwardOptions forward,
        std::unique_ptr<tyloi::Executor::Graph>&& graph, std::unique_ptr<Batch>&& batch) {
    if (executor_) unload_model();
    model_ = std::move(model);
    executor_ = std::move(executor);
    config_ = std::move(config);
    source_ = std::move(source);
    ledger_ = std::move(memory);
    weights_ = std::move(weights);
    idle_state_ = std::move(model_state);
    cache_ = std::move(sessions);
    batch_ = std::move(batch);
    text_ = std::move(text);
    forward_options_ = std::move(forward);
    install(std::move(graph));
}
Engine::~Engine() {
    scheduler_.reset();
    if (executor_) {
        try { unload_model(); }
        catch (const std::exception& error) { std::cerr << "runtime_unload_cleanup error=" << error.what() << std::endl; std::terminate(); }
        catch (...) { std::cerr << "runtime_unload_cleanup error=unknown" << std::endl; std::terminate(); }
    }
}

EngineInfo Engine::describe() const { return info_; }

bool Engine::ready() const { return ready_; }

std::unique_ptr<tyloi::Executor::Scope> Engine::enter() { return executor_->enter(); }

void Engine::prepare_request(Request& request, bool batched) {
    engine::prepare_request(request, text_.get(), info_, batched, false);
    if (batched) {
        if (!batch_) throw std::logic_error("continuous batching is unavailable for this model");
        return;
    }
    if (!idle_state_) {
        request.output.reset();
        request.input = {};
        throw std::logic_error("native engine state is already in use");
    }
    request.state = std::move(idle_state_);
}

void Engine::retire_request(Request& request) noexcept {
    request.generation = {};
    if (!request.state) return;
    idle_state_ = std::move(request.state);
    try { drain(); }
    catch (...) {
        ready_ = false;
        if (request.output && !request.output->cleanup_failure)
            request.output->cleanup_failure = std::current_exception();
    }
}

SessionResult Engine::session_command(const SessionCommand& command) {
    if (!ready_) throw std::runtime_error("CUDA engine is unloaded");
    auto scope = executor_->enter();
    return command.kind == engine::SessionCommand::Kind::memory_budget
        ? set_budget(command) : cache_->control(command);
}

void Engine::drain() {
    auto scope = executor_->enter();
    std::exception_ptr failure;
    try { if (weights_) weights_->drain(); }
    catch (...) { failure = std::current_exception(); }
    try { executor_->drain(); }
    catch (...) { if (!failure) failure = std::current_exception(); }
    if (failure) std::rethrow_exception(failure);
}

void Engine::unload_model() {
    ready_ = false;
    if (executor_) {
        auto scope = executor_->enter();
        if (graph_) graph_->invalidate();
        drain();
        if (weights_) weights_->release_workspace();
        drain();
        // No owner is released until consumers and loan restoration have completed.
        clear_bindings();
        graph_.reset();
        batch_.reset();
        idle_state_.reset();
        if (weights_) weights_->clear_workspace();
        budget_.reset(); cache_.reset();
        model_.reset();
        weights_.reset();
        text_.reset();
        // Execution resources retire last: state and weight backing free first.
        executor_->clear();
        scope.reset();
        executor_.reset();
    }
    ledger_.reset(); source_.reset();
}

ControlResult Engine::control_command(ControlRequest request) {
    if (!ready_) throw std::runtime_error("engine is unloaded");
    auto scope = executor_->enter();
    return std::visit([&](auto&& value) -> engine::ControlResult {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, engine::DecodeTokens> || std::is_same_v<T, engine::CountTokens> ||
                      std::is_same_v<T, engine::PrepareDuplex> || std::is_same_v<T, engine::PrepareDuplexStep>) {
            if (!text_) throw std::invalid_argument("text control requires a tokenizer");
            if constexpr (std::is_same_v<T, engine::DecodeTokens>) return text_->decode_tokens(value.tokens, value.excluded);
            else if constexpr (std::is_same_v<T, engine::CountTokens>) return engine::TokenCount{text_->count_tokens(value.prompt), info_.max_context};
            else if constexpr (std::is_same_v<T, engine::PrepareDuplex>) {
                text_->prepare_duplex_session(value.prompt, value.parameters); return std::move(value.parameters);
            } else { text_->prepare_duplex_step(value.text, value.input); return std::move(value.input); }
        } else if constexpr (std::is_same_v<T, engine::GetMetrics>) {
            return resource_metrics(model_->metrics(), {}, std::nullopt, false,
                batch_ ? batch_->metrics() : Metrics{}, {}, std::nullopt, {}, source_.get(),
                *ledger_, *weights_, *cache_, *budget_,
                executor_->memory(config_.devices), config_.device, config_.online_budget);
        } else throw std::invalid_argument("model has no duplex component");
    }, std::move(request));
}

bool Engine::can_batch(const EngineRequest& request) const {
    return batch_ && batch_compatible(request, cache_->persistent_prefix_enabled(), model_->predictor() != nullptr);
}

bool Engine::exclusive() const { return false; }

void Engine::execute(const std::vector<RequestId>& eligible) {
    if (batch_) batch_->step(eligible);
}

void Engine::clear_bindings() noexcept {
    if (budget_) budget_->unbind();
}

SessionResult Engine::set_budget(const SessionCommand& command) {
    if (!config_.online_budget)
        throw std::invalid_argument("online memory budgeting requires a single-device CUDA worker");
    auto change = budget_->change(command);
    try {
        weights_->settle();
        graph_->invalidate();
        if (batch_) batch_->release_idle_buffers();
    } catch (...) {
        ready_ = false;
        throw;
    }
    if (ready_) reset_states(1);
    if (idle_state_) idle_state_->reset(1);
    return change.apply();
}

} // namespace tyloi::engine

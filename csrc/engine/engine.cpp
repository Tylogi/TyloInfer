#include "engine/engine.h"
#include "engine/scheduler/scheduler.h"
#include "engine/generation/batch/batch.h"
#include "engine/generation/batch/ops.h"
#include "engine/generation/mtp/metrics.h"
#include "engine/models/common/mtp.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <utility>


#include "engine/metrics.h"
#include "engine/executor.h"
#include "engine/memory/budget.h"
#include "engine/state/session/sessions.h"
#include "engine/weights/cache/cache.h"
#include "engine/weights/memory.h"
#include "engine/weights/source/source.h"

#include <cstddef>
#include <optional>
#include <set>
#include <span>
#include <string>
#include "engine/models/prepare.h"
#include "engine/models/source/model.h"
#include "engine/scheduler/request.h"
#include "engine/requests/processor.h"
#include <stdexcept>
#include <type_traits>

namespace tyloi::engine {

inline Metrics resource_metrics(
        const Metrics& component_metrics, const Metrics& profile_metrics,
        std::optional<Metrics> memory, bool streamed_kv,
        const Metrics& batch_metrics, const Metrics& execution_metrics,
        const Metrics& predictor_metrics,
        const source::Source* source, const weights::Memory& ledger,
        const weights::Cache& weights, const state::Sessions& cache,
        const Budget& budget, std::span<const tyloi::Executor::Memory> devices,
        Device primary, bool online_budget_supported) {
    Metrics metrics;
    metrics.insert(metrics.end(), component_metrics.begin(), component_metrics.end());
    auto experts = weights.metrics();
    metrics.insert(metrics.end(), experts.begin(), experts.end());
    metrics.insert(metrics.end(), profile_metrics.begin(), profile_metrics.end());
    if (source) {
        const auto io = source->file_read_stats();
        metrics.insert(metrics.end(), {
            {"model_io_direct", io.mode == tyloi::FileReadMode::Direct ? 1.0 : 0.0},
            {"model_io_read_calls", double(io.calls)},
            {"model_io_logical_bytes", double(io.logical_bytes)},
            {"model_io_physical_bytes", double(io.physical_bytes)},
            {"model_io_read_ms", double(io.read_nanoseconds) / 1e6},
            {"model_io_staging_bytes", double(io.staging_bytes)},
            {"model_io_cached_staging_bytes", double(io.staging_cached_bytes)},
            {"model_io_staging_allocations", double(io.staging_allocations)}});
    }
    metrics.insert(metrics.end(), execution_metrics.begin(), execution_metrics.end());
    std::size_t resident_weights = ledger.resident_bytes();
    for (const auto& metric : experts)
        if (metric.first == "resident_expert_cache_bytes")
            resident_weights += std::size_t(metric.second);
    const auto ple = ledger.ple_bytes();
    metrics.insert(metrics.end(), {
        {"resident_weight_bytes", double(resident_weights)},
        {"ssd_ple_enabled", ple > 0 ? 1.0 : 0.0},
        {"ssd_ple_payload_bytes", double(ple)}});
    if (memory) {
        std::size_t shared_kv_bytes = 0;
        bool shared_kv_reported = false;
        for (const auto& [name, value] : execution_metrics)
            if (name == "kv_memory_gpu_bytes" || name == "qsa_kv_resident_bytes") {
                shared_kv_bytes += std::size_t(value);
                shared_kv_reported = true;
            }
        const auto primary_live = "gpu_" + std::to_string(primary.index) + "_live_kv_bytes";
        bool primary_reported = false;
        for (auto& [name, value] : *memory) {
            if (name == "kv_cache_bytes" || name == primary_live)
                value += double(shared_kv_bytes);
            if (name == primary_live) primary_reported = true;
        }
        if (shared_kv_reported && !primary_reported)
            memory->emplace_back(primary_live, double(shared_kv_bytes));
        for (auto& [name, value] : *memory)
            for (const auto& [batch_name, batch_value] : batch_metrics) {
                if (name == "kv_cache_bytes" && batch_name == "continuous_batching_request_kv_bytes")
                    value += batch_value;
                if (name == "kv_cache_contexts" && batch_name == "continuous_batching_request_contexts")
                    value += batch_value;
            }
        metrics.insert(metrics.end(), memory->begin(), memory->end());
    }
    const auto metric_bytes = [&](const std::string& key) {
        for (const auto& [name, value] : metrics)
            if (name == key)
                return std::size_t(value);
        return std::size_t(0);
    };
    auto prefix_bytes = cache.device_hot_bytes();
    std::size_t allocated = 0, reserved = 0, free_bytes = 0, total_bytes = 0, buffers = 0;
    std::size_t charged = 0, wired = 0, cached = 0, cuda_allocated = 0, cuda_reserved = 0;
    bool shared_pool = false, cuda_devices = false;
    std::set<std::pair<DeviceType, int>> pools;
    for (const auto& snapshot : devices) {
        const auto accounting = snapshot.pool.value_or(snapshot.device);
        if (snapshot.pool && !pools.emplace(accounting.type, accounting.index).second) continue;
        const auto device = accounting.index;
        const auto external = !snapshot.pool && accounting == primary ? metric_bytes("kv_memory_vmm_bytes") : 0;
        const auto resident = snapshot.pool
            ? std::max(snapshot.resident, std::max(snapshot.reserved, snapshot.allocated))
            : std::max(snapshot.reserved, snapshot.allocated) + external;
        const auto stem = "gpu_" + std::to_string(device) + "_";
        auto device_weights = snapshot.pool ? ledger.resident_bytes() : ledger.resident_bytes(device);
        if (accounting == primary)
            device_weights += metric_bytes("resident_expert_cache_bytes");
        const auto hot = prefix_bytes[device];
        auto live = metric_bytes(stem + "live_kv_bytes");
        if (accounting == primary)
            for (const auto& [name, value] : batch_metrics)
                if (name == "continuous_batching_request_kv_bytes")
                    live += std::size_t(value);
        metrics.insert(metrics.end(), {
            {stem + "resident_bytes", double(resident)},
            {stem + "resident_weight_bytes", double(device_weights)},
            {stem + "prefix_cache_bytes", double(hot)},
            {stem + "memory_total_bytes", double(snapshot.total)},
            {stem + "memory_available_bytes", double(snapshot.available)}});
        if (memory) {
            const auto categorized = device_weights + live + hot;
            const auto buffer = resident > categorized ? resident - categorized : 0;
            metrics.emplace_back(stem + "resident_buffer_bytes", double(buffer));
            metrics.emplace_back(stem + "kv_bytes", double(live + hot));
            buffers += buffer;
        }
        allocated += snapshot.allocated + external;
        reserved += snapshot.reserved + external;
        free_bytes += snapshot.available;
        total_bytes += snapshot.total;
        charged += resident;
        wired += snapshot.wired;
        cached += snapshot.reserved > snapshot.allocated ? snapshot.reserved - snapshot.allocated : 0;
        shared_pool = shared_pool || snapshot.pool.has_value();
        if (accounting.is_cuda()) {
            cuda_devices = true;
            cuda_allocated += snapshot.allocated + external;
            cuda_reserved += snapshot.reserved + external;
        }
    }
    metrics.insert(metrics.end(), {
        {"device_free_bytes", double(free_bytes)},
        {"device_total_bytes", double(total_bytes)},
        {"device_allocated_bytes", double(allocated)},
        {"device_reserved_bytes", double(reserved)},
        {"device_wired_bytes", double(wired)},
        {"device_allocator_cache_bytes", double(cached)}});
    if (cuda_devices)
        metrics.insert(metrics.end(), {{"cuda_allocated_bytes", double(cuda_allocated)},
            {"cuda_reserved_bytes", double(cuda_reserved)}});
    if (memory)
        metrics.emplace_back("resident_buffer_bytes", double(buffers));
    const auto expert_bytes = metric_bytes("resident_expert_cache_bytes");
    const auto minimum = weights.minimum();
    metrics.insert(metrics.end(), {
        {"resident_memory_dynamic_budget", online_budget_supported ? 1.0 : 0.0},
        {"resident_weight_dynamic_budget", online_budget_supported ? 1.0 : 0.0},
        {"resident_memory_budget_bytes", double(budget.resident())},
        {"resident_weight_budget_bytes", double(budget.weights())},
        {"resident_memory_used_bytes", double(shared_pool ? charged : std::max(allocated, reserved))},
        {"resident_reclaimable_weight_bytes", double(expert_bytes > minimum ? expert_bytes - minimum : 0)}});
    metrics.emplace_back("qsa_kv_offload_supported", streamed_kv ? 1.0 : 0.0);
    const bool has_offload = std::any_of(execution_metrics.begin(), execution_metrics.end(),
        [](const auto& metric) { return metric.first == "qsa_kv_offload_enabled"; });
    if (!has_offload) {
        metrics.insert(metrics.end(), {{"qsa_kv_offload_enabled", 0.0},
            {"qsa_kv_ram_enabled", 0.0}, {"qsa_kv_ram_bytes", 0.0},
            {"qsa_kv_ssd_bytes", 0.0}});
    }
    metrics.insert(metrics.end(), predictor_metrics.begin(), predictor_metrics.end());
    metrics.insert(metrics.end(), batch_metrics.begin(), batch_metrics.end());
    return metrics;
}

} // namespace tyloi::engine

namespace tyloi::engine {

Engine::Engine() = default;
Engine::Engine(std::unique_ptr<models::CausalModel> model, std::unique_ptr<tyloi::Executor> executor,
        Config config, std::shared_ptr<const ModelSource> source,
        std::shared_ptr<weights::Memory> memory, std::unique_ptr<weights::Cache> weights,
        std::unique_ptr<models::State> model_state, std::unique_ptr<state::Sessions> sessions,
        std::unique_ptr<TextProcessor> text, models::ForwardOptions forward,
        std::unique_ptr<tyloi::Executor::Graph> graph, std::unique_ptr<Batch> batch,
        std::unique_ptr<tyloi::Executor> predictor_execution) {
    rebind(std::move(model), std::move(executor), std::move(config), std::move(source),
        std::move(memory), std::move(weights), std::move(model_state), std::move(sessions),
        std::move(text), std::move(forward), std::move(graph), std::move(batch), std::move(predictor_execution));
    start();
}

// Wiring that needs this Engine: caches, budget, expert space, reservations.
void Engine::install(std::unique_ptr<tyloi::Executor::Graph> graph) {
    // Keep the graph with its Executor even when metadata or scope setup fails.
    graph_.emplace(0, config_.device, std::move(graph));
    try {
        auto* predictor = model_->predictor();
        if (predictor && (!idle_state_ || !idle_state_->predictor || !idle_state_->predictor->belongs_to(*predictor) ||
                          !predictor_execution_ || !model_->supports_speculation()))
            throw std::invalid_argument("predictor resources are unavailable or incompatible");
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
        budget_->bind(predictor != nullptr);
        executor_->bind_budget(&*budget_);
        if (!batch_ && config_.continuous_batch.max_sequences)
            batch_ = engine::make_batch(*model_, *executor_, *weights_, *budget_, *cache_, config_, forward_options_, predictor_execution_.get());
        if (batch_) batch_->bind(*budget_);
        if (const auto bytes = config_.resident_budget_bytes) {
            engine::SessionCommand command; command.kind = engine::SessionCommand::Kind::memory_budget; command.bytes = bytes;
            set_budget(command);
        }
        if (config_.preallocate_kv) {
            if (!batch_) {
                budget_->reserve(idle_state_->preallocate_growth());
                idle_state_->preallocate();
            }
            if (predictor) {
                budget_->reserve(predictor->reset_growth(*idle_state_->predictor, 1));
                predictor->reset(*idle_state_->predictor, 1);
            }
        }
        models::DecodeOptions decode;
        decode.enabled = config_.decode_graph.enabled && weights_->decode() != weights::Cache::Decode::none;
        decode.routed = weights_->decode() == weights::Cache::Decode::routed;
        engine::warmup(*model_, *idle_state_, *executor_, *weights_, *budget_, config_.device, decode);
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
        std::unique_ptr<tyloi::Executor::Graph>&& graph, std::unique_ptr<Batch>&& batch,
        std::unique_ptr<tyloi::Executor>&& predictor_execution) {
    if (executor_) unload_model();
    model_ = std::move(model);
    executor_ = std::move(executor);
    predictor_execution_ = std::move(predictor_execution);
    mtp_state_ = {};
    config_ = std::move(config);
    source_ = std::move(source);
    ledger_ = std::move(memory);
    weights_ = std::move(weights);
    idle_state_ = std::move(model_state);
    serial_state_ = idle_state_.get();
    cache_ = std::move(sessions);
    batch_ = std::move(batch);
    text_ = std::move(text);
    forward_options_ = std::move(forward);
    install(std::move(graph));
}
Engine::~Engine() {
    try { shutdown(); }
    catch (const std::exception& error) {
        std::cerr << "runtime_unload_cleanup error=" << error.what() << std::endl;
        std::terminate();
    }
    catch (...) {
        std::cerr << "runtime_unload_cleanup error=unknown" << std::endl;
        std::terminate();
    }
    scheduler_.reset();
    if (executor_) {
        try { unload_model(); }
        catch (const std::exception& error) { std::cerr << "runtime_unload_cleanup error=" << error.what() << std::endl; std::terminate(); }
        catch (...) { std::cerr << "runtime_unload_cleanup error=unknown" << std::endl; std::terminate(); }
    }
}

void Engine::stop_reload() noexcept {
    if (reload_worker_.joinable()) reload_worker_.join();
}

EngineInfo Engine::describe() const { return info_; }

bool Engine::ready() const { return ready_; }

std::unique_ptr<tyloi::Executor::Scope> Engine::enter() { return executor_->enter(); }

void Engine::prepare_request(Request& request, bool batched) {
    engine::prepare_request(request, text_.get(), info_, batched, model_->predictor() != nullptr);
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
    try {
        drain();
        weights_->discard_prefill();
        if (batch_ && config_.preallocate_kv) {
            auto scope = executor_->enter();
            graph_->invalidate();
            budget_->reserve(idle_state_->graph_exit_growth());
            idle_state_->release_graphs();
            serial_state_ = nullptr;
            idle_state_.reset();
            idle_state_ = model_->make_state();
            serial_state_ = idle_state_.get();
            if (auto* predictor = model_->predictor(); predictor && !idle_state_->predictor)
                idle_state_->predictor = predictor->make_state();
        }
    }
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
    try {
        if (predictor_execution_) {
            auto predictor_scope = predictor_execution_->enter();
            predictor_execution_->drain();
        }
    } catch (...) { failure = std::current_exception(); }
    try { if (weights_) weights_->drain(); }
    catch (...) { if (!failure) failure = std::current_exception(); }
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
        serial_state_ = nullptr;
        idle_state_.reset();
        if (weights_) weights_->clear_workspace();
        budget_.reset(); cache_.reset();
        model_.reset();
        text_.reset();
        mtp_state_ = {};
        if (predictor_execution_) {
            auto predictor_scope = predictor_execution_->enter();
            predictor_execution_->clear();
        }
        predictor_execution_.reset();
        // Retire execution-owned KV after state, while its weight cache is alive.
        executor_->clear();
        weights_.reset();
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
            Metrics predictor_metrics;
            if (model_->predictor()) mtp::append_generation_metrics(predictor_metrics, mtp_state_.stats);
            return resource_metrics(model_->metrics(), {}, serial_state_ ? std::optional<Metrics>(serial_state_->metrics()) : std::nullopt,
                model_info_.streamed_kv, batch_ ? batch_->metrics() : Metrics{},
                executor_->metrics(),
                predictor_metrics, source_.get(),
                *ledger_, *weights_, *cache_, *budget_,
                executor_->memory(config_.devices), config_.device, config_.online_budget);
        } else throw std::invalid_argument("model has no duplex component");
    }, std::move(request));
}

bool Engine::can_batch(const EngineRequest& request) const {
    return batch_ && batch_compatible(request, cache_->persistent_prefix_enabled(),
        model_->predictor() != nullptr, batch_->supports_sessions(), batch_->supports_mtp());
}

bool Engine::exclusive() const { return false; }

void Engine::execute(const std::vector<RequestId>& eligible) {
    if (batch_) batch_->step(eligible);
}

void Engine::clear_bindings() noexcept {
    if (executor_) executor_->bind_budget(nullptr);
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
    if (idle_state_) {
        budget_->reserve(idle_state_->graph_exit_growth());
        idle_state_->release_graphs();
        budget_->reserve(idle_state_->reset_growth(1));
        idle_state_->reset(1);
        if (auto* predictor = model_->predictor()) {
            budget_->reserve(predictor->reset_growth(*idle_state_->predictor, 1));
            predictor->reset(*idle_state_->predictor, 1);
        }
    }
    return change.apply();
}

void Engine::reset_states(std::int64_t batch) { scheduler_->reset_states(batch); }

void Scheduler::reset_states(std::int64_t batch) {
    for (auto& [id, request] : requests_) {
        if (!request.state) continue;
        auto& state = *request.state;
        engine_.budget_->reserve(state.graph_exit_growth());
        state.release_graphs();
        engine_.budget_->reserve(state.reset_growth(batch));
        state.reset(batch);
        if (auto* predictor = engine_.model_->predictor(); predictor && state.predictor) {
            engine_.budget_->reserve(predictor->reset_growth(*state.predictor, batch));
            predictor->reset(*state.predictor, batch);
        }
    }
}

} // namespace tyloi::engine

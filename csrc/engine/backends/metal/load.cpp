#include "engine/load.h"
#include "engine/engine.h"
#include "engine/config.h"
#include "engine/backends/metal/executor/executor.h"
#include "engine/backends/metal/native/stream.h"
#include "engine/models/causal_model.h"
#include "engine/models/state.h"
#include "engine/models/load.h"
#include "engine/models/prepare.h"
#include "engine/models/session/cache.h"
#include "engine/models/source.h"
#include "engine/models/source/model_config.h"
#include "engine/requests/processor.h"
#include "engine/state/session/sessions.h"
#include "engine/weights/cache.h"
#include "engine/weights/memory.h"

#include <mlx/mlx.h>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <typeinfo>
#include <utility>

namespace tyloi::metal {
using namespace tyloi::engine;
namespace {
std::unique_ptr<tyloi::Executor> executor() {
    return std::make_unique<metal::Executor>(std::make_shared<metal::Stream>(
        mlx::core::new_thread_unsafe_stream(mlx::core::Device::gpu)));
}

void validate(const EngineOptions& options) {
    if (options.device != Device(DeviceType::metal))
        throw std::invalid_argument("this build supports Metal device 0 only");
    if (options.context_size < 0) throw std::invalid_argument("context size must be non-negative");
    if (options.n_gpu_layers_set || !options.cpu_offload_layers.empty() ||
        options.tensor_parallel.enabled() || options.expert_parallel.enabled() ||
        !options.tensor_parallel.devices.empty() || !options.expert_parallel.devices.empty() ||
        !options.layer_placement.devices.empty() || !options.layer_placement.split.empty())
        throw std::invalid_argument("Metal does not support CPU layer offload or parallel placement");
    if (options.cpu_threads_set || options.cpu_threads || options.yarn_context_extension)
        throw std::invalid_argument("Metal does not support thread or YaRN load options");
    if (!std::isfinite(options.moe_gpu_cache_gb) || options.moe_gpu_cache_gb < 0)
        throw std::invalid_argument("expert cache budget must be finite and non-negative");
    if (!options.moe_cache_profile_path.empty())
        throw std::invalid_argument("Metal expert cache profiles are not supported");
    if (options.continuous_batching)
        throw std::invalid_argument("Metal continuous batching is not supported");
}

std::unique_ptr<Engine> assemble(EngineOptions options, Engine* target) {
    validate(options);
    const auto quantization = kv_quantization();
    const auto kv_fingerprint = quantization.enabled()
        ? "tq-v1-k" + std::to_string(quantization.key_bits()) +
            "v" + std::to_string(quantization.value_bits())
        : std::string{};
    if (!options.context_size) options.context_size = 32768;
    auto config = resolve_config(options.prefill_chunk_size, options.continuous_batching);
    config.device = options.device;
    config.devices = {options.device};
    config.decode_graph.enabled = false;
    config.continuous_batch.cuda_graph = false;
    auto execution = executor();
    std::unique_ptr<tyloi::Executor> predictor_execution;
    auto weights = std::make_unique<weights::Cache>();
    auto memory = std::make_shared<weights::Memory>();
    std::shared_ptr<const ModelSource> source;
    std::unique_ptr<models::CausalModel> model;
    std::unique_ptr<models::State> state;
    std::unique_ptr<tyloi::state::Sessions> sessions;
    std::unique_ptr<TextProcessor> text;
    try {
        {
            auto scope = execution->enter();
            weights::LoadScope memory_load(*memory);
            weights->begin_load();
            if (options.moe_gpu_cache_gb > 0) {
                const auto bytes = options.moe_gpu_cache_gb * double(std::size_t(1) << 30);
                if (bytes < 1 || bytes >= double(std::numeric_limits<std::size_t>::max()))
                    throw std::invalid_argument("expert cache budget is outside the supported range");
                weights->configure(static_cast<std::size_t>(bytes), {}, false, {});
            }
            source = open_model_source(options.model_path);
            auto definition = models::config_json(*source, options.config_path);
            auto loader = weights->loader(*source, {}, {.target = options.device});
            LayerPlacementConfig placement;
            models::LoadOptions policy;
            policy.context = options.context_size;
            policy.kv_quantization = quantization;
            models::load(model, source, definition, loader, placement, policy);
            state = model->make_state();
            if (model->predictor()) predictor_execution = executor();
            const auto info = model->info();
            sessions = models::make_sessions(*source, config.session_cache, config.prefix_cache,
                info.max_context, state->supports_text_session_state(),
                state->supports_paged_text_session_state(), options.device, false, kv_fingerprint);
            text = TextProcessor::load(*source, options.tokenizer_model,
                static_cast<std::int32_t>(info.vocab_size), info.model_type);
            config.stop_token_ids = models::stop_token_ids(definition, info.vocab_size);
            weights->drain();
            execution->drain();
        }
        models::ForwardOptions forward{};
        if (target) {
            target->rebind(std::move(model), std::move(execution), std::move(config),
                std::move(source), std::move(memory), std::move(weights), std::move(state),
                std::move(sessions), std::move(text), forward, {}, {}, std::move(predictor_execution));
            return {};
        }
        return std::make_unique<Engine>(std::move(model), std::move(execution), std::move(config),
            std::move(source), std::move(memory), std::move(weights), std::move(state),
            std::move(sessions), std::move(text), forward, nullptr, nullptr, std::move(predictor_execution));
    } catch (...) {
        const auto failure = std::current_exception();
        if (execution) {
            try {
                auto scope = execution->enter();
                if (weights) weights->drain();
                execution->drain();
                state.reset(); sessions.reset(); model.reset();
                if (weights) { weights->release_workspace(); weights->clear_workspace(); }
                execution->drain();
                execution->clear();
                weights.reset();
            } catch (const std::exception& error) {
                std::cerr << "runtime_load_cleanup error=" << error.what() << '\n';
                std::terminate();
            } catch (...) {
                std::cerr << "runtime_load_cleanup error=unknown\n";
                std::terminate();
            }
        }
        std::rethrow_exception(failure);
    }
}
} // namespace

std::unique_ptr<Engine> load(EngineOptions options) {
    return assemble(std::move(options), nullptr);
}
void reload(Engine& engine, EngineOptions options) {
    if (typeid(engine) != typeid(Engine))
        throw std::invalid_argument("unsupported Metal engine for reload");
    assemble(std::move(options), &engine);
}
} // namespace tyloi::metal

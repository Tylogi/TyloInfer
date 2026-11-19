#include "engine/backends/cuda/load.h"
#include "engine/engine.h"
#include "engine/config.h"
#include "engine/models/common/load.h"
#include "engine/models/prepare.h"
#include "engine/ops/ops.h"
#include "engine/models/load.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/mtp.h"
#include "engine/models/dflash2/model.h"
#include "engine/models/minicpmo45/engine.h"
#include "engine/models/minicpmo45/text/model.h"
#include "engine/models/source/model_config.h"
#include "engine/models/session/cache.h"
#include "engine/models/input.h"
#include "engine/requests/processor.h"
#include "engine/state/session/sessions.h"

#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/device.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/ops/attention/kv/quantization.h"
#include "engine/weights/cache/cache.h"
#include "engine/weights/memory.h"
#include "engine/backends/cuda/executor/kv/memory.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/native/tensor/backend.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>

namespace tyloi::cuda {

namespace {

std::unique_ptr<engine::Engine> load_native_engine(tyloi::engine::EngineOptions options, engine::Engine* target);

} // namespace

std::unique_ptr<engine::Engine> load(tyloi::engine::EngineOptions options) {
    return load_native_engine(std::move(options), nullptr);
}

void reload(engine::Engine& engine, tyloi::engine::EngineOptions options) {
    if (typeid(engine) == typeid(tyloi::engine::Engine) ||
        typeid(engine) == typeid(minicpmo45::Engine)) {
        load_native_engine(std::move(options), &engine);
        return;
    }
    throw std::invalid_argument("unsupported CUDA engine for reload");
}

namespace {
std::unique_ptr<engine::Engine> load_native_engine(tyloi::engine::EngineOptions options, engine::Engine* target) {
    std::shared_ptr<const ModelSource> source;
    std::shared_ptr<tyloi::weights::Memory> ledger;
    auto weights = std::make_unique<tyloi::weights::Cache>();
    CudaLoadConfig load;
    tyloi::engine::Config config;
    std::unique_ptr<Executor> root;
    std::unique_ptr<tyloi::Executor> execution;
    std::unique_ptr<tyloi::Executor> predictor_execution;
    std::unique_ptr<models::CausalModel> model;
    std::unique_ptr<models::State> state;
    std::unique_ptr<tyloi::state::Sessions> sessions;
    std::unique_ptr<tyloi::Executor::Graph> graph;
    try {
        const bool dump_env = env_flag("TYLOINFER_CUDA_ENV_DUMP", false);
        env_freeze();
        if (dump_env) std::cerr << "cuda knobs at load:\n" << env_dump();
        if (!options.context_size) options.context_size = 32768;
        auto [execution_config, numeric_options] = load_cuda_execution_config();
        load = std::move(execution_config);
        configure_load(load, options);
        config = engine::resolve_config(options.prefill_chunk_size, options.continuous_batching);
        config.preallocate_kv = ::tyloi::cuda::options::kv_vram_preallocated();
        if (options.cpu_threads > 0) mfq_set_num_threads(options.cpu_threads);
        root = std::make_unique<Executor>(std::move(numeric_options), options.device.index);
        {
            auto scope = root->enter(options.device.index);
            configure_placement(load, root->collectives(), options.device.index);
            root->set_primary(load.layer_placement.primary_device());
        }
        root->profiler().enabled = false;
        {
            auto outer_scope = root->enter(load.layer_placement.primary_device());
            weights->begin_load();
            if (load.cache_bytes) {
                weights->configure(load.cache_bytes, load.cache, load.weights.preload, load.cache_profile);
                if (!load.cache_profile.empty() && !weights->empty())
                    std::cerr << "moe_cache_profile source=" << (options.moe_cache_profile_path.empty() ? "sidecar" : "explicit")
                        << " path=" << load.cache_profile << '\n';
            }
            ledger = std::make_shared<tyloi::weights::Memory>();
            tyloi::weights::LoadScope memory_load(*ledger);
            root->init_kv();
            {
                auto scope = root->enter(load.layer_placement.primary_device());
                source = tyloi::open_model_source(options.model_path);
                auto definition = models::configure_yarn(load.yarn, load.yarn_context_extension,
                    models::config_json(*source, options.config_path), models::graph(*source).backbone,
                    options.context_size);
                const auto loader = weights->loader(*source, load.weights,
                    {Device(kCUDA, load.layer_placement.primary_device()), load.drop_file_cache,
                        load.tensor_parallel, moe_parallel_config(load)});
                auto policy = model_options(load, options.context_size, true, root->mode().transposed);
                policy.dflash2_draft_path = options.dflash2_draft_path;
                if (policy.dflash2_draft_path.empty())
                    if (const auto* path = env_lookup("TYLOINFER_DFLASH2_DRAFT")) policy.dflash2_draft_path = path;
                models::load(model, source, definition, loader, load.layer_placement, policy);
                load.dense_cpu_layer_count = load.n_gpu_layers >= 0
                    ? static_cast<int>(std::max<std::int64_t>(models::graph(*source).topology.text_layers - load.n_gpu_layers, 0)) : 0;
                if (root->kv_offload() && !model->info().streamed_kv)
                    throw std::invalid_argument("CUDA streamed KV requires a QSA model");
                if (model->predictor())
                    predictor_execution = root->fork(load.layer_placement.primary_device());
                state = model->make_state();
                if (auto* predictor = model->predictor(); predictor && !state->predictor)
                    state->predictor = predictor->make_state();
                root->finalize_kv(*weights);
                std::size_t pending_kv = 0;
                if (config.preallocate_kv) {
                    const Device primary(kCUDA, load.layer_placement.primary_device());
                    for (const auto& [device, bytes] : state->preallocate_growth()) {
                        if (device != primary) continue;
                        if (pending_kv > std::numeric_limits<std::size_t>::max() - bytes)
                            throw std::overflow_error("pending KV reserve overflow");
                        pending_kv += bytes;
                    }
                }
                if (weights->needs_finalize()) {
                    const auto kv = root->kv_memory();
                    const auto reserve = kv ? kv->growth_reserve() : 0;
                    if (reserve > std::numeric_limits<std::size_t>::max() - pending_kv)
                        throw std::overflow_error("KV reserve overflow");
                    weights->finalize(reserve + pending_kv, kv ? kv->expert_budget_charge() : 0);
                }
                weights->drain();
                root->drain();
                report_cuda_memory(load.report_cuda_memory, "loaded");
                if (root->kv_offload() && !model->info().streamed_kv)
                    throw std::invalid_argument("CUDA streamed KV requires a QSA model");
            }
            graph = root->graph();
            auto kv_key = cuda_kv_quantization_tag();
            const auto architecture = source->architecture();
            if (architecture == "qwen4_exp" || architecture == "qwen4") {
                if (const auto bits = env_size("TYLOINFER_QSA_RAM_KV_BITS", 0); bits)
                    kv_key += ";qsa-ram-session-v3=" + std::to_string(bits);
            }
            if (const auto* draft = dynamic_cast<const models::dflash2::Draft*>(model->predictor()))
                kv_key += draft->cache_key;
            sessions = models::make_sessions(*source, config.session_cache, config.prefix_cache,
                model->info().max_context, state->supports_text_session_state(),
                state->supports_paged_text_session_state(),
                Device(kCUDA, load.layer_placement.primary_device()),
                separate_memory(load.layer_placement.primary_device()), kv_key);
        }
        root->mode().ffn.independent |= model->info().separate_projections;
        const auto mode = root->forward_mode(load.layer_placement.primary_device());
        config.device = Device(kCUDA, load.layer_placement.primary_device());
        configure_generation(config, load, *root, *weights, model->info());
        const auto model_info = model->info();
        const auto vocab_size = static_cast<std::int32_t>(model_info.vocab_size);
        auto text = engine::TextProcessor::load(*source, options.tokenizer_model, vocab_size, model_info.model_type);
        config.stop_token_ids = models::stop_token_ids(models::config_json(*source, options.config_path), vocab_size);
        config.resident_budget_bytes = ::tyloi::cuda::options::resident_budget_bytes();
        models::ForwardOptions forward{mode.kv_capacity, mode.trace, mode.trace_layer, mode.primary, mode.ffn.batch_serial};
        const bool mini = dynamic_cast<minicpmo45::text::Model*>(model.get()) != nullptr;
        if (target && (dynamic_cast<minicpmo45::Engine*>(target) != nullptr) != mini)
            throw std::invalid_argument("CUDA reload cannot change the engine model family");
        execution = std::move(root);
        if (target) {
            target->rebind(std::move(model), std::move(execution), std::move(config), std::move(source),
                std::move(ledger), std::move(weights), std::move(state), std::move(sessions),
                std::move(text), forward, std::move(graph), {}, std::move(predictor_execution));
            return {};
        }
        if (mini)
            return std::make_unique<minicpmo45::Engine>(std::move(model), std::move(execution), std::move(config),
                std::move(source), std::move(ledger), std::move(weights), std::move(state),
                std::move(sessions), std::move(text), forward, std::move(graph),
                std::unique_ptr<engine::Batch>{}, std::move(predictor_execution));
        return std::make_unique<engine::Engine>(std::move(model), std::move(execution), std::move(config),
            std::move(source), std::move(ledger), std::move(weights), std::move(state),
            std::move(sessions), std::move(text), forward, std::move(graph),
            std::unique_ptr<engine::Batch>{}, std::move(predictor_execution));
    } catch (...) {
        const auto failure = std::current_exception();
        auto* owner = execution ? execution.get() : root.get();
        if (owner) {
            try {
                auto scope = owner->enter();
                std::exception_ptr cleanup;
                const auto finish = [&](auto&& work) {
                    try { work(); } catch (...) { if (!cleanup) cleanup = std::current_exception(); }
                };
                if (graph) graph->invalidate();
                finish([&] { if (weights) weights->drain(); });
                finish([&] {
                    if (predictor_execution) {
                        auto predictor_scope = predictor_execution->enter();
                        predictor_execution->drain();
                    }
                });
                finish([&] { owner->drain(); });
                if (cleanup) std::rethrow_exception(cleanup);
                if (weights) weights->release_workspace();
                owner->drain();
                graph.reset();
                state.reset(); sessions.reset();
                if (weights) weights->clear_workspace();
                model.reset();
                if (predictor_execution) {
                    auto predictor_scope = predictor_execution->enter();
                    predictor_execution->clear();
                }
                predictor_execution.reset();
                owner->clear();
                weights.reset();
            } catch (const std::exception& error) {
                std::cerr << "runtime_load_cleanup error=" << error.what() << std::endl;
                std::terminate();
            } catch (...) {
                std::cerr << "runtime_load_cleanup error=unknown" << std::endl;
                std::terminate();
            }
            execution.reset();
            root.reset();
        }
        std::rethrow_exception(failure);
    }
}
} // namespace

} // namespace tyloi::cuda

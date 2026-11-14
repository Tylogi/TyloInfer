#pragma once

#include "engine/models/plan.h"
#include "engine/load.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/models/deepseek_v41/load.h"
#include "engine/models/common/components.h"
#include "engine/models/glm5_next/components.h"
#include "engine/models/deepseek_v41/components.h"
#include "engine/models/nemotron_h/components.h"
#include "engine/models/common/load.h"
#include "engine/models/source/model_config.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/qwen35/load.h"
#include "engine/models/load.h"
#include "engine/models/glm_dsa/load.h"
#include "engine/models/glm_dsa/state.h"
#include "engine/models/deepseek_v4/load.h"
#include "engine/models/deepseek_v4/state.h"
#include "engine/models/deepseek_v41/ops.h"
#include "engine/models/deepseek_v41/state.h"
#include "engine/models/glm5_next/load.h"
#include "engine/models/nemotron_h/load.h"
#include "engine/models/nemotron_h/state.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/state/kv_memory.h"
#include "engine/weights/cache.h"
#include "engine/models/qwen35/components.h"
#include "engine/models/qwen35/state.h"
#include "engine/models/minicpmo45/state.h"
#include "engine/models/minicpmo45/text/load.h"
#include "engine/models/minicpmo45/text/state.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

namespace tyloi::cuda {

template <bool IncludeQwen = true>
inline constexpr auto with_model = []<typename F>(
        CudaLoadConfig& config, Executor& root, tyloi::weights::Cache& cache,
        std::shared_ptr<KvMemoryManager>& kv, std::shared_ptr<KvOffloadStore>& offload,
        const CudaLoadOptions& options,
        bool load_optional_components,
        F&& run) -> auto {
    root.configure_kv(kv, offload);
    auto scope = root.enter(config.layer_placement.primary_device());
    auto source = tyloi::open_model_source(options.model_path);
    auto finish = [&](auto& owner, auto& state, auto& executor, auto& components, auto started,
                      std::unique_ptr<tyloi::Executor> predictor_execution = {}) {
        auto& model = *owner;
        using Model = std::remove_cvref_t<decltype(model)>;
        if (auto* predictor = component_predictor(components)) {
            if constexpr (requires { state->predictor; })
                state->predictor = predictor->make_state();
            else
                throw std::logic_error("model state has no predictor state");
        }
        tyloi::cuda::finalize_kv(kv.get(), cache, offload);
        root.configure_kv(kv, offload);
        const auto loaded_at = tyloi::cuda::finish_load(
            kv.get(), cache, config.report_cuda_memory);
        return run(owner, state, executor, components, started, loaded_at, source, std::move(predictor_execution));
    };
    auto load_tts = [&] {
        auto started = std::chrono::steady_clock::now();
        auto definition = tyloi::models::config_json(*source, options.config_path);
        definition = configure_yarn(config.yarn, config.yarn_context_extension, std::move(definition),
            tyloi::models::graph(*source).backbone, options.context_size);
        auto owner = std::make_shared<minicpmo45::Model>(
            minicpmo45::read(source, definition, options.context_size));
        auto& model = *owner;
        if (config.expert_parallel.enabled())
            throw std::runtime_error("--expert-parallel requires a model with routed experts");
        config.layer_placement.prepare(model.config.num_hidden_layers);
        config.dense_cpu_layer_count = config.n_gpu_layers >= 0
            ? static_cast<int>(std::max<int64_t>(model.config.num_hidden_layers - config.n_gpu_layers, 0)) : 0;
        if (config.dense_cpu_layer_count > 0) {
            if (model_parallel_enabled(config) || config.layer_placement.enabled())
                throw std::runtime_error("--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
            std::cerr << "dense_layer_placement cpu=0-" << config.dense_cpu_layer_count - 1
                << " gpu=" << config.dense_cpu_layer_count << '-' << model.config.num_hidden_layers - 1
                << " cpu_threads=" << mfq_get_num_threads() << '\n';
        }
        auto weights = cache.loader(*source, config.weights,
            {config.layer_placement.primary_device(), false, config.drop_file_cache,
             config.tensor_parallel, moe_parallel_config(config)});
        minicpmo45::load(model, weights, config.layer_placement,
            config.dense_cpu_layer_count, config.yarn);
        minicpmo45::State state(model);
        if (::tyloi::cuda::options::kv_vram_preallocated()) state.preallocate();
        std::unique_ptr<minicpmo45::State> owned_state;
        auto executor = root.fork(model_parallel_primary_device(config));
        std::monostate components;
        owned_state = std::make_unique<minicpmo45::State>(std::move(state));
        return finish(owner, owned_state, executor, components, started);
    };

    switch (::tyloi::models::backbone(tyloi::models::graph(*source).backbone)) {
        case ::tyloi::models::Backbone::generic_qwen:
            if constexpr (IncludeQwen) {
                auto started = std::chrono::steady_clock::now();
                auto weights = cache.loader(*source, config.weights,
                    {config.layer_placement.primary_device(), false, config.drop_file_cache,
                     config.tensor_parallel, moe_parallel_config(config)});
                std::shared_ptr<qwen35::Model> owner;
                std::unique_ptr<qwen35::State> state;
                auto executor = root.fork(model_parallel_primary_device(config));
                const bool preallocate = ::tyloi::cuda::options::kv_vram_preallocated();
                auto definition = tyloi::models::config_json(*source, options.config_path);
                definition = configure_yarn(config.yarn, config.yarn_context_extension,
                    std::move(definition), tyloi::models::graph(*source).backbone, options.context_size);
                owner = qwen35::load(
                    source, definition, weights, config.layer_placement,
                    tyloi::models::LoadOptions{
                        .context = options.context_size,
                        .gpu_layers = config.n_gpu_layers,
                        .expert_parallel = config.expert_parallel.enabled(),
                        .recurrent_transposed = root.mode().transposed,
                        .mirror_scalars = config.tensor_parallel_mirror_linear_attention_scalars,
                        .yarn = config.yarn},
                    config.dense_cpu_layer_count);
                state = std::make_unique<qwen35::State>(*owner, root.mode().transposed);
                if (preallocate) state->preallocate();
                std::unique_ptr<tyloi::Executor> predictor_execution;
                auto components = qwen35::Components::load(
                    *owner, weights, config.layer_placement, config.dense_cpu_layer_count,
                    *executor, predictor_execution, load_optional_components,
                    !config.dsv4_cpu_offload_layers.empty(), preallocate);
                return finish(owner, state, executor, components, started, std::move(predictor_execution));
            } else {
                throw std::runtime_error("unsupported CUDA model backbone");
            }
        case ::tyloi::models::Backbone::minicpmo45: {
            auto started = std::chrono::steady_clock::now();
            auto definition = tyloi::models::config_json(*source, options.config_path);
            definition = configure_yarn(config.yarn, config.yarn_context_extension,
                std::move(definition), tyloi::models::graph(*source).backbone, options.context_size);
            auto owner = std::make_shared<minicpmo45::text::Model>(
                minicpmo45::text::read(source, definition, options.context_size));
            auto weights = cache.loader(*source, config.weights,
                {config.layer_placement.primary_device(), false, config.drop_file_cache,
                 config.tensor_parallel, moe_parallel_config(config)});
            config.dense_cpu_layer_count = config.n_gpu_layers >= 0
                ? static_cast<int>(std::max<int64_t>(owner->config.num_hidden_layers - config.n_gpu_layers, 0)) : 0;
            minicpmo45::text::load(*owner, weights, config.layer_placement,
                config.n_gpu_layers, config.expert_parallel.enabled(), true, true, config.yarn);
            auto state = std::make_unique<minicpmo45::text::State>(*owner);
            state->preallocate();
            auto executor = root.fork(model_parallel_primary_device(config));
            auto components = minicpmo45::load_components(*owner, *executor, weights,
                load_optional_components, config.layer_placement.primary_device());
            return finish(owner, state, executor, components, started);
        }
        case ::tyloi::models::Backbone::minicpmo_tts:
            return load_tts();
        case ::tyloi::models::Backbone::glm_dsa: {
            auto started = std::chrono::steady_clock::now();
            auto definition = tyloi::models::config_json(*source, options.config_path);
            definition = configure_yarn(config.yarn, config.yarn_context_extension,
                std::move(definition), tyloi::models::graph(*source).backbone, options.context_size);
            auto owner = std::make_shared<glm_dsa::Model>(
                glm_dsa::read(source, definition, options.context_size));
            auto weights = cache.loader(*source, config.weights,
                {config.layer_placement.primary_device(), false, config.drop_file_cache,
                 config.tensor_parallel, moe_parallel_config(config)});
            config.dense_cpu_layer_count = 0;
            glm_dsa::load(*owner, weights, config.layer_placement,
                          config.n_gpu_layers, true, true, config.yarn);
            auto state = std::make_unique<glm_dsa::State>(*owner);
            state->preallocate();
            auto executor = root.fork(model_parallel_primary_device(config));
            if (load_optional_components && (owner->plan.vision != ::tyloi::models::VisionAdapter::none ||
                    owner->plan.predictor != ::tyloi::models::PredictorAdapter::none))
                throw std::runtime_error("CUDA component adapter is unsupported for this model");
            std::monostate components;
            return finish(owner, state, executor, components, started);
        }
        case ::tyloi::models::Backbone::glm5_next: {
            auto started = std::chrono::steady_clock::now();
            auto definition = tyloi::models::config_json(*source, options.config_path);
            definition = configure_yarn(config.yarn, config.yarn_context_extension,
                std::move(definition), tyloi::models::graph(*source).backbone, options.context_size);
            auto owner = std::make_shared<glm5_next::Model>(
                glm5_next::read(source, definition, options.context_size));
            auto weights = cache.loader(*source, config.weights,
                {config.layer_placement.primary_device(), false, config.drop_file_cache,
                 config.tensor_parallel, moe_parallel_config(config)});
            config.dense_cpu_layer_count = 0;
            glm5_next::load(*owner, weights, config.layer_placement, config.n_gpu_layers);
            auto state = std::make_unique<glm5_next::State>(*owner);
            state->preallocate();
            auto executor = root.fork(model_parallel_primary_device(config));
            auto components = glm5_next::load_components(*owner, weights, load_optional_components);
            return finish(owner, state, executor, components, started);
        }
        case ::tyloi::models::Backbone::deepseek_v4: {
            auto started = std::chrono::steady_clock::now();
            auto definition = tyloi::models::config_json(*source, options.config_path);
            definition = configure_yarn(config.yarn, config.yarn_context_extension,
                std::move(definition), tyloi::models::graph(*source).backbone, options.context_size);
            auto owner = std::make_shared<deepseek_v4::Model>(
                deepseek_v4::read(source, definition, options.context_size));
            deepseek_v4::validate_offload(owner->config, config.dsv4_cpu_offload_layers);
            if (!config.dsv4_cpu_offload_layers.empty()) {
                config.dsv4_cpu_offload_host_bytes = 0;
                config.drop_file_cache = true;
            }
            auto weights = cache.loader(*source, config.weights,
                {config.layer_placement.primary_device(), false, config.drop_file_cache,
                 config.tensor_parallel, moe_parallel_config(config)});
            config.dense_cpu_layer_count = 0;
            deepseek_v4::load(*owner, weights, config.layer_placement, root.collectives(),
                model_parallel_primary_device(config), config.dsv4_cpu_offload_layers,
                config.dsv4_cpu_offload_host_bytes, config.n_gpu_layers);
            auto state = std::make_unique<deepseek_v4::State>(*owner);
            state->preallocate();
            auto executor = root.fork(model_parallel_primary_device(config));
            if (load_optional_components && (owner->plan.vision != ::tyloi::models::VisionAdapter::none ||
                    owner->plan.predictor != ::tyloi::models::PredictorAdapter::none))
                throw std::runtime_error("CUDA component adapter is unsupported for this model");
            std::monostate components;
            return finish(owner, state, executor, components, started);
        }
        case ::tyloi::models::Backbone::deepseek_v41: {
            auto started = std::chrono::steady_clock::now();
            auto definition = tyloi::models::config_json(*source, options.config_path);
            definition = configure_yarn(config.yarn, config.yarn_context_extension,
                std::move(definition), tyloi::models::graph(*source).backbone, options.context_size);
            auto weights = cache.loader(*source, config.weights,
                {config.layer_placement.primary_device(), false, config.drop_file_cache,
                 config.tensor_parallel, moe_parallel_config(config)});
            std::unique_ptr<tyloi::models::CausalModel> model;
            deepseek_v41::load(model, source, definition, weights, config.layer_placement,
                {.context = options.context_size, .gpu_layers = config.n_gpu_layers,
                 .expert_parallel = config.expert_parallel.enabled(), .yarn = config.yarn,
                 .cpu_offload_layers = config.dsv4_cpu_offload_layers,
                 .engram_cache_rows = config.deepseek_v41_engram_cache_rows,
                 .host_tokens = config.cache.pipeline, .optional_components = false});
            auto owner = std::shared_ptr<deepseek_v41::Model>(
                static_cast<deepseek_v41::Model*>(model.release()));
            config.dense_cpu_layer_count = 0;
            auto components = deepseek_v41::load_components(*owner, weights, load_optional_components);
            auto state = std::unique_ptr<deepseek_v41::State>(static_cast<deepseek_v41::State*>(
                owner->make_state().release()));
            auto executor = root.fork(model_parallel_primary_device(config));
            return finish(owner, state, executor, components, started);
        }
        case ::tyloi::models::Backbone::nemotron_h: {
            auto started = std::chrono::steady_clock::now();
            auto definition = tyloi::models::config_json(*source, options.config_path);
            definition = configure_yarn(config.yarn, config.yarn_context_extension,
                std::move(definition), tyloi::models::graph(*source).backbone, options.context_size);
            auto owner = std::make_shared<nemotron_h::Model>(
                nemotron_h::read(source, definition, options.context_size));
            auto weights = cache.loader(*source, config.weights,
                {config.layer_placement.primary_device(), false, config.drop_file_cache,
                 config.tensor_parallel, moe_parallel_config(config)});
            config.dense_cpu_layer_count = 0;
            nemotron_h::load(*owner, weights, config.layer_placement,
                             config.n_gpu_layers, config.expert_parallel.enabled());
            auto state = std::make_unique<nemotron_h::State>(*owner);
            state->preallocate();
            auto executor = root.fork(model_parallel_primary_device(config));
            auto components = nemotron_h::load_components(*owner, weights, load_optional_components);
            return finish(owner, state, executor, components, started);
        }
        case ::tyloi::models::Backbone::unsupported:
        default:
            throw std::runtime_error("unsupported CUDA model backbone");
    }
    throw std::runtime_error("invalid CUDA model backbone");
};

} // namespace tyloi::cuda

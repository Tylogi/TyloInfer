#include "engine/program.h"

#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"
#include "engine/backends/cuda/native/device.h"
#include "engine/budget.h"
#include "engine/config.h"
#include "engine/generation/decode.h"
#include "engine/generation/batch_driver.h"
#include "engine/generation/batch.h"
#include "engine/generation/batch_ops.h"
#include "engine/generation/prefill_budget.h"
#include "engine/generation/prefill.h"
#include "engine/generation/run.h"
#include "engine/generation/mtp/state.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/load.h"
#include "engine/models/common/mtp.h"
#include "engine/models/layers.h"
#include "engine/models/groups.h"
#include "engine/models/ffn.h"
#include "engine/models/load.h"
#include "engine/models/session/cache.h"
#include "engine/models/source.h"
#include "engine/models/source/model_config.h"
#include "engine/models/source/model_graph.h"
#include "engine/backends/cuda/executor/kv_memory.h"
#include "engine/state/session/sessions.h"
#include "engine/weights/cache.h"
#include "engine/weights/memory.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda {
namespace {
/** Drain before retiring a request's coroutine and release its exclusive owner flag. */
struct RequestWork {
    Program& program;
    engine::Generation job;
    bool& running;
    bool complete = false;
    void drain() noexcept {
        try { program.drain(); }
        catch (const std::exception& error) { std::cerr << "request_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "request_cleanup error=unknown\n"; }
    }
    ~RequestWork() noexcept {
        if (!complete) drain();
        job = {};
        if (!complete) drain();
        running = false;
    }
};

} // namespace

struct Program::Impl {
    tyloi::engine::LoadOptions options;
    CudaLoadConfig config;
    std::unique_ptr<Executor> executor;
    std::unique_ptr<::tyloi::weights::Cache> weights;
    std::shared_ptr<const ModelSource> source;
    std::shared_ptr<::tyloi::weights::Memory> ledger;
    std::unique_ptr<models::CausalModel> model;
    std::unique_ptr<models::State> state;
    std::unique_ptr<models::Layers> layers;
    std::unique_ptr<models::Groups> groups;
    std::unique_ptr<models::RoutedFfn> ffn;
    engine::Config run_config;
    models::Info run_info{};
    models::ForwardOptions forward{};
    std::unique_ptr<tyloi::state::Sessions> sessions;
    std::optional<engine::Decode> graph;
    std::optional<engine::Budget> budget;
    std::unique_ptr<engine::BatchDriver> batch_driver;
    engine::mtp::State mtp_policy;
    bool attempted = false;
    bool run_attempted = false, run_ready = false, running = false;

    std::string definition(std::int64_t context) {
        return models::configure_yarn(config.yarn, config.yarn_context_extension,
            models::config_json(*source, options.config_path), models::graph(*source).backbone, context);
    }

    void drain() {
        auto scope = executor->enter(config.layer_placement.primary_device());
        std::exception_ptr failure;
        try { weights->drain(); } catch (...) { failure = std::current_exception(); }
        try { executor->drain(); } catch (...) { if (!failure) failure = std::current_exception(); }
        if (failure) std::rethrow_exception(failure);
    }

    void finish() {
        if (weights->needs_finalize()) {
            const auto kv = executor->kv_memory();
            weights->finalize(kv ? kv->growth_reserve() : 0, kv ? kv->expert_budget_charge() : 0);
        }
        drain();
        report_cuda_memory(config.report_cuda_memory, "loaded");
    }

    /** Open the source only for a model-loading operation, after execution setup. */
    void open_source() {
        if (options.model_path.empty()) throw std::logic_error("program has no model source");
        if (!source) source = open_model_source(options.model_path);
    }

    /** Drain before destroying execution-dependent resources, including on partial setup. */
    void cleanup() noexcept {
        if (!executor) return;
        try { if (weights) drain(); }
        catch (const std::exception& error) { std::cerr << "program_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "program_cleanup error=unknown\n"; }
        std::unique_ptr<tyloi::Executor::Scope> scope;
        try { scope = executor->enter(config.layer_placement.primary_device()); }
        catch (const std::exception& error) { std::cerr << "program_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "program_cleanup error=unknown\n"; }
        batch_driver.reset();
        graph.reset();
        state.reset();
        layers.reset();
        groups.reset();
        ffn.reset();
        model.reset();
        executor->bind_budget(nullptr);
        budget.reset();
        sessions.reset();
        try { executor->clear(); }
        catch (const std::exception& error) { std::cerr << "program_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "program_cleanup error=unknown\n"; }
        weights.reset(); ledger.reset(); source.reset();
        scope.reset();
        executor.reset();
    }
};

Program::Program() : impl_(std::make_unique<Impl>()) {}
Program::~Program() noexcept { impl_->cleanup(); }

std::unique_ptr<Program> Program::bare() {
    auto program = std::unique_ptr<Program>(new Program());
    auto& p = *program->impl_;
    p.options.device = Device(DeviceType::cuda);
    const bool dump_env = env_flag("TYLOINFER_CUDA_ENV_DUMP", false);
    if (dump_env) std::cerr << "cuda knobs at load:\n" << env_dump();
    auto [config, numeric] = load_cuda_execution_config();
    p.config = std::move(config);
    p.weights = std::make_unique<::tyloi::weights::Cache>();
    p.executor = std::make_unique<Executor>(std::move(numeric), 0);
    p.executor->profiler().enabled = false;
    p.ledger = std::make_shared<::tyloi::weights::Memory>();
    return program;
}

std::unique_ptr<Program> Program::open(tyloi::engine::LoadOptions options, bool adapt) {
    if (!options.device.is_cuda()) throw std::invalid_argument("CUDA Program requires a CUDA device");
    auto program = bare();
    auto& p = *program->impl_;
    p.options = std::move(options);
    if (!adapt) p.config.cache.adapt = false;
    configure_load(p.config, p.options);
    if (p.options.cpu_threads > 0) mfq_set_num_threads(p.options.cpu_threads);
    {
        auto scope = p.executor->enter(p.options.device.index);
        configure_placement(p.config, p.executor->collectives(), p.options.device.index);
        p.executor->set_primary(p.config.layer_placement.primary_device());
    }
    auto scope = p.executor->enter(p.config.layer_placement.primary_device());
    p.weights->begin_load();
    if (p.config.cache_bytes) {
        p.weights->configure(p.config.cache_bytes, p.config.cache, p.config.weights.preload, p.config.cache_profile);
        if (!p.config.cache_profile.empty() && !p.weights->empty())
            std::cerr << "moe_cache_profile source=" << (p.options.moe_cache_profile_path.empty() ? "sidecar" : "explicit")
                << " path=" << p.config.cache_profile << '\n';
    }
    p.executor->init_kv();
    return program;
}

void Program::load(bool components) {
    auto& p = *impl_;
    if (p.attempted || !p.executor) throw std::logic_error("program load already attempted");
    p.attempted = true;
    p.open_source();
    {
        auto scope = p.executor->enter(p.config.layer_placement.primary_device());
        ::tyloi::weights::LoadScope memory(*p.ledger);
        const auto definition = p.definition(p.options.context_size);
        const auto input = loader();
        const auto policy = model_options(p.config, p.options.context_size, components, p.executor->mode().transposed);
        models::load(p.model, p.source, definition, input, p.config.layer_placement, policy);
        p.config.dense_cpu_layer_count = p.config.n_gpu_layers >= 0
            ? static_cast<int>(std::max<std::int64_t>(models::graph(*p.source).topology.text_layers - p.config.n_gpu_layers, 0)) : 0;
        if (p.executor->kv_offload() && !p.model->info().streamed_kv)
            throw std::invalid_argument("CUDA streamed KV requires a QSA model");
        p.state = p.model->make_state();
        if (auto* predictor = p.model->predictor(); predictor && !p.state->predictor)
            p.state->predictor = predictor->make_state();
        if (cuda::options::kv_vram_preallocated()) p.state->preallocate();
        Executor::prepare(p.model->info().vocab_size);
        p.executor->finalize_kv(*p.weights);
        p.finish();
    }
}

void Program::load_layers() {
    auto& p = *impl_;
    if (p.attempted || !p.executor) throw std::logic_error("program load already attempted");
    p.attempted = true;
    p.open_source();
    {
        auto scope = p.executor->enter(p.config.layer_placement.primary_device());
        ::tyloi::weights::LoadScope memory(*p.ledger);
        if (p.executor->kv_offload()) throw std::invalid_argument("layer-wise execution does not support streamed KV");
        const auto definition = p.definition(p.options.context_size);
        const auto input = loader();
        const auto policy = model_options(p.config, p.options.context_size, false, p.executor->mode().transposed);
        models::load_layers(p.layers, p.source, definition, input, p.config.layer_placement, policy);
        p.finish();
    }
}

void Program::load_groups(std::int64_t context) {
    auto& p = *impl_;
    if (p.attempted || !p.executor) throw std::logic_error("program load already attempted");
    if (context < 1) throw std::invalid_argument("grouped execution requires a positive context");
    p.attempted = true;
    p.open_source();
    auto scope = p.executor->enter(p.config.layer_placement.primary_device());
    ::tyloi::weights::LoadScope memory(*p.ledger);
    if (p.executor->kv_offload()) throw std::invalid_argument("grouped execution does not support streamed KV");
    const auto definition = p.definition(context);
    const auto input = loader();
    const auto policy = model_options(p.config, context, false, p.executor->mode().transposed);
    p.groups = models::load_groups(p.source, definition, input, p.config.layer_placement, policy);
    p.groups->prepare();
    p.finish();
}

void Program::load_ffn(int layer) {
    auto& p = *impl_;
    if (p.attempted || !p.executor) throw std::logic_error("program load already attempted");
    p.attempted = true;
    p.open_source();
    auto scope = p.executor->enter(p.config.layer_placement.primary_device());
    ::tyloi::weights::LoadScope memory(*p.ledger);
    if (p.executor->kv_offload()) throw std::invalid_argument("FFN execution does not support streamed KV");
    const auto definition = p.definition(p.options.context_size);
    const auto input = loader();
    const auto policy = model_options(p.config, p.options.context_size, false, p.executor->mode().transposed);
    p.ffn = models::load_ffn(p.source, definition, input, p.config.layer_placement, policy, layer);
    p.ffn->prepare();
    p.finish();
}

Executor& Program::executor() { if (!impl_->executor) throw std::logic_error("program closed"); return *impl_->executor; }
::tyloi::weights::Cache& Program::weights() { if (!impl_->weights) throw std::logic_error("program closed"); return *impl_->weights; }
::tyloi::weights::Loader Program::loader() {
    auto& p = *impl_;
    if (!p.executor || !p.weights) throw std::logic_error("program closed");
    p.open_source();
    return p.weights->loader(*p.source, p.config.weights,
        {Device(kCUDA, p.config.layer_placement.primary_device()), p.config.drop_file_cache,
            p.config.tensor_parallel, moe_parallel_config(p.config)});
}
CudaLoadConfig& Program::config() { return impl_->config; }
models::CausalModel& Program::model() { if (!impl_->model) throw std::logic_error("model not loaded"); return *impl_->model; }
models::State& Program::state() { if (!impl_->state) throw std::logic_error("state not loaded"); return *impl_->state; }
models::Layers& Program::layers() { if (!impl_->layers) throw std::logic_error("layers not loaded"); return *impl_->layers; }
models::Groups& Program::groups() { if (!impl_->groups) throw std::logic_error("groups not loaded"); return *impl_->groups; }
models::RoutedFfn& Program::ffn() { if (!impl_->ffn) throw std::logic_error("FFN not loaded"); return *impl_->ffn; }

void Program::configure(engine::Config config) {
    auto& p = *impl_;
    if (!p.model || !p.state) throw std::logic_error("model not loaded");
    if (p.running) throw std::logic_error("program request is already running");
    if (p.run_attempted && !p.run_ready) throw std::logic_error("generation configuration failed");
    p.run_attempted = true;
    p.run_ready = false;
    drain();
    auto scope = p.executor->enter(p.config.layer_placement.primary_device());
    if (p.budget) p.budget->reserve(p.state->graph_exit_growth());
    p.state->release_graphs();
    p.executor->bind_budget(nullptr);
    p.budget.reset();
    p.graph.reset();
    p.sessions.reset();
    ::tyloi::weights::LoadScope memory(*p.ledger);
    p.run_config = std::move(config);
    auto& runtime = p.run_config;
    p.run_info = p.model->info();
    if (auto* predictor = p.model->predictor(); predictor &&
        (!p.state->predictor || !p.state->predictor->belongs_to(*predictor)))
        throw std::invalid_argument("predictor state is unavailable or belongs to another model");
    configure_generation(runtime, p.config, *p.executor, *p.weights, p.run_info);
    runtime.device = Device(kCUDA, p.config.layer_placement.primary_device());
    const auto mode = p.executor->forward_mode(p.config.layer_placement.primary_device());
    p.forward = {mode.kv_capacity, mode.trace, mode.trace_layer, mode.primary};
    p.sessions = models::make_sessions(*p.source, runtime.session_cache, runtime.prefix_cache,
        p.run_info.max_context, p.state->supports_text_session_state(), p.state->supports_paged_text_session_state(),
        runtime.device, separate_memory(p.config.layer_placement.primary_device()), cuda_kv_quantization_tag());
    p.graph.emplace(p.run_info.max_context,
        Device(kCUDA, p.config.layer_placement.primary_device()), p.executor->graph());
    p.budget.emplace(*p.executor, *p.ledger, *p.weights, *p.sessions, runtime.device);
    p.budget->bind(p.model->predictor() != nullptr);
    p.executor->bind_budget(&*p.budget);
    if (runtime.resident_budget_bytes) {
        if (!runtime.online_budget)
            throw std::invalid_argument("online memory budgeting requires a single-device CUDA worker");
        engine::SessionCommand command;
        command.kind = engine::SessionCommand::Kind::memory_budget;
        command.bytes = runtime.resident_budget_bytes;
        auto change = p.budget->change(command);
        p.weights->settle();
        p.graph->invalidate();
        p.state->release_graphs();
        auto* predictor = p.model->predictor();
        if (predictor && (!p.state->predictor || !p.state->predictor->belongs_to(*predictor)))
            throw std::invalid_argument("predictor state is unavailable or belongs to another model");
        for (const auto& [device, bytes] : p.state->reset_growth(1))
            if (bytes) p.budget->reserve_kv(device, engine::prefill_budget_add(bytes, std::size_t(64) << 20));
        p.state->reset(1);
        if (predictor) {
            for (const auto& [device, bytes] : predictor->reset_growth(*p.state->predictor, 1))
                if (bytes) p.budget->reserve_kv(device, engine::prefill_budget_add(bytes, std::size_t(64) << 20));
            predictor->reset(*p.state->predictor, 1);
        }
        (void)change.apply();
    }
    models::DecodeOptions decode;
    decode.enabled = runtime.decode_graph.enabled && p.weights->decode() != weights::Cache::Decode::none;
    decode.routed = p.weights->decode() == weights::Cache::Decode::routed;
    engine::warmup(*p.model, *p.state, *p.executor, *p.weights, *p.budget, runtime.device, decode);
    p.run_ready = true;
}

Program::BatchLease::BatchLease(Program& owner) : owner_(owner) {}

Program::BatchLease::~BatchLease() noexcept {
    try { owner_.drain(); }
    catch (const std::exception& error) { std::cerr << "batch_cleanup error=" << error.what() << '\n'; }
    catch (...) { std::cerr << "batch_cleanup error=unknown\n"; }
    std::unique_ptr<tyloi::Executor::Scope> scope;
    try { scope = owner_.executor().enter(owner_.config().layer_placement.primary_device()); }
    catch (const std::exception& error) { std::cerr << "batch_cleanup error=" << error.what() << '\n'; }
    catch (...) { std::cerr << "batch_cleanup error=unknown\n"; }
    batch_.reset();
    owner_.impl_->running = false;
}

engine::Batch& Program::BatchLease::batch() {
    if (!batch_) throw std::logic_error("batch is not available");
    return *batch_;
}

std::unique_ptr<Program::BatchLease> Program::make_batch() {
    auto& p = *impl_;
    if (p.running) throw std::logic_error("program request is already running");
    if (!p.run_ready) configure(engine::resolve_config());
    auto lease = std::unique_ptr<BatchLease>(new BatchLease(*this));
    p.running = true;
    auto scope = p.executor->enter(p.config.layer_placement.primary_device());
    p.graph->invalidate();
    p.budget->reserve(p.state->graph_exit_growth());
    p.state->release_graphs();
    lease->batch_ = engine::make_batch(*p.model, *p.executor,
        *p.weights, *p.budget, p.run_config, p.forward);
    return lease;
}

engine::BatchDriver& Program::begin_batch(models::BatchConfig config) {
    auto& p=*impl_;
    if(p.running)throw std::logic_error("program request is already running");
    if(!p.run_ready)configure(engine::resolve_config());
    drain();
    p.running=true;
    try {
        auto scope=p.executor->enter(p.config.layer_placement.primary_device());
        p.budget->reserve(p.state->graph_exit_growth());
        p.graph->invalidate();
        p.state->release_graphs();
        config.forward=p.forward;
        p.batch_driver=std::make_unique<engine::BatchDriver>(
            p.model->make_batch(config),*p.executor,*p.weights,*p.budget,p.run_config.device,
            p.run_config.workspace_reserve);
        p.batch_driver->initialize();
        return *p.batch_driver;
    } catch(...) {
        auto failure=std::current_exception();
        try { drain(); } catch(...) {}
        p.batch_driver.reset(); p.running=false;
        std::rethrow_exception(failure);
    }
}

void Program::end_batch() {
    auto& p=*impl_;
    if(!p.running || !p.batch_driver)throw std::logic_error("numerical batch is not active");
    drain();
    auto scope=p.executor->enter(p.config.layer_placement.primary_device());
    p.batch_driver->close(); p.batch_driver.reset(); p.running=false;
}

models::Prefill Program::prefill(models::Input input, std::int64_t compute, engine::InferenceOutput& output) {
    auto& p = *impl_;
    if (p.running) throw std::logic_error("program request is already running");
    if (!p.run_ready) configure(engine::resolve_config());
    if (compute <= 0) throw std::invalid_argument("prefill compute chunk must be positive");
    p.running = true;
    models::Prefill result;
    try {
        auto scope = p.executor->enter(p.config.layer_placement.primary_device());
        result = engine::prefill(*p.model, *p.state, *p.executor, *p.weights,
            *p.budget, p.forward, input, compute, output);
        p.running = false;
        return result;
    } catch (...) {
        auto failure = std::current_exception();
        try { drain(); }
        catch (const std::exception& error) { std::cerr << "prefill_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "prefill_cleanup error=unknown\n"; }
        p.running = false;
        std::rethrow_exception(failure);
    }
}

std::optional<models::Decoded> Program::decode(models::Input input, models::DecodeOptions options) {
    auto& p = *impl_;
    if (p.running) throw std::logic_error("program request is already running");
    if (!p.run_ready) configure(engine::resolve_config());
    auto scope = p.executor->enter(p.config.layer_placement.primary_device());
    p.budget->reserve(p.state->graph_exit_growth());
    options.enabled &= p.weights->decode() != weights::Cache::Decode::none;
    return engine::decode(*p.model, *p.state, *p.executor, *p.weights, *p.budget,
        p.forward, std::move(input), options);
}

Tensor Program::restore(const TextSessionState& snapshot) {
    auto& p = *impl_;
    if (p.running) throw std::logic_error("program request is already running");
    if (!p.run_ready) configure(engine::resolve_config());
    auto scope = p.executor->enter(p.config.layer_placement.primary_device());
    auto* predictor = snapshot.mtp ? p.model->predictor() : nullptr;
    if (snapshot.mtp && !predictor)
        throw std::invalid_argument("session predictor is unavailable");
    p.graph->invalidate();
    return tyloi::state::restore(*p.state, snapshot, *p.executor, *p.weights, *p.budget, predictor);
}

StepSequence<engine::EventData> Program::generate(engine::InferenceRequest& request,
        engine::InferenceOutput& output, bool predictor, std::string id) {
    auto& p = *impl_;
    if (p.running) throw std::logic_error("program request is already running");
    if (!p.run_ready) configure(engine::resolve_config());
    p.running = true;
    RequestWork work{*this, {}, p.running};
    work.job = engine::generate({*p.model, *p.state, *p.executor, *p.budget, *p.graph, *p.weights,
        *p.sessions, p.run_info, p.run_config, p.forward, p.source.get(), p.mtp_policy,
        predictor ? p.model->predictor() : nullptr}, request, output, std::move(id));
    for (;;) {
        StepResult<engine::EventData> step;
        { auto scope = p.executor->enter(p.config.layer_placement.primary_device()); step = work.job.next(); }
        if (!step) break;
        co_yield std::move(step);
    }
    work.complete = true;
}

Program::Stats Program::stats() const {
    const auto& p = *impl_;
    return {p.mtp_policy.stats, p.mtp_policy.cycles, p.mtp_policy.accepted, p.mtp_policy.rejected,
        p.graph ? p.graph->captures : 0, p.graph ? p.graph->reuses : 0};
}

void Program::drain() {
    auto& p = *impl_;
    if (!p.executor || !p.weights) throw std::logic_error("program closed");
    p.drain();
}
} // namespace tyloi::cuda

#include "engine/models/minicpmo45/text/load.h"
#include "engine/models/minicpmo45/text/forward.h"
#include "engine/models/common/ffn.h"
#include "engine/models/common/mlp.h"
#include "engine/models/common/load.h"
#include "engine/weights/loader.h"
#include "engine/models/common/check.h"
#include "engine/models/session/state.h"
#include "engine/models/source/model_graph.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/native/collectives.h"
#endif
#include "engine/backends/cuda/executor/options.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#endif

#include <algorithm>
#include <iostream>
#include <limits>

namespace tyloi::cuda::minicpmo45::text {

Model read(std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition, int64_t context) {
    validate_model_source(*source);
    Model model;
    model.source = std::move(source);
    model.graph = tyloi::models::graph(*model.source);
    model.plan = cuda_model_plan(model.graph);
    tyloi::models::require_model(model.graph.backbone == "minicpmo45",
        "model backbone does not match the requested causal model");
    auto& config = model.config;
    config = Config::from_json(definition);
    if (config.model_type.empty()) config.model_type = model.graph.architecture;
    config.rotary_dim = config.head_dim;
    config.layer_types.assign(static_cast<size_t>(config.num_hidden_layers), "full_attention");
    if (config.hidden_size != 4096 || config.intermediate_size != 12288 ||
        config.num_hidden_layers != 36 || config.num_attention_heads != 32 ||
        config.num_key_value_heads != 8 || config.head_dim != 128 ||
        config.hidden_act != "silu" || config.attention_bias || config.use_sliding_window)
        throw std::runtime_error("unsupported MiniCPM-o 4.5 Qwen3 configuration");
    auto& metadata = model.metadata;
    metadata.vocab_size = config.vocab_size;
    metadata.hidden_size = config.hidden_size;
    metadata.num_hidden_layers = config.num_hidden_layers;
    metadata.num_attention_heads = config.num_attention_heads;
    metadata.num_key_value_heads = config.num_key_value_heads;
    metadata.head_dim = config.head_dim;
    metadata.max_position_embeddings = config.max_position_embeddings;
    metadata.rotary_dim = config.rotary_dim;
    metadata.rope_base = config.rope_base;
    metadata.rms_norm_eps = config.rms_norm_eps;
    metadata.tie_word_embeddings = config.tie_word_embeddings;
    metadata.model_type = config.model_type;
    metadata.layer_types = config.layer_types;
    metadata.rope_interleaved = true;
    metadata.decode_graph_double_warmup = true;
    tyloi::models::require_model(config.num_hidden_layers == model.graph.topology.text_layers,
        "model graph/config text-layer topology mismatch");
    if (context > 0) {
        tyloi::models::require_model(context <= model.metadata.max_position_embeddings,
            "--ctx-size exceeds max_position_embeddings");
        config.max_position_embeddings = model.metadata.max_position_embeddings = context;
    }
    return model;
}

Layer load_layer(const tyloi::weights::Loader& weights, const Config& config, int index,
                 const std::string& type) {
    if (type != "full_attention")
        throw std::runtime_error("unsupported MiniCPM layer type: " + type);
    Layer layer;
    layer.device = weights.placement.device;
    layer.cpu = weights.placement.cpu;
    layer.attention.official_bf16 = true;
    TransformerWeightLoader loader{weights};
    loader.preserve_projection_boundaries = true;
    const auto prefix = "model.block." + std::to_string(index) + ".";
    tyloi::models::load_attention_mixer(layer.attention, loader, config, index, prefix, 0.0);
#ifdef TYLOINFER_NATIVE_CUDA
    layer.ffn_norm = weights.tensor(prefix + "mlp.norm.weight");
#else
    layer.ffn_norm = loader.dense(prefix + "mlp.norm.weight");
#endif
    layer.ffn = tyloi::models::load_dense_ffn<tyloi::models::Mlp>(loader, config, prefix + "mlp.");
    return layer;
}

void load(Model& model, const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
          int gpu_layers, bool expert_parallel, bool load_layers, bool load_input_output,
          const tyloi::ops::YarnScaling& yarn) {
    const auto& config = model.config;
    if (expert_parallel)
        throw std::runtime_error("--expert-parallel requires a model with routed experts");
    placement.prepare(config.num_hidden_layers);
    const int cpu_layers = gpu_layers < 0 ? 0 :
        static_cast<int>(std::max<int64_t>(config.num_hidden_layers - gpu_layers, 0));
    if (cpu_layers > 0 && (weights.placement.tensor.enabled() || placement.enabled()))
        throw std::runtime_error("--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
    if (model.plan.vision == CudaVisionAdapter::grid_vit && cpu_layers > 0)
        throw std::runtime_error("CUDA grid-Vision currently requires GPU-resident text layers");
    MfqCudaGuard guard(placement.primary_device());
    const auto make_rope = [&](::tyloi::Device device) {
        RopeCache rope(config.max_position_embeddings, config.rotary_dim, config.rope_base,
                         0, -1, device, true, yarn);

#ifdef TYLOINFER_NATIVE_CUDA
        return rope.view();
#else
        return rope;
#endif
    };
    model.rope = make_rope(::tyloi::Device(::tyloi::kCUDA, placement.primary_device()));
    if (cpu_layers > 0) model.cpu_rope = make_rope(::tyloi::Device(::tyloi::kCPU));
    if (placement.enabled())
        for (int device : placement.devices) {
            MfqCudaGuard device_guard(device);
            model.device_ropes.emplace(device, make_rope(::tyloi::Device(::tyloi::kCUDA, device)));
        }
    if (load_input_output) {
        const bool separate = !config.tie_word_embeddings && weights.has("model.output.weight");
        model.embed = weights.embedding("model.token_embedding.weight", separate);
#ifdef TYLOINFER_NATIVE_CUDA
        model.output_norm = weights.tensor("model.output_norm.weight");
#else
        model.output_norm = weights.tensor("model.output_norm.weight").dense();
#endif
        model.lm_head = separate ? weights.weight("model.output.weight", TensorParallelAxis::Output)
            : weights.placement.tensor.enabled()
                ? weights.weight("model.token_embedding.weight", TensorParallelAxis::Output) : model.embed;
        if (load_layers) {
            model.layers.reserve(config.num_hidden_layers);
            for (int index = 0; index < config.num_hidden_layers; ++index) {
                const int device = placement.device_for_layer(index);
                const bool cpu = index < cpu_layers;
                MfqCudaGuard layer_guard(device);
                const auto& type = model.metadata.layer_types.at(index);
                std::cerr << "loading layer " << index << ' ' << type << ' ' << (cpu ? "CPU" : "CUDA") << std::endl;
                model.layers.push_back(load_layer(weights.for_layer(device, cpu), config, index, type));
            }
        }
    } else {
        if (load_layers) throw std::invalid_argument("streamed model setup requires deferred blocks");
#ifdef TYLOINFER_NATIVE_CUDA
        model.output_norm = weights.tensor("model.output_norm.weight");
#else
        model.output_norm = weights.tensor("model.output_norm.weight").dense();
#endif
    }
}

#ifdef TYLOINFER_NATIVE_CUDA
tyloi::models::Info Model::info() const {
    auto result = tyloi::models::model_info(metadata);
    result.separate_projections = true;
    return result;
}
MfqModelCapabilities Model::capabilities() const {
    return {graph.architecture, graph.has_component("text"), false, false,
            false, false, false, false, "model-graph"};
}
tyloi::engine::Metrics Model::metrics() const {
    return {{"kv_memory_supported", 0.0}, {"kv_quantization_supported", 0.0},
            {"kv_quantization_bits", 0.0}, {"vision_declared", double(graph.has_component("vision"))},
            {"vision_supported", 0.0}, {"vision_available", 0.0},
            {"mtp_declared", double(graph.has_component("predictor"))},
            {"mtp_supported", 0.0}, {"mtp_available", 0.0}};
}
std::unique_ptr<tyloi::models::State> Model::make_state() const {
    return std::make_unique<State>(*this);
}
std::vector<std::pair<tyloi::Device, std::size_t>> State::growth(
        std::int64_t tokens, const tyloi::models::ForwardOptions& options) const {
    if (batch <= 0 || tokens <= 0 || cache_pos < 0)
        throw std::invalid_argument("MiniCPM text KV planning requires positive batch and tokens and a nonnegative position");
    if (cache_pos > std::numeric_limits<std::int64_t>::max() - tokens)
        throw std::overflow_error("MiniCPM text KV planning position overflow");
    const auto end = cache_pos + tokens;
    std::vector<std::pair<tyloi::Device, std::size_t>> result;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        if (layers[index].defined()) continue;
        const auto& layout = layout_[index];
        const auto capacity = layout.sliding ? layout.capacity :
            (options.kv_capacity > 0 ? std::max(end, options.kv_capacity) : layout.capacity);
        const auto bytes = tyloi::state::kv_bytes(batch, layout.heads, capacity, layout.head_dim,
            layout.sliding, tyloi::kBFloat16, layout.quantization);
        const tyloi::Device device(layout.cpu ? tyloi::kCPU : tyloi::kCUDA, layout.cpu ? -1 : layout.device);
        auto entry = std::find_if(result.begin(), result.end(),
                                  [device](const auto& item) { return item.first == device; });
        if (entry == result.end()) result.emplace_back(device, bytes);
        else {
            if (entry->second > std::numeric_limits<std::size_t>::max() - bytes)
                throw std::overflow_error("MiniCPM text KV planning byte count overflow");
            entry->second += bytes;
        }
    }
    return result;
}
#endif

State::State(const Model& model) : layers(model.layers.size()) {
    for (const auto& layer : model.layers) {
        const auto& a = layer.attention;
        layout_.push_back({layer.device, layer.cpu, a.sliding, a.kv_heads, a.attention_head_dim,
                           a.sliding ? a.attention_window : a.max_position_embeddings, a.kv_quantization});
    }
}
void State::preallocate() {
    if (!::tyloi::cuda::options::kv_vram_preallocated()) return;
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& l = layout_[i];
        if (l.cpu || layers[i].defined()) continue;
        MfqCudaGuard guard(l.device);
        layers[i] = tyloi::state::allocate_kv(
            1, l.heads, l.capacity, l.head_dim, l.sliding,
            ::tyloi::Device(::tyloi::kCUDA, l.device), ::tyloi::kBFloat16,
            l.quantization);
    }
}
void State::reset(int64_t next_batch) {
    cache_pos = decode_position_delta = 0;
    batch = next_batch;
    for (size_t i = 0; i < layers.size(); ++i)
        if (layers[i].defined() && layers[i].batch != batch) {
            MfqCudaGuard guard(layout_[i].device);
            layers[i] = {};
        }
}
std::vector<::tyloi::Tensor> State::storage() const {
    std::vector<::tyloi::Tensor> result;
    for (const auto& cache : layers) {
        if (cache.paged()) {
            if (cache.key_chunk_ptrs.defined()) result.push_back(cache.key_chunk_ptrs);
            if (cache.value_chunk_ptrs.defined()) result.push_back(cache.value_chunk_ptrs);
            if (cache.page_table.defined()) result.push_back(cache.page_table);
        } else {
            if (cache.physical_key.defined()) result.push_back(cache.physical_key);
            if (cache.physical_value.defined()) result.push_back(cache.physical_value);
        }
    }
    return result;
}
TextSessionStateKind State::text_session_state_kind() const {
    return layers.empty() ? TextSessionStateKind::Unsupported : TextSessionStateKind::FullAttention;
}
bool State::supports_text_session_state() const { return !layers.empty(); }
bool State::supports_paged_text_session_state() const {
    return !layers.empty() && std::none_of(layout_.begin(), layout_.end(), [](const auto& l) { return l.sliding; });
}
TextSessionState State::capture_text_session_state(const std::vector<int64_t>& tokens) const {
    if (layers.empty()) throw std::runtime_error("text session state is unsupported by this block layout");
    if (cache_pos <= 0 || static_cast<size_t>(cache_pos) != tokens.size())
        throw std::runtime_error("text session token count does not match the model cache");
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = cache_pos;
    state.decode_position_delta = decode_position_delta;
    state.payload = std::vector<KvSnapshot>{};
    auto& saved = std::get<std::vector<KvSnapshot>>(state.payload);
    saved.reserve(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        MfqCudaGuard guard(layout_[i].device);
        saved.push_back(capture_kv(layers[i], cache_pos, state.bytes));
    }
    return state;
}
void State::restore_text_session_state(const TextSessionState& state) {
    const auto* saved = std::get_if<std::vector<KvSnapshot>>(&state.payload);
    if (layers.empty() || !saved || state.cache_pos <= 0 ||
        static_cast<size_t>(state.cache_pos) != state.tokens.size() || saved->size() != layers.size())
        throw CudaSessionStateError("full-attention text session state is incompatible");
    for (size_t i = 0; i < layers.size(); ++i) {
        MfqCudaGuard guard(layout_[i].device);
        restore_kv(layers[i], (*saved)[i], layout_[i].head_dim, ::tyloi::kBFloat16, layout_[i].quantization);
    }
    cache_pos = state.cache_pos;
    decode_position_delta = state.decode_position_delta;
    batch = 1;
}

#ifdef TYLOINFER_NATIVE_CUDA
namespace {
Tensor token_ids(int primary, const Tensor& ids) {
    tyloi::models::require_model(ids.defined() && (ids.dim() == 1 || ids.dim() == 2),
        "token IDs must have shape [tokens] or [batch,tokens]");
    auto shaped = ids.dim() == 1 ? tyloi::ops::reshape(ids, {1, ids.size(0)}) : ids;
    tyloi::models::require_model(shaped.size(0) > 0 && shaped.size(1) > 0, "token IDs must not be empty");
    return tyloi::ops::contiguous(tyloi::ops::copy(tyloi::ops::cast(shaped, tyloi::kInt64),
        tyloi::Device(tyloi::kCUDA, primary)));
}
}
Tensor Model::embedding(const Tensor& ids, int primary) const {
    auto normalized = token_ids(primary, ids);
    auto result = tyloi::ops::measure("model.embed", normalized.device(), [&] {
        return tyloi::ops::embedding(normalized, embed);
    });
    return tyloi::ops::contiguous(tyloi::ops::cast(result, tyloi::kBFloat16));
}
Tensor Model::normalize(const Tensor& hidden, int64_t batch, int64_t tokens) const {
    return tyloi::ops::measure("model.output_norm", hidden.device(), [&] {
        return tyloi::ops::rms_norm(tyloi::ops::reshape(hidden, {batch, tokens, metadata.hidden_size}),
            output_norm, metadata.rms_norm_eps, metadata.norm_weight_offset, tyloi::kBFloat16);
    });
}
Tensor Model::softcap(const Tensor& value) const { return value; }
Tensor Model::logits(const Tensor& hidden, bool) const {
    auto result = tyloi::ops::measure("model.lm_head", hidden.device(), [&] {
        return tyloi::ops::project(hidden, lm_head);
    });
    return tyloi::ops::contiguous(tyloi::ops::cast(result, tyloi::kBFloat16));
}
tyloi::StepSequence<tyloi::models::Prepared> Model::prepare(
        const std::vector<std::int64_t>&, const MfqMultimodalInput&, tyloi::models::State&) const {
    throw std::invalid_argument("model has no multimodal component");
}
Tensor Model::forward(tyloi::models::Input input, tyloi::models::State& state_value,
                      const tyloi::models::ForwardOptions& mode) const {
    if (input.suffix) throw std::invalid_argument("MiniCPM text does not support speculative suffix forwarding");
    auto& state = dynamic_cast<State&>(state_value);
    auto ids = token_ids(mode.primary, input.ids);
    auto hidden = input.embeddings ? input.embeddings : embedding(ids, mode.primary);
    const auto B = ids.size(0), T = ids.size(1);
    tyloi::models::require_model(hidden.dim() == 3 && hidden.size(0) == B && hidden.size(1) == T &&
        hidden.size(2) == metadata.hidden_size, "inputs_embeds shape must match [batch,tokens,hidden_size]");
    tyloi::models::require_model(input.confirmed == 0, "unsupported speculative backbone geometry");
    if (state.cache_pos == 0) state.reset(B);
    const tyloi::Device primary(tyloi::kCUDA, mode.primary);
    const auto copy_position = [&](const Tensor& value) {
        return tyloi::ops::contiguous(tyloi::ops::copy(tyloi::ops::cast(value, tyloi::kInt64), primary));
    };
    auto physical = input.cache_positions ? copy_position(input.cache_positions)
        : tyloi::ops::arange(state.cache_pos, state.cache_pos + T, primary);
    auto positions = input.positions ? copy_position(input.positions)
        : (state.decode_position_delta ? tyloi::ops::add(physical, state.decode_position_delta) : physical);
    tyloi::models::require_model((physical.dim() == 1 && physical.numel() == T) ||
        (physical.dim() == 2 && physical.size(0) == B && physical.size(1) == T),
        "cache_positions must have shape [tokens] or [batch,tokens]");
    tyloi::models::require_model((positions.dim() == 1 && positions.size(0) == T) ||
        (positions.dim() == 2 && positions.size(0) == B && positions.size(1) == T),
        "position_ids must have shape [tokens] or [batch,tokens]");
    hidden = tyloi::ops::contiguous(tyloi::ops::cast(tyloi::ops::copy(hidden, primary), tyloi::kBFloat16));
    const auto trace = [&] {
        if (input.trace) input.trace->push_back(tyloi::ops::clone(
            tyloi::ops::cast(tyloi::ops::copy(hidden, primary), tyloi::kFloat32)));
    };
    trace();
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        const auto& attention = layer.attention;
        const tyloi::Device device(layer.cpu ? tyloi::kCPU : tyloi::kCUDA, layer.cpu ? -1 : layer.device);
        const auto local = [&](const Tensor& value) {
            return tyloi::ops::contiguous(tyloi::ops::copy(value, device));
        };
        hidden = local(hidden);
        AttentionStep step{local(positions), {}, local(physical), state.cache_pos,
                           input.kv_length, input.attention_parts};
        if (input.lengths) step.sequence_lengths = local(input.lengths);
        auto& cache = state.layers[i];
        const auto capacity = attention.sliding ? attention.attention_window :
            (mode.kv_capacity > 0 ? std::max(state.cache_pos + T, mode.kv_capacity)
                                  : attention.max_position_embeddings);
        if (!cache.defined())
            cache = tyloi::state::allocate_kv(B, attention.kv_heads, capacity, attention.attention_head_dim,
                attention.sliding, device, tyloi::kBFloat16, attention.kv_quantization);
        cache.quantization = attention.kv_quantization;
        const auto* rope = layer.cpu ? &cpu_rope : &this->rope;
        if (!layer.cpu && !device_ropes.empty()) rope = &device_ropes.at(layer.device);
        auto* layer_trace = mode.trace_layer == attention.layer ? mode.trace : nullptr;
        auto output = attention.forward(hidden, step, cache, *rope, layer_trace);
        hidden = tyloi::models::full_ffn(layer.ffn, layer.ffn_norm, output[0], output[1],
            attention.rms_norm_eps, attention.norm_weight_offset,
            attention.official_bf16, attention.attention_output_gate, layer_trace);
        trace();
    }
    if (!input.positions || input.advance) state.cache_pos += T;
    hidden = tyloi::ops::copy(hidden, primary);
    auto finalized = normalize(hidden, B, T);
    if (input.raw_hidden) *input.raw_hidden = std::move(hidden);
    return finalized;
}
#else
namespace {
DenseTensor token_ids(Collectives& copies, int primary, DenseTensor ids) {
    tyloi::models::require_model(ids.dim() == 1 || ids.dim() == 2,
        "token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = ids.unsqueeze(0);
    tyloi::models::require_model(ids.size(0) > 0 && ids.size(1) > 0, "token IDs must not be empty");
    return copies.copy(ids.to(::tyloi::kInt64), primary).contiguous();
}
}

DenseTensor embed(const Model& model, Executor& executor, Collectives& copies, int primary,
             DenseTensor ids, const Executor::Mode& mode) {
    auto normalized = token_ids(copies, primary, std::move(ids));
    return executor.profiler().measure("model.embed", [&] {
        return ::tyloi::cuda::quant_embedding_lookup(model.embed, std::move(normalized));
    }).to(::tyloi::kBFloat16).contiguous();
}
DenseTensor normalize(const Model& model, Executor& executor, DenseTensor hidden, int64_t batch, int64_t tokens) {
    return executor.profiler().measure("model.output_norm", [&] {
        return qwen_rms_norm_bf16(executor.attention_options().fused_norm,
            hidden.reshape({batch * tokens, model.metadata.hidden_size}),
            model.output_norm, model.metadata.rms_norm_eps, model.metadata.norm_weight_offset)
            .reshape({batch, tokens, model.metadata.hidden_size});
    });
}
DenseTensor logits(const Model& model, Executor& executor, DenseTensor hidden, bool,
              const Executor::Mode& mode) {
    return ::tyloi::cuda::logits(model.lm_head, std::move(hidden),
        executor.projector(mode.ffn), executor.profiler(), 0.0)
        .to(::tyloi::kBFloat16).contiguous();
}
DenseTensor forward(const Model& model, State& state, Executor& executor, Collectives& copies,
               const Executor::Mode& run_mode, tyloi::models::Input input) {
    auto mode = run_mode;
    mode.ffn.independent = true;
    MfqCudaGuard guard(mode.primary);
    auto ids = token_ids(copies, mode.primary, std::move(input.ids).dense());
    auto hidden = input.embeddings ? *input.embeddings
        : embed(model, executor, copies, mode.primary, ids, mode);
    const auto B = ids.size(0), T = ids.size(1);
    tyloi::models::require_model(hidden.dim() == 3 && hidden.size(0) == B && hidden.size(1) == T &&
        hidden.size(2) == model.metadata.hidden_size,
        "inputs_embeds shape must match [batch,tokens,hidden_size]");
    tyloi::models::require_model(input.confirmed == 0, "unsupported speculative backbone geometry");
    if (state.cache_pos == 0) state.reset(B);
    const auto options = ::tyloi::TensorOptions().dtype(::tyloi::kInt64).device(::tyloi::Device(::tyloi::kCUDA, mode.primary));
    auto physical = input.cache_positions
        ? copies.copy(
            input.cache_positions->to(::tyloi::kInt64),
            mode.primary).contiguous()
        : ::tyloi::cuda::arange(state.cache_pos, state.cache_pos + T, options);
    auto positions = input.positions
        ? copies.copy(
            input.positions->to(::tyloi::kInt64),
            mode.primary).contiguous()
        : (state.decode_position_delta ? physical + state.decode_position_delta : physical);
    tyloi::models::require_model((physical.dim() == 1 && physical.numel() == T) ||
        (physical.dim() == 2 && physical.size(0) == B && physical.size(1) == T),
        "cache_positions must have shape [tokens] or [batch,tokens]");
    tyloi::models::require_model((positions.dim() == 1 && positions.size(0) == T) ||
        (positions.dim() == 2 && positions.size(0) == B && positions.size(1) == T),
        "position_ids must have shape [tokens] or [batch,tokens]");
    hidden = copies.copy(std::move(hidden), mode.primary).to(::tyloi::kBFloat16).contiguous();
    const auto trace = [&] {
        if (input.trace) input.trace->push_back(copies.copy(hidden, mode.primary).to(::tyloi::kFloat32).clone());
    };
    trace();
    for (size_t i = 0; i < model.layers.size(); ++i) {
        const auto& layer = model.layers[i];
        const auto& attention = layer.attention;
        MfqCudaGuard layer_guard(layer.device);
        const auto local = [&](DenseTensor value) {
            return layer.cpu ? value.to(::tyloi::kCPU).contiguous() : copies.copy(value, layer.device).contiguous();
        };
        hidden = local(std::move(hidden));
        AttentionStep step{local(positions), {}, local(physical), state.cache_pos,
                           input.kv_length, input.attention_parts};
        if (input.lengths) step.sequence_lengths = local(*input.lengths);
        auto& cache = state.layers[i];
        const auto capacity = attention.sliding ? attention.attention_window :
            (mode.kv_capacity > 0 ? std::max(state.cache_pos + T, mode.kv_capacity)
                                  : attention.max_position_embeddings);
        if (!cache.defined())
            cache = tyloi::state::allocate_kv(
                B, attention.kv_heads, capacity, attention.attention_head_dim,
                attention.sliding, hidden.device(), ::tyloi::kBFloat16,
                attention.kv_quantization);
        cache.quantization = attention.kv_quantization;
        const auto* rope = layer.cpu ? &model.cpu_rope : &model.rope;
        if (!layer.cpu && !model.device_ropes.empty()) rope = &model.device_ropes.at(layer.device);
        auto* layer_trace = mode.trace_layer == attention.layer ? mode.trace : nullptr;
        auto output = attention.forward(executor.projector(mode.ffn), executor.profiler(),
            executor.attention_options(), std::move(hidden), step, cache,
            executor.attention_scratch(2 * i), *rope, layer_trace);
        hidden = ::tyloi::models::full_ffn(
            layer.ffn, layer.ffn_norm, std::move(output[0]), std::move(output[1]),
            attention.rms_norm_eps, attention.norm_weight_offset,
            attention.official_bf16, attention.attention_output_gate, layer_trace).dense();
        trace();
    }
    if (!input.positions || input.advance) state.cache_pos += T;
    hidden = copies.copy(std::move(hidden), mode.primary);
    auto finalized = normalize(model, executor, hidden, B, T);
    if (input.raw_hidden) *input.raw_hidden = std::move(hidden);
    return finalized;
}
#endif

} // namespace tyloi::cuda::minicpmo45::text

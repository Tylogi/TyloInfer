#include "engine/models/minicpmo45/text/load.h"
#include "engine/models/minicpmo45/text/state.h"
#include "engine/models/minicpmo45/vision.h"
#include "engine/models/minicpmo45/audio.h"
#include "engine/models/minicpmo45/tts.h"
#include "engine/models/common/ffn.h"
#include "engine/models/common/mlp.h"
#include "engine/models/common/load.h"
#include "engine/weights/loader.h"
#include "engine/models/common/check.h"
#include "engine/models/session/state.h"
#include "engine/models/source/model_graph.h"
#include "engine/requests/media.h"
#include "engine/backends/cuda/executor/options.h"

#include <algorithm>
#include <iostream>
#include <limits>

namespace tyloi::cuda::minicpmo45::text {

Model read(std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition, int64_t context) {
    ::tyloi::models::validate_model_source(*source);
    Model model;
    model.source = std::move(source);
    model.graph = tyloi::models::graph(*model.source);
    tyloi::models::require_model(model.graph.backbone == "minicpmo45",
        "model backbone does not match the requested causal model");
    const auto* vision = model.graph.component("vision");
    const auto* audio = model.graph.component("audio_input");
    const auto* tts = model.graph.component("audio_output");
    if (vision || audio || tts) {
        if (!vision || !audio || !tts ||
                vision->implementation != "minicpmo45_vision" || vision->tensor_root != "vision" ||
                vision->input_contract != "minicpmo45.v1" ||
                vision->position_policy != "minicpmo45_positions" ||
                audio->implementation != "minicpmo45_audio" || audio->tensor_root != "audio" ||
                tts->implementation != "minicpmo45_tts" || tts->tensor_root != "tts")
            throw std::runtime_error("MiniCPM-o requires its vision, audio and TTS components together");
    }
    if (const auto* duplex = model.graph.component("duplex")) {
        if (!vision || !audio || !tts || duplex->implementation != "minicpmo45_duplex" ||
                duplex->tensor_root != "runtime")
            throw std::runtime_error("MiniCPM-o duplex requires its media components");
    }
    if (model.graph.has_component("predictor"))
        throw std::runtime_error("MiniCPM-o does not support a predictor component");
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
    layer.device = weights.placement.target.is_cuda()
        ? weights.placement.target.index : mfq_current_cuda_device();
    layer.cpu = weights.placement.target.is_cpu();
    layer.attention.official_bf16 = true;
    TransformerWeightLoader loader{weights};
    loader.preserve_projection_boundaries = true;
    const auto prefix = "model.block." + std::to_string(index) + ".";
    tyloi::models::load_attention_mixer(layer.attention, loader, config, index, prefix, 0.0);
    layer.ffn_norm = weights.tensor(prefix + "mlp.norm.weight");
    layer.ffn = tyloi::models::load_dense_ffn(loader, config, prefix + "mlp.");
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
    MfqCudaGuard guard(placement.primary_device());
    const auto make_rope = [&](::tyloi::Device device) {
        RopeCache rope(config.max_position_embeddings, config.rotary_dim, config.rope_base,
                         0, -1, device, true, yarn);

        return rope.view();
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
        model.output_norm = weights.tensor("model.output_norm.weight");
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
                model.layers.push_back(load_layer(weights.for_layer(cpu ? ::tyloi::Device(::tyloi::kCPU) : ::tyloi::Device(::tyloi::kCUDA, device)), config, index, type));
            }
        }
    } else {
        if (load_layers) throw std::invalid_argument("streamed model setup requires deferred blocks");
        model.output_norm = weights.tensor("model.output_norm.weight");
    }
}

void load_media(Model& model, const tyloi::weights::Loader& weights) {
    model.vision = std::make_shared<MiniCPMO45VisionEncoder>(MiniCPMO45VisionEncoder::load(weights));
    model.resampler = std::make_shared<MiniCPMO45Resampler>(MiniCPMO45Resampler::load(weights));
    model.audio = std::make_shared<MiniCPMO45AudioEncoder>(MiniCPMO45AudioEncoder::load(weights));
    model.tts = std::make_shared<Tts>(Tts::load(weights));
}

void load(std::unique_ptr<tyloi::models::CausalModel>& owner,
        std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
        const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
        const tyloi::models::LoadOptions& options) {
    owner = std::make_unique<Model>(read(std::move(source), definition, options.context));
    auto* model = static_cast<Model*>(owner.get());
    load(*model, weights, placement, options.gpu_layers, options.expert_parallel, true, true, options.yarn);
    model->primary = placement.primary_device();
    if (options.optional_components && model->graph.has_component("vision")) {
        load_media(*model, weights);
    }
}

tyloi::models::Info Model::info() const {
    auto result = tyloi::models::model_info(metadata);
    result.separate_projections = true;
    return result;
}
MfqModelCapabilities Model::capabilities() const {
    const bool media = vision && resampler && audio;
    return {graph.architecture, graph.has_component("text"), media, media,
            media, media && tts, media && tts && graph.has_component("duplex"), false, "model"};
}
tyloi::engine::Metrics Model::metrics() const {
    return {{"kv_memory_supported", 0.0}, {"kv_quantization_supported", 0.0},
            {"kv_quantization_bits", 0.0}, {"vision_declared", double(graph.has_component("vision"))},
            {"vision_supported", double(graph.has_component("vision"))}, {"vision_available", double(bool(vision && resampler))},
            {"mtp_declared", double(graph.has_component("predictor"))},
            {"mtp_supported", 0.0}, {"mtp_available", 0.0}};
}
std::unique_ptr<tyloi::models::State> Model::make_state() const {
    return std::make_unique<State>(*this);
}
std::vector<std::pair<tyloi::Device, std::size_t>> State::preallocate_growth() const {
    std::vector<std::pair<tyloi::Device, std::size_t>> result;
    if (!::tyloi::cuda::options::kv_vram_preallocated()) return result;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto& l = layout_[i];
        if (l.cpu || layers[i].defined()) continue;
        const tyloi::Device device(tyloi::kCUDA, l.device);
        const auto bytes = tyloi::state::kv_bytes(1, l.heads, l.capacity, l.head_dim,
            l.sliding, tyloi::kBFloat16, l.quantization);
        auto entry = std::find_if(result.begin(), result.end(), [device](const auto& v) { return v.first == device; });
        if (entry == result.end()) result.emplace_back(device, bytes);
        else {
            if (entry->second > std::numeric_limits<std::size_t>::max() - bytes)
                throw std::overflow_error("MiniCPM text KV planning byte count overflow");
            entry->second += bytes;
        }
    }
    return result;
}
std::vector<std::pair<tyloi::Device, std::size_t>> State::growth(
        const tyloi::models::Input& input, const tyloi::models::ForwardOptions& options) const {
    const auto [next_batch, tokens] = input.shape();
    if (batch <= 0 || next_batch != batch || cache_pos < 0 || input.confirmed != 0 || input.suffix)
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
        const tyloi::Device device(layout.cpu ? tyloi::kCPU : tyloi::kCUDA, layout.cpu ? 0 : layout.device);
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
    attention_mask = {};
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
std::unique_ptr<tyloi::models::Restore> State::prepare_restore(
        const TextSessionState& state, const tyloi::models::Mtp* predictor_model) {
    if (predictor_model) throw CudaSessionStateError("MiniCPM text does not support a predictor");
    const auto* saved = std::get_if<std::vector<KvSnapshot>>(&state.payload);
    if (layers.empty() || !saved || state.cache_pos <= 0 ||
        static_cast<size_t>(state.cache_pos) != state.tokens.size() || saved->size() != layers.size())
        throw CudaSessionStateError("full-attention text session state is incompatible");
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& l = layout_[i];
        const auto& kv = (*saved)[i];
        validate_kv(kv);
        const auto expected = tyloi::state::kv_layout(1, l.heads, kv.capacity, l.head_dim,
            l.sliding, tyloi::kBFloat16, l.quantization);
        const auto device = tyloi::Device(l.cpu ? tyloi::kCPU : tyloi::kCUDA, l.cpu ? 0 : l.device);
        const auto dtype = expected.storage_dtype;
        if (kv.target_device() != device || kv.ring != l.sliding ||
            kv.k.scalar_type() != dtype || kv.v.scalar_type() != dtype || kv.k.size(1) != l.heads ||
            kv.k.size(3) != expected.key_width || kv.v.size(3) != expected.value_width ||
            kv.k.size(2) != (kv.ring ? kv.capacity : state.cache_pos) ||
            (!kv.ring && state.cache_pos > l.capacity))
            throw CudaSessionStateError("MiniCPM text session cache geometry changed");
    }
    struct Candidate final : tyloi::models::Restore {
        State& owner;
        TextSessionState snapshot;
        std::vector<tyloi::state::KvView> prepared;
        Candidate(State& value, const TextSessionState& source)
            : owner(value), snapshot(source), prepared(value.layers.size()) {}
        std::vector<std::pair<tyloi::Device, std::size_t>> growth() const override {
            std::vector<std::pair<tyloi::Device, std::size_t>> result;
            for (const auto& kv : std::get<std::vector<KvSnapshot>>(snapshot.payload)) {
                add_kv_restore_growth(result, kv);
            }
            return result;
        }
        void stage() override {
            const auto& values = std::get<std::vector<KvSnapshot>>(snapshot.payload);
            for (size_t i = 0; i < prepared.size(); ++i) {
                const auto& l = owner.layout_[i];
                MfqCudaGuard guard(l.device);
                stage_kv(prepared[i], values[i], l.head_dim, tyloi::kBFloat16, l.quantization);
            }
        }
        std::vector<::tyloi::Tensor> storage() const override {
            std::vector<::tyloi::Tensor> result;
            for (const auto& kv : prepared) {
                if (kv.physical_key) result.push_back(kv.physical_key);
                if (kv.physical_value) result.push_back(kv.physical_value);
            }
            return result;
        }
        void publish() noexcept override {
            owner.layers.swap(prepared);
            owner.cache_pos = snapshot.cache_pos;
            owner.decode_position_delta = snapshot.decode_position_delta;
            owner.batch = 1;
        }
    };
    return std::make_unique<Candidate>(*this, state);
}

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
Tensor Model::logits(const Tensor& hidden, bool, tyloi::models::LogitPrecision precision) const {
    auto projection_input = precision == tyloi::models::LogitPrecision::model
        ? tyloi::ops::contiguous(tyloi::ops::cast(hidden, tyloi::kBFloat16)) : hidden;
    auto result = tyloi::ops::measure("model.lm_head", hidden.device(), [&] {
        return tyloi::ops::project(projection_input, lm_head);
    });
    return tyloi::ops::contiguous(tyloi::ops::cast(result, tyloi::kBFloat16));
}
tyloi::StepSequence<tyloi::models::Prepared> Model::prepare(
        const std::vector<std::int64_t>& tokens, const MfqMultimodalInput& media,
        tyloi::models::State& state) const {
    if (!dynamic_cast<State*>(&state)) throw std::invalid_argument("MiniCPM text state has the wrong type");
    if (!vision || !resampler || !audio) throw std::invalid_argument("model has no multimodal component");
    const auto gpu = tyloi::TensorOptions().device(tyloi::Device(tyloi::kCUDA, primary));
    auto ids = tyloi::cuda::tensor(tokens, gpu.dtype(tyloi::kInt64)).reshape({1, static_cast<int64_t>(tokens.size())});
    auto embeddings = embedding(Tensor(ids), primary);
    std::vector<Tensor> retained{Tensor(ids), embeddings};
    std::optional<MiniCPMO45AudioEncoder::State> audio_state;
    tyloi::StepSequence<Tensor> vision_steps, audio_steps;
    const auto scatter = [&](const Tensor& encoded, const tyloi::models::minicpmo45::MediaBound& bound, int64_t length) {
        tyloi::models::require_model(bound.batch == 0 && bound.source >= 0 && bound.source < encoded.size(0) &&
            bound.begin >= 0 && bound.end <= ids.size(1) && bound.end - bound.begin == length,
            "MiniCPM-o media bound does not match encoded length");
        auto& target = embeddings.dense();
        target.index({0, Slice(bound.begin, bound.end), Slice()}).copy_(
            encoded.dense().index({bound.source, Slice(0, length), Slice()}).to(target.scalar_type()));
    };
    const auto images = tyloi::models::minicpmo45::media_bounds(media.image_bounds);
    if (!images.empty()) {
        auto pixels = tyloi::cuda::tensor(media.pixel_values, gpu.dtype(tyloi::kFloat32)).reshape(media.pixel_shape);
        auto mask = tyloi::cuda::tensor(media.patch_mask).reshape(media.patch_mask_shape).to(tyloi::kBool);
        auto sizes = tyloi::cuda::tensor(media.target_sizes).reshape(media.target_sizes_shape);
        retained.insert(retained.end(), {Tensor(pixels), Tensor(mask), Tensor(sizes)});
        vision_steps = vision->forward_steps(Tensor(pixels), Tensor(mask), Tensor(sizes));
        Tensor states;
        while (auto step = vision_steps.next()) {
            if (step.value) states = std::move(*step.value);
            else co_yield step.state;
        }
        auto encoded = resampler->forward(states, Tensor(sizes));
        retained.push_back(states);
        retained.push_back(encoded);
        for (const auto& bound : images) scatter(encoded, bound, encoded.size(1));
        co_yield tyloi::StepState::advanced;
    }
    const auto audios = tyloi::models::minicpmo45::media_bounds(media.audio_bounds);
    if (!audios.empty()) {
        auto features = tyloi::cuda::tensor(media.audio_features, gpu.dtype(tyloi::kFloat32)).reshape(media.audio_features_shape);
        auto lengths = tyloi::cuda::tensor(media.audio_lengths);
        retained.insert(retained.end(), {Tensor(features), Tensor(lengths)});
        audio_state.emplace(*audio);
        audio_steps = audio->forward_steps(Tensor(features), Tensor(lengths), *audio_state, false);
        Tensor encoded;
        while (auto step = audio_steps.next()) {
            if (step.value) encoded = std::move(*step.value);
            else co_yield step.state;
        }
        retained.push_back(encoded);
        const auto pooled = MiniCPMO45AudioEncoder::pooled_lengths(lengths);
        for (const auto& bound : audios) {
            tyloi::models::require_model(bound.source >= 0 && static_cast<size_t>(bound.source) < pooled.size(),
                "MiniCPM-o audio bound source is out of range");
            scatter(encoded, bound, pooled[bound.source]);
        }
        co_yield tyloi::StepState::advanced;
    }
    tyloi::models::Prepared prepared;
    prepared.token_ids = tokens;
    prepared.embeddings = std::move(embeddings);
    prepared.positions = tyloi::ops::arange(0, static_cast<int64_t>(tokens.size()), tyloi::Device(tyloi::kCUDA, primary));
    co_yield std::move(prepared);
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
    if (state.cache_pos == 0) {
        auto mask = state.attention_mask;
        state.reset(B);
        state.attention_mask = std::move(mask);
    }
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
        ::tyloi::models::AttentionStep step{local(positions), {}, local(physical), state.cache_pos,
                           input.kv_length, input.attention_parts};
        if (input.lengths) step.sequence_lengths = local(input.lengths);
        if (state.attention_mask) step.attention_mask = local(state.attention_mask);
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

} // namespace tyloi::cuda::minicpmo45::text

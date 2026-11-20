#include "engine/models/load.h"
#include "engine/models/glm5_next/components.h"
#include "engine/backends/cuda/ops/ffn/experts/launch.h"
#include "model.h"
#include "mtp.h"
#include "load.h"
#include "forward.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/scope.h"
#include <stdexcept>
#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <variant>
#include "engine/models/common/load.h"
#include "engine/weights/loader.h"
#include "engine/models/session/paged.h"
#include "engine/ops/ops.h"

namespace tyloi::cuda::glm5_next {
using Config = tyloi::models::glm5_next::Config;
namespace {
DenseTensor dense_ffn(const tyloi::models::Mlp& weights, const DenseTensor& source) {
    DenseTensor hidden;
    {
        auto g = ::tyloi::ops::project(source, weights.gate_up[0]).dense();
        auto u = ::tyloi::ops::project(source, weights.gate_up[1]).dense();
        const auto limit = weights.swiglu_limit;
        // GLM retains its unconditional clamp, including a zero limit.
        g = ::tyloi::cuda::clamp_max(g, limit);
        u = ::tyloi::cuda::clamp(u, -limit, limit);
        hidden = (g * ::tyloi::cuda::sigmoid(g)) * u;
    }
    return ::tyloi::ops::project(hidden, weights.down).dense();
}

DenseTensor feed_forward(const tyloi::models::Ffn& weights, const DenseTensor& x) {
    if (const auto* dense = std::get_if<tyloi::models::Mlp>(&weights))
        return dense_ffn(*dense, x);
    const auto& moe = std::get<tyloi::models::Moe>(weights);
    const auto hidden = moe.down.size(1), intermediate = moe.down.size(2);
    const auto limit = moe.swiglu_limit;
    auto source = x.reshape({-1, hidden}).to(::tyloi::kFloat16);
    auto logits = ::tyloi::ops::project(source.to(::tyloi::kFloat32), moe.router).dense()
        .to(::tyloi::kFloat32).contiguous();
    auto selected = ::tyloi::ops::topk(logits, moe.top_k, moe.routing, moe.router_bias);
    auto shared_output = dense_ffn(moe.shared, source);
    DenseTensor routed;
    {
        DenseTensor activation;
        {
            auto gu = ::tyloi::ops::project(source, moe.gate_up[0], selected.ids).dense();
            if (moe.gate_up.size() == 2) {
                auto routed_up = ::tyloi::ops::project(source, moe.gate_up[1], selected.ids).dense();
                gu = ::tyloi::cuda::cat({gu, routed_up}, -1);
            }
            auto g = ::tyloi::cuda::clamp_max(gu.narrow(-1, 0, intermediate), limit);
            auto u = ::tyloi::cuda::clamp(gu.narrow(-1, intermediate, intermediate), -limit, limit);
            activation = (g * ::tyloi::cuda::sigmoid(g)) * u;
        }
        auto pairs = ::tyloi::ops::project(activation, moe.down, selected.ids).dense();
        activation = {};
        auto reduced = ::tyloi::cuda::zeros({source.size(0), hidden},
                                            source.options().dtype(::tyloi::kFloat32));
        for (int64_t r = 0; r < moe.top_k; ++r)
            reduced = reduced + pairs.select(1, r).to(::tyloi::kFloat32) *
                                selected.weights.dense().select(1, r).unsqueeze(-1);
        routed = reduced.to(pairs.scalar_type());
    }
    return (routed + shared_output).reshape(x.sizes());
}
} // namespace

struct LayerLoader : ::tyloi::weights::Loader {
    using KdaWeights = glm5_next::KdaWeights;
    using MlaWeights = glm5_next::MlaWeights;

    static Mhc mhc(Parameter function, Parameter base, Parameter scale) {
        return {std::move(function), std::move(base), std::move(scale)};
    }
    static Parameter concat(const std::vector<Parameter> &parts) {
        std::vector<DenseTensor> dense;
        dense.reserve(parts.size());
        for (const auto& part : parts) dense.push_back(part.dense());
        return std::make_shared<DenseTensor>(::tyloi::cuda::cat(dense, 0));
    }
    static auto kda(KdaWeights weights, const Config &c) {
        return std::make_unique<Kda>(std::move(weights), c.kda_heads, c.kda_width, c.kda_kernel,
                                     c.lower_bound, c.eps);
    }
    static auto mla(MlaWeights weights, const Config &c) {
        MlaConfig geometry{c.heads, c.nope, c.latent, c.value_width, c.index_heads, c.index_width,
                           c.pool, c.budget, c.maximum, c.tail, c.eps};
        return std::make_unique<SparseMla>(std::move(weights), geometry);
    }
};

static DenseTensor execute(const Layer& layer, LayerState& activity,
                      const DenseTensor &x, int64_t position, int64_t confirmed = 0) {
    const auto& [config, attention_hc, ffn_hc, attention_norm, ffn_norm, ffn, kda, mla, device] = layer;
    if (mla)
        {
            if (!(std::get<MlaState>(activity).position() == position))
                throw std::runtime_error("GLM MLA/model cache positions diverged");
        }
    auto attention_mix = attention_hc.pre(x, config);
    auto normalized = rms_norm(attention_mix[2], attention_norm.dense(), config.eps);
    auto branch = kda ? kda->forward(std::get<KdaState>(activity), std::move(normalized), true, confirmed)
                      : mla->forward(std::get<MlaState>(activity), std::move(normalized), true);
    auto hidden = mfq_glm5_next::mhc_post(branch, x, attention_mix[0], attention_mix[1]);
    auto ffn_mix = ffn_hc.pre(hidden, config);
    normalized = rms_norm(ffn_mix[2], ffn_norm.dense(), config.eps);
    branch = feed_forward(ffn, normalized);
    return mfq_glm5_next::mhc_post(branch, hidden, ffn_mix[0], ffn_mix[1]);
}

void load_layer(Layer& layer, const ::tyloi::weights::Loader& weights, const Config& config, int index) {
    MfqCudaGuard guard(weights.placement.target.index);
    layer.device = weights.placement.target.index;
    LayerLoader loader{weights};
    loader.role = "glm5_next";
    tyloi::models::glm5_next::load_layer_weights(layer, loader, config, index);
}

Model read(std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
           int64_t context) {
    ::tyloi::models::validate_model_source(*source);
    Model model;
    model.source = std::move(source);
    model.graph = tyloi::models::graph(*model.source);
    if (!(model.graph.backbone == "glm5_next"))
        throw std::runtime_error("model backbone does not match the requested causal model");
    for (const auto& component : model.graph.components)
        if (component.kind != "text" && component.kind != "predictor")
            throw std::runtime_error("GLM5-Next CUDA does not support component: " + component.kind);
    if (const auto* predictor = model.graph.component("predictor"); predictor &&
        (predictor->tensor_root != "predictor" || predictor->implementation != "next_token_prediction"))
        throw std::runtime_error("GLM5-Next CUDA requires a next_token_prediction predictor under the predictor tensor root");
    tyloi::models::glm5_next::load_config(definition, model.config, model.metadata);
    if (!(model.config.layers == model.graph.topology.text_layers))
        throw std::runtime_error("model graph/config text-layer topology mismatch");
    if (context > 0) {
        if (!(context <= model.metadata.max_position_embeddings))
            throw std::runtime_error("--ctx-size exceeds max_position_embeddings");
        model.config.maximum = context;
        model.metadata.max_position_embeddings = context;
    }
    return model;
}

void load(std::unique_ptr<tyloi::models::CausalModel>& owner,
          std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
          const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
          const tyloi::models::LoadOptions& options) {
    if (!options.cpu_offload_layers.empty())
        throw std::invalid_argument("GLM5-Next does not support --cpu-offload-layers");
    owner = std::make_unique<Model>(read(std::move(source), definition, options.context));
    auto& model = static_cast<Model&>(*owner);
    if (!weights.placement.target.is_cuda() || weights.placement.target.index < 0)
        throw std::invalid_argument("GLM5-Next requires a CUDA target");
    if (placement.devices.empty()) placement.devices = {weights.placement.target.index};
    if (placement.primary_device() != weights.placement.target.index)
        throw std::invalid_argument("GLM5-Next weight and layer primary devices differ");
    if (options.expert_parallel && model.config.experts <= 0)
        throw std::invalid_argument("--expert-parallel requires a model with routed experts");
    load(model, weights, placement, options.gpu_layers);
    load_components(model, weights, options.optional_components);
}

void load(Model& model, const ::tyloi::weights::Loader& weights,
          LayerPlacementConfig& placement, int gpu_layers, bool load_layers,
          bool load_input_output) {
    tyloi::models::validate_native_placement(weights.placement.tensor, placement, gpu_layers);
    placement.prepare(model.config.layers);
    MfqCudaGuard guard(placement.primary_device());
    if (load_input_output) {
        const bool separate = !model.config.tied_embeddings && weights.has("model.output.weight");
        model.embed = weights.embedding("model.token_embedding.weight", separate);
        model.output_norm = weights.tensor("model.output_norm.weight");
        model.lm_head = separate ? weights.weight("model.output.weight", TensorParallelAxis::Output)
                                 : model.embed;
        if (load_layers) {
            model.layers.reserve(static_cast<size_t>(model.config.layers));
            for (int index = 0; index < model.config.layers; ++index) {
                model.layers.emplace_back();
                load_layer(model.layers.back(), weights.for_layer(::tyloi::Device(::tyloi::kCUDA, placement.device_for_layer(index))),
                           model.config, index);
            }
        }
    } else {
        if (load_layers) throw std::invalid_argument("streamed model setup requires deferred blocks");
        model.output_norm = weights.tensor("model.output_norm.weight");
    }
}

State::State(const Model& model)
    : maximum_(model.config.maximum), kda_heads_(model.config.kda_heads),
      kda_width_(model.config.kda_width), kda_kernel_(model.config.kda_kernel)
{
    layers.reserve(model.layers.size());
    for (const auto& layer : model.layers) {
        devices.push_back(layer.device);
        if (layer.kda) layers.emplace_back(std::in_place_type<KdaState>);
        else layers.emplace_back(std::in_place_type<MlaState>, model.config.maximum,
                                 model.config.latent, model.config.index_width);
    }
}

void State::reset(int64_t next_batch) {
    cache_pos = decode_position_delta = 0;
    speculative_start = -1;
    speculative_confirmed = 0;
    batch = next_batch;
    for (auto& layer : layers)
        std::visit([](auto& activity) { activity.reset(); }, layer);
}

void State::commit_speculative() {
    if (!(speculative_start >= 0))
        throw std::runtime_error("no speculative transaction");
    for (auto& layer : layers)
        if (auto* kda = std::get_if<KdaState>(&layer)) kda->commit();
    speculative_start = -1;
    speculative_confirmed = 0;
}

TextSessionState State::capture_text_session_state(const std::vector<int64_t>&) const {
    throw std::runtime_error("text session state is unsupported by this model");
}
void State::restore_text_session_state(const TextSessionState&) {
    throw CudaSessionStateError("text session state is unsupported by this model");
}

std::vector<std::pair<tyloi::Device, std::size_t>> State::growth(
        const tyloi::models::Input& input, const tyloi::models::ForwardOptions&) const {
    const auto [input_batch, tokens] = input.shape();
    if (input.suffix || input.positions.defined() || input.cache_positions.defined() || input_batch != batch || input.confirmed < 0 ||
        (input.confirmed > 0 && (input.confirmed >= tokens || batch != 1 || cache_pos <= 0)) ||
        batch <= 0 || tokens <= 0 || maximum_ <= 0 || cache_pos < 0 || cache_pos > maximum_ ||
        tokens > maximum_ - cache_pos || speculative_start >= 0 ||
        kda_heads_ <= 0 || kda_width_ <= 0 || kda_kernel_ <= 1)
        throw std::invalid_argument("invalid GLM Flash-Next ordinary state growth");
    if (tokens > std::numeric_limits<std::int64_t>::max() - (kda_kernel_ - 1))
        throw std::overflow_error("GLM KDA convolution length overflow");
    struct Growth { std::size_t retained = 0, peak = 0; };
    std::map<int, Growth> growth;
    const auto add = [](std::size_t& total, std::size_t value) {
        if (total > std::numeric_limits<std::size_t>::max() - value)
            throw std::overflow_error("GLM state growth byte count overflow");
        total += value;
    };
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (devices[i] < 0) throw std::invalid_argument("invalid GLM state device");
        const tyloi::Device device(tyloi::kCUDA, devices[i]);
        std::size_t retained = 0, temporary = 0;
        if (const auto* kda = std::get_if<KdaState>(&layers[i])) {
            if (kda->pending() || (kda->conv.defined() && kda->conv.size(0) != batch))
                throw std::invalid_argument("invalid GLM KDA state for growth");
            const std::array<int64_t, 5> conv{batch, kda_kernel_ - 1, 3, kda_heads_, kda_width_};
            const std::array<int64_t, 4> recurrent{batch, kda_heads_, kda_width_, kda_width_};
            const auto conv_bytes = tyloi::make_contiguous_view(nullptr, conv, tyloi::kFloat32, device).nbytes();
            const auto recurrent_bytes = tyloi::make_contiguous_view(nullptr, recurrent, tyloi::kFloat32, device).nbytes();
            if (input.confirmed > 0) {
                if (!kda->conv.defined() || !kda->recurrent.defined())
                    throw std::invalid_argument("GLM KDA verification requires existing state");
                const std::array<int64_t, 5> prefix{batch, kda_kernel_ - 1 + input.confirmed, 3, kda_heads_, kda_width_};
                const std::array<int64_t, 5> suffix{batch, kda_kernel_ - 1 + tokens - input.confirmed, 3, kda_heads_, kda_width_};
                add(retained, tyloi::make_contiguous_view(nullptr, prefix, tyloi::kFloat32, device).nbytes());
                add(retained, tyloi::make_contiguous_view(nullptr, suffix, tyloi::kFloat32, device).nbytes());
                add(retained, recurrent_bytes);
                add(retained, recurrent_bytes);
            } else {
                const std::array<int64_t, 5> joined{batch, kda_kernel_ - 1 + tokens, 3, kda_heads_, kda_width_};
                const auto joined_bytes = tyloi::make_contiguous_view(nullptr, joined, tyloi::kFloat32, device).nbytes();
                if (batch == 1) add(retained, joined_bytes);
                else { add(temporary, joined_bytes); add(retained, conv_bytes); }
                if (!kda->conv.defined()) add(temporary, conv_bytes);
                add(retained, recurrent_bytes);
            }
        } else {
            const auto& mla = std::get<MlaState>(layers[i]);
            if (mla.latent.position() != cache_pos || mla.index.position() != cache_pos)
                throw std::invalid_argument("GLM MLA/model cache positions diverged");
            add(retained, mla.latent.growth_bytes(batch, tokens));
            add(retained, mla.index.growth_bytes(batch, tokens));
        }
        auto& device_growth = growth[devices[i]];
        auto peak = device_growth.retained;
        add(peak, retained);
        add(peak, temporary);
        device_growth.peak = std::max(device_growth.peak, peak);
        add(device_growth.retained, retained);
    }
    std::vector<std::pair<tyloi::Device, std::size_t>> result;
    for (const auto& [device, value] : growth)
        result.emplace_back(tyloi::Device(tyloi::kCUDA, device), value.peak);
    return result;
}

std::vector<::tyloi::Tensor> State::storage() const {
    std::vector<::tyloi::Tensor> result;
    const auto add = [&](const ::tyloi::Tensor& value) { if (value.defined()) result.emplace_back(value); };
    for (const auto& layer : layers) {
        if (const auto* kda = std::get_if<KdaState>(&layer)) {
            add(kda->conv); add(kda->recurrent); add(kda->rollback_conv); add(kda->rollback_recurrent);
        } else {
            const auto& mla = std::get<MlaState>(layer);
            add(mla.latent.storage()); add(mla.index.storage());
        }
    }
    return result;
}

void State::rollback_speculative(std::int64_t accepted) {
    if (!(speculative_start >= 0 && accepted >= 0))
        throw std::runtime_error("no speculative transaction");
    const auto keep = speculative_start + speculative_confirmed + accepted;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        MfqCudaGuard guard(devices[i]);
        if (auto* kda = std::get_if<KdaState>(&layers[i])) kda->rollback();
        else std::get<MlaState>(layers[i]).truncate(keep);
    }
    cache_pos = keep;
    speculative_start = -1;
    speculative_confirmed = 0;
}

tyloi::models::Info Model::info() const { return tyloi::models::model_info(metadata); }
MfqModelCapabilities Model::capabilities() const {
    return {graph.architecture, graph.has_component("text"), false, false,
        false, false, false, bool(mtp), "model"};
}
tyloi::engine::Metrics Model::metrics() const {
    const bool mtp_declared = graph.has_component("predictor");
    return {{"kv_memory_supported", 0.0}, {"kv_quantization_supported", 0.0},
        {"kv_quantization_bits", 0.0}, {"vision_declared", double(graph.has_component("vision"))},
        {"vision_supported", 0.0}, {"vision_available", 0.0},
        {"mtp_declared", double(mtp_declared)}, {"mtp_supported", double(mtp_declared)},
        {"mtp_available", double(bool(mtp))}};
}
std::unique_ptr<tyloi::models::State> Model::make_state() const {
    auto state = std::make_unique<State>(*this);
    if (mtp) state->predictor = mtp->make_state();
    return state;
}
tyloi::StepSequence<tyloi::models::Prepared> Model::prepare(
        const std::vector<std::int64_t>&, const MfqMultimodalInput&, tyloi::models::State&) const {
    throw std::invalid_argument("model has no multimodal component");
}

::tyloi::Tensor Model::embedding(const ::tyloi::Tensor& input_ids, int primary) const {
    auto ids = input_ids.dense();
    if (!(ids.dim() == 1 || ids.dim() == 2))
        throw std::runtime_error("token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = ids.unsqueeze(0);
    if (!(ids.size(0) > 0 && ids.size(1) > 0))
        throw std::runtime_error("token IDs must not be empty");
    auto normalized = tyloi::ops::contiguous(tyloi::ops::copy(tyloi::ops::cast(ids, tyloi::kInt64), tyloi::Device(tyloi::kCUDA, primary)));
    return tyloi::ops::measure("model.embed", normalized.device(), [&] {
        return tyloi::ops::embedding(normalized, embed);
    });
}
::tyloi::Tensor Model::normalize(const ::tyloi::Tensor& hidden, std::int64_t, std::int64_t) const {
    return rms_norm(hidden.dense().mean(2), output_norm.dense(), metadata.rms_norm_eps);
}
::tyloi::Tensor Model::softcap(const ::tyloi::Tensor& value) const { return value; }
::tyloi::Tensor Model::logits(const ::tyloi::Tensor& hidden, bool, tyloi::models::LogitPrecision) const {
    return tyloi::ops::project(hidden, lm_head);
}

bool supports_speculation(const Model& model) noexcept { return !model.layers.empty(); }
bool Model::supports_speculation() const noexcept {
    return glm5_next::supports_speculation(*this);
}

::tyloi::Tensor Model::forward(tyloi::models::Input input, tyloi::models::State& state_value,
                               const tyloi::models::ForwardOptions& mode) const {
    auto& state = static_cast<State&>(state_value);
    if (input.suffix)
        throw std::runtime_error("GLM Flash-Next does not support speculative suffix forwarding");
    MfqCudaGuard guard(mode.primary);
    auto ids = input.ids.dense();
    if (!(ids.dim() == 1 || ids.dim() == 2))
        throw std::runtime_error("token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = ids.unsqueeze(0);
    if (!(ids.size(0) > 0 && ids.size(1) > 0))
        throw std::runtime_error("token IDs must not be empty");
    ids = tyloi::ops::contiguous(tyloi::ops::copy(tyloi::ops::cast(ids, tyloi::kInt64), tyloi::Device(tyloi::kCUDA, mode.primary))).dense();
    auto embeddings = input.embeddings.defined() ? input.embeddings.dense() : embedding(ids, mode.primary).dense();
    const auto B = ids.size(0), T = ids.size(1);
    if (!(embeddings.dim() == 3 && embeddings.size(0) == B &&
        embeddings.size(1) == T && embeddings.size(2) == metadata.hidden_size))
        throw std::runtime_error("inputs_embeds shape must match [batch,tokens,hidden_size]");
    if (!(T > 0 && state.cache_pos + T <= metadata.max_position_embeddings &&
        !input.positions.defined() && !input.cache_positions.defined()))
        throw std::runtime_error("GLM Flash-Next currently requires contiguous causal cache positions");
    if (!(state.speculative_start < 0))
        throw std::runtime_error("commit or roll back the pending speculative pass before forwarding");
    if (!(input.confirmed >= 0 &&
        (input.confirmed == 0 || (input.confirmed < T && B == 1 && state.cache_pos > 0 && supports_speculation()))))
        throw std::runtime_error("unsupported speculative backbone geometry");
    if (input.confirmed > 0) {
        state.speculative_start = state.cache_pos;
        state.speculative_confirmed = input.confirmed;
    }
    if (state.cache_pos == 0) state.reset(B);
    auto hidden = tyloi::ops::copy(embeddings, tyloi::Device(tyloi::kCUDA, mode.primary)).dense().contiguous()
        .to(::tyloi::kFloat16).unsqueeze(2).expand({B, T, config.streams, config.hidden}).contiguous();
    const auto trace = [&](const DenseTensor& value) {
        if (input.trace) input.trace->push_back(tyloi::ops::clone(tyloi::ops::cast(
            tyloi::ops::copy(value, tyloi::Device(tyloi::kCUDA, mode.primary)), tyloi::kFloat32)));
    };
    trace(hidden);
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        MfqCudaGuard layer_guard(layer.device);
        hidden = execute(layer, state.layers[i],
                         tyloi::ops::copy(hidden, tyloi::Device(tyloi::kCUDA, layer.device)).dense(),
                         state.cache_pos, input.confirmed);
        trace(hidden);
    }
    state.cache_pos += T;
    hidden = tyloi::ops::copy(hidden, tyloi::Device(tyloi::kCUDA, mode.primary)).dense();
    auto finalized = normalize(hidden, B, T);
    if (input.raw_hidden) *input.raw_hidden = finalized;
    return finalized;
}

Mtp::Mtp(::tyloi::Tensor embedding, int primary)
    : embedding_(std::move(embedding)), primary_(primary) {}
Mtp::~Mtp() = default;
Mtp::Mtp(Mtp&&) noexcept = default;

bool Mtp::load_weights(const ::tyloi::weights::Loader& weights,
                       const tyloi::models::glm5_next::Config &main) {
    LayerLoader loader{weights};
    loader.role = "glm5_next";
    return tyloi::models::glm5_next::load_predictor(*this, loader, main);
}

void Mtp::reset(tyloi::models::MtpState& value, int64_t next_batch) const {
    auto& state = this->state<PredictorState>(value);
    if (!(next_batch > 0))
        throw std::runtime_error("GLM5-Next MTP batch must be positive");
    for (auto &layer : state.states)
        layer.reset();
    std::fill(state.lengths.begin(), state.lengths.end(), 0);
    state.batch = next_batch;
}

std::pair<DenseTensor, DenseTensor> Mtp::evaluate(tyloi::models::MtpState& value,
                                       const DenseTensor &hidden,
                                       const DenseTensor &ids, int64_t depth, bool cache,
                                       const DenseTensor &supplied_positions,
                                       const DenseTensor &supplied_embeddings) const {
    auto& state = this->state<PredictorState>(value);
    if (!(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 &&
                          hidden.dim() == 3 && hidden.size(0) == ids.size(0) &&
                          hidden.size(1) == ids.size(1) && hidden.size(2) == hidden_norm.numel()))
        throw std::runtime_error("GLM5-Next MTP input geometry mismatch");
    const auto b = ids.size(0), t = ids.size(1);
    auto dtype = ::tyloi::kFloat16;
    const auto capacity = config.maximum;
    const auto count = int64_t(state.lengths.size());
    if (!(count > 0 && b > 0 && t > 0))
        throw std::runtime_error("invalid predictor geometry");
    const auto layer = (depth % count + count) % count;
    if (cache && state.batch && state.batch != b)
        reset(state, b);
    const auto start = cache ? state.lengths[layer] : 0;
    if (!(start >= 0 && t <= capacity - start))
        throw std::runtime_error("predictor history exceeds context capacity");
    auto embeds = supplied_embeddings;
    if (!embeds.defined()) {
        const auto device = ::tyloi::Device(::tyloi::kCUDA, primary_);
        auto normalized = ::tyloi::ops::contiguous(::tyloi::ops::copy(::tyloi::ops::cast(ids, ::tyloi::kInt64), device));
        embeds = ::tyloi::ops::measure("model.embed", device, [&] {
            return ::tyloi::ops::embedding(normalized, embedding_);
        }).dense();
    }
    if (!(embeds.sizes().vec() ==
                          std::vector<int64_t>({b, t, config.hidden})))
        throw std::runtime_error("GLM5-Next MTP embedding shape mismatch");
    DenseTensor pos;
    {
        auto current = supplied_positions.defined()
                           ? supplied_positions
                           : ::tyloi::cuda::arange(start, start + t,
                                        ids.options().dtype(::tyloi::kInt32));
        if (!((current.dim() == 1 || current.dim() == 2) &&
                              current.size(-1) == t &&
                              (current.dim() == 1 || current.size(0) == b)))
            throw std::runtime_error("GLM MTP positions require [T] or [B,T]");
        current = current.to(::tyloi::kInt32);
        pos = std::move(current);
    }
    DenseTensor x;
    {
        DenseTensor e;
        {
            auto mask = (pos == 0).reshape({pos.dim() == 1 ? 1 : b, t, 1});
            auto masked = ::tyloi::cuda::where(mask, ::tyloi::cuda::zeros_like(embeds), embeds);
            e = rms_norm(masked, embedding_norm.dense(), config.eps);
        }
        // Release prior-stage tensors before later projections allocate.
        embeds = {};
        DenseTensor streams;
        streams = rms_norm(hidden, hidden_norm.dense(), config.eps);
        x = ::tyloi::ops::project(::tyloi::cuda::cat({e, streams}, -1), fusion).dense();
    }
    DenseTensor multi;
    {
        auto &block = layers[layer];
        dtype = x.scalar_type() == ::tyloi::kFloat32 ? ::tyloi::kFloat32 : ::tyloi::kFloat16;
        auto normalized = rms_norm(x, block.attention_norm.dense(), config.eps);
        auto branch = block.attention->forward(state.states[layer], normalized, cache);
        x = x.to(::tyloi::kFloat32) + branch.to(::tyloi::kFloat32);
        normalized = rms_norm(x, block.ffn_norm.dense(), config.eps).to(dtype);
        branch = feed_forward(block.ffn, normalized);
        multi = x.to(::tyloi::kFloat32) + branch.to(::tyloi::kFloat32);
    }
    x = {};
    DenseTensor result;
    result = rms_norm(multi, output_norm.dense(), config.eps).to(dtype);
    if (cache) {
        state.batch = b;
        state.lengths[layer] = start + t;
    }
    return {std::move(result), std::move(multi)};
}

tyloi::models::MtpStep Mtp::step(tyloi::models::MtpState& value,
                                  const ::tyloi::Tensor& hidden, const ::tyloi::Tensor& ids,
                                  const ::tyloi::Tensor& positions,
                                  const tyloi::models::ForwardOptions&) const {
    if (positions.defined())
        throw std::runtime_error("this CUDA MTP predictor does not accept explicit positions");
    auto result = evaluate(value, hidden.dense(),
                           ids.dense());
    return {std::make_shared<DenseTensor>(std::move(result.first)),
            std::make_shared<DenseTensor>(std::move(result.second))};
}

void Mtp::trim_cache_to(tyloi::models::MtpState& value, int64_t position) const {
    auto& state = this->state<PredictorState>(value);
    if (!(position >= 0 && position <= state.position()))
        throw std::runtime_error("GLM5-Next MTP cache trim position is invalid");
    constexpr size_t index = 0;
    if (index < layers.size() && layers[index].attention)
        state.states[index].truncate(position);
    state.lengths[index] = position;
}

bool Mtp::teacher_forced_prompt_prime() const noexcept { return false; }

bool Mtp::target_bootstrap_decode() const noexcept { return true; }

bool Mtp::preserve_output_dtype() const noexcept { return true; }

} // namespace tyloi::cuda::glm5_next

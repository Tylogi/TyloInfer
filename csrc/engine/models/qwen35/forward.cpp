#include "engine/models/qwen35/model.h"
#include "engine/models/qwen35/forward.h"
#include "engine/models/qwen35/state.h"
#include "engine/models/common/ffn.h"

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <utility>

namespace tyloi::models::qwen35 {
namespace {

void require_forward(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void capture_target(std::span<const std::int64_t> layers, std::size_t index,
                    const Tensor& hidden, Device primary, std::vector<Tensor>& values) {
    if (values.size() < layers.size() && layers[values.size()] == std::int64_t(index)) {
        ops::DeviceScope scope(primary);
        values.push_back(ops::contiguous(ops::cast(ops::copy(hidden, primary), kFloat32)));
    }
}

Tensor target_raw(std::span<const std::int64_t> layers, std::vector<Tensor>& values) {
    require_forward(values.size() == layers.size(), "DFlash2 target hidden states were not fully captured");
    return ops::contiguous(ops::cat(values, -1));
}

Tensor normalize_ids(Tensor ids, Device primary) {
    require_forward(ids.dim() == 1 || ids.dim() == 2,
                    "token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = ops::unsqueeze(ids, 0);
    require_forward(ids.size(0) > 0 && ids.size(1) > 0, "token IDs must not be empty");
    return ops::contiguous(ops::copy(ops::cast(ids, kInt64), primary));
}

const ops::RopeTable& rope_for(const Model& model, const Layer& layer) {
    if (layer.device.is_cpu()) return model.cpu_rope;
    const auto found=model.device_ropes.find(layer.device.index);
    return found==model.device_ropes.end() ? model.rope : found->second;
}

models::AttentionStep layer_step(const Layer& layer, const models::AttentionStep& step) {
    auto local = step;
    local.positions = ops::contiguous(ops::copy(step.positions, layer.device));
    if (step.sequence_lengths.defined())
        local.sequence_lengths = ops::contiguous(ops::copy(step.sequence_lengths, layer.device));
    if (step.cache_positions.defined())
        local.cache_positions = ops::contiguous(ops::copy(step.cache_positions, layer.device));
    return local;
}

void prepare_linear_state(const models::LinearAttention& attention, LinearState& state,
                          const Tensor& input, bool transposed) {
    if (state.mixer.defined()) return;
    const auto batch = input.size(0);
    const auto width = 2 * attention.key_heads * attention.key_dim +
                       attention.value_heads * attention.value_dim;
    const auto options = input.options().dtype(kFloat32);
    state.mixer.view() = {
        ops::zeros({batch, attention.conv_kernel - 1, width}, options),
        ops::zeros({batch, attention.value_heads, attention.value_dim, attention.value_dim}, options),
        !input.device().is_cpu() && transposed};
}

void clear_linear_speculative(LinearState& state) noexcept {
    state.projected = {};
    state.mixer.commit();
}

struct PreparedRow {
    State* state = nullptr;
    models::Input input;
    Tensor embeddings;
    models::AttentionStep step;
    std::int64_t batch = 0;
    std::int64_t tokens = 0;
};

PreparedRow prepare_row(const Model& model, ForwardRow row,
                        const models::ForwardOptions& options, bool prepare_pages,
                        bool capture = false) {
    require_forward(row.state != nullptr, "Qwen3.5 forward row has no state");
    require_forward(row.state->belongs_to(model),
                    "Qwen3.5 state belongs to another model");
    auto& state = *row.state;
    auto input = std::move(row.input);
    require_forward(!input.suffix, "Qwen3.5 does not support speculative suffix forwarding");
    const Device primary{model.device.type, options.primary};
    const auto [B, T] = input.shape();
    auto ids = input.ids && input.ids.defined() ? normalize_ids(input.ids, primary) : Tensor{};
    auto embeddings = input.embeddings && input.embeddings.defined()
        ? input.embeddings
        : model.embedding(ids, options.primary);
    require_forward(embeddings.dim() == 3 && embeddings.size(0) == B &&
                        embeddings.size(1) == T && embeddings.size(2) == model.metadata.hidden_size,
                    "inputs_embeds shape must match [batch,tokens,hidden_size]");
    require_forward(B > 0 && T > 0, "Qwen3.5 input batch and tokens must be positive");
    require_forward(input.confirmed >= 0 &&
        (input.confirmed == 0 || (input.confirmed < T && B == 1 && state.cache_pos > 0 &&
         (capture || (!input.positions && !input.cache_positions)) && model.supports_speculation())),
        "unsupported speculative backbone geometry");
    require_forward(state.speculative_start < 0,
                    "commit or roll back the pending speculative pass before forwarding");
    require_forward(state.cache_pos + T <= model.metadata.max_position_embeddings,
                    "Qwen3.5 history exceeds context capacity");
    if (state.cache_pos == 0) state.reset(B);
    else require_forward(state.batch == B, "Qwen3.5 cache batch does not match input batch");
    if (prepare_pages) state.prepare_pages(state.cache_pos + T);
    if (input.lengths)
        require_forward(input.lengths.dim() == 1 && input.lengths.size(0) == B,
                        "sequence lengths must have shape [batch]");
    auto physical = input.cache_positions
        ? ops::copy(ops::cast(input.cache_positions, kInt64), primary)
        : ops::arange(state.cache_pos, state.cache_pos + T, primary);
    physical = ops::contiguous(physical);
    require_forward((physical.dim() == 1 && physical.numel() == T) ||
                    (physical.dim() == 2 && physical.size(0) == B && physical.size(1) == T),
                    "cache_positions must have shape [tokens] or [batch,tokens]");
    const bool has_mrope = model.rope.sections && model.rope.sections.numel() > 0;
    auto semantic = input.positions
        ? ops::copy(ops::cast(input.positions, kInt64), primary)
        : (state.decode_position_delta == 0 ? physical
                                            : ops::add(physical, state.decode_position_delta));
    semantic = ops::contiguous(semantic);
    const auto rank = semantic.dim();
    require_forward(Config::valid_positions(rank, rank > 1 ? semantic.size(0) : 0,
                                           semantic.size(-1), B, T, has_mrope),
                    "position_ids must have shape [tokens], [batch,tokens], or configured grid-MRoPE [3,tokens]");
    Tensor lengths;
    if (input.lengths) lengths = ops::contiguous(ops::copy(ops::cast(input.lengths, kInt64), primary));
    else if (state.cache_pos > 0 && T == 1) {
        const std::array<std::int64_t, 1> shape{B};
        lengths = ops::full(shape, state.cache_pos + 1, primary);
    }
    models::AttentionStep step{semantic, lengths,
        has_mrope || input.positions || input.cache_positions ? physical : Tensor{},
        state.cache_pos, input.kv_length, input.attention_parts};
    if (capture) step.contiguous_tail = true;
    if (input.confirmed > 0) {
        state.speculative_start = state.cache_pos;
        state.speculative_confirmed = input.confirmed;
    }
    return {&state, std::move(input), std::move(embeddings), std::move(step), B, T};
}

using AttentionResult = ::tyloi::models::LinearOutput;

std::array<Tensor, 2> prepare_ffn(const Linear& layer, const AttentionResult& attention) {
    return ops::recurrent_norm(attention.residual, attention.output, layer.ffn_norm,
        layer.attention.rms_norm_eps, layer.attention.norm_weight_offset);
}

} // namespace

static AttentionResult forward_attention(const Layer& layer, LayerState& state, Tensor input,
                     const ::tyloi::models::AttentionStep& step, std::int64_t confirmed_prefix,
                     const tyloi::ops::RopeTable& rope,
                     const models::ForwardOptions& options, bool transposed,
                     bool scaled_readout = true) {
    ops::DeviceScope scope(layer.device);
    auto local_step = layer_step(layer, step);
    input = ops::contiguous(ops::copy(input, layer.device));
    const auto* full = std::get_if<Full>(&layer.weights);
    auto* trace = full != nullptr && options.trace_layer >= 0 &&
                                options.trace_layer == full->attention.layer
                            ? options.trace
                            : nullptr;
    if (full != nullptr) {
        auto* cache = std::get_if<tyloi::state::KvView>(&state);
        require_forward(cache != nullptr, "Qwen3.5 full layer state kind mismatch");
        const auto& attention = full->attention;
        const auto capacity = kv_capacity_for(attention.sliding,
            attention.sliding ? attention.attention_window : attention.max_position_embeddings,
            local_step.cache_position + input.size(1), options.kv_capacity);
        if (!cache->defined()) {
            *cache = tyloi::state::allocate_kv(
                input.size(0), attention.kv_heads, capacity, attention.attention_head_dim,
                attention.sliding, input.device(),
                attention.official_bf16 ? ::tyloi::kBFloat16
                                        : ::tyloi::kFloat16,
                attention.kv_quantization);
        }
        cache->quantization = attention.kv_quantization;
        auto output = attention.forward(input, local_step, *cache,
                                        rope, trace);
        return {std::move(output[0]), std::move(output[1]), {}};
    }
    const auto* linear = std::get_if<Linear>(&layer.weights);
    auto* linear_state = std::get_if<LinearState>(&state);
    require_forward(linear != nullptr && linear_state != nullptr,
                    "Qwen3.5 linear layer state kind mismatch");
    const auto& attention = linear->attention;
    prepare_linear_state(attention, *linear_state, input, transposed);
    if (confirmed_prefix == 0) {
        require_forward(!linear_state->mixer.pending(),
                        "linear attention has a pending speculative pass");
    } else {
        require_forward(!input.device().is_cpu() && !linear_state->mixer.pending() &&
                            linear_state->mixer.defined() && confirmed_prefix > 0 &&
                            confirmed_prefix < input.size(1),
                        "invalid Qwen3.5 speculative linear-attention transaction");
        linear_state->mixer.begin(local_step.cache_position, confirmed_prefix, input.size(1));
    }
    auto& state_view = linear_state->mixer.view();
    const auto captured = confirmed_prefix > 0 ? &linear_state->projected : nullptr;
    const ::tyloi::models::LinearView linear_view{
        state_view.convolution, state_view.recurrent, state_view.transposed};
    try {
        return linear->attention.forward(input, linear_view, captured,
            scaled_readout ? linear->ffn_norm : Tensor{});
    } catch (...) {
        if (confirmed_prefix > 0) {
            linear_state->mixer.abort();
            clear_linear_speculative(*linear_state);
        }
        throw;
    }
}

static Tensor finish_layer(const Layer& layer, AttentionResult attention,
                           const models::ForwardOptions& options) {
    ops::DeviceScope scope(layer.device);
    const auto* full = std::get_if<Full>(&layer.weights);
    if (full != nullptr) {
        const auto* trace = options.trace_layer >= 0 &&
                                    options.trace_layer == full->attention.layer
                                ? options.trace
                                : nullptr;
        return ::tyloi::models::full_ffn(
            full->ffn, full->ffn_norm, std::move(attention.residual),
            std::move(attention.output), full->attention.rms_norm_eps,
            full->attention.norm_weight_offset, full->attention.official_bf16,
            full->attention.attention_output_gate, trace, options.serial_experts);
    }
    const auto* linear = std::get_if<Linear>(&layer.weights);
    require_forward(linear != nullptr, "Qwen3.5 layer weight kind mismatch");
    auto pair = attention.normalized
        ? std::array<Tensor, 2>{std::move(attention.residual), std::move(attention.normalized)}
        : prepare_ffn(*linear, attention);
    return ::tyloi::models::linear_ffn(linear->ffn, pair, options.serial_experts);
}

static models::Decoded forward_one(const Model& model, State& selected, models::Input input,
                                   const models::ForwardOptions& options, bool capture) {
    const Device primary{model.device.type, options.primary};
    ops::DeviceScope scope(primary);
    auto prepared = prepare_row(model, ForwardRow{&selected, std::move(input)}, options, !capture, capture);
    auto& state = *prepared.state;
    const auto targets = model.mtp && (capture || prepared.input.raw_hidden)
        ? model.mtp->target_layers() : std::span<const std::int64_t>{};
    std::vector<Tensor> captured;
    auto hidden = ops::contiguous(ops::copy(prepared.embeddings, primary));
    auto snapshot = [&](const Tensor& value) {
        return ops::clone(ops::contiguous(ops::cast(ops::copy(value, primary), kFloat32)));
    };
    if (prepared.input.trace) prepared.input.trace->push_back(snapshot(hidden));
    try {
        for (std::size_t index = 0; index < model.layers.size(); ++index) {
            const auto& layer = model.layers[index];
            auto attention = forward_attention(layer, state.layers[index], hidden,
                                               prepared.step, prepared.input.confirmed,
                                               rope_for(model,layer), options, state.transposed());
            hidden = finish_layer(layer, std::move(attention), options);
            capture_target(targets, index, hidden, primary, captured);
            if (!capture && std::get_if<Linear>(&layer.weights) != nullptr && prepared.input.confirmed > 0) {
                auto& linear = std::get<LinearState>(state.layers[index]);
                ++linear.speculative_projection_batches;
                ++linear.speculative_ffn_batches;
            }
            if (prepared.input.trace) prepared.input.trace->push_back(snapshot(hidden));
        }
    } catch (...) {
        state.abort_speculative();
        throw;
    }
    if (!capture && (!prepared.input.positions || prepared.input.advance)) state.cache_pos += prepared.tokens;
    hidden = ops::contiguous(ops::copy(hidden, primary));
    models::Decoded result;
    result.hidden = model.normalize(hidden, prepared.batch, prepared.tokens);
    result.raw = targets.empty() ? std::move(hidden) : target_raw(targets, captured);
    if (prepared.input.raw_hidden) *prepared.input.raw_hidden = result.raw;
    return result;
}

Tensor Model::forward(models::Input input, models::State& value, const models::ForwardOptions& options) const {
    auto* selected = dynamic_cast<State*>(&value);
    require_forward(selected && selected->belongs_to(*this), "Qwen3.5 state belongs to another model");
    return forward_one(*this, *selected, std::move(input), options, false).hidden;
}

models::Decoded capture_forward(const Model& model, State& state, models::Input input,
                                const models::ForwardOptions& options) {
    const Device primary{model.device.type, options.primary};
    const auto [batch, tokens] = input.shape();
    const auto stable = [primary](const Tensor& tensor) {
        return tensor.defined() && tensor.device() == primary && tensor.scalar_type() == kInt64 &&
               tensor.is_contiguous();
    };
    require_forward(primary.is_cuda() && state.belongs_to(model) && state.cache_pos > 0 &&
                        state.batch == 1 && batch == 1 &&
                        tokens <= 8 && input.confirmed >= 0 && input.confirmed < tokens &&
                        input.kv_length >= state.cache_pos + tokens &&
                        input.attention_parts > 0 && stable(input.positions) &&
                        stable(input.cache_positions) && stable(input.lengths),
                    "invalid Qwen3.5 capture geometry");
    require_forward(state.layers.size() == model.layers.size(), "invalid Qwen3.5 capture state");
    for (std::size_t index = 0; index < state.layers.size(); ++index) {
        require_forward(model.layers[index].device == primary, "Qwen3.5 capture requires one CUDA device");
        if (const auto* cache = std::get_if<tyloi::state::KvView>(&state.layers[index])) {
            require_forward(cache->defined() && !cache->paged() && !cache->ring && cache->batch == batch &&
                                cache->physical_key.device() == primary && cache->physical_value.device() == primary &&
                                cache->physical_key.dim() == 4 && cache->physical_value.dim() == 4 &&
                                cache->physical_key.size(2) >= input.kv_length &&
                                cache->physical_value.size(2) >= input.kv_length,
                            "Qwen3.5 capture requires preallocated nonpaged KV state");
        } else {
            const auto& mixer = std::get<LinearState>(state.layers[index]).mixer;
            require_forward(mixer.defined() && !mixer.pending() &&
                                mixer.view().convolution.device() == primary &&
                                mixer.view().recurrent.device() == primary,
                            "Qwen3.5 capture requires prefilled recurrent state");
        }
    }
    return forward_one(model, state, std::move(input), options, true);
}

StepSequence<std::vector<models::Decoded>> forward_batch(
    const Model& model, std::vector<ForwardRow> rows,
    const models::ForwardOptions& options, bool advance) {
    require_forward(!rows.empty(), "Qwen3.5 packed forward requires at least one row");
    const Device primary{model.device.type, options.primary};
    ops::DeviceScope scope(primary);
    std::vector<PreparedRow> prepared;
    try {
        prepared.reserve(rows.size());
        for (auto& row : rows) {
            auto value = prepare_row(model, std::move(row), options, advance);
            require_forward(value.batch == 1,
                            "Qwen3.5 packed forward requires B=1 rows");
            prepared.push_back(std::move(value));
        }

        std::vector<Tensor> hidden;
        hidden.reserve(prepared.size());
        for (auto& row : prepared)
            hidden.push_back(ops::contiguous(ops::copy(row.embeddings, primary)));
        const auto snapshot = [&](const Tensor& value) {
            return ops::clone(ops::contiguous(ops::cast(ops::copy(value, primary), kFloat32)));
        };
        for (std::size_t row_index = 0; row_index < prepared.size(); ++row_index)
            if (prepared[row_index].input.trace)
                prepared[row_index].input.trace->push_back(snapshot(hidden[row_index]));

        std::vector<std::int64_t> offsets(prepared.size() + 1, 0);
        for (std::size_t index = 0; index < prepared.size(); ++index)
            offsets[index + 1] = offsets[index] + prepared[index].tokens;

        for (std::size_t layer_index = 0; layer_index < model.layers.size(); ++layer_index) {
            co_yield LayerEvent{LayerEvent::Phase::begin, LayerEvent::Mode::packed};
            const auto& layer = model.layers[layer_index];
            ops::DeviceScope layer_scope(layer.device);
            const auto& rope=rope_for(model,layer);
            const bool scaled_readout = layer.device.is_cuda() && std::holds_alternative<Linear>(layer.weights) &&
                std::all_of(hidden.begin(), hidden.end(),
                    [](const auto& value) { return value.scalar_type() == kFloat16; });
            std::vector<AttentionResult> attended;
            attended.reserve(prepared.size());
            for (std::size_t row_index = 0; row_index < prepared.size(); ++row_index) {
                attended.push_back(forward_attention(
                    layer, prepared[row_index].state->layers[layer_index],
                    std::move(hidden[row_index]), prepared[row_index].step,
                    prepared[row_index].input.confirmed, rope, options,
                    prepared[row_index].state->transposed(), scaled_readout));
            }
            const bool normalized = std::any_of(attended.begin(), attended.end(),
                [](const auto& attention) { return attention.normalized.defined(); });
            std::vector<Tensor> residuals, values;
            residuals.reserve(attended.size());
            values.reserve(attended.size());
            for (auto& attention : attended) {
                if (normalized && !attention.normalized) {
                    auto pair = prepare_ffn(std::get<Linear>(layer.weights), attention);
                    attention = {std::move(pair[0]), {}, std::move(pair[1])};
                }
                residuals.push_back(std::move(attention.residual));
                values.push_back(normalized ? std::move(attention.normalized) : std::move(attention.output));
            }
            AttentionResult packed_attention{
                residuals.size() == 1 ? std::move(residuals.front()) : ops::cat(residuals, 1), {}, {}};
            (normalized ? packed_attention.normalized : packed_attention.output) =
                values.size() == 1 ? std::move(values.front()) : ops::cat(values, 1);
            auto packed = finish_layer(layer, std::move(packed_attention), options);
            if (std::get_if<Linear>(&layer.weights) != nullptr) {
                for (auto& row : prepared) {
                    if (row.input.confirmed == 0) continue;
                    auto& linear = std::get<LinearState>(row.state->layers[layer_index]);
                    ++linear.speculative_projection_batches;
                    ++linear.speculative_ffn_batches;
                }
            }
            for (std::size_t row_index = 0; row_index < prepared.size(); ++row_index) {
                hidden[row_index] = ops::slice(
                    packed, 1, offsets[row_index], offsets[row_index + 1]);
                if (prepared[row_index].input.trace)
                    prepared[row_index].input.trace->push_back(snapshot(hidden[row_index]));
            }
            co_yield LayerEvent{LayerEvent::Phase::end, LayerEvent::Mode::packed};
        }

        for (auto& row : prepared) {
            if (advance && (!row.input.positions || row.input.advance))
                row.state->cache_pos += row.tokens;
        }
        std::vector<Tensor> packed_parts;
        packed_parts.reserve(hidden.size());
        for (auto& value : hidden) packed_parts.push_back(std::move(value));
        auto raw_packed = packed_parts.size() == 1 ? std::move(packed_parts.front())
                                                   : ops::cat(packed_parts, 1);
        raw_packed = ops::contiguous(ops::copy(raw_packed, primary));
        auto finalized = model.normalize(raw_packed, 1, offsets.back());
        std::vector<models::Decoded> result;
        result.reserve(prepared.size());
        for (std::size_t row_index = 0; row_index < prepared.size(); ++row_index) {
            models::Decoded decoded;
            decoded.hidden = ops::slice(finalized, 1, offsets[row_index], offsets[row_index + 1]);
            decoded.raw = ops::slice(raw_packed, 1, offsets[row_index], offsets[row_index + 1]);
            if (prepared[row_index].input.raw_hidden) {
                *prepared[row_index].input.raw_hidden = decoded.raw;
            }
            result.push_back(std::move(decoded));
        }
        co_yield std::move(result);
    } catch (...) {
        for (auto& row : prepared)
            if (row.state) row.state->abort_speculative();
        throw;
    }
}

bool Model::supports_speculation() const noexcept {
    return !layers.empty() && std::all_of(layers.begin(), layers.end(), [](const Layer& layer) {
        if (const auto* full = std::get_if<Full>(&layer.weights)) return !full->attention.sliding;
        return true;
    });
}

#if !TYLOINFER_HAS_CUDA
StepSequence<models::Decoded> Model::decode(models::Input, models::State&,
        const models::ForwardOptions&, const models::DecodeOptions&) const {
    return {};
}
#endif

void replay_layers(const Model& model, State& state, std::int64_t retained) {
    for (std::size_t index = 0; index < state.layers.size(); ++index) {
        auto* linear = std::get_if<LinearState>(&state.layers[index]);
        if (linear == nullptr) continue;
        const auto& block = model.layers[index];
        ops::DeviceScope scope(block.device);
        const auto* layer = std::get_if<Linear>(&block.weights);
        require_forward(layer != nullptr, "Qwen3.5 recurrent layer weight kind mismatch");
        auto& view = linear->mixer.view();
        const models::LinearView linear_view{view.convolution, view.recurrent, view.transposed};
        layer->attention.replay(linear_view, linear->projected, retained);
    }
}

StepSequence<bool> Model::replay(State& state, std::int64_t accepted, [[maybe_unused]] bool capture) const {
    require_forward(state.belongs_to(*this), "Qwen3.5 state belongs to another model");
    require_forward(state.speculative_start >= 0 && accepted >= 0, "no speculative transaction");
    struct Rollback {
        State& state;
        ~Rollback() { if (state.speculating()) state.abort_speculative(); }
    } rollback{state};
    const auto keep = state.rewind(accepted);
    const auto retained = keep - state.speculative_start;
    std::optional<bool> graph_result;
#if TYLOINFER_HAS_CUDA
    auto graph = replay_graph(*this, state, retained, capture);
    while (auto step = graph.next()) {
        if (step.graph) co_yield std::move(*step.graph);
        else if (step.value) graph_result = *step.value;
    }
#endif
    if (!graph_result.has_value()) replay_layers(*this, state, retained);
    state.commit_speculative();
    state.cache_pos = keep;
    co_yield graph_result.value_or(false);
}

} // namespace tyloi::models::qwen35

#include "engine/models/qwen35/model.h"
#include "engine/models/qwen35/state.h"
#include "engine/models/common/ffn.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::qwen35 {
namespace {

void require_forward(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Tensor normalize_ids(Tensor ids, Device primary) {
    require_forward(ids.dim() == 1 || ids.dim() == 2,
                    "token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = ops::unsqueeze(ids, 0);
    require_forward(ids.size(0) > 0 && ids.size(1) > 0, "token IDs must not be empty");
    return ops::contiguous(ops::copy(ops::cast(ids, kInt64), primary));
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

Tensor forward_layer(const Layer& layer, LayerState& state, Tensor input,
                     const ::tyloi::models::AttentionStep& step, std::int64_t confirmed_prefix,
                     const tyloi::ops::RopeTable& rope,
                     const models::ForwardOptions& options, bool transposed) {
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
        return ::tyloi::models::full_ffn(
            full->ffn, full->ffn_norm, output[0], output[1],
            attention.rms_norm_eps, attention.norm_weight_offset,
            attention.official_bf16, attention.attention_output_gate, trace,
            options.serial_experts);
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
        auto output = linear->attention.forward(
            input, linear_view, captured);
        auto result = ::tyloi::models::linear_ffn(
            linear->ffn, linear->ffn_norm, std::move(output[0]), std::move(output[1]),
            attention.rms_norm_eps, attention.norm_weight_offset,
            options.serial_experts);
        if (confirmed_prefix > 0) {
            ++linear_state->speculative_projection_batches;
            ++linear_state->speculative_ffn_batches;
        }
        return result;
    } catch (...) {
        if (confirmed_prefix > 0) {
            linear_state->mixer.abort();
            clear_linear_speculative(*linear_state);
        }
        throw;
    }
}

} // namespace

Tensor Model::forward(models::Input input, State& state, const models::ForwardOptions& options) const {
    require_forward(!input.suffix, "Qwen3.5 does not support speculative suffix forwarding");
    const Device primary{device.type, options.primary};
    ops::DeviceScope scope(primary);
    auto ids = normalize_ids(input.ids, primary);
    const auto B = ids.size(0);
    const auto T = ids.size(1);
    auto embeddings = input.embeddings ? input.embeddings : embedding(ids, options.primary);
    require_forward(embeddings.dim() == 3 && embeddings.size(0) == B &&
                        embeddings.size(1) == T && embeddings.size(2) == metadata.hidden_size,
                    "inputs_embeds shape must match [batch,tokens,hidden_size]");
    require_forward(input.confirmed >= 0 &&
        (input.confirmed == 0 || (input.confirmed < T && B == 1 && state.cache_pos > 0 &&
         !input.positions && !input.cache_positions && supports_speculation())),
        "unsupported speculative backbone geometry");
    require_forward(state.speculative_start < 0,
                    "commit or roll back the pending speculative pass before forwarding");
    require_forward(state.cache_pos + T <= metadata.max_position_embeddings,
                    "Qwen3.5 history exceeds context capacity");
    if (state.cache_pos == 0) state.reset(B);
    else require_forward(state.batch == B, "Qwen3.5 cache batch does not match input batch");
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
    auto semantic = input.positions ? ops::copy(ops::cast(input.positions, kInt64), primary)
        : (state.decode_position_delta == 0 ? physical : ops::add(physical, state.decode_position_delta));
    semantic = ops::contiguous(semantic);
    const auto rank = semantic.dim();
    const bool has_mrope = rope.sections && rope.sections.numel() > 0;
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
    auto hidden = ops::contiguous(ops::copy(embeddings, primary));
    auto snapshot = [&](const Tensor& value) {
        return ops::clone(ops::contiguous(ops::cast(ops::copy(value, primary), kFloat32)));
    };
    if (input.trace) input.trace->push_back(snapshot(hidden));
    if (input.confirmed > 0) {
        state.speculative_start = state.cache_pos;
        state.speculative_confirmed = input.confirmed;
    }
    try {
        for (std::size_t index = 0; index < layers.size(); ++index) {
            const auto& layer = layers[index];
            const auto* active_rope = layer.device.is_cpu() ? &cpu_rope : &rope;
            if (!layer.device.is_cpu() && !device_ropes.empty()) {
                const auto it = device_ropes.find(layer.device.index);
                if (it != device_ropes.end()) active_rope = &it->second;
            }
            hidden = forward_layer(layer, state.layers[index], hidden, step, input.confirmed,
                                   *active_rope, options, state.transposed());
            if (input.trace) input.trace->push_back(snapshot(hidden));
        }
    } catch (...) {
        state.abort_speculative();
        throw;
    }
    if (!input.positions || input.advance) state.cache_pos += T;
    hidden = ops::contiguous(ops::copy(hidden, primary));
    auto finalized = normalize(hidden, B, T);
    if (input.raw_hidden) *input.raw_hidden = std::move(hidden);
    return finalized;
}

bool Model::supports_speculation() const noexcept {
    return !layers.empty() && std::all_of(layers.begin(), layers.end(), [](const Layer& layer) {
        if (const auto* full = std::get_if<Full>(&layer.weights)) return !full->attention.sliding;
        return true;
    });
}

void Model::replay(State& state, std::int64_t accepted) const {
    require_forward(state.speculative_start >= 0 && accepted >= 0, "no speculative transaction");
    try {
        const auto keep = state.rewind(accepted);
        for (std::size_t index = 0; index < state.layers.size(); ++index) {
            auto* linear = std::get_if<LinearState>(&state.layers[index]);
            if (linear == nullptr) continue;
            ops::DeviceScope scope(layers[index].device);
            const auto* layer = std::get_if<Linear>(&layers[index].weights);
            require_forward(layer != nullptr, "Qwen3.5 recurrent layer weight kind mismatch");
            auto& view = linear->mixer.view();
            const models::LinearView linear_view{view.convolution, view.recurrent, view.transposed};
            layer->attention.replay(linear_view, linear->projected, keep - state.speculative_start);
        }
        state.commit_speculative();
        state.cache_pos = keep;
    } catch (...) {
        state.abort_speculative();
        throw;
    }
}

} // namespace tyloi::cuda::qwen35

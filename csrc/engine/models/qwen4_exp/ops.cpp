#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/qwen4_exp/state.h"
#include "engine/models/qwen4_exp/taps.h"
#include "engine/models/common/mtp.h"
#include "engine/weights/cache/workspace.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace tyloi::models::qwen4_exp {

Tensor Model::embedding(const Tensor& ids, int) const {
    return embed_tokens(device_ids(ids));
}

Tensor Model::normalize(const Tensor& hidden, std::int64_t, std::int64_t) const { return hidden; }
Tensor Model::softcap(const Tensor& logits) const { return logits; }

Tensor Model::logits(const Tensor& hidden, bool, tyloi::models::LogitPrecision) const {
    ops::ProfileRange range("model.lm_head", hidden.device());
    auto output = ops::project(hidden, lm_head);
    range.finish();
    return output;
}

Tensor Model::forward(Input input, models::State& state_value, const ForwardOptions&) const {
    auto* state = dynamic_cast<State*>(&state_value);
    if (!state || !state->belongs_to(*this))
        throw std::invalid_argument("Qwen4 state belongs to a different model");
    if (input.suffix)
        if (!(state->speculative_start >= 0 && state->speculative_confirmed == 0 &&
            !state->speculative_suffix_forward))
            throw std::runtime_error("speculative suffix is not active");
    struct SuffixScope {
        State& state;
        bool active;
        ~SuffixScope() { if (active) state.speculative_suffix_forward = false; }
    } suffix{*state, input.suffix};
    if (input.suffix) state->speculative_suffix_forward = true;

    if (!bool(input.ids))
        throw std::runtime_error("Qwen4 forward requires token ids");
    auto ids = device_ids(input.ids);
    auto embeddings = input.embeddings ? ops::contiguous(ops::copy(input.embeddings, device)) : embed_tokens(ids);
    const auto batch = ids.size(0), tokens = ids.size(1);
    if (!(embeddings.dim() == 3 && embeddings.size(0) == batch &&
        embeddings.size(1) == tokens && embeddings.size(2) == hidden_size()))
        throw std::runtime_error("inputs_embeds shape must match [batch,tokens,hidden_size]");
    if (!(tokens > 0 && state->cache_pos + tokens <= metadata.max_position_embeddings &&
        !input.cache_positions))
        throw std::runtime_error("Qwen4 requires unpadded causal cache positions");
    if (!(state->speculative_start < 0 || state->speculative_suffix_forward))
        throw std::runtime_error("commit or roll back the pending speculative pass before forwarding");
    const auto confirmed = input.confirmed;
    if (!(confirmed >= 0 && (confirmed == 0 ||
        (confirmed < tokens && batch == 1 && state->cache_pos > 0 && supports_speculation()))))
        throw std::runtime_error("unsupported speculative backbone geometry");
    if (confirmed > 0) {
        if (!(state->cache_pos + tokens <= max_position_embeddings()))
            throw std::runtime_error("speculative pass exceeds context capacity");
        state->speculative_start = state->cache_pos;
        state->speculative_confirmed = confirmed;
    }
    if ((state->batch != 0 && state->batch != batch) || state->cache_pos == 0)
        state->reset(batch);
    auto cache_positions = ops::arange(state->cache_pos, state->cache_pos + tokens, device);
    auto current = input.positions ? device_ids(input.positions)
        : state->decode_position_delta == 0 ? cache_positions : ops::add(cache_positions, state->decode_position_delta);
    if ((current.dim() == 2 || current.dim() == 3) && current.size(0) == 4)
        current = ops::slice(current, 0, 1, 4);
    Tensor full_positions;
    {
        weights::Resident resident;
        full_positions = state->positions.defined()
            ? ops::cat(std::array<Tensor, 2>{state->positions, current}, -1) : ops::clone(current);
    }
    if (!((current.dim() == 1 || (current.dim() == 2 && current.size(0) == 3) ||
        (current.dim() == 3 && current.size(0) == 3 && current.size(1) == batch)) &&
        current.size(-1) == tokens))
        throw std::runtime_error("Qwen4 positions require [T], [3,T], [4,T], [3,B,T] or [4,B,T]");

    Tensor host_ids, cpu_positions;
    if (host_tokens || std::any_of(layers.begin(), layers.end(),
            [](const auto& layer) { return layer->device.is_cpu(); })) {
        host_ids = ops::contiguous(ops::copy(ids, Device(kCPU)));
        cpu_positions = ops::contiguous(ops::copy(current, Device(kCPU)));
        for (std::size_t i = 0; i < layers.size(); ++i)
            if (layers[i]->ple) layers[i]->ple->prefetch_tokens(state->layers[i]->ple, host_ids);
    }
    auto hidden = repeat_hidden(ops::contiguous(ops::copy(embeddings, device)), metadata.hc_mult);
    const auto prefill_rows = input.advance && batch == 1 && tokens > 8 && !input.suffix && !confirmed ? tokens : 0;
    const auto* mtp = predictor();
    const auto targets = device.is_cuda() && input.raw_hidden && mtp
        ? mtp->target_layers() : std::span<const std::int64_t>{};
    Taps taps(targets, metadata.hc_mult, device);
    state->begin_forward();
    const auto trace = [&] {
        if (input.trace) input.trace->push_back(ops::clone(ops::cast(ops::copy(hidden, device), kFloat32)));
    };
    trace();
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = *layers[i];
        ops::DeviceScope scope(layer.device);
        auto local_ids = layer.device.is_cpu() ? host_ids : ops::copy(ids, layer.device);
        auto local_positions = layer.device.is_cpu() ? cpu_positions : ops::copy(current, layer.device);
        auto local_full = ops::copy(full_positions, layer.device);
        hidden = ops::contiguous(ops::copy(hidden, layer.device));
        hidden = qwen4_exp::forward(layer, *state->layers[i], std::move(hidden),
            local_ids, local_positions, local_full, confirmed, host_ids, 0, prefill_rows);
        taps.capture(std::int64_t(i), hidden);
        trace();
    }
    state->finish_forward(full_positions, batch, tokens);
    state->cache_pos += tokens;
    hidden = ops::copy(hidden, device);
    if (input.raw_hidden) *input.raw_hidden = taps.finish(hidden);
    return final_mixer->pre(hidden, ops::ResidualRole::final).mixed;
}

void Model::load_config(std::string_view payload) {
    auto& config = this->config;
    auto& metadata = this->metadata;
    config = CausalConfig::from_json(payload);
    metadata.vocab_size = config.vocab;
    metadata.hidden_size = config.hidden;
    metadata.num_hidden_layers = config.layers;
    metadata.num_attention_heads = config.heads;
    metadata.num_key_value_heads = config.kv_heads;
    metadata.head_dim = config.width;
    metadata.max_position_embeddings = config.maximum;
    metadata.rotary_dim = config.rotary;
    metadata.num_experts = config.experts;
    metadata.hc_mult = config.streams;
    metadata.rope_base = config.rope_base;
    metadata.rms_norm_eps = config.eps;
    metadata.tie_word_embeddings = config.tied_embeddings;
    metadata.multi_axis_positions = true;
    metadata.flash_next = true;
    metadata.model_type = config.experts ? "qwen4_exp" : "qwen4";
    metadata.layer_types = config.layer_types;
}


} // namespace tyloi::models::qwen4_exp

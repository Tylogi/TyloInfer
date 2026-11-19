#include "engine/models/qwen4_exp/predictor.h"
#include "engine/models/qwen4_exp/state.h"
#include "engine/models/qwen4_exp/layer.h"
#include "engine/models/common/check.h"
#include "engine/config.h"
#include "engine/ops/ops.h"
#include "engine/ops/residual.h"
#include "engine/ops/ffn.h"
#include "engine/ops/qsa.h"
#include "engine/weights/cache/workspace.h"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace tyloi::models::qwen4_exp {

PredictorState::PredictorState(const Predictor& owner, bool initialize_layers)
    : MtpState(owner), owner(owner), positions(owner.layers.size()), lengths(owner.layers.size(),0) {
    if (initialize_layers)
        for (const auto& layer : owner.layers)
            layer_states.push_back(std::make_unique<LayerState>(*layer));
}
PredictorState::~PredictorState() = default;
int64_t PredictorState::position() const noexcept {
    return lengths.empty() ? 0 : lengths.front();
}
std::vector<Tensor> PredictorState::storage() const {
    std::vector<Tensor> result;
    const auto add = [&](const Tensor& value) { if (value.defined()) result.push_back(value); };
    for (const auto& value : positions) add(value);
    for (std::size_t i = 0; i < layer_states.size(); ++i) {
        const auto& layer = *owner.layers[i];
        const auto& state = *layer_states[i];
        if (layer.qsa) for (const auto& value : layer.qsa->storage(*state.qsa)) add(value);
        if (layer.gdn) for (const auto& value : layer.gdn->storage(state.gdn)) add(value);
        if (layer.ple) {
            add(state.ple.conv); add(state.ple.saved_conv);
            for (const auto& value : state.ple.saved_prefix_conv) add(value);
        }
    }
    return result;
}
std::unique_ptr<tyloi::models::MtpState> Predictor::make_state() const {
    return std::make_unique<PredictorState>(*this);
}
Predictor::Predictor() = default;
Predictor::~Predictor() = default;
Predictor::Predictor(Predictor&&) noexcept = default;
Predictor& Predictor::operator=(Predictor&&) noexcept = default;

Tensor Predictor::embed(Tensor ids) const {
    auto normalized = ::tyloi::ops::contiguous(::tyloi::ops::copy(
        ::tyloi::ops::cast(ids, ::tyloi::kInt64), device));
    ::tyloi::ops::ProfileRange range("model.embed", normalized.device());
    auto output = ::tyloi::ops::embedding(normalized, embedding);
    range.finish();
    return output;
}

Tensor Predictor::logits(Tensor hidden) const {
    ::tyloi::ops::ProfileRange range("model.lm_head", hidden.device());
    if (draft_output) {
        auto subset = ops::project(hidden, draft_output);
        auto shape = subset.dense().sizes().vec();
        shape.back() = config.vocab;
        auto logits = ops::empty(shape, subset.options());
        ops::fill(logits, -std::numeric_limits<float>::infinity());
        ops::index_copy(logits, -1, draft_token_ids, subset);
        range.finish();
        return logits;
    }
    auto projected = ::tyloi::ops::project(hidden, output);
    range.finish();
    return ::tyloi::ops::softcap(projected, softcap);
}

Tensor Predictor::fuse(const Tensor& hidden, Tensor embeddings) const {
    const auto b = hidden.size(0), t = hidden.size(1);
    auto e = ops::project(ops::rms_norm(embeddings, embedding_norm, config.eps, 1.0, embeddings.scalar_type()),
                          embedding_fusion);
    // Release prior-stage tensors before later projections allocate.
    embeddings = {};
    auto normalized = ops::rms_norm(hidden, hidden_norm, config.eps, 1.0, hidden.scalar_type());
    auto streams = ops::project(ops::reshape(normalized, {b, t, config.streams, config.hidden}), hidden_fusion);
    return ops::reshape(ops::add(streams, ops::unsqueeze(e, -2)), {b, t, config.streams * config.hidden});
}

void Predictor::reset(tyloi::models::MtpState& value, int64_t next_batch) const {
    auto& request=state<PredictorState>(value);
    tyloi::models::require_model(next_batch > 0, "Qwen4-Exp MTP batch must be positive");
    request.release_graphs();
    for (std::size_t i=0; i<layers.size(); ++i)
        request.layer_states[i]->reset(*layers[i]);
    for (auto &pos : request.positions)
        pos = {};
    std::fill(request.lengths.begin(), request.lengths.end(), 0);
    request.batch = next_batch;
}

std::vector<std::pair<Device, std::size_t>> Predictor::growth(
        const tyloi::models::MtpState& value, const Tensor& hidden, const Tensor& ids,
        const Tensor& positions, const tyloi::models::ForwardOptions&) const {
    const auto& request = state<PredictorState>(value);
    tyloi::models::require_model(!layers.empty() && request.layer_states.size() == layers.size() &&
        ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 && hidden.dim() == 3 &&
        hidden.size(0) == ids.size(0) && hidden.size(1) == ids.size(1) &&
        hidden.size(2) == hidden_norm.numel(), "invalid Qwen4 predictor growth geometry");
    const auto b = ids.size(0), t = ids.size(1);
    const bool reset = request.batch && request.batch != b;
    const auto start = reset ? 0 : request.lengths.front();
    tyloi::models::require_model(start >= 0 && start <= config.maximum && t <= config.maximum - start,
        "predictor history exceeds context capacity");
    std::vector<std::pair<Device, std::size_t>> result;
    const auto add = [&](Device target, std::size_t bytes) {
        if (!bytes) return;
        add_session_growth(result, target, bytes);
    };
    const auto multiply = [](std::size_t a, std::size_t b) {
        if (b && a > std::numeric_limits<std::size_t>::max() / b)
            throw std::overflow_error("predictor position growth overflow");
        return a * b;
    };
    const auto current = multiply(multiply(multiply(std::size_t(3), std::size_t(b)), std::size_t(t)), sizeof(std::int32_t));
    if (hidden.device() != device) add(device, hidden.nbytes());
    if (positions.defined()) {
        if (positions.device() != device) add(device, positions.nbytes());
        if (positions.scalar_type() != kInt32) add(device, multiply(std::size_t(positions.numel()), sizeof(std::int32_t)));
    } else {
        add(device, multiply(std::size_t(t), sizeof(std::int64_t)));
        add(device, multiply(std::size_t(t), sizeof(std::int32_t)));
    }
    add(device, current);
    if (!reset && request.positions.front().defined())
        add(device, multiply(multiply(multiply(std::size_t(3), std::size_t(b)),
            std::size_t(start + t)), sizeof(std::int32_t)));
    const auto& block = *layers.front();
    const auto& layer_state = *request.layer_states.front();
    if (block.qsa) {
        const auto& qsa = *layer_state.qsa;
        const bool exit_graph = !reset && block.device.is_cuda() && block.qsa->graph_exit_growth_bytes(qsa);
        add(block.device, exit_graph ? block.qsa->exit_growth(qsa, t)
                                     : block.qsa->allocation_bytes(qsa, b, t, reset));
    }
    return result;
}

std::pair<Tensor, Tensor> Predictor::evaluate(tyloi::models::MtpState& value, const Tensor &hidden,
                                       const Tensor &ids, int64_t depth, bool cache,
                                       const Tensor &supplied_positions,
                                       const Tensor &supplied_embeddings, bool history_only) const {
    auto& request=state<PredictorState>(value);
    tyloi::models::require_model(!history_only || (cache && layers.size()==1),
        "cache-only MTP priming requires a single predictor layer");
    tyloi::models::require_model(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 &&
                          hidden.dim() == 3 && hidden.size(0) == ids.size(0) &&
                          hidden.size(1) == ids.size(1) &&
                          hidden.size(2) == hidden_norm.numel(),
                      "Qwen4-Exp MTP input geometry mismatch");
    const auto b = ids.size(0), t = ids.size(1);
    const auto capacity = config.maximum;
    const auto count = int64_t(request.lengths.size());
    tyloi::models::require_model(count > 0 && b > 0 && t > 0, "invalid predictor geometry");
    const auto layer = (depth % count + count) % count;
    if (cache && request.batch && request.batch != b)
        reset(value,b);
    const auto start = cache ? request.lengths[layer] : 0;
    tyloi::models::require_model(start >= 0 && t <= capacity - start,
                                 "predictor history exceeds context capacity");
    auto embeds =
        supplied_embeddings.defined() ? supplied_embeddings : embed(ids);
    tyloi::models::require_model(embeds.dim() == 3 && embeds.size(0) == b &&
                          embeds.size(1) == t && embeds.size(2) == config.hidden,
                      "Qwen4-Exp MTP embedding shape mismatch");
    std::array<Tensor, 2> pos;
    {
        // Full positions persist across predictor priming tiles and must
        // not occupy expert storage borrowed by this prefill phase.
        weights::Resident resident;
        auto current = supplied_positions.defined()
                           ? supplied_positions
                           : ::tyloi::ops::arange(start, start + t, device);
        current = ops::cast(ops::copy(current, device), kInt32);
        if (current.dim() == 1)
            current = ops::expand(ops::reshape(current, {1, 1, t}), {3, 1, t});
        else if (current.dim() == 2)
            current = ops::unsqueeze(current, 1);
        if (current.dim() == 3 && current.size(0) == 4)
            current = ops::slice(current, 0, 1, 4);
        tyloi::models::require_model(current.dim() == 3 && current.size(0) == 3 &&
                              current.size(-1) == t &&
                              (current.size(1) == 1 || current.size(1) == b),
                          "Qwen4 MTP positions require [T]/[3,T]/[3,B,T]");
        current = ops::contiguous(ops::expand(current, {3, b, t}));
        const std::array<Tensor, 2> segments{request.positions[layer], current};
        auto full = cache && request.positions[layer].defined() ? ops::cat(segments, -1) : current;
        pos = {current, full};
    }
    auto x = fuse(hidden, std::move(embeds));
    Tensor multi;
    {
        auto &block = *layers[layer];
        auto& qsa = *request.layer_states[layer]->qsa;
        if (block.device.is_cuda() && block.qsa->graph_exit_growth_bytes(qsa)) {
            request.release_graphs();
            block.qsa->leave_graph(qsa);
        }
        // Teacher-forced rows each receive the target hidden state. Only
        // the attention input contributes persistent predictor history;
        // attention readout, FFN and final mixer have no consumer here.
        if(history_only) {
            auto mix=block.attention_gr.pre(x);
            block.qsa->forward(*request.layer_states[layer]->qsa,mix.mixed,pos[0],pos[1],true,nullptr,{},true,true);
            multi = Tensor{};
        } else {
            auto [attention_input,attention_residual,attention_gate]=block.attention_gr.pre(x);
            auto attention_output=block.qsa->forward(*request.layer_states[layer]->qsa,attention_input,pos[0],pos[1],cache);
            x=block.attention_gr.post(attention_output,attention_residual,attention_gate);
            attention_output={};
            auto [ffn_input,ffn_residual,ffn_gate]=block.ffn_gr.pre(x);
            auto ffn_output=tyloi::models::qwen4_exp::feed_forward(block.ffn,ffn_input);
            multi = block.ffn_gr.post(ffn_output,ffn_residual,ffn_gate);
        }
    }
    x = {};
    auto result = history_only?Tensor{}:final_mixer->pre(multi).mixed;
    if (cache) {
        request.positions[layer] = pos[1];
        request.batch = b;
        request.lengths[layer] = start + t;
    }
    return {std::move(result), std::move(multi)};
}

tyloi::models::MtpStep Predictor::step(tyloi::models::MtpState& value, const tyloi::Tensor& input_hidden, const tyloi::Tensor& input_ids,
                                      const tyloi::Tensor& input_positions,
                                      const tyloi::models::ForwardOptions&) const {
    auto result = evaluate(value,input_hidden, input_ids, 0, true, input_positions);
    auto projected = draft_output ? logits(ops::contiguous(ops::slice(
        result.first, 1, result.first.size(1) - 1, result.first.size(1)))) : Tensor{};
    return {std::move(result.first), std::move(result.second), std::move(projected)};
}

std::vector<tyloi::models::MtpStep> Predictor::batch_step(
        std::span<const tyloi::models::MtpRow> rows,
        const tyloi::models::ForwardOptions&) const {
    tyloi::models::require_model(!rows.empty() && !layers.empty(),
                                 "invalid Qwen4 predictor batch");

    const auto& block = *layers.front();
    tyloi::models::require_model(block.qsa && rows.size() <=
                                     static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()),
                                 "Qwen4 predictor batch requires QSA layers");

    std::vector<PredictorState*> requests;
    std::vector<Tensor> hidden_rows, id_rows, current_positions, full_positions;
    requests.reserve(rows.size());
    hidden_rows.reserve(rows.size());
    id_rows.reserve(rows.size());
    current_positions.reserve(rows.size());
    full_positions.reserve(rows.size());

    const auto batch = static_cast<std::int64_t>(rows.size());
    for (const auto& row : rows) {
        tyloi::models::require_model(row.state && row.state->belongs_to(*this),
                                     "Qwen4 predictor batch state belongs to another model");
        auto& request = state<PredictorState>(*row.state);
        tyloi::models::require_model(request.layer_states.size() == layers.size() &&
                request.positions.size() == layers.size() && request.lengths.size() == layers.size() &&
                request.batch <= 1 && row.hidden.dim() == 3 && row.hidden.size(0) == 1 &&
                row.hidden.size(1) == 1 && row.hidden.size(2) == hidden_norm.numel() &&
                row.ids.dim() == 2 && row.ids.size(0) == 1 && row.ids.size(1) == 1,
            "invalid Qwen4 predictor batch row geometry");
        tyloi::models::require_model(request.layer_states.front() &&
                request.layer_states.front()->qsa,
            "Qwen4 predictor batch state has no QSA cache");
        tyloi::models::require_model(request.lengths.front() >= 0 &&
                request.lengths.front() <= config.maximum - 1,
            "Qwen4 predictor batch history exceeds context capacity");

        requests.push_back(&request);
        hidden_rows.push_back(ops::copy(row.hidden, device));
        id_rows.push_back(ops::contiguous(ops::copy(ops::cast(row.ids, ::tyloi::kInt64), device)));

        auto current = row.positions.defined()
            ? row.positions
            : ::tyloi::ops::arange(request.lengths.front(), request.lengths.front() + 1, device);
        current = ops::cast(ops::copy(current, device), kInt32);
        if (current.dim() == 1)
            current = ops::expand(ops::reshape(current, {1, 1, 1}), {3, 1, 1});
        else if (current.dim() == 2)
            current = ops::unsqueeze(current, 1);
        if (current.dim() == 3 && current.size(0) == 4)
            current = ops::slice(current, 0, 1, 4);
        tyloi::models::require_model(current.dim() == 3 && current.size(0) == 3 &&
                current.size(1) == 1 && current.size(2) == 1,
            "Qwen4 predictor batch positions require [T], [3,T] or [3,1,T]");
        current = ops::contiguous(ops::expand(current, {3, 1, 1}));
        current_positions.push_back(current);
        full_positions.push_back(request.positions.front().defined()
            ? ops::cat(std::array<Tensor, 2>{request.positions.front(), current}, -1)
            : current);
    }

    auto ids = ops::cat(id_rows, 0);
    auto hidden = ops::cat(hidden_rows, 0);
    auto embeds = embed(ids);
    auto e = ops::project(
        ops::rms_norm(embeds, embedding_norm, config.eps, 1.0, embeds.scalar_type()),
        embedding_fusion);
    embeds = {};
    auto normalized = ops::rms_norm(hidden, hidden_norm, config.eps, 1.0, hidden.scalar_type());
    auto streams = ops::project(
        ops::reshape(normalized, {batch, 1, config.streams, config.hidden}), hidden_fusion);
    auto x = ops::reshape(ops::add(streams, ops::unsqueeze(e, -2)),
                          {batch, 1, config.streams * config.hidden});

    auto [attention_input, attention_residual, attention_gate] = block.attention_gr.pre(x);
    auto projections = block.qsa->project_inputs(attention_input);
    std::vector<Tensor> attended;
    attended.reserve(rows.size());
    for (std::size_t index = 0; index < rows.size(); ++index) {
        std::vector<Tensor> row_projections;
        row_projections.reserve(projections.size());
        for (const auto& projection : projections)
            row_projections.push_back(ops::slice(projection, 0,
                static_cast<std::int64_t>(index), static_cast<std::int64_t>(index + 1)));
        auto& qsa = *requests[index]->layer_states.front()->qsa;
        if (block.device.is_cuda() && block.qsa->graph_exit_growth_bytes(qsa)) {
            requests[index]->release_graphs();
            block.qsa->leave_graph(qsa);
        }
        attended.push_back(block.qsa->forward(
            *requests[index]->layer_states.front()->qsa,
            ops::slice(attention_input, 0, static_cast<std::int64_t>(index),
                       static_cast<std::int64_t>(index + 1)),
            current_positions[index], full_positions[index], true, nullptr,
            std::move(row_projections), false));
    }
    auto attention = block.qsa->project_output(ops::cat(attended, 0));
    x = block.attention_gr.post(attention, attention_residual, attention_gate);
    auto [ffn_input, ffn_residual, ffn_gate] = block.ffn_gr.pre(x);
    auto ffn_output = tyloi::models::qwen4_exp::feed_forward(block.ffn, ffn_input);
    auto multi = block.ffn_gr.post(ffn_output, ffn_residual, ffn_gate);
    auto sample = final_mixer->pre(multi).mixed;
    auto projected = draft_output ? logits(sample) : Tensor{};

    std::vector<tyloi::models::MtpStep> result;
    result.reserve(rows.size());
    for (std::size_t index = 0; index < rows.size(); ++index) {
        requests[index]->positions.front() = std::move(full_positions[index]);
        requests[index]->lengths.front() += 1;
        requests[index]->batch = 1;
        result.push_back({
            ops::slice(sample, 0, static_cast<std::int64_t>(index),
                       static_cast<std::int64_t>(index + 1)),
            ops::slice(multi, 0, static_cast<std::int64_t>(index),
                       static_cast<std::int64_t>(index + 1)),
            projected ? ops::slice(projected, 0, static_cast<std::int64_t>(index),
                                   static_cast<std::int64_t>(index + 1)) : Tensor{}});
    }
    return result;
}

void Predictor::trim_cache_to(tyloi::models::MtpState& value, int64_t position) const {
    auto& request=state<PredictorState>(value);
    tyloi::models::require_model(position >= 0 && position <= request.position(),
                      "Qwen4-Exp MTP cache trim position is invalid");
    constexpr size_t index = 0;
    if (index < layers.size() && layers[index]->qsa)
        layers[index]->qsa->truncate(*request.layer_states[index]->qsa,position);
    if (request.positions[index].defined())
        request.positions[index] = ops::slice(request.positions[index],-1,0,position);
    request.lengths[index] = position;
}

bool Predictor::teacher_forced_prompt_prime() const noexcept { return true; }

bool Predictor::target_bootstrap_decode() const noexcept { return true; }

bool Predictor::preserve_output_dtype() const noexcept { return true; }
bool Predictor::stream_prompt_history(std::int64_t tokens) const {
    const auto* option=tyloi::engine::environment_value("TYLOINFER_MTP_STREAM_PROMPT_HISTORY");
    return tokens>16384 && (!option || option[0]!='0');
}
bool Predictor::host_prompt_history(std::int64_t tokens, std::size_t available,
                                    std::size_t raw_row_bytes) const {
    const auto* option=tyloi::engine::environment_value("TYLOINFER_MTP_HOST_PROMPT_HISTORY");
    return option ? option[0]!='0' : tyloi::models::Mtp::host_prompt_history(tokens,available,raw_row_bytes);
}
bool Predictor::bootstrap_before_publication() const {
    const auto* option=tyloi::engine::environment_value("TYLOINFER_MTP_EARLY_BOOTSTRAP");
    return !option || option[0]!='0';
}
std::size_t Predictor::predictor_history_flush_threshold() const {
    const auto* option=tyloi::engine::environment_value("TYLOINFER_MTP_HISTORY_EARLY_FLUSH");
    return !option || option[0]!='0' ? 4 : 512;
}
std::size_t Predictor::predictor_parked_history_flush_threshold() const {
    const auto* option=tyloi::engine::environment_value("TYLOINFER_MTP_HISTORY_EARLY_FLUSH");
    return !option || option[0]!='0' ? 16 : 512;
}
std::size_t Predictor::prefill_workspace_bytes(std::int64_t tokens) const {
    const auto* option=tyloi::engine::environment_value("TYLOINFER_MTP_CACHE_ONLY_PRIME");
    if (layers.size()!=1 || (option && option[0]=='0') || tokens<=128) return 0;
    return (std::size_t(64)<<20)+std::size_t(tokens)*config.hidden*config.streams*sizeof(float)*12;
}
tyloi::models::MtpStep Predictor::prefill(tyloi::models::MtpState& value,
        const tyloi::Tensor& hidden, const tyloi::Tensor& ids, const tyloi::Tensor& positions,
        const tyloi::models::ForwardOptions& options) const {
    const auto* option=tyloi::engine::environment_value("TYLOINFER_MTP_CACHE_ONLY_PRIME");
    if (layers.size()!=1 || (option && option[0]=='0'))
        return step(value,hidden,ids,positions,options);
    (void)evaluate(value,hidden,ids,0,true,positions,{},true);
    return {};
}

std::unique_ptr<tyloi::engine::mtp::DepthPolicy> Predictor::depth_policy(
        int maximum_depth, tyloi::engine::mtp::PolicyState& state) const {
    const auto* option = tyloi::engine::environment_value("TYLOINFER_MTP_METAL_POLICY");
    const bool batch = !option || option[0] != '0';
    return std::make_unique<tyloi::engine::mtp::AdaptivePolicy>(maximum_depth, &state, true, batch);
}

bool Predictor::supports_session_state(const tyloi::models::MtpState& value) const noexcept {
    if (!value.belongs_to(*this) || layers.empty()) return false;
    const auto& request = static_cast<const PredictorState&>(value);
    if (request.layer_states.size() != layers.size()) return false;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (!request.layer_states[i] ||
            (layers[i]->qsa && (!request.layer_states[i]->qsa ||
                !layers[i]->qsa->supports_session(*request.layer_states[i]->qsa)))) return false;
    }
    return true;
}

MtpSessionState Predictor::capture_session_state(const tyloi::models::MtpState& value, int64_t target_position,const tyloi::Tensor& input_hidden) const {
    const auto& request=this->state<PredictorState>(value);
    const auto& hidden=input_hidden;
    const auto position=target_position-1;
    tyloi::models::require_model(supports_session_state(value) && position>0 && position<=request.position() &&
        hidden.defined() && hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==1 && hidden.size(2)==hidden_norm.numel(),
        "Qwen4 MTP session boundary is unavailable");
    MtpSessionState state;state.cache_pos=position;
    auto& snapshot=state.payload.emplace<tyloi::models::qwen4_exp::MtpSnapshot>();
    for(std::size_t i=0;i<layers.size();++i) {
        const auto count=std::min(position,request.lengths[i]);
        snapshot.lengths.push_back(count);
        snapshot.layers.push_back(count?capture_layer_session(*layers[i],*request.layer_states[i],count,state.bytes):tyloi::models::qwen4_exp::LayerSnapshot{});
        ::tyloi::Tensor pos;
        if(count) {
            pos=ops::clone(ops::slice(request.positions[i],-1,0,count));
            state.bytes+=session_tensor_bytes(pos);
        }
        snapshot.positions.push_back(std::move(pos));
    }
    auto saved_hidden=ops::clone(hidden);state.bytes+=session_tensor_bytes(saved_hidden);
    state.last_target_hidden=std::move(saved_hidden);
    return state;
}

} // namespace tyloi::models::qwen4_exp

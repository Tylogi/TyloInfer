#include "engine/models/qwen4_exp/state.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/qwen4_exp/snapshot.h"
#include "engine/models/qwen4_exp/predictor.h"
#include "engine/models/session/state.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace tyloi::cuda::qwen4_exp {

LayerState::LayerState(const Layer& layer) {
    if (layer.qsa)
        qsa = std::make_unique<tyloi::ops::Qsa::State>(*layer.qsa);
}

void LayerState::reset(const Layer& layer) {
    if (layer.gdn)
        layer.gdn->reset(gdn, retain_graph_storage);
    if (layer.qsa)
        layer.qsa->reset(*qsa);
    if (layer.ple)
        layer.ple->reset(ple, retain_graph_storage);
}

void LayerState::commit(const Layer& layer) {
    if (layer.gdn)
        layer.gdn->commit(gdn);
    if (layer.ple)
        layer.ple->commit(ple);
}

void LayerState::rollback(const Layer& layer, std::int64_t keep) {
    if (layer.gdn)
        layer.gdn->rollback(gdn, keep);
    if (layer.qsa)
        layer.qsa->truncate(*qsa, keep);
    if (layer.ple)
        layer.ple->rollback(ple, keep);
}

std::vector<::tyloi::Tensor*> LayerState::graph_state(const Layer& layer) {
    std::vector<::tyloi::Tensor*> result;
    if (layer.gdn)
        for (auto* value : layer.gdn->graph_state(gdn))
            if (value->defined()) result.push_back(value);
    if (layer.ple)
        for (auto* value : layer.ple->graph_state(ple))
            if (value->defined()) result.push_back(value);
    return result;
}

tyloi::models::qwen4_exp::LayerSnapshot capture_layer_session(
        const Layer& layer, const LayerState& state, std::int64_t position, std::size_t& bytes) {
    tyloi::models::qwen4_exp::LayerSnapshot snapshot;
    const auto save = [&](const DenseTensor& tensor) {
        TYLOINFER_RUNTIME_CHECK(tensor.defined(), "Qwen4 recurrent session state is unavailable");
        auto saved = std::make_shared<DenseTensor>(tensor.clone());
        bytes += session_tensor_bytes(*saved);
        return saved;
    };
    if (layer.qsa) {
        tyloi::ops::Qsa::Snapshot attention;
        layer.qsa->capture_session(*state.qsa, attention, position, bytes);
        snapshot.keys = std::move(attention.keys);
        snapshot.values = std::move(attention.values);
        snapshot.index = std::move(attention.index);
    }
    if (layer.gdn) {
        snapshot.convolution = save(state.gdn.conv.dense());
        snapshot.recurrent = save(state.gdn.recurrent.dense());
    }
    if (layer.ple) {
        snapshot.ple_convolution = save(layer.ple->conv_state(state.ple).dense());
        const auto& context = layer.ple->session_context(state.ple);
        snapshot.ple_batch = context.batch;
        snapshot.ple_tokens = context.tokens;
        bytes += snapshot.ple_tokens.size() * sizeof(std::int64_t);
    }
    return snapshot;
}

void add_session_growth(std::vector<std::pair<tyloi::Device, std::size_t>>& growth,
        tyloi::Device device, std::size_t bytes) {
    if (!bytes) return;
    auto entry = std::find_if(growth.begin(), growth.end(),
        [device](const auto& value) { return value.first == device; });
    if (entry == growth.end()) growth.emplace_back(device, bytes);
    else {
        if (entry->second > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::overflow_error("Qwen4 session allocation size overflow");
        entry->second += bytes;
    }
}

void stage_session_tensor(::tyloi::Tensor& target, const ::tyloi::Tensor& source, tyloi::Device device) {
    target = tyloi::ops::empty(source.dense().sizes(), source.options().device(device));
    tyloi::ops::copy_into(target, source);
}

static bool session_device_matches(const ::tyloi::Tensor& value, int original, tyloi::Device device) {
    return original == -1 ? value.device() == device :
        original >= 0 && value.device() == tyloi::Device(tyloi::kCPU) && device == tyloi::Device(tyloi::kCUDA, original);
}

void validate_layer_session(const Layer& layer, const LayerState& current,
        const tyloi::models::qwen4_exp::LayerSnapshot& saved, std::int64_t position) {
    const auto device = layer.device;
    const auto& c = layer.config;
    const auto tensor = [&](const ::tyloi::Tensor& value, int original, const std::vector<int64_t>& shape) {
        return value.defined() && value.representation<DenseTensor>() && session_device_matches(value, original, device) &&
            value.scalar_type() == tyloi::kFloat32 && value.dense().sizes().vec() == shape;
    };
    if (layer.qsa) {
        try {
            if (!session_device_matches(saved.keys, saved.devices[0], device) ||
                !session_device_matches(saved.values, saved.devices[1], device) ||
                !session_device_matches(saved.index, saved.devices[2], device) ||
                saved.keys.device() != saved.values.device() || saved.keys.device() != saved.index.device())
                throw CudaSessionStateError("incompatible Qwen4 QSA session device");
            layer.qsa->validate_session(*current.qsa, {saved.keys, saved.values, saved.index},
                position, saved.keys.device());
        } catch (const std::exception&) {
            throw CudaSessionStateError("incompatible Qwen4 QSA session state");
        }
    } else if (saved.keys || saved.values || saved.index) {
        throw CudaSessionStateError("unexpected Qwen4 QSA session state");
    }
    if (layer.gdn) {
        if (!tensor(saved.convolution, saved.devices[3], {1, c.kernel - 1, (2 * c.key_heads + c.value_heads) * c.linear_width}) ||
            !tensor(saved.recurrent, saved.devices[4], {1, c.value_heads, c.linear_width, c.linear_width}))
            throw CudaSessionStateError("incompatible Qwen4 GDN session state");
    } else if (saved.convolution || saved.recurrent) {
        throw CudaSessionStateError("unexpected Qwen4 GDN session state");
    }
    if (layer.ple) {
        if (!tensor(saved.ple_convolution, saved.devices[5], {1, (c.ple_kernel - 1) * c.ngram, c.streams * c.hidden}) ||
            saved.ple_batch != 1 || saved.ple_tokens.size() != std::size_t(c.ngram - 1))
            throw CudaSessionStateError("incompatible Qwen4 PLE session state");
    } else if (saved.ple_convolution || saved.ple_batch || !saved.ple_tokens.empty()) {
        throw CudaSessionStateError("unexpected Qwen4 PLE session state");
    }
}

std::size_t layer_session_growth(const Layer& layer, const LayerState& current,
        const tyloi::models::qwen4_exp::LayerSnapshot& saved, std::int64_t position) {
    std::size_t bytes = layer.qsa ? layer.qsa->fresh_session_growth(*current.qsa, position) : 0;
    for (const auto* tensor : {&saved.convolution, &saved.recurrent, &saved.ple_convolution}) {
        const auto next = session_tensor_bytes(*tensor);
        if (bytes > std::numeric_limits<std::size_t>::max() - next)
            throw std::overflow_error("Qwen4 session allocation size overflow");
        bytes += next;
    }
    return bytes;
}

void stage_layer_session(const Layer& layer, LayerState& target,
        const tyloi::models::qwen4_exp::LayerSnapshot& saved, std::int64_t position) {
    const auto device = layer.device;
    tyloi::ops::DeviceScope scope(device);
    if (layer.qsa)
        layer.qsa->stage_session(*target.qsa, {saved.keys, saved.values, saved.index}, position, device);
    if (layer.gdn) {
        stage_session_tensor(target.gdn.conv, saved.convolution, device);
        stage_session_tensor(target.gdn.recurrent, saved.recurrent, device);
    }
    if (layer.ple) {
        stage_session_tensor(target.ple.conv, saved.ple_convolution, device);
        layer.ple->restore_graph_context(target.ple, {saved.ple_batch, saved.ple_tokens});
    }
}

tyloi::models::SnapshotSize snapshot_size(const State& state,const tyloi::models::Mtp* mtp) {
    const auto& model = state.model_;
    std::size_t bytes=session_tensor_bytes(state.positions);
    const auto count_block=[&](const Layer& block,const LayerState& layer,int64_t count) {
        if(count<=0)return;
        if(block.qsa)bytes+=block.qsa->session_bytes(*layer.qsa,count);
        if(block.gdn) {
            bytes+=session_tensor_bytes(layer.gdn.conv);
            bytes+=session_tensor_bytes(layer.gdn.recurrent);
        }
        if(block.ple)bytes+=session_tensor_bytes(block.ple->conv_state(layer.ple));
    };
    for(std::size_t i=0;i<model.layers.size();++i)count_block(*model.layers[i],*state.layers[i],state.cache_pos);
    if(const auto* predictor=dynamic_cast<const Predictor*>(mtp); predictor && state.predictor && state.predictor->belongs_to(*predictor)) {
        const auto& request=static_cast<const PredictorState&>(*state.predictor);
        for(std::size_t i=0;i<predictor->layers.size();++i) {
            const auto count=std::min(state.cache_pos-1,request.lengths[i]);
            count_block(*predictor->layers[i],*request.layer_states[i],count);
            if(count>0)bytes+=std::size_t(count)*3*request.positions[i].element_size();
        }
        bytes+=std::size_t(model.config.hidden)*model.config.streams*sizeof(float);
    }
    // Transposed graph KV can temporarily require a contiguous copy of the checkpoint.
    return {bytes, bytes * 2};
}

std::vector<std::pair<tyloi::Device, std::size_t>> State::graph_exit_growth() const {
    std::vector<std::pair<tyloi::Device, std::size_t>> growth;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto& block = *model_.layers[i];
        if (!block.qsa) continue;
        const auto bytes = block.qsa->graph_exit_growth_bytes(*layers[i]->qsa);
        if (!bytes) continue;
        const auto device = block.device;
        auto entry = std::find_if(growth.begin(), growth.end(),
            [device](const auto& value) { return value.first == device; });
        if (entry == growth.end()) growth.emplace_back(device, bytes);
        else {
            if (entry->second > std::numeric_limits<std::size_t>::max() - bytes)
                throw std::overflow_error("Qwen4 graph exit size overflow");
            entry->second += bytes;
        }
    }
    return growth;
}

std::vector<std::pair<tyloi::Device, std::size_t>> State::growth(
        const tyloi::models::Input& input, const tyloi::models::ForwardOptions& options) const {
    const auto [input_batch, tokens] = input.shape();
    if (batch <= 0 || input_batch != batch || input.cache_positions ||
        (input.suffix && (input.confirmed != 0 || batch != 1 || cache_pos <= 0 ||
                          !model_.supports_suffix_speculation())) ||
        input.confirmed < 0 || (input.confirmed > 0 &&
            (input.confirmed >= tokens || batch != 1 || cache_pos <= 0 || !model_.supports_speculation())) ||
        tokens <= 0 || cache_pos < 0 || cache_pos > model_.max_position_embeddings() ||
        tokens > model_.max_position_embeddings() - cache_pos || speculative_start >= 0 ||
        speculative_suffix_forward || options.primary < 0 || layers.size() != model_.layers.size())
        throw std::invalid_argument("invalid Qwen4 ordinary state growth");
    struct Growth { tyloi::Device device; std::size_t retained = 0, peak = 0; };
    std::vector<Growth> growth;
    const auto add = [&](tyloi::Device device, std::size_t retained, std::size_t temporary = 0) {
        if (device.is_cuda() && device.index < 0) throw std::invalid_argument("invalid Qwen4 state device");
        auto entry = std::find_if(growth.begin(), growth.end(),
            [device](const auto& value) { return value.device == device; });
        if (entry == growth.end()) { growth.push_back({device}); entry = std::prev(growth.end()); }
        const auto checked = [](std::size_t a, std::size_t b) {
            if (a > std::numeric_limits<std::size_t>::max() - b)
                throw std::overflow_error("Qwen4 state growth overflow");
            return a + b;
        };
        entry->retained = checked(entry->retained, retained);
        entry->peak = std::max(entry->peak, checked(entry->retained, temporary));
    };
    if (input.suffix) {
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& block = *model_.layers[i];
            const auto& layer = *layers[i];
            const auto device = block.device;
            if (block.gdn) {
                TYLOINFER_RUNTIME_CHECK(!layer.gdn.saved_conv.defined() && layer.gdn.conv.dense().defined() &&
                    layer.gdn.recurrent.dense().defined(), "GDN suffix requires committed recurrent state");
                add(device, layer.gdn.conv.dense().nbytes());
                add(device, layer.gdn.recurrent.dense().nbytes());
            }
            if (block.ple) {
                TYLOINFER_RUNTIME_CHECK(!layer.ple.saved && layer.ple.conv.defined(), "PLE suffix requires committed convolution state");
                add(device, layer.ple.conv.dense().nbytes());
                const auto& context = block.ple->session_context(layer.ple);
                if (context.tokens.size() > std::numeric_limits<std::size_t>::max() / sizeof(int64_t))
                    throw std::overflow_error("PLE suffix context overflow");
                add(tyloi::Device(tyloi::kCPU), context.tokens.size() * sizeof(int64_t));
            }
        }
    }
    std::vector<int64_t> shape;
    if (positions) {
        const auto& dense = positions.dense();
        if (dense.dim() < 1 || dense.dim() > 3 || dense.size(-1) != cache_pos)
            throw std::invalid_argument("invalid Qwen4 position history");
        shape = dense.sizes().vec();
        shape.back() = cache_pos + tokens;
    } else if (input.positions) {
        shape = input.positions.dense().sizes().vec();
        if (shape.empty() || shape.size() > 3 || shape.back() != tokens ||
            (shape.size() > 1 && shape.front() != 3 && shape.front() != 4) ||
            (shape.size() == 3 && shape[1] != batch))
            throw std::invalid_argument("invalid Qwen4 input positions");
        if (shape.size() > 1 && shape.front() == 4) shape.front() = 3;
    } else {
        shape = {tokens};
    }
    if (positions) {
        std::vector<int64_t> current_shape = input.positions ? input.positions.dense().sizes().vec() : std::vector<int64_t>{tokens};
        if (current_shape.size() > 1 && current_shape.front() == 4) current_shape.front() = 3;
        if (current_shape.size() != shape.size() || current_shape.empty() || current_shape.back() != tokens ||
            !std::equal(current_shape.begin(), current_shape.end() - 1, shape.begin()))
            throw std::invalid_argument("Qwen4 position history geometry changed");
    }
    const tyloi::Device primary(tyloi::kCUDA, options.primary);
    add(primary, tyloi::make_contiguous_view(nullptr, shape, tyloi::kInt64, primary).nbytes());
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto& block = *model_.layers[i];
        const auto& state = *layers[i];
        if ((block.gdn && state.gdn.saved_conv.defined()) || (block.ple && state.ple.saved))
            throw std::invalid_argument("pending Qwen4 speculative state");
        const auto device = block.device;
        if (block.ple) {
            if (input.suffix) {
                const auto [retained, temporary] = block.ple->context_growth_bytes(state.ple, batch, tokens);
                add(tyloi::Device(tyloi::kCPU), retained, temporary);
            }
            const auto [retained, temporary] = block.ple->ordinary_growth_bytes(state.ple, batch, tokens);
            if (input.confirmed > 0) add(device, block.ple->checkpoint_growth_bytes(state.ple, batch, tokens, input.confirmed));
            add(device, retained, temporary);
        }
        if (block.qsa) add(device, block.qsa->allocation_bytes(*state.qsa, batch, tokens, false));
        if (block.gdn) {
            const auto [retained, temporary] = block.gdn->ordinary_growth_bytes(state.gdn, batch, tokens);
            if (input.confirmed > 0) add(device, block.gdn->checkpoint_growth_bytes(state.gdn, batch, tokens, input.confirmed));
            add(device, retained, temporary);
        }
    }
    std::vector<std::pair<tyloi::Device, std::size_t>> result;
    for (const auto& value : growth) result.emplace_back(value.device, value.peak);
    return result;
}

std::vector<::tyloi::Tensor> State::storage() const {
    std::vector<::tyloi::Tensor> result;
    const auto add=[&](const ::tyloi::Tensor& value) { if(value.defined())result.push_back(value); };
    add(positions);
    for (std::size_t i=0;i<model_.layers.size();++i) {
        const auto& block=*model_.layers[i];
        const auto& layer=*layers[i];
        if(block.gdn)
            for(const auto& value:block.gdn->storage(layer.gdn))add(value);
        if(block.ple) {
            const auto& ple=layer.ple;
            add(ple.conv);add(ple.saved_conv);
            for(const auto& value:ple.saved_prefix_conv)add(value);
        }
        if(block.qsa)
            for(const auto& value:block.qsa->storage(*layer.qsa))add(value);
    }
    return result;
}

std::vector<tyloi::Tensor*> State::graph_state() {
    std::vector<tyloi::Tensor*> warmup_states;
    for (std::size_t i=0; i<model_.layers.size(); ++i)
        for (auto* tensor : layers[i]->graph_state(*model_.layers[i]))
            warmup_states.push_back(tensor);
    return warmup_states;
}

void State::begin_speculative_suffix(std::int64_t draft_tokens) {
    tyloi::models::require_model(model_.supports_suffix_speculation() && draft_tokens > 0 && batch == 1 &&
                                     cache_pos > 0 && cache_pos <= model_.max_position_embeddings() &&
                                     draft_tokens <= model_.max_position_embeddings() - cache_pos &&
                                     speculative_start < 0 && !speculative_suffix_forward,
                                 "invalid speculative suffix");
    struct Checkpoint {
        DenseTensor conv, recurrent, ple_conv;
        NgramEmbedding::Context context;
    };
    std::vector<Checkpoint> saved(layers.size());
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto& block = *model_.layers[i];
        const auto& layer = *layers[i];
        ::tyloi::ops::DeviceScope guard(block.device);
        if (block.gdn) {
            TYLOINFER_RUNTIME_CHECK(!layer.gdn.saved_conv.defined() && layer.gdn.conv.dense().defined() &&
                layer.gdn.recurrent.dense().defined(), "GDN suffix requires committed recurrent state");
            saved[i].conv = layer.gdn.conv.dense().clone();
            saved[i].recurrent = layer.gdn.recurrent.dense().clone();
        }
        if (block.ple) {
            TYLOINFER_RUNTIME_CHECK(!layer.ple.saved && layer.ple.conv.defined(), "PLE suffix requires committed convolution state");
            saved[i].ple_conv = layer.ple.conv.dense().clone();
            saved[i].context = block.ple->session_context(layer.ple);
        }
    }
    for (std::size_t i = 0; i < layers.size(); ++i) {
        auto& layer = *layers[i];
        if (model_.layers[i]->gdn) {
            layer.gdn.saved_conv = std::move(saved[i].conv);
            layer.gdn.saved_state = std::move(saved[i].recurrent);
        }
        if (model_.layers[i]->ple) {
            layer.ple.saved_conv = std::move(saved[i].ple_conv);
            layer.ple.saved_context = std::move(saved[i].context);
            layer.ple.saved = true;
        }
    }
    speculative_prefix_retention = false;
    speculative_suffix_journal = true;
    speculative_start = cache_pos;
    speculative_confirmed = 0;
}

void State::commit_speculative() {
    tyloi::models::require_model(speculative_start >= 0, "no speculative transaction");
    for (std::size_t i=0; i<model_.layers.size(); ++i)
        layers[i]->commit(*model_.layers[i]);
    speculative_prefix_retention = false;
    speculative_suffix_journal = false;
    speculative_start = -1;
    speculative_confirmed = 0;
}

TextSessionStateKind State::text_session_state_kind() const {
    if (model_.layers.empty() || layers.size() != model_.layers.size())
        return TextSessionStateKind::Unsupported;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (!layers[i] || (model_.layers[i]->qsa && (!layers[i]->qsa ||
                !model_.layers[i]->qsa->supports_session(*layers[i]->qsa))))
            return TextSessionStateKind::Unsupported;
    }
    return TextSessionStateKind::Qwen4;
}

bool State::supports_text_session_state() const {
    return text_session_state_kind() != TextSessionStateKind::Unsupported;
}

bool State::supports_paged_text_session_state() const {
    return text_session_state_kind() == TextSessionStateKind::Qwen4;
}

TextSessionState State::capture_text_session_state(const std::vector<std::int64_t>& tokens) const {
    TYLOINFER_RUNTIME_CHECK(text_session_state_kind() == TextSessionStateKind::Qwen4 && batch == 1 &&
                                cache_pos > 0 && std::size_t(cache_pos) == tokens.size() &&
                                speculative_start < 0,
                            "Qwen4 text session boundary is unavailable");
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = cache_pos;
    state.decode_position_delta = decode_position_delta;
    tyloi::models::qwen4_exp::Snapshot snapshot;
    for (std::size_t i=0; i<model_.layers.size(); ++i)
        snapshot.layers.push_back(capture_layer_session(*model_.layers[i], *layers[i], cache_pos, state.bytes));
    snapshot.positions = std::make_shared<DenseTensor>(positions.dense().clone());
    state.bytes += session_tensor_bytes(snapshot.positions);
    state.payload = std::move(snapshot);
    return state;
}

namespace {
class SessionRestore final : public tyloi::models::Restore {
public:
    SessionRestore(State& state, const Model& model, const TextSessionState& snapshot,
                   const tyloi::models::Mtp* predictor_model)
        : state_(state), model_(model), snapshot_(snapshot) {
        const auto* saved = std::get_if<tyloi::models::qwen4_exp::Snapshot>(&snapshot_.payload);
        const auto* positions = saved ? saved->positions.representation<DenseTensor>() : nullptr;
        if (!state_.supports_text_session_state() || state_.speculative_start >= 0 ||
            state_.speculative_suffix_forward || !saved || saved->layers.size() != model_.layers.size() ||
            snapshot_.cache_pos <= 0 || snapshot_.cache_pos > model_.config.maximum ||
            snapshot_.tokens.size() != std::size_t(snapshot_.cache_pos) || !positions || !positions->defined() ||
            !session_device_matches(saved->positions, saved->positions_device, tyloi::Device(tyloi::kCUDA, model_.primary_device)) ||
            saved->positions.scalar_type() != tyloi::kInt64 || positions->dim() < 1 || positions->dim() > 3 ||
            positions->size(-1) != snapshot_.cache_pos ||
            (positions->dim() > 1 && positions->size(0) != 3) ||
            (positions->dim() == 3 && positions->size(1) != 1))
            throw CudaSessionStateError("incompatible Qwen4 text session state");
        for (std::size_t i = 0; i < saved->layers.size(); ++i)
            validate_layer_session(*model_.layers[i], *state_.layers[i], saved->layers[i], snapshot_.cache_pos);
        if (predictor_model) {
            if (!state_.predictor || !state_.predictor->belongs_to(*predictor_model) ||
                !snapshot_.mtp || snapshot_.mtp->cache_pos != snapshot_.cache_pos - 1)
                throw CudaSessionStateError("incompatible Qwen4 predictor session boundary");
            predictor_ = state_.predictor->prepare_restore(*snapshot_.mtp);
        }
    }

    std::vector<std::pair<tyloi::Device, std::size_t>> growth() const override {
        std::vector<std::pair<tyloi::Device, std::size_t>> result;
        const auto& saved = std::get<tyloi::models::qwen4_exp::Snapshot>(snapshot_.payload);
        add_session_growth(result, tyloi::Device(tyloi::kCUDA, model_.primary_device), session_tensor_bytes(saved.positions));
        for (std::size_t i = 0; i < saved.layers.size(); ++i) {
            const auto& layer = *model_.layers[i];
            const auto device = layer.device;
            add_session_growth(result, device,
                layer_session_growth(layer, *state_.layers[i], saved.layers[i], snapshot_.cache_pos));
            add_session_growth(result, tyloi::Device(tyloi::kCPU), saved.layers[i].ple_tokens.size() * sizeof(int64_t));
        }
        if (predictor_) for (const auto& [device, bytes] : predictor_->growth())
            add_session_growth(result, device, bytes);
        return result;
    }

    void stage() override {
        candidate_ = std::make_unique<State>(model_, false);
        candidate_->layers.reserve(model_.layers.size());
        for (const auto& layer : model_.layers)
            candidate_->layers.push_back(std::make_unique<LayerState>(*layer));
        const auto& saved = std::get<tyloi::models::qwen4_exp::Snapshot>(snapshot_.payload);
        for (std::size_t i = 0; i < saved.layers.size(); ++i)
            stage_layer_session(*model_.layers[i], *candidate_->layers[i], saved.layers[i], snapshot_.cache_pos);
        stage_session_tensor(candidate_->positions, saved.positions, tyloi::Device(tyloi::kCUDA, model_.primary_device));
        candidate_->cache_pos = snapshot_.cache_pos;
        candidate_->decode_position_delta = snapshot_.decode_position_delta;
        candidate_->batch = 1;
        if (predictor_) predictor_->stage();
    }

    void publish() noexcept override {
        state_.publish_restore(*candidate_);
        if (predictor_) predictor_->publish();
    }
private:
    State& state_;
    const Model& model_;
    TextSessionState snapshot_;
    std::unique_ptr<State> candidate_;
    std::unique_ptr<tyloi::models::Restore> predictor_;
};
} // namespace

std::unique_ptr<tyloi::models::Restore> State::prepare_restore(
        const TextSessionState& snapshot, const tyloi::models::Mtp* predictor_model) {
    return std::make_unique<SessionRestore>(*this, model_, snapshot, predictor_model);
}

namespace {
class PredictorRestore final : public tyloi::models::Restore {
public:
    PredictorRestore(PredictorState& state, const MtpSessionState& snapshot)
        : state_(state), model_(state.owner), snapshot_(snapshot) {
        const auto* saved = std::get_if<tyloi::models::qwen4_exp::MtpSnapshot>(&snapshot_.payload);
        const auto& hidden = snapshot_.last_target_hidden;
        if (!model_.supports_session_state(state_) || snapshot_.cache_pos <= 0 ||
            snapshot_.cache_pos > model_.config.maximum || !saved ||
            saved->layers.size() != model_.layers.size() || saved->positions.size() != model_.layers.size() ||
            saved->lengths.size() != model_.layers.size() || saved->lengths.front() != snapshot_.cache_pos ||
            (!saved->positions_devices.empty() && saved->positions_devices.size() != saved->positions.size()) ||
            !hidden.defined() || !hidden.representation<DenseTensor>() || hidden.dim() != 3 ||
            !session_device_matches(hidden, saved->hidden_device, tyloi::Device(tyloi::kCUDA, model_.primary)) ||
            hidden.size(0) != 1 || hidden.size(1) != 1 || hidden.size(2) != model_.hidden_norm.numel() ||
            (hidden.scalar_type() != tyloi::kFloat16 && hidden.scalar_type() != tyloi::kBFloat16 &&
             hidden.scalar_type() != tyloi::kFloat32))
            throw CudaSessionStateError("incompatible Qwen4 predictor session state");
        for (std::size_t i = 0; i < saved->layers.size(); ++i) {
            const auto length = saved->lengths[i];
            const auto& position = saved->positions[i];
            if (length < 0 || length > snapshot_.cache_pos ||
                (length == 0 && position.defined()) ||
                (length > 0 && (!position.defined() || !position.representation<DenseTensor>() ||
                    !session_device_matches(position, saved->positions_devices.empty() ? -1 : saved->positions_devices[i],
                        tyloi::Device(tyloi::kCUDA, model_.primary)) ||
                    position.scalar_type() != tyloi::kInt32 || position.dim() != 3 ||
                    position.size(0) != 3 || position.size(1) != 1 || position.size(2) != length)))
                throw CudaSessionStateError("incompatible Qwen4 predictor position history");
            if (length)
                validate_layer_session(*model_.layers[i], *state_.layer_states[i], saved->layers[i], length);
            else {
                const auto& layer = saved->layers[i];
                if (layer.keys || layer.values || layer.index || layer.convolution || layer.recurrent ||
                    layer.ple_convolution || layer.ple_batch || !layer.ple_tokens.empty())
                    throw CudaSessionStateError("unexpected empty Qwen4 predictor layer state");
            }
        }
    }

    std::vector<std::pair<tyloi::Device, std::size_t>> growth() const override {
        std::vector<std::pair<tyloi::Device, std::size_t>> result;
        const auto& saved = std::get<tyloi::models::qwen4_exp::MtpSnapshot>(snapshot_.payload);
        if (saved.hidden_device >= 0)
            add_session_growth(result, tyloi::Device(tyloi::kCUDA, model_.primary), session_tensor_bytes(snapshot_.last_target_hidden));
        for (std::size_t i = 0; i < saved.layers.size(); ++i) {
            if (!saved.lengths[i]) continue;
            const auto& layer = *model_.layers[i];
            const auto device = layer.device;
            add_session_growth(result, device,
                layer_session_growth(layer, *state_.layer_states[i], saved.layers[i], saved.lengths[i]));
            add_session_growth(result, tyloi::Device(tyloi::kCUDA, model_.primary), session_tensor_bytes(saved.positions[i]));
            add_session_growth(result, tyloi::Device(tyloi::kCPU), saved.layers[i].ple_tokens.size() * sizeof(int64_t));
        }
        return result;
    }

    void stage() override {
        candidate_ = std::make_unique<PredictorState>(model_, false);
        candidate_->layer_states.reserve(model_.layers.size());
        for (const auto& layer : model_.layers)
            candidate_->layer_states.push_back(std::make_unique<LayerState>(*layer));
        const auto& saved = std::get<tyloi::models::qwen4_exp::MtpSnapshot>(snapshot_.payload);
        if (saved.hidden_device >= 0)
            stage_session_tensor(hidden_, snapshot_.last_target_hidden, tyloi::Device(tyloi::kCUDA, model_.primary));
        else hidden_ = snapshot_.last_target_hidden;
        for (std::size_t i = 0; i < saved.layers.size(); ++i) {
            if (!saved.lengths[i]) continue;
            stage_layer_session(*model_.layers[i], *candidate_->layer_states[i], saved.layers[i], saved.lengths[i]);
            stage_session_tensor(candidate_->positions[i], saved.positions[i], tyloi::Device(tyloi::kCUDA, model_.primary));
            candidate_->lengths[i] = saved.lengths[i];
        }
        candidate_->batch = 1;
    }

    void publish() noexcept override {
        state_.layer_states.swap(candidate_->layer_states);
        state_.positions.swap(candidate_->positions);
        state_.lengths.swap(candidate_->lengths);
        state_.batch = candidate_->batch;
    }
    ::tyloi::Tensor target_hidden() const override { return hidden_; }
private:
    PredictorState& state_;
    const Predictor& model_;
    MtpSessionState snapshot_;
    ::tyloi::Tensor hidden_;
    std::unique_ptr<PredictorState> candidate_;
};
} // namespace

std::unique_ptr<tyloi::models::Restore> PredictorState::prepare_restore(const MtpSessionState& snapshot) {
    return std::make_unique<PredictorRestore>(*this, snapshot);
}

} // namespace tyloi::cuda::qwen4_exp

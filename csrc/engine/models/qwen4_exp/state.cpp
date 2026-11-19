#include "engine/models/qwen4_exp/state.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/qwen4_exp/snapshot.h"
#include "engine/models/qwen4_exp/predictor.h"
#include "engine/models/session/state.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>

namespace tyloi::models::qwen4_exp {

namespace {
std::vector<std::int64_t> shape_of(const Tensor& value) {
    std::vector<std::int64_t> result;
    for (std::int64_t i = 0; i < value.dim(); ++i) result.push_back(value.size(i));
    return result;
}
}

LayerState::LayerState(const Layer& layer) {
    if (layer.qsa)
        qsa = std::make_unique<tyloi::ops::Qsa::State>(*layer.qsa);
}

State::State(const Model& model, bool initialize_layers) : model_(model) {
    if (initialize_layers)
        for (const auto& layer : model_.layers)
            layers.push_back(std::make_unique<LayerState>(*layer));
}

State::~State() = default;

tyloi::engine::Metrics State::metrics() const {
    std::map<std::int32_t, std::vector<std::shared_ptr<TensorStorage>>> devices;
    const auto add = [&](const Tensor& value) {
        if (!value.defined() || value.device().is_cpu()) return;
        for (auto& backing : value.backings())
            if (backing && !backing->device.is_cpu()) devices[backing->device.index].push_back(std::move(backing));
    };
    const auto block_cache = [&](const Layer& block, const LayerState& state) {
        if (block.qsa) for (const auto& value : block.qsa->resident_cache_tensors(*state.qsa)) add(value);
        if (block.gdn) { add(state.gdn.conv); add(state.gdn.recurrent); }
        if (block.ple) add(state.ple.conv);
    };
    for (std::size_t i = 0; i < layers.size(); ++i) block_cache(*model_.layers[i], *layers[i]);
    add(positions);
    if (graph_registry) for (const auto& value : graph_registry->storage()) add(value);
    if (const auto* model = dynamic_cast<const Predictor*>(model_.predictor());
        model && predictor && predictor->belongs_to(*model)) {
        const auto& state = static_cast<const PredictorState&>(*predictor);
        for (std::size_t i = 0; i < state.layer_states.size(); ++i)
            block_cache(*model->layers[i], *state.layer_states[i]);
        for (const auto& value : state.positions) add(value);
    }
    std::size_t total = 0;
    tyloi::engine::Metrics result;
    for (const auto& [device, backings] : devices) {
        const auto bytes = tyloi::backing_bytes(backings);
        total += bytes;
        result.emplace_back("gpu_" + std::to_string(device) + "_live_kv_bytes", double(bytes));
    }
    result.emplace_back("kv_cache_bytes", double(total));
    result.emplace_back("kv_cache_contexts", cache_pos > 0 ? double(batch) : 0.0);
    if (graph_registry) {
        auto values = graph_registry->metrics();
        result.insert(result.end(), values.begin(), values.end());
    } else result.emplace_back("decode_graph_count", 0.0);
    return result;
}

void State::release_graphs() {
    if (graph_registry) graph_registry->release();
    graph_registry.reset();
    for (auto& layer : layers) layer->retain_graph_storage = false;
    if (predictor) predictor->release_graphs();
}

void State::publish_restore(State& candidate) noexcept {
    layers.swap(candidate.layers);
    graph_registry.swap(candidate.graph_registry);
    std::swap(positions, candidate.positions);
    cache_pos = candidate.cache_pos;
    decode_position_delta = candidate.decode_position_delta;
    batch = candidate.batch;
    speculative_start = -1;
    speculative_confirmed = 0;
    speculative_suffix_forward = false;
    speculative_suffix_journal = false;
    speculative_prefix_retention = false;
}

void State::reset(std::int64_t new_batch) {
    cache_pos = decode_position_delta = 0;
    speculative_start = -1;
    speculative_confirmed = 0;
    speculative_suffix_forward = false;
    speculative_suffix_journal = false;
    speculative_prefix_retention = false;
    const bool retain = graph_registry && graph_registry->reset(new_batch, reuse_decode_graphs);
    for (auto& layer : layers) layer->retain_graph_storage = retain;
    if (!retain) graph_registry.reset();
    positions = {};
    batch = new_batch;
    for (std::size_t i = 0; i < layers.size(); ++i) layers[i]->reset(*model_.layers[i]);
}

void State::begin_forward() {
    for (auto& layer : layers) layer->retained_prefix = false;
    if (graph_registry && !graph_registry->begin_forward(reuse_decode_graphs)) graph_registry.reset();
}

void State::finish_forward(const Tensor& full_positions, std::int64_t new_batch, std::int64_t tokens) {
    positions = full_positions;
    batch = new_batch;
    speculative_prefix_retention = tokens <= 8 && !layers.empty() &&
        std::all_of(layers.begin(), layers.end(), [](const auto& layer) { return layer->retained_prefix; });
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
    const auto save = [&](const Tensor& tensor) {
        tyloi::models::require_model(tensor.defined(), "Qwen4 recurrent session state is unavailable");
        auto saved = tyloi::ops::clone(tensor);
        bytes += session_tensor_bytes(saved);
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
        snapshot.convolution = save(state.gdn.conv);
        snapshot.recurrent = save(state.gdn.recurrent);
    }
    if (layer.ple) {
        snapshot.ple_convolution = save(layer.ple->conv_state(state.ple));
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
    if (source.device() != device) {
        target = tyloi::ops::copy(source, device);
        return;
    }
    target = tyloi::ops::empty(shape_of(source), source.options().device(device));
    tyloi::ops::copy_into(target, source);
}

static bool session_device_matches(const ::tyloi::Tensor& value, std::optional<tyloi::Device> original, tyloi::Device device) {
    return value.defined() && !value.is_encoded() && (original
        ? value.device() == tyloi::Device(tyloi::kCPU) && *original == device
        : value.device() == device);
}

void validate_layer_session(const Layer& layer, const LayerState& current,
        const tyloi::models::qwen4_exp::LayerSnapshot& saved, std::int64_t position) {
    const auto device = layer.device;
    const auto& c = layer.config;
    const auto tensor = [&](const ::tyloi::Tensor& value, std::optional<tyloi::Device> original, const std::vector<int64_t>& shape) {
        if (!session_device_matches(value, original, device) || value.scalar_type() != tyloi::kFloat32 ||
            value.dim() != std::int64_t(shape.size())) return false;
        for (std::int64_t i = 0; i < value.dim(); ++i) if (value.size(i) != shape[i]) return false;
        return true;
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
    return plan_growth(input_batch, tokens, input, options);
}

std::vector<std::pair<Device, std::size_t>> State::batch_growth(std::int64_t tokens) const {
    tyloi::models::ForwardOptions options;
    options.primary = model_.device.index;
    return plan_growth(1, tokens, {}, options);
}

std::vector<std::pair<Device, std::size_t>> State::plan_growth(std::int64_t input_batch,
        std::int64_t tokens, const tyloi::models::Input& input,
        const tyloi::models::ForwardOptions& options) const {
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
                tyloi::models::require_model(!layer.gdn.saved_conv.defined() && layer.gdn.conv.defined() &&
                    layer.gdn.recurrent.defined(), "GDN suffix requires committed recurrent state");
                add(device, layer.gdn.conv.nbytes());
                add(device, layer.gdn.recurrent.nbytes());
            }
            if (block.ple) {
                tyloi::models::require_model(!layer.ple.saved && layer.ple.conv.defined(), "PLE suffix requires committed convolution state");
                add(device, layer.ple.conv.nbytes());
                const auto& context = block.ple->session_context(layer.ple);
                if (context.tokens.size() > std::numeric_limits<std::size_t>::max() / sizeof(int64_t))
                    throw std::overflow_error("PLE suffix context overflow");
                add(tyloi::Device(tyloi::kCPU), context.tokens.size() * sizeof(int64_t));
            }
        }
    }
    std::vector<int64_t> shape;
    if (positions) {
        if (positions.dim() < 1 || positions.dim() > 3 || positions.size(-1) != cache_pos)
            throw std::invalid_argument("invalid Qwen4 position history");
        shape = shape_of(positions);
        shape.back() = cache_pos + tokens;
    } else if (input.positions) {
        shape = shape_of(input.positions);
        if (shape.empty() || shape.size() > 3 || shape.back() != tokens ||
            (shape.size() > 1 && shape.front() != 3 && shape.front() != 4) ||
            (shape.size() == 3 && shape[1] != batch))
            throw std::invalid_argument("invalid Qwen4 input positions");
        if (shape.size() > 1 && shape.front() == 4) shape.front() = 3;
    } else {
        shape = {tokens};
    }
    if (positions) {
        std::vector<int64_t> current_shape = input.positions ? shape_of(input.positions) : std::vector<int64_t>{tokens};
        if (current_shape.size() > 1 && current_shape.front() == 4) current_shape.front() = 3;
        if (current_shape.size() != shape.size() || current_shape.empty() || current_shape.back() != tokens ||
            !std::equal(current_shape.begin(), current_shape.end() - 1, shape.begin()))
            throw std::invalid_argument("Qwen4 position history geometry changed");
    }
    const auto primary = model_.device;
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
    if (graph_registry) for (const auto& value : graph_registry->storage()) add(value);
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
        Tensor conv, recurrent, ple_conv;
        NgramEmbedding::Context context;
    };
    std::vector<Checkpoint> saved(layers.size());
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto& block = *model_.layers[i];
        const auto& layer = *layers[i];
        ::tyloi::ops::DeviceScope guard(block.device);
        if (block.gdn) {
            tyloi::models::require_model(!layer.gdn.saved_conv.defined() && layer.gdn.conv.defined() &&
                layer.gdn.recurrent.defined(), "GDN suffix requires committed recurrent state");
            saved[i].conv = tyloi::ops::clone(layer.gdn.conv);
            saved[i].recurrent = tyloi::ops::clone(layer.gdn.recurrent);
        }
        if (block.ple) {
            tyloi::models::require_model(!layer.ple.saved && layer.ple.conv.defined(), "PLE suffix requires committed convolution state");
            saved[i].ple_conv = tyloi::ops::clone(layer.ple.conv);
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

void State::rollback_speculative(std::int64_t accepted) {
    tyloi::models::require_model(speculative_start >= 0 && accepted >= 0, "no speculative transaction");
    tyloi::models::require_model(!speculative_suffix_journal || accepted == 0,
        "suffix checkpoint cannot retain an accepted prefix");
    const auto keep = speculative_start + speculative_confirmed + accepted;
    speculative_prefix_retention = false;
    for (std::size_t i = 0; i < model_.layers.size(); ++i)
        layers[i]->rollback(*model_.layers[i], keep);
    if (positions.defined()) positions = ops::slice(positions, -1, 0, keep);
    cache_pos = keep;
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
    tyloi::models::require_model(text_session_state_kind() == TextSessionStateKind::Qwen4 && batch == 1 &&
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
    snapshot.positions = tyloi::ops::clone(positions);
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
        const auto* positions = saved ? &saved->positions : nullptr;
        if (!state_.supports_text_session_state() || state_.speculative_start >= 0 ||
            state_.speculative_suffix_forward || !saved || saved->layers.size() != model_.layers.size() ||
            snapshot_.cache_pos <= 0 || snapshot_.cache_pos > model_.config.maximum ||
            snapshot_.tokens.size() != std::size_t(snapshot_.cache_pos) || !positions || !positions->defined() ||
            !session_device_matches(saved->positions, saved->positions_device, model_.device) ||
            saved->positions.scalar_type() != tyloi::kInt64 || positions->dim() < 1 || positions->dim() > 3 ||
            positions->size(-1) != snapshot_.cache_pos ||
            (positions->dim() > 1 && positions->size(0) != 3) ||
            (positions->dim() == 3 && positions->size(1) != 1))
            throw CudaSessionStateError("incompatible Qwen4 text session state");
        for (std::size_t i = 0; i < saved->layers.size(); ++i)
            validate_layer_session(*model_.layers[i], *state_.layers[i], saved->layers[i], snapshot_.cache_pos);
        if (predictor_model) {
            if (!state_.predictor || !state_.predictor->belongs_to(*predictor_model) ||
                !snapshot_.mtp || !predictor_model->session_boundary(snapshot_.cache_pos, snapshot_.mtp->cache_pos))
                throw CudaSessionStateError("incompatible Qwen4 predictor session boundary");
            predictor_ = state_.predictor->prepare_restore(*snapshot_.mtp);
        }
    }

    std::vector<std::pair<tyloi::Device, std::size_t>> growth() const override {
        std::vector<std::pair<tyloi::Device, std::size_t>> result;
        const auto& saved = std::get<tyloi::models::qwen4_exp::Snapshot>(snapshot_.payload);
        add_session_growth(result, model_.device, session_tensor_bytes(saved.positions));
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
        stage_session_tensor(candidate_->positions, saved.positions, model_.device);
        candidate_->cache_pos = snapshot_.cache_pos;
        candidate_->decode_position_delta = snapshot_.decode_position_delta;
        candidate_->batch = 1;
        if (predictor_) predictor_->stage();
    }

    std::vector<::tyloi::Tensor> storage() const override {
        auto result = candidate_ ? candidate_->storage() : std::vector<::tyloi::Tensor>{};
        if (predictor_) {
            auto more = predictor_->storage();
            result.insert(result.end(), more.begin(), more.end());
        }
        return result;
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
            !hidden.defined() || hidden.dim() != 3 ||
            !session_device_matches(hidden, saved->hidden_device, model_.device) ||
            hidden.size(0) != 1 || hidden.size(1) != 1 || hidden.size(2) != model_.hidden_norm.numel() ||
            (hidden.scalar_type() != tyloi::kFloat16 && hidden.scalar_type() != tyloi::kBFloat16 &&
             hidden.scalar_type() != tyloi::kFloat32))
            throw CudaSessionStateError("incompatible Qwen4 predictor session state");
        for (std::size_t i = 0; i < saved->layers.size(); ++i) {
            const auto length = saved->lengths[i];
            const auto& position = saved->positions[i];
            if (length < 0 || length > snapshot_.cache_pos ||
                (length == 0 && position.defined()) ||
                (length > 0 && (!position.defined() || !session_device_matches(position, saved->positions_devices.empty() ? std::nullopt : saved->positions_devices[i],
                        model_.device) ||
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
        if (saved.hidden_device)
            add_session_growth(result, model_.device, session_tensor_bytes(snapshot_.last_target_hidden));
        for (std::size_t i = 0; i < saved.layers.size(); ++i) {
            if (!saved.lengths[i]) continue;
            const auto& layer = *model_.layers[i];
            const auto device = layer.device;
            add_session_growth(result, device,
                layer_session_growth(layer, *state_.layer_states[i], saved.layers[i], saved.lengths[i]));
            add_session_growth(result, model_.device, session_tensor_bytes(saved.positions[i]));
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
        if (saved.hidden_device)
            stage_session_tensor(hidden_, snapshot_.last_target_hidden, model_.device);
        else hidden_ = snapshot_.last_target_hidden;
        for (std::size_t i = 0; i < saved.layers.size(); ++i) {
            if (!saved.lengths[i]) continue;
            stage_layer_session(*model_.layers[i], *candidate_->layer_states[i], saved.layers[i], saved.lengths[i]);
            stage_session_tensor(candidate_->positions[i], saved.positions[i], model_.device);
            candidate_->lengths[i] = saved.lengths[i];
        }
        candidate_->batch = 1;
    }

    std::vector<::tyloi::Tensor> storage() const override {
        std::vector<::tyloi::Tensor> result;
        const auto add = [&](const ::tyloi::Tensor& value) { if (value) result.push_back(value); };
        add(hidden_);
        if (!candidate_) return result;
        for (const auto& value : candidate_->positions) add(value);
        for (std::size_t i = 0; i < candidate_->layer_states.size(); ++i) {
            const auto& block = *model_.layers[i];
            const auto& layer = *candidate_->layer_states[i];
            if (block.gdn) for (const auto& value : block.gdn->storage(layer.gdn)) add(value);
            if (block.ple) {
                add(layer.ple.conv);
                add(layer.ple.saved_conv);
                for (const auto& value : layer.ple.saved_prefix_conv) add(value);
            }
            if (block.qsa) for (const auto& value : block.qsa->storage(*layer.qsa)) add(value);
        }
        return result;
    }

    void publish() noexcept override {
        state_.release_graphs();
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

} // namespace tyloi::models::qwen4_exp

#include "engine/models/qwen35/forward.h"
#include "engine/models/qwen35/model.h"
#include "engine/models/qwen35/state.h"
#include "engine/backends/cuda/executor/gdn_recorded_replay_batch.h"
#include "engine/backends/cuda/executor/graph/decode.h"
#include "engine/backends/cuda/executor/graph/window.h"
#include "engine/backends/cuda/executor/graph_pool_lease.h"
#include "engine/backends/cuda/ops/attention/scratch.h"
#include "engine/generation/prefill/budget.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace tyloi::models::qwen35 {

struct DecodeGraphs {
    struct Checkpoint {
        std::size_t layer;
        tyloi::state::Recurrent::State initial;
        models::LinearInputs projected;
    };
    struct Cell {
        explicit Cell(cudaStream_t stream) : window(stream) {}

        Tensor ids, positions, lengths;
        models::Decoded output;
        std::vector<Checkpoint> checkpoints;
        std::unique_ptr<cuda::RecordedGdnReplayBatch> replay;
        cuda::DecodeWindow window;
    };

    explicit DecodeGraphs(cuda::StreamHandle stream) : stream(std::move(stream)) {}

    using Key = std::tuple<std::int64_t, std::int64_t, std::int64_t, bool>;
    cuda::StreamHandle stream;
    std::shared_ptr<cuda::GraphPoolLease> pool;
    std::vector<DenseTensor> replay_workspace;
    std::map<Key, std::unique_ptr<Cell>> cells;
    Cell* active = nullptr;
};

namespace {

std::int64_t capacity(const Model& model, const State& state) {
    auto limit = std::min<std::int64_t>(model.metadata.max_position_embeddings, std::numeric_limits<int>::max());
    if (state.layers.size() != model.layers.size()) return 0;
    for (std::size_t i = 0; i < model.layers.size(); ++i) {
        const auto& layer = model.layers[i];
        if (layer.device != model.device) return 0;
        if (const auto* full = std::get_if<Full>(&layer.weights)) {
            const auto* cache = std::get_if<tyloi::state::KvView>(&state.layers[i]);
            if (!cache || !cache->defined() || cache->paged() || cache->ring ||
                full->attention.sliding || full->attention.kv_quantization.enabled() ||
                cache->head_dim > 512) return 0;
            limit = std::min(limit, cache->physical_key.size(2));
        } else {
            const auto* linear = std::get_if<LinearState>(&state.layers[i]);
            if (!linear || !linear->mixer.defined() || linear->mixer.pending()) return 0;
        }
    }
    return limit;
}

void prepare_replay(const Model& model, State& state, DecodeGraphs& graphs, DecodeGraphs::Cell& cell) {
    std::vector<cuda::RecordedGdnReplayBatch::Layer> layers;
    for (const auto& checkpoint : cell.checkpoints) {
        if (!checkpoint.projected.updates) return;
        const auto index = checkpoint.layer;
        const auto& view = std::get<LinearState>(state.layers[index]).mixer.view();
        const auto& attention = std::get<Linear>(model.layers[index].weights).attention;
        // State::rewind restores the initial convolution before this batch runs.
        layers.push_back({checkpoint.projected.qkv.dense(), checkpoint.projected.updates.dense(),
            view.convolution.dense(), view.convolution.dense(), view.recurrent.dense(),
            int(attention.key_heads), view.transposed, attention.tiled_v_heads});
    }
    cell.replay = cuda::RecordedGdnReplayBatch::create(std::move(layers), graphs.replay_workspace);
}

} // namespace

StepSequence<models::Decoded> Model::decode(models::Input input, models::State& value,
        const models::ForwardOptions& forward, const models::DecodeOptions& options) const {
    auto* state = dynamic_cast<State*>(&value);
    if (!options.enabled || options.routed || options.projection != models::DecodeOptions::Projection::hidden ||
        !state || !state->belongs_to(*this) || state->batch != 1 || state->cache_pos <= 0 ||
        state->speculating() || state->decode_position_delta != 0 || metadata.num_experts ||
        !device.is_cuda() || forward.primary != device.index || forward.trace ||
        !input.ids || input.ids.device() != device || input.ids.dim() != 2 || input.ids.size(0) != 1 ||
        (input.ids.scalar_type() != kInt32 && input.ids.scalar_type() != kInt64) ||
        input.embeddings || input.positions || input.cache_positions || input.lengths ||
        input.image_mask || input.trace || input.suffix || input.kv_length || input.attention_parts ||
        (rope.sections && rope.sections.numel()) || !mfq_cuda_graph_capture_supported()) co_return;

    const auto start = state->cache_pos;
    const auto tokens = input.ids.size(1);
    const auto confirmed = input.confirmed;
    const auto limit = capacity(*this, *state);
    if (tokens < 1 || tokens > 8 || tokens > limit - start ||
        confirmed < 0 || confirmed >= tokens) co_return;

    const auto end = start + tokens;
    const auto extent = std::clamp(options.reserve_until > 0 ? options.reserve_until : limit, end, limit);
    const DecodeGraphs::Key key{tokens, confirmed, extent, options.greedy_readout};
    const auto stream = mfq_get_current_cuda_stream(device.index);
    auto& graphs = state->decode_graphs;
    if (graphs && graphs->stream.stream() != stream.stream()) state->release_graphs();
    if (!graphs) graphs = std::make_shared<DecodeGraphs>(stream);
    graphs->active = nullptr;
    auto& slot = graphs->cells[key];
    if (slot && !slot->window.matches(state->storage())) slot.reset();
    const bool capture = !slot;
    if (capture) {
        co_yield GraphPhase::ready;
        std::size_t bytes = (2 * tokens + 1) * sizeof(std::int64_t);
        std::size_t descriptors = 0;
        const auto tensors = state->graph_state();
        for (const auto* tensor : tensors) bytes = engine::prefill_budget_add(bytes, tensor->nbytes());
        if (confirmed > 0) {
            for (const auto& layer : state->layers) {
                if (const auto* linear = std::get_if<LinearState>(&layer)) {
                    bytes = engine::prefill_budget_add(bytes, linear->mixer.view().convolution.nbytes());
                    descriptors = engine::prefill_budget_add(descriptors, sizeof(GdnReplayLayer));
                }
            }
            bytes = engine::prefill_budget_add(bytes, descriptors);
        }
        auto growth = std::vector<std::pair<Device, std::size_t>>{{device, bytes}};
        if (descriptors) growth.emplace_back(Device(kCPU), descriptors);
        co_yield GraphRequest{GraphPhase::admit, nullptr, std::move(growth)};
        if (!graphs->pool)
            graphs->pool = cuda::GraphPoolLease::acquire(cuda::default_context(device.index), stream.stream());
        slot = std::make_unique<DecodeGraphs::Cell>(stream.stream());
        const auto opts = TensorOptions().device(device).dtype(kInt64);
        slot->ids = slot->window.buffer({1, tokens}, opts);
        slot->positions = slot->window.buffer({tokens}, opts);
        slot->lengths = slot->window.buffer({1}, opts);
    }

    auto& cell = *slot;
    ops::copy_into(cell.ids, ops::cast(input.ids, kInt64));
    ops::copy_into(cell.positions, ops::arange(start, end, device));
    ops::fill(cell.lengths, end);

    if (capture) {
        models::Input packet;
        packet.ids = cell.ids;
        packet.positions = packet.cache_positions = cell.positions;
        packet.lengths = cell.lengths;
        packet.confirmed = confirmed;
        packet.kv_length = extent;
        packet.attention_parts = ::decode_graph_attention_parts(extent, cuda::AttentionScratch::kMaxParts);
        struct Saved { Tensor* live; Tensor copy; };
        std::vector<Saved> saved;
        for (auto* tensor : state->graph_state()) saved.push_back({tensor, ops::clone(*tensor)});
        const auto restore = [&] {
            if (state->speculating()) state->commit_speculative();
            for (const auto& item : saved) ops::copy_into(*item.live, item.copy);
        };
        try {
            auto recording = cell.window.begin_capture();
            while (recording->next()) {
                std::exception_ptr failure;
                bool warm = false;
                try {
                    cell.output = capture_forward(*this, *state, packet, forward);
                    {
                        auto readout = ops::dense_scope(nullptr, device);
                        cell.output.logits = logits(cell.output.hidden);
                    }
                    if (options.greedy_readout) {
                        auto rows = ops::reshape(cell.output.logits, {tokens, -1});
                        cell.output.ids = ops::sample_greedy(ops::contiguous(rows));
                    }
                    if (confirmed > 0) {
                        for (std::size_t i = 0; i < state->layers.size(); ++i) {
                            if (const auto* linear = std::get_if<LinearState>(&state->layers[i]))
                                cell.checkpoints.push_back({i, linear->mixer.checkpoint().state, linear->projected});
                        }
                    }
                    warm = recording->finish();
                } catch (...) { failure = std::current_exception(); }
                if (failure) {
                    if (!recording->retry(failure)) std::rethrow_exception(failure);
                    warm = true;
                }
                restore();
                if (warm) {
                    cell.output = {};
                    cell.checkpoints.clear();
                    recording->release_warm_outputs();
                    recording->pause();
                    co_yield GraphRequest{GraphPhase::admit, nullptr, recording->growth()};
                    recording->resume();
                }
            }
            recording->complete();
        } catch (...) {
            restore();
            throw;
        }
        cell.window.retain(state->storage());
        prepare_replay(*this, *state, *graphs, cell);
    }

    models::Decoded result = cell.output;
    try {
        if (!cell.window.enqueue_if_device_only()) cell.window.run();
        if (confirmed > 0) {
            state->speculative_start = start;
            state->speculative_confirmed = confirmed;
            for (const auto& checkpoint : cell.checkpoints) {
                auto& linear = std::get<LinearState>(state->layers[checkpoint.layer]);
                linear.mixer.begin(start, confirmed, tokens, checkpoint.initial);
                linear.projected = checkpoint.projected;
                ++linear.speculative_projection_batches;
                ++linear.speculative_ffn_batches;
            }
        }
        state->cache_pos = end;
        graphs->active = &cell;
        result.raw = input.raw_hidden ? ops::clone(result.raw) : Tensor{};
        result.initialized = capture;
    } catch (...) {
        if (state->speculating()) state->abort_speculative();
        state->release_graphs();
        throw;
    }
    co_yield std::move(result);
}

bool replay_graph(State& state, std::int64_t retained) {
    const auto& graphs = state.decode_graphs;
    if (!graphs || !graphs->active || !graphs->active->replay ||
        graphs->stream.stream() != mfq_get_current_cuda_stream(graphs->stream.device_index()).stream())
        return false;
    auto& cell = *graphs->active;
    for (const auto& checkpoint : cell.checkpoints) {
        const auto& linear = std::get<LinearState>(state.layers[checkpoint.layer]);
        if (!checkpoint.projected.qkv.same_view(linear.projected.qkv) ||
            !checkpoint.projected.updates.same_view(linear.projected.updates))
            return false;
    }
    cell.replay->run(retained);
    return true;
}

} // namespace tyloi::models::qwen35

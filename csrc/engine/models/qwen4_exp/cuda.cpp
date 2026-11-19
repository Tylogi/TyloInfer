#include "engine/models/session/snapshot.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/backends/cuda/executor/graph/decode.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/executor/tq/resources.h"
#include "engine/backends/cuda/ops/ffn/tq/banks.h"
#include "engine/backends/cuda/ops/ffn/experts/launch.h"
#include "engine/backends/cuda/ops/ffn/experts/shared_gate.h"
#include "engine/backends/cuda/ops/sampling/launch.h"
#include "engine/backends/cuda/ops/qwen4_exp.h"
#include "components/error.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/memory.h"
#include "engine/backends/cuda/native/resident_allocations.h"
#include "engine/backends/cuda/native/stream.h"
#include "ops.h"
#include "engine/models/qwen4_exp/cuda.h"
#include "model.h"
#include "engine/models/qwen4_exp/snapshot.h"
#include "engine/models/qwen4_exp/batch.h"
#include "engine/models/qwen4_exp/taps.h"
#include "engine/models/common/mtp.h"
#include "predictor.h"
#include "engine/models/common/load.h"
#include "engine/weights/encoding/quant/nint_rows.h"
#include "engine/weights/encoding/tq/tensor.h"
#include "engine/weights/tq/source.h"
#include "engine/backends/cuda/executor/graph/window.h"
#include "engine/backends/cuda/executor/graph_pool_lease.h"
#include "engine/backends/cuda/ops/profile.h"
#include "engine/backends/cuda/kernels/moe/pipeline.h"
#include <iostream>
#include "engine/ops/ops.h"
#include "engine/ops/residual.h"
#include "engine/ops/ffn.h"
#include "engine/ops/gdn.h"
#include "engine/ops/qsa.h"
#include "engine/generation/sequence.h"
#include "engine/generation/prefill/budget.h"
#include "engine/backends/cuda/native/matmul_rows.h"
#include "engine/backends/cuda/native/graph.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <exception>
#include <limits>
#include <map>
#include <tuple>

namespace tyloi::models::qwen4_exp {
using namespace ::tyloi::cuda;
namespace {

bool routed_graph_supported(const tyloi::models::Moe& weights) {
    if (weights.gate_up.size() != 2 || !weights.down) return false;
    const auto banks = ::tq_banks(weights.gate_up[0], weights.gate_up[1], weights.down);
    return banks.gate && banks.up && banks.down;
}

struct GraphCheckpoint {
    std::vector<::tyloi::Tensor> gdn_conv, gdn_state;
    std::vector<::tyloi::Tensor> ple_conv;
};

void add_growth(std::vector<std::pair<Device, std::size_t>>& growth, Device device,
        std::size_t bytes, std::size_t count = 1) {
    if (count && bytes > std::numeric_limits<std::size_t>::max() / count)
        throw std::overflow_error("Qwen4 graph growth overflow");
    bytes *= count;
    if (!bytes) return;
    auto found = std::find_if(growth.begin(), growth.end(),
        [device](const auto& entry) { return entry.first == device; });
    if (found == growth.end()) growth.emplace_back(device, bytes);
    else {
        if (found->second > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::overflow_error("Qwen4 graph growth overflow");
        found->second += bytes;
    }
}

tyloi::ops::Residual::Result graph_attention(const qwen4_exp::Layer& weights, qwen4_exp::LayerState& state,
        ::tyloi::DenseTensor x, const ::tyloi::DenseTensor& ids, const ::tyloi::DenseTensor& positions,
        const ::tyloi::DenseTensor& cache_positions, const ::tyloi::DenseTensor& ple_rows, int layer,
        int64_t confirmed = 0, GraphCheckpoint* checkpoint = nullptr,
        const std::optional<tyloi::ops::Residual::Result>& pending_residual = {}) {
    if(auto* window=DecodeWindow::recording())window->set_profile_layer_kind(layer,weights.gdn?"gdn":"qsa");
    const auto mark=[&](const char* phase){if(auto* window=DecodeWindow::recording())window->mark(layer,phase);};
    if(weights.ple)x=x+(confirmed
        ? weights.ple->forward_graph(state.ple, x,ids,ple_rows,confirmed,checkpoint->ple_conv)
        : weights.ple->forward_chunk(state.ple, x,ids,true,{},ple_rows)).dense();
    auto [attention_input,attention_residual,attention_gate]=pending_residual
        ? weights.attention_gr.pre_after(x,pending_residual->residual,pending_residual->gate,tyloi::ops::ResidualRole::attention)
        : weights.attention_gr.pre(x,tyloi::ops::ResidualRole::attention);
    mark("attention_core_begin");
    auto attention_output=weights.gdn
        ? (confirmed ? weights.gdn->forward_graph(state.gdn, attention_input,confirmed,checkpoint->gdn_conv,checkpoint->gdn_state)
                     : weights.gdn->forward_chunk(state.gdn, attention_input,true))
        : weights.qsa->forward_graph(*state.qsa, attention_input,positions,cache_positions);
    mark("attention_core_end");
    auto ffn_result=weights.ffn_gr.pre_after(attention_output,attention_residual,attention_gate,tyloi::ops::ResidualRole::ffn);
    mark("ffn_pre_end");
    attention_output={};
    return ffn_result;
}

} // namespace

static bool streamed_decode_graphs() {
    return tyloi::ops::Qsa::Graph({}, false).eligible();
}
static ::tyloi::DenseTensor join_request_outputs(
        const std::vector<::tyloi::DenseTensor>& outputs,int64_t axis) {
    const auto option=::tyloi::cuda::options::single_output_view();
    if(outputs.size()==1 && outputs.front().is_contiguous() && (!option || *option))
        return outputs.front();
    return ::tyloi::cuda::cat(outputs,axis);
}
static ::tyloi::DenseTensor append_decode_positions(
        const ::tyloi::DenseTensor& previous,const ::tyloi::DenseTensor& current,
        ::tyloi::DenseTensor& storage,int64_t maximum,
        const char* option_name="TYLOINFER_QWEN4_POSITION_BUFFER") {
    if(const auto option=::tyloi::cuda::options::position_buffer(option_name))if(!*option)
        return previous.defined()?::tyloi::cuda::cat({previous,current},-1):current.clone();
    TYLOINFER_RUNTIME_CHECK(current.dim()>=1 && (!previous.defined() || (previous.dim()==current.dim() &&
        previous.scalar_type()==current.scalar_type() && previous.device()==current.device())),
        "invalid decode position history");
    const int64_t count=previous.defined()?previous.size(-1):0,required=count+current.size(-1);
    TYLOINFER_RUNTIME_CHECK(required<=maximum,"decode position history exceeds context");
    for(int axis=0;previous.defined() && axis+1<current.dim();++axis)
        TYLOINFER_RUNTIME_CHECK(previous.size(axis)==current.size(axis),"decode position history axes changed");
    // Rollback retains a prefix view; restored histories receive a new arena.
    // The same last-axis append serves 1D target and 3D MTP rotary positions.
    const bool owns_prefix=storage.defined() && previous.defined() &&
        previous.data_ptr()==storage.data_ptr();
    if(!owns_prefix || storage.size(-1)<required) {
        int64_t capacity=owns_prefix?storage.size(-1):std::min<int64_t>(256,maximum);
        while(capacity<required)capacity=capacity>maximum/2?maximum:capacity*2;
        auto shape=current.sizes().vec();shape.back()=capacity;
        auto expanded=::tyloi::cuda::empty(shape,current.options());
        if(count)expanded.narrow(-1,0,count).copy_(previous);
        storage=std::move(expanded);
    }
    storage.narrow(-1,count,current.size(-1)).copy_(current);
    return storage.narrow(-1,0,required);
}
}

namespace tyloi::models::qwen4_exp {
using namespace ::tyloi::cuda;

struct PredictorGraphs {
    struct Cell {
        explicit Cell(cudaStream_t stream) : window(stream) {}
        Tensor ids, hidden, positions, cache_positions, greedy_host, compact_host, greedy;
        MtpStep output;
        DecodeWindow window;
    };

    explicit PredictorGraphs(StreamHandle stream) : stream(std::move(stream)) {}

    StreamHandle stream;
    std::shared_ptr<GraphPoolLease> pool;
    std::map<std::tuple<int64_t, ScalarType, MtpStepMode>, std::unique_ptr<Cell>> cells;
    DenseTensor history;
    int64_t capacity = 0;

    static void prepare(Cell& cell, const MtpRow& row, int64_t start) {
        ::tyloi::ops::copy_into(cell.ids, row.ids);
        ::tyloi::ops::copy_into(cell.hidden, row.hidden);
        mfq_qwen4_exp::fill_decode_positions(cell.cache_positions.dense(), cell.positions.dense(), start);
    }

    std::size_t history_growth(const PredictorState& state, int64_t tokens) const {
        const auto required = state.position() + tokens, maximum = state.owner.config.maximum;
        if (!cuda::options::position_buffer("TYLOINFER_MTP_POSITION_BUFFER").value_or(true))
            return std::size_t(required) * 3 * sizeof(int32_t);
        if (history.defined() && history.data_ptr() == state.positions[0].dense().data_ptr() &&
            required <= history.size(-1)) return 0;
        const auto doubled = required > maximum / 2 ? maximum : required * 2;
        return std::size_t(std::min(maximum, std::max<int64_t>(256, doubled))) * 3 * sizeof(int32_t);
    }

    Tensor positions(const PredictorState& state, const Cell& cell) {
        return append_decode_positions(state.positions[0].dense(), cell.positions.dense(),
            history, state.owner.config.maximum, "TYLOINFER_MTP_POSITION_BUFFER");
    }

    static void advance(PredictorState& state, Tensor positions, int64_t tokens) {
        state.owner.layers.front()->qsa->advance_graph(*state.layer_states.front()->qsa, tokens);
        state.positions[0] = std::move(positions);
        state.lengths[0] += tokens;
    }
};

namespace {

bool predictor_graph_enabled() {
    return cuda::options::mtp_predictor_graph().value_or(true) && mfq_cuda_graph_capture_supported();
}

std::vector<Tensor> predictor_graph_storage(const Predictor& model, PredictorState& state) {
    std::vector<Tensor> owners;
    const auto& attention = *model.layers.front()->qsa;
    for (auto* tensor : attention.graph_state(*state.layer_states.front()->qsa)) owners.push_back(*tensor);
    return owners;
}

} // namespace

StepSequence<MtpStep> Predictor::graph_step(const CausalModel& target, MtpRow row, MtpStepMode mode,
        const ForwardOptions& options) const {
    auto* request = row.state ? dynamic_cast<PredictorState*>(row.state) : nullptr;
    if (!request || !request->belongs_to(*this) || target.predictor() != this ||
        !device.is_cuda() || options.trace || options.primary != device.index || !predictor_graph_enabled() ||
        request->batch != 1 || request->position() <= 0 || layers.size() != 1 || row.positions ||
        !row.ids || row.ids.device() != device || row.ids.scalar_type() != kInt64 ||
        row.ids.dim() != 2 || row.ids.size(0) != 1 || row.ids.size(1) < 1 || row.ids.size(1) > 8 ||
        !row.hidden || row.hidden.device() != device || row.hidden.is_encoded() || row.hidden.dim() != 3 ||
        row.hidden.size(0) != 1 || row.hidden.size(1) != row.ids.size(1) || row.hidden.size(2) != hidden_norm.numel())
        co_return;
    const auto& block = *layers.front();
    const auto tokens = row.ids.size(1), start = request->position();
    if (!request->positions[0] || block.device != device || !block.qsa || block.gdn || block.ple ||
        tokens > config.maximum - start ||
        (mode != MtpStepMode::history && (!tq_graph_supports_tokens(tokens) || !routed_graph_supported(block.ffn))))
        co_return;
    if (mode == MtpStepMode::tail && tokens == 1) mode = MtpStepMode::full;
    const auto* history_chain = cuda::env_lookup("TYLOINFER_MTP_HISTORY_CHAIN");
    if (mode == MtpStepMode::tail && ((history_chain && history_chain[0] == '0') || cuda::Executor::current().kv_offload()))
        co_return;

    auto& qsa = *block.qsa;
    auto& cache = *request->layer_states.front()->qsa;
    ::tyloi::ops::Qsa::Graph graph({{{{&qsa, &cache}}, start, config.maximum, 0}}, false);
    if (!graph.eligible()) co_return;
    const auto stream = cuda::current_stream(device.index);
    auto& graphs = request->graphs;
    if (graphs && graphs->stream.stream() != stream.stream()) request->release_graphs();
    if (!graphs) graphs = std::make_shared<PredictorGraphs>(stream);
    graph.admit(tokens, graphs->capacity, true);
    auto growth = graph.growth();
    add_growth(growth, device, graphs->history_growth(*request, tokens));
    if (!growth.empty())
        co_yield GraphRequest{GraphPhase::admit, nullptr, std::move(growth)};
    graph.materialize();
    if (graph.clear_required()) {
        co_yield GraphPhase::ready;
        graphs->cells.clear();
        if (const auto bytes = qsa.graph_exit_growth_bytes(cache)) {
            std::vector<std::pair<Device, std::size_t>> growth{{device, bytes}};
            co_yield GraphRequest{GraphPhase::admit, nullptr, std::move(growth)};
            qsa.leave_graph(cache);
        }
        graph.layout();
    }
    graphs->capacity = graph.rows().front().capacity;
    if (const auto bytes = qsa.managed_memory(cache) ? 0 : qsa.graph_growth_bytes(cache, graphs->capacity)) {
        std::vector<std::pair<Device, std::size_t>> growth{{device, bytes}};
        co_yield GraphRequest{GraphPhase::admit, nullptr, std::move(growth)};
    }
    graph.prepare(0, request->positions[0]);

    const auto owners = predictor_graph_storage(*this, *request);
    auto& slot = graphs->cells[{tokens, row.hidden.scalar_type(), mode}];
    if (slot && !slot->window.matches(owners)) slot.reset();
    const bool capture = !slot;
    const bool readout = mode != MtpStepMode::history;
    const bool compact = readout && row.compact_sampling && config.vocab >= 64;
    if (capture) {
        co_yield GraphPhase::ready;
        auto fixed = DecodeWindow::fixed_growth(device, false, readout, 0);
        add_growth(fixed, device, row.hidden.nbytes());
        add_growth(fixed, device, std::size_t(tokens) * (2 * sizeof(int64_t) + 3 * sizeof(int32_t)));
        if (readout) add_growth(fixed, Device(kCPU), 65 * sizeof(int64_t));
        co_yield GraphRequest{GraphPhase::admit, nullptr, std::move(fixed)};
        if (!graphs->pool) graphs->pool = GraphPoolLease::acquire(cuda::default_context(device.index), stream.stream());
        auto cell = std::make_unique<PredictorGraphs::Cell>(stream.stream());
        const auto integer = TensorOptions().device(device).dtype(kInt64);
        cell->ids = cell->window.buffer({1, tokens}, integer);
        cell->hidden = cell->window.buffer({1, tokens, row.hidden.size(2)}, row.hidden.options());
        cell->positions = cell->window.buffer({3, 1, tokens}, integer.dtype(kInt32));
        cell->cache_positions = cell->window.buffer({tokens}, integer);
        if (readout) {
            auto packet = cuda::empty({65}, integer.device(kCPU).pinned_memory(true));
            cell->compact_host = packet.narrow(0, 0, 64).reshape({1, 64});
            cell->greedy_host = packet.narrow(0, 64, 1);
        }
        slot = std::move(cell);
    }

    auto& cell = *slot;
    PredictorGraphs::prepare(cell, row, start);
    if (readout) cell.window.set_compact_sampling(compact, "predictor", 1);
    if (capture) {
        if (readout) co_yield GraphPhase::prepare;
        auto recording = cell.window.begin_capture();
        while (recording->next()) {
            std::exception_ptr failure;
            bool warm = false;
            try {
                auto input = fuse(cell.hidden, embed(cell.ids));
                auto mix = block.attention_gr.pre(input);
                const auto prefix = mode == MtpStepMode::tail ? tokens - 1 : 0;
                if (prefix) {
                    qsa.forward_graph(cache, ::tyloi::ops::slice(mix.mixed, 1, 0, prefix),
                        ::tyloi::ops::contiguous(::tyloi::ops::slice(cell.positions, 2, 0, prefix)),
                        ::tyloi::ops::slice(cell.cache_positions, 0, 0, prefix), {}, true, true);
                }
                auto attention = qsa.forward_graph(cache, ::tyloi::ops::slice(mix.mixed, 1, prefix, tokens),
                    ::tyloi::ops::contiguous(::tyloi::ops::slice(cell.positions, 2, prefix, tokens)),
                    ::tyloi::ops::slice(cell.cache_positions, 0, prefix, tokens), {}, true, !readout);
                if (readout) {
                    auto residual = block.attention_gr.post(attention,
                        ::tyloi::ops::slice(mix.residual, 1, prefix, tokens), ::tyloi::ops::slice(mix.gate, 1, prefix, tokens));
                    auto ffn = block.ffn_gr.pre(residual);
                    auto chain = block.ffn_gr.post(feed_forward(block.ffn, ffn.mixed), ffn.residual, ffn.gate);
                    auto sample = final_mixer->pre(chain).mixed;
                    const auto end = sample.size(1);
                    auto last = ::tyloi::ops::slice(sample, 1, end - 1, end);
                    auto scores = draft_token_ids ? logits(std::move(last)) : target.logits(last);
                    cell.output = {std::move(sample), std::move(chain), std::move(scores)};
                    cell.greedy = cell.window.sample(cell.output.sample_logits, cell.greedy_host, cell.compact_host, 1);
                }
                warm = recording->finish();
            } catch (...) { failure = std::current_exception(); }
            if (failure) {
                if (!recording->retry(failure)) std::rethrow_exception(failure);
                warm = true;
            }
            if (warm) {
                cell.output = {};
                cell.greedy = {};
                recording->release_warm_outputs();
                recording->pause();
                co_yield GraphRequest{GraphPhase::admit, nullptr, recording->growth()};
                recording->resume();
            }
        }
        recording->complete();
        cell.window.retain(owners);
    }
    auto positions = graphs->positions(*request, cell);
    if (readout || !cell.window.enqueue_if_device_only()) cell.window.run();
    graphs->advance(*request, std::move(positions), tokens);
    if (readout) {
        auto result = cell.output;
        result.sample_hidden = ::tyloi::ops::clone(result.sample_hidden);
        result.chain_hidden = ::tyloi::ops::clone(result.chain_hidden);
        result.sample_logits = ::tyloi::ops::clone(result.sample_logits);
        result.greedy_token = int32_t(*cell.greedy_host.dense().data_ptr<int64_t>());
        if (compact) result.sample_compact = ::tyloi::ops::clone(cell.compact_host);
        co_yield std::move(result);
    } else co_yield MtpStep{};
}

StepSequence<MtpDraft> Predictor::greedy_draft(const CausalModel& target, MtpRow row, int depth,
        const ForwardOptions& options) const {
    auto* request = row.state ? dynamic_cast<PredictorState*>(row.state) : nullptr;
    if (!request || !request->belongs_to(*this) || target.predictor() != this || options.trace ||
        !device.is_cuda() || !predictor_graph_enabled() || row.positions || request->batch != 1 ||
        depth < 2 || depth > engine::mtp::kMaximumReportedDraftDepth || !request->graphs ||
        !row.ids || row.ids.device() != device || row.ids.scalar_type() != kInt64 || row.ids.dim() != 2 ||
        row.ids.size(0) != 1 || row.ids.size(1) < 1 || !row.hidden || row.hidden.device() != device ||
        row.hidden.dim() != 3 || row.hidden.size(0) != 1 || row.hidden.size(1) != row.ids.size(1) ||
        row.hidden.size(2) != hidden_norm.numel()) co_return;

    auto& graphs = *request->graphs;
    const auto tokens = row.ids.size(1), start = request->position();
    const auto written = tokens + depth - 1;
    if (start <= 0 || written > graphs.capacity - start ||
        graphs.stream.stream() != cuda::current_stream(device.index).stream()) co_return;
    const auto owners = predictor_graph_storage(*this, *request);
    const auto ready = [&](int64_t width, ScalarType type, MtpStepMode mode) -> PredictorGraphs::Cell* {
        const auto found = graphs.cells.find({width, type, mode});
        return found != graphs.cells.end() && found->second->window.valid() &&
            found->second->window.matches(owners) ? found->second.get() : nullptr;
    };
    auto* first = ready(tokens, row.hidden.scalar_type(), tokens > 1 ? MtpStepMode::tail : MtpStepMode::full);
    if (!first) co_return;
    auto* next = ready(1, first->output.chain_hidden.scalar_type(), MtpStepMode::full);
    if (!next || next->output.chain_hidden.scalar_type() != first->output.chain_hidden.scalar_type()) co_return;

    auto& qsa = *layers.front()->qsa;
    auto& cache = *request->layer_states.front()->qsa;
    ::tyloi::ops::Qsa::Graph graph({{{{&qsa, &cache}}, start, config.maximum, 0}}, false);
    graph.admit(written, graphs.capacity, true);
    auto growth = graph.growth();
    add_growth(growth, device, graphs.history_growth(*request, written));
    if (!growth.empty())
        co_yield GraphRequest{GraphPhase::admit, nullptr, std::move(growth)};
    graph.materialize();
    graph.prepare(0, request->positions[0]);

    auto packet = ::tyloi::ops::empty({1, depth + 1}, row.ids.options());
    ::tyloi::ops::copy_into(::tyloi::ops::slice(packet, 1, 0, 1), ::tyloi::ops::slice(row.ids, 1, tokens - 1, tokens));
    for (int index = 0; index < depth; ++index) {
        auto& cell = index == 0 ? *first : *next;
        PredictorGraphs::prepare(cell, row, request->position());
        auto positions = graphs.positions(*request, cell);
        cell.window.set_compact_sampling(false, "predictor", 1);
        if (!cell.window.enqueue_if_device_only()) cell.window.run();
        graphs.advance(*request, std::move(positions), cell.ids.size(1));
        row.ids = ::tyloi::ops::reshape(cell.greedy, {1, 1});
        row.hidden = cell.output.chain_hidden;
        ::tyloi::ops::copy_into(::tyloi::ops::slice(packet, 1, index + 1, index + 2), row.ids);
    }
    co_yield MtpDraft{std::move(packet)};
}

namespace {

::tyloi::DenseTensor qwen4_last_hidden(::tyloi::DenseTensor hidden) {
    return hidden.index({::tyloi::indexing::Slice(), -1, ::tyloi::indexing::Slice()});
}
::tyloi::DenseTensor qwen4_logits_from_hidden(const qwen4_exp::Model& model, ::tyloi::DenseTensor hidden) {
    const auto cap = model.metadata.final_logit_softcapping;
    auto logits = model.logits(hidden).dense();
    return cap > 0.0 ? ::tyloi::cuda::tanh(logits / cap) * cap : logits;
}
::tyloi::DenseTensor qwen4_finalize_hidden(const qwen4_exp::Model& model, ::tyloi::DenseTensor hidden,
                                           std::int64_t, std::int64_t) {
    return model.final_mixer->pre(hidden, tyloi::ops::ResidualRole::final).mixed.dense();
}

::tyloi::DenseTensor qwen4_next_token(const qwen4_exp::Model& model, ::tyloi::DenseTensor hidden) {
    auto* window=DecodeWindow::recording();
    if(window)window->mark(-1,"lm_head_begin");
    auto projected=model.logits(hidden).dense();
    if(window)window->mark(-1,"lm_head_projection_end");
    auto output=::tyloi::cuda::argmax(projected,-1).to(::tyloi::kInt64);
    if(window)window->mark(-1,"lm_head_sample_end");
    return output;
}


} // namespace
bool Model::prefetch_token_rows(State& state,const ::tyloi::Tensor& host_ids) {
    if (!state.reuse_decode_graphs || !state.graph_registry) return false;
    bool issued=false;
    for(std::size_t i=0;i<layers.size();++i) {
        const auto& block=layers[i];
        if(block->ple && block->ple->graph_supported()) {
            block->ple->prefetch_tokens(state.layers[i]->ple,host_ids,true);
            issued=true;
        }
    }
    return issued;
}

struct Qwen4DecodeGraph final : DecodeGraph {
    bool startup_warmup=false;
    struct PleStage {
        qwen4_exp::Ple* ple;
        qwen4_exp::Ple::State* state;
        ::tyloi::DenseTensor rows;
        std::shared_ptr<RowStage> transfer;
        std::vector<qwen4_exp::NgramEmbedding::Context> prefix_context;
        std::unique_ptr<DecodeWindow::Signal> signal;
        PleStage(qwen4_exp::Ple* p,qwen4_exp::Ple::State& request,int tokens,int hidden,
                Device device, DecodeWindow& window)
            :ple(p),state(&request),signal(window.signal()) {
            transfer=ple->graph_stage(*state,tokens,device);
            signal->set_mapped_rows(transfer->mapped_rows());
            rows=transfer->output().dense().reshape({1,tokens,hidden});
        }
    };
    struct Cell {
        ::tyloi::DenseTensor ids,positions,cache_positions,host_ids,output,raw_hidden,logits,greedy_host,compact_host;
        Tensor verification;
        unsigned used_kinds=0;
        std::vector<std::unique_ptr<PleStage>> ple;
        std::vector<GraphCheckpoint> checkpoints;
        DecodeWindow window;
        Cell(int tokens,int hidden,Device device,const std::vector<std::unique_ptr<qwen4_exp::Layer>>& weights,
                const std::vector<std::unique_ptr<qwen4_exp::LayerState>>& layers,int confirmed,
                std::vector<GraphCheckpoint>& shared_checkpoints)
            :window(::tyloi::cuda::options::layer_timings()) {
            const auto options=::tyloi::TensorOptions().device(device).dtype(::tyloi::kInt64);
            ids=window.buffer({1,tokens},options,true).dense();
            positions=window.buffer({tokens},options,true).dense();
            cache_positions=window.buffer({tokens},options,true).dense();
            host_ids=window.buffer({1,tokens},options.device(::tyloi::kCPU),true).dense();
            greedy_host=window.buffer({tokens},options.device(::tyloi::kCPU).pinned_memory(true)).dense();
            compact_host=window.buffer({tokens,64},options.device(::tyloi::kCPU).pinned_memory(true)).dense();
            if(confirmed && shared_checkpoints.empty())shared_checkpoints.resize(weights.size());
            std::size_t layer=0;
            for(const auto& block:weights) {
                auto& b=*block;
                if(b.ple)ple.push_back(std::make_unique<PleStage>(b.ple.get(),layers[layer]->ple,tokens,hidden,device,window));
                if(confirmed) {
                    auto& shared=shared_checkpoints[layer];
                    GraphCheckpoint checkpoint;
                    if(b.gdn) {
                        const auto state=b.gdn->graph_state(layers[layer]->gdn);
                        while(shared.gdn_conv.size()<std::size_t(tokens-confirmed)) {
                            shared.gdn_conv.push_back(::tyloi::cuda::empty_like(state[0]->dense()));
                            shared.gdn_state.push_back(::tyloi::cuda::empty_like(state[1]->dense()));
                        }
                        checkpoint.gdn_conv.assign(shared.gdn_conv.begin(),shared.gdn_conv.begin()+tokens-confirmed);
                        checkpoint.gdn_state.assign(shared.gdn_state.begin(),shared.gdn_state.begin()+tokens-confirmed);
                    }
                    if(b.ple) {
                        while(shared.ple_conv.size()<std::size_t(tokens-confirmed))
                            shared.ple_conv.push_back(::tyloi::cuda::empty_like(b.ple->conv_state(layers[layer]->ple).dense()));
                        checkpoint.ple_conv.assign(shared.ple_conv.begin(),shared.ple_conv.begin()+tokens-confirmed);
                    }
                    checkpoints.push_back(std::move(checkpoint));
                }
                ++layer;
            }
        }
    };
    DecodeStream stream;
    std::map<std::tuple<int,int,int,bool,bool>,std::unique_ptr<Cell>> cells;
    // Verification widths execute serially and commit/rollback before the next
    // window. Their snapshots can share stable allocations across graph shapes.
    std::vector<GraphCheckpoint> shared_checkpoints;
    qwen4_exp::State& request;
    std::vector<std::pair<const tyloi::ops::Qsa*,tyloi::ops::Qsa::State*>> qsa;
    ::tyloi::DenseTensor position_storage;
    int64_t kv_capacity=0;
    bool failed=false;
    explicit Qwen4DecodeGraph(const qwen4_exp::Model& model,qwen4_exp::State& state)
        :request(state) {
        for(std::size_t i=0;i<model.layers.size();++i)
            if(auto* attention=model.layers[i]->qsa.get())qsa.emplace_back(attention,request.layers[i]->qsa.get());
    }
    void restore_legacy() {
        cells.clear();shared_checkpoints.clear();
        for(auto [attention,state]:qsa)attention->leave_graph(*state);
        for(auto& layer:request.layers)layer->retain_graph_storage=false;
    }
    bool reusable() const {
        return !failed && !qsa.empty() && std::all_of(qsa.begin(),qsa.end(),
            [](const auto& attention){return attention.first->graph_capacity(*attention.second)>0;});
    }

    bool reset(std::int64_t batch, bool reuse) override {
        const bool retain = reuse && batch == 1 && reusable();
        if (retain) for (auto& [key, cell] : cells) cell->used_kinds = 0;
        return retain;
    }
    bool begin_forward(bool reuse) override {
        if (reuse && reusable()) {
            for (auto [attention, state] : qsa) attention->leave_graph(*state);
            return true;
        }
        restore_legacy();
        return false;
    }
    void release() override { restore_legacy(); }
    tyloi::engine::Metrics metrics() const override {
        tyloi::engine::Metrics result;
        for (const auto& [key, cell] : cells) {
            const auto prefix = std::string("cuda_graph_profile_target_rows_") +
                std::to_string(std::get<0>(key)) + "_kind_" + std::to_string(std::get<1>(key)) +
                "_confirmed_" + std::to_string(std::get<2>(key)) +
                (std::get<3>(key) ? "_raw" : "") + (std::get<4>(key) ? "_verify_" : "_");
            append_graph_profile_metrics(cell->window, prefix, result);
        }
        result.emplace_back("decode_graph_count", double(cells.size()));
        return result;
    }
    std::vector<Tensor> storage() const override {
        std::vector<Tensor> result;
        if (position_storage.defined()) result.emplace_back(position_storage);
        for (const auto& [key, cell] : cells)
            for (const auto& checkpoint : cell->checkpoints) {
                result.insert(result.end(), checkpoint.gdn_conv.begin(), checkpoint.gdn_conv.end());
                result.insert(result.end(), checkpoint.gdn_state.begin(), checkpoint.gdn_state.end());
                result.insert(result.end(), checkpoint.ple_conv.begin(), checkpoint.ple_conv.end());
            }
        return result;
    }

    tyloi::StepSequence<tyloi::models::Decoded> run(const qwen4_exp::Model& model,
            tyloi::models::Input input, tyloi::models::DecodeOptions options) {
        auto ids=model.device_ids(input.ids).dense();
        const auto tokens=static_cast<int>(ids.size(1));
        const auto confirmed=static_cast<int>(input.confirmed);
        const bool want_raw=input.raw_hidden!=nullptr;
        const auto tap_layers=want_raw && model.predictor() ? model.predictor()->target_layers() : std::span<const int64_t>{};
        const auto projection=options.projection;
        int kind=projection==tyloi::models::DecodeOptions::Projection::greedy ? 2 :
            projection==tyloi::models::DecodeOptions::Projection::logits ? 1 :
            projection==tyloi::models::DecodeOptions::Projection::hidden ? 4 : 3;
        const bool want_logits=kind==4;
        const bool verify=want_logits && tokens>1 && options.readout==tyloi::models::DecodeOptions::Readout::verification;
        tyloi::models::Decoded result;
        if(failed)throw std::runtime_error("MFQ model session requires recovery after its previous error");
        stream.require_current();
        // These consumers use the same single-row backbone. Keep its hidden
        // state, logits and selected token in one graph instead of reserving
        // a complete set of layer buffers for each output representation.
        const int result_kind=kind;
        if(tokens==1 && kind>=1 && kind<=4)kind=4;
        co_yield tyloi::GraphPhase::ready;
        const auto growth=[&](std::int64_t capacity) {
            std::vector<std::pair<tyloi::Device,std::size_t>> result;
            for(std::size_t i=0;i<model.layers.size();++i) {
                const auto& block=*model.layers[i];
                if(!block.qsa)continue;
                const auto bytes=block.qsa->graph_growth_bytes(*request.layers[i]->qsa,capacity);
                if(!bytes)continue;
                const auto device = block.device;
                auto it=std::find_if(result.begin(),result.end(),[&](const auto& item){return item.first==device;});
                if(it==result.end())result.emplace_back(device,bytes);
                else {
                    if(it->second>std::numeric_limits<std::size_t>::max()-bytes)
                        throw std::overflow_error("Qwen4 graph growth overflow");
                    it->second+=bytes;
                }
            }
            return result;
        };
        tyloi::ops::Qsa::Graph graph({{qsa, request.cache_pos, model.max_position_embeddings(),
                                      request.decode_reserve_until}}, false);
        graph.admit(tokens, kv_capacity, want_raw);
        co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr, graph.growth()};
        graph.materialize();
        if (graph.clear_required()) {
            co_yield tyloi::GraphPhase::ready;
            co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit,nullptr,request.graph_exit_growth()};
            restore_legacy();
            graph.layout();
        }
        kv_capacity = graph.rows().front().capacity;
        co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit,nullptr,growth(kv_capacity)};
        const auto observed=[&] {
            std::vector<::tyloi::Tensor> result;
            for(std::size_t i=0;i<model.layers.size();++i) {
                auto& layer=*request.layers[i];const auto& weights=*model.layers[i];
                for(auto* tensor:layer.graph_state(weights))result.push_back(*tensor);
                if(weights.qsa)for(auto* tensor:weights.qsa->graph_state(*layer.qsa))result.push_back(*tensor);
            }
            return result;
        };
        auto& owner=cells[{tokens,kind,confirmed,!tap_layers.empty() || (want_raw && kind<3),verify}];
        if(owner && owner->window.valid() && !owner->window.matches(observed()))owner.reset();
        if(!owner) {
            const Device device = model.device;
            const auto ple_count = std::count_if(model.layers.begin(), model.layers.end(),
                [](const auto& layer) { return bool(layer->ple); });
            auto fixed = DecodeWindow::fixed_growth(device, ::tyloi::cuda::options::layer_timings(), kind == 4 && tokens > 1, ple_count);
            add_growth(fixed, device, std::size_t(tokens) * sizeof(int64_t), 3);
            add_growth(fixed, Device(kCPU), std::size_t(tokens) * sizeof(int64_t), 66);
            for (std::size_t i = 0; i < model.layers.size(); ++i) {
                const auto& layer = *model.layers[i];
                if (layer.ple) for (const auto& [target, bytes] : layer.ple->graph_stage_growth(tokens, device))
                    add_growth(fixed, target, bytes);
                if (!confirmed) continue;
                const GraphCheckpoint empty;
                const auto& shared = i < shared_checkpoints.size() ? shared_checkpoints[i] : empty;
                const auto count = std::size_t(tokens - confirmed);
                if (layer.gdn) {
                    const auto tensors = layer.gdn->graph_state(request.layers[i]->gdn);
                    if (shared.gdn_conv.size() < count)
                        add_growth(fixed, tensors[0]->device(), tensors[0]->nbytes(), count - shared.gdn_conv.size());
                    if (shared.gdn_state.size() < count)
                        add_growth(fixed, tensors[1]->device(), tensors[1]->nbytes(), count - shared.gdn_state.size());
                }
                if (layer.ple && shared.ple_conv.size() < count) {
                    const auto& tensor = layer.ple->conv_state(request.layers[i]->ple);
                    add_growth(fixed, tensor.device(), tensor.nbytes(), count - shared.ple_conv.size());
                }
            }
            co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr, std::move(fixed)};
            owner=std::make_unique<Cell>(tokens,model.config.hidden,device,model.layers,request.layers,confirmed,shared_checkpoints);
        }
        owner->window.set_profile_geometry(want_raw?"mtp_target":"target_decode",tokens);
        auto& cell=*owner;
        if(kind==4 && tokens>1 && !verify)cell.window.set_compact_sampling(options.compact_sampling,"target",tokens);
        // A retained graph's first replay can still pay cold transfer/cache
        // costs after prefill. Exclude that replay from steady MTP policy data.
        result.initialized=!cell.window.valid() || !(cell.used_kinds&(1u<<result_kind));
        if(!cell.ple.empty()) {
            if(input.host_ids) {
                TYLOINFER_RUNTIME_CHECK(!input.host_ids.dense().is_cuda() && input.host_ids.dense().scalar_type()==::tyloi::kInt64 &&
                    input.host_ids.dense().sizes()==ids.sizes(),"Qwen4 graph host token geometry changed");
                cell.host_ids.copy_(input.host_ids.dense());
            } else cell.host_ids.copy_(ids.to(::tyloi::kCPU,::tyloi::kInt64));
        }
        cell.ids.copy_(ids);
        mfq_qwen4_exp::fill_decode_positions(cell.cache_positions,cell.positions,request.cache_pos,request.decode_position_delta);
        try {
            // PLE depends only on token IDs and the committed ngram context.
            // Start its read before waiting for adaptive expert exchanges,
            // rather than leaving that CPU interval outside the I/O overlap.
            // Task reset repeats issue() idempotently; it never advances the
            // context, which is still committed by upload_graph().
            const auto* early_ple=tyloi::cuda::env_lookup("TYLOINFER_PLE_EARLY_ISSUE");
            if(!early_ple || early_ple[0]!='0')
                for(const auto& stage:cell.ple)stage->ple->prefetch_tokens(*stage->state,cell.host_ids);
            co_yield tyloi::GraphPhase::ready;
            // Prefix restoration and eager prefill update the same KV/index
            // storage; refresh pooled keys even when this graph is already valid.
            graph.prepare(0, request.positions);
            if(!cell.window.valid()) {
                co_yield tyloi::GraphPhase::prepare;
                struct Saved {::tyloi::Tensor* target;::tyloi::DenseTensor saved;const void* address;};
                std::vector<Saved> states;
                std::vector<std::pair<Device, std::size_t>> copy_growth;
                if (!startup_warmup) for (std::size_t i = 0; i < model.layers.size(); ++i)
                    for (auto* value : request.layers[i]->graph_state(*model.layers[i]))
                        add_growth(copy_growth, value->device(), value->nbytes());
                if (!copy_growth.empty())
                    co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr, std::move(copy_growth)};
                if(!startup_warmup) for(std::size_t i=0;i<model.layers.size();++i)
                        for(auto* value:request.layers[i]->graph_state(*model.layers[i]))
                            states.push_back({value,value->dense().clone(),value->dense().data_ptr()});
                auto restore=[&] {
                    for(const auto& state:states) {
                        TYLOINFER_RUNTIME_CHECK(state.target->dense().data_ptr()==state.address,"MFQ capture changed recurrent storage");
                        state.target->dense().copy_(state.saved);
                    }
                };
                auto recording=cell.window.begin_capture();
                try { while(recording->next()) {
                    std::exception_ptr failure;
                    bool warm = false;
                    try {
                    cell.window.mark(-1,"window_begin");
                    auto hidden=model.repeat_hidden(model.embed_tokens(cell.ids),model.metadata.hc_mult).dense();
                    qwen4_exp::Taps taps(tap_layers,model.metadata.hc_mult,model.device);
                    std::size_t ple_index=0;int layer=0;
                    std::optional<tyloi::ops::Residual::Result> pending_residual;
                    const auto chain_setting=::tyloi::cuda::options::gr_chain_ffn_post();
                    const bool chain_post=(!chain_setting || *chain_setting) && tokens<=8;
                    for(const auto& block:model.layers) {
                        cell.window.mark(layer,"block_begin");
                        auto& b=*block;::tyloi::DenseTensor rows;
                        if(b.ple) {
                            cell.window.mark(layer,"ple_begin");
                            auto* stage=cell.ple[ple_index++].get();rows=stage->rows;
                            cell.window.enroll({stage,
                                [&cell,stage] {stage->signal->reset();stage->ple->prefetch_tokens(*stage->state,cell.host_ids);},
                                [this,&cell,stage,confirmed] {
                                    stage->signal->wait_route();
                                    auto guard=cell.window.transfer_scope();
                                    if(confirmed) {
                                        stage->prefix_context.clear();
                                        for(int i=confirmed;i<cell.ids.size(1);++i)
                                            stage->prefix_context.push_back(stage->ple->graph_prefix_context(*stage->state,cell.host_ids,i));
                                    }
                                    stage->ple->upload_graph(*stage->state,cell.host_ids,*stage->transfer);
                                    stage->signal->publish(stage->transfer->mapped_rows());
                                },{},[stage]{stage->signal->cancel();}});
                            stage->signal->wait_device();
                            stage->transfer->decode();
                            cell.window.mark(layer,"ple_end");
                        }
                        cell.window.mark(layer,"attention_begin");
                        std::optional<tyloi::ops::Residual::Result> next_residual;
                        bool defer=false;
                        if(chain_post && size_t(layer+1)<model.layers.size()) {
                            auto& next=*model.layers[layer+1];
                            defer=!taps.contains(layer) && !next.ple && next.attention_gr.can_chain(pending_residual?pending_residual->residual:hidden);
                        }
                        {
                            auto& layer_state=*request.layers[layer];
                            auto ffn_result=graph_attention(b,layer_state,std::move(hidden),cell.ids,cell.positions,
                                cell.cache_positions,rows,layer,confirmed,confirmed?&cell.checkpoints[layer]:nullptr,pending_residual);
                            const auto& ffn_input=ffn_result.mixed;
                            auto ffn_output=tyloi::models::qwen4_exp::feed_forward(b.ffn,ffn_input).dense();
                            if(defer) {
                                next_residual=std::move(ffn_result);
                                hidden=std::move(ffn_output);
                            } else hidden=b.ffn_gr.post(ffn_output,ffn_result.residual,ffn_result.gate).dense();
                        }
                        pending_residual=std::move(next_residual);
                        taps.capture(layer,hidden);
                        cell.window.mark(layer++,"block_end");
                    }
                    if(kind>=3 || want_raw)cell.raw_hidden=taps.finish(hidden).dense().clone();
                    hidden=qwen4_finalize_hidden(model,std::move(hidden),1,tokens);
                    if(kind>=3) {
                        cell.output=std::move(hidden);
                        if(kind==4) {
                            cell.logits=qwen4_logits_from_hidden(model, cell.output);
                            if(verify) {
                                auto rows=tyloi::ops::reshape(Tensor(cell.logits),{tokens,-1});
                                const std::array<Tensor,2> packet{
                                    cell.ids.narrow(1,1,tokens-1).reshape({tokens-1}),
                                    tyloi::ops::reshape(tyloi::ops::sample_greedy(tyloi::ops::contiguous(rows)),{tokens})};
                                cell.verification=tyloi::ops::cat(packet,0);
                            } else cell.window.sample(cell.logits,cell.greedy_host,cell.compact_host,tokens);
                        }
                    }
                    else if(kind==2)cell.output=qwen4_next_token(model,qwen4_last_hidden(std::move(hidden)));
                    else if(kind==1)cell.output=model.logits(qwen4_last_hidden(std::move(hidden))).dense();
                    else cell.output=qwen4_logits_from_hidden(model, std::move(hidden));
                    cell.window.mark(-1,"window_end");
                    warm=recording->finish();
                    } catch (...) { failure = std::current_exception(); }
                    if (failure) {
                        if (!recording->retry(failure)) std::rethrow_exception(failure);
                        warm = true;
                    }
                    restore();
                    if(warm) {
                        cell.output={};cell.raw_hidden={};cell.logits={};
                        cell.verification={};
                        recording->release_warm_outputs();
                        recording->pause();
                        co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr, recording->growth()};
                        recording->resume();
                    }
                }
                recording->complete();
                cell.window.retain_result(cell.output);
                cell.window.retain_result(cell.raw_hidden);
                cell.window.retain_result(cell.logits);
                cell.window.retain_result(cell.verification);
                cell.window.retain(observed());
                } catch(...) {recording.reset();restore();throw;}
                ++request.decode_graph_captures;
                std::cerr<<"runtime_qwen4_graph_capture role=target rows="<<tokens<<" kind="<<kind
                    <<" confirmed="<<confirmed<<" segments="<<cell.window.segments()<<" count="<<request.decode_graph_captures
                    <<" ms="<<cell.window.capture_ms()<<'\n';
            }
            cell.window.run();cell.used_kinds|=1u<<result_kind;
            if(confirmed) {
                // Captured copies save the confirmed prefix. Bind their owned
                // storage only after replay, so generic commit/rollback keeps
                // exactly the same transaction lifecycle as eager execution.
                std::size_t ple_index=0,layer=0;
                for(const auto& block:model.layers) {
                    auto& b=*block;
                    const auto& checkpoint=cell.checkpoints[layer];
                    auto& layer_state=*request.layers[layer++];
                    const auto position=request.cache_pos+confirmed;
                    if(b.gdn)b.gdn->bind_graph_checkpoint(layer_state.gdn,checkpoint.gdn_conv,checkpoint.gdn_state,position);
                    if(b.ple)b.ple->bind_graph_checkpoint(layer_state.ple,checkpoint.ple_conv,cell.ple[ple_index++]->prefix_context,position);
                }
                request.speculative_start=request.cache_pos;request.speculative_confirmed=confirmed;
                request.speculative_prefix_retention=true;
            }
            for(std::size_t i=0;i<model.layers.size();++i)
                if(model.layers[i]->qsa)model.layers[i]->qsa->advance_graph(*request.layers[i]->qsa,tokens);
            request.cache_pos+=tokens;
            request.positions=append_decode_positions(request.positions ? request.positions.dense() : DenseTensor{},cell.positions,position_storage,
                model.max_position_embeddings());request.batch=1;
            // MTP can retain history until a later probe. Each step must own its
            // raw residual rather than alias the next replay's capture buffer.
            if(want_raw)result.raw=cell.raw_hidden.clone();
            if(kind==4 && tokens>1 && !verify && options.compact_sampling && cell.logits.size(-1)>=64)
                result.compact=cell.compact_host;
            if(want_logits) {
                result.logits=cell.logits.clone();
                if(verify)result.ids=cell.verification;
                else {
                    const auto* first=cell.greedy_host.data_ptr<int64_t>();
                    result.tokens.assign(first,first+tokens);
                }
            }
            if(tokens==1 && result_kind==1)result.logits=cell.logits.reshape({1,-1});
            else if(tokens==1 && result_kind==2)result.ids=cell.greedy_host;
            else result.hidden=cell.output;
            co_yield std::move(result);
        } catch(...) {failed=true;throw;}
    }
};
tyloi::StepSequence<tyloi::models::Decoded> qwen4_exp::Model::decode(
        tyloi::models::Input input, tyloi::models::State& value,
        const tyloi::models::ForwardOptions&, const tyloi::models::DecodeOptions& options) const {
    auto& state=dynamic_cast<qwen4_exp::State&>(value);
    if (!state.belongs_to(*this)) throw std::invalid_argument("Qwen4 state belongs to a different model");
    if (!options.enabled || !streamed_decode_graphs() || !input.ids || input.embeddings ||
        input.positions || input.lengths || input.cache_positions || input.image_mask || input.suffix ||
        state.speculative_start>=0) co_return;
    auto ids=input.ids.dense();
    if(ids.dim()==1) { ids=ids.unsqueeze(0); input.ids=ids; }
    const auto kind=options.projection;
    const bool hidden=kind==tyloi::models::DecodeOptions::Projection::hidden ||
                      kind==tyloi::models::DecodeOptions::Projection::hidden_only;
    auto* graph = dynamic_cast<Qwen4DecodeGraph*>(state.graph_registry.get());
    if(state.cache_pos==0 && (!hidden || !graph || !graph->reusable())) co_return;
    if(ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 || ids.size(1)>8 ||
        (state.cache_pos!=0 && state.positions.dim()!=1) || input.confirmed<0 ||
        (input.confirmed && input.confirmed>=ids.size(1))) co_return;
    if(options.routed && !tq_graph_supports_tokens(ids.size(1)))co_return;
    for(const auto& block:layers)
        if(block->device != device || ((hidden || options.routed) && (!routed_graph_supported(block->ffn) ||
            (block->ple && !block->ple->graph_supported())))) co_return;
    TYLOINFER_RUNTIME_CHECK(state.cache_pos+ids.size(1)<=max_position_embeddings(),"MFQ window exceeds context");
    state.decode_reserve_until=options.reserve_until;
    if(!state.graph_registry)state.graph_registry=std::make_unique<Qwen4DecodeGraph>(*this,state);
    graph = dynamic_cast<Qwen4DecodeGraph*>(state.graph_registry.get());
    if (!graph) throw std::logic_error("Qwen4 state has a different graph implementation");
    auto job=graph->run(*this,std::move(input),options);
    while(auto step=job.next()) co_yield std::move(step);
}

tyloi::StepSequence<tyloi::models::Decoded> qwen4_exp::Model::warmup(
        tyloi::models::State& value, const tyloi::models::DecodeOptions& options) const {
    auto& state=dynamic_cast<qwen4_exp::State&>(value);
    if (!state.belongs_to(*this)) throw std::invalid_argument("Qwen4 state belongs to a different model");
    const auto* reuse=tyloi::cuda::env_lookup("TYLOINFER_QWEN4_REUSE_GRAPHS");
    state.reuse_decode_graphs=!reuse || reuse[0]!='0';
    const auto* setting=tyloi::cuda::env_lookup("TYLOINFER_QWEN4_WARMUP_GRAPHS");
    if(!state.reuse_decode_graphs || (setting && setting[0]=='0') || !options.enabled ||
        max_position_embeddings()<32 || !streamed_decode_graphs()) co_return;
    for(std::size_t i=0;i<layers.size();++i) {
        const auto& block=*layers[i];
        if(block.device != device || !routed_graph_supported(block.ffn) || (block.qsa && !block.qsa->graph_capacity(*state.layers[i]->qsa)) ||
            (block.ple && !block.ple->graph_supported())) co_return;
    }
    struct Reset {
        qwen4_exp::State& state;
        ~Reset() {
            if(auto* graph=dynamic_cast<Qwen4DecodeGraph*>(state.graph_registry.get())) graph->startup_warmup=false;
            try { state.reset(1); } catch (...) {}
        }
    } reset{state};
    const auto start=std::chrono::steady_clock::now();
    co_yield tyloi::GraphPhase::warmup;
    const auto tensor_options=::tyloi::TensorOptions().device(device).dtype(::tyloi::kInt64);
    std::vector<int64_t> warm_ids(16);
    for(std::size_t i=0;i<warm_ids.size();++i)warm_ids[i]=int64_t((i*7919+17)%metadata.vocab_size);
    auto ids=::tyloi::cuda::tensor(warm_ids,tensor_options.device(::tyloi::kCPU)).to(device).reshape({1,16});
    co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit,nullptr,state.reset_growth(1)};
    state.reset(1);
    tyloi::models::ForwardOptions forward_options;
    forward_options.primary=device.index;
    for(int row=0;row<(options.routed?1:16);++row) {
        tyloi::models::Input input;
        input.ids=ids.narrow(1,row,1);
        co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit,nullptr,state.growth(input,forward_options)};
        (void)forward(std::move(input),state,forward_options);
        co_yield tyloi::StepState::advanced;
    }
    co_yield tyloi::GraphPhase::target;
    if(!state.graph_registry)state.graph_registry=std::make_unique<Qwen4DecodeGraph>(*this,state);
    auto* graph=dynamic_cast<Qwen4DecodeGraph*>(state.graph_registry.get());
    if (!graph) throw std::logic_error("Qwen4 state has a different graph implementation");
    graph->startup_warmup=true;
    for(auto projection : {tyloi::models::DecodeOptions::Projection::greedy,
                           tyloi::models::DecodeOptions::Projection::logits}) {
        tyloi::models::Input input;
        input.ids=ids.narrow(1,0,1);
        auto policy=options;
        policy.projection=projection;
        auto job=decode(std::move(input),state,{},policy);
        while(auto step=job.next()) {
            if(step.graph || step.workspace || step.layer || !step.value) co_yield std::move(step);
        }
    }
    state.reset(1);
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_get_current_cuda_stream()));
    std::cerr<<"runtime_qwen4_graph_warmup target_captures="<<state.decode_graph_captures<<" ms="
        <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<'\n';
    co_yield tyloi::GraphPhase::settle;
}

} // namespace tyloi::models::qwen4_exp

namespace tyloi::models::qwen4_exp {
using namespace ::tyloi::cuda;
struct Qwen4BatchDecodeGraph::Impl {
    using DenseTensor = ::tyloi::DenseTensor;
    qwen4_exp::Model& model;
    DecodeStream stream;
    std::unique_ptr<DecodeWindow> window;
    std::vector<State*> requests;
    std::vector<int64_t> capacities;
    std::vector<DenseTensor> positions, cache_positions, position_storage;
    std::vector<DenseTensor> gdn_addresses;
    std::vector<std::vector<std::shared_ptr<RowStage>>> ple;
    DenseTensor ids, logits;
    bool greedy = false;
    int64_t captures = 0, replays = 0;
    explicit Impl(qwen4_exp::Model& owner)
        : model(owner) {}
    void clear() {
        if (requests.empty() && !window) return;
        window.reset();
        for (auto* request : requests) {
            for (std::size_t i = 0; i < request->layers.size(); ++i) {
                const auto& weights = *model.layers[i];
                auto& state = *request->layers[i];
                if (weights.qsa) weights.qsa->leave_graph(*state.qsa);
            }
        }
        requests.clear(); capacities.clear(); positions.clear(); cache_positions.clear(); position_storage.clear(); ple.clear();gdn_addresses.clear();
        ids = {}; logits = {};
    }
};
Qwen4BatchDecodeGraph::Qwen4BatchDecodeGraph(qwen4_exp::Model& model)
    : graph_(std::make_unique<Impl>(model)) {}
Qwen4BatchDecodeGraph::~Qwen4BatchDecodeGraph() = default;
void Qwen4BatchDecodeGraph::clear() { graph_->clear(); }
int64_t Qwen4BatchDecodeGraph::captures() const { return graph_->captures; }
int64_t Qwen4BatchDecodeGraph::replays() const { return graph_->replays; }
tyloi::StepSequence<Qwen4BatchDecodeGraph::Plan> Qwen4BatchDecodeGraph::plan(
        std::vector<State*> requests, bool greedy) {
    auto& graph = *graph_;
    auto& model = graph.model;
    std::vector<tyloi::ops::Qsa::Graph::Request> rows;
    for (auto* request : requests) {
        tyloi::ops::Qsa::Graph::Request row;
        row.position = request->position();
        row.maximum = model.max_position_embeddings();
        for (std::size_t i = 0; i < model.layers.size(); ++i)
            if (model.layers[i]->qsa)
                row.layers.emplace_back(model.layers[i]->qsa.get(), request->layers[i]->qsa.get());
        rows.push_back(std::move(row));
    }
    auto attention = std::make_unique<tyloi::ops::Qsa::Graph>(std::move(rows), true);
    if (!attention->eligible()) co_return;
    for (auto* request : requests) {
        for (std::size_t i = 0; i < request->layers.size(); ++i) {
            const auto& weights = *model.layers[i];
            if (weights.ple && !weights.ple->graph_supported()) co_return;
        }
    }
    const auto batch = static_cast<int64_t>(requests.size());
    attention->admit();
    auto stream_scope=graph.stream.enter();
    bool reuse = graph.requests == requests && graph.window && graph.greedy == greedy;
    if (reuse) {
        std::vector<::tyloi::Tensor> owners;
        for(auto* request:requests)for(std::size_t i=0;i<model.layers.size();++i) {
            auto& state=*request->layers[i];const auto& weights=*model.layers[i];
            for(auto* tensor:state.graph_state(weights))owners.push_back(*tensor);
            if(weights.qsa)for(auto* tensor:weights.qsa->graph_state(*state.qsa))owners.push_back(*tensor);
        }
        reuse=graph.window->matches(owners);
    }
    if (reuse) for (int64_t row = 0; row < batch; ++row)
        reuse = reuse && requests[row]->position() + 1 <= graph.capacities[row];
    if (!reuse) {
        std::vector<std::pair<Device, std::size_t>> exit_growth;
        for (auto* request : graph.requests)
            for (std::size_t i = 0; i < model.layers.size(); ++i)
                if (model.layers[i]->qsa)
                    add_growth(exit_growth, model.layers[i]->device,
                        model.layers[i]->qsa->graph_exit_growth_bytes(*request->layers[i]->qsa));
        stream_scope.reset();
        if (!exit_growth.empty())
            co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr, std::move(exit_growth)};
        stream_scope = graph.stream.enter();
        graph.clear();
    }
    Plan plan;
    plan.rebuild = !graph.window;
    if (plan.rebuild) {
        attention->layout();
        for (std::size_t row = 0; row < requests.size(); ++row) {
            plan.capacities.push_back(attention->rows()[row].capacity);
            for (std::size_t i = 0; i < model.layers.size(); ++i)
                if (model.layers[i]->gdn)
                    add_growth(plan.growth, model.layers[i]->device,
                        model.layers[i]->gdn->request_cache_growth_bytes(requests[row]->layers[i]->gdn));
        }
    }
    for (const auto& [device, bytes] : attention->growth()) add_growth(plan.growth, device, bytes);
    plan.attention = std::move(attention);
    co_yield std::move(plan);
}

tyloi::StepSequence<::tyloi::Tensor> Qwen4BatchDecodeGraph::run(
        const std::vector<State*>& requests,
        const ::tyloi::Tensor& ids_handle,const ::tyloi::Tensor& host_ids_handle,bool greedy,const Plan& plan,bool prepared) {
    const auto& ids = ids_handle.dense();
    const auto& host_ids = host_ids_handle.dense();
    auto& graph = *graph_;
    auto& model = graph.model;
    const auto batch = static_cast<int64_t>(requests.size());
    auto stream_scope=graph.stream.enter();
    graph.greedy = greedy;
    stream_scope.reset();
    co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr, plan.attention->growth()};
    stream_scope = graph.stream.enter();
    plan.attention->materialize();
    if (plan.rebuild) {
        auto fixed = DecodeWindow::fixed_growth(ids.device(), ::tyloi::cuda::options::layer_timings(), false, 0);
        add_growth(fixed, ids.device(), std::size_t(batch) * sizeof(int64_t), 3);
        for (std::size_t i = 0; i < model.layers.size(); ++i) {
            const auto& layer = *model.layers[i];
            if (layer.ple) for (const auto& [device, bytes] : layer.ple->graph_stage_growth(1, ids.device()))
                add_growth(fixed, device, bytes, std::size_t(batch));
            if (layer.gdn && batch > 1 && ::tyloi::cuda::options::batch_attention_projections() &&
                    ::tyloi::cuda::options::gdn_independent_batch()) {
                add_growth(fixed, ids.device(), std::size_t(batch) * sizeof(int64_t), 3);
                add_growth(fixed, Device(kCPU), std::size_t(batch) * sizeof(int64_t), 3);
            }
        }
        stream_scope.reset();
        co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr, std::move(fixed)};
        stream_scope = graph.stream.enter();
        graph.requests = requests;
        graph.capacities = plan.capacities;
        graph.window = std::make_unique<DecodeWindow>(::tyloi::cuda::options::layer_timings(),true);
        graph.ids = graph.window->buffer(ids.sizes().vec(),ids.options()).dense();
        graph.ple.resize(batch);
        graph.position_storage.resize(batch);
        for (int64_t row = 0; row < batch; ++row) {
            auto& state = *requests[row];
            graph.positions.push_back(graph.window->buffer({1},ids.options()).dense());
            graph.cache_positions.push_back(graph.window->buffer({1},ids.options()).dense());
            graph.ple[row].resize(state.layers.size());
            for (std::size_t i = 0; i < state.layers.size(); ++i) {
                const auto& weights = *model.layers[i];
                auto& layer = *state.layers[i];
                if (weights.gdn) weights.gdn->initialize_request_cache(layer.gdn);
                if (weights.qsa) weights.qsa->prepare_graph(*layer.qsa,state.positions,graph.capacities[row]);
                if (weights.ple) graph.ple[row][i] = weights.ple->graph_stage(layer.ple,1,ids.device());
            }
        }
        graph.gdn_addresses.resize(model.layers.size());
        if(batch>1 && ::tyloi::cuda::options::batch_attention_projections() && ::tyloi::cuda::options::gdn_independent_batch())
            for(std::size_t i=0;i<model.layers.size();++i) {
                std::vector<std::pair<const tyloi::ops::Gdn*,tyloi::ops::Gdn::State*>> rows;
                for(auto* request:requests) {
                    const auto& weights = *model.layers[i];
                    auto& state = *request->layers[i];
                    if(weights.gdn) rows.emplace_back(weights.gdn.get(),&state.gdn);
                }
                if(rows.size()==requests.size())graph.gdn_addresses[i]=tyloi::ops::Gdn::prepare_independent_batch(rows).dense();
            }
        // Cached MFE owns its graph pool; dense execution needs its own.
        if (!prepared) graph.window->prepare_pool();
        graph.window->set_profile_geometry("batch_decode",static_cast<int>(batch));
    }
    graph.ids.copy_(ids);
    std::vector<std::tuple<qwen4_exp::Ple*,qwen4_exp::Ple::State*,qwen4_exp::NgramEmbedding::Context>> ple_contexts;
    for (int64_t row = 0; row < batch; ++row) {
        auto& state = *requests[row];
        mfq_qwen4_exp::fill_decode_positions(graph.cache_positions[row],graph.positions[row],
                                            state.cache_pos,state.decode_position_delta);
        for (std::size_t i = 0; i < state.layers.size(); ++i) {
            const auto& weights = *model.layers[i];
            auto& layer = *state.layers[i];
            if (weights.ple) {
                if(!graph.window->valid())ple_contexts.emplace_back(weights.ple.get(),&layer.ple,weights.ple->session_context(layer.ple));
                weights.ple->upload_graph(layer.ple,host_ids.narrow(0,row,1),*graph.ple[row][i]);
            }
        }
    }
    if (!graph.window->valid()) {
        struct Saved { ::tyloi::Tensor* tensor; ::tyloi::DenseTensor snapshot; };
        std::vector<Saved> recurrent;
        std::vector<::tyloi::Tensor*> tensors;
        std::vector<std::pair<Device, std::size_t>> copy_growth;
        for (auto* request : requests) for (std::size_t i = 0; i < model.layers.size(); ++i)
            for (auto* tensor : request->layers[i]->graph_state(*model.layers[i])) {
                tensors.push_back(tensor);
                add_growth(copy_growth, batch > 8 ? Device(kCPU) : tensor->device(), tensor->nbytes());
            }
        stream_scope.reset();
        co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr, std::move(copy_growth)};
        stream_scope = graph.stream.enter();
        const auto save=[&](::tyloi::Tensor* tensor) {
            // Capture restores these values after warmup. They are never read
            // by replay, so large batches can keep this temporary copy in RAM.
            if(tensor->defined())recurrent.push_back({tensor,batch>8 ? tensor->dense().to(::tyloi::kCPU) : tensor->dense().clone()});
        };
        for (auto* tensor : tensors) save(tensor);
        try {
            auto recording=graph.window->begin_capture();
            while(recording->next()) {
            std::exception_ptr failure;
            bool warm = false;
            try {
            graph.window->mark(-1,"window_begin");
            auto hidden = model.repeat_hidden(model.embed_tokens(graph.ids),model.metadata.hc_mult).dense();
            std::optional<tyloi::ops::Residual::Result> pending_residual;
            const auto chain_setting=::tyloi::cuda::options::gr_chain_ffn_post();
            const bool chain_post=(!chain_setting || *chain_setting) && batch<=8;
            for (std::size_t i = 0; i < model.layers.size(); ++i) {
                graph.window->mark(static_cast<int>(i),"block_begin");
                const auto& block = *model.layers[i];
                graph.window->set_profile_layer_kind(static_cast<int>(i),block.gdn?"gdn":"qsa");
                if (block.ple) {
                    graph.window->mark(static_cast<int>(i),"ple_begin");
                    std::vector<::tyloi::DenseTensor> positional;
                    for (int64_t row = 0; row < batch; ++row) {
                        auto x = hidden.narrow(0,row,1);
                        auto& layer = *requests[row]->layers[i];
                        graph.ple[row][i]->decode();
                        auto rows = graph.ple[row][i]->output().dense().reshape({1,1,model.hidden_size()});
                        positional.push_back(x + block.ple->forward_chunk(
                            layer.ple,x,graph.ids.narrow(0,row,1),true,{},rows).dense());
                    }
                    hidden = join_request_outputs(positional,0);
                    graph.window->mark(static_cast<int>(i),"ple_end");
                }
                auto mix = pending_residual
                    ? block.attention_gr.pre_after(hidden,pending_residual->residual,pending_residual->gate,tyloi::ops::ResidualRole::attention)
                    : block.attention_gr.pre(hidden,tyloi::ops::ResidualRole::attention);
                graph.window->mark(static_cast<int>(i),"attention_core_begin");
                const bool project_batch=batch>1 && ::tyloi::cuda::options::batch_attention_projections();
                const auto projections=project_batch ? (block.gdn ? block.gdn->project_inputs(mix.mixed)
                    : block.qsa->project_inputs(mix.mixed)) : std::vector<::tyloi::Tensor>{};
                std::vector<::tyloi::DenseTensor> attended;
                DenseTensor attention;
                if(project_batch && block.gdn && graph.gdn_addresses[i].defined())
                    attention=block.gdn->forward_independent_batch(requests[0]->layers[i]->gdn,mix.mixed,projections,graph.gdn_addresses[i]).dense();
                if(!attention.defined())for (int64_t row = 0; row < batch; ++row) {
                    auto& layer = *requests[row]->layers[i];
                    auto attention_input = mix.mixed.dense().narrow(0,row,1);
                    std::vector<::tyloi::Tensor> projected;
                    for(const auto& value:projections)projected.push_back(value.dense().narrow(0,row,1));
                    attended.push_back((block.gdn ? block.gdn->forward_chunk(layer.gdn,attention_input,true,std::move(projected),!project_batch)
                        : block.qsa->forward_graph(*layer.qsa,attention_input,graph.positions[row],graph.cache_positions[row],
                            std::move(projected),!project_batch)).dense());
                }
                if(!attention.defined())attention=join_request_outputs(attended,0);
                if(project_batch)attention=(block.gdn ? block.gdn->project_output(attention)
                    : block.qsa->project_output(attention)).dense();
                graph.window->mark(static_cast<int>(i),"attention_core_end");
                auto ffn_mix = block.ffn_gr.pre_after(std::move(attention),mix.residual,mix.gate,tyloi::ops::ResidualRole::ffn);
                graph.window->mark(static_cast<int>(i),"ffn_pre_end");
                auto ffn_output=tyloi::models::qwen4_exp::feed_forward(block.ffn,ffn_mix.mixed).dense();
                bool defer=false;
                if(chain_post && i+1<model.layers.size()) {
                    const auto& next=*model.layers[i+1];
                    defer=!next.ple && next.attention_gr.can_chain(ffn_mix.residual);
                }
                if(defer){hidden=std::move(ffn_output);pending_residual=std::move(ffn_mix);}
                else {hidden=block.ffn_gr.post(ffn_output,ffn_mix.residual,ffn_mix.gate).dense();pending_residual.reset();}
                graph.window->mark(static_cast<int>(i),"block_end");
            }
            auto finalized=qwen4_finalize_hidden(model, std::move(hidden),batch,1);
            graph.logits = greedy ? qwen4_next_token(model,qwen4_last_hidden(std::move(finalized)))
                .reshape({batch,1}) : qwen4_logits_from_hidden(model, std::move(finalized)).reshape({batch,-1});
            graph.window->mark(-1,"window_end");
            warm=recording->finish();
            } catch (...) { failure = std::current_exception(); }
            if (failure) {
                if (!recording->retry(failure)) std::rethrow_exception(failure);
                warm = true;
            }
            for (const auto& saved : recurrent) saved.tensor->dense().copy_(saved.snapshot);
            if(warm) {
                graph.logits={};
                recording->release_warm_outputs();
                recording->pause();
                stream_scope.reset();
                co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr, recording->growth()};
                stream_scope=graph.stream.enter();
                recording->resume();
            }
            }
            recording->complete();
            std::vector<::tyloi::Tensor> owners;
            for(auto* request:requests)for(std::size_t i=0;i<model.layers.size();++i) {
                auto& state=*request->layers[i];const auto& weights=*model.layers[i];
                for(auto* tensor:state.graph_state(weights))owners.push_back(*tensor);
                if(weights.qsa)for(auto* tensor:weights.qsa->graph_state(*state.qsa))owners.push_back(*tensor);
            }
            graph.window->retain_result(graph.logits);
            graph.window->retain(owners);
        }
        catch(const std::exception& error) {
            for (const auto& saved : recurrent) saved.tensor->dense().copy_(saved.snapshot);
            const std::string message=error.what();
            if(!dynamic_cast<const MfqResourceExhausted*>(&error) &&
               !message.starts_with("CUDA process VRAM limit exceeded:") &&
               !message.starts_with("CUDA VRAM limit exceeded:") &&
               !message.starts_with("CUDA allocation failed:") &&
               !message.starts_with("CUDA pinned allocation failed:"))throw;
            // Capture has not accepted a token. Retire its consumers before
            // restoring state, then let admission reject only the newest row.
            graph.clear();
            (void)cudaGetLastError();
            for(const auto& saved:recurrent)saved.tensor->dense().copy_(saved.snapshot);
            for(auto& [layer,layer_state,context]:ple_contexts)layer->restore_graph_context(*layer_state,std::move(context));
            throw;
        }
        ++graph.captures;
    }
    graph.window->run();
    ++graph.replays;
    for (std::size_t row=0;row<requests.size();++row) {
        auto& state = *requests[row];
        for (std::size_t i = 0; i < state.layers.size(); ++i)
            if (model.layers[i]->qsa) model.layers[i]->qsa->advance_graph(*state.layers[i]->qsa,1);
        state.positions=append_decode_positions(state.positions ? state.positions.dense() : DenseTensor{},
            graph.cache_positions[row], graph.position_storage[row],model.max_position_embeddings());
        ++state.cache_pos;
    }
    stream_scope.reset();
    co_yield ::tyloi::Tensor(graph.logits);
}

namespace {
class CudaBatch final : public Batch {
    struct GraphPlan final : models::Batch::Plan { Qwen4BatchDecodeGraph::Plan layout; };
public:
    CudaBatch(Model& model, const BatchConfig& config)
        : Batch(model, config, cuda::options::batch_attention_projections()) {
        if (config.graph) graph_ = std::make_unique<Qwen4BatchDecodeGraph>(model);
    }
    void release(BatchState&) override { clear(); }
    Tensor decode(std::span<const BatchRow> rows, Tensor ids, const Tensor& host_ids) override {
        auto policy = cuda::Executor::current_ffn();
        policy.batch_serial = active(rows).size() > 8;
        auto scope = cuda::Executor::current().enter(policy);
        return Batch::decode(rows, std::move(ids), host_ids);
    }
    StepSequence<std::unique_ptr<Plan>> plan(std::span<const BatchRow> rows, bool greedy) override {
        if (!graph_) co_return;
        co_yield GraphPhase::ready;
        auto job = graph_->plan(active(rows), greedy);
        std::optional<Qwen4BatchDecodeGraph::Plan> layout;
        while (auto step = job.next()) {
            if (step.graph) co_yield *step.graph;
            else if (step.value) layout = std::move(*step.value);
            else co_yield step.state;
        }
        if (!layout) co_return;
        auto plan = std::make_unique<GraphPlan>();
        plan->growth = layout->growth;
        plan->rebuild = layout->rebuild;
        plan->layout = std::move(*layout);
        co_yield std::unique_ptr<Plan>(std::move(plan));
    }
    StepSequence<std::optional<Tensor>> capture(std::span<const BatchRow> rows,
            const Tensor& ids, const Tensor& host_ids, bool greedy, const Plan& plan) override {
        const auto& layout = static_cast<const GraphPlan&>(plan).layout;
        bool prepared = false;
        if (layout.rebuild) co_yield GraphRequest{GraphPhase::prepare, &prepared};
        auto requests = active(rows);
        std::optional<Tensor> result;
        auto job = graph_->run(requests, ids, host_ids, greedy, layout, prepared);
        for (;;) {
            StepResult<Tensor> step;
            {
                auto policy = cuda::Executor::current_ffn();
                policy.batch_serial = requests.size() > 8;
                auto scope = cuda::Executor::current().enter(policy);
                step = job.next();
            }
            if (!step) break;
            if (step.graph) co_yield *step.graph;
            else if (step.value) result = std::move(*step.value);
            else co_yield step.state;
        }
        co_yield std::move(result);
    }
    void clear() override { if (graph_) graph_->clear(); }
    engine::Metrics metrics() const override {
        return {{"continuous_batching_cuda_graph_captures", double(graph_ ? graph_->captures() : 0)},
                {"continuous_batching_cuda_graph_replays", double(graph_ ? graph_->replays() : 0)}};
    }
private:
    std::unique_ptr<Qwen4BatchDecodeGraph> graph_;
};
} // namespace

std::unique_ptr<tyloi::models::Batch> Model::make_batch(const BatchConfig& config) const {
    return std::make_unique<CudaBatch>(const_cast<Model&>(*this), config);
}

// Keep each compressed expert image resident while advancing all prompt chunks
// through that layer. Attention and PLE still advance in causal token order.
tyloi::models::PrefillPlan qwen4_exp::Model::prefill_plan(
        const tyloi::models::State& value, const tyloi::models::PrefillOptions& policy) const {
    const auto& state = dynamic_cast<const qwen4_exp::State&>(value);
    if (!state.belongs_to(*this)) throw std::invalid_argument("Qwen4 state belongs to a different model");
    const auto enabled=::tyloi::cuda::options::moe_prefill_layer_major();
    const auto whole=::tyloi::cuda::options::moe_prefill_layer();
    if ((enabled && *enabled==0) || (whole && *whole==0) || (!policy.routed && !whole) ||
        policy.tokens<=128 || policy.tokens>16384 || state.speculative_start>=0 ||
        (!policy.routed && (!host_tokens || !policy.coordinated)) || policy.transformed ||
        policy.compute<=8 || policy.first!=0 || (policy.last!=0 && policy.last!=policy.tokens)) return {false,8192};
    for (const auto& block:layers)
        if (block->device.is_cpu() || block->device.index!=device.index) return {false,8192};
    return {true,policy.tokens};
}

tyloi::StepSequence<tyloi::models::Prefill> qwen4_exp::Model::prefill(
        tyloi::models::Input input, tyloi::models::State& value,
        const tyloi::models::ForwardOptions& options, const tyloi::models::PrefillOptions& policy) const {
    if (!prefill_plan(value, policy).layered) co_return;
    auto& state=dynamic_cast<qwen4_exp::State&>(value);
    const auto& model=*this;
    auto ids_handle=input.ids;
    auto compute_chunk=policy.compute;
    const bool raw_hidden=input.raw_hidden!=nullptr;
    const bool raw_hidden_host=policy.raw_host;
    const auto tap_layers=raw_hidden && model.predictor() ? model.predictor()->target_layers() : std::span<const int64_t>{};
    using DenseTensor=::tyloi::DenseTensor;
    auto ids = ids_handle.dense();
    TYLOINFER_RUNTIME_CHECK(ids.dim()==2 && ids.size(0)==1 &&
        prefill_plan(state,policy).layered,
        "layer-major prefill requires a bounded single-device Qwen4 chunk and whole-layer cache");
    TYLOINFER_RUNTIME_CHECK(compute_chunk>8,"layer-major compute chunk must exceed decode width");
    const auto tokens=ids.size(1);
    compute_chunk=std::min(compute_chunk,tokens);
    const auto cache_start=state.cache_pos;
    const auto maximum=model.metadata.max_position_embeddings;
    tyloi::models::require_model(tokens > 0 && cache_start >= 0 && cache_start <= maximum &&
        tokens <= maximum - cache_start, "Qwen4 requires unpadded causal cache positions");
    const Device cpu(kCPU);
    std::size_t ple_host = 0, previous = 0;
    for (const auto& block : model.layers) {
        std::size_t current = 0;
        if (block->ple && block->ple->graph_supported()) {
            for (const auto& [target, bytes] : block->ple->graph_stage_growth(1, compute_chunk, model.device))
                if (target.is_cpu()) current = ::tyloi::engine::prefill_budget_add(current, bytes);
        }
        ple_host = std::max(ple_host, ::tyloi::engine::prefill_budget_add(previous, current));
        previous = current;
    }
    const auto raw_rows = tap_layers.empty() ? model.config.streams : int64_t(tap_layers.size());
    const auto raw_host = raw_hidden && raw_hidden_host
        ? make_contiguous_view(nullptr, std::array<int64_t, 3>{tokens, model.config.hidden, raw_rows}, kFloat32, cpu).nbytes() : 0;
    std::vector<tyloi::ops::Qsa::State*> qsa_states;
    for (const auto& layer : state.layers)
        if (layer->qsa) qsa_states.push_back(layer->qsa.get());
    std::optional<tyloi::ops::Qsa::HostLoan> host_workspace(std::in_place, qsa_states,
        std::max(ple_host, raw_host), cache_start + tokens, compute_chunk);
    auto growth = state.growth(input, options);
    add_growth(growth, cpu, std::size_t(tokens) * sizeof(int64_t));
    add_growth(growth, cpu, host_workspace->bytes());
    co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr, std::move(growth)};

    ids=model.device_ids(ids).dense();
    auto capture = tyloi::ops::ProfileCapture::prefill_once(tokens);
    const auto reserve=std::size_t(1)<<30;
    const auto row_bytes=std::size_t(model.config.hidden)*model.config.streams*sizeof(float)*8;
    const auto sequence_reserve=std::size_t(tokens)*model.config.hidden*sizeof(float)*
        (model.config.streams*3+tap_layers.size()*2);
    capture.suspend();
    co_yield tyloi::Workspace{reserve+sequence_reserve, nullptr,
        tyloi::Workspace::Tile{tokens,row_bytes,&compute_chunk}};
    capture.resume();
    tyloi::cuda::ResidentAllocationScope persistent_preamble;
    if(cache_start==0)state.reset(1);
    state.begin_forward();
    auto current=::tyloi::cuda::arange(cache_start,cache_start+tokens,ids.options());
    if(state.decode_position_delta)current=current+state.decode_position_delta;
    auto prepared=std::move(current);
    if ((prepared.dim() == 2 || prepared.dim() == 3) && prepared.size(0) == 4)
        prepared = prepared.narrow(0,1,3);
    auto prepared_full=state.positions.defined()?
        ::tyloi::cuda::cat({state.positions.dense(),prepared},-1):prepared;
    tyloi::models::require_model((prepared.dim() == 1 || (prepared.dim() == 2 && prepared.size(0) == 3) ||
        (prepared.dim() == 3 && prepared.size(0) == 3 && prepared.size(1) == 1)) && prepared.size(-1) == tokens,
        "Qwen4 positions require [T], [3,T], [4,T], [3,B,T] or [4,B,T]");
    auto cpu_ids=ids.to(::tyloi::kCPU,::tyloi::kInt64).contiguous();
    host_workspace->prepare();
    const auto first_ids = cpu_ids.narrow(1, 0, compute_chunk);
    const auto prefetch = [&](std::size_t layer) {
        if (layer < model.layers.size() && model.layers[layer]->ple)
            model.layers[layer]->ple->prefetch_tokens(state.layers[layer]->ple, first_ids);
    };
    prefetch(0);
    persistent_preamble.finish();
    DenseTensor hidden=::tyloi::cuda::empty({1,tokens,model.metadata.hidden_size*model.metadata.hc_mult},
                           ids.options().dtype(::tyloi::kFloat16));
    for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
        capture.suspend(); co_yield tyloi::StepState::advanced; capture.resume();
        const auto count=std::min(compute_chunk,tokens-begin);
        auto input=model.repeat_hidden(model.embed_tokens(ids.narrow(1,begin,count)),model.metadata.hc_mult).dense();
        TYLOINFER_RUNTIME_CHECK(input.scalar_type()==hidden.scalar_type() &&
            input.size(-1)==hidden.size(-1),"layer-major embedding geometry changed");
        hidden.narrow(1,begin,count).copy_(input);
    }
    const auto trace_option=::tyloi::cuda::options::prefill_layer_major_trace();
    const bool trace=trace_option && *trace_option!=0;
    const auto batch_option=::tyloi::cuda::options::moe_prefill_expert_batch();
    const bool batch_experts=!batch_option || *batch_option!=0;
    const auto bulk_gr_option=::tyloi::cuda::options::gr_prefill_layer_batch();
    // Long prompts retain the established memory footprint unless explicitly
    // requested; the combined residual tensors have been verified to 16k.
    const bool bulk_gr=batch_experts && (bulk_gr_option?*bulk_gr_option:tokens<=16384);
    const auto bulk_gdn_option=::tyloi::cuda::options::gdn_prefill_layer_batch();
    // Allow 8k chat framing while keeping the extra whole-sequence QKV and
    // gate tensors out of the 16k service path, which also retains MTP weights.
    const bool bulk_gdn=bulk_gr && (bulk_gdn_option?*bulk_gdn_option:tokens<=9216);
    const auto reference_rows=batch_experts?std::min<int64_t>(1024,compute_chunk):0;
    const auto batch_tokens_option=::tyloi::cuda::options::moe_prefill_expert_batch_tokens();
    const int64_t batch_tokens=batch_tokens_option?*batch_tokens_option:8192;
    const auto append_tail_option=::tyloi::cuda::options::moe_prefill_append_tail();
    const bool append_tail=append_tail_option && *append_tail_option;
    const auto expert_batch_count=[&](int64_t remaining) {
        // Chat framing can leave a few tokens beyond the batch boundary.
        // Absorb at most one reference chunk instead of running a second
        // almost-empty routed GEMM for every layer.
        return append_tail && remaining<=batch_tokens+1024?remaining:std::min(batch_tokens,remaining);
    };
    TYLOINFER_RUNTIME_CHECK(!batch_experts || batch_tokens>=1024,"expert prefill batch must cover at least 1024 tokens");
    const auto audit_option=::tyloi::cuda::options::prefill_expert_batch_audit();
    const bool audit_batch=audit_option && *audit_option!=0;
    const auto started=std::chrono::steady_clock::now();
    std::optional<qwen4_exp::Taps> taps(std::in_place,tap_layers,model.metadata.hc_mult,model.device);
    int layer=0;
    for(auto& block:model.layers) {
        auto& layer_state=*state.layers[layer];
        capture.suspend(); co_yield tyloi::LayerEvent{tyloi::LayerEvent::Phase::begin, tyloi::LayerEvent::Mode::tiled}; capture.resume();
        prefetch(std::size_t(layer) + 1);
        DenseTensor next_hidden=hidden;
        DenseTensor ffn_inputs,ffn_gate;
        tyloi::ops::Residual::Result attention_result;
        DenseTensor attention_output;
        if(bulk_gr) {
            auto& qwen_block=*block;
            DenseTensor source=hidden;
            if(qwen_block.ple) {
                for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                    capture.suspend(); co_yield tyloi::StepState::advanced; capture.resume();
                    const auto count=std::min(compute_chunk,tokens-begin);
                    auto chunk_ids=ids.narrow(1,begin,count);
                    auto host_ids=cpu_ids.narrow(1,begin,count);
                    if (begin) qwen_block.ple->prefetch_tokens(layer_state.ple, host_ids);
                    auto input=qwen4_exp::prefill_ple(qwen_block,layer_state,
                        hidden.narrow(1,begin,count),chunk_ids,host_ids).dense();
                    if(begin==0)source=::tyloi::cuda::empty(hidden.sizes(),input.options());
                    source.narrow(1,begin,count).copy_(input);
                }
            }
            {
                tyloi::ops::ProfileRange range("prefill.gr_attention");
                attention_result = qwen_block.attention_gr.pre(source);
                range.finish();
            }
            if(bulk_gdn && qwen_block.gdn) {
                capture.suspend(); co_yield tyloi::StepState::advanced; capture.resume();
                const tyloi::ops::Gdn::ProjectionScope projection_rows(compute_chunk);
                tyloi::ops::ProfileRange range("prefill.gdn");
                attention_output = qwen_block.gdn->forward(layer_state.gdn,attention_result.mixed,true,0).dense();
                range.finish();
            }
        }
        for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
            capture.suspend(); co_yield tyloi::StepState::advanced; capture.resume();
            if(bulk_gdn && block->gdn)continue;
            const auto count=std::min(compute_chunk,tokens-begin);
            capture.resume();
            auto chunk_ids=ids.narrow(1,begin,count);
            auto host_ids=cpu_ids.narrow(1,begin,count);
            auto positions=prepared.narrow(-1,begin,count);
            auto full_positions=prepared_full.narrow(-1,0,cache_start+begin+count);
            if (!bulk_gr && begin && block->ple)
                block->ple->prefetch_tokens(layer_state.ple, host_ids);
            auto input=hidden.narrow(1,begin,count);
            DenseTensor output;
            if(bulk_gr) {
                auto& qwen_block=*block;
                auto attention_chunk=qwen4_exp::prefill_attention_core(qwen_block,layer_state,
                    attention_result.mixed.dense().narrow(1,begin,count),positions,full_positions).dense();
                if(!attention_output.defined()) {
                    auto shape=attention_chunk.sizes().vec();shape[1]=tokens;
                    attention_output=::tyloi::cuda::empty(shape,attention_chunk.options());
                }
                attention_output.narrow(1,begin,count).copy_(attention_chunk);
                continue;
            } else if(batch_experts) {
                auto& qwen_block=*block;
                auto mix=qwen4_exp::prefill_attention(qwen_block,layer_state,
                    input,chunk_ids,host_ids,positions,full_positions);
                if(!ffn_inputs.defined()) {
                    ffn_inputs=::tyloi::cuda::empty({1,tokens,model.hidden_size()},mix.mixed.dense().options().dtype(::tyloi::kFloat16));
                    ffn_gate=::tyloi::cuda::empty({1,tokens,mix.gate.size(-1)},mix.gate.dense().options());
                }
                // The MoE casts the router/FFN source to half at this boundary.
                ffn_inputs.narrow(1,begin,count).copy_(mix.mixed.dense().to(::tyloi::kFloat16));
                ffn_gate.narrow(1,begin,count).copy_(mix.gate.dense());
                output=mix.residual.dense();
            } else output=qwen4_exp::forward(*block,layer_state,
                input,chunk_ids,positions,full_positions,0,host_ids).dense();
            // Retain the original rounding at every layer boundary. The next
            // chunk is independent of this input once its cache writes retire.
            TYLOINFER_RUNTIME_CHECK(output.sizes()==input.sizes(),"layer-major decoder changed hidden shape");
            if(begin==0 && output.scalar_type()!=hidden.scalar_type()) {
                next_hidden=::tyloi::cuda::empty(hidden.sizes(),output.options());
                if(trace)std::cout<<"prefill_layer_major_hidden layer="<<layer
                    <<" input_bytes="<<hidden.nbytes()<<" output_bytes="<<next_hidden.nbytes()<<std::endl;
            }
            TYLOINFER_RUNTIME_CHECK(output.scalar_type()==next_hidden.scalar_type(),
                "layer-major decoder output dtype changed within a layer");
            next_hidden.narrow(1,begin,count).copy_(output);
        }
        // Attention's decoded matrices are no longer needed by FFN. Keep
        // reuse within each phase, without retaining both sets at peak memory.
        if(tokens>9216) { capture.suspend(); co_yield tyloi::LayerEvent{tyloi::LayerEvent::Phase::clear}; capture.resume(); }
        if(bulk_gr) {
            auto& qwen_block=*block;
            tyloi::ops::Residual::Result mix;
            {
                tyloi::ops::ProfileRange range("prefill.gr_ffn_pre");
                mix = qwen_block.ffn_gr.pre_after(attention_output,attention_result.residual,attention_result.gate);
                range.finish();
            }
            ffn_inputs=mix.mixed.dense().to(::tyloi::kFloat16);ffn_gate=mix.gate.dense();
            next_hidden=mix.residual.dense();
            attention_result={};attention_output={};
        }
        if(tokens>9216) { capture.suspend(); co_yield tyloi::LayerEvent{tyloi::LayerEvent::Phase::clear}; capture.resume(); }
        // The new residual owns the next layer's values. Prior hidden readers
        // have already been submitted before this stream-ordered release.
        if(tokens>9216)hidden={};
        if(batch_experts) {
            auto& qwen_block=*block;
            DenseTensor ffn_output;
            for(int64_t begin=0;begin<tokens;) {
                capture.suspend(); co_yield tyloi::StepState::advanced; capture.resume();
                auto count=expert_batch_count(tokens-begin);
                const auto tail=tokens-begin-count;
                // Keep tiny tails in the prefill path to retain shared prefill images.
                if(tail>0 && tail<=8)count-=16-tail;
                tyloi::ops::ProfileRange range("prefill.ffn");
                auto part=tyloi::models::qwen4_exp::feed_forward(qwen_block.ffn,
                    ffn_inputs.narrow(1,begin,count),reference_rows).dense();
                range.finish();
                if(count==tokens) {
                    ffn_output=std::move(part);
                    break;
                }
                if(!ffn_output.defined()) {
                    auto shape=part.sizes().vec();shape[1]=tokens;
                    ffn_output=::tyloi::cuda::empty(shape,part.options());
                }
                ffn_output.narrow(1,begin,count).copy_(part);
                begin+=count;
            }
            TYLOINFER_RUNTIME_CHECK(ffn_output.defined(),"expert prefill produced no output");
            if(audit_batch) {
                auto expected=::tyloi::cuda::empty(ffn_output.sizes(),ffn_output.options());
                for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                    const auto count=std::min(compute_chunk,tokens-begin);
                    expected.narrow(1,begin,count).copy_(
                        tyloi::models::qwen4_exp::feed_forward(qwen_block.ffn,ffn_inputs.narrow(1,begin,count),reference_rows).dense());
                }
                auto a=ffn_output.to(::tyloi::kCPU,::tyloi::kFloat32).contiguous();
                auto b=expected.to(::tyloi::kCPU,::tyloi::kFloat32).contiguous();
                double error=0,scale=0,maximum=0;
                for(int64_t i=0;i<a.numel();++i) {
                    const double delta=double(a.data_ptr<float>()[i])-b.data_ptr<float>()[i];
                    error+=delta*delta;scale+=double(b.data_ptr<float>()[i])*b.data_ptr<float>()[i];
                    maximum=std::max(maximum,std::abs(delta));
                }
                std::cout<<"prefill_expert_batch_layer_oracle layer="<<layer<<" tokens="<<tokens
                    <<" relative_rms="<<std::sqrt(error/std::max(1e-30,scale))
                    <<" maximum_absolute="<<maximum<<std::endl;
                // Diagnostic only: continue with the reference so later
                // layers are tested with their original input distribution.
                ffn_output=std::move(expected);
            }
            DenseTensor post_hidden=next_hidden;
            for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                capture.suspend(); co_yield tyloi::StepState::advanced; capture.resume();
                const auto count=std::min(compute_chunk,tokens-begin);
                if(mfq_qwen4_exp::try_gated_residual_post_out(ffn_output.narrow(1,begin,count),
                    next_hidden.narrow(1,begin,count),ffn_gate.narrow(1,begin,count),
                    post_hidden.narrow(1,begin,count),model.metadata.hc_mult))continue;
                auto output=qwen_block.ffn_gr.post(ffn_output.narrow(1,begin,count),
                    next_hidden.narrow(1,begin,count),ffn_gate.narrow(1,begin,count));
                if(begin==0 && output.scalar_type()!=next_hidden.scalar_type())
                    post_hidden=::tyloi::cuda::empty(next_hidden.sizes(),output.options());
                TYLOINFER_RUNTIME_CHECK(output.scalar_type()==post_hidden.scalar_type(),
                    "expert-batched residual dtype changed");
                post_hidden.narrow(1,begin,count).copy_(output.dense());
            }
            next_hidden=std::move(post_hidden);
        }
        hidden=std::move(next_hidden);
        taps->capture(layer,hidden);
        capture.suspend(); co_yield tyloi::LayerEvent{tyloi::LayerEvent::Phase::end}; capture.resume();
        ++layer;
        if(trace && (layer%4==0 || layer==int(model.layers.size()))) {
            tyloi::models::Prefill report;
            report.phase=tyloi::models::Prefill::Phase::progress;
            report.progress=tyloi::models::Prefill::Progress{
                model.device,tokens,compute_chunk,
                layer,static_cast<std::int64_t>(model.layers.size()),started};
            capture.suspend(); co_yield std::move(report); capture.resume();
        }
    }
    capture.resume();
    const auto begin=((tokens-1)/compute_chunk)*compute_chunk;
    auto finalized=qwen4_finalize_hidden(model, hidden.narrow(1,begin,tokens-begin),1,tokens-begin);
    finalized=qwen4_last_hidden(std::move(finalized)).unsqueeze(1);
    Tensor raw;
    if(raw_hidden) {
        raw=taps->finish(hidden);
        if(raw_hidden_host)raw=raw.dense().to(::tyloi::kCPU);
    }
    taps.reset();
    hidden={};
    tyloi::models::Prefill result;
    result.mode=tyloi::models::Prefill::Mode::layered;
    result.phase=tyloi::models::Prefill::Phase::computed;
    capture.suspend(); co_yield result; capture.resume();
    result.hidden=std::move(finalized);
    result.raw=std::move(raw);
    capture.suspend();
    host_workspace->collect();
    co_yield tyloi::Workspace{0,&result.hidden,std::nullopt,raw_hidden && !raw_hidden_host ? &result.raw : nullptr};
    capture.resume();
    state.finish_forward(prepared_full,1,tokens);
    state.cache_pos=cache_start+tokens;
    host_workspace.reset();
    result.phase=tyloi::models::Prefill::Phase::complete;
    capture.suspend(); co_yield std::move(result); capture.resume();
}

} // namespace tyloi::models::qwen4_exp

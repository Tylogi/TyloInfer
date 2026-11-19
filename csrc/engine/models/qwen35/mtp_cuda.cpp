#include "engine/models/qwen35/mtp.h"
#include "engine/models/qwen35/model.h"
#include "engine/backends/cuda/executor/graph/decode.h"
#include "engine/backends/cuda/executor/graph/window.h"
#include "engine/backends/cuda/executor/graph_pool_lease.h"
#include "engine/backends/cuda/ops/attention/scratch.h"
#include "engine/generation/prefill/budget.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace tyloi::models::qwen35 {

struct MtpGraphs {
    struct Cell {
        explicit Cell(cudaStream_t stream) : window(stream) {}
        Tensor hidden, ids, positions, lengths, result, logits;
        cuda::DecodeWindow window;
    };
    explicit MtpGraphs(cuda::StreamHandle value, const CausalModel* owner)
        : stream(std::move(value)), target(owner) {}
    cuda::StreamHandle stream;
    const CausalModel* target;
    std::shared_ptr<cuda::GraphPoolLease> pool;
    std::map<std::tuple<int, int64_t, MtpStepMode, int, int64_t>, std::unique_ptr<Cell>> cells;
};

StepSequence<models::MtpDraft> Mtp::greedy_draft(const CausalModel& target,
        models::MtpRow row, int depth, const ForwardOptions& options) const {
    if (depth < 2 || depth > engine::mtp::kMaximumReportedDraftDepth) co_return;
    auto job = graph(target, std::move(row), MtpStepMode::tail, depth, options);
    while (auto step = job.next()) {
        if (step.graph) co_yield std::move(*step.graph);
        else if (step.value) co_yield MtpDraft{std::move(step.value->output), step.value->initialized};
    }
}

StepSequence<models::MtpStep> Mtp::graph_step(const CausalModel& target,
        MtpRow row, MtpStepMode mode, const ForwardOptions& options) const {
    auto job = graph(target, std::move(row), mode, 0, options);
    while (auto step = job.next()) {
        if (step.graph) co_yield std::move(*step.graph);
        else if (step.value) {
            auto hidden = step.value->output ? ops::clone(step.value->output) : Tensor{};
            co_yield MtpStep{hidden, hidden, std::move(step.value->logits)};
        }
    }
}

StepSequence<Mtp::GraphResult> Mtp::graph(const CausalModel& target, MtpRow row,
        MtpStepMode mode, int depth, const ForwardOptions& options) const {
    auto* request = row.state ? dynamic_cast<MtpState*>(row.state) : nullptr;
    const auto* backbone = dynamic_cast<const Model*>(&target);
    if (!request || !request->belongs_to(*this) || !backbone || target.predictor() != this ||
        !model_.device.is_cuda() || backbone->device != model_.device ||
        (backbone->rope.sections && backbone->rope.sections.numel()) ||
        std::any_of(backbone->layers.begin(), backbone->layers.end(), [&](const auto& layer) {
            return layer.device != model_.device;
        }) ||
        options.primary != model_.device.index || options.trace || model_.config.num_experts ||
        model_.layers.empty() || !model_.config.mrope_sections.empty() ||
        (model_.rope.sections && model_.rope.sections.numel()) || row.positions ||
        !row.hidden || row.hidden.is_encoded() || row.hidden.device() != model_.device ||
        row.hidden.dim() != 3 || row.hidden.size(0) != 1 ||
        row.hidden.size(2) != model_.config.hidden_size ||
        !row.ids || row.ids.is_encoded() || row.ids.device() != model_.device ||
        row.ids.scalar_type() != kInt64 || row.ids.dim() != 2 || row.ids.size(0) != 1 ||
        row.ids.size(1) < 1 || row.ids.size(1) > 1 + engine::mtp::kMaximumReportedDraftDepth ||
        row.hidden.size(1) != row.ids.size(1) || request->position() <= 0)
        co_return;

    const auto start = request->position(), tokens = row.ids.size(1);
    if (depth == 0 && mode == MtpStepMode::tail && tokens == 1) mode = MtpStepMode::full;
    if (mode != MtpStepMode::full && model_.layers.size() != 1) co_return;
    const auto storage = request->storage();
    if (storage.size() != model_.layers.size() * 2) co_return;
    const auto kv = request->views();
    auto extent = std::min<std::int64_t>(model_.config.max_position_embeddings, std::numeric_limits<int>::max());
    for (std::size_t i = 0; i < model_.layers.size(); ++i) {
        const auto& attention = model_.layers[i].attention;
        const auto& cache = kv[i];
        if (attention.sliding || attention.kv_quantization.enabled() || attention.attention_head_dim <= 0 ||
            attention.attention_head_dim > 512 || attention.kv_heads <= 0 ||
            attention.attention_heads <= 0 || attention.attention_heads % attention.kv_heads ||
            (attention.attention_rope.sections && attention.attention_rope.sections.numel()) ||
            !cache.defined() || cache.paged() || cache.ring || cache.quantization.enabled() ||
            cache.batch != 1 || cache.physical_key.device() != model_.device ||
            cache.physical_value.device() != model_.device || cache.physical_key.is_encoded() ||
            cache.physical_value.is_encoded() || !cache.physical_key.is_contiguous() ||
            !cache.physical_value.is_contiguous() || cache.physical_key.dim() != 4 ||
            cache.physical_value.dim() != 4 ||
            cache.physical_key.size(0) != 1 || cache.physical_key.size(1) != attention.kv_heads ||
            cache.physical_key.size(3) != attention.attention_head_dim ||
            cache.physical_value.size(0) != 1 || cache.physical_value.size(1) != attention.kv_heads ||
            cache.physical_value.size(2) != cache.physical_key.size(2) ||
            cache.physical_value.size(3) != attention.attention_head_dim ||
            cache.physical_key.scalar_type() != (attention.official_bf16 ? kBFloat16 : kFloat16) ||
            cache.physical_value.scalar_type() != cache.physical_key.scalar_type()) co_return;
        extent = std::min(extent, cache.physical_key.size(2));
    }
    const auto written = tokens + std::max(0, depth - 1);
    if (start > extent || written > extent - start ||
        mfq_current_cuda_device() != model_.device.index || !mfq_cuda_graph_capture_supported()) co_return;
    const auto parts = ::decode_graph_attention_parts(extent, cuda::AttentionScratch::kMaxParts);

    const auto stream = mfq_get_current_cuda_stream(model_.device.index);
    auto& graphs = request->graphs;
    if (graphs && (graphs->stream.stream() != stream.stream() || graphs->target != &target)) graphs.reset();
    if (!graphs) graphs = std::make_shared<MtpGraphs>(stream, &target);
    const auto key = std::tuple{int(row.hidden.scalar_type()), tokens, mode, depth, extent};
    auto& slot = graphs->cells[key];
    if (slot && !slot->window.matches(storage)) slot.reset();
    const bool capture = !slot;
    if (capture) {
        co_yield GraphPhase::ready;
        auto bytes = engine::prefill_budget_add(row.hidden.nbytes(), (2 * tokens + 1) * sizeof(int64_t));
        std::vector<std::pair<Device, std::size_t>> growth{{model_.device, bytes}};
        co_yield GraphRequest{GraphPhase::admit, nullptr, std::move(growth)};
        try {
            if (!graphs->pool)
                graphs->pool = cuda::GraphPoolLease::acquire(cuda::default_context(model_.device.index), stream.stream());
            slot = std::make_unique<MtpGraphs::Cell>(stream.stream());
            const auto integer = TensorOptions().device(model_.device).dtype(kInt64);
            slot->hidden = slot->window.buffer({1, tokens, model_.config.hidden_size}, row.hidden.options());
            slot->ids = slot->window.buffer({1, tokens}, integer);
            slot->positions = slot->window.buffer({tokens}, integer);
            slot->lengths = slot->window.buffer({1}, integer);
        } catch (...) {
            request->release_graphs();
            throw;
        }
    }
    auto& cell = *slot;
    try {
        ops::copy_into(cell.hidden, row.hidden);
        ops::copy_into(cell.ids, row.ids);
        ops::copy_into(cell.positions, ops::arange(start, start + tokens, model_.device));
        ops::fill(cell.lengths, start + tokens);
        if (capture) {
            auto recording = cell.window.begin_capture();
            while (recording->next()) {
                std::exception_ptr failure;
                bool warm = false;
                try {
                    auto evaluate = [&](const Tensor& hidden, const Tensor& ids,
                                        const Tensor& positions, const Tensor& lengths,
                                        int64_t offset, bool history_only = false) {
                        auto embedded = ops::embedding(ids, model_.embedding);
                        auto value = ops::cast(ops::contiguous(ops::cast(embedded, kFloat16)), hidden.scalar_type());
                        AttentionStep step;
                        step.positions = step.cache_positions = positions;
                        step.sequence_lengths = lengths;
                        step.cache_position = start + offset;
                        step.planned_kv_length = extent;
                        step.decode_attention_parts = parts;
                        step.contiguous_tail = true;
                        return model_.forward(hidden, value, step, kv, model_.rope, nullptr, -1, history_only);
                    };
                    if (mode == MtpStepMode::history) {
                        (void)evaluate(cell.hidden, cell.ids, cell.positions, {}, 0, true);
                    } else if (mode == MtpStepMode::full) {
                        cell.result = evaluate(cell.hidden, cell.ids, cell.positions, cell.lengths, 0);
                    } else {
                        if (tokens > 1)
                            (void)evaluate(ops::slice(cell.hidden, 1, 0, tokens - 1),
                                ops::slice(cell.ids, 1, 0, tokens - 1),
                                ops::slice(cell.positions, 0, 0, tokens - 1), {}, 0, true);
                        cell.result = evaluate(ops::slice(cell.hidden, 1, tokens - 1, tokens),
                            ops::slice(cell.ids, 1, tokens - 1, tokens),
                            ops::slice(cell.positions, 0, tokens - 1, tokens), cell.lengths, tokens - 1);
                    }
                    if (depth > 0) {
                        auto output = std::move(cell.result);
                        std::vector<Tensor> ids{ops::slice(cell.ids, 1, tokens - 1, tokens)};
                        for (int index = 0; index < depth; ++index) {
                            auto logits = ops::reshape(target.logits(output), {1, -1});
                            auto token = ops::reshape(ops::sample_greedy(ops::contiguous(logits)), {1, 1});
                            ids.push_back(token);
                            if (index + 1 < depth) {
                                auto position = ops::add(ops::slice(cell.positions, 0, tokens - 1, tokens), index + 1);
                                auto lengths = ops::add(cell.lengths, index + 1);
                                output = evaluate(output, token, position, lengths, tokens + index);
                            }
                        }
                        cell.result = ops::cat(ids, 1);
                    } else if (cell.result) {
                        const auto end = cell.result.size(1);
                        cell.logits = target.logits(ops::slice(cell.result, 1, end - 1, end));
                    }
                    warm = recording->finish();
                } catch (...) { failure = std::current_exception(); }
                if (failure) {
                    if (!recording->retry(failure)) std::rethrow_exception(failure);
                    warm = true;
                }
                if (warm) {
                    cell.result = {};
                    cell.logits = {};
                    recording->release_warm_outputs();
                    recording->pause();
                    co_yield GraphRequest{GraphPhase::admit, nullptr, recording->growth()};
                    recording->resume();
                }
            }
            recording->complete();
            cell.window.retain(storage);
        }
        if (!cell.window.enqueue_if_device_only()) cell.window.run();
        request->advance(written);
    } catch (...) {
        request->release_graphs();
        throw;
    }
    co_yield GraphResult{cell.result, cell.logits, capture};
}

} // namespace tyloi::models::qwen35

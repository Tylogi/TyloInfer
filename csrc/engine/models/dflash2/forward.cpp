#include "model.h"

#include "engine/models/causal_model.h"
#include "engine/ops/ops.h"
#include "engine/backends/cuda/executor/graph/window.h"
#include "engine/backends/cuda/executor/graph_pool_lease.h"
#include "engine/backends/cuda/kernels/attention/dflash2/attention.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/device.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/ops/sampling/launch.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>

namespace tyloi::models::dflash2 {

struct DraftGraph {
    struct Branch {
        Branch(int device, int layers) : stream(cuda::stream_from_pool(false, device)), ready(layers) {}

        cuda::StreamHandle stream;
        cuda::Event fork, joined;
        std::vector<cuda::Event> ready;
        std::shared_ptr<void> pool;
    };

    DraftGraph(cuda::StreamHandle stream, const std::shared_ptr<cuda::Context>& context)
        : stream(std::move(stream)), pool(cuda::GraphPoolLease::acquire(context, this->stream.stream())),
          window(this->stream.stream()) {}

    cuda::StreamHandle stream;
    std::shared_ptr<cuda::GraphPoolLease> pool;
    Tensor ids, positions;
    Tensor features, context_positions;
    std::unique_ptr<Branch> branch;
    DenseTensor hidden;
    cuda::DecodeWindow window;
};

namespace {
namespace raw = tyloi::cuda::dflash2;

void record(cuda::StreamHandle stream, std::initializer_list<DenseTensor> tensors) {
    for (const auto& tensor : tensors)
        if (tensor.defined()) tensor.record_stream(reinterpret_cast<std::uintptr_t>(stream.stream()), stream.lifetime());
}

DenseTensor empty(std::span<const std::int64_t> shape, TensorOptions options) {
    return ops::empty(shape, options).dense();
}
DenseTensor empty(std::initializer_list<std::int64_t> shape, TensorOptions options) {
    return empty(std::span<const std::int64_t>(shape.begin(), shape.size()), options);
}

struct Activation {
    DenseTensor values, codes, scales;
};

Activation activation(DenseTensor values) {
    Activation result{std::move(values), {}, {}};
    const auto width = result.values.size(-1), rows = result.values.numel() / width;
    if (rows >= 1 && rows <= 8 && width % 32 == 0) {
        const auto padded = (width + 511) / 512 * 512;
        result.codes = empty({8 * padded}, result.values.options().dtype(ScalarType::uint8));
        result.scales = empty({8 * (padded / 32)}, result.values.options());
    }
    return result;
}

std::vector<DenseTensor> project(std::initializer_list<const Matrix*> weights,
                                DenseTensor input, cuda::StreamHandle stream,
                                const Activation* prepared = nullptr) {
    const auto& first = **weights.begin();
    if (input.dim() < 1 || input.size(-1) != first.columns || input.numel() < first.columns)
        throw std::invalid_argument("DFlash2 projection input differs");
    input = input.to(ScalarType::float32).contiguous();
    const auto rows = input.numel() / first.columns;
    if (rows > std::numeric_limits<int>::max() - 7)
        throw std::invalid_argument("DFlash2 projection rows overflow");
    const auto padded_rows = (rows + 7) / 8 * 8;
    const auto padded_columns = (std::int64_t(first.columns) + 511) / 512 * 512;
    auto codes = prepared && prepared->codes.defined() ? prepared->codes :
        empty({padded_rows * padded_columns}, input.options().dtype(ScalarType::uint8));
    auto scales = prepared && prepared->scales.defined() ? prepared->scales :
        empty({padded_rows * (padded_columns / 32)}, input.options());
    record(stream, {input, codes, scales});
    if (!prepared || !prepared->codes.defined())
        raw::q8_quantize(input.data_ptr<float>(), codes.data_ptr(), scales.data_ptr<float>(),
                         int(rows), first.columns, stream.stream());
    std::vector<DenseTensor> outputs;
    raw::Q8TiledMatrix matrices[3];
    int index = 0;
    for (const auto* weight : weights) {
        if (weight->q4 != first.q4 || weight->columns != first.columns || weight->tiles.device() != first.tiles.device())
            throw std::invalid_argument("DFlash2 projection group differs");
        auto shape = input.sizes().vec();
        shape.back() = weight->rows;
        outputs.push_back(empty(shape, input.options()));
        matrices[index++] = {weight->tiles.data_ptr(), outputs.back().data_ptr<float>(), weight->rows, weight->padded_rows};
    }
    if (first.q4) raw::q4_gemm(matrices, index, codes.data_ptr(), scales.data_ptr<float>(), int(rows), first.columns, stream.stream());
    else raw::q8_gemm(matrices, index, codes.data_ptr(), scales.data_ptr<float>(), int(rows), first.columns, stream.stream());
    return outputs;
}

DenseTensor project(const Matrix& weight, DenseTensor input, cuda::StreamHandle stream, const Activation* prepared = nullptr) {
    return project({&weight}, std::move(input), stream, prepared).front();
}

Activation normalize(DenseTensor input, const DenseTensor& norm, float epsilon, cuda::StreamHandle stream) {
    input = input.to(ScalarType::float32).contiguous();
    auto result = activation(empty(input.sizes(), input.options()));
    const int width = int(input.size(-1)), rows = int(input.numel() / width);
    record(stream, {input, result.values, result.codes, result.scales});
    if (result.codes.defined())
        raw::rms_norm_q8(input.data_ptr<float>(), norm.data_ptr<float>(), result.values.data_ptr<float>(),
            result.codes.data_ptr(), result.scales.data_ptr<float>(), rows, width, epsilon, stream.stream());
    else raw::rms_norm(input.data_ptr<float>(), norm.data_ptr<float>(), result.values.data_ptr<float>(), rows, width, epsilon, stream.stream());
    return result;
}

Activation convolve(const DenseTensor& input, const DenseTensor& dynamic, const DenseTensor& base,
                    const DenseTensor& residual, int group, bool finish, cuda::StreamHandle stream) {
    auto result = finish ? Activation{empty(input.sizes(), input.options()), {}, {}} : activation(empty(input.sizes(), input.options()));
    record(stream, {input, dynamic, base, residual, result.values, result.codes, result.scales});
    if (result.codes.defined())
        raw::convolution_q8(input.data_ptr<float>(), dynamic.data_ptr<float>(), base.data_ptr<float>(), nullptr,
            result.values.data_ptr<float>(), result.codes.data_ptr(), result.scales.data_ptr<float>(),
            int(input.size(1)), int(input.size(2)), group, false, stream.stream());
    else raw::convolution(input.data_ptr<float>(), dynamic.data_ptr<float>(), base.data_ptr<float>(),
            finish ? residual.data_ptr<float>() : nullptr, result.values.data_ptr<float>(),
            int(input.size(1)), int(input.size(2)), group, finish, stream.stream());
    return result;
}

DenseTensor table(const DenseTensor& positions, const Config& c, cuda::StreamHandle stream) {
    auto result = empty({positions.numel(), c.dimension}, positions.options().dtype(ScalarType::float32));
    record(stream, {positions, result});
    raw::rotary_table(positions.data_ptr<std::int64_t>(), result.data_ptr<float>(), int(positions.numel()), c.dimension, c.theta, stream.stream());
    return result;
}

DenseTensor rotate(const DenseTensor& input, const DenseTensor& norm, const DenseTensor& rotary,
                   int heads, const Config& c, cuda::StreamHandle stream) {
    auto result = empty(input.sizes(), input.options());
    record(stream, {input, norm, rotary, result});
    raw::norm_rope_tabled(input.data_ptr<float>(), norm.data_ptr<float>(), rotary.data_ptr<float>(), result.data_ptr<float>(),
        int(input.size(1)), heads, c.dimension, c.epsilon, stream.stream());
    return result;
}

DenseTensor block_hidden(const Draft& model, State& state, const DenseTensor& ids,
                         const DenseTensor& positions, cuda::StreamHandle stream,
                         const DraftGraph::Branch* branch = nullptr) {
    const auto& c = model.config;
    auto embedded = ops::contiguous(model.target.embedding(model.mask_embedding.defined() ? ids.narrow(1, 0, 1) : ids, model.device.index)).dense();
    if (embedded.numel() != std::int64_t(c.hidden) * (model.mask_embedding.defined() ? 1 : c.block) || embedded.device() != model.device)
        throw std::invalid_argument("DFlash2 target embedding shape or device differs");
    raw::DraftEmbeddingType type;
    switch (embedded.scalar_type()) {
        case ScalarType::float32: type = raw::DraftEmbeddingType::Float32; break;
        case ScalarType::float16: type = raw::DraftEmbeddingType::Float16; break;
        case ScalarType::bfloat16: type = raw::DraftEmbeddingType::BFloat16; break;
        default: throw std::invalid_argument("DFlash2 target embedding type differs");
    }
    auto hidden = empty({1, c.block, c.hidden}, embedded.options().dtype(ScalarType::float32));
    record(stream, {embedded, model.mask_embedding, hidden});
    raw::prepare_draft_hidden(embedded.data_ptr(), type, model.mask_embedding.defined() ? model.mask_embedding.data_ptr<float>() : nullptr,
                              hidden.data_ptr<float>(), c.block, c.hidden, stream.stream());
    auto rotary = table(positions, c, stream);
    const int chunks = (c.window - 1 + c.block + 127) / 128;
    auto partial = empty({chunks, c.block, c.heads, c.dimension + 2}, hidden.options());
    for (int i = 0; i < c.layers; ++i) {
        const auto& layer = model.layers[i];
        auto norm = normalize(hidden, layer.input_norm, c.epsilon, stream);
        auto dynamic = project(layer.dynamic_attention, norm.values, stream, &norm);
        auto conv = convolve(norm.values, dynamic, layer.attention_base, {}, c.group, false, stream);
        auto qkv = project({&layer.q, &layer.k, &layer.v}, conv.values, stream, &conv);
        auto query = rotate(qkv[0], layer.q_norm, rotary, c.heads, c, stream);
        auto key = rotate(qkv[1], layer.k_norm, rotary, c.kv_heads, c, stream);
        auto attended = empty({1, c.block, c.heads * c.dimension}, hidden.options());
        record(stream, {query, key, qkv[2], state.keys[i], state.values[i], partial, attended, positions});
        if (branch) TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), branch->ready[i].get(), 0));
        raw::attention(query.data_ptr<float>(), state.keys[i].data_ptr(), state.values[i].data_ptr(),
            key.data_ptr<float>(), qkv[2].data_ptr<float>(), partial.data_ptr<float>(), attended.data_ptr<float>(),
            c.block, c.heads, c.kv_heads, c.dimension, c.window, 0, stream.stream(), positions.data_ptr<std::int64_t>(), model.compact_kv);
        auto output = project(layer.output, attended, stream);
        auto residual = convolve(output, dynamic, layer.attention_base, hidden, c.group, true, stream).values;
        norm = normalize(residual, layer.post_norm, c.epsilon, stream);
        dynamic = project(layer.dynamic_mlp, norm.values, stream, &norm);
        conv = convolve(norm.values, dynamic, layer.mlp_base, {}, c.group, false, stream);
        auto gu = project({&layer.gate, &layer.up}, conv.values, stream, &conv);
        auto activated = activation(empty({1, c.block, c.intermediate}, hidden.options()));
        record(stream, {gu[0], gu[1], activated.values, activated.codes, activated.scales});
        if (activated.codes.defined())
            raw::swiglu_float_q8(gu[0].data_ptr<float>(), gu[1].data_ptr<float>(), activated.values.data_ptr<float>(),
                activated.codes.data_ptr(), activated.scales.data_ptr<float>(), c.block, c.intermediate, stream.stream());
        else raw::swiglu_float(gu[0].data_ptr<float>(), gu[1].data_ptr<float>(), activated.values.data_ptr<float>(), activated.values.numel(), stream.stream());
        output = project(layer.down, activated.values, stream, &activated);
        hidden = convolve(output, dynamic, layer.mlp_base, residual, c.group, true, stream).values;
    }
    return hidden;
}

void validate_context(const Config& c, const State& s, const Tensor& hidden, std::int64_t start) {
    if (start < s.pos || hidden.dim() != 3 || hidden.size(0) != 1 || hidden.size(1) < 1 ||
        hidden.size(2) != c.hidden * std::int64_t(c.target_layers.size()) ||
        (start != s.pos && hidden.size(1) < c.window) ||
        hidden.size(1) > std::numeric_limits<std::int64_t>::max() - start)
        throw std::invalid_argument("DFlash2 target features differ");
}

void store_context(const Draft& model, State& s, const DenseTensor& input, const DenseTensor& positions,
                   cuda::StreamHandle stream, DraftGraph::Branch* branch = nullptr) {
    const auto& c = model.config;
    auto projected = project(model.context_projection, input, stream);
    auto norm = normalize(projected, model.hidden_norm, c.epsilon, stream);
    auto rotary = table(positions, c, stream);
    for (int i = 0; i < c.layers; ++i) {
        const auto& layer = model.layers[i];
        auto kv = project({&layer.k, &layer.v}, norm.values, stream, &norm);
        record(stream, {kv[0], kv[1], rotary, positions, s.keys[i], s.values[i]});
        raw::store_context(kv[0].data_ptr<float>(), kv[1].data_ptr<float>(), layer.k_norm.data_ptr<float>(),
            rotary.data_ptr<float>(), positions.data_ptr<std::int64_t>(), s.keys[i].data_ptr(), s.values[i].data_ptr(),
            int(positions.numel()), c.kv_heads, c.dimension, c.window, c.epsilon, stream.stream(), model.compact_kv);
        if (branch) branch->ready[i].record(stream.stream());
    }
}

void commit_context(State& s, std::int64_t end) {
    s.pos = end;
    s.compact = {};
    s.selector = {};
    s.successor_rows = {};
    s.row = 0;
}

DenseTensor record_block(const Draft& model, State& s, DraftGraph& cell) {
    auto* branch = cell.branch.get();
    if (branch) {
        cell.window.prepare_branch_stream(branch->stream, branch->pool);
        // The parent joins every read of the shared model weights before retiring the request.
        branch->fork.record(cell.stream.stream());
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(branch->stream.stream(), branch->fork.get(), 0));
        cuda::StreamGuard selected(branch->stream);
        store_context(model, s, cell.features.dense(), cell.context_positions.dense(), branch->stream, branch);
        branch->joined.record(branch->stream.stream());
    } else if (cell.features) {
        store_context(model, s, cell.features.dense(), cell.context_positions.dense(), cell.stream);
    }
    auto hidden = block_hidden(model, s, cell.ids.dense(), cell.positions.dense(), cell.stream, branch);
    if (branch) TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(cell.stream.stream(), branch->joined.get(), 0));
    return hidden;
}
} // namespace

void Draft::append_target_context(MtpState& value, const Tensor& hidden, std::int64_t start) const {
    auto& s = state<State>(value);
    validate_context(config, s, hidden, start);
    const auto count = hidden.size(1), offset = count - std::min<std::int64_t>(count, config.window);
    auto positions = ops::arange(start + offset, start + count, device).dense();
    auto input = ops::contiguous(ops::cast(ops::slice(hidden, 1, offset, count), ScalarType::float32, device)).dense();
    store_context(*this, s, input, positions, cuda::current_stream(device.index));
    commit_context(s, start + count);
}

void Draft::seed_target_context(MtpState& value, const Tensor& hidden, std::int64_t count) const {
    auto& s = state<State>(value);
    if (s.pos || hidden.dim() != 3 || count < 1 || hidden.size(1) < std::min<std::int64_t>(count, config.window) || count < hidden.size(1))
        throw std::invalid_argument("DFlash2 prompt context is incomplete");
    append_target_context(value, hidden, count - hidden.size(1));
}

StepSequence<Tensor> Draft::draft_block(MtpState& value, const Tensor& anchor, int requested,
        bool graph, const Tensor& features) const {
    auto& s = state<State>(value);
    const auto& c = config;
    const auto start = s.pos;
    if (features) validate_context(c, s, features, start);
    const auto count = features ? features.size(1) : 0;
    const auto end = start + count;
    if (!end || requested < 1 || requested > maximum_draft_depth() || !anchor.defined() ||
        anchor.numel() != 1 || anchor.scalar_type() != ScalarType::int64 || anchor.device() != device || !anchor.is_contiguous() ||
        end > std::numeric_limits<std::int64_t>::max() - c.block)
        throw std::invalid_argument("DFlash2 draft depth or anchor differs");
    const auto stream = cuda::current_stream(device.index);
    const std::int64_t shape[] = {1, c.block};
    const auto context = cuda::default_context(device.index);
    const auto* setting = cuda::env_lookup("TYLOINFER_DFLASH2_GRAPH");
    const bool use_graph = graph && context->supports_async_allocations() && (!setting || setting[0] != '0');
    const bool capture_context = use_graph && count && count <= raw::kQ8TileRows;
    if (std::any_of(s.graphs.begin(), s.graphs.end(), [&](const auto& cell) {
            return cell && cell->stream.stream() != stream.stream();
        })) s.release_graphs();
    if (count && !capture_context) append_target_context(value, features, start);
    DenseTensor hidden;
    if (use_graph) {
        std::vector<Tensor> rings;
        rings.reserve(s.keys.size() + s.values.size());
        rings.insert(rings.end(), s.keys.begin(), s.keys.end());
        rings.insert(rings.end(), s.values.begin(), s.values.end());
        auto& slot = s.graphs[capture_context];
        if (slot && !slot->window.matches(rings)) slot.reset();
        const bool capture = !slot;
        if (capture) {
            // Warm passes may revisit the same slots or fail partway through a write.
            if (capture_context) append_target_context(value, features, start);
            co_yield GraphPhase::ready;
            auto bytes = 2 * std::size_t(c.block) * sizeof(std::int64_t);
            if (capture_context)
                bytes += raw::kQ8TileRows * (sizeof(std::int64_t) + std::size_t(features.size(2)) * sizeof(float));
            std::vector<std::pair<Device, std::size_t>> growth{{device, bytes}};
            co_yield GraphRequest{GraphPhase::admit, nullptr, std::move(growth)};
            auto cell = std::make_shared<DraftGraph>(stream, context);
            cell->ids = cell->window.buffer({1, c.block}, anchor.options());
            cell->positions = cell->window.buffer({c.block}, anchor.options());
            ops::fill(cell->ids, c.mask);
            if (capture_context) {
                cell->features = cell->window.buffer({1, raw::kQ8TileRows, features.size(2)}, anchor.options().dtype(kFloat32));
                cell->context_positions = cell->window.buffer({raw::kQ8TileRows}, anchor.options());
                const auto* overlap = cuda::env_lookup("TYLOINFER_DFLASH2_CONTEXT_OVERLAP");
                if (!overlap || overlap[0] != '0') cell->branch = std::make_unique<DraftGraph::Branch>(device.index, c.layers);
            }
            slot = std::move(cell);
        }
        auto& cell = *slot;
        if (capture_context) {
            const auto retained = std::min<std::int64_t>(count, c.window);
            if (retained < raw::kQ8TileRows) {
                ops::fill(cell.features, 0);
                ops::fill(cell.context_positions, -1);
            }
            auto rows = ops::cast(ops::slice(features, 1, count - retained, count), kFloat32, device);
            ops::copy_into(ops::slice(cell.features, 1, 0, retained), rows);
            ops::copy_into(ops::slice(cell.context_positions, 0, 0, retained), ops::arange(end - retained, end, device));
        }
        record(stream, {anchor.dense()});
        raw::prepare_draft_inputs(anchor.dense().data_ptr<std::int64_t>(), cell.ids.dense().data_ptr<std::int64_t>(),
            cell.positions.dense().data_ptr<std::int64_t>(), c.block, end, stream.stream());
        if (capture) {
            auto recording = cell.window.begin_capture();
            while (recording->next()) {
                std::exception_ptr failure;
                bool warm = false;
                try {
                    cell.hidden = record_block(*this, s, cell);
                    warm = recording->finish();
                } catch (...) { failure = std::current_exception(); }
                if (failure) {
                    if (!recording->retry(failure)) std::rethrow_exception(failure);
                    warm = true;
                }
                if (warm) {
                    cell.hidden = {};
                    recording->release_warm_outputs();
                    recording->pause();
                    co_yield GraphRequest{GraphPhase::admit, nullptr, recording->growth()};
                    recording->resume();
                }
            }
            recording->complete();
            cell.window.retain(rings);
        }
        if (!cell.window.enqueue_if_device_only()) cell.window.run();
        if (capture_context) commit_context(s, end);
        hidden = cell.hidden;
    } else {
        auto ids = ops::full(shape, c.mask, device).dense();
        auto positions = ops::arange(s.pos, s.pos + c.block, device).dense();
        ids.narrow(1, 0, 1).copy_(anchor.dense().reshape({1, 1}));
        hidden = block_hidden(*this, s, ids, positions, stream);
    }
    auto norm = normalize(hidden.narrow(1, 1, requested), final_norm, c.epsilon, stream);
    auto logits = ops::contiguous(ops::cast(target.logits(norm.values, true, LogitPrecision::input), ScalarType::float32)).dense();
    if (logits.numel() != std::int64_t(requested) * c.vocabulary || logits.device() != device)
        throw std::invalid_argument("DFlash2 target logits differ");
    logits = logits.reshape({1, requested, c.vocabulary});
    auto compact = sample_top_k_compact_cuda(logits.reshape({requested, c.vocabulary}), c.candidates);
    auto selector = project(selector_projection, norm.values, stream);
    DenseTensor successor_rows;
    if (successor.is_cpu()) {
        const auto candidates = compact.cpu();
        const auto rows = std::int64_t(requested) * c.candidates;
        auto host = empty({rows, c.rank}, successor.options());
        const auto* indices = candidates.data_ptr<std::int64_t>();
        const auto* table = successor.data_ptr<std::uint16_t>();
        auto* destination = host.data_ptr<std::uint16_t>();
        for (std::int64_t row = 0; row < rows; ++row) {
            const auto token = std::uint32_t(indices[row]);
            auto* output = destination + row * c.rank;
            if (token < std::uint32_t(c.vocabulary))
                std::memcpy(output, table + std::int64_t(token) * c.rank, c.rank * sizeof(std::uint16_t));
            else std::memset(output, 0, c.rank * sizeof(std::uint16_t));
        }
        successor_rows = host.to(device);
    }
    s.compact = std::move(compact);
    s.selector = std::move(selector);
    s.successor_rows = std::move(successor_rows);
    s.row = 0;
    co_yield Tensor{std::move(logits)};
}

Tensor Draft::draft_next(MtpState& value, const Tensor& logits, const Tensor& previous) const {
    auto& s = state<State>(value);
    const auto& c = config;
    if (!s.compact.defined() || s.row >= s.compact.size(0))
        throw std::logic_error("DFlash2 selector has no draft row");
    if (!logits.defined() || logits.numel() != c.vocabulary || logits.device() != device ||
        !previous.defined() || previous.numel() != 1 || previous.device() != device ||
        previous.scalar_type() != ScalarType::int64 || !previous.is_contiguous())
        throw std::invalid_argument("DFlash2 selector input differs");
    auto output = ops::empty(logits.dense().sizes(), logits.options().dtype(ScalarType::float32));
    ops::fill(output, -std::numeric_limits<float>::infinity());
    const auto stream = cuda::current_stream(device.index);
    const bool staged = successor.is_cpu();
    const auto& rows = staged ? s.successor_rows : successor;
    record(stream, {s.compact, s.selector, previous.dense(), predecessor, rows, output.dense()});
    raw::selector_logits(s.compact.data_ptr<std::int64_t>() + std::int64_t(s.row) * c.candidates, c.candidates,
        previous.dense().data_ptr<std::int64_t>(), predecessor.data_ptr<std::uint16_t>(),
        rows.data_ptr<std::uint16_t>() + (staged ? std::int64_t(s.row) * c.candidates * c.rank : 0),
        s.selector.data_ptr<float>() + std::int64_t(s.row) * c.rank, output.dense().data_ptr<float>(),
        c.rank, c.vocabulary, stream.stream(), staged);
    ++s.row;
    return output;
}

StepSequence<MtpDraft> Draft::greedy_draft(const CausalModel& model, MtpRow input,
        int depth, const ForwardOptions&) const {
    if (&model != &target || !input.state || !input.state->belongs_to(*this) ||
        input.ids.dim() != 2 || input.ids.size(0) != 1 || input.ids.size(1) != 1) co_return;
    auto& request = state<State>(*input.state);
    auto block = draft_block(request, input.ids, depth, true, input.context);
    while (auto step = block.next())
        if (step.graph) co_yield std::move(*step.graph);
    auto tokens = ops::empty({1, depth}, TensorOptions().device(device).dtype(kInt64));
    const auto stream = cuda::current_stream(device.index);
    const bool staged = successor.is_cpu();
    const auto& rows = staged ? request.successor_rows : successor;
    record(stream, {request.compact, request.selector, input.ids.dense(), predecessor, rows, tokens.dense()});
    raw::selector_greedy_walk(request.compact.data_ptr<std::int64_t>(), depth, config.candidates,
        input.ids.dense().data_ptr<std::int64_t>(), predecessor.data_ptr<std::uint16_t>(),
        rows.data_ptr<std::uint16_t>(), request.selector.data_ptr<float>(),
        tokens.dense().data_ptr<std::int64_t>(), config.rank, config.vocabulary, stream.stream(), staged);
    request.row = depth;
    const std::array<Tensor, 2> window{input.ids, tokens};
    co_yield MtpDraft{ops::cat(window, 1)};
}
} // namespace tyloi::models::dflash2

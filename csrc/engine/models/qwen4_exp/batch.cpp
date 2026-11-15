#include "engine/models/qwen4_exp/batch.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/common/check.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace tyloi::models::qwen4_exp {
namespace {

class Request final : public BatchState {
public:
    explicit Request(const Model& model) : state(model), model_(model) {}
    std::int64_t position() const override { return state.position(); }
    std::size_t resident_bytes() const override {
        for (const auto& [name, value] : state.metrics())
            if (name == "kv_cache_bytes") return static_cast<std::size_t>(value);
        return 0;
    }
    BatchGrowth initial_growth_bytes() const override {
        BatchGrowth growth;
        for (std::size_t i = 0; i < model_.layers.size(); ++i) {
            const auto& layer = *model_.layers[i];
            if (layer.gdn) add_session_growth(growth, layer.device,
                layer.gdn->request_cache_growth_bytes(state.layers[i]->gdn));
        }
        return growth;
    }
    void initialize() override {
        state.reset(1);
        for (std::size_t i = 0; i < model_.layers.size(); ++i)
            if (const auto& gdn = model_.layers[i]->gdn)
                gdn->initialize_request_cache(state.layers[i]->gdn, false);
    }
    BatchGrowth prefill_growth_bytes(std::int64_t tokens) const override {
        return state.batch_growth(tokens);
    }
    std::vector<Tensor> storage() const override { return state.storage(); }
    State state;
private:
    const Model& model_;
};

struct Rows {
    std::vector<Tensor> current, full;
    std::vector<std::int64_t> offsets{0};
};

Rows prepare(const Model& model, const std::vector<State*>& states, const std::vector<Tensor>& ids) {
    require_model(!states.empty() && states.size() == ids.size(), "invalid Qwen4 batch rows");
    Rows result;
    for (std::size_t row = 0; row < states.size(); ++row) {
        auto& state = *states[row];
        const auto& chunk = ids[row];
        require_model(state.belongs_to(model) && chunk.dim() == 2 && chunk.size(0) == 1 &&
            chunk.size(1) > 0 && state.cache_pos >= 0 &&
            chunk.size(1) <= model.max_position_embeddings() - state.cache_pos,
            "invalid Qwen4 batch input");
        state.begin_forward();
        auto current = ops::arange(state.cache_pos, state.cache_pos + chunk.size(1), model.device);
        if (state.decode_position_delta) current = ops::add(current, state.decode_position_delta);
        result.current.push_back(current);
        result.full.push_back(state.positions
            ? ops::cat(std::array<Tensor, 2>{state.positions, current}, -1) : current);
        result.offsets.push_back(result.offsets.back() + chunk.size(1));
    }
    return result;
}

// Process each request's mutable attention and PLE state independently while
// projecting and applying the FFN across the combined rows.
Tensor layer(const Layer& weights, const std::vector<State*>& requests, std::size_t index,
        const std::vector<Tensor>& ids, const std::vector<Tensor>& host, const Rows& rows,
        Tensor hidden, bool decode, bool batch_projection) {
    const auto axis = decode ? 0 : 1;
    if (weights.ple) {
        std::vector<Tensor> values;
        for (std::size_t row = 0; row < requests.size(); ++row) {
            const auto first = decode ? std::int64_t(row) : rows.offsets[row];
            const auto last = decode ? first + 1 : rows.offsets[row + 1];
            auto input = ops::slice(hidden, axis, first, last);
            auto ple = weights.ple->forward(requests[row]->layers[index]->ple, input,
                ids[row], true, 0, host[row]);
            values.push_back(ops::add(input, ple));
        }
        hidden = ops::cat(values, axis);
    }
    auto mix = weights.attention_gr.pre(hidden);
    const bool project = requests.size() > 1 && batch_projection;
    const auto projections = project ? (weights.gdn ? weights.gdn->project_inputs(mix.mixed)
        : weights.qsa->project_inputs(mix.mixed)) : std::vector<Tensor>{};
    std::vector<Tensor> attended;
    for (std::size_t row = 0; row < requests.size(); ++row) {
        const auto first = decode ? std::int64_t(row) : rows.offsets[row];
        const auto last = decode ? first + 1 : rows.offsets[row + 1];
        auto input = ops::slice(mix.mixed, axis, first, last);
        std::vector<Tensor> row_projections;
        for (const auto& value : projections) row_projections.push_back(ops::slice(value, axis, first, last));
        auto& state = *requests[row]->layers[index];
        attended.push_back(weights.gdn
            ? weights.gdn->forward_chunk(state.gdn, input, true, std::move(row_projections), !project)
            : weights.qsa->forward(*state.qsa, input, rows.current[row], rows.full[row], true,
                nullptr, std::move(row_projections), !project));
    }
    auto attention = ops::cat(attended, axis);
    if (project) attention = weights.gdn ? weights.gdn->project_output(attention)
                                        : weights.qsa->project_output(attention);
    auto ffn = weights.ffn_gr.pre_after(attention, mix.residual, mix.gate);
    auto output = decode ? ops::batch_ffn(weights.ffn, ffn.mixed) : ops::ffn(weights.ffn, ffn.mixed);
    return weights.ffn_gr.post(output, ffn.residual, ffn.gate);
}
} // namespace

Batch::Batch(Model& model, const BatchConfig& config, bool batch_attention_projections)
    : model_(model), batch_attention_projections_(batch_attention_projections) {
    require_model(config.slots > 0 && config.slots <= 32, "Flash-Next continuous batching supports 1..32 requests");
    for (const auto& layer : model.layers)
        require_model(layer->device == model.device, "Flash-Next continuous batching requires one device");
}

std::unique_ptr<BatchState> Batch::make_state(std::int64_t) { return std::make_unique<Request>(model_); }
std::size_t Batch::prefill_row_bytes() const {
    return std::size_t(model_.metadata.hidden_size) * model_.metadata.hc_mult * sizeof(float) * 8;
}

std::vector<State*> Batch::active(std::span<const BatchRow> rows) const {
    std::vector<State*> result;
    for (const auto& row : rows) if (row.eligible) {
        auto* request = dynamic_cast<Request*>(row.state);
        require_model(request && request->state.belongs_to(model_) && request->position() == row.position,
            "Flash-Next batch decode position mismatch");
        result.push_back(&request->state);
    }
    return result;
}

StepSequence<std::vector<Tensor>> Batch::prefill(std::vector<BatchState*> batch, std::vector<Tensor> ids) {
    std::vector<State*> requests;
    for (auto* value : batch) {
        auto* request = dynamic_cast<Request*>(value);
        require_model(request && request->state.belongs_to(model_), "Qwen4 batch state belongs to a different model");
        requests.push_back(&request->state);
    }
    auto rows = prepare(model_, requests, ids);
    auto all_ids = ops::cat(ids, 1);
    auto all_host = ops::copy(all_ids, Device(kCPU));
    std::vector<Tensor> host;
    for (std::size_t row = 0; row < requests.size(); ++row)
        host.push_back(ops::slice(all_host, 1, rows.offsets[row], rows.offsets[row + 1]));
    if (requests.size() == 1 && model_.host_tokens)
        for (std::size_t i = 0; i < model_.layers.size(); ++i)
            if (model_.layers[i]->ple) model_.layers[i]->ple->prefetch_tokens(requests[0]->layers[i]->ple, host[0]);
    auto hidden = model_.repeat_hidden(model_.embed_tokens(all_ids), model_.metadata.hc_mult);
    for (std::size_t i = 0; i < model_.layers.size(); ++i) {
        co_yield LayerEvent{LayerEvent::Phase::begin, requests.size() > 1 ? LayerEvent::Mode::packed : LayerEvent::Mode::single};
        hidden = layer(*model_.layers[i], requests, i, ids, host, rows, std::move(hidden), false, true);
        co_yield LayerEvent{LayerEvent::Phase::end};
    }
    hidden = model_.final_mixer->pre(hidden, ops::ResidualRole::final).mixed;
    std::vector<Tensor> result;
    for (std::size_t row = 0; row < requests.size(); ++row) {
        result.push_back(ops::slice(hidden, 1, rows.offsets[row + 1] - 1, rows.offsets[row + 1]));
        requests[row]->finish_forward(rows.full[row], 1, ids[row].size(1));
        requests[row]->cache_pos += ids[row].size(1);
    }
    co_yield std::move(result);
}

Tensor Batch::decode(std::span<const BatchRow> rows, Tensor ids, const Tensor& host_ids) {
    auto requests = active(rows);
    require_model(!requests.empty() && ids.dim() == 2 && ids.size(0) == std::int64_t(requests.size()) &&
        ids.size(1) == 1, "invalid Flash-Next batch input");
    auto host_all = host_ids ? host_ids : ops::copy(ids, Device(kCPU));
    std::vector<Tensor> chunks, host;
    for (std::size_t row = 0; row < requests.size(); ++row) {
        chunks.push_back(ops::slice(ids, 0, row, row + 1));
        host.push_back(ops::slice(host_all, 0, row, row + 1));
    }
    auto positions = prepare(model_, requests, chunks);
    auto hidden = model_.repeat_hidden(model_.embed_tokens(ids), model_.metadata.hc_mult);
    for (std::size_t i = 0; i < model_.layers.size(); ++i) {
        const auto& weights = *model_.layers[i];
        const bool project = batch_attention_projections_ && (weights.gdn || !model_.device.is_cuda());
        hidden = layer(weights, requests, i, chunks, host, positions, std::move(hidden), true, project);
    }
    auto final = model_.final_mixer->pre(hidden, ops::ResidualRole::final).mixed;
    auto logits = ops::reshape(ops::softcap(model_.logits(final, false), model_.metadata.final_logit_softcapping),
        {std::int64_t(requests.size()), -1});
    for (std::size_t row = 0; row < requests.size(); ++row) {
        requests[row]->finish_forward(positions.full[row], 1, 1);
        ++requests[row]->cache_pos;
    }
    return logits;
}
} // namespace tyloi::models::qwen4_exp

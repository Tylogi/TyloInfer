#include "engine/models/qwen35/batch.h"
#include "engine/models/qwen35/forward.h"
#include "engine/weights/moe/experts.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tyloi::models::qwen35 {
namespace {

void add(BatchGrowth& result, const BatchGrowth& values) {
    for (const auto& [device, bytes] : values) {
        auto found=std::find_if(result.begin(),result.end(),[&](const auto& value) { return value.first==device; });
        if (found==result.end()) result.emplace_back(device,bytes);
        else {
            if (bytes>std::numeric_limits<std::size_t>::max()-found->second)
                throw std::overflow_error("Qwen batch growth overflow");
            found->second+=bytes;
        }
    }
}

bool cached(const models::Ffn& ffn) {
    const auto* moe=std::get_if<models::Moe>(&ffn);
    return moe && (weights::cached(moe->down) ||
        std::any_of(moe->gate_up.begin(),moe->gate_up.end(),weights::cached));
}

std::vector<state::PagedKv::Layout> layouts(const Model& model) {
    std::vector<state::PagedKv::Layout> result;
    for (const auto& layer:model.layers) if (const auto* full=std::get_if<Full>(&layer.weights)) {
        const auto& attention=full->attention;
        result.push_back({layer.device,attention.kv_heads,attention.attention_head_dim,
            attention.official_bf16 ? kBFloat16 : kFloat16,attention.kv_quantization});
    }
    return result;
}

} // namespace

class Batch::Request final : public models::BatchState {
public:
    explicit Request(Batch& owner) : state(owner.model_,owner.model_.recurrent_transposed), owner_(owner) {
        if (owner_.pages_) state.bind_pages(*owner_.pages_);
    }
    models::State* target() noexcept override { return &state; }
    std::int64_t position() const override { return state.position(); }
    std::size_t resident_bytes() const override {
        std::vector<std::shared_ptr<TensorStorage>> backings;
        for (const auto& value:state.storage()) {
            auto storage=value.backings();
            backings.insert(backings.end(),storage.begin(),storage.end());
        }
        if (state.predictor) for (const auto& value:state.predictor->storage()) {
            auto storage=value.backings();
            backings.insert(backings.end(),storage.begin(),storage.end());
        }
        return backing_bytes(backings);
    }
    BatchGrowth initial_growth_bytes() const override {
        return owner_.config_.preallocate && !owner_.pages_ ? state.preallocate_growth() : BatchGrowth{};
    }
    void initialize() override {
        state.reset(1);
        if (owner_.config_.preallocate && !owner_.pages_) state.preallocate();
    }
    BatchGrowth prefill_growth_bytes(std::int64_t tokens) const override {
        return state.growth_for(1,tokens,owner_.config_.forward);
    }
    std::vector<Tensor> storage() const override { return state.storage(); }
    State state;
    bool admitted=false;
private:
    friend class Batch;
    Batch& owner_;
};

Batch::Batch(const Model& model, BatchConfig config) : model_(model),config_(std::move(config)) {
    if (!(config_.slots>0 && config_.slots<=32))
        throw std::runtime_error("Qwen continuous batching supports 1..32 requests");
    if (model_.layers.empty())
        throw std::runtime_error("continuous batching requires at least one model block");
    if (config_.cpu_offload)
        throw std::runtime_error("continuous batching requires GPU-resident model blocks");
    for (const auto& layer:model_.layers) {
        if (layer.device.is_cpu())
            throw std::runtime_error("continuous batching cannot use CPU-offloaded blocks");
        std::visit([&](const auto& value) {
            if constexpr (std::is_same_v<std::decay_t<decltype(value)>,Full>)
                if (value.attention.sliding)
                    throw std::runtime_error("continuous batching requires non-sliding Qwen attention");
            if (const auto* moe=std::get_if<models::Moe>(&value.ffn); moe && cached(value.ffn))
                if (!(moe->top_k<=config_.cached_slots))
                    throw std::runtime_error("continuous batching requires one cached slot per routed expert");
        },layer.weights);
    }
}

BatchGrowth Batch::initial_growth_bytes() const {
    return config_.paged ? state::PagedKv::construction_growth(layouts(model_),model_.metadata.max_position_embeddings,
        std::int32_t(config_.slots),config_.preallocate) : BatchGrowth{};
}
void Batch::initialize() {
    if (config_.paged) pages_=std::make_unique<state::PagedKv>(layouts(model_),model_.metadata.max_position_embeddings,
        std::int32_t(config_.slots),config_.preallocate);
}
std::unique_ptr<models::BatchState> Batch::make_state(std::int64_t prompt) {
    if (!(prompt>0))
        throw std::runtime_error("invalid Qwen batch prompt");
    return std::make_unique<Request>(*this);
}
Batch::Request& Batch::request(models::BatchState& value) const {
    auto* result=dynamic_cast<Request*>(&value);
    if (!(result && &result->owner_==this))
        throw std::runtime_error("Qwen batch state belongs to another batch");
    return *result;
}
void Batch::activate(models::BatchState& value) { request(value).admitted=true; }
void Batch::release(models::BatchState& value) {
    clear();
    auto& row=request(value);
    if (row.admitted) { row.admitted=false; ++releases_; }
}
std::vector<State*> Batch::active(std::span<const BatchRow> rows) const {
    std::vector<State*> states;
    for (const auto& row:rows) if (row.eligible) {
        auto& value=request(*row.state);
        if (!(value.state.position()==row.position))
            throw std::runtime_error("Qwen batch position mismatch");
        states.push_back(&value.state);
    }
    return states;
}
BatchGrowth Batch::decode_growth(std::span<const BatchRow> rows) const {
    BatchGrowth result;
    for (const auto* state:active(rows)) add(result,state->growth_for(1,1,config_.forward));
    return result;
}
void Batch::prepare_decode(std::span<const BatchRow> rows) {
    for (auto* state:active(rows)) state->prepare_pages(state->position()+1);
}
std::size_t Batch::prefill_row_bytes() const {
    return std::size_t(model_.metadata.hidden_size)*model_.metadata.hc_mult*sizeof(float)*8;
}
StepSequence<std::vector<Decoded>> Batch::prefill(std::vector<models::BatchState*> states,std::vector<Tensor> ids) {
    return forward(std::move(states),std::move(ids),false);
}
StepSequence<std::vector<Decoded>> Batch::verify(std::vector<models::BatchState*> states,std::vector<Tensor> ids) {
    return forward(std::move(states),std::move(ids),true);
}
StepSequence<std::vector<Decoded>> Batch::forward(std::vector<models::BatchState*> states,
        std::vector<Tensor> ids,bool verify) {
    if (!(!states.empty() && states.size()==ids.size()))
        throw std::runtime_error("invalid Qwen batch rows");
    std::vector<ForwardRow> rows;
    for (std::size_t i=0;i<states.size();++i) {
        models::Input input;
        input.ids=std::move(ids[i]);
        input.confirmed=verify && input.ids.size(1)>1 ? 1 : 0;
        rows.push_back({&request(*states[i]).state,std::move(input)});
    }
    auto options=config_.forward;
    options.serial_experts=options.serial_experts || has_cached_moe();
    auto job=forward_batch(model_,std::move(rows),options);
    while (auto step=job.next()) {
        if (step.value) {
            auto output=std::move(*step.value);
            if (verify) {
                std::vector<Tensor> hidden;
                for (const auto& row:output) hidden.push_back(row.hidden);
                auto logits=model_.logits(ops::cat(hidden,1));
                std::int64_t offset=0;
                for (auto& row:output) {
                    row.logits=ops::slice(logits,1,offset,offset+row.hidden.size(1));
                    offset+=row.hidden.size(1);
                }
            } else for (std::size_t i=0;i<output.size();++i) {
                auto& row=output[i];
                row.hidden=ops::slice(row.hidden,1,row.hidden.size(1)-1,row.hidden.size(1));
                if (!request(*states[i]).state.predictor) row.raw={};
            }
            co_yield std::move(output);
        } else if (step.layer) co_yield *step.layer;
        else if (step.workspace) co_yield *step.workspace;
        else if (step.graph) co_yield *step.graph;
        else if (step.state==StepState::waiting) co_yield StepState::waiting;
    }
}
Tensor Batch::decode(std::span<const BatchRow> rows,Tensor ids,const Tensor&) {
    const auto states=active(rows);
    if (!(!states.empty() && ids.dim()==2 && ids.size(0)==std::int64_t(states.size()) && ids.size(1)==1))
        throw std::runtime_error("invalid Qwen batch decode rows");
    std::vector<ForwardRow> inputs;
    for (std::size_t i=0;i<states.size();++i) {
        models::Input input;
        input.ids=ops::slice(ids,0,i,i+1);
        inputs.push_back({states[i],std::move(input)});
    }
    auto options=config_.forward;
    options.serial_experts=options.serial_experts || has_cached_moe();
    auto job=forward_batch(model_,std::move(inputs),options);
    std::vector<Tensor> hidden;
    while (auto step=job.next()) {
        if (step.value) for (const auto& row:*step.value) hidden.push_back(row.hidden);
        else if (!step.layer) throw std::logic_error("unexpected ordinary Qwen batch decode event");
    }
    if (!(hidden.size()==states.size()))
        throw std::runtime_error("Qwen batch decode returned no hidden rows");
    return ops::reshape(model_.logits(ops::cat(hidden,0)),{std::int64_t(states.size()),-1});
}
bool Batch::has_cached_moe() const {
    return std::any_of(model_.layers.begin(),model_.layers.end(),[](const auto& layer) {
        return std::visit([](const auto& value) { return cached(value.ffn); },layer.weights);
    });
}
engine::Metrics Batch::metrics() const {
    const auto stats=pages_ ? pages_->stats() : state::PagedKv::Stats{};
    const bool moe=std::any_of(model_.layers.begin(),model_.layers.end(),[](const auto& layer) {
        return std::visit([](const auto& value) { return std::holds_alternative<models::Moe>(value.ffn); },layer.weights);
    });
    return {{"continuous_batching_request_releases",double(releases_)},
        {"continuous_batching_moe",double(moe)}, {"continuous_batching_moe_cached_row_serial",double(has_cached_moe())},
        {"continuous_batching_paged_kv",double(bool(pages_))},
        {"paged_kv_page_size",double(stats.page_size)}, {"paged_kv_live_pages",double(stats.live_pages)},
        {"paged_kv_peak_live_pages",double(stats.peak_live_pages)}, {"paged_kv_capacity_pages",double(stats.capacity_pages)},
        {"paged_kv_reserved_bytes",double(stats.reserved_bytes)}, {"paged_kv_page_allocations",double(stats.allocations)},
        {"paged_kv_page_reuses",double(stats.reuses)}, {"paged_kv_page_releases",double(stats.releases)},
        {"paged_kv_table_updates",double(stats.table_updates)}};
}

} // namespace tyloi::models::qwen35

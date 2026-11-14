#include "engine/models/qwen4_exp/batch.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/backends/cuda/executor/executor.h"
#include "components/error.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::qwen4_exp {
namespace {

class Request final : public models::BatchState {
public:
    explicit Request(Model& model) : state(model), device(kCUDA, model.primary_device) {}
    std::int64_t position() const override { return state.position(); }
    std::size_t resident_bytes() const override { return state.resident_bytes(); }
    models::BatchGrowth initial_growth_bytes() const override { return {{device, state.initial_growth_bytes()}}; }
    void initialize() override { state.initialize(); }
    models::BatchGrowth prefill_growth_bytes(std::int64_t tokens) const override {
        return {{device, state.prefill_growth_bytes(tokens)}};
    }
    Qwen4BatchRequestState state;
    Device device;
};

std::vector<Qwen4BatchRequestState*> active(std::span<const models::BatchRow> rows) {
    std::vector<Qwen4BatchRequestState*> result;
    for (const auto& row : rows) if (row.eligible) {
        auto& request = static_cast<Request&>(*row.state);
        TYLOINFER_RUNTIME_CHECK(request.position() == row.position, "Flash-Next batch decode position mismatch");
        result.push_back(&request.state);
    }
    return result;
}

class Batch final : public models::Batch {
    struct GraphPlan final : Plan { Qwen4BatchDecodeGraph::Plan layout; };
public:
    Batch(Model& model, const models::BatchConfig& config)
        : model_(model), device_(kCUDA, model.primary_device) {
        TYLOINFER_RUNTIME_CHECK(config.slots > 0 && config.slots <= 32,
            "Flash-Next continuous batching supports 1..32 requests");
        if (config.graph) graph_ = std::make_unique<Qwen4BatchDecodeGraph>(model_);
    }
    std::unique_ptr<models::BatchState> make_state(std::int64_t) override {
        return std::make_unique<Request>(model_);
    }
    bool supports_packed() const override { return true; }
    void release(models::BatchState&) override { clear(); }
    std::size_t prefill_row_bytes() const override {
        return std::size_t(model_.metadata.hidden_size) * model_.metadata.hc_mult * sizeof(float) * 8;
    }
    StepSequence<std::vector<Tensor>> prefill(std::vector<models::BatchState*> states,
            std::vector<Tensor> ids) override {
        std::vector<Qwen4BatchRequestState*> requests;
        for (auto* state : states) requests.push_back(&static_cast<Request&>(*state).state);
        auto job = qwen4_batch_prefill(model_, requests, ids, model_.host_tokens);
        while (auto step = job.next()) {
            if (step.layer) co_yield *step.layer;
            else if (step.workspace) co_yield *step.workspace;
            else if (step.graph) co_yield *step.graph;
            else if (step.value) co_yield std::move(*step.value);
            else co_yield step.state;
        }
    }
    Tensor decode(std::span<const models::BatchRow> rows, Tensor ids, const Tensor& host_ids) override {
        auto requests = active(rows);
#ifdef TYLOINFER_NATIVE_CUDA
        auto policy = Executor::current_ffn();
        policy.batch_serial = requests.size() > 8;
        auto scope = Executor::current().enter(policy);
#endif
        return qwen4_batch_decode(model_, requests, std::move(ids), host_ids);
    }
    StepSequence<std::unique_ptr<Plan>> plan(std::span<const models::BatchRow> rows, bool greedy) override {
        if (!graph_) co_return;
        co_yield GraphPhase::ready;
        auto layout = graph_->plan(active(rows), greedy);
        if (!layout) co_return;
        auto plan = std::make_unique<GraphPlan>();
        plan->growth = {{device_, layout->growth}};
        plan->rebuild = layout->rebuild;
        plan->layout = std::move(*layout);
        co_yield std::unique_ptr<Plan>(std::move(plan));
    }
    StepSequence<std::optional<Tensor>> capture(std::span<const models::BatchRow> rows,
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
#ifdef TYLOINFER_NATIVE_CUDA
                auto policy = Executor::current_ffn();
                policy.batch_serial = requests.size() > 8;
                auto scope = Executor::current().enter(policy);
#endif
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
    Model& model_;
    Device device_;
    std::unique_ptr<Qwen4BatchDecodeGraph> graph_;
};
}

std::unique_ptr<models::Batch> make_batch(Model& model, const models::BatchConfig& config) {
    return std::make_unique<Batch>(model, config);
}
}

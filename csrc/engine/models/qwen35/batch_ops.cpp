#include "engine/models/qwen35/batch.h"
#include "engine/models/qwen35/forward.h"
#include "engine/models/common/check.h"
#include "engine/backends/cuda/executor/graph/window.h"
#include "engine/backends/cuda/executor/graph/decode.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/attention/scratch.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <utility>

namespace tyloi::models::qwen35 {
namespace {

void add(BatchGrowth& result, Device device, std::size_t bytes) {
    auto found=std::find_if(result.begin(),result.end(),[&](const auto& item) { return item.first==device; });
    if (found==result.end()) result.emplace_back(device,bytes);
    else {
        if (bytes>std::numeric_limits<std::size_t>::max()-found->second)
            throw std::overflow_error("Qwen batch graph growth overflow");
        found->second+=bytes;
    }
}

std::vector<Tensor> storage(const std::vector<State*>& states) {
    std::vector<Tensor> result;
    for (const auto* state:states) {
        auto values=state->storage();
        result.insert(result.end(),values.begin(),values.end());
    }
    return result;
}

class CudaBatch final : public Batch {
    struct GraphPlan final : Plan { std::int64_t length=0; };
public:
    using Batch::Batch;
    StepSequence<std::unique_ptr<Plan>> plan(std::span<const BatchRow>,bool) override;
    StepSequence<std::optional<Tensor>> capture(std::span<const BatchRow>,const Tensor&,
        const Tensor&,bool,const Plan&) override;
    void clear() override {
        window_.reset(); metadata_={}; output_={}; states_.clear(); length_=0;
    }
    engine::Metrics metrics() const override {
        auto result=Batch::metrics();
        result.emplace_back("continuous_batching_cuda_graph_captures",captures_);
        result.emplace_back("continuous_batching_cuda_graph_replays",replays_);
        return result;
    }
private:
    std::unique_ptr<cuda::DecodeStream> stream_;
    std::unique_ptr<cuda::DecodeWindow> window_;
    Tensor metadata_,output_;
    std::vector<State*> states_;
    std::int64_t length_=0,captures_=0,replays_=0;
};

StepSequence<std::unique_ptr<models::Batch::Plan>> CudaBatch::plan(std::span<const BatchRow> rows,bool greedy) {
    auto states=active(rows);
    if (!config_.graph || !greedy || config_.slots<2 || has_cached_moe() || states.empty() ||
        !mfq_cuda_graph_capture_supported()) co_return;
    std::int64_t requested=0,remaining=std::numeric_limits<std::int64_t>::max();
    for (const auto& row:rows) if (row.eligible) {
        requested=std::max(requested,row.position+row.remaining);
        remaining=std::min(remaining,row.remaining);
    }
    const auto length=::decode_graph_bucket(requested,model_.metadata.max_position_embeddings);
    const bool reuse=window_ && length_==length && states_==states && window_->matches(storage(states));
    if (!reuse && remaining<config_.minimum) co_return;
    co_yield GraphPhase::ready;
    auto result=std::make_unique<GraphPlan>();
    result->length=length; result->rebuild=!reuse;
    if (!reuse) {
        result->growth=cuda::DecodeWindow::fixed_growth(model_.device,false,false,0);
        const auto bytes=3*states.size()*sizeof(std::int64_t);
        add(result->growth,model_.device,bytes);
        add(result->growth,Device(kCPU),bytes);
        for (auto* state:states) for (const auto* tensor:state->graph_state())
            add(result->growth,tensor->device(),tensor->nbytes());
    }
    co_yield std::unique_ptr<models::Batch::Plan>(std::move(result));
}

StepSequence<std::optional<Tensor>> CudaBatch::capture(std::span<const BatchRow> rows,
        const Tensor&,const Tensor&,bool,const models::Batch::Plan& value) {
    const auto& plan=static_cast<const GraphPlan&>(value);
    auto states=active(rows);
    bool prepared=false;
    if (plan.rebuild) co_yield GraphRequest{GraphPhase::prepare,&prepared};
    if (!stream_) stream_=std::make_unique<cuda::DecodeStream>();
    auto stream_scope=stream_->enter();
    const auto count=std::int64_t(states.size());
    if (plan.rebuild) {
        clear();
        window_=std::make_unique<cuda::DecodeWindow>(false,true);
        if (!prepared) window_->prepare_pool();
        metadata_=window_->buffer({3,count},TensorOptions().device(model_.device).dtype(kInt64));
        length_=plan.length; states_=states;
    }
    require_model(window_ && states_==states,"Qwen batch graph membership changed");
    std::vector<std::int64_t> values(std::size_t(3*count));
    std::size_t index=0;
    for (const auto& row:rows) if (row.eligible) {
        values[index]=row.token; values[count+index]=row.position+states[index]->decode_position_delta;
        values[2*count+index]=row.position+1; ++index;
    }
    auto host=ops::reshape(ops::tensor(values,Device(kCPU)),{3,count});
    ops::copy_into(metadata_,host);
    if (!window_->valid()) {
        struct Saved { Tensor* target; Tensor copy; };
        std::vector<Saved> saved;
        for (auto* state:states) for (auto* tensor:state->graph_state()) saved.push_back({tensor,ops::clone(*tensor)});
        try {
            auto recording=window_->begin_capture();
            while (recording->next()) {
                std::exception_ptr failure;
                bool warm=false;
                try {
                    auto policy=cuda::Executor::current_ffn();
                    policy.serial=true; policy.projection_major=true;
                    auto execution=cuda::Executor::current().enter(policy);
                    std::vector<ForwardRow> inputs;
                    for (std::int64_t i=0;i<count;++i) {
                        models::Input input;
                        input.ids=ops::slice(ops::slice(metadata_,0,0,1),1,i,i+1);
                        input.positions=ops::slice(ops::slice(metadata_,0,1,2),1,i,i+1);
                        input.cache_positions=states[i]->decode_position_delta
                            ? ops::add(input.positions,-states[i]->decode_position_delta) : input.positions;
                        input.lengths=ops::reshape(ops::slice(ops::slice(metadata_,0,2,3),1,i,i+1),{1});
                        input.kv_length=plan.length;
                        input.attention_parts=::decode_graph_attention_parts(plan.length,cuda::AttentionScratch::kMaxParts);
                        inputs.push_back({states[i],std::move(input)});
                    }
                    auto job=forward_batch(model_,std::move(inputs),config_.forward,false);
                    std::vector<Tensor> hidden;
                    while (auto step=job.next()) {
                        if (step.value) for (const auto& row:*step.value) hidden.push_back(row.hidden);
                        else if (!step.layer) throw std::logic_error("unexpected Qwen batch graph event");
                    }
                    require_model(hidden.size()==states.size(),"Qwen batch graph returned no hidden rows");
                    auto logits=model_.logits(ops::cat(hidden,0));
                    output_=ops::reshape(ops::sample_greedy(ops::contiguous(logits)),{count,1});
                    warm=recording->finish();
                } catch (...) { failure=std::current_exception(); }
                if (failure) {
                    if (!recording->retry(failure)) std::rethrow_exception(failure);
                    warm=true;
                }
                for (const auto& item:saved) ops::copy_into(*item.target,item.copy);
                if (warm) {
                    output_={}; recording->release_warm_outputs(); recording->pause();
                    stream_scope.reset();
                    co_yield GraphRequest{GraphPhase::admit,nullptr,recording->growth()};
                    stream_scope=stream_->enter(); recording->resume();
                }
            }
            recording->complete();
            window_->retain_result(output_); window_->retain(storage(states)); ++captures_;
        } catch (...) {
            for (const auto& item:saved) ops::copy_into(*item.target,item.copy);
            clear(); throw;
        }
    }
    window_->run(); ++replays_;
    for (auto* state:states) state->advance(1);
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_get_current_cuda_stream()));
    stream_scope.reset();
    co_yield std::optional<Tensor>(output_);
}

} // namespace

std::unique_ptr<models::Batch> Model::make_batch(const BatchConfig& config) const {
    return std::make_unique<CudaBatch>(*this,config);
}
} // namespace tyloi::models::qwen35

#include "engine/generation/batch/ops.h"
#include "engine/engine.h"

#include "components/error.h"
#include "engine/memory/budget.h"
#include "engine/config.h"
#include "engine/executor.h"
#include "engine/generation/batch/batch.h"
#include "engine/generation/batch/driver.h"
#include "engine/generation/decode.h"
#include "engine/generation/mtp/sampling.h"
#include "engine/generation/prefill/budget.h"
#include "engine/generation/sampling.h"
#include "engine/generation/session.h"
#include "engine/models/batch.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/mtp.h"
#include "engine/ops/ops.h"
#include "engine/state/session/sessions.h"
#include "engine/weights/cache/cache.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::engine {

bool batch_compatible(const EngineRequest& request, bool persistent_prefix, bool has_mtp, bool sessions, bool mtp) {
    const auto& input = request.input;
    return !input.media && (sessions || (input.cache_plan.session_id.empty() &&
           !(input.prefix_cache_enabled &&
               (request.token_ids.empty() || input.cache_plan.stable_prefix_tokens) &&
               persistent_prefix))) &&
           (!(input.sampling.enable_mtp && has_mtp) || mtp);
}

namespace {

struct BatchRow : BatchRequest {
    using BatchRequest::BatchRequest;
    std::optional<Sampler> sampler;
    Tensor counts, ids;
    std::vector<std::unique_ptr<Executor::Timer>> timers;
    std::unique_ptr<models::BatchState> state;
    std::int64_t row = -1;
    double packed_ms = 0;
    bool caching = false;
    bool speculative = false;
    Tensor raw;
    models::Decoded verified;
    std::vector<std::int32_t> drafts;
    std::vector<mtp::CompactDistribution> compact_proposals;
    std::vector<std::vector<float>> proposals;
    std::vector<std::int64_t> candidates;
    mtp::PolicyState policy;
    std::unique_ptr<mtp::DepthPolicy> depth;
    mtp::GenerationStats mtp_stats;
    mtp::DepthPolicy::Clock::time_point started;
    std::int64_t predictor_base = 0;
    std::size_t accepted = 0, route_offset = 0;
    bool unresolved = false;
};

std::exception_ptr prefill_failure(std::exception_ptr failure) {
    try { std::rethrow_exception(failure); }
    catch (const std::exception& error) {
        const std::string message = error.what();
        if (dynamic_cast<const MfqResourceExhausted*>(&error) ||
            message.starts_with("CUDA process VRAM limit exceeded:") ||
            message.starts_with("CUDA VRAM limit exceeded:") ||
            message.starts_with("CUDA pinned allocation failed:"))
            return std::make_exception_ptr(InferenceInputError(InferenceInputErrorCode::ContextLength,
                "insufficient GPU or pinned RAM for prefill workspace; reduce concurrent requests or the packed prefill token budget"));
    } catch (...) {}
    return failure;
}

void merge_growth(models::BatchGrowth& target, const models::BatchGrowth& source) {
    for (const auto& [device, bytes] : source) {
        auto it=std::find_if(target.begin(),target.end(),[&](const auto& item) { return item.first==device; });
        if (it==target.end()) target.emplace_back(device,bytes);
        else it->second=prefill_budget_add(it->second,bytes);
    }
}

class Ops {
public:
    using Request = BatchRow;
    using State = BatchState<Request>;
    using Sample = models::BatchSample;
    struct Decoded { Tensor logits; std::vector<std::int64_t> greedy; bool speculative = false; };
    static constexpr bool isolate_prefill_failures = true;
    static constexpr bool isolate_context_growth_failures = true;

    Ops(models::CausalModel& model, std::unique_ptr<models::Batch> batch, Executor& executor,
        weights::Cache& weights, Budget& budget, state::Sessions& sessions, ContinuousBatchConfig config,
        Device device, std::size_t workspace, bool packed, std::int64_t packed_tokens,
        models::ForwardOptions forward, Executor* predictor_execution)
        : model_(model), batch_(std::move(batch)), executor_(executor), weights_(weights), budget_(&budget), sessions_(sessions),
          config_(std::move(config)), device_(device), workspace_(workspace), packed_(packed), packed_tokens_(packed_tokens),
          forward_(std::move(forward)), mtp_(model.predictor()), predictor_execution_(predictor_execution) {
        if (!batch_ || config_.max_sequences == 0 || config_.max_sequences > 32)
            throw std::invalid_argument("continuous batching supports 1..32 requests");
    }

    void bind(Budget& budget) noexcept { budget_ = &budget; }
    bool supports_sessions() const noexcept { return batch_->supports_sessions(); }
    bool supports_mtp() const noexcept {
        return batch_->supports_mtp() && mtp_ && !mtp_->blockwise_drafting() && predictor_execution_;
    }
    std::int64_t cache_length(const std::shared_ptr<Request>& request) const { return request->state->position(); }
    std::int64_t vocab_size() const { return model_.info().vocab_size; }
    std::int64_t max_context() const { return model_.info().max_context; }
    Metrics metrics() const {
        std::size_t bytes = 0, contexts = 0;
        for (const auto& weak : requests_)
            if (const auto request = weak.lock(); request && request->state) {
                bytes += request->state->resident_bytes();
                contexts += request->state->position() > 0;
            }
        auto result = batch_->metrics();
        result.emplace_back("continuous_batching_decode_batches", decode_batches_);
        result.emplace_back("continuous_batching_decode_tokens", decode_tokens_);
        result.emplace_back("continuous_batching_batched_greedy_batches", greedy_batches_);
        result.emplace_back("continuous_batching_packed_prefill_batches", packed_batches_);
        result.emplace_back("continuous_batching_packed_prefill_requests", packed_requests_);
        result.emplace_back("continuous_batching_packed_prefill_tokens", packed_rows_);
        if (std::none_of(result.begin(),result.end(),[](const auto& metric) {
            return metric.first=="continuous_batching_moe_cached_row_serial";
        })) result.emplace_back("continuous_batching_moe_cached_row_serial", 0.0);
        result.emplace_back("continuous_batching_request_kv_bytes", bytes);
        result.emplace_back("continuous_batching_request_contexts", contexts);
        result.emplace_back("continuous_batching_mtp_batches",mtp_batches_);
        result.emplace_back("continuous_batching_mtp_draft_batches",draft_batches_);
        result.emplace_back("continuous_batching_mtp_draft_rows",draft_rows_);
        return result;
    }
    void suspend_decode() {
        auto scope=executor_.enter();
        std::exception_ptr failure;
        try { weights_.finish_exchange(); } catch (...) { failure=std::current_exception(); }
        try { batch_->clear(); } catch (...) { if (!failure) failure=std::current_exception(); }
        if (failure) std::rethrow_exception(failure);
    }
    void resume_decode(std::int64_t position) {
        auto scope=executor_.enter();
        batch_->resume(position);
    }
    std::int64_t prepare(const std::shared_ptr<Request>& request) {
        auto scope = executor_.enter();
        suspend_decode();
        initialize(request, {});
        auto* target = request->state->target();
        if (supports_sessions() && !target)
            throw std::logic_error("batch session state is unavailable");
        request->caching = supports_sessions() && target && target->supports_text_session_state() &&
            request->cache_plan.stable_prefix_tokens > 0 &&
            (!request->cache_plan.session_id.empty() || sessions_.persistent_prefix_enabled());
        if (!request->caching) return 0;
        request->prefix = std::min(request->cache_plan.stable_prefix_tokens, request->prompt.size() - 1);
        const auto restored = sessions_.restore_best(*target, executor_, weights_, *budget_, request->speculative ? mtp_ : nullptr,
            request->cache_plan.session_id, request->prompt, request->cache_plan.stable_prefix_tokens);
        request->raw = restored.mtp_last_target_hidden;
        request->checkpoint = restored.tokens;
        return static_cast<std::int64_t>(restored.tokens);
    }
    std::int64_t prefix_tokens(const std::shared_ptr<Request>& request) const {
        return static_cast<std::int64_t>(request->prefix);
    }
    PrefillChunk prefill_chunk(const std::shared_ptr<Request>& request, PrefillChunk chunk) {
        auto scope = executor_.enter();
        if (!request->state) weights_.finish_exchange();
        initialize(request, chunk);
        std::size_t primary=0;
        for (const auto& [device, bytes] : request->state->prefill_growth_bytes(chunk.count)) {
            if (device==device_) primary=prefill_budget_add(primary,bytes);
            else if (bytes && device.is_cpu()) budget_->reserve_ram(bytes);
            else if (bytes) budget_->reserve_kv(device,prefill_budget_add(bytes,std::size_t(64)<<20));
        }
        return budget_->batch(device_, batch_->prefill_row_bytes(), chunk,
            primary);
    }
    std::size_t prefill_batch_limit() const { return packed_ && batch_->supports_packed() ? config_.max_sequences : 1; }
    std::int64_t prefill_batch_token_budget() const { return packed_tokens_; }

    Sample prefill(const std::shared_ptr<Request>& request, PrefillChunk chunk) {
        try {
            auto scope = executor_.enter();
            initialize(request, chunk);
            auto timer = executor_.timer();
            if (chunk.count > 1024) weights_.workspace(workspace_);
            reserve(request->state->prefill_growth_bytes(chunk.count),std::size_t(256)<<20,true);
            std::vector<models::BatchState*> states{request->state.get()};
            std::vector<Tensor> ids{ops::contiguous(ops::slice(request->ids,1,chunk.offset,chunk.offset+chunk.count))};
            auto result = run_prefill(std::move(states), std::move(ids)).at(0);
            prime_prompt(request, chunk, result.raw);
            auto hidden = std::move(result.hidden);
            Tensor next;
            if (chunk.offset + chunk.count == std::int64_t(request->prompt.size()))
                next = sample_logits(*request->sampler,
                    ops::reshape(model_.logits(ops::select(hidden,1,-1)), {1,-1}),
                    request->counts, cursor(request));
            timer->stop({hidden, next});
            request->timers.push_back(std::move(timer));
            if (request->state->position() == static_cast<std::int64_t>(request->prefix)) checkpoint(request);
            return finish(request, std::move(next));
        } catch (...) { std::rethrow_exception(prefill_failure(std::current_exception())); }
    }

    std::vector<PrefillBatchResult<Sample>> prefill_batch(
        const std::vector<std::pair<std::shared_ptr<Request>, PrefillChunk>>& items) {
        auto scope = executor_.enter();
        weights_.retire_decode();
        std::vector<PrefillBatchResult<Sample>> result(items.size());
        std::vector<std::size_t> ready;
        for (std::size_t i=0;i<items.size();++i) {
            try { initialize(items[i].first, items[i].second); ready.push_back(i); }
            catch (...) { result[i].failure = prefill_failure(std::current_exception()); }
        }
        const auto execute = [&](auto&& self, const std::vector<std::size_t>& rows) -> void {
            if (rows.empty()) return;
            models::BatchGrowth growth;
            for (auto row : rows) merge_growth(growth,items[row].first->state->prefill_growth_bytes(items[row].second.count));
            try { reserve(growth,std::size_t(256)<<20,true); }
            catch (const InferenceInputError& error) {
                if (error.code != InferenceInputErrorCode::ContextLength) throw;
                if (rows.size()==1) { result[rows.front()].failure=std::current_exception(); return; }
                const auto middle=rows.begin()+rows.size()/2;
                self(self, std::vector<std::size_t>(rows.begin(),middle));
                self(self, std::vector<std::size_t>(middle,rows.end()));
                return;
            }
            try {
                if (rows.size()==1) {
                    const auto row=rows.front();
                    result[row].sample=prefill(items[row].first,items[row].second);
                    return;
                }
                std::vector<models::BatchState*> states;
                std::vector<Tensor> ids;
                std::int64_t tokens=0;
                for (auto row : rows) {
                    const auto& [request,chunk]=items[row];
                    states.push_back(request->state.get());
                    ids.push_back(ops::contiguous(ops::slice(request->ids,1,chunk.offset,chunk.offset+chunk.count)));
                    tokens+=chunk.count;
                }
                auto timer=executor_.timer();
                if (tokens>1024) weights_.workspace(workspace_);
                auto outputs=run_prefill(std::move(states),std::move(ids));
                if (outputs.size()!=rows.size()) throw std::runtime_error("invalid packed prefill output count");
                std::vector<Tensor> hidden;
                for (std::size_t i=0;i<rows.size();++i) {
                    prime_prompt(items[rows[i]].first, items[rows[i]].second, outputs[i].raw);
                    hidden.push_back(outputs[i].hidden);
                }
                std::vector<Tensor> last, next(items.size());
                std::vector<std::size_t> finished;
                for (std::size_t i=0;i<rows.size();++i) {
                    const auto& [request,chunk]=items[rows[i]];
                    if (chunk.offset+chunk.count==std::int64_t(request->prompt.size())) {
                        last.push_back(hidden[i]); finished.push_back(rows[i]);
                    }
                }
                if (!last.empty()) {
                    auto logits=ops::reshape(model_.logits(ops::cat(last,0)), {std::int64_t(last.size()),-1});
                    for (std::size_t i=0;i<finished.size();++i) {
                        const auto row=finished[i];
                        const auto& request=items[row].first;
                        next[row]=sample_logits(*request->sampler,ops::slice(logits,0,i,i+1),
                            request->counts,cursor(request));
                    }
                }
                executor_.submit(hidden);
                executor_.submit(next);
                timer->stop();
                const auto milliseconds=timer->elapsed_ms();
                for (auto row : rows) {
                    items[row].first->packed_ms += milliseconds*items[row].second.count/tokens;
                    const auto& request = items[row].first;
                    if (request->state->position() == static_cast<std::int64_t>(request->prefix)) checkpoint(request);
                    result[row].sample=finish(items[row].first,std::move(next[row]));
                }
                ++packed_batches_; packed_requests_+=rows.size(); packed_rows_+=tokens;
            } catch (...) {
                const auto error=prefill_failure(std::current_exception());
                for (auto row : rows) result[row].failure=error;
            }
        };
        execute(execute,ready);
        return result;
    }

    void activate(const std::shared_ptr<Request>& request, const Sample&) {
        auto scope=executor_.enter();
        reserve(batch_->activate_growth(*request->state),std::size_t(64)<<20);
        batch_->activate(*request->state);
    }
    void discard_prefill(const std::shared_ptr<Request>& request) {
        auto scope=executor_.enter();
        suspend_decode();
        rollback(request);
        drain_predictor();
        executor_.drain();
        if (request->state) batch_->release(*request->state);
        request->state.reset(); request->ids={}; request->timers.clear(); request->packed_ms=0;
        request->raw={}; request->verified={}; request->depth.reset();
    }
    void retire(const std::vector<std::shared_ptr<Request>>& retired, std::int64_t position) {
        auto scope=executor_.enter();
        weights_.finish_exchange();
        for (const auto& request : retired) rollback(request);
        drain_predictor();
        executor_.drain();
        for (const auto& request : retired) {
            if (request->completed) checkpoint(request);
            if (request->state) batch_->release(*request->state);
            request->state.reset(); request->ids={}; request->timers.clear(); request->packed_ms=0;
            request->raw={}; request->verified={}; request->depth.reset();
        }
        batch_->retire(position);
        const bool active=std::any_of(requests_.begin(),requests_.end(),[](const auto& weak) {
            const auto request=weak.lock(); return request && request->state;
        });
        if (!active) { weights_.retire_decode(); executor_.reclaim(device_); }
    }
    Decoded decode(State& state) {
        auto scope=executor_.enter();
        if (std::any_of(state.active.begin(),state.active.end(),[](const auto& request) {
            return request->eligible && request->speculative;
        })) return decode_mtp(state);
        std::vector<models::BatchRow> rows;
        std::vector<std::int64_t> tokens;
        for (const auto& request : state.active) {
            request->row=-1;
            if (!request->state || request->state->position()!=request->cache_length)
                throw std::runtime_error("batch decode position mismatch");
            const bool row_greedy=request->sampler->greedy() && !request->sampler->has_penalties() && !request->token_constraint;
            rows.push_back({request->state.get(),request->pending_token,request->cache_length,
                request->generation_limit-request->produced,request->eligible,row_greedy});
            if (!request->eligible) continue;
            request->row=std::int64_t(tokens.size());
            tokens.push_back(request->pending_token);
        }
        auto host=ops::reshape(ops::tensor(tokens,Device(kCPU)), {-1,1});
        auto ids=ops::copy(host,device_);
        const bool greedy=config_.greedy && std::all_of(state.active.begin(),state.active.end(),[](const auto& request) {
            return !request->eligible || (request->sampler->greedy() && !request->sampler->has_penalties() && !request->token_constraint);
        });
        std::optional<Tensor> captured;
        try {
            reserve(batch_->decode_growth(rows),std::size_t(64)<<20);
            batch_->prepare_decode(rows);
            if (config_.cuda_graph) {
                auto plan=plan_graph(rows,greedy);
                if (plan) {
                    if (plan->rebuild) {
                        executor_.reclaim(device_);
                        reserve(plan->growth,std::size_t(64)<<20);
                    }
                    captured=capture_batch(*batch_,executor_,weights_,*budget_,device_,rows,ids,host,greedy,*plan);
                }
            }
        } catch (const std::exception& error) {
            const std::string message=error.what();
            if (!dynamic_cast<const MfqResourceExhausted*>(&error) &&
                !message.starts_with("CUDA process VRAM limit exceeded:") &&
                !message.starts_with("CUDA VRAM limit exceeded:") &&
                !message.starts_with("CUDA pinned allocation failed:")) throw;
            throw InferenceInputError(InferenceInputErrorCode::ContextLength,
                "insufficient GPU or pinned RAM for the batch decode workspace; reduce concurrent requests");
        }
        if (!captured) {
            models::BatchGrowth growth;
            for (const auto& row:rows) if (row.eligible) merge_growth(growth,row.state->prefill_growth_bytes(1));
            reserve(growth,std::size_t(64)<<20);
        }
        auto output=captured ? std::move(*captured) : batch_->decode(rows,std::move(ids),host);
        if (!captured) {
            std::vector<models::BatchState*> states;
            for (const auto& row:rows) if (row.eligible) states.push_back(row.state);
            submit_batch(executor_,std::span<const Tensor>{&output,1},states);
        }
        std::vector<std::int64_t> selected;
        if (greedy) {
            selected=ops::read_integers(captured ? output : ops::sample_greedy(ops::contiguous(output)));
            ++greedy_batches_;
        }
        ++decode_batches_; decode_tokens_+=tokens.size();
        return {std::move(output),std::move(selected)};
    }
    Sample sample(const std::shared_ptr<Request>& request, const Decoded& decoded) {
        if (request->row<0) throw std::runtime_error("batch sampled an inactive row");
        if (decoded.speculative) {
            Sample result;
            result.tokens = request->candidates;
            result.token = result.tokens.back();
            return result;
        }
        if (!decoded.greedy.empty()) return {{},decoded.greedy.at(std::size_t(request->row)),{}};
        auto next=sample_logits(*request->sampler,ops::slice(decoded.logits,0,request->row,request->row+1),
            request->counts,cursor(request));
        const auto token=ops::item(next);
        return {std::move(next),token,{}};
    }
    void accept(const std::shared_ptr<Request>& request, const Sample&, std::span<const std::int64_t> emitted) {
        if (request->unresolved) resolve(request, emitted);
        for (const auto token : emitted) {
            if (request->token_constraint) request->token_constraint->accept(token);
            if (request->counts) ops::add_token_counts(request->counts, ids_for({token}));
        }
    }
    void report(const std::shared_ptr<Request>& request, InferenceOutput& output) const {
        output.metrics.mtp=request->mtp_stats;
    }
    void recover(State& state) {
        auto scope=executor_.enter();
        std::exception_ptr failure;
        try { weights_.drain(); } catch (...) { failure=std::current_exception(); }
        try { drain_predictor(); } catch (...) { if (!failure) failure=std::current_exception(); }
        try { executor_.drain(); } catch (...) { if (!failure) failure=std::current_exception(); }
        routes_.reset(); route_keep_.clear(); unresolved_=0; committed_.clear();
        for (const auto& request:state.active) discard_prefill(request);
        for (const auto& request:state.prefilling) discard_prefill(request);
        batch_->recover();
        if (failure) std::rethrow_exception(failure);
    }

private:
    void drain_predictor() {
        if (!predictor_execution_) return;
        auto scope=predictor_execution_->enter();
        predictor_execution_->drain();
    }
    void rollback(const std::shared_ptr<Request>& request) {
        if (request->state) {
            auto* target=request->state->target();
            if (target && target->speculating()) target->rollback_speculative(0);
        }
        if (request->unresolved) {
            request->unresolved=false;
            if (unresolved_ && !--unresolved_) {
                routes_->resolve(route_keep_);
                routes_.reset(); route_keep_.clear();
            }
        }
    }
    static MfqTokenConstraintPtr cursor(const std::shared_ptr<Request>& request) {
        auto value=request->token_constraint ? request->token_constraint->clone() : MfqTokenConstraintPtr{};
        if (request->token_constraint && !value) throw std::logic_error("batch constraint cursor is unavailable");
        return value;
    }
    Tensor ids_for(std::vector<std::int64_t> tokens) const {
        return ops::reshape(ops::tensor(tokens,device_),{1,-1});
    }
    models::MtpState& predictor(const std::shared_ptr<Request>& request) const {
        auto* target=request->state->target();
        if (!target || !target->predictor) throw std::logic_error("batch predictor state is unavailable");
        return *target->predictor;
    }
    void prime(const std::shared_ptr<Request>& request, Tensor hidden, Tensor ids) {
        auto options=forward_;
        options.serial_experts=false;
        const auto count=ids.size(1);
        const auto limit=mtp_->prefill_chunk_size(count);
        if (limit<=0) throw std::logic_error("invalid batch predictor prefill chunk");
        for (std::int64_t first=0;first<count;first+=limit) {
            const auto last=std::min(count,first+limit);
            auto chunk=ops::contiguous(ops::slice(hidden,1,first,last));
            auto tokens=ops::contiguous(ops::slice(ids,1,first,last));
            auto& state=predictor(request);
            budget_->reserve(mtp_->growth(state,chunk,tokens,{},options));
            auto scope=ops::dense_scope(predictor_execution_,device_);
            auto result=mtp_->prefill(state,chunk,tokens,{},options);
            predictor_execution_->submit(state.storage());
            const std::array<Tensor,3> roots{result.sample_hidden,result.chain_hidden,result.sample_logits};
            predictor_execution_->submit(roots);
        }
    }
    void prime_prompt(const std::shared_ptr<Request>& request, PrefillChunk chunk, const Tensor& raw) {
        if (!request->speculative) return;
        if (!raw || raw.dim()!=3 || raw.size(0)!=1 || raw.size(1)!=chunk.count)
            throw std::logic_error("batch MTP prefill requires complete raw hidden rows");
        if (chunk.offset) {
            if (!request->raw) throw std::logic_error("batch MTP prompt boundary is unavailable");
            prime(request,request->raw,ops::slice(request->ids,1,chunk.offset,chunk.offset+1));
        }
        if (chunk.count>1)
            prime(request,ops::slice(raw,1,0,chunk.count-1),
                ops::slice(request->ids,1,chunk.offset+1,chunk.offset+chunk.count));
        request->raw=ops::clone(ops::slice(raw,1,chunk.count-1,chunk.count));
        executor_.submit(std::span<const Tensor>{&request->raw,1});
        if (predictor(request).position()!=request->state->position()-1)
            throw std::logic_error("batch MTP prompt history boundary mismatch");
    }
    Tensor probability_logits(Tensor logits, Tensor counts, const MfqSamplingParams& sampling) const {
        logits=ops::reshape(ops::contiguous(logits),{1,-1});
        if (counts) {
            logits=ops::clone(logits);
            ops::apply_penalties(logits,counts,sampling.presence_penalty,sampling.frequency_penalty,sampling.repetition_penalty);
        }
        return logits;
    }
    static bool compact(const Request& request) {
        return !request.sampler->greedy() && request.sampling.top_k>0 && request.sampling.top_k<=64;
    }
    void draft(const std::vector<std::shared_ptr<Request>>& requests) {
        struct Row {
            std::shared_ptr<Request> request;
            Tensor hidden, ids, counts;
            int depth;
        };
        std::vector<Row> active;
        for (const auto& request:requests) if (request->speculative) {
            request->started=mtp::DepthPolicy::Clock::now();
            request->predictor_base=predictor(request).position();
            if (request->predictor_base!=request->state->position()-1 || !request->raw)
                throw std::logic_error("batch MTP committed history mismatch");
            const auto remaining=std::min<std::int64_t>(request->generation_limit-request->produced-1,
                max_context()-request->state->position()-1);
            const auto depth=std::max(0,int(std::min<std::int64_t>(request->depth->depth(),remaining)));
            if (depth) active.push_back({request,request->raw,ids_for({request->pending_token}),
                request->counts ? ops::clone(request->counts) : Tensor{},depth});
        }
        auto options=forward_;
        options.serial_experts=false;
        for (int depth=0;!active.empty();++depth) {
            std::vector<models::MtpRow> rows;
            models::BatchGrowth growth;
            for (const auto& row:active) {
                auto& state=predictor(row.request);
                rows.push_back({&state,row.hidden,row.ids,{}});
                merge_growth(growth,mtp_->growth(state,row.hidden,row.ids,{},options));
            }
            reserve(growth,std::size_t(64)<<20);
            std::vector<models::MtpStep> outputs;
            {
                auto scope=ops::dense_scope(predictor_execution_,device_);
                outputs=mtp_->batch_step(rows,options);
                if (outputs.size()!=rows.size()) throw std::logic_error("invalid batched predictor output count");
                for (std::size_t i=0;i<rows.size();++i) {
                    predictor_execution_->submit(rows[i].state->storage());
                    const auto& value=outputs[i];
                    const std::array<Tensor,3> roots{value.sample_hidden,value.chain_hidden,value.sample_logits};
                    predictor_execution_->submit(roots);
                }
            }
            std::vector<Tensor> hidden;
            for (const auto& value:outputs) hidden.push_back(value.sample_hidden);
            auto input=ops::contiguous(ops::cat(hidden,0));
            if (!mtp_->preserve_output_dtype()) input=ops::cast(input,kFloat16);
            auto logits=ops::reshape(model_.logits(input,true,models::LogitPrecision::input),
                {std::int64_t(active.size()),vocab_size()});
            for (std::size_t i=0;i<active.size();++i) {
                auto& row=active[i];
                auto& request=*row.request;
                const auto sampling=mtp_->draft_sampling(request.sampling,compact(request));
                auto values=probability_logits(ops::slice(logits,0,i,i+1),row.counts,sampling);
                std::int32_t token;
                if (request.sampler->greedy()) token=ops::item(ops::sample_greedy(values));
                else if (compact(request)) {
                    auto proposal=mtp::compact_rows(values,sampling).at(0);
                    token=mtp::sample_compact(proposal,request.sampler->next_uniform());
                    request.compact_proposals.push_back(std::move(proposal));
                } else {
                    auto proposal=mtp::probabilities(values,sampling);
                    token=mtp::sample(proposal,request.sampler->next_uniform());
                    request.proposals.push_back(std::move(proposal));
                }
                if (token<0 || token>=vocab_size()) throw std::logic_error("batch predictor token outside vocabulary");
                request.drafts.push_back(token);
                row.hidden=outputs[i].chain_hidden;
                row.ids=ids_for({token});
                if (row.counts) ops::add_token_counts(row.counts,row.ids);
            }
            ++draft_batches_; draft_rows_+=active.size();
            std::erase_if(active,[&](const auto& row) { return row.depth<=depth+1; });
        }
    }
    void verify(const std::shared_ptr<Request>& request) {
        auto& value=*request;
        const auto count=value.drafts.size();
        auto logits=ops::reshape(value.verified.logits,{std::int64_t(count)+1,vocab_size()});
        auto counts=value.counts ? ops::clone(value.counts) : Tensor{};
        mtp::ChainVerification result;
        if (!count) {
            auto next=sample_logits(*value.sampler,ops::slice(logits,0,0,1),counts,cursor(request));
            result={0,std::int32_t(ops::item(next)),true};
        } else {
            std::vector<std::int32_t> targets;
            std::vector<mtp::CompactDistribution> compact_targets;
            std::vector<std::vector<float>> probabilities;
            for (std::size_t i=0;i<=count;++i) {
                auto row=probability_logits(ops::slice(logits,0,i,i+1),counts,value.sampling);
                if (value.sampler->greedy()) targets.push_back(ops::item(ops::sample_greedy(row)));
                else if (compact(value)) compact_targets.push_back(mtp::compact_rows(row,value.sampling).at(0));
                else probabilities.push_back(mtp::probabilities(row,value.sampling));
                if (counts && i<count) ops::add_token_counts(counts,ids_for({value.drafts[i]}));
            }
            if (value.sampler->greedy()) result=mtp::verify_greedy(value.drafts,targets);
            else {
                std::vector<double> uniforms(count);
                std::generate(uniforms.begin(),uniforms.end(),[&] { return value.sampler->next_uniform(); });
                result=compact(value)
                    ? mtp::verify_compact_chain(value.drafts,value.compact_proposals,compact_targets,uniforms,value.sampler->next_uniform())
                    : mtp::verify_stochastic_chain(value.drafts,value.proposals,probabilities,uniforms,value.sampler->next_uniform());
            }
            auto constraint=cursor(request);
            if (constraint) {
                counts=value.counts ? ops::clone(value.counts) : Tensor{};
                for (std::size_t i=0;i<=result.accepted_drafts;++i) {
                    const auto token=i<result.accepted_drafts ? value.drafts[i] : result.next_token;
                    if (!constraint->allows(token)) {
                        result.accepted_drafts=i;
                        auto next=sample_logits(*value.sampler,ops::slice(logits,0,i,i+1),counts,constraint);
                        result.next_token=ops::item(next);
                        break;
                    }
                    constraint->accept(token);
                    if (counts) ops::add_token_counts(counts,ids_for({token}));
                }
            }
        }
        if (result.accepted_drafts>count || result.next_token<0 || result.next_token>=vocab_size())
            throw std::logic_error("invalid batch MTP verification result");
        value.accepted=result.accepted_drafts;
        value.candidates.assign(value.drafts.begin(),value.drafts.begin()+value.accepted);
        value.candidates.push_back(result.next_token);
    }
    Decoded decode_mtp(State& state) {
        std::vector<std::shared_ptr<Request>> requests;
        std::vector<models::BatchState*> states;
        for (const auto& request:state.active) if (request->eligible) {
            if (!request->state || request->state->position()!=request->cache_length || request->unresolved)
                throw std::logic_error("batch MTP target position mismatch");
            request->row=std::int64_t(requests.size());
            requests.push_back(request);
            states.push_back(request->state.get());
        }
        suspend_decode();
        const auto draft_started=mtp::DepthPolicy::Clock::now();
        draft(requests);
        const auto draft_ms=std::chrono::duration<double,std::milli>(mtp::DepthPolicy::Clock::now()-draft_started).count();
        for (const auto& request:requests) if (request->speculative) request->mtp_stats.head_ms+=draft_ms;
        std::vector<Tensor> ids;
        models::BatchGrowth growth;
        std::size_t tokens=0;
        for (const auto& request:requests) {
            std::vector<std::int64_t> window{request->pending_token};
            window.insert(window.end(),request->drafts.begin(),request->drafts.end());
            ids.push_back(ids_for(std::move(window)));
            models::Input input;
            input.ids=ids.back(); input.confirmed=request->drafts.empty() ? 0 : 1;
            merge_growth(growth,request->state->target()->growth(input,forward_));
            request->route_offset=tokens;
            tokens+=ids.back().size(1);
        }
        reserve(growth,std::size_t(64)<<20);
        routes_=std::make_unique<weights::Cache::Routes>();
        routes_->begin(weights_,int(tokens));
        route_keep_.assign(tokens,0);
        unresolved_=requests.size();
        auto timer=executor_.timer();
        auto outputs=run_prefill(std::move(states),std::move(ids),true);
        if (outputs.size()!=requests.size()) throw std::logic_error("invalid MTP target output count");
        timer->stop();
        const auto elapsed=timer->elapsed_ms();
        for (std::size_t i=0;i<requests.size();++i) {
            const auto& request=requests[i];
            request->verified=std::move(outputs[i]);
            request->unresolved=true;
            verify(request);
            if (request->speculative) request->mtp_stats.target_ms+=elapsed;
        }
        ++mtp_batches_; ++decode_batches_; decode_tokens_+=tokens;
        return {{},{},true};
    }
    void resolve(const std::shared_ptr<Request>& request, std::span<const std::int64_t> emitted) {
        auto scope=executor_.enter();
        const auto accepted=std::min(request->accepted,emitted.size());
        auto* target=request->state->target();
        if (!request->drafts.empty()) {
            if (accepted==request->drafts.size()) target->commit_speculative();
            else {
                if (!target->retains_partial_prefix(accepted))
                    throw std::logic_error("batched MTP target cannot retain the accepted prefix");
                target->rollback_speculative(accepted);
            }
        }
        request->accepted=accepted;
        if (!emitted.empty())
            std::fill_n(route_keep_.begin()+request->route_offset,accepted+1,std::uint8_t(1));
        request->unresolved=false;
        committed_.push_back(request);
        if (--unresolved_) return;
        routes_->resolve(route_keep_);
        routes_.reset(); route_keep_.clear();
        // Predictor replay can visit the same expert pool after target learning commits.
        for (const auto& row:committed_) commit_predictor(row);
        committed_.clear();
    }
    void commit_predictor(const std::shared_ptr<Request>& request) {
        const auto started=mtp::DepthPolicy::Clock::now();
        const auto accepted=request->accepted;
        auto* target=request->state->target();
        if (request->speculative) {
            auto& state=predictor(request);
            const auto base=request->predictor_base;
            if (!request->drafts.empty()) {
                mtp_->trim_cache_to(state,base+1);
                if (accepted) {
                    std::vector<std::int64_t> ids(request->drafts.begin(),request->drafts.begin()+accepted);
                    prime(request,ops::slice(request->verified.raw,1,0,accepted),ids_for(std::move(ids)));
                }
            } else prime(request,request->raw,ids_for({request->history.at(base+1)}));
            request->raw=ops::clone(ops::slice(request->verified.raw,1,accepted,accepted+1));
            executor_.submit(std::span<const Tensor>{&request->raw,1});
            if (state.position()!=target->position()-1)
                throw std::logic_error("batch MTP committed predictor boundary mismatch");
            const auto finished=mtp::DepthPolicy::Clock::now();
            const int depth=request->drafts.size();
            if (depth) request->depth->observe_window(depth,accepted,request->started,finished);
            else request->depth->observe_plain_window(request->started,finished);
            auto& stats=request->mtp_stats;
            ++stats.cycles; stats.drafted_tokens+=depth; stats.accepted_tokens+=accepted;
            ++stats.depth_cycles.at(depth);
            for (int i=0;i<depth;++i) ++stats.position_drafted.at(i);
            for (std::size_t i=0;i<accepted;++i) ++stats.position_accepted.at(i);
            if (!depth) ++stats.standard_tokens;
            stats.selected_depth=request->depth->depth();
            stats.park_count=request->depth->parks(); stats.reentry_probes=request->depth->reentries();
            stats.rollback_ms+=std::chrono::duration<double,std::milli>(finished-started).count();
            for (int i=0;i<=request->depth->maximum_depth();++i)
                if (auto cost=request->depth->measured_cycle_ms(i)) stats.measured_depth_ms.at(i)=*cost;
        }
        request->verified={}; request->drafts.clear(); request->proposals.clear();
        request->compact_proposals.clear(); request->candidates.clear();
    }
    void checkpoint(const std::shared_ptr<Request>& request) {
        if (!request->caching || !request->state ||
            request->state->position() == static_cast<std::int64_t>(request->checkpoint)) return;
        const auto saved = save_session(*request->state->target(), sessions_,
            request->cache_plan.session_id, request->history, {},
            request->speculative ? mtp_ : nullptr, request->raw);
        if (saved) request->checkpoint = saved;
    }
    void reserve(const models::BatchGrowth& values, std::size_t margin, bool primary=false) {
        reserve_batch(*budget_,device_,values,margin,primary);
    }
    std::vector<models::Decoded> run_prefill(std::vector<models::BatchState*> states, std::vector<Tensor> ids, bool verify=false) {
        return tyloi::engine::prefill_batch(*batch_,executor_,weights_,*budget_,device_,std::move(states),std::move(ids),verify);
    }
    std::unique_ptr<models::Batch::Plan> plan_graph(const std::vector<models::BatchRow>& rows, bool greedy) {
        return plan_batch(*batch_,executor_,weights_,*budget_,device_,rows,greedy);
    }
    void initialize(const std::shared_ptr<Request>& request, PrefillChunk chunk) {
        if (!request->state) {
            if (chunk.offset!=0) throw std::runtime_error("batch lost request state");
            std::erase_if(requests_,[](const auto& weak){return weak.expired();});
            const bool active=std::any_of(requests_.begin(),requests_.end(),[](const auto& weak) {
                const auto row=weak.lock(); return row && row->state;
            });
            if (!active) { weights_.retire_decode(); executor_.reclaim(device_); }
            requests_.push_back(request);
            auto state=batch_->make_state(static_cast<std::int64_t>(request->prompt.size()));
            reserve(state->initial_growth_bytes(),std::size_t(64)<<20);
            request->state=std::move(state);
            request->state->initialize();
            request->speculative = request->sampling.enable_mtp && supports_mtp();
            if (request->speculative) {
                auto* target=request->state->target();
                if (!target) throw std::logic_error("batch predictor requires an independent target state");
                target->predictor=mtp_->make_state();
                budget_->reserve(mtp_->reset_growth(*target->predictor,1));
                mtp_->reset(*target->predictor,1);
                const bool compact=request->sampling.temperature>0 &&
                    request->sampling.top_k>1 && request->sampling.top_k<=64;
                const bool greedy=request->sampling.temperature<=0 || request->sampling.top_k==1;
                const int maximum=std::min(mtp_->maximum_draft_depth(),!greedy && !compact ? 1 :
                    std::clamp<int>(request->sampling.mtp_max_draft_tokens,1,mtp::kMaximumDraftDepth));
                request->depth=mtp_->depth_policy(maximum,request->policy);
                request->mtp_stats.available=request->mtp_stats.used=true;
            }
            request->ids=ops::reshape(ops::tensor(request->prompt,device_),{1,-1});
            request->sampler.emplace(request->sampling,device_);
            if (request->sampler->has_penalties()) {
                request->counts=ops::zeros({vocab_size()},TensorOptions().device(device_).dtype(kInt32));
                ops::add_token_counts(request->counts,request->ids);
            }
        }
        if (request->state->position()!=chunk.offset) throw std::runtime_error("batch prefill position mismatch");
    }
    Sample finish(const std::shared_ptr<Request>& request, Tensor next) {
        if (!next) return {};
        const auto token=ops::item(next);
        double ms=request->packed_ms;
        for (const auto& timer:request->timers) ms+=timer->elapsed_ms();
        request->timers.clear(); request->ids={}; request->packed_ms=0;
        return {std::move(next),token,{request->prompt.size(),ms,0.0,ms}};
    }
    models::CausalModel& model_;
    std::unique_ptr<models::Batch> batch_;
    Executor& executor_;
    weights::Cache& weights_;
    Budget* budget_;
    state::Sessions& sessions_;
    ContinuousBatchConfig config_;
    Device device_;
    std::size_t workspace_;
    bool packed_;
    std::int64_t packed_tokens_;
    models::ForwardOptions forward_;
    models::Mtp* mtp_;
    Executor* predictor_execution_;
    std::unique_ptr<weights::Cache::Routes> routes_;
    std::vector<std::uint8_t> route_keep_;
    std::size_t unresolved_=0;
    std::vector<std::shared_ptr<Request>> committed_;
    std::vector<std::weak_ptr<Request>> requests_;
    std::int64_t decode_batches_=0,decode_tokens_=0,greedy_batches_=0;
    std::int64_t packed_batches_=0,packed_requests_=0,packed_rows_=0;
    std::int64_t mtp_batches_=0,draft_batches_=0,draft_rows_=0;
};
} // namespace

std::unique_ptr<Batch> make_batch(models::CausalModel& model, Executor& executor,
    weights::Cache& weights, Budget& budget, state::Sessions& sessions,
    const Config& config, const models::ForwardOptions& forward, Executor* predictor_execution) {
    const auto& settings=config.continuous_batch;
    auto scope=executor.enter();
    auto batch = model.make_batch({settings.max_sequences,settings.paged_kv,settings.preallocate,
        settings.cpu_offload,settings.cached_slots,settings.cuda_graph,settings.greedy,
        settings.cuda_graph_minimum_tokens,forward});
    if (!batch) throw std::invalid_argument("continuous batching is unavailable for this model");
    try {
        reserve_batch(budget, config.device, batch->initial_growth_bytes(), std::size_t(64)<<20);
        batch->initialize();
        weights.finish_exchange();
        executor.drain();
    } catch (...) {
        const auto failure=std::current_exception();
        try { weights.finish_exchange(); } catch (...) {}
        try { executor.drain(); } catch (...) {}
        batch.reset();
        std::rethrow_exception(failure);
    }
    return std::make_unique<ContinuousBatch<Ops>>(config.generation.prefill_chunk_size,
        config.continuous_batch.prefill_token_budget,model,std::move(batch),executor,weights,budget,
        sessions,config.continuous_batch,config.device,config.workspace_reserve,config.continuous_batch.packed_prefill,
        config.continuous_batch.packed_tokens,forward,predictor_execution);
}
} // namespace tyloi::engine

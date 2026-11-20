#pragma once

#include "continuous_batch.h"
#include "../models/dflash2/mlx_dflash2.h"
#include "mlx_dflash2_serving.h"
#include "mlx_qwen4_causal_lm.h"
#include "mlx_resident_budget.h"
#include "mlx_stream_sync.h"
#include "mtp_batch_policy.h"
#include "mtp_policy.h"

#include <chrono>
#include <iostream>
#include <unordered_map>

namespace mfq::metal {

struct MlxMtpBatchRound {
    std::shared_ptr<mfq::engine::mtp::BatchDepthPolicy> policy;
    mfq::engine::mtp::BatchDepthPolicy::Clock::time_point started;
    mfq::engine::mtp::BatchDepthPolicy::Clock::time_point finished;
    std::vector<mfq::engine::mtp::BatchAcceptance> accepted;
    std::size_t expected;
    std::uint64_t prefill_epoch;
    int depth;
    int maximum_drafts;
    bool stable;
};

inline MlxSamplingParams mlx_batch_sampling(const MfqSamplingParams& input) {
    MlxSamplingParams result;
    result.temperature = input.temperature;
    result.top_k = input.top_k;
    result.top_p = input.top_p;
    result.presence_penalty = input.presence_penalty;
    result.frequency_penalty = input.frequency_penalty;
    result.repetition_penalty = input.repetition_penalty;
    result.enable_mtp = input.enable_mtp;
    result.mtp_max_draft_tokens = input.mtp_max_draft_tokens;
    result.seed = input.seed;
    return result;
}

struct MlxQwen4BatchRequest : mfq::engine::BatchRequest {
    MlxQwen4BatchRequest(std::string id, const mfq::engine::InferenceRequest& input)
        : BatchRequest(std::move(id), input), sampler(mlx_batch_sampling(input.sampling)),
          cache_plan(input.cache_plan), history(input.prompt) {}
    MlxSampler sampler;
    MfqPromptCachePlan cache_plan;
    std::shared_ptr<MlxQwen4RequestCache> cache;
    std::optional<mlx::core::array> counts;
    std::optional<mlx::core::array> initial_logits;
    std::optional<MlxMtpTargetRequest> target_work;
    std::optional<MlxMtpTargetBatch> target_result;
    MlxMtpGenerationStats mtp_stats;
    std::optional<MlxDFlash2Context> dflash_context;
    int next_depth = 0, budget_depth = 0, budget_rows = 0;
    int scheduled_budget = 0, scheduled_rows = 0;
    std::array<std::uint64_t, mfq::engine::mtp::kMaximumReportedDraftDepth> reached{}, hits{};
    std::uint64_t mtp_parks = 0, mtp_reentries = 0;
    bool mtp_parked = false;
    int observe_mtp_depth(int depth, bool parked) {
        if (parked && !mtp_parked) { ++mtp_parks; mtp_parked = true; }
        else if (depth && mtp_parked) { ++mtp_reentries; mtp_parked = false; }
        return depth;
    }
    bool mtp = false;
    std::shared_ptr<MlxMtpBatchRound> round;
    int round_depth = 0;
    std::chrono::steady_clock::time_point draft_started;
    std::uint64_t draft_epoch = 0;
    bool draft_calibration = true;
    std::vector<std::int64_t> history;
    std::size_t prefix = 0, checkpoint = 0;
};

class MlxQwen4BatchOperations {
public:
    using Request = MlxQwen4BatchRequest;
    using State = mfq::engine::BatchState<Request>;
    struct Sample {
        std::int64_t token = 0;
        MfqPrefillTiming timing;
    };
    struct Decoded {
        mlx::core::array logits;
        std::unordered_map<std::string, int> rows;
        std::vector<std::int64_t> sampled_tokens;
        bool only_draft_heads = false;
    };
    struct Hooks {
        std::size_t block_size = 0;
        std::function<std::size_t(Request&)> restore;
        std::function<void(Request&, MlxQwen4TextSessionState)> store;
        std::shared_ptr<MlxDFlash2> dflash2;
    };

    MlxQwen4BatchOperations(MlxQwen4CausalLm& runtime, Hooks hooks)
        : runtime_(runtime), hooks_(std::move(hooks)) {}

    std::int64_t prepare(const std::shared_ptr<Request>& request) {
        if (hooks_.dflash2 && mlx_dflash2_sampling_supported(request->sampling, request->token_constraint))
            hooks_.dflash2->reset(request->dflash_context.emplace());
        request->mtp = request->dflash_context || (request->sampling.enable_mtp && runtime_.supports_mtp());
        runtime_.clear_cache();
        request->cache = runtime_.make_request_cache();
        const auto restored = hooks_.restore && !request->dflash_context ? hooks_.restore(*request) : 0;
        if (restored) runtime_.adopt_request_cache(*request->cache);
        if (request->sampler.params().has_penalties()) {
            request->counts = sample_token_counts_add(mlx::core::zeros(
                {1, static_cast<int>(runtime_.config().vocab_size)}, mlx::core::float32),
                mlx::core::array(request->prompt.begin(),
                    {1, static_cast<int>(request->prompt.size())}, mlx::core::int32));
        }
        return restored;
    }
    std::int64_t prefix_tokens(const std::shared_ptr<Request>& request) { return request->prefix; }
    mfq::engine::PrefillChunk prefill_chunk(const std::shared_ptr<Request>& request,
        mfq::engine::PrefillChunk chunk) {
        if (request->prefix && hooks_.block_size) {
            const auto end = std::min<std::int64_t>(chunk.offset + chunk.count,
                (chunk.offset / hooks_.block_size + 1) * hooks_.block_size);
            chunk.count = end - chunk.offset;
            chunk.last = end == static_cast<std::int64_t>(request->prompt.size());
        }
        return chunk;
    }
    Sample prefill(const std::shared_ptr<Request>& request, mfq::engine::PrefillChunk chunk) {
        return prefill_batch({{request, chunk}}).front();
    }
    std::vector<Sample> prefill_batch(
        const std::vector<std::pair<std::shared_ptr<Request>, mfq::engine::PrefillChunk>>& pending) {
        ++prefill_epoch_;
        if (policy_) policy_->interrupt();
        const auto started = std::chrono::steady_clock::now();
        std::vector<std::shared_ptr<MlxQwen4RequestCache>> caches;
        std::vector<std::vector<std::int64_t>> tokens;
        std::vector<bool> mtp;
        bool capture_dflash = false;
        for (const auto& [request, chunk] : pending) {
            caches.push_back(request->cache);
            tokens.emplace_back(request->prompt.begin() + chunk.offset,
                request->prompt.begin() + chunk.offset + chunk.count);
            mtp.push_back(request->mtp && !request->dflash_context);
            capture_dflash |= request->dflash_context.has_value();
        }
        DFlashCapture capture(runtime_, capture_dflash ? hooks_.dflash2.get() : nullptr);
        auto logits = runtime_.prefill_requests(caches, tokens, mtp);
        if (auto features = capture.take()) {
            int offset = 0;
            for (const auto& [request, chunk] : pending) {
                const int end = offset + static_cast<int>(chunk.count);
                if (request->dflash_context) hooks_.dflash2->append_context(*request->dflash_context,
                    feature_rows(*features, offset, end), static_cast<int>(chunk.offset));
                offset = end;
            }
        }
        const auto computed = std::chrono::steady_clock::now();
        for (const auto& [request, chunk] : pending) checkpoint(*request, chunk.offset + chunk.count);
        const auto checkpointed = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double, std::milli>(
            checkpointed - started).count();
        prefill_compute_ms_ += std::chrono::duration<double, std::milli>(computed - started).count();
        prefill_checkpoint_ms_ += std::chrono::duration<double, std::milli>(checkpointed - computed).count();
        prefill_maximum_ms_ = std::max(prefill_maximum_ms_, elapsed);
        std::vector<Sample> result(pending.size());
        for (std::size_t row = 0; row < pending.size(); ++row) {
            const auto& [request, chunk] = pending[row];
            result[row].timing.llm_ms = elapsed;
            if (chunk.offset + chunk.count == static_cast<int64_t>(request->prompt.size())) {
                auto last = mlx_last_token_logits(logits[row], static_cast<int>(runtime_.config().vocab_size));
                if (request->mtp) request->initial_logits = std::move(last);
                else result[row].token = sample_logits(*request, last);
            }
        }
        ++prefill_batches_;
        prefill_rows_ += pending.size();
        prefill_maximum_ = std::max(prefill_maximum_, pending.size());
        return result;
    }
    void suspend_decode() {}
    void resume_decode(std::int64_t) {}
    bool ready_decode(const State& state, const std::vector<std::string>& eligible) const {
        const bool speculative = std::any_of(state.active.begin(), state.active.end(),
            [](const auto& request) { return request->mtp && request->eligible; });
        if (!speculative) return true;
        for (const auto& request : state.active) if (request->mtp && !request->eligible &&
            request->produced < request->generation_limit &&
            std::find(eligible.begin(), eligible.end(), request->id) != eligible.end()) return false;
        return true;
    }
    void synchronize(const State& state) {
        std::vector<std::string> ids;
        std::vector<std::string> ordinary;
        int maximum = 1;
        std::vector<Request*> lanes;
        for (const auto& request : state.active) if (request->mtp && request->produced < request->generation_limit) {
            ids.push_back(request->id);
            maximum = std::max(maximum, draft_cap(*request));
            lanes.push_back(request.get());
        }
        std::sort(ids.begin(), ids.end());
        for (const auto& request : state.active) if (!request->mtp && request->produced < request->generation_limit)
            ordinary.push_back(request->id);
        std::sort(ordinary.begin(), ordinary.end());
        if (!policy_ || policy_->requests() != ids || policy_->maximum() != maximum || ordinary != ordinary_cohort_) {
            if (policy_) park_memory_.retired(*policy_);
            policy_.reset();
            if (!ids.empty()) {
                policy_ = std::make_shared<mfq::engine::mtp::BatchDepthPolicy>(std::move(ids), maximum, ordinary.size());
                park_memory_.seed(*policy_);
            }
            ordinary_cohort_ = std::move(ordinary);
        }
        for (const auto& round : completed_rounds_) {
            if (round->policy && round->policy == policy_) {
                const auto previous_parks = policy_->parks();
                const bool was_parked = policy_->remaining() != 0;
                policy_->observe(round->accepted, round->started,
                    round->finished, round->stable, round->prefill_epoch != prefill_epoch_, round->depth);
                if (policy_->parks() > previous_parks) park_memory_.parked(*policy_);
                if (was_parked && !policy_->remaining()) park_memory_.resumed(*policy_);
            }
        }
        completed_rounds_.clear();
        if (policy_) assign_verify_rows(lanes);
    }
    void invalidate_runtime_policy() {
        completed_rounds_.clear();
        policy_.reset();
        park_memory_ = {};
    }
    void activate(const std::shared_ptr<Request>& request, const Sample& result) {
        if (!request->mtp) accept(request, result);
    }
    std::int64_t cache_length(const std::shared_ptr<Request>& request) const {
        return runtime_.request_cache_position(*request->cache);
    }
    std::optional<mfq::engine::Generation> generate_decode(const std::shared_ptr<Request>& request,
        mfq::engine::InferenceOutput& output, std::function<void()> schedule,
        std::function<bool()> ready, std::function<void()> check) {
        if (!request->mtp) return std::nullopt;
        if (request->dflash_context)
            return generate_dflash2(request, output, std::move(schedule), std::move(ready), std::move(check));
        return generate_mtp(request, output, std::move(schedule), std::move(ready), std::move(check));
    }
    void accept(const std::shared_ptr<Request>& request, const Sample& result) {
        if (request->token_constraint) request->token_constraint->accept(result.token);
        if (request->counts) *request->counts = sample_token_counts_add(*request->counts,
            mlx::core::array({result.token}, {1, 1}, mlx::core::int32));
        request->history.push_back(result.token);
    }
    Decoded decode(const State& state) {
        bool speculative = false;
        for (const auto& request : state.active) if (request->eligible && request->mtp) speculative = true;
        if (speculative) return decode_mtp(state);
        std::vector<std::shared_ptr<MlxQwen4RequestCache>> caches;
        std::vector<std::int64_t> tokens;
        std::unordered_map<std::string, int> rows;
        for (const auto& request : state.active) if (request->eligible) {
            rows.emplace(request->id, static_cast<int>(caches.size()));
            caches.push_back(request->cache);
            tokens.push_back(request->pending_token);
        }
        if (!caches.empty()) park_memory_.tick();
        auto logits = runtime_.decode_requests(caches, tokens);
        ++decode_batches_;
        decode_rows_ += caches.size();
        maximum_batch_ = std::max(maximum_batch_, caches.size());
        for (const auto& request : state.active) if (request->eligible)
            checkpoint(*request, request->cache_length + 1);
        const auto vocab = static_cast<int>(runtime_.config().vocab_size);
        if (caches.size() > 1 && std::all_of(state.active.begin(), state.active.end(),
            [](const auto& request) { return !request->eligible ||
                (request->sampler.params().greedy() && !request->counts); })) {
            auto selected = sample_greedy(mlx::core::reshape(logits,
                {static_cast<int>(caches.size()), vocab}));
            selected.eval();
            const auto* values = selected.data<std::int32_t>();
            return {std::move(logits), std::move(rows), {values, values + caches.size()}};
        }
        std::vector<mlx::core::array> selected;
        selected.reserve(caches.size());
        for (const auto& request : state.active) if (request->eligible) {
            const int row = rows.at(request->id);
            auto value = mlx_last_token_logits(mlx::core::slice(logits,
                {row, 0, 0}, {row + 1, 1, vocab}), vocab);
            selected.push_back(request->counts ? request->sampler.sample(value, *request->counts)
                : request->sampler.sample(value));
        }
        mlx::core::eval(selected);
        std::vector<std::int64_t> sampled;
        sampled.reserve(selected.size());
        for (const auto& value : selected) sampled.push_back(value.data<std::int32_t>()[0]);
        return {std::move(logits), std::move(rows), std::move(sampled)};
    }
    Sample sample(const std::shared_ptr<Request>& request, const Decoded& decoded) {
        if (request->mtp) return {};
        const auto row = decoded.rows.at(request->id);
        auto logits = mlx::core::slice(decoded.logits, {row, 0, 0},
            {row + 1, 1, static_cast<int>(runtime_.config().vocab_size)});
        return {sample_logits(*request, mlx_last_token_logits(logits,
            static_cast<int>(runtime_.config().vocab_size)), decoded.sampled_tokens.at(row)), {}};
    }
    bool decode_completed(const std::shared_ptr<Request>& request, const Decoded& decoded) const {
        return !decoded.only_draft_heads || (request->mtp &&
            request->target_work->kind != MlxMtpTargetRequest::Kind::target);
    }
    void discard_prefill(const std::shared_ptr<Request>& request) {
        request->cache.reset();
        request->dflash_context.reset();
    }
    void retire(const std::vector<std::shared_ptr<Request>>& requests, std::int64_t) {
        for (const auto& request : requests) {
            runtime_.record_mtp_stats(request->mtp_stats);
            if (request->target_result && request->target_work && request->target_work->draft_count) {
                runtime_.resolve_request(*request->cache, 0);
                request->cache_length = runtime_.request_cache_position(*request->cache);
            }
            checkpoint(*request, request->cache_length, true);
            request->cache.reset();
            request->target_work.reset();
            request->target_result.reset();
            request->dflash_context.reset();
            request->round.reset();
        }
    }
    void recover(const State& state) {
        for (const auto& request : state.active) request->cache.reset();
        for (const auto& request : state.prefilling) request->cache.reset();
        runtime_.clear_cache();
        completed_rounds_.clear();
        policy_.reset();
    }
    bool recoverable_failure(std::exception_ptr error) const {
        try { std::rethrow_exception(error); }
        catch (const MlxResidentBudgetExceeded&) { return true; }
        catch (...) { return false; }
    }
    mfq::engine::Metrics metrics() const {
        mfq::engine::Metrics result{{"continuous_batching_decode_batches", double(decode_batches_)},
            {"continuous_batching_decode_rows", double(decode_rows_)},
            {"continuous_batching_maximum_batch", double(maximum_batch_)},
            {"continuous_batching_mtp_batches", double(mtp_batches_)},
            {"continuous_batching_mtp_rows", double(mtp_rows_)},
            {"continuous_batching_mtp_maximum_batch", double(mtp_maximum_batch_)},
            {"continuous_batching_mtp_verification_rows", double(mtp_verify_rows_)},
            {"continuous_batching_mtp_ragged_batches", double(mtp_ragged_batches_)},
            {"continuous_batching_mtp_draft_batches", double(mtp_draft_batches_)},
            {"continuous_batching_mtp_draft_rows", double(mtp_draft_rows_)},
            {"continuous_batching_mtp_draft_maximum_batch", double(mtp_draft_maximum_)},
            {"continuous_batching_dflash2_draft_batches", double(dflash_draft_batches_)},
            {"continuous_batching_dflash2_draft_rows", double(dflash_draft_rows_)},
            {"continuous_batching_dflash2_draft_maximum_batch", double(dflash_draft_maximum_)},
            {"continuous_batching_prefill_batches", double(prefill_batches_)},
            {"continuous_batching_prefill_rows", double(prefill_rows_)},
            {"continuous_batching_prefill_maximum_batch", double(prefill_maximum_)},
            {"continuous_batching_prefill_compute_ms", prefill_compute_ms_},
            {"continuous_batching_prefill_checkpoint_ms", prefill_checkpoint_ms_},
            {"continuous_batching_prefill_maximum_ms", prefill_maximum_ms_}};
        for (int depth = 0; depth <= mfq::engine::mtp::kMaximumReportedDraftDepth; ++depth) {
            const auto prefix = "continuous_batching_mtp_depth_" + std::to_string(depth);
            result.emplace_back(prefix + "_cycles", double(mtp_cycle_counts_[depth]));
            result.emplace_back(prefix + "_total_ms", mtp_cycle_totals_[depth]);
            result.emplace_back(prefix + "_maximum_ms", mtp_cycle_maxima_[depth]);
        }
        return result;
    }

private:
    int draft_cap(const Request& request) const {
        if (!request.dflash_context && request.sampling.temperature > 0 &&
            (request.sampling.top_k < 1 || request.sampling.top_k > 128)) return 1;
        const int maximum = request.dflash_context ? hooks_.dflash2->config().block_size - 1
            : kMlxMtpEngineMaximumDraftDepth;
        return std::clamp(request.sampling.mtp_max_draft_tokens, 1, maximum);
    }

    void assign_verify_rows(const std::vector<Request*>& requests) {
        struct Lane {
            Request* request;
            int cap;
            double reach = 1;
        };
        const int nominal = policy_->depth();
        int rows = 0;
        std::vector<Lane> lanes;
        for (auto* request : requests) {
            const int cap = draft_cap(*request);
            rows += std::min(nominal, cap);
            const auto position = std::int64_t(request->prompt.size()) + request->produced - 1;
            const auto remaining = std::min<std::int64_t>(request->generation_limit - request->produced - 1,
                runtime_.context_capacity() - position - 1);
            lanes.push_back({request, static_cast<int>(std::clamp<std::int64_t>(remaining, 0, cap))});
            request->next_depth = 0;
        }
        for (int row = 0; row < rows; ++row) {
            Lane* best = nullptr;
            double best_score = 0;
            for (auto& lane : lanes) {
                const int depth = lane.request->next_depth;
                if (depth == lane.cap) continue;
                const auto prior = policy_->conditional_acceptance(depth);
                const double score = lane.reach * (lane.request->hits[depth] + 8 * prior) /
                    (lane.request->reached[depth] + 8);
                if (!best || score > best_score || (score == best_score && depth < best->request->next_depth)) {
                    best = &lane;
                    best_score = score;
                }
            }
            if (!best) break;
            best->reach = best_score;
            ++best->request->next_depth;
        }
        for (auto* request : requests) {
            request->budget_depth = nominal;
            request->budget_rows = rows;
        }
    }

    int select_draft_depth(Request& request) const {
        request.scheduled_budget = request.budget_depth;
        request.scheduled_rows = request.budget_rows;
        return request.observe_mtp_depth(request.next_depth, policy_ && policy_->remaining());
    }

    void finish_round(Request& request, int accepted) {
        if (auto round = std::exchange(request.round, {})) {
            const int depth = request.round_depth;
            for (int position = 0; position < depth && position <= accepted; ++position) {
                ++request.reached[position];
                request.hits[position] += position < accepted;
            }
            round->accepted.push_back({depth, accepted});
            if (round->accepted.size() == round->expected) {
                round->finished = std::chrono::steady_clock::now();
                const auto ms = std::chrono::duration<double, std::milli>(round->finished - round->started).count();
                const int depth = round->maximum_drafts;
                ++mtp_cycle_counts_[depth];
                mtp_cycle_totals_[depth] += ms;
                mtp_cycle_maxima_[depth] = std::max(mtp_cycle_maxima_[depth], ms);
                completed_rounds_.push_back(std::move(round));
            }
        }
    }

    void publish_mtp_metrics(Request& request, mfq::engine::InferenceOutput& output) {
        auto& stats = request.mtp_stats;
        stats.park_count = request.mtp_parks;
        stats.reentry_probes = request.mtp_reentries;
        runtime_.record_mtp_stats(stats);
        output.metrics.mtp = {stats.available, stats.used, stats.cycles, stats.drafted_tokens,
            stats.accepted_tokens, stats.depth_cycles, stats.position_drafted, stats.position_accepted,
            stats.measured_depth_ms, stats.selected_depth, stats.standard_tokens, stats.park_count,
            stats.reentry_probes};
    }

    class DFlashCapture {
    public:
        DFlashCapture(MlxQwen4CausalLm& runtime, const MlxDFlash2* draft)
            : runtime_(runtime), enabled_(draft != nullptr) {
            if (enabled_) runtime_.set_hidden_capture_layers(draft->config().target_layers);
        }
        ~DFlashCapture() { if (enabled_) runtime_.set_hidden_capture_layers({}); }
        std::optional<mlx::core::array> take() {
            if (!enabled_) return std::nullopt;
            auto features = runtime_.take_hidden_capture();
            if (!features) throw std::logic_error("DFlash2 target omitted batch features");
            return features;
        }
    private:
        MlxQwen4CausalLm& runtime_;
        bool enabled_;
    };

    static mlx::core::array feature_rows(const mlx::core::array& features, int first, int end) {
        return mlx::core::slice(features, {0, first, 0}, {1, end, features.shape(2)});
    }
    mfq::engine::Generation generate_mtp(std::shared_ptr<Request> request,
        mfq::engine::InferenceOutput& output, std::function<void()> schedule,
        std::function<bool()> ready, std::function<void()> check) {
        std::deque<mfq::engine::OutputDelta> deltas;
        const std::array<std::int64_t, 1> eos{runtime_.config().eos_token_id};
        auto callbacks = runtime_.request_mtp_callbacks(request->cache, true);
        callbacks.draft_depth = [this, request] {
            return select_draft_depth(*request);
        };
        callbacks.batch_cycle_ms = [this](int depth) {
            return policy_ ? policy_->cost(depth) : std::optional<double>{};
        };
        callbacks.batch_park_count = [request] { return request->mtp_parks; };
        const auto prepare = callbacks.prepare_draft;
        callbacks.prepare_draft = [this, request, prepare](const MlxMtpDraftContext& context,
            const MlxMtpTokenSelector& select) {
            request->draft_started = std::chrono::steady_clock::now();
            request->draft_epoch = prefill_epoch_;
            prepare(context, select);
            request->draft_calibration = !context.time_sample || *context.time_sample;
        };
        const auto history = callbacks.prepare_draft_history;
        callbacks.prepare_draft_history = [this, request, history](const MlxMtpDraftContext& context) {
            request->draft_started = std::chrono::steady_clock::now();
            request->draft_epoch = prefill_epoch_;
            history(context);
            request->draft_calibration = !context.time_sample || *context.time_sample;
        };
        callbacks.verify_target = [request](std::int32_t, const mlx::core::array&, int) {
            if (!request->target_result) throw std::logic_error("MTP batch result is not ready");
            auto result = std::move(*request->target_result);
            request->target_result.reset();
            return result;
        };
        callbacks.decode_target = [request](std::int32_t) {
            if (!request->target_result) throw std::logic_error("MTP batch result is not ready");
            auto result = std::move(*request->target_result);
            request->target_result.reset();
            return result;
        };
        const auto resolve = callbacks.resolve_target;
        callbacks.resolve_target = [this, request, resolve](int accepted, int drafts) {
            resolve(accepted, drafts);
            request->cache_length = runtime_.request_cache_position(*request->cache);
        };
        callbacks.draft_limit = [this, request](int position, int depth) {
            if (!request->prefix || !hooks_.block_size) return depth;
            const auto boundary = (position / hooks_.block_size + 1) * hooks_.block_size;
            return static_cast<int>(std::min<std::size_t>(depth, boundary - position - 1));
        };
        const auto commit = callbacks.committed_target;
        callbacks.committed_target = [this, request, commit](const MlxMtpDraftContext& context) {
            commit(context);
            request->cache_length = runtime_.request_cache_position(*request->cache);
            checkpoint(*request, request->cache_length);
            finish_round(*request, context.accepted_drafts);
        };
        auto sequence = mlx_mtp_generation(MlxMtpEngineRequest{
            static_cast<int>(runtime_.config().vocab_size), request->generation_limit,
            static_cast<int>(runtime_.config().max_position_embeddings), *request->initial_logits,
            request->sampler.params(), request->counts,
            runtime_.config().eos_token_id > 0 ? std::span<const std::int64_t>(eos) : std::span<const std::int64_t>{},
            [&](std::int64_t token) {
                request->history.push_back(token);
                ++request->produced;
                deltas.push_back(output.append({token}));
                return !output.stopped();
            }, 0u, request->token_constraint}, std::move(callbacks), request->mtp_stats);
        while (!output.stopped()) {
            auto step = sequence.next();
            if (step.value) {
                request->target_work = std::move(*step.value);
                schedule();
            }
            publish_mtp_metrics(*request, output);
            if (!deltas.empty()) {
                mfq::engine::OutputDelta value;
                while (!deltas.empty()) {
                    auto& part = deltas.front();
                    value.token_ids.insert(value.token_ids.end(), part.token_ids.begin(), part.token_ids.end());
                    for (auto& diff : part.diffs) value.diffs.push_back(std::move(diff));
                    deltas.pop_front();
                }
                co_yield std::move(value);
            } else if (step.value) co_yield mfq::StepState::advanced;
            if (!step || output.stopped()) break;
            while (!ready() && !output.stopped()) co_yield mfq::StepState::waiting;
            if (output.stopped()) break;
            check();
            request->target_work.reset();
        }
    }

    mfq::engine::Generation generate_dflash2(std::shared_ptr<Request> request,
        mfq::engine::InferenceOutput& output, std::function<void()> schedule,
        std::function<bool()> ready, std::function<void()> check) {
        auto& context = *request->dflash_context;
        auto& stats = request->mtp_stats;
        stats.available = true;
        auto pending = static_cast<std::int32_t>(sample_logits(*request, *request->initial_logits));
        request->initial_logits.reset();
        auto delta = output.append({pending});
        request->produced += delta.token_ids.size();
        stats.standard_tokens = request->produced;
        for (;;) {
            const int position = runtime_.request_cache_position(*request->cache);
            if (output.stopped() || request->produced >= request->generation_limit ||
                position >= runtime_.context_capacity()) {
                publish_mtp_metrics(*request, output);
                if (!delta.token_ids.empty()) co_yield std::move(delta);
                co_return;
            }
            if (position != context.position) throw std::logic_error("DFlash2 request cache positions diverged");
            const int depth = std::min({select_draft_depth(*request),
                request->generation_limit - request->produced - 1, runtime_.context_capacity() - position - 1});
            request->draft_started = std::chrono::steady_clock::now();
            request->draft_epoch = prefill_epoch_;
            request->target_work.emplace(MlxMtpTargetRequest{pending, mlx::core::zeros({0}, mlx::core::int32), depth});
            schedule();
            publish_mtp_metrics(*request, output);
            co_yield std::move(delta);
            while (!ready() && !output.stopped()) co_yield mfq::StepState::waiting;
            if (output.stopped()) co_return;
            check();

            auto& verified = *request->target_result;
            auto predictions = verified.greedy_tokens ? *verified.greedy_tokens : request->sampler.sample(verified.logits);
            predictions.eval();
            const auto* target = predictions.data<std::int32_t>();
            const auto* drafts = depth ? request->target_work->draft_tokens.data<std::int32_t>() : nullptr;
            int matched = 0;
            pending = target[0];
            if (depth) {
                const auto decision = mfq::engine::mtp::verify_greedy(
                    {drafts, static_cast<std::size_t>(depth)}, {target, static_cast<std::size_t>(depth + 1)});
                matched = static_cast<int>(decision.accepted_drafts);
                pending = decision.next_token;
            }
            std::vector<std::int64_t> tokens;
            if (matched) tokens.assign(drafts, drafts + matched);
            tokens.push_back(pending);
            delta = output.append(tokens);
            const int retained = std::min(matched, static_cast<int>(delta.token_ids.size()));
            if (depth) runtime_.resolve_request(*request->cache, retained);
            auto features = std::move(verified.hidden);
            request->target_result.reset();
            request->target_work.reset();
            request->cache_length = runtime_.request_cache_position(*request->cache);
            request->produced += delta.token_ids.size();
            if (!output.stopped() && request->produced < request->generation_limit &&
                request->cache_length < runtime_.context_capacity())
                hooks_.dflash2->append_context(context, feature_rows(features, 0, retained + 1), position);

            stats.selected_depth = depth;
            ++stats.depth_cycles[depth];
            if (!depth) ++stats.standard_tokens;
            else {
                stats.used = true;
                ++stats.cycles;
                stats.drafted_tokens += depth;
                stats.accepted_tokens += retained;
                for (int i = 0; i < depth; ++i) ++stats.position_drafted[i];
                for (int i = 0; i < retained; ++i) ++stats.position_accepted[i];
            }
            stats.measured_depth_ms[depth] += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - request->draft_started).count();
            finish_round(*request, retained);
        }
    }

    void draft_dflash2(const std::vector<Request*>& lanes, std::vector<mlx::core::array>& draft_tokens) {
        if (lanes.empty()) return;
        const auto& draft = *hooks_.dflash2;
        const auto& config = draft.config();
        const int count = static_cast<int>(lanes.size());
        std::vector<const MlxDFlash2Context*> contexts;
        std::vector<std::int32_t> anchors;
        std::vector<int> depths;
        std::vector<std::int32_t> ids(lanes.size() * config.block_size, config.mask_token);
        for (int lane = 0; lane < count; ++lane) {
            auto& request = *lanes[lane];
            contexts.push_back(&*request.dflash_context);
            anchors.push_back(request.target_work->pending_token);
            depths.push_back(request.target_work->draft_count);
            ids[lane * config.block_size] = anchors.back();
        }
        auto proposal = draft.propose_batch(contexts, anchors, runtime_.embed_token_ids(
            mlx::core::array(ids.begin(), {count, config.block_size}, mlx::core::int32)),
            [this](const auto& hidden) { return runtime_.project_hidden_logits(hidden); }, depths);
        for (int lane = 0; lane < count; ++lane) {
            auto& work = *lanes[lane]->target_work;
            work.draft_tokens = mlx::core::contiguous(mlx::core::reshape(
                mlx::core::slice(proposal.tokens, {lane, 0}, {lane + 1, work.draft_count}), {work.draft_count}));
            draft_tokens.push_back(work.draft_tokens);
        }
        ++dflash_draft_batches_;
        dflash_draft_rows_ += lanes.size();
        dflash_draft_maximum_ = std::max(dflash_draft_maximum_, lanes.size());
    }

    Decoded decode_mtp(const State& state) {
        std::vector<std::shared_ptr<MlxQwen4RequestCache>> draft_caches;
        std::vector<std::optional<mlx::core::array>> previous;
        for (const auto& request : state.active) if (request->eligible && request->mtp &&
            request->target_work->kind != MlxMtpTargetRequest::Kind::target) {
            draft_caches.push_back(request->cache);
            previous.push_back(request->target_work->kind == MlxMtpTargetRequest::Kind::first_draft ? std::optional<mlx::core::array>{}
                : request->target_work->draft_tokens);
        }
        if (!draft_caches.empty()) {
            runtime_.draft_request_heads(draft_caches, previous);
            ++mtp_draft_batches_;
            mtp_draft_rows_ += draft_caches.size();
            mtp_draft_maximum_ = std::max(mtp_draft_maximum_, draft_caches.size());
            // A lane with a shorter draft waits for the other native heads so
            // their target rows still share one backbone pass.
            return {mlx::core::zeros({0, 1, static_cast<int>(runtime_.config().vocab_size)}), {}, {}, true};
        }
        State cohort;
        std::vector<Request*> dflash_lanes;
        std::vector<mlx::core::array> drafts;
        bool capture_dflash = false;
        for (const auto& request : state.active) if (request->eligible &&
            (!request->mtp || request->target_work->kind == MlxMtpTargetRequest::Kind::target)) {
            cohort.active.push_back(request);
            capture_dflash |= request->dflash_context.has_value();
            if (request->mtp && request->target_work->draft_count) {
                if (request->dflash_context) dflash_lanes.push_back(request.get());
                else drafts.push_back(request->target_work->draft_tokens);
            }
        }
        if (!cohort.active.empty()) park_memory_.tick();
        draft_dflash2(dflash_lanes, drafts);
        if (!drafts.empty()) mlx::core::eval(drafts);
        Decoded result{mlx::core::zeros({0, 1, static_cast<int>(runtime_.config().vocab_size)}), {}, {}};
        if (!cohort.active.empty()) {
            std::vector<std::shared_ptr<MlxQwen4RequestCache>> caches;
            std::vector<std::vector<std::int64_t>> windows;
            int verify_rows = 0;
            int maximum_drafts = 0;
            for (const auto& request : cohort.active) {
                const int depth = request->mtp ? request->target_work->draft_count : 0;
                verify_rows += depth;
                maximum_drafts = std::max(maximum_drafts, depth);
                caches.push_back(request->cache);
                const auto token = request->mtp ? request->target_work->pending_token : request->pending_token;
                windows.push_back({token});
                if (depth) {
                    const auto* ids = request->target_work->draft_tokens.data<std::int32_t>();
                    windows.back().insert(windows.back().end(), ids, ids + depth);
                }
            }
            const bool joint_greedy = caches.size() > 1 &&
                std::all_of(cohort.active.begin(), cohort.active.end(), [](const auto& request) {
                    return request->sampler.params().greedy() && !request->counts;
                });
            std::optional<mlx::core::array> packed_logits, joint_tokens;
            DFlashCapture capture(runtime_, capture_dflash ? hooks_.dflash2.get() : nullptr);
            auto target = runtime_.forward_request_windows(caches, windows,
                joint_greedy ? &packed_logits : nullptr);
            auto features = capture.take();
            if (packed_logits) joint_tokens = sample_greedy(mlx::core::reshape(*packed_logits,
                {packed_logits->shape(1), packed_logits->shape(2)}));
            auto round = std::make_shared<MlxMtpBatchRound>();
            round->policy = policy_;
            round->expected = std::count_if(cohort.active.begin(), cohort.active.end(),
                [](const auto& request) { return request->mtp; });
            round->prefill_epoch = prefill_epoch_;
            round->depth = 0;
            round->maximum_drafts = maximum_drafts;
            for (const auto& request : cohort.active) if (request->mtp)
                round->depth = std::max(round->depth, request->scheduled_budget);
            if (policy_) round->depth = std::min(round->depth, policy_->maximum());
            round->stable = policy_ && policy_->requests().size() == round->expected &&
                policy_->ordinary_requests() == cohort.active.size() - round->expected;
            round->started = std::chrono::steady_clock::now();
            for (const auto& request : cohort.active) if (request->mtp) {
                request->round_depth = request->target_work->draft_count;
                round->started = std::min(round->started, request->draft_started);
                round->prefill_epoch = std::min(round->prefill_epoch, request->draft_epoch);
                round->stable = round->stable && request->draft_calibration &&
                    request->scheduled_budget == round->depth && request->scheduled_rows == verify_rows;
                request->round = round;
            }
            ++decode_batches_;
            decode_rows_ += caches.size();
            maximum_batch_ = std::max(maximum_batch_, caches.size());
            if (round->expected) {
                ++mtp_batches_;
                mtp_rows_ += round->expected;
                mtp_maximum_batch_ = std::max(mtp_maximum_batch_, round->expected);
                mtp_verify_rows_ += verify_rows;
                mtp_ragged_batches_ += verify_rows != static_cast<int>(caches.size()) * maximum_drafts;
            }
            std::vector<mlx::core::array> selected;
            std::vector<mlx::core::array> first_logits;
            std::vector<int> selected_rows;
            std::vector<mlx::core::array> greedy_tokens;
            int selection_offset = 0;
            for (int row = 0; row < static_cast<int>(caches.size()); ++row) {
                const auto& request = cohort.active[row];
                const int depth = request->mtp ? request->target_work->draft_count : 0;
                const int first_selection = selection_offset;
                selection_offset += depth + 1;
                auto logits = target[row].first;
                first_logits.push_back(mlx::core::slice(logits, {0, 0, 0}, {1, 1, logits.shape(2)}));
                if (!request->mtp) {
                    checkpoint(*request, runtime_.request_cache_position(*request->cache));
                    auto value = mlx_last_token_logits(logits, static_cast<int>(runtime_.config().vocab_size));
                    selected.push_back(joint_tokens ? mlx::core::slice(*joint_tokens,
                        {first_selection}, {selection_offset}) : request->counts
                        ? request->sampler.sample(value, *request->counts) : request->sampler.sample(value));
                    selected_rows.push_back(row);
                    result.rows.emplace(request->id, row);
                    continue;
                }
                auto verify_logits = mlx::core::reshape(logits, {depth + 1, logits.shape(2)});
                std::optional<mlx::core::array> prepared;
                if (caches.size() > 1 && request->sampler.params().greedy() && !request->counts) {
                    prepared = joint_tokens ? mlx::core::slice(*joint_tokens,
                        {first_selection}, {selection_offset}) : sample_greedy(verify_logits);
                    greedy_tokens.push_back(*prepared);
                }
                auto hidden = request->dflash_context ? feature_rows(*features, first_selection, selection_offset) :
                    std::move(target[row].second);
                request->target_result.emplace(MlxMtpTargetBatch{
                    std::move(verify_logits), std::move(hidden), std::move(prepared)});
            }
            if (!greedy_tokens.empty()) {
                greedy_tokens.insert(greedy_tokens.end(), selected.begin(), selected.end());
                mlx::core::eval(greedy_tokens);
            }
            if (!selected.empty()) {
                if (greedy_tokens.empty()) mlx::core::eval(selected);
                result.logits = mlx::core::concatenate(first_logits, 0);
                result.sampled_tokens.resize(caches.size());
                for (std::size_t row = 0; row < selected.size(); ++row)
                    result.sampled_tokens[selected_rows[row]] = selected[row].data<std::int32_t>()[0];
            }
        }
        return result;
    }
    std::int64_t sample_logits(Request& request, const mlx::core::array& logits,
        std::optional<std::int64_t> prepared = std::nullopt) {
        auto token = prepared.value_or(0);
        if (!prepared) {
            auto selected = request.counts ? request.sampler.sample(logits, *request.counts)
                : request.sampler.sample(logits);
            selected.eval();
            token = selected.data<std::int32_t>()[0];
        }
        const auto vocab = static_cast<int>(runtime_.config().vocab_size);
        if (request.token_constraint && !request.token_constraint->allows(token)) {
            auto adjusted = request.counts ? request.sampler.apply_penalties(logits, *request.counts) : logits;
            adjusted = mlx::core::contiguous(mlx::core::astype(adjusted, mlx::core::float32));
            adjusted.eval();
            std::vector<float> masked(adjusted.data<float>(), adjusted.data<float>() + vocab);
            request.token_constraint->apply(masked.data(), masked.size());
            auto selected = request.sampler.sample(mlx::core::array(masked.begin(), {1, vocab}, mlx::core::float32));
            selected.eval();
            token = selected.data<std::int32_t>()[0];
            if (!request.token_constraint->allows(token)) throw std::runtime_error("invalid constrained batch token");
        }
        if (token < 0 || token >= vocab) throw std::runtime_error("invalid batch token");
        return token;
    }
    void checkpoint(Request& request, std::size_t position, bool terminal = false) {
        if (!hooks_.store || !request.prefix || !request.cache || !position ||
            position > request.history.size() || position == request.checkpoint ||
            (!request.cache_plan.cache_output_tokens && position > request.prefix) ||
            (!terminal && position != request.prefix &&
                (!hooks_.block_size || position % hooks_.block_size))) return;
        try {
            std::vector<std::int64_t> tokens(request.history.begin(), request.history.begin() + position);
            hooks_.store(request, runtime_.capture_request_cache(*request.cache, tokens));
            request.checkpoint = position;
        } catch (const std::exception& error) {
            std::cerr << "server_session_cache backend=metal action=skip error=" << error.what() << '\n';
        }
    }
    MlxQwen4CausalLm& runtime_;
    Hooks hooks_;
    std::size_t decode_batches_ = 0, decode_rows_ = 0, maximum_batch_ = 0;
    std::size_t mtp_batches_ = 0, mtp_rows_ = 0, mtp_maximum_batch_ = 0;
    std::size_t mtp_verify_rows_ = 0, mtp_ragged_batches_ = 0;
    std::size_t mtp_draft_batches_ = 0, mtp_draft_rows_ = 0, mtp_draft_maximum_ = 0;
    std::size_t dflash_draft_batches_ = 0, dflash_draft_rows_ = 0, dflash_draft_maximum_ = 0;
    std::size_t prefill_batches_ = 0, prefill_rows_ = 0, prefill_maximum_ = 0;
    double prefill_compute_ms_ = 0, prefill_checkpoint_ms_ = 0, prefill_maximum_ms_ = 0;
    std::array<std::uint64_t, mfq::engine::mtp::kMaximumReportedDraftDepth + 1> mtp_cycle_counts_{};
    std::array<double, mfq::engine::mtp::kMaximumReportedDraftDepth + 1> mtp_cycle_totals_{}, mtp_cycle_maxima_{};
    std::shared_ptr<mfq::engine::mtp::BatchDepthPolicy> policy_;
    mfq::engine::mtp::BatchParkMemory park_memory_;
    std::uint64_t prefill_epoch_ = 0;
    std::vector<std::shared_ptr<MlxMtpBatchRound>> completed_rounds_;
    std::vector<std::string> ordinary_cohort_;
};

} // namespace mfq::metal

#pragma once

#include "mlx_request_batch.h"
#include "mlx_mtp.h"
#include "mlx_sampling.h"
#include "tyloinfer/token_constraint.h"

#include <concepts>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <span>
#include <unordered_map>

namespace mfq::metal {

template<class Runtime>
concept MlxBatchTextRuntime = requires(Runtime& model, typename Runtime::BatchCache& cache,
    const mlx::core::array& ids, std::span<typename Runtime::BatchCache* const> caches,
    const std::vector<std::int64_t>& tokens) {
    { model.batch_vocabulary() } -> std::convertible_to<int>;
    { model.make_batch_cache() } -> std::same_as<typename Runtime::BatchCache>;
    { model.take_batch_cache() } -> std::same_as<typename Runtime::BatchCache>;
    { model.prefill_batch_request(cache, ids) } -> std::same_as<mlx::core::array>;
    { model.decode_batch_requests(caches, ids) } -> std::same_as<mlx::core::array>;
    model.capture_batch_cache(cache, tokens);
    { model.batch_cache_position(cache) } -> std::convertible_to<int>;
};

template<class Runtime>
concept MlxBatchSpeculativeRuntime = MlxBatchTextRuntime<Runtime> &&
    requires(Runtime& model, typename Runtime::BatchCache& cache,
        std::span<typename Runtime::BatchCache* const> caches, std::span<const MlxMtpTargetRequest> steps,
        MlxMtpGenerationStats stats) {
        { model.supports_batch_mtp() } -> std::convertible_to<bool>;
        { model.batch_context() } -> std::convertible_to<int>;
        { model.batch_mtp_callbacks(cache) } -> std::same_as<MlxMtpEngineCallbacks>;
        { model.verify_batch_requests(caches, steps) } -> std::same_as<std::vector<MlxMtpTargetBatch>>;
        model.discard_batch_verification(cache);
        model.record_mtp_stats(std::move(stats));
    };

template<class Runtime>
concept MlxBatchDraftRuntime = MlxBatchSpeculativeRuntime<Runtime> &&
    requires(Runtime& model, std::span<typename Runtime::BatchCache* const> caches,
        std::span<const MlxMtpTargetRequest> steps) {
        { model.draft_batch_requests(caches, steps) } -> std::same_as<std::vector<mlx::core::array>>;
    };

template<MlxBatchTextRuntime Runtime, class SessionCache>
class MlxTextBatchOperations {
    using Cache = typename Runtime::BatchCache;
    using DepthPolicy = mfq::engine::mtp::BatchDepthPolicy;
    using Clock = DepthPolicy::Clock;

    struct DepthRound {
        std::shared_ptr<DepthPolicy> policy;
        Clock::time_point started, finished;
        std::vector<mfq::engine::mtp::BatchAcceptance> acceptance;
        std::size_t expected = 0;
        std::uint64_t prefill_epoch = 0;
        int depth = 0;
        bool calibrate = true;
    };

public:
    struct Request : mfq::engine::BatchRequest {
        Request(std::string id, const mfq::engine::InferenceRequest& input)
            : BatchRequest(std::move(id), input), cache_plan(input.cache_plan),
              sampler(mlx_batch_sampling(input.sampling)), history(prompt) {}

        MfqPromptCachePlan cache_plan;
        MlxSampler sampler;
        std::vector<std::int64_t> history;
        std::optional<mlx::core::array> counts;
        std::optional<Cache> cache;
        bool speculative = false;
        bool draft_batchable = false;
        std::optional<mlx::core::array> first_logits;
        std::optional<MlxMtpTargetRequest> target_work;
        std::optional<MlxMtpTargetBatch> target_result;
        std::optional<mlx::core::array> draft_result;
        MlxMtpGenerationStats mtp_stats;
        int draft_cap = 0;
        std::shared_ptr<DepthPolicy> draft_policy;
        Clock::time_point draft_started;
        int draft_budget = 0;
        std::uint64_t draft_epoch = 0;
        bool draft_calibration = false;
        std::shared_ptr<DepthRound> depth_round;
        int verified_drafts = 0;
        std::size_t prefix = 0;
        std::size_t checkpoint = 0;
    };

    using State = mfq::engine::BatchState<Request>;
    struct Sample {
        std::int64_t token = 0;
        MfqPrefillTiming timing;
    };
    using Decoded = std::unordered_map<std::string, std::int64_t>;

    MlxTextBatchOperations(Runtime& runtime, SessionCache& sessions)
        : runtime_(runtime), sessions_(sessions), batch_depth_enabled_([] {
            const auto* value = std::getenv("TYLOINFER_METAL_TEXT_BATCH_MTP_POLICY");
            return MlxBatchSpeculativeRuntime<Runtime> && value && std::string_view(value) == "batch";
        }()), batch_drafts_enabled_([] {
            const auto* value = std::getenv("TYLOINFER_METAL_BATCH_DRAFT");
            return MlxBatchDraftRuntime<Runtime> && (!value || std::string_view(value) != "0");
        }()) {}

    std::int64_t prepare(const std::shared_ptr<Request>& request) {
        using namespace mlx::core;
        request->cache.emplace(runtime_.make_batch_cache());
        if constexpr (MlxBatchSpeculativeRuntime<Runtime>) {
            request->mtp_stats.available = runtime_.supports_batch_mtp();
            request->speculative = request->mtp_stats.available && request->sampling.enable_mtp &&
                request->generation_limit > 1;
        }
        request->prefix = sessions_.normalize_stable_prefix_tokens(
            std::min(request->cache_plan.stable_prefix_tokens, request->prompt.size()));
        if (request->cache_plan.session_id.empty() && !sessions_.persistent_prefix_enabled())
            request->prefix = 0;
        std::size_t restored = 0;
        constexpr bool restores_draft_state = requires { runtime_.enable_batch_mtp(*request->cache); };
        if (request->prefix) {
            runtime_.clear_cache();
            restored = sessions_.restore_best(runtime_, request->cache_plan.session_id, request->prompt,
                std::min(request->prefix, request->prompt.size() - 1), request->speculative && restores_draft_state);
            if (restored) request->cache.emplace(runtime_.take_batch_cache());
        }
        if constexpr (restores_draft_state)
            if (request->speculative) runtime_.enable_batch_mtp(*request->cache);
        if (request->sampler.params().has_penalties()) {
            request->counts = sample_token_counts_add(
                zeros({runtime_.batch_vocabulary()}, int32),
                array(request->prompt.begin(), {1, static_cast<int>(request->prompt.size())}, int32));
        }
        return restored;
    }

    std::int64_t prefix_tokens(const std::shared_ptr<Request>& request) const { return request->prefix; }

    mfq::engine::PrefillChunk prefill_chunk(const std::shared_ptr<Request>& request,
        mfq::engine::PrefillChunk chunk) const {
        if (request->prefix) {
            const auto block = static_cast<std::int64_t>(sessions_.block_size_tokens());
            const auto end = std::min(chunk.offset + chunk.count, (chunk.offset / block + 1) * block);
            chunk.count = end - chunk.offset;
            chunk.last = end == static_cast<std::int64_t>(request->prompt.size());
        }
        return chunk;
    }

    Sample prefill(const std::shared_ptr<Request>& request, mfq::engine::PrefillChunk chunk) {
        using namespace mlx::core;
        const auto start = std::chrono::steady_clock::now();
        const auto input = array(request->prompt.begin() + chunk.offset, {1, static_cast<int>(chunk.count)}, int32);
        auto logits = runtime_.prefill_batch_request(*request->cache, input);
        logits.eval();
        checkpoint(*request, chunk.offset + chunk.count);
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        Sample sample{0, {static_cast<std::size_t>(chunk.count), elapsed, 0, elapsed}};
        if (chunk.last) {
            if (request->speculative) request->first_logits = std::move(logits);
            else sample.token = checked_token(*request, raw_sample(*request, logits).template item<std::int32_t>());
        }
        return sample;
    }

    void suspend_decode() {
        if (depth_policy_) {
            ++prefill_epoch_;
            depth_policy_->interrupt();
        }
    }
    void resume_decode(std::int64_t) {}

    void synchronize(const State& state) {
        if (batch_drafts_enabled_) {
            draft_participants_ = std::count_if(state.active.begin(), state.active.end(),
                [](const auto& request) {
                    return request->draft_batchable && request->produced < request->generation_limit - 1;
                });
        }
        if (!batch_depth_enabled_) return;
        std::vector<std::string> speculative, ordinary;
        int maximum = 0;
        bool all_caps_known = true;
        for (const auto& request : state.active) {
            if (request->produced >= request->generation_limit) continue;
            if (request->speculative) {
                speculative.push_back(request->id);
                maximum = std::max(maximum, request->draft_cap);
                all_caps_known = all_caps_known && request->draft_cap > 0;
            } else ordinary.push_back(request->id);
        }
        std::sort(speculative.begin(), speculative.end());
        std::sort(ordinary.begin(), ordinary.end());
        const bool paired = all_caps_known && !speculative.empty() && speculative.size() + ordinary.size() > 1;
        if (!paired || !depth_policy_ || depth_policy_->requests() != speculative ||
                depth_policy_->maximum() != maximum || ordinary_cohort_ != ordinary) {
            if (depth_policy_) park_memory_.retired(*depth_policy_);
            depth_policy_.reset();
            if (paired) {
                depth_policy_ = std::make_shared<DepthPolicy>(std::move(speculative), maximum, ordinary.size());
                park_memory_.seed(*depth_policy_);
            }
            ordinary_cohort_ = std::move(ordinary);
        }
        for (const auto& round : completed_rounds_) if (round->policy == depth_policy_) {
            const auto parks = depth_policy_->parks();
            const bool was_parked = depth_policy_->remaining() != 0;
            const bool interrupted = round->prefill_epoch != prefill_epoch_;
            depth_policy_->observe(round->acceptance, round->started, round->finished,
                round->calibrate, interrupted, round->depth);
            if (round->calibrate && !interrupted) ++depth_samples_;
            if (depth_policy_->parks() > parks) park_memory_.parked(*depth_policy_);
            if (was_parked && !depth_policy_->remaining()) park_memory_.resumed(*depth_policy_);
        }
        completed_rounds_.clear();
    }

    void invalidate_runtime_policy() {
        depth_policy_.reset();
        ordinary_cohort_.clear();
        completed_rounds_.clear();
        park_memory_ = {};
    }

    void activate(const std::shared_ptr<Request>& request, const Sample& result) {
        if (!request->speculative) accept(request, result);
    }

    std::int64_t cache_length(const std::shared_ptr<Request>& request) const {
        return runtime_.batch_cache_position(*request->cache);
    }

    std::optional<mfq::engine::Generation> generate_decode(const std::shared_ptr<Request>& request,
        mfq::engine::InferenceOutput& output, std::function<void()> schedule,
        std::function<bool()> ready, std::function<void()> check) {
        if constexpr (MlxBatchSpeculativeRuntime<Runtime>) {
            if (request->speculative)
                return generate_mtp(request, output, std::move(schedule), std::move(ready), std::move(check));
        }
        return std::nullopt;
    }

    void accept(const std::shared_ptr<Request>& request, const Sample& result) {
        using namespace mlx::core;
        if (request->token_constraint) request->token_constraint->accept(result.token);
        if (request->counts) *request->counts = sample_token_counts_add(*request->counts,
            array({result.token}, {1, 1}, int32));
        request->history.push_back(result.token);
    }

    Decoded decode(const State& state) {
        using namespace mlx::core;
        std::vector<Request*> eligible;
        std::vector<Cache*> caches;
        std::vector<std::int64_t> tokens;
        for (const auto& request : state.active) if (request->eligible) {
            eligible.push_back(request.get());
            caches.push_back(&*request->cache);
            tokens.push_back(request->pending_token);
        }
        if constexpr (MlxBatchDraftRuntime<Runtime>) {
            if (batch_drafts_enabled_ && draft_windows(eligible)) return {};
        }
        const auto rows = static_cast<int>(eligible.size());
        ++decode_batches_;
        if (batch_depth_enabled_) park_memory_.tick();
        decode_rows_ += rows;
        maximum_batch_ = std::max(maximum_batch_, eligible.size());
        std::vector<array> sources;
        std::vector<Request*> ordinary;
        sources.reserve(rows);
        ordinary.reserve(rows);
        bool has_verification = false;

        if constexpr (MlxBatchSpeculativeRuntime<Runtime>) {
            has_verification = std::any_of(eligible.begin(), eligible.end(),
                [](const auto* request) { return request->speculative; });
            if (has_verification) {
                if (depth_policy_) track_depth_round(eligible);
                std::vector<MlxMtpTargetRequest> steps;
                steps.reserve(rows);
                for (const auto* request : eligible) {
                    if (request->speculative) steps.push_back(*request->target_work);
                    else steps.push_back({static_cast<std::int32_t>(request->pending_token), zeros({0}, int32), 0});
                }
                auto results = runtime_.verify_batch_requests(caches, steps);
                for (int row = 0; row < rows; ++row) {
                    auto* request = eligible[row];
                    if (request->speculative) request->target_result = std::move(results[row]);
                    else {
                        ordinary.push_back(request);
                        sources.push_back(std::move(results[row].logits));
                    }
                }
                ++mtp_batches_;
                mtp_maximum_batch_ = std::max(mtp_maximum_batch_, eligible.size());
            }
        }
        if (!has_verification) {
            const auto logits = reshape(runtime_.decode_batch_requests(caches,
                array(tokens.begin(), {rows, 1}, int32)), {rows, runtime_.batch_vocabulary()});
            ordinary = eligible;
            for (int row = 0; row < rows; ++row)
                sources.push_back(slice(logits, {row, 0}, {row + 1, logits.shape(1)}));
        }

        std::vector<array> samples;
        samples.reserve(ordinary.size());
        for (std::size_t row = 0; row < ordinary.size(); ++row)
            samples.push_back(raw_sample(*ordinary[row], sources[row]));
        if (!samples.empty()) eval(samples);
        Decoded result;
        for (std::size_t row = 0; row < ordinary.size(); ++row) {
            auto& request = *ordinary[row];
            checkpoint(request, runtime_.batch_cache_position(*request.cache));
            result.emplace(request.id, checked_token(request, samples[row].data<std::int32_t>()[0]));
        }
        return result;
    }

    Sample sample(const std::shared_ptr<Request>& request, const Decoded& result) const {
        return request->speculative ? Sample{} : Sample{result.at(request->id), {}};
    }

    bool decode_completed(const std::shared_ptr<Request>& request, const Decoded& result) const
        requires MlxBatchDraftRuntime<Runtime> {
        if (!batch_drafts_enabled_) return true;
        return request->speculative ? request->target_result.has_value() || request->draft_result.has_value()
            : result.contains(request->id);
    }

    void discard_prefill(const std::shared_ptr<Request>& request) { request->cache.reset(); }
    void retire(const std::vector<std::shared_ptr<Request>>& requests, std::int64_t) {
        for (const auto& request : requests) {
            if constexpr (MlxBatchSpeculativeRuntime<Runtime>) {
                if (request->depth_round && request->depth_round->policy == depth_policy_)
                    depth_policy_->interrupt();
                if (request->target_result) runtime_.discard_batch_verification(*request->cache);
                runtime_.record_mtp_stats(std::move(request->mtp_stats));
            }
            if (!request->target_result)
                checkpoint(*request, runtime_.batch_cache_position(*request->cache), true);
            request->cache.reset();
        }
    }

    void recover(const State& state) {
        invalidate_runtime_policy();
        for (const auto& request : state.active) request->cache.reset();
        for (const auto& request : state.prefilling) request->cache.reset();
        runtime_.clear_cache();
    }

    mfq::engine::Metrics metrics() const {
        return {{"continuous_batching_decode_batches", double(decode_batches_)},
            {"continuous_batching_decode_rows", double(decode_rows_)},
            {"continuous_batching_maximum_batch", double(maximum_batch_)},
            {"continuous_batching_mtp_batches", double(mtp_batches_)},
            {"continuous_batching_mtp_maximum_batch", double(mtp_maximum_batch_)},
            {"continuous_batching_mtp_draft_batches", double(draft_batches_)},
            {"continuous_batching_mtp_draft_maximum_batch", double(draft_maximum_batch_)},
            {"continuous_batching_mtp_policy_samples", double(depth_samples_)}};
    }

private:
    bool draft_windows(std::span<Request* const> eligible) requires MlxBatchDraftRuntime<Runtime> {
        std::vector<Request*> lanes;
        std::vector<Cache*> caches;
        std::vector<MlxMtpTargetRequest> steps;
        for (auto* request : eligible) {
            if (!request->target_work || request->target_work->kind != MlxMtpTargetRequest::Kind::draft_window)
                continue;
            lanes.push_back(request);
            caches.push_back(&*request->cache);
            steps.push_back(*request->target_work);
        }
        if (lanes.empty()) return false;
        auto proposals = runtime_.draft_batch_requests(caches, steps);
        if (proposals.size() != lanes.size()) throw std::logic_error("invalid Metal draft batch size");
        for (std::size_t lane = 0; lane < lanes.size(); ++lane)
            lanes[lane]->draft_result = std::move(proposals[lane]);
        ++draft_batches_;
        draft_maximum_batch_ = std::max(draft_maximum_batch_, lanes.size());
        // Wait for each proposal before scheduling a joint target pass.
        return true;
    }

    void track_depth_round(std::span<Request* const> eligible) {
        auto round = std::make_shared<DepthRound>();
        round->policy = depth_policy_;
        round->expected = depth_policy_->requests().size();
        round->prefill_epoch = prefill_epoch_;
        round->started = Clock::now();
        if (eligible.size() != round->expected + depth_policy_->ordinary_requests()) {
            depth_policy_->interrupt();
            return;
        }
        std::optional<int> budget;
        for (auto* request : eligible) {
            if (!request->speculative) continue;
            if (request->draft_policy != depth_policy_ || (budget && *budget != request->draft_budget)) {
                depth_policy_->interrupt();
                return;
            }
            budget = request->draft_budget;
            round->started = std::min(round->started, request->draft_started);
            round->prefill_epoch = std::min(round->prefill_epoch, request->draft_epoch);
            round->calibrate = round->calibrate && request->draft_calibration &&
                request->target_work->draft_count == std::min(*budget, request->draft_cap);
        }
        round->depth = *budget;
        round->acceptance.reserve(round->expected);
        for (auto* request : eligible) if (request->speculative) {
            request->depth_round = round;
            request->verified_drafts = request->target_work->draft_count;
        }
    }

    mlx::core::array raw_sample(Request& request, const mlx::core::array& logits) {
        using namespace mlx::core;
        array values = request.counts ? request.sampler.apply_penalties(logits, *request.counts) : logits;
        if (request.token_constraint) {
            values = contiguous(astype(values, float32));
            values.eval();
            const auto vocab = runtime_.batch_vocabulary();
            std::vector<float> masked(values.data<float>(), values.data<float>() + vocab);
            request.token_constraint->apply(masked.data(), masked.size());
            values = array(masked.begin(), {1, vocab}, float32);
        }
        return request.sampler.sample(values);
    }

    int64_t checked_token(const Request& request, int64_t token) const {
        if (token < 0 || token >= runtime_.batch_vocabulary() ||
            (request.token_constraint && !request.token_constraint->allows(token)))
            throw std::runtime_error("invalid Metal text batch token");
        return token;
    }

    mfq::engine::Generation generate_mtp(std::shared_ptr<Request> request,
        mfq::engine::InferenceOutput& output, std::function<void()> schedule,
        std::function<bool()> ready, std::function<void()> check)
        requires MlxBatchSpeculativeRuntime<Runtime> {
        std::deque<mfq::engine::OutputDelta> deltas;
        auto callbacks = runtime_.batch_mtp_callbacks(*request->cache);
        if constexpr (MlxBatchDraftRuntime<Runtime>) {
            const auto& params = request->sampler.params();
            if (batch_drafts_enabled_ && params.greedy() && !params.has_penalties() && !params.warmup &&
                    !callbacks.prepare_draft_history) {
                request->draft_batchable = true;
                callbacks.can_batch_draft = [this] { return draft_participants_ > 1; };
                callbacks.batched_draft_tokens = [request] {
                    if (!request->draft_result) throw std::logic_error("Metal batch draft is not ready");
                    auto result = std::move(*request->draft_result);
                    request->draft_result.reset();
                    return result;
                };
            }
        }
        if (batch_depth_enabled_ && !request->sampler.params().warmup &&
                !callbacks.predictor.dspark && !callbacks.prepare_draft_history && !callbacks.draft_depth) {
            request->draft_cap = mlx_mtp_maximum_depth(request->sampler.params(), callbacks.predictor.maximum_depth);
            callbacks.draft_depth = [this, request]() -> std::optional<int> {
                request->draft_policy.reset();
                if (!depth_policy_ || !std::binary_search(depth_policy_->requests().begin(),
                        depth_policy_->requests().end(), request->id)) return std::nullopt;
                request->draft_policy = depth_policy_;
                request->draft_budget = depth_policy_->depth();
                return std::min(request->draft_budget, request->draft_cap);
            };
            callbacks.batch_cycle_ms = [request](int depth) {
                return request->draft_policy->cost(depth);
            };
            callbacks.batch_park_count = [request] { return request->draft_policy->parks(); };
            const auto prepare = std::move(callbacks.prepare_draft);
            callbacks.prepare_draft = [this, request, prepare](const MlxMtpDraftContext& context,
                const MlxMtpTokenSelector& select) {
                request->draft_started = Clock::now();
                request->draft_epoch = prefill_epoch_;
                prepare(context, select);
                request->draft_calibration = !context.time_sample || *context.time_sample;
            };
        }
        callbacks.verify_target = [request](std::int32_t, const mlx::core::array&, int) {
            if (!request->target_result) throw std::logic_error("Metal batch verification is not ready");
            auto result = std::move(*request->target_result);
            request->target_result.reset();
            return result;
        };
        callbacks.draft_limit = [this, request](int position, int depth) {
            if (!request->prefix) return depth;
            const auto size = sessions_.block_size_tokens();
            const auto next = (position / size + 1) * size;
            return static_cast<int>(std::min<std::size_t>(depth, next - position - 1));
        };
        const auto commit = std::move(callbacks.committed_target);
        callbacks.committed_target = [this, request, commit](const MlxMtpDraftContext& context) {
            if (commit) commit(context);
            request->cache_length = runtime_.batch_cache_position(*request->cache);
            checkpoint(*request, request->cache_length);
            if (auto round = std::exchange(request->depth_round, {})) {
                round->acceptance.push_back({request->verified_drafts, context.accepted_drafts});
                if (round->acceptance.size() == round->expected) {
                    round->finished = Clock::now();
                    completed_rounds_.push_back(std::move(round));
                }
            }
        };
        auto initial = std::move(*request->first_logits);
        request->first_logits.reset();
        auto sequence = mlx_mtp_generation(MlxMtpEngineRequest{
            runtime_.batch_vocabulary(), request->generation_limit, runtime_.batch_context(), std::move(initial),
            request->sampler.params(), request->counts, {},
            [&](std::int64_t token) {
                request->history.push_back(token);
                ++request->produced;
                deltas.push_back(output.append({token}));
                return !output.stopped();
            }, 0u, request->token_constraint}, std::move(callbacks), request->mtp_stats);
        while (!output.stopped()) {
            auto step = sequence.next();
            if (step.value) {
                if (step.value->kind == MlxMtpTargetRequest::Kind::draft_window && request->draft_policy) {
                    request->draft_started = Clock::now();
                    request->draft_epoch = prefill_epoch_;
                    request->draft_calibration = true;
                }
                request->target_work = std::move(*step.value);
                schedule();
            }
            const auto& stats = request->mtp_stats;
            output.metrics.mtp = {stats.available, stats.used, stats.cycles, stats.drafted_tokens,
                stats.accepted_tokens, stats.depth_cycles, stats.position_drafted, stats.position_accepted,
                stats.measured_depth_ms, stats.selected_depth, stats.standard_tokens, stats.park_count,
                stats.reentry_probes};
            if (!deltas.empty()) {
                mfq::engine::OutputDelta delta;
                while (!deltas.empty()) {
                    auto& part = deltas.front();
                    delta.token_ids.insert(delta.token_ids.end(), part.token_ids.begin(), part.token_ids.end());
                    for (auto& diff : part.diffs) delta.diffs.push_back(std::move(diff));
                    deltas.pop_front();
                }
                co_yield std::move(delta);
            } else if (step.value) co_yield mfq::StepState::advanced;
            if (!step || output.stopped()) break;
            while (!ready() && !output.stopped()) co_yield mfq::StepState::waiting;
            if (output.stopped()) break;
            check();
            request->target_work.reset();
        }
    }

    void checkpoint(Request& request, std::size_t position, bool terminal = false) {
        if (!request.prefix || !request.cache || !position || position > request.history.size() ||
            position == request.checkpoint || (!request.cache_plan.cache_output_tokens && position > request.prefix) ||
            (!terminal && position != request.prefix && position % sessions_.block_size_tokens())) return;
        try {
            auto state = runtime_.capture_batch_cache(*request.cache,
                {request.history.begin(), request.history.begin() + position});
            sessions_.store(request.cache_plan.session_id, std::move(state));
            request.checkpoint = position;
        } catch (const std::exception& error) {
            std::cerr << "server_session_cache backend=metal action=skip error=" << error.what() << '\n';
        }
    }

    Runtime& runtime_;
    SessionCache& sessions_;
    const bool batch_depth_enabled_;
    const bool batch_drafts_enabled_;
    std::shared_ptr<DepthPolicy> depth_policy_;
    mfq::engine::mtp::BatchParkMemory park_memory_;
    std::vector<std::string> ordinary_cohort_;
    std::vector<std::shared_ptr<DepthRound>> completed_rounds_;
    std::uint64_t prefill_epoch_ = 0, depth_samples_ = 0;
    std::size_t decode_batches_ = 0, decode_rows_ = 0, maximum_batch_ = 0;
    std::size_t mtp_batches_ = 0, mtp_maximum_batch_ = 0;
    std::size_t draft_batches_ = 0, draft_maximum_batch_ = 0, draft_participants_ = 0;
};

class MlxDFlash2;

template<MlxBatchTextRuntime Runtime, class SessionCache>
std::unique_ptr<MlxRequestBatch> make_mlx_model_batch(Runtime& runtime, SessionCache& sessions,
    int chunk_size, bool mtp_available, const std::shared_ptr<MlxDFlash2>& draft) {
    const bool batch_mtp = [&] {
        if constexpr (MlxBatchSpeculativeRuntime<Runtime>) return runtime.supports_batch_mtp();
        else return false;
    }();
    auto admit = [speculative = (mtp_available && !batch_mtp) || bool(draft)](const mfq::engine::EngineRequest& request) {
        return !(request.input.sampling.enable_mtp && speculative);
    };
    return make_mlx_request_batch<MlxTextBatchOperations<Runtime, SessionCache>>(
        chunk_size, chunk_size, admit, runtime, sessions);
}
} // namespace mfq::metal

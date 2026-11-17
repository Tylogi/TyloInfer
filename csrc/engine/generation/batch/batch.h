#pragma once
#include "../step.h"
#include "../prefill/activity.h"
#include "../../requests/policy.h"
#include <deque>
#include <functional>
#include <memory>
#include <span>
#include <unordered_map>

namespace tyloi::engine {

class Budget;

class Batch {
  public:
    virtual ~Batch() = default;
    // Bind before accepting requests; the budget must outlive the batch.
    virtual void bind(Budget&) = 0;
    virtual Generation generate(std::string id, InferenceRequest&, InferenceOutput&) = 0;
    virtual void step(const std::vector<std::string>& eligible) = 0;
    virtual Metrics metrics() const = 0;
    virtual void release_idle_buffers() = 0;
    virtual bool supports_sessions() const noexcept { return false; }
    virtual bool supports_mtp() const noexcept { return false; }
};

// Request metadata shared by batch drivers; the scheduler owns lifecycle and output.
struct BatchRequest {
    BatchRequest(std::string id, const InferenceRequest& input)
        : id(std::move(id)), prompt(input.prompt), sampling(input.sampling),
          token_constraint(input.token_constraint), generation_limit(input.sampling.max_tokens),
          cache_plan(input.cache_plan), history(input.prompt) {}
    std::string id;
    std::vector<int64_t> prompt;
    MfqSamplingParams sampling;
    MfqTokenConstraintPtr token_constraint;
    int32_t generation_limit = 0, produced = 0;
    int64_t pending_token = 0;
    bool eligible = false;
    int64_t prefill_offset = 0, cache_length = 0;
    MfqPromptCachePlan cache_plan;
    std::vector<std::int64_t> history;
    std::size_t prefix = 0, checkpoint = 0;
    bool completed = false;
};

template <class Request> struct BatchState {
    std::deque<std::shared_ptr<Request>> prefilling;
    std::vector<std::shared_ptr<Request>> active;
};

template <class Sample> struct PrefillBatchResult {
    Sample sample;
    std::exception_ptr failure;
};

// Groups pending numerical operations and publishes each request's progress.
template <class Ops> class ContinuousBatch : public Batch {
    using Request = typename Ops::Request;
    using Sample = typename Ops::Sample;
    using State = BatchState<Request>;
    struct Work {
        std::shared_ptr<Request> request;
        InferenceOutput* output = nullptr;
        std::optional<PrefillChunk> prefill;
        std::optional<Sample> result;
        std::optional<OutputDelta> delta;
        std::exception_ptr failure;
        bool decode = false, active = false, custom = false;
        void schedule_prefill(PrefillChunk chunk) {
            result.reset(); delta.reset(); prefill = chunk;
        }
        void schedule_decode() {
            result.reset(); delta.reset(); decode = true;
        }
        bool ready() const { return result.has_value() || delta.has_value() || failure; }
        const Sample& sample() const {
            if (failure) std::rethrow_exception(failure);
            return result.value();
        }
    };
    int64_t prefill_chunk_size, prefill_token_budget;
    State state;
    std::unordered_map<std::string, std::shared_ptr<Work>> requests;
    Ops operations;
    bool decode_next = true;
    int64_t admissions = 0, prefill_chunks = 0, prefill_yields = 0;

    int64_t cache_position() const {
        int64_t position = 0;
        for (const auto& request : state.active)
            position = std::max(position, request->cache_length);
        return position;
    }
    void publish(Work& work, bool report_stats = true) {
        if (!work.output) throw std::logic_error("batch output is unavailable");
        const auto& sample = work.sample();
        std::vector<std::int64_t> candidates;
        if (!sample.tokens.empty()) candidates = sample.tokens;
        if (candidates.empty()) candidates.push_back(sample.token);
        work.delta = work.output->append(candidates);
        auto& request = *work.request;
        request.history.insert(request.history.end(), work.delta->token_ids.begin(),
            work.delta->token_ids.end());
        operations.accept(work.request, sample,
            std::span<const std::int64_t>(work.delta->token_ids.data(), work.delta->token_ids.size()));
        if (!work.delta->token_ids.empty()) request.pending_token = work.delta->token_ids.back();
        request.produced += static_cast<std::int32_t>(work.delta->token_ids.size());
        if (report_stats) operations.report(work.request, *work.output);
    }
    void recover(std::exception_ptr error) {
        State failed;
        for (const auto& [id, work] : requests) {
            work->failure = error;
            work->prefill.reset();
            work->decode = work->active = false;
            failed.prefilling.push_back(work->request);
        }
        state = {};
        operations.recover(failed);
    }
    void release(Work& work) {
        if (!requests.contains(work.request->id)) return;
        try {
            if (work.active) {
                std::erase(state.active, work.request);
                operations.retire({work.request}, cache_position());
            } else if (!work.failure) {
                operations.suspend_decode();
                operations.discard_prefill(work.request);
                operations.resume_decode(cache_position());
            }
        } catch (...) {
            auto error = std::current_exception();
            try { recover(error); } catch (...) { error = std::current_exception(); }
            requests.erase(work.request->id);
            std::rethrow_exception(error);
        }
        requests.erase(work.request->id);
    }
  public:
    void bind(Budget& budget) override { operations.bind(budget); }
    bool supports_sessions() const noexcept override { return operations.supports_sessions(); }
    bool supports_mtp() const noexcept override { return operations.supports_mtp(); }
    bool empty() const noexcept { return requests.empty(); }
    void invalidate_runtime_policy() {
        if constexpr (requires { operations.invalidate_runtime_policy(); }) operations.invalidate_runtime_policy();
    }
    void release_idle_buffers() override {
        if(!empty())throw std::runtime_error("cannot release active batch buffers");
        operations.suspend_decode();
    }
    template <class... Args>
    explicit ContinuousBatch(int64_t chunk_size, int64_t token_budget, Args&&... args)
        : prefill_chunk_size(std::min(chunk_size, token_budget)), prefill_token_budget(token_budget),
          operations(std::forward<Args>(args)...) {
        if (prefill_chunk_size <= 0)
            throw std::invalid_argument("continuous batching requires a positive prefill chunk");
    }

    /** Advance one batch request and retire it after its child generator is destroyed. */
    Generation generate(std::string id, InferenceRequest& input, InferenceOutput& output) override {
        if (output.stopped()) co_return;
        auto work = std::make_shared<Work>();
        work->request = std::make_shared<Request>(std::move(id), input);
        if (!requests.emplace(work->request->id, work).second)
            throw std::invalid_argument("duplicate batch request id");
        ExecutionCleanup cleanup{output.cleanup_failure, [&] { release(*work); }};
        work->output = &output;
        const auto reused = operations.prepare(work->request);
        const auto prefix = operations.prefix_tokens(work->request);
        const auto prompt_tokens = static_cast<int64_t>(input.prompt.size());
        double elapsed = 0.0;
        auto offset = reused;
        while (offset < prompt_tokens && !output.stopped()) {
            const auto end = offset < prefix ? std::min(prompt_tokens, prefix) : prompt_tokens;
            auto chunk = operations.prefill_chunk(work->request, next_prefill_chunk(end, offset, prefill_chunk_size));
            work->schedule_prefill(chunk);
            co_yield StepState::advanced;
            while (!work->ready() && !output.stopped()) co_yield StepState::waiting;
            if (output.stopped()) break;
            {
                PrefillActivity activity;
                elapsed += work->sample().timing.llm_ms;
            }
            offset += chunk.count;
            co_yield PrefillProgress{
                {static_cast<std::size_t>(offset - reused), elapsed, 0.0, elapsed}};
        }
        if (!output.stopped()) {
            bool generated = false;
            if constexpr (requires { operations.generate_decode(work->request, output,
                    std::function<void()>{}, std::function<bool()>{}, std::function<void()>{}); }) {
                auto generation = operations.generate_decode(work->request, output,
                    [&work] { work->schedule_decode(); }, [&work] { return work->ready(); },
                    [&work] { (void)work->sample(); });
                if (generation) {
                    generated = true;
                    work->custom = true;
                    while (auto step = generation->next()) co_yield std::move(step);
                }
            }
            if (!generated) {
                publish(*work);
                while (work->delta) {
                    auto delta = std::move(*work->delta);
                    work->delta.reset();
                    co_yield std::move(delta);
                    if (output.stopped()) break;
                    work->schedule_decode();
                    while (!work->ready() && !output.stopped()) co_yield StepState::waiting;
                    // A stopped output still has to publish the delta cached by
                    // step() once. release() handles a row with no delta.
                    if (!work->delta) {
                        if (output.stopped()) break;
                        (void)work->sample();
                        throw std::logic_error("batch decode produced no output delta");
                    }
                }
            }
        }
        auto tail = output.finish();
        if (!tail.diffs.empty()) co_yield std::move(tail);
        work->request->completed = !output.result.cancelled;
        cleanup.finish();
    }

    void step(const std::vector<std::string>& eligible) override {
        for (const auto& request : state.active) {
            const auto& work = *requests.at(request->id);
            request->eligible = work.decode &&
                std::find(eligible.begin(), eligible.end(), request->id) != eligible.end();
        }
        auto prefill = requests.end();
        for (const auto& id : eligible) {
            auto it = requests.find(id);
            if (it != requests.end() && it->second->prefill) { prefill = it; break; }
        }
        const bool rows_ready = std::all_of(state.active.begin(), state.active.end(),
            [&](const auto& request) {
                return std::find(eligible.begin(), eligible.end(), request->id) == eligible.end() ||
                    requests.at(request->id)->decode;
            });
        // Let eligible peers consume their previous sample and schedule the
        // next decode before executing it. Otherwise staggered coroutines
        // alternate single-row decodes and never form a stable batch.
        const bool has_decode = rows_ready && std::any_of(state.active.begin(), state.active.end(),
            [](const auto& request) { return request->eligible; });
        bool run_decode = has_decode && (prefill == requests.end() || decode_next);
        if constexpr (requires { operations.ready_decode(state, eligible); })
            run_decode = run_decode && operations.ready_decode(state, eligible);
        bool prefilling_operation = false;
        bool decoding_operation = false;
        std::vector<std::shared_ptr<Work>> prefill_work;
        try {
            if constexpr (requires { operations.synchronize(state); }) operations.synchronize(state);
            if (run_decode) {
                decoding_operation = true;
                auto decoded = operations.decode(state);
                decoding_operation = false;
                for (const auto& request : state.active) if (request->eligible) {
                    auto& work = *requests.at(request->id);
                    work.result = operations.sample(request, decoded);
                    if (!work.custom) publish(work, false);
                    request->cache_length = operations.cache_length(request);
                    work.decode = false;
                }
                for (const auto& request : state.active) if (request->eligible) {
                    const auto& work = *requests.at(request->id);
                    if (!work.custom) operations.report(request, *work.output);
                }
            } else if (prefill != requests.end()) {
                prefill_work.push_back(prefill->second);
                using Item = std::pair<std::shared_ptr<Request>, PrefillChunk>;
                std::vector<Item> items{{prefill->second->request, *prefill->second->prefill}};
                auto budget=prefill_chunk_size;
                if constexpr (requires { operations.prefill_batch_limit(); }) budget=prefill_token_budget;
                if constexpr(requires {operations.prefill_batch_token_budget();})
                    budget=std::min(budget,operations.prefill_batch_token_budget());
                auto remaining = budget - items.front().second.count;
                const auto limit = [&] {
                    if constexpr (requires { operations.prefill_batch_limit(); }) return operations.prefill_batch_limit();
                    else return eligible.size();
                }();
                for (const auto& id : eligible) {
                    auto it = requests.find(id);
                    if (it == requests.end() || !it->second->prefill ||
                        std::find(prefill_work.begin(),prefill_work.end(),it->second) != prefill_work.end()) continue;
                    const auto chunk = *it->second->prefill;
                    if (items.size() >= limit || chunk.count > remaining) continue;
                    remaining -= chunk.count;
                    prefill_work.push_back(it->second);
                    items.emplace_back(it->second->request,chunk);
                }
                operations.suspend_decode();
                PrefillActivity prefill_activity;
                prefilling_operation = true;
                std::vector<PrefillBatchResult<Sample>> samples;
                const bool use_batch = items.size() > 1 || !requires { operations.prefill_batch_limit(); };
                if (use_batch) {
                    auto results = operations.prefill_batch(items);
                    for (auto& result : results) {
                        if constexpr (requires { result.failure; result.sample; }) samples.push_back(std::move(result));
                        else samples.push_back({std::move(result), {}});
                    }
                }
                if (samples.empty()) {
                    const auto& [request,chunk]=items.front();
                    samples.push_back({operations.prefill(request, chunk),{}});
                }
                prefilling_operation = false;
                if (samples.size() != items.size()) throw std::runtime_error("invalid packed prefill result count");
                for (std::size_t row = 0; row < items.size(); ++row) {
                    auto& work = *prefill_work[row];
                    auto& request = work.request;
                    auto& result = samples[row];
                    const auto chunk = items[row].second;
                    if (result.failure) {
                        work.failure = result.failure;
                        operations.discard_prefill(request);
                    } else {
                        request->prefill_offset = chunk.offset + chunk.count;
                        ++prefill_chunks;
                        if (request->prefill_offset == static_cast<int64_t>(request->prompt.size())) {
                            request->cache_length = request->prompt.size();
                            operations.activate(request,result.sample);
                            state.active.push_back(request);
                            work.active = true;
                            ++admissions;
                        } else ++prefill_yields;
                        work.result = std::move(result.sample);
                    }
                    work.prefill.reset();
                }
                operations.resume_decode(cache_position());
            }
            decode_next = !decode_next;
        } catch (...) {
            if constexpr (requires { Ops::isolate_context_growth_failures; }) {
                bool capacity_failure = false;
                try { throw; }
                catch (const InferenceInputError& error) {
                    capacity_failure = error.code == InferenceInputErrorCode::ContextLength;
                } catch (...) {}
                if (Ops::isolate_context_growth_failures && decoding_operation && capacity_failure) {
                    // Capacity checks run before numerical decode. Release the
                    // newest eligible request while preserving established rows.
                    const auto victim = std::find_if(state.active.rbegin(),state.active.rend(),
                        [](const auto& request) { return request->eligible; });
                    if (victim != state.active.rend()) {
                        auto request = *victim;
                        auto& work = *requests.at(request->id);
                        work.failure = std::current_exception(); work.decode = false;
                        std::erase(state.active,request);
                        try {
                            operations.retire({request},cache_position());
                            work.active = false;
                            return;
                        } catch (...) { recover(std::current_exception()); throw; }
                    }
                }
            }
            if constexpr (requires { Ops::isolate_prefill_failures; }) {
                if (Ops::isolate_prefill_failures && prefilling_operation && prefill != requests.end()) {
                    const auto error = std::current_exception();
                    try {
                        for (const auto& work : prefill_work) {
                            work->failure = error;
                            work->prefill.reset();
                            work->decode = false;
                            operations.discard_prefill(work->request);
                        }
                        operations.resume_decode(cache_position());
                        return;
                    } catch (...) { recover(std::current_exception()); throw; }
                }
            }
            const auto error = std::current_exception();
            if constexpr (requires { operations.recoverable_failure(error); }) {
                if (operations.recoverable_failure(error)) {
                    if (!run_decode && prefill != requests.end()) {
                        for (const auto& item : prefill_work) {
                            auto& work = *item;
                            work.failure = error;
                            work.prefill.reset();
                            work.decode = work.active = false;
                            std::erase(state.active, work.request);
                            operations.discard_prefill(work.request);
                        }
                        operations.resume_decode(cache_position());
                    } else recover(error);
                    return;
                }
            }
            recover(error);
            throw;
        }
    }

    Metrics metrics() const override {
        auto result = operations.metrics();
        result.emplace_back("continuous_batching_active", state.active.size());
        result.emplace_back("continuous_batching_prefilling", requests.size() - state.active.size());
        result.emplace_back("continuous_batching_requests", admissions);
        result.emplace_back("continuous_batching_admissions", admissions);
        result.emplace_back("continuous_batching_prefill_chunks", prefill_chunks);
        result.emplace_back("continuous_batching_prefill_yields", prefill_yields);
        return result;
    }
};
} // namespace tyloi::engine

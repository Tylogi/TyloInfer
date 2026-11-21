#pragma once

#include "mlx_mtp.h"
#include "mlx_prefix_cache.h"
#include "mlx_sampling.h"
#include "tyloinfer/token_constraint.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mfq::metal {

class MfqContainer;

// Shared serial generation around backbones that own their cache geometry.
// The engine owns prefix lookup, scheduling, token streaming and cancellation.
template <class Model>
class MlxTextBackboneRuntime {
public:
    using SessionState = typename Model::TextSessionState;
    static constexpr bool prefix_checkpoints = true;
    int vocabulary() const noexcept { return model_.vocabulary(); }
    int preferred_prefill_chunk_size(int requested) const noexcept {
        return std::min(512, requested);
    }

    template <class... Args>
    static MlxTextBackboneRuntime load(const MfqContainer& model, int context, Args&&... args) {
        return MlxTextBackboneRuntime(Model::load(model, context, std::forward<Args>(args)...));
    }

    bool supports_mtp() const noexcept {
        if constexpr (requires { model_.supports_mtp(); }) return model_.supports_mtp();
        return false;
    }
    const MlxMtpGenerationStats& last_mtp_stats() const noexcept { return mtp_stats_; }
    auto qsa_kv_offload_stats() const
        requires requires(const Model& model) { model.qsa_kv_offload_stats(); } {
        return model_.qsa_kv_offload_stats();
    }
    std::size_t kv_cache_bytes() const
        requires requires(const Model& model, const typename Model::Cache& cache) { model.kv_cache_bytes(cache); } {
        return model_.kv_cache_bytes(cache_);
    }

    bool supports_text_session_state() const noexcept { return true; }
    int cache_position() const noexcept { return cache_.position; }
    void reset_cache(int batch = 1) {
        if (batch != 1) throw std::invalid_argument("serial Metal text runtime requires batch one");
        cache_ = {};
        last_logits_.reset();
    }
    void set_mtp_limit(int) noexcept {}
    void clear_cache() { reset_cache(); }
    void reset_generation_state() { reset_cache(); }

    SessionState capture_text_session_state(const std::vector<std::int64_t>& tokens, bool detached = true) const {
        auto state = model_.capture(cache_, tokens, detached);
        if (last_logits_) {
            state.last_logits = last_logits_;
            state.bytes += last_logits_->nbytes();
        }
        return state;
    }

    void restore_text_session_state(SessionState state) {
        if (!state.last_logits || state.last_logits->shape() != mlx::core::Shape{1, model_.vocabulary()})
            throw std::invalid_argument("restored Metal text logits do not match the loaded model");
        auto cache = model_.restore(state);
        cache_ = std::move(cache);
        last_logits_ = std::move(state.last_logits);
    }

    std::int32_t generate(const std::vector<std::int64_t>& prompt,
        const MlxSamplingParams& sampling, int max_tokens,
        const std::function<bool(std::int64_t)>& callback,
        const std::function<void(std::size_t, double)>& prefill_callback,
        const MfqTokenConstraintPtr& constraint, std::optional<std::size_t> stable_prefix_tokens,
        int chunk_size, const MlxPrefixCacheHooks& prefix = {}) {
        using namespace mlx::core;
        if (sampling.warmup) reset_cache();
        mtp_stats_ = {.available = supports_mtp()};
        const int vocab = model_.vocabulary();
        if (prompt.empty() || prompt.size() > static_cast<std::size_t>(model_.context()) ||
            max_tokens < 0 || chunk_size <= 0 || cache_.position > static_cast<int>(prompt.size()))
            throw std::invalid_argument("invalid Metal text generation lengths");
        mlx_validate_token_set(prompt, vocab);
        const int prompt_count = static_cast<int>(prompt.size());
        std::vector<std::int32_t> input(prompt.begin(), prompt.end());
        const array ids(input.begin(), Shape{1, prompt_count}, int32);
        const int begin = cache_.position;
        const auto started = std::chrono::steady_clock::now();
        for (int row = begin; row < prompt_count;) {
            auto end = prefix.prefill_end(row, row + std::min(chunk_size, prompt_count - row));
            if (!prefix && stable_prefix_tokens && *stable_prefix_tokens > static_cast<std::size_t>(row))
                end = std::min(end, *stable_prefix_tokens);
            last_logits_ = mlx_last_token_logits(model_.forward(
                slice(ids, {0, row}, {1, static_cast<int>(end)}), cache_, true), vocab);
            last_logits_->eval();
            row = static_cast<int>(end);
            prefix.checkpoint(row);
        }
        if (prefill_callback && prompt_count > begin)
            prefill_callback(prompt_count - begin, std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
        if (!last_logits_) throw std::runtime_error("Metal text runtime has no logits for the prompt");

        auto counts = mlx_generation_token_counts(sampling, ids, vocab);
        const int limit = std::min(max_tokens, model_.context() - prompt_count);
        if constexpr (requires { model_.supports_mtp(); }) {
            if (sampling.enable_mtp && supports_mtp() && limit > 1) {
                auto request = MlxMtpEngineRequest{vocab, limit, model_.context(), *last_logits_, sampling,
                    std::move(counts), {}, callback, 0, constraint};
                return model_.generate_speculative(cache_, std::move(request), prefix, mtp_stats_, last_logits_);
            }
        }
        MlxSampler sampler(sampling);
        int generated = 0;
        while (generated < limit) {
            auto logits = *last_logits_;
            if (counts) logits = sampler.apply_penalties(logits, *counts);
            if (constraint) {
                array values = contiguous(astype(logits, float32));
                values.eval();
                std::vector<float> masked(values.data<float>(), values.data<float>() + vocab);
                constraint->apply(masked.data(), masked.size());
                logits = array(masked.begin(), {1, vocab}, float32);
            }
            const auto token = sampler.sample(logits).template item<std::int32_t>();
            if (token < 0 || token >= vocab || (constraint && !constraint->allows(token)))
                throw std::runtime_error("Metal text sampler returned an invalid token");
            if (constraint) constraint->accept(token);
            const array next({token}, Shape{1, 1}, int32);
            if (counts) *counts = sample_token_counts_add(*counts, next);
            ++generated;
            const bool keep = !callback || callback(token);
            if (prefix.wants(cache_.position + 1, !keep || generated == limit) || (keep && generated < limit)) {
                last_logits_ = mlx_last_token_logits(model_.forward(next, cache_, true), vocab);
                last_logits_->eval();
                prefix.checkpoint(cache_.position, !keep || generated == limit);
            }
            if (!keep) break;
        }
        return generated;
    }

private:
    explicit MlxTextBackboneRuntime(Model model) : model_(std::move(model)) {}
    Model model_;
    typename Model::Cache cache_;
    std::optional<mlx::core::array> last_logits_;
    MlxMtpGenerationStats mtp_stats_;
};

}  // namespace mfq::metal

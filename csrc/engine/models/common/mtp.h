#pragma once

#include "engine/generation/sequence.h"
#include "engine/models/session/snapshot.h"
#include "engine/tensor/tensor.h"
#include "engine/models/state.h"
#include "engine/models/input.h"
#include "engine/generation/mtp/policy.h"
#include "engine/requests/sampling.h"

#include <cstdint>
#include <algorithm>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>

namespace tyloi::models {

class CausalModel;

struct MtpStep {
    Tensor sample_hidden;
    Tensor chain_hidden;
    // Optional last-row vocabulary projection computed in the predictor graph.
    Tensor sample_logits;
    int32_t greedy_token=-1;
};

struct MtpRow {
    MtpState* state = nullptr;
    Tensor hidden;
    Tensor ids;
    Tensor positions;
};

struct MtpDraft {
    // The last history ID followed by the device-generated proposals.
    Tensor ids;
    bool initialized = false;
};

// Predictor weights and math use the explicit state owned by the request.
struct Mtp {
    virtual ~Mtp() = default;
    virtual std::unique_ptr<MtpState> make_state() const = 0;
    // Predictors that allocate on reset report new backing before admission.
    virtual std::vector<std::pair<Device, std::size_t>> reset_growth(
            const MtpState& state, std::int64_t batch) const {
        if (!state.belongs_to(*this) || batch <= 0)
            throw std::invalid_argument("invalid predictor reset state or batch");
        return {};
    }
    virtual void reset(MtpState& state, int64_t batch = 1) const = 0;
    // Additional predictor state backing required by the next step or prefill.
    virtual std::vector<std::pair<Device, std::size_t>> growth(
            const MtpState&, const Tensor&, const Tensor&, const Tensor&, const ForwardOptions&) const {
        return {};
    }
    virtual MtpStep step(
        MtpState& state,
        const Tensor& previous_hidden,
        const Tensor& next_ids,
        const Tensor& positions = {},
        const ForwardOptions& options = {}) const = 0;
    virtual std::vector<MtpStep> batch_step(
        std::span<const MtpRow>, const ForwardOptions& = {}) const {
        throw std::logic_error("batched MTP step is unsupported");
    }
    virtual StepSequence<MtpDraft> greedy_draft(
        const CausalModel&, MtpRow, int, const ForwardOptions&) const { return {}; }
    virtual void trim_cache_to(MtpState& state, int64_t position) const = 0;
    virtual bool teacher_forced_prompt_prime() const noexcept = 0;
    virtual bool target_bootstrap_decode() const noexcept = 0;
    virtual bool preserve_output_dtype() const noexcept = 0;
    virtual std::int64_t prefill_chunk_size(std::int64_t requested) const noexcept {
        return std::min<std::int64_t>(512, requested);
    }
    virtual bool stream_prompt_history(std::int64_t) const { return false; }
    virtual bool host_prompt_history(std::int64_t tokens, std::size_t available,
                                     std::size_t raw_row_bytes) const {
        if (tokens <= 0) return false;
        const auto rows=std::size_t(tokens);
        if (raw_row_bytes && rows>std::numeric_limits<std::size_t>::max()/raw_row_bytes) return true;
        const auto margin=std::size_t(32)<<20;
        return available<margin || rows*raw_row_bytes>available-margin;
    }
    virtual bool bootstrap_before_publication() const { return false; }
    virtual std::size_t predictor_history_merge_limit() const { return 0; }
    virtual std::size_t predictor_history_flush_threshold() const { return 512; }
    virtual std::size_t predictor_parked_history_flush_threshold() const { return 512; }
    virtual std::size_t prefill_workspace_bytes(std::int64_t) const { return 0; }
    virtual MtpStep prefill(MtpState& state, const Tensor& hidden, const Tensor& ids,
                            const Tensor& positions, const ForwardOptions& options) const {
        return step(state, hidden, ids, positions, options);
    }
    virtual int maximum_draft_depth() const noexcept {
        return tyloi::engine::mtp::kMaximumDraftDepth;
    }
    virtual bool blockwise_drafting() const noexcept { return false; }
    virtual std::span<const std::int64_t> target_layers() const noexcept { return {}; }
    virtual std::int64_t target_context_window() const noexcept { return 0; }
    virtual std::unique_ptr<tyloi::engine::mtp::DepthPolicy> depth_policy(
            int maximum_depth, tyloi::engine::mtp::PolicyState& state) const {
        return std::make_unique<tyloi::engine::mtp::AdaptivePolicy>(maximum_depth, &state);
    }
    virtual MfqSamplingParams draft_sampling(MfqSamplingParams sampling, bool) const {
        return sampling;
    }
    virtual bool split_target_verification() const noexcept { return false; }
    virtual bool retains_partial_target_prefix() const noexcept {
        return false;
    }
    virtual bool supports_session_state(const MtpState&) const noexcept { return false; }
    virtual bool session_boundary(std::int64_t target, std::int64_t predictor) const noexcept {
        return target > 0 && predictor == target - 1;
    }
    virtual std::unique_ptr<Restore> prepare_restore(MtpState& state, const MtpSessionState& snapshot) const {
        if (!state.belongs_to(*this)) throw std::invalid_argument("predictor state belongs to another model");
        return state.prepare_restore(snapshot);
    }
    virtual MtpSessionState capture_session_state(
        const MtpState&,
        std::int64_t,
        const Tensor&) const {
        throw std::runtime_error(
            "this CUDA MTP predictor has no session state");
    }
    virtual void restore_session_state(MtpState&, const MtpSessionState&) const {
        throw std::runtime_error(
            "this CUDA MTP predictor has no session state");
    }
    virtual void seed_target_context(MtpState& state, const Tensor& hidden, std::int64_t tokens) const {
        if (hidden.dim() != 3 || hidden.size(1) != tokens)
            throw std::runtime_error("this CUDA MTP predictor requires complete prompt context");
        append_target_context(state, hidden, 0);
    }
    virtual void append_target_context(
        MtpState&,
        const Tensor&,
        std::int64_t) const {
        throw std::runtime_error(
            "this CUDA MTP predictor has no target-context adapter");
    }
    virtual StepSequence<Tensor> draft_block(
        MtpState&,
        const Tensor&,
        int,
        bool) const {
        throw std::runtime_error(
            "this CUDA MTP predictor has no block-draft adapter");
    }
    virtual Tensor draft_next(
        MtpState&,
        const Tensor&,
        const Tensor&) const {
        throw std::runtime_error("this CUDA MTP predictor has no block-draft adapter");
    }

protected:
    template<class T> T& state(MtpState& value) const {
        if (!value.belongs_to(*this)) throw std::invalid_argument("predictor state belongs to another model");
        return static_cast<T&>(value);
    }
    template<class T> const T& state(const MtpState& value) const {
        if (!value.belongs_to(*this)) throw std::invalid_argument("predictor state belongs to another model");
        return static_cast<const T&>(value);
    }
};

}  // namespace tyloi::models

#pragma once

#include "engine/models/qwen35/predictor.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/qwen35/mtp_state.h"
#include "engine/models/common/mtp.h"

#include <memory>

namespace tyloi::models::qwen35 {

class Mtp final : public tyloi::models::Mtp {
public:
    Mtp(tyloi::models::qwen35::Predictor model, bool preallocate);

    std::unique_ptr<tyloi::models::MtpState> make_state() const override;
    std::vector<std::pair<Device, std::size_t>> reset_growth(
        const tyloi::models::MtpState&, int64_t batch) const override;
    void reset(tyloi::models::MtpState& state, int64_t batch = 1) const override;
    std::vector<std::pair<Device, std::size_t>> growth(const tyloi::models::MtpState&,
        const Tensor& hidden, const Tensor& ids, const Tensor& positions,
        const tyloi::models::ForwardOptions&) const override;
    // The caller owns the predictor execution scope.
    tyloi::models::MtpStep step(tyloi::models::MtpState& state, const ::tyloi::Tensor& hidden,
        const ::tyloi::Tensor& next_ids, const ::tyloi::Tensor& positions = {},
        const tyloi::models::ForwardOptions& options = {}) const override;
    tyloi::models::MtpStep prefill(tyloi::models::MtpState& state, const ::tyloi::Tensor& hidden,
        const ::tyloi::Tensor& ids, const ::tyloi::Tensor& positions,
        const tyloi::models::ForwardOptions& options) const override;
    std::vector<tyloi::models::MtpStep> batch_step(
        std::span<const tyloi::models::MtpRow>,
        const tyloi::models::ForwardOptions& options = {}) const override;
    StepSequence<tyloi::models::MtpDraft> greedy_draft(const tyloi::models::CausalModel& target,
        tyloi::models::MtpRow row, int depth,
        const tyloi::models::ForwardOptions& options = {}) const override;
#if TYLOINFER_HAS_CUDA
    StepSequence<tyloi::models::MtpStep> graph_step(const tyloi::models::CausalModel& target,
        tyloi::models::MtpRow row, MtpStepMode mode, const ForwardOptions& options) const override;
#endif

    void trim_cache_to(tyloi::models::MtpState& state, int64_t position) const override;
    bool supports_session_state(const tyloi::models::MtpState&) const noexcept override { return true; }
    MtpSessionState capture_session_state(const tyloi::models::MtpState& state, int64_t target_cache_position,
        const ::tyloi::Tensor& last_target_hidden) const override;

    bool teacher_forced_prompt_prime() const noexcept override { return true; }
    bool target_bootstrap_decode() const noexcept override { return false; }
    bool preserve_output_dtype() const noexcept override { return false; }
    bool retains_partial_target_prefix() const noexcept override { return true; }
    std::size_t predictor_history_merge_limit() const override {
        return model_.device.is_cuda() ? 1 + engine::mtp::kMaximumReportedDraftDepth : 0;
    }

private:
#if TYLOINFER_HAS_CUDA
    struct GraphResult { Tensor output, logits; bool initialized; };
    StepSequence<GraphResult> graph(const CausalModel& target, MtpRow row,
        MtpStepMode mode, int depth, const ForwardOptions& options) const;
#endif
    tyloi::models::qwen35::Predictor model_;
    bool preallocate_;
};

} // namespace tyloi::models::qwen35

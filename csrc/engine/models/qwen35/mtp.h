#pragma once

#include "engine/models/qwen35/predictor.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/qwen35/mtp_state.h"
#include "engine/models/common/mtp.h"

#include <functional>
#include <memory>

namespace tyloi::cuda::qwen35 {

class State;
class Mtp;

void restore_session(State&, tyloi::models::Mtp*, const TextSessionState&, const std::function<void()>&);

class Mtp final : public tyloi::models::Mtp {
public:
    Mtp(tyloi::models::qwen35::Predictor model, bool preallocate);

    std::unique_ptr<tyloi::models::MtpState> make_state() const override;
    void reset(tyloi::models::MtpState& state, int64_t batch = 1) const override;
    // The caller owns the predictor execution scope.
    tyloi::models::MtpStep step(tyloi::models::MtpState& state, const ::tyloi::Tensor& hidden,
        const ::tyloi::Tensor& next_ids, const ::tyloi::Tensor& positions = {},
        const tyloi::models::ForwardOptions& options = {}) const override;

    void trim_cache_to(tyloi::models::MtpState& state, int64_t position) const override;
    bool supports_session_state(const tyloi::models::MtpState&) const noexcept override { return true; }
    MtpSessionState capture_session_state(const tyloi::models::MtpState& state, int64_t target_cache_position,
        const ::tyloi::Tensor& last_target_hidden) const override;

    bool teacher_forced_prompt_prime() const noexcept override { return true; }
    bool target_bootstrap_decode() const noexcept override { return false; }
    bool preserve_output_dtype() const noexcept override { return false; }
    bool retains_partial_target_prefix() const noexcept override { return true; }

private:
    tyloi::models::qwen35::Predictor model_;
    bool preallocate_;
};

} // namespace tyloi::cuda::qwen35

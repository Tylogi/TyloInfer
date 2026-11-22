#pragma once

#include "continuous_batch.h"
#include "mlx_sampling.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {

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

class MlxRequestBatch {
public:
    virtual ~MlxRequestBatch() = default;
    virtual bool can_batch(const mfq::engine::EngineRequest& request) const = 0;
    virtual bool empty() const noexcept = 0;
    virtual void step(const std::vector<std::string>& eligible) = 0;
    virtual mfq::engine::Generation generate(std::string id,
        mfq::engine::InferenceRequest& input, mfq::engine::InferenceOutput& output) = 0;
    virtual mfq::engine::Metrics metrics() const = 0;
    virtual void invalidate_runtime_policy() = 0;
};

template<class Operations, class Admission>
class MlxRequestBatchAdapter final : public MlxRequestBatch {
public:
    template<class... Args>
    MlxRequestBatchAdapter(int chunk_size, int token_budget, Admission admission, Args&&... args)
        : batch_(chunk_size, token_budget, std::forward<Args>(args)...), admission_(std::move(admission)) {}

    bool can_batch(const mfq::engine::EngineRequest& request) const override {
        return !request.input.media && admission_(request);
    }
    bool empty() const noexcept override { return batch_.empty(); }
    void step(const std::vector<std::string>& eligible) override { batch_.step(eligible); }
    mfq::engine::Generation generate(std::string id,
        mfq::engine::InferenceRequest& input, mfq::engine::InferenceOutput& output) override {
        return batch_.generate(std::move(id), input, output);
    }
    mfq::engine::Metrics metrics() const override { return batch_.metrics(); }
    void invalidate_runtime_policy() override { batch_.invalidate_runtime_policy(); }

private:
    mfq::engine::ContinuousBatch<Operations> batch_;
    Admission admission_;
};

template<class Operations, class Admission, class... Args>
std::unique_ptr<MlxRequestBatch> make_mlx_request_batch(int chunk_size, int token_budget,
    Admission admission, Args&&... args) {
    return std::make_unique<MlxRequestBatchAdapter<Operations, Admission>>(
        chunk_size, token_budget, std::move(admission), std::forward<Args>(args)...);
}

} // namespace mfq::metal

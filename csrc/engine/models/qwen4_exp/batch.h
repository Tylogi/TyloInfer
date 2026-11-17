#pragma once

#include "engine/models/batch.h"
#include "engine/models/qwen4_exp/state.h"

namespace tyloi::models::qwen4_exp {
struct Model;

// Each admitted request owns one independent model state.
class Batch : public models::Batch {
public:
    Batch(Model&, const BatchConfig&, bool batch_attention_projections);
    std::unique_ptr<BatchState> make_state(std::int64_t) override;
    bool supports_sessions() const noexcept override { return true; }
    bool supports_mtp() const noexcept override { return true; }
    bool supports_packed() const override { return true; }
    std::size_t prefill_row_bytes() const override;
    StepSequence<std::vector<Decoded>> prefill(std::vector<BatchState*>, std::vector<Tensor>) override;
    StepSequence<std::vector<Decoded>> verify(std::vector<BatchState*>, std::vector<Tensor>) override;
    Tensor decode(std::span<const BatchRow>, Tensor ids, const Tensor& host_ids) override;
    StepSequence<std::unique_ptr<Plan>> plan(std::span<const BatchRow>, bool) override { return {}; }
    StepSequence<std::optional<Tensor>> capture(std::span<const BatchRow>, const Tensor&,
        const Tensor&, bool, const Plan&) override { return {}; }
    void clear() override {}
    engine::Metrics metrics() const override { return {}; }
protected:
    std::vector<State*> active(std::span<const BatchRow>) const;
    Model& model_;
private:
    StepSequence<std::vector<Decoded>> forward(std::vector<BatchState*>,
        std::vector<Tensor>, bool verify);
    bool batch_attention_projections_;
};
} // namespace tyloi::models::qwen4_exp

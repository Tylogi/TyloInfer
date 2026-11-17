#pragma once

#include "engine/models/batch.h"
#include "engine/models/qwen35/model.h"
#include "engine/models/qwen35/state.h"
#include "engine/state/kv/paged.h"

namespace tyloi::models::qwen35 {

class Batch : public models::Batch {
public:
    Batch(const Model&, BatchConfig);
    BatchGrowth initial_growth_bytes() const override;
    void initialize() override;
    bool supports_sessions() const noexcept override { return true; }
    bool supports_mtp() const noexcept override { return model_.supports_speculation(); }
    bool supports_packed() const override { return true; }
    std::unique_ptr<models::BatchState> make_state(std::int64_t prompt) override;
    void activate(models::BatchState&) override;
    void release(models::BatchState&) override;
    BatchGrowth decode_growth(std::span<const BatchRow>) const override;
    void prepare_decode(std::span<const BatchRow>) override;
    std::size_t prefill_row_bytes() const override;
    StepSequence<std::vector<Decoded>> prefill(std::vector<models::BatchState*>, std::vector<Tensor>) override;
    StepSequence<std::vector<Decoded>> verify(std::vector<models::BatchState*>, std::vector<Tensor>) override;
    Tensor decode(std::span<const BatchRow>, Tensor ids, const Tensor& host_ids) override;
    StepSequence<std::unique_ptr<Plan>> plan(std::span<const BatchRow>, bool) override { return {}; }
    StepSequence<std::optional<Tensor>> capture(std::span<const BatchRow>, const Tensor&,
        const Tensor&, bool, const Plan&) override { return {}; }
    void clear() override {}
    engine::Metrics metrics() const override;

protected:
    std::vector<State*> active(std::span<const BatchRow>) const;
    bool has_cached_moe() const;
    const Model& model_;
    BatchConfig config_;

private:
    class Request;
    Request& request(models::BatchState&) const;
    StepSequence<std::vector<Decoded>> forward(std::vector<models::BatchState*>, std::vector<Tensor>, bool verify);
    std::unique_ptr<state::PagedKv> pages_;
    std::int64_t releases_ = 0;
};

} // namespace tyloi::models::qwen35

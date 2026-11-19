#pragma once

#include "engine/ops/ffn.h"

#include <memory>
#include <span>

namespace tyloi::weights {

// Create before shared-expert work; retain it through the selected projections.
// Preparation and consumption require the same execution scope.
class ExpertBatch {
public:
    ExpertBatch(std::span<const Tensor> gate_up, const Tensor& down, const Tensor& ids,
                std::int64_t row_offset, std::int64_t row_total,
                ops::MlpActivation activation = ops::MlpActivation::silu_gated, bool prefill = false);
    ~ExpertBatch();
    ExpertBatch(ExpertBatch&&) noexcept;
    ExpertBatch& operator=(ExpertBatch&&) noexcept;
    ExpertBatch(const ExpertBatch&) = delete;
    ExpertBatch& operator=(const ExpertBatch&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend ops::RoutedExperts ops::Ffn::routed(const Tensor&, const Tensor&,
                                             ExpertBatch&, const ops::ExpertSpec&);
    friend Tensor ops::Ffn::routed_shared(const Tensor&, const Tensor&, ExpertBatch&,
        std::span<const Tensor>, const Tensor&, const Tensor&, const Tensor&,
        const ops::ExpertSpec&, std::int64_t);
};

} // namespace tyloi::weights

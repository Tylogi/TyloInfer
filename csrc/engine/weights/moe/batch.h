#pragma once

#include "engine/ops.h"

#include <memory>
#include <span>

namespace tyloi::weights {

// Create before shared-expert work; retain it through the selected projections.
// Preparation and consumption require the same execution scope.
class ExpertBatch {
public:
    ExpertBatch(std::span<const Tensor> gate_up, const Tensor& down, const Tensor& ids);
    ~ExpertBatch();
    ExpertBatch(ExpertBatch&&) noexcept;
    ExpertBatch& operator=(ExpertBatch&&) noexcept;
    ExpertBatch(const ExpertBatch&) = delete;
    ExpertBatch& operator=(const ExpertBatch&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend ops::RoutedExperts ops::routed(const Tensor&, const Tensor&, const Tensor&,
                                           ExpertBatch&, const ops::ExpertSpec&);
};

} // namespace tyloi::weights

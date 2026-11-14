#pragma once

#include "mlx_tensor.h"

namespace tyloi::metal {
class MlxRmsNorm;

// Register independent projections over one activation with the backend-wide
// grouped coordinator. Model adapters only identify the participating
// tensors; format eligibility and kernel selection remain runtime-owned.
std::optional<MlxGroupedLinear> mlx_group_linears(
    std::span<const MlxLinear* const> linears);

// Runtime-owned coordinator for an arbitrary number of independent
// projections over the same activation. Eligible contiguous projections are
// partitioned into the largest genuinely fused groups supported by the
// packed kernel; dense or unsupported members remain exact standalone
// calls. Model adapters therefore describe topology without duplicating
// format- or row-count dispatch policy.
class MlxProjectionBatch {
public:
    explicit MlxProjectionBatch(
        std::vector<const MlxLinear*> linears);

    std::vector<mlx::core::array> operator()(
        const mlx::core::array& input) const;
    std::vector<mlx::core::array> forward_normalized(
        const mlx::core::array& input, const MlxRmsNorm& norm) const;

    // Execute a two-projection Gate/Up batch and apply SwiGLU. When the
    // registered formats expose a fused small-M implementation, projection
    // and activation stay in that kernel; otherwise the same coordinator
    // performs the best grouped/standalone projection partition first.
    mlx::core::array swiglu(
        const mlx::core::array& input,
        float limit = 0.0f) const;
    bool supports_fused_swiglu(
        const mlx::core::array& input) const noexcept;

    std::size_t projection_count() const noexcept;
    std::size_t grouped_projection_count() const noexcept;
    bool projections_share_group(
        std::size_t begin,
        std::size_t count) const noexcept;

private:
    std::vector<mlx::core::array> project(
        const mlx::core::array& input,
        std::optional<mlx::core::Dtype> native_output_dtype) const;
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

 } // namespace tyloi::metal

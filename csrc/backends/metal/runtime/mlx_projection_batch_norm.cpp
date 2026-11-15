#include "mlx_projection_batch.h"
#include "mlx_transformer.h"

namespace tyloi::metal {
std::vector<mlx::core::array> MlxProjectionBatch::forward_normalized(
    const mlx::core::array& input, const MlxRmsNorm& norm) const {
    const bool bfloat_consumers = input.dtype() == mlx::core::float16 && input.ndim() >= 2 &&
        input.shape(-1) == norm.width() && input.size() / std::size_t(norm.width()) <= 8 &&
        bfloat_input();
    if (bfloat_consumers) return project(norm(input, mlx::core::bfloat16), input.dtype());
    return (*this)(norm(input));
}
} // namespace tyloi::metal

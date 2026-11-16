#pragma once

#include "engine/ops/ops.h"


namespace tyloi::metal::ops {

::tyloi::ops::TailRopeTable tail_rope_table(std::int64_t max_positions, double base, std::int64_t compress_ratio,
    Device device, double compress_base = 0.0, std::int64_t original_positions = 0,
    double factor = 1.0, double beta_fast = 32.0, double beta_slow = 1.0);
Tensor tail_rope(const Tensor& input, const Tensor& positions,
    const ::tyloi::ops::TailRopeTable& table, bool inverse = false);

} // namespace tyloi::metal::ops

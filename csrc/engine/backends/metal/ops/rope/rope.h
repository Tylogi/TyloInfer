#pragma once

#include "engine/ops.h"

namespace tyloi::metal::ops {

::tyloi::ops::RopeTable rope_table(std::int64_t positions, std::int64_t dimension, double base,
    Device device, std::int64_t frequency_dim = 0, std::int64_t active_pairs = -1);
Tensor rope(const Tensor& input, const Tensor& positions, const ::tyloi::ops::RopeTable& table,
    bool bf16 = false, bool grid_mrope_positions = false);

} // namespace tyloi::metal::ops

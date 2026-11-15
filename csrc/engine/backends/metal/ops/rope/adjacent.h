#pragma once

#include <mlx/array.h>

namespace tyloi::metal::ops {

mlx::core::array adjacent_rope(const mlx::core::array& value, int rotary_dimension,
    const mlx::core::array& cosine, const mlx::core::array& sine, bool inverse = false);

} // namespace tyloi::metal::ops

#pragma once
#include <mlx/mlx.h>
#include <vector>

namespace tyloi::metal {
struct MlxGgufRotation {
    int block;
    std::vector<float> pattern;
    mlx::core::array signs;
};
}

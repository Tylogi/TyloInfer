#pragma once

#include "engine/weights/encoding/quant/nint_rows.h"
#include <span>
#include <mlx/mlx.h>

namespace tyloi::metal {
class MlxMappedNintRows : public tyloi::NintRows {
public:
    explicit MlxMappedNintRows(std::span<const std::uint8_t> blob)
        : NintRows(blob.data(), blob.size()) {}
};

class MlxNintRowBatch : public tyloi::NintRowBatch {
public:
    mlx::core::array decode(mlx::core::Dtype dtype = mlx::core::float16) const;
};
} // namespace tyloi::metal

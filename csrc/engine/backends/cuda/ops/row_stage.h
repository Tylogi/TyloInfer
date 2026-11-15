#pragma once

#include "engine/weights/row_stage.h"

namespace tyloi::cuda {

std::shared_ptr<RowStage> make_dense_row_stage(const std::vector<Tensor>& tables,
    std::int64_t rows_per_shard, const std::vector<std::int64_t>& shape, Device device);

} // namespace tyloi::cuda

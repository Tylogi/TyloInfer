#pragma once

#include "engine/weights/rows/stage.h"

namespace tyloi::weights { class GgufRows; class GgufRead; }

namespace tyloi::cuda::gguf {

std::shared_ptr<RowStage> make_row_stage(
    const std::vector<std::shared_ptr<weights::GgufRows>>& tables,
    std::int64_t rows_per_shard, const std::vector<std::int64_t>& shape,
    Device device, std::shared_ptr<weights::GgufRead> read = {});

std::pair<std::size_t, std::size_t> row_stage_growth(
    const std::vector<std::shared_ptr<weights::GgufRows>>& tables,
    std::int64_t rows_per_shard, const std::vector<std::int64_t>& shape, Device device);

Tensor lookup_rows(const std::vector<std::shared_ptr<weights::GgufRows>>& tables,
    std::int64_t rows_per_shard, const std::vector<std::int64_t>& ids,
    const std::vector<std::int64_t>& shape, Device device,
    std::shared_ptr<weights::GgufRead> read = {});

} // namespace tyloi::cuda::gguf

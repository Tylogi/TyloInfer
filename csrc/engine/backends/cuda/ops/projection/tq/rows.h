#pragma once

#include "engine/weights/rows/stage.h"
#include "engine/weights/encoding/tq/table.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tyloi::weights { class TqRead; }

namespace tyloi::cuda::tq {

::tyloi::Tensor lookup_rows(
    const tyloi::TqTable&, const std::int64_t*, std::size_t,
    const ::tyloi::Device&);

::tyloi::Tensor lookup_rows(
    const tyloi::TqTable&, const ::tyloi::Tensor& ids);

::tyloi::Tensor lookup_rows(
    const std::vector<std::shared_ptr<tyloi::TqTable>>& tables,
    std::int64_t rows_per_shard, const std::vector<std::int64_t>& ids,
    const ::tyloi::Device& device);

std::shared_ptr<RowStage> make_row_stage(
    const std::vector<std::shared_ptr<tyloi::TqTable>>& tables,
    std::int64_t rows_per_shard, const std::vector<std::int64_t>& shape,
    const ::tyloi::Device& device, std::shared_ptr<weights::TqRead> read);

std::pair<std::size_t, std::size_t> row_stage_growth(
    const std::vector<std::shared_ptr<tyloi::TqTable>>& tables,
    std::int64_t rows_per_shard, const std::vector<std::int64_t>& shape,
    const ::tyloi::Device& device);

} // namespace tyloi::cuda::tq

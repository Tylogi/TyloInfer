#pragma once

#include "engine/ops/kv_storage.h"

namespace tyloi::cuda::ops {

void bind_kv_chunk(Tensor keys, Tensor values, std::size_t index,
    const Tensor& key_chunk, const Tensor& value_chunk);
std::size_t gather_kv(std::span<const tyloi::ops::KvGatherRow> rows,
    const tyloi::ops::KvGatherWorkspace& workspace);

} // namespace tyloi::cuda::ops

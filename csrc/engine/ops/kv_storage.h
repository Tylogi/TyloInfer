#pragma once

#include "engine/tensor.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>

namespace tyloi::ops {

// Dense CPU/CUDA arena view; retains the original backing without copying.
Tensor storage_view(const Tensor& arena, std::size_t offset,
    std::span<const std::int64_t> shape, ScalarType dtype);
inline Tensor storage_view(const Tensor& arena, std::size_t offset,
        std::initializer_list<std::int64_t> shape, ScalarType dtype) {
    return storage_view(arena, offset, std::span<const std::int64_t>(shape.begin(), shape.size()), dtype);
}

// CUDA pointer tables retain their addresses; the upload completes before return.
void bind_kv_chunk(Tensor keys, Tensor values, std::size_t index,
    const Tensor& key_chunk, const Tensor& value_chunk);

struct KvGatherWorkspace {
    Tensor rows, upload, host, descriptors, device_descriptors;
    std::int64_t heads = 1, key_width = 0;
};

struct KvGatherRow {
    Tensor source;
    std::int64_t row = 0, output = 0;
};

// Copies CPU rows into pinned staging before submission; returns uploaded bytes.
// The caller waits for the prior consumer before reusing the fixed workspace.
std::size_t gather_kv(std::span<const KvGatherRow> rows, const KvGatherWorkspace& workspace);

} // namespace tyloi::ops

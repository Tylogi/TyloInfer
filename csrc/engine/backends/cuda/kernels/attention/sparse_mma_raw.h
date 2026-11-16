#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace tyloi::cuda::sparse_mma_raw {
struct Hot {
    const void* keys = nullptr;
    const void* values = nullptr;
    const int32_t* mapping = nullptr;
    int64_t cells = 0;
    const int32_t* cold_mapping = nullptr;
    int64_t cold_cells = 0;
};
struct Plan {
    int batch, tokens, heads, kv_heads, max_seq, selected, width, value_width;
    int blocks, parts, tiles, iter_j, iter_z_gqa, threads, device;
    std::size_t shared_bytes, workspace_bytes;
    bool hot;
    int head_columns = 16;
    bool resolved_hot = false;
    int key_tile = 0;
    bool native_selection = false;
};

/** Plan the original stream-K launch on the current CUDA device. */
Plan plan(int batch, int tokens, int heads, int kv_heads, int max_seq,
          int selected, int width, int value_width, bool hot,
          bool native_selection = false, bool resolved_hot = false);

// The selected kernel reads complete device-dependent MMA key tiles.
int selection_alignment(int width, int value_width, int device);

/** Launch into caller-owned float output and workspace on the supplied stream. */
void launch(const Plan& plan, const void* query, const void* key, const void* value,
            const int32_t* indices, const void* mask, const float* sinks,
            const void* positions, const int64_t query_strides[4], bool query_half, bool query_contiguous,
            bool positions_i64, int causal_columns, Hot hot, float scale,
            float* output, void* workspace, std::size_t workspace_bytes,
            bool gated, cudaStream_t stream);
}

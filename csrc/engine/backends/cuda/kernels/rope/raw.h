#pragma once

#include "../numeric_type.h"
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <cstdint>

namespace tyloi::cuda::rope_raw {

struct Layout { int64_t batch, token, head, column; };

struct Projection {
    const void* input;
    const float* weight;
    const void* positions;
    const float* frequencies;
    const int32_t* selected_axes;
    void* output;
    int heads, tokens, width, pairs, axes, batches, maximum, threads;
    bool wide_positions;
    float eps;
    Layout layout;
    void* key_cache;
    const void* cache_positions;
    bool wide_cache_positions;
    int64_t capacity;
    const void* projected_value;
    bool float_value;
    Layout value_layout;
    void* value_cache;
    int64_t rows;
    const float2* rotations = nullptr;
    double attention_factor = 1.0;
};

struct Group { Projection entries[3]; int count; };

enum class PoolType { f32, f16, affine };
struct Pool {
    const void* input;
    const float* weight;
    const void* positions;
    const int64_t* cache_positions;
    const float* frequencies;
    const int32_t* selected_axes;
    void* output;
    int tokens, width, pool, maximum, rotary_maximum, pairs, axes;
    float eps;
    int64_t row_stride, column_stride;
    bool wide_positions, compact;
    double attention_factor;
    PoolType type;
};

void pool(const Pool&, cudaStream_t);
bool share_angles(NumericType, bool grouped, bool uniform, int threads, int64_t rows);
void angles(const Projection&, float2* output, int64_t count, cudaStream_t);
void normalized(const Projection&, NumericType, cudaStream_t);
void normalized_group(const Group&, int64_t rows, int threads, NumericType, cudaStream_t);
void cached(const void*, const int32_t*, const float*, const int32_t*, void*, int64_t rows,
    int heads, int tokens, int width, int pairs, int axes, int batches, int maximum,
    double attention_factor, NumericType, cudaStream_t);
void rope(const float*, const float*, float*, int rows, int tokens, int width, int dimension,
    int axes, int batches, int s0, int s1, int s2, float base, cudaStream_t);
void table(const float*, const int64_t*, const float*, const float*, float*, int rows,
    int tokens, int width, int dimension, int length, int axes, int batches,
    int s0, int s1, int s2, cudaStream_t);
void table_bf16(const void*, const int64_t*, const float*, const float*, void*,
    int rows, int tokens, int width, int dimension, int length, cudaStream_t);
void minicpm(const void*, const void*, const void*, const int64_t*, const int64_t*,
    const float*, const float*, void*, void*, void*, int batches, int query_heads,
    int key_heads, int length, int capacity, cudaStream_t);

} // namespace tyloi::cuda::rope_raw

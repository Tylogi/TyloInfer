#include "raw.h"
#include <cuda_runtime.h>

constexpr int ROPE_BD = 256;

__device__ int rope_axis_for_pair(int j, int s0, int s1, int s2)
{
    if (s0 <= 0 && s1 <= 0 && s2 <= 0) {
        return 0;
    }
    if (j < s0) {
        return 0;
    }
    if (j < s0 + s1) {
        return 1;
    }
    return 2;
}

__global__ void rope_kernel(const float* __restrict__ x, const float* __restrict__ pos,
                            float* __restrict__ out, int MT, int T, int D, int rotary_dim,
                            int pos_axes, int pos_batches,
                            int s0, int s1, int s2, float base)
{
    int mt = blockIdx.x;
    if (mt >= MT) {
        return;
    }
    int m = mt / T;
    int t = mt % T;
    int half = rotary_dim / 2;
    size_t base0 = ((size_t)m * T + t) * D;
    int tid = threadIdx.x;

    for (int i = tid; i < D; i += ROPE_BD) {
        out[base0 + i] = x[base0 + i];
    }
    __syncthreads();

    for (int j = tid; j < half; j += ROPE_BD) {
        float p;
        if (pos_batches > 0) {
            const int leading_rows = MT / T;
            const int rows_per_batch = leading_rows / pos_batches;
            const int batch = m / rows_per_batch;
            p = pos[(size_t)batch * T + t];
        } else {
            int axis = rope_axis_for_pair(j, s0, s1, s2);
            if (axis >= pos_axes) {
                axis = 0;
            }
            p = pos_axes == 1 ? pos[t] : pos[(size_t)axis * T + t];
        }
        float freq = powf(base, -2.0f * (float)j / (float)rotary_dim);
        float ang = p * freq;
        float cs = cosf(ang);
        float sn = sinf(ang);
        float x0 = x[base0 + j];
        float x1 = x[base0 + j + half];
        out[base0 + j] = x0 * cs - x1 * sn;
        out[base0 + j + half] = x1 * cs + x0 * sn;
    }
}

__global__ void rope_table_kernel(const float* __restrict__ x, const int64_t* __restrict__ pos,
                                  const float* __restrict__ cos, const float* __restrict__ sin,
                                  float* __restrict__ out, int MT, int T, int D, int rotary_dim,
                                  int table_len, int pos_axes, int pos_batches,
                                  int s0, int s1, int s2)
{
    int mt = blockIdx.x;
    if (mt >= MT) {
        return;
    }
    int m = mt / T;
    int t = mt % T;
    int half = rotary_dim / 2;
    size_t base0 = ((size_t)m * T + t) * D;
    int tid = threadIdx.x;

    for (int i = rotary_dim + tid; i < D; i += ROPE_BD) {
        out[base0 + i] = x[base0 + i];
    }

    for (int j = tid; j < half; j += ROPE_BD) {
        int64_t p;
        if (pos_batches > 0) {
            const int leading_rows = MT / T;
            const int rows_per_batch = leading_rows / pos_batches;
            const int batch = m / rows_per_batch;
            p = pos[(size_t)batch * T + t];
        } else {
            int axis = rope_axis_for_pair(j, s0, s1, s2);
            if (axis >= pos_axes) {
                axis = 0;
            }
            p = pos_axes == 1 ? pos[t] : pos[(size_t)axis * T + t];
        }
        if (p < 0) {
            p = 0;
        }
        if (p >= table_len) {
            p = table_len - 1;
        }
        float cs = cos[(size_t)p * half + j];
        float sn = sin[(size_t)p * half + j];
        float x0 = x[base0 + j];
        float x1 = x[base0 + j + half];
        out[base0 + j] = x0 * cs - x1 * sn;
        out[base0 + j + half] = x1 * cs + x0 * sn;
    }
}

namespace tyloi::cuda::rope_raw {
void rope(const float* x, const float* pos, float* out,
        int rows, int tokens, int width, int rotary_dim, int axes, int batches,
        int s0, int s1, int s2, float base, cudaStream_t stream) {
    rope_kernel<<<rows, ROPE_BD, 0, stream>>>(
        x, pos, out, rows, tokens, width, rotary_dim, axes, batches, s0, s1, s2, base);
}

void table(const float* x, const int64_t* pos, const float* cos, const float* sin, float* out,
        int rows, int tokens, int width, int rotary_dim, int table_len, int axes, int batches,
        int s0, int s1, int s2, cudaStream_t stream) {
    rope_table_kernel<<<rows, ROPE_BD, 0, stream>>>(
        x, pos, cos, sin, out, rows, tokens, width, rotary_dim, table_len,
        axes, batches, s0, s1, s2);
}
} // namespace tyloi::cuda::rope_raw

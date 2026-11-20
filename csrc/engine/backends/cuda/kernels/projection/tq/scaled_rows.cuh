#pragma once

#include "../gguf/prefill/scale.cuh"

template<bool Columns>
__global__ void tq_row_exponents(TrellisProjectionView weight, int first, std::int32_t* exponents) {
    const int row = first + blockIdx.x;
    const auto descriptor = __ldg(weight.descriptors + row);
    float maximum = 0;
    for (std::int64_t col = threadIdx.x * 16; col < weight.input_columns; col += blockDim.x * 16) {
        alignas(16) half values[16];
        tq_decode_tile_dispatch<Columns>(weight, descriptor, row, static_cast<int>(col), values);
        #pragma unroll
        for (int j = 0; j < 16; ++j)
            if (col + j < weight.input_columns) maximum = fmaxf(maximum, fabsf(__half2float(values[j])));
    }
    for (int step = 16; step; step >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, step));
    __shared__ float partial[4];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (!lane) partial[warp] = maximum;
    __syncthreads();
    if (warp == 0) {
        maximum = lane < 4 ? partial[lane] : 0;
        for (int step = 16; step; step >>= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, step));
        if (!lane) exponents[blockIdx.x] = prefill_unit_exponent(maximum, weight.input_columns);
    }
}

template<bool Columns>
__global__ void tq_scaled_rows(TrellisProjectionView weight, int first, const std::int32_t* exponents, half* output) {
    const auto col = (std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 16;
    if (col >= std::uint64_t(weight.input_columns)) return;
    const int row = first + blockIdx.y;
    const auto descriptor = __ldg(weight.descriptors + row);
    alignas(16) half values[16];
    tq_decode_tile_dispatch<Columns>(weight, descriptor, row, static_cast<int>(col), values);
    const int exponent = __ldg(exponents + blockIdx.y);
    auto* target = output + std::size_t(blockIdx.y) * weight.input_columns + col;
    #pragma unroll
    for (int j = 0; j < 16; ++j)
        if (col + j < std::uint64_t(weight.input_columns))
            target[j] = __float2half_rn(prefill_ldexp(__half2float(values[j]), -exponent));
}

int tq_decode_scaled_rows(TrellisProjectionView weight, int first, int count, std::int32_t* exponents,
        void* output, cudaStream_t stream) {
    if (!exponents || !output || !weight.words || !weight.descriptors || !weight.scales || !weight.relative ||
        weight.input_columns < 1 || weight.input_columns > weight.stored_columns || weight.stored_columns % 32 ||
        first < 0 || first > weight.rows || count < 1 || count > 65535 || count > weight.rows - first ||
        (!weight.columns && (weight.gslog < 4 || weight.gslog > 20))) return -1;
    const dim3 grid((std::uint64_t(weight.input_columns) + 511) / 512, count);
    if (weight.columns) tq_row_exponents<true><<<count, 128, 0, stream>>>(weight, first, exponents);
    else tq_row_exponents<false><<<count, 128, 0, stream>>>(weight, first, exponents);
    if (const auto error = cudaGetLastError(); error != cudaSuccess) return int(error);
    if (weight.columns) tq_scaled_rows<true><<<grid, 32, 0, stream>>>(weight, first, exponents, static_cast<half*>(output));
    else tq_scaled_rows<false><<<grid, 32, 0, stream>>>(weight, first, exponents, static_cast<half*>(output));
    return int(cudaGetLastError());
}

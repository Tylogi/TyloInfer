#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace mfq::cuda::gguf::w8a8 {

// Both operands use the same orthonormal 32-coordinate transform. Padding
// belongs to the transformed dot product, never to the caller's source rows.
std::int64_t padded_columns(std::int64_t columns);
void pack(const void* input, bool half_input, std::int64_t rows,
          std::int64_t columns, std::int64_t source_stride,
          std::int8_t* codes, float* scales, cudaStream_t stream);
void pack_gated(const void* input,const void* gate,bool half_input,bool silu,
                std::int64_t rows,std::int64_t columns,void* materialized,
                std::int8_t* codes,float* scales,cudaStream_t stream);

struct Plan {
    std::shared_ptr<void> implementation;
    std::size_t workspace_bytes = 0;
    explicit operator bool() const noexcept { return bool(implementation); }
};

// Plans retain host library descriptors only. All device scratch remains
// caller-owned, including the INT32 accumulator and the library workspace.
Plan prepare(std::int64_t tokens, std::int64_t rows, std::int64_t columns,
             std::size_t workspace_limit, cudaStream_t stream);
// Operand bases require 16-byte alignment; nonempty scratch requires 256.
void multiply(const Plan& plan, const std::int8_t* input,
              const std::int8_t* weight, std::int32_t* accumulator,
              void* workspace, std::size_t workspace_bytes,
              cudaStream_t stream);
void restore(const std::int32_t* accumulator, const float* input_scales,
             const float* weight_scales, void* output, bool half_output,
             std::int64_t tokens, std::int64_t rows,
             std::int64_t output_stride, cudaStream_t stream,
             const void* half_residual = nullptr);

}

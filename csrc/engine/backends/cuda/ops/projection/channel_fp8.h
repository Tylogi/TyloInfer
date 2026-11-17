#pragma once

#include "engine/weights/encoding/fp8/tensor.h"

struct CudaProfiler;

namespace tyloi::cuda {

// The prepared activation owns the source handle as well as its FP8 codes and
// per-token scales.  It is an execution input, not part of the weight record.
struct ChannelFp8Input {
    DenseTensor source;
    DenseTensor values;
    DenseTensor scales;
};

ChannelFp8Input prepare_channel_fp8_input(
    const DenseTensor& input, CudaProfiler* profiler = nullptr);

DenseTensor channel_fp8_projection(
    const ChannelFp8Tensor& weight,
    DenseTensor input,
    CudaProfiler* profiler = nullptr,
    const ChannelFp8Input* prepared = nullptr);

DenseTensor channel_fp8_rows(
    const ChannelFp8Tensor& weight, DenseTensor ids);

DenseTensor dequant_channel_fp8(const ChannelFp8Tensor& weight);

} // namespace tyloi::cuda

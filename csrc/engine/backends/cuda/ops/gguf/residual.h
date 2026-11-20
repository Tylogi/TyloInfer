#pragma once
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/tensor/encoding/ggml.h"

#include <cstdint>
#include <vector>

namespace tyloi::cuda::gguf {

DenseTensor residual_mix(
    const GgmlQuantizedTensor&, const DenseTensor& input,
    const DenseTensor& normalized, std::int64_t streams);

std::vector<DenseTensor> gated_residual_pre(
    const DenseTensor& input, const DenseTensor& norm,
    const GgmlQuantizedTensor& down, const GgmlQuantizedTensor& up, const GgmlQuantizedTensor& injection,
    std::int64_t streams, double eps,
    const DenseTensor& branch = {}, const DenseTensor& prior_injection = {});

} // namespace tyloi::cuda::gguf

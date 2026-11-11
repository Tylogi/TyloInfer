#pragma once
#include "engine/backends/cuda/native/tensor/backend.h"

// Cached causal windows use device-resident sequence lengths, including
// changing lengths during Graph replay. Empty means the MMA resources or
// operand geometry are unsupported; callers retain their ordinary kernel.
::tyloi::DenseTensor try_attention_mma_verify_cuda(
    ::tyloi::DenseTensor query,::tyloi::DenseTensor keys,
    ::tyloi::DenseTensor values,::tyloi::DenseTensor lengths,
    double scale,int64_t planned_length);

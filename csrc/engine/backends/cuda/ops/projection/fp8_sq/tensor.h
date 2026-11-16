#pragma once

#include "engine/weights/prepare/sq.h"
#include "engine/backends/cuda/kernels/projection/sq/fp8.h"

::tyloi::DenseTensor sq_matmul_cpu(
    const tyloi::Fp8SqTensor& weight,
    ::tyloi::DenseTensor input);

namespace tyloi::cuda {
::tyloi::DenseTensor sq_matmul(const tyloi::Fp8SqTensor& weight, ::tyloi::DenseTensor x);
}

::tyloi::DenseTensor dequant_fp8_sq(
    const tyloi::Fp8SqTensor& weight, bool fp32);

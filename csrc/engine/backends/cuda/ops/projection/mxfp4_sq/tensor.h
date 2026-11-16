#pragma once

#include "engine/weights/prepare/sq.h"
#include "engine/backends/cuda/kernels/projection/sq/mxfp4.h"

::tyloi::DenseTensor sq_matmul_cpu(
    const tyloi::Mxfp4SqTensor& weight,
    ::tyloi::DenseTensor input);

::tyloi::DenseTensor dequant_mxfp4_sq_cpu(
    const tyloi::Mxfp4SqTensor& weight);

namespace tyloi::cuda {
::tyloi::DenseTensor sq_matmul(const tyloi::Mxfp4SqTensor& weight, ::tyloi::DenseTensor x);
}

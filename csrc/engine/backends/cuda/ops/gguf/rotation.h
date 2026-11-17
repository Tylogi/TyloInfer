#pragma once

#include "engine/weights/encoding/gguf/tensor.h"

namespace tyloi::cuda::gguf {

DenseTensor apply_rotation(const GgufRotation&, DenseTensor, bool adjoint = false);
void apply_rotation_inplace(const GgufRotation&, DenseTensor, bool adjoint = false);

} // namespace tyloi::cuda::gguf

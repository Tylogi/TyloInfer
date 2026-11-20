#pragma once

#include "engine/tensor/encoding/ggml.h"

namespace tyloi::cuda::gguf {

DenseTensor apply_rotation(const GgmlInputRotation&, DenseTensor, bool adjoint = false);
void apply_rotation_inplace(const GgmlInputRotation&, DenseTensor, bool adjoint = false);

} // namespace tyloi::cuda::gguf

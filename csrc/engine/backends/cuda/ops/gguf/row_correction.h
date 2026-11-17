#pragma once

#include "engine/weights/encoding/gguf/tensor.h"

namespace tyloi::cuda::gguf {

DenseTensor add_correction(const GgufCorrection&, DenseTensor input, DenseTensor output,
    DenseTensor ids = {}, std::int64_t experts = 1, bool routed_input = false);
DenseTensor add_correction_embedding(const GgufCorrection&, DenseTensor ids, DenseTensor output);
DenseTensor dequantize_correction(const GgufCorrection&, int device);

} // namespace tyloi::cuda::gguf

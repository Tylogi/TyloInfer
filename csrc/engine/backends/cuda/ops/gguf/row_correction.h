#pragma once

#include "engine/tensor/encoding/ggml.h"

namespace tyloi::cuda::gguf {

DenseTensor add_correction(const GgmlRowCorrection&, DenseTensor input, DenseTensor output,
    DenseTensor ids = {}, std::int64_t experts = 1, bool routed_input = false);
DenseTensor add_correction_embedding(const GgmlRowCorrection&, DenseTensor ids, DenseTensor output);
DenseTensor dequantize_correction(const GgmlRowCorrection&, int device);

} // namespace tyloi::cuda::gguf

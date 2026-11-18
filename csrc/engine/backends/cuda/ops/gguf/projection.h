#pragma once

#include "engine/weights/encoding/gguf/tensor.h"
#include <cstdint>
#include <vector>

namespace tyloi::cuda::gguf {

struct ProjectionInput {
    DenseTensor native, interleaved, original;
    std::vector<std::int64_t> shape;
    std::int64_t tokens = 0;
    ScalarType dtype = kFloat32;
    bool aligned_m1 = false;
};

DenseTensor project(const GgufTensor&, DenseTensor, bool allow_mmq = true);
DenseTensor embedding(const GgufTensor&, DenseTensor);
DenseTensor dequantize(const GgufTensor&);
DenseTensor gated_residual(const GgufTensor&, const DenseTensor& gate,
    const DenseTensor& up, const DenseTensor& residual);
std::vector<DenseTensor> project_group(const std::vector<GgufTensor>&, DenseTensor);
bool same_input_basis(const GgufTensor&, const GgufTensor&);
bool original_input(const GgufTensor&);
DenseTensor project_q8(const GgufTensor&, const DenseTensor&, std::int64_t, ScalarType, bool);
bool position_major_q8(const GgufTensor&, std::int64_t);
bool m1_aligned(const std::vector<GgufTensor>&, std::int64_t);
DenseTensor canonical_packed(const GgufTensor&);
ProjectionInput prepare_input(const std::vector<GgufTensor>&, DenseTensor);
ProjectionInput prepare_rms_input(const std::vector<GgufTensor>&, const DenseTensor& input,
    const DenseTensor& norm, double eps, double offset, DenseTensor& normalized);
std::vector<DenseTensor> project_input(const ProjectionInput&, const std::vector<GgufTensor>&);

} // namespace tyloi::cuda::gguf

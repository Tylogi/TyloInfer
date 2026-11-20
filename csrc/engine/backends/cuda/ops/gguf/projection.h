#pragma once

#include "engine/tensor/encoding/ggml.h"
#include <cstdint>
#include <span>
#include <vector>

namespace tyloi::weights { struct ExpertRows; }

namespace tyloi::cuda::gguf {

struct ProjectionInput {
    DenseTensor native, interleaved, original;
    std::vector<std::int64_t> shape;
    std::int64_t tokens = 0;
    ScalarType dtype = kFloat32;
    bool aligned_m1 = false;
};

DenseTensor project(const GgmlQuantizedTensor&, DenseTensor, bool allow_mmq = true);
bool can_prefill_residual(const GgmlQuantizedTensor&, std::int64_t tokens, const DenseTensor& residual);
DenseTensor project_prefill_residual(const GgmlQuantizedTensor&, DenseTensor, const DenseTensor& residual);
void project_routed(std::span<const GgmlQuantizedTensor> images, const weights::ExpertRows&,
    std::size_t first, const DenseTensor& input, DenseTensor& output);
DenseTensor project_indexed(const GgmlQuantizedTensor& weight, const DenseTensor& addresses,
    const weights::ExpertRows& rows, const DenseTensor& input);
DenseTensor embedding(const GgmlQuantizedTensor&, DenseTensor);
DenseTensor dequantize(const GgmlQuantizedTensor&);
DenseTensor gated_residual(const GgmlQuantizedTensor&, const DenseTensor& gate,
    const DenseTensor& up, const DenseTensor& residual);
std::vector<DenseTensor> project_group(const std::vector<GgmlQuantizedTensor>&, DenseTensor);
bool same_input_basis(const GgmlQuantizedTensor&, const GgmlQuantizedTensor&);
bool original_input(const GgmlQuantizedTensor&);
DenseTensor project_q8(const GgmlQuantizedTensor&, const DenseTensor&, std::int64_t, ScalarType, bool);
bool position_major_q8(const GgmlQuantizedTensor&, std::int64_t);
bool m1_aligned(const std::vector<GgmlQuantizedTensor>&, std::int64_t);
DenseTensor canonical_packed(const GgmlQuantizedTensor&);
ProjectionInput prepare_input(const std::vector<GgmlQuantizedTensor>&, DenseTensor);
ProjectionInput prepare_rms_input(const std::vector<GgmlQuantizedTensor>&, const DenseTensor& input,
    const DenseTensor& norm, double eps, double offset, DenseTensor& normalized);
std::vector<DenseTensor> project_input(const ProjectionInput&, const std::vector<GgmlQuantizedTensor>&);

} // namespace tyloi::cuda::gguf

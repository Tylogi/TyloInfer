#pragma once
#include "engine/backends/cuda/native/tensor/backend.h"
#include "gated_residual.h"
struct GgufWeight;
// Empty asks the caller to retain its general projection+mix path.
::tyloi::DenseTensor try_gguf_residual_mix(const GgufWeight& weight,
    const ::tyloi::DenseTensor& input,const ::tyloi::DenseTensor& normalized,
    std::int64_t streams);
std::vector<::tyloi::DenseTensor> try_gguf_gated_residual_pre(
    const ::tyloi::DenseTensor& input,const ::tyloi::DenseTensor& norm,
    const GgufWeight& down,const GgufWeight& up,const GgufWeight& injection,
    std::int64_t streams,double eps,const ::tyloi::DenseTensor& branch={},
    const ::tyloi::DenseTensor& prior_injection={},const GatedResidualMarker& marker={});

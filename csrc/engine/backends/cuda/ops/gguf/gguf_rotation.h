#pragma once
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/models/source.h"
#include <memory>
#include <vector>

// Immutable activation basis of one loaded matrix. CPU signs remain available
// for host projections; the device copy follows the weight's device/lifetime.
struct GgufInputRotation {
    int block = 1;
    std::vector<std::int8_t> signs_cpu;
    ::tyloi::DenseTensor signs_device;
    bool equivalent(const GgufInputRotation&) const;
    ::tyloi::DenseTensor apply(::tyloi::DenseTensor input, bool adjoint = false) const;
    void apply_inplace(::tyloi::DenseTensor input, bool adjoint = false) const;
    void apply_cpu(float* values, std::int64_t rows, std::int64_t width,
        bool half = false, bool adjoint = false) const;
};
bool same_gguf_rotation(const std::shared_ptr<const GgufInputRotation>&,
    const std::shared_ptr<const GgufInputRotation>&);
std::shared_ptr<const GgufInputRotation> load_gguf_input_rotation(const tyloi::ModelSource&,
    const std::string& name, std::int64_t source_width, std::int64_t first, std::int64_t last,
    int device, bool cuda);

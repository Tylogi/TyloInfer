#pragma once
#include "engine/backends/cuda/native/tensor/backend.h"
#include <optional>

// One BF16 scalar weight row and Half inputs. The dot follows the dense
// adapter's BF16 rounding, with FP16 activation/output boundaries. Unsupported
// views return nullopt so the caller keeps its ordinary dense projection.
std::optional<::tyloi::DenseTensor> try_shared_gate_sigmoid_cuda(
    const ::tyloi::DenseTensor& input,
    const ::tyloi::DenseTensor& weight);

// FP32 scalar weights: native FP32 dot/sigmoid and expert-output scaling in
// one launch. Keep the ordinary FP32 result; no weight/input requantization.
std::optional<::tyloi::DenseTensor> try_shared_gate_multiply_cuda(
    const ::tyloi::DenseTensor& input,
    const ::tyloi::DenseTensor& weight,
    const ::tyloi::DenseTensor& expert_output);

struct NintWeight;
std::optional<::tyloi::DenseTensor> try_nint_shared_gate_sigmoid_cuda(
    const ::tyloi::DenseTensor& input,const NintWeight& weight,
    const ::tyloi::DenseTensor& quantized={},const ::tyloi::DenseTensor& scales={});

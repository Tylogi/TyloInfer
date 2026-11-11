#pragma once

#include "engine/models/minicpmo45/model.h"
#include "engine/models/minicpmo45/state.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/executor.h"
#endif
#include "engine/models/input.h"

namespace tyloi::cuda::minicpmo45 {

#ifndef TYLOINFER_NATIVE_CUDA
DenseTensor embed(const Model&, Executor&, Collectives&, int primary, DenseTensor ids,
             const Executor::Mode& = {});
DenseTensor normalize(const Model&, Executor&, DenseTensor hidden, int64_t batch, int64_t tokens);
DenseTensor logits(const Model&, Executor&, DenseTensor hidden, bool capped = true,
              const Executor::Mode& = {});
DenseTensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, tyloi::models::Input);
#endif

} // namespace tyloi::cuda::minicpmo45

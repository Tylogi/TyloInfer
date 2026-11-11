#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/qwen35/model.h"
#include "engine/models/qwen35/state.h"

namespace tyloi::cuda::qwen35 {

::tyloi::Tensor embed(const Model&, Executor&, Collectives&, int primary, const ::tyloi::Tensor& ids,
                      const Executor::Mode& = {});
::tyloi::Tensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, tyloi::models::Input);
::tyloi::Tensor logits(const Model&, Executor&, const ::tyloi::Tensor& hidden, bool capped = true,
                       const Executor::Mode& = {});
bool supports_speculation(const Model&) noexcept;
void replay(const Model&, State&, Executor&, std::int64_t accepted = 0);

} // namespace tyloi::cuda::qwen35

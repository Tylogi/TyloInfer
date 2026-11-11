#include "engine/models/qwen35/forward_cuda.h"

#include <utility>

namespace tyloi::cuda::qwen35 {

::tyloi::Tensor embed(const Model& model, Executor& executor, Collectives&, int primary,
                      const ::tyloi::Tensor& ids, const Executor::Mode& mode) {
    auto scope = executor.enter(mfq_current_cuda_device(), mode.ffn);
    return model.embedding(ids, primary);
}

::tyloi::Tensor logits(const Model& model, Executor& executor, const ::tyloi::Tensor& hidden,
                       bool capped, const Executor::Mode& mode) {
    auto scope = executor.enter(mfq_current_cuda_device(), mode.ffn);
    return model.logits(hidden, capped);
}

::tyloi::Tensor forward(const Model& model, State& state, Executor& executor, Collectives&,
                        const Executor::Mode& mode, tyloi::models::Input input) {
    auto scope = executor.enter(mfq_current_cuda_device(), mode.ffn);
    tyloi::models::ForwardOptions options;
    options.kv_capacity = mode.kv_capacity;
    options.trace = mode.trace;
    options.trace_layer = mode.trace_layer;
    options.primary = mode.primary;
    options.serial_experts = mode.ffn.batch_serial;
    return model.forward(std::move(input), state, options);
}

bool supports_speculation(const Model& model) noexcept { return model.supports_speculation(); }

void replay(const Model& model, State& state, Executor& executor, std::int64_t accepted) {
    auto scope = executor.enter(mfq_current_cuda_device(), FfnMode{});
    model.replay(state, accepted);
}

} // namespace tyloi::cuda::qwen35

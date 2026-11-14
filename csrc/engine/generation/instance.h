#pragma once

#include "engine/text.h"
#include "engine/generation/generation.h"

namespace tyloi::cuda::engine_detail {

template <class M, class S, class P>
tyloi::StepSequence<tyloi::models::Prepared> Text<M, S, P>::prepare(InferenceRequest& input, State& state) {
    if constexpr (std::is_same_v<Parts, std::unique_ptr<tyloi::cuda::minicpmo45::Components>>) {
        if (!components) throw std::invalid_argument("model has no multimodal component");
        return components->prepare(input.prompt, *input.vision);
    } else if constexpr (std::is_same_v<Model, qwen4_exp::Model> ||
                         std::is_same_v<Model, nemotron_h::Model>) {
        return components.prepare(language, images,
#ifndef TYLOINFER_NATIVE_CUDA
        root_executor.projector(root_executor.mode().ffn, model_parallel_primary_device(load_config)),
        root_executor.collectives(),
#endif
        load_config.layer_placement.primary_device(),
#ifndef TYLOINFER_NATIVE_CUDA
        executor.get(),
#endif
        input.prompt, *input.vision);
    }
#ifdef TYLOINFER_NATIVE_CUDA
    else if constexpr (requires { static_cast<const tyloi::models::CausalModel&>(language); }) {
        const tyloi::models::CausalModel& causal = language;
        return causal.prepare(input.prompt, *input.vision, state);
    }
#endif
    else throw std::invalid_argument("model has no multimodal component");
}

template <class M, class S, class P>
Generation Text<M, S, P>::generate_text(InferenceRequest& input, InferenceOutput& output, State& state,
                                       std::optional<Prepared> prepared, const RequestId& id, bool batched) {
    return internal::generate(
        language, state, executor.get(), root_executor.collectives(),
        root_executor.forward_mode(load_config.layer_placement.primary_device()), root_executor.profiler(), weights, *budget,
        root_executor.mode().ffn.serial, load_config.report_cuda_memory,
        load_config.dsv4_cpu_offload_layers.empty() && load_config.dense_cpu_layer_count == 0 &&
            model_parallel_cuda_graph_enabled(load_config, root_executor, weights),
        [this] {
            drain_cuda(root_executor, weights, offload.get(), load_config.layer_placement,
                       load_config.tensor_parallel, load_config.expert_parallel);
        },
        info, source.get(), graph, cache, config, input, output,
        mtp_state,
        batched ? nullptr : component_predictor(components), std::move(prepared),
        batched ? batching.get() : nullptr, id, predictor_execution.get());
}

template <class M, class S, class P>
Generation Text<M, S, P>::generate(const RequestId& id, InferenceRequest& input, InferenceOutput& output, bool batched) {
    if (batched) {
        graph.invalidate();
        if constexpr (std::is_same_v<Model,qwen4_exp::Model>) {
            // Exclusive MTP generation has ended before a batch starts.
            // Its graph consumers must retire before idle batch buffers.
            qwen4_release_decode_graphs(language,state);
            if (components.mtp) components.mtp->reset(predictor_state(), 1);
        }
    }
    return generate_prepared(std::ref(*this), state, *executor, input, output, id, batched);
}

} // namespace tyloi::cuda::engine_detail

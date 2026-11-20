#include "engine/models/deepseek_v41/ops.h"
#include "engine/models/deepseek_v41/load.h"
#include "engine/models/deepseek_v41/vision.h"

#include "engine/models/common/load.h"
#include "engine/models/load.h"
#include "engine/weights/loader.h"
#include "engine/models/source/model_config.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/common/mtp.h"
#include "engine/models/deepseek_v41/components.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>


namespace tyloi::cuda::deepseek_v41 {


void read(Model& model, std::shared_ptr<const tyloi::ModelSource> model_source,
          std::string_view definition, std::int64_t context_size) {
    model.source = std::move(model_source);
    const auto& source = *model.source;
    ::tyloi::models::validate_model_source(source);
    model.graph = tyloi::models::graph(source);
    if (!(Model::accepts_backbone(model.graph.backbone)))
        throw std::runtime_error("model backbone does not match the requested causal model");
    if (const auto* vision = model.graph.component("vision")) {
        if (!(vision->tensor_root == "vision" && vision->implementation == "deepseek_v41_vision" &&
            vision->input_contract == "deepseek_v41_vision.v1" &&
            vision->position_policy == "deepseek_v41_positions"))
            throw std::runtime_error("unsupported DeepSeek-V4.1 vision component declaration");
    }
    if (const auto* predictor = model.graph.component("predictor")) {
        if (!(predictor->tensor_root == "predictor" && predictor->implementation == "deepseek_v41_dspark" &&
            predictor->input_contract.empty() && predictor->position_policy.empty()))
            throw std::runtime_error("unsupported DeepSeek-V4.1 predictor component declaration");
    }
    model.load_config(definition);
    if (!(model.num_hidden_layers() == model.graph.topology.text_layers))
        throw std::runtime_error("model graph/config text-layer topology mismatch");
    if (context_size > 0) {
        if (!(context_size <= model.max_position_embeddings()))
            throw std::runtime_error("--ctx-size exceeds max_position_embeddings");
        model.config.max_position_embeddings = context_size;
        model.metadata.max_position_embeddings = context_size;
    }
}

namespace {
void load_weights(Model& model, const tyloi::weights::Loader& tensors,
                  LayerPlacementConfig& placement, std::size_t cache_rows, bool load_blocks) {
    constexpr auto embedding = "model.token_embedding.weight";
    constexpr auto output = "model.output.weight";
    const bool independent_output = !model.tie_word_embeddings() && tensors.has(output);
    model.embed = tensors.embedding(embedding, independent_output);
    model.output_norm = tensors.tensor("model.output_norm.weight");
    model.lm_head = model.tie_word_embeddings() || !tensors.has(output)
        ? (tensors.placement.tensor.enabled() ? tensors.weight(embedding, TensorParallelAxis::Output) : model.embed)
        : tensors.weight(output, TensorParallelAxis::Output);
    if (load_blocks) {
        model.tables.engram_parameters = deepseek_v41_runtime::EngramHashState::load_parameters(tensors.source, model.config);
        model.blocks.reserve(static_cast<std::size_t>(model.num_hidden_layers()));
        for (int layer = 0; layer < model.num_hidden_layers(); ++layer) {
            const auto type = std::string(model.metadata.layer_types.at(static_cast<std::size_t>(layer)));
            const int device = placement.device_for_layer(layer);
            const bool cpu_offloaded = false;
            std::cerr << "loading layer " << layer << ' ' << type << ' '
                      << (cpu_offloaded ? "CPU" : "CUDA") << std::endl;
            auto placed = tensors.for_layer(cpu_offloaded ? ::tyloi::Device(::tyloi::kCPU) : ::tyloi::Device(::tyloi::kCUDA, device));
            if (!(type == "deepseek_v41" && model.tables.engram_parameters))
                throw std::runtime_error("invalid DeepSeek-V4.1 block loader state");
            auto block = deepseek_v41_runtime::load_block(placed,
                layer, model.config, model.tables, cache_rows);
            block->cuda_device = device;
            block->cpu_offloaded = cpu_offloaded;
            model.blocks.push_back(std::move(block));
        }
    }
}
} // namespace

void load(std::unique_ptr<tyloi::models::CausalModel>& owner,
        std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
        const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
        const tyloi::models::LoadOptions& options) {
    owner = std::make_unique<Model>();
    auto* model = static_cast<Model*>(owner.get());
    read(*model, std::move(source), definition, options.context);
    if (weights.placement.tensor.enabled() || placement.enabled() || options.gpu_layers >= 0)
        throw std::runtime_error("DeepSeek-V4.1 native CUDA currently supports single-device dense "
                                 "placement or expert parallelism");
    if (!options.cpu_offload_layers.empty())
        throw std::runtime_error("DeepSeek-V4.1 does not support CPU layer offloading");
    if (options.expert_parallel && model->metadata.num_experts <= 0)
        throw std::runtime_error("--expert-parallel requires a model with routed experts");
    placement.prepare(model->num_hidden_layers());
    model->primary_device = placement.primary_device();
    model->needs_host_tokens = options.host_tokens;
    load_weights(*model, weights, placement, options.engram_cache_rows, true);
    model->mtp = load_components(*model, weights, options.optional_components);
}

} // namespace tyloi::cuda::deepseek_v41

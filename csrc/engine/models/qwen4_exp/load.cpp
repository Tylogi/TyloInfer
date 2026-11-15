#include "engine/models/plan.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/qwen4_exp/load.h"
#include "engine/models/qwen4_exp/components.h"
#include "engine/models/qwen4_exp/layer.h"
#include "engine/models/qwen4_exp/causal_lm.h"
#include "engine/weights/loader.h"
#include "engine/weights/source/read.h"
#include "engine/weights/group.h"
#include "engine/models/qwen4_exp/predictor.h"
#include "engine/ops/ffn.h"
#include "engine/ops/gdn.h"
#include "engine/ops/qsa.h"
#include "engine/ops/residual.h"

#include "engine/models/load.h"
#include "engine/models/common/check.h"
#include "engine/models/common/load.h"
#include "engine/models/source/model_config.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/source.h"
#include "engine/weights/cache.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace tyloi::models::qwen4_exp {

using Config = tyloi::models::qwen4_exp::Config;

struct LayerLoader : tyloi::weights::Loader {
    using PleWeights = qwen4_exp::PleWeights;
    const tyloi::ops::YarnScaling& yarn;
    const tyloi::state::KvQuantization& kv_quantization;
    using Embedding = ::tyloi::Tensor;

    auto residual(::tyloi::Tensor norm, ::tyloi::Tensor down, ::tyloi::Tensor up,
                  ::tyloi::Tensor injection, const Config& c, bool final) const {
        return tyloi::ops::Residual(
            {std::move(norm), std::move(down), std::move(up), std::move(injection)},
            {c.hidden, c.streams, c.eps, final, placement.target,
                placement.target.is_metal() && source.metadata().contains("gguf_metadata")});
    }
    static auto final_mixer(tyloi::ops::Residual value) {
        return std::make_unique<tyloi::ops::Residual>(std::move(value));
    }
    auto gdn_weights(const std::string &p) const {
        const bool independent = has(p + ".query.weight") || has(p + ".key.weight") || has(p + ".value.weight");
        TYLOINFER_RUNTIME_CHECK(!independent || !has(p + ".qkv.weight"), "ambiguous Qwen4 GDN input projections");
        tyloi::ops::GdnWeights w;
        if(independent)for(const auto* name:{"query","key","value"})
            w.input_projection.push_back(weight(p+"."+name+".weight"));
        else w.input_projection.push_back(weight(p+".qkv.weight"));
        w.input_projection.push_back(weight(p+".gate.weight"));
        w.input_projection.push_back(weight(p+".alpha.weight"));
        w.input_projection.push_back(weight(p+".beta.weight"));
        w.qkv_count=independent?3:1;
        w.output=weight(p+".output.weight");
        w.conv=dense(p+".conv.weight");
        w.dt_bias=dense(p+".dt_bias");
        w.a_log=dense(p+".a");
        w.norm=dense(p+".norm.weight");
        return w;
    }
    auto qsa_weights(const std::string &p) const {
        tyloi::ops::QsaWeights w;
        for(const auto* name:{"query","key","value"})
            w.input_projection.push_back(weight(p+"."+name+".weight"));
        w.input_projection.push_back(weight(p+".indexer.query_key.weight"));
        w.output=weight(p+".output.weight");
        w.query_norm=dense(p+".query_norm.weight");
        w.key_norm=dense(p+".key_norm.weight");
        w.index_query_norm=dense(p+".indexer.query_norm.weight");
        w.index_key_norm=dense(p+".indexer.key_norm.weight");
        if(has(p+".gate.weight")) {
            w.input_projection.push_back(weight(p+".gate.weight"));
            w.has_gate=true;
        }
        return w;
    }
    auto gdn(tyloi::ops::GdnWeights weights, const Config& c) const {
        weights.tiled_heads = c.linear_tiled_heads;
        weights.input_projection = tyloi::weights::pack_projections(
            std::move(weights.input_projection), options.grouped, placement.target.is_cpu());
        return std::make_unique<tyloi::ops::Gdn>(std::move(weights), c.key_heads, c.value_heads,
            c.linear_width, c.kernel, c.eps, c.silu_gate, placement.target);
    }
    auto qsa(tyloi::ops::QsaWeights weights, const Config& c) const {
        weights.input_projection = tyloi::weights::pack_projections(
            std::move(weights.input_projection), options.grouped, placement.target.is_cpu());
        for (auto* norm : {&weights.query_norm, &weights.key_norm, &weights.index_query_norm, &weights.index_key_norm})
            if (norm->defined()) *norm = tyloi::ops::contiguous(tyloi::ops::cast(*norm, tyloi::kFloat32));
        tyloi::ops::QsaConfig geometry{c.heads, c.kv_heads, c.width, c.index_heads, c.index_width,
            c.pool, c.budget, c.maximum, c.eps, c.rotary, c.rope_base, c.sections, c.interleaved,
            yarn, placement.target, kv_quantization};
        return std::make_unique<tyloi::ops::Qsa>(std::move(weights), std::move(geometry));
    }
    auto moe(std::vector<tyloi::Tensor> gate_up, tyloi::Tensor down, tyloi::Tensor router,
             tyloi::Tensor shared_gate, tyloi::Tensor sg, tyloi::Tensor su, tyloi::Tensor sd,
             const Config& c, int layer) const {
        tyloi::ops::FfnWeights f;
        f.is_moe = true;
        f.gate_up = std::move(gate_up);
        f.down = std::move(down);
        f.moe_router = std::move(router);
        f.moe_shared_gate = std::move(shared_gate);
        f.shared = std::make_unique<tyloi::ops::FfnWeights>();
        f.shared->gate_up = {std::move(sg), std::move(su)};
        f.shared->down = std::move(sd);
        const auto routing = c.routing();
        f.moe_top_k = static_cast<int>(c.topk);
        f.moe_use_sigmoid = routing.activation == tyloi::ops::RouterActivation::sigmoid;
        f.moe_use_sqrt_softplus = routing.activation == tyloi::ops::RouterActivation::sqrt_softplus;
        f.moe_normalize = routing.normalize;
        f.moe_delayed_softmax = routing.delayed_softmax;
        f.moe_router_scale = routing.scale;
        f.moe_layer = layer;
        f.qwen_hidden = c.hidden;
        f.qwen_moe_width = c.moe_width;
        f.target = placement.target;
        return tyloi::ops::qwen4_ffn_weights(std::move(f));
    }
    auto embedding(const std::string &name) const {
        auto table = rows(name);
        const std::array<int64_t,2> shape{table.size(0), table.size(1)};
        return std::pair{std::move(table), shape};
    }
    std::vector<int64_t> integers(const std::string &name) const {
        const auto &type = require_tensor(source, name).dtype;
        TYLOINFER_RUNTIME_CHECK(type == "I64" || type == "I32",
                          "Qwen4 PLE metadata must be a dense integer array: ", name);
        auto value = host(name);
        TYLOINFER_RUNTIME_CHECK(value.dim() == 1, "Qwen4 PLE metadata must be a vector");
        return tyloi::ops::read_integers(value);
    }
    static auto ple(std::vector<Embedding> shards, int64_t rows, int64_t width, const Config &c,
                    std::vector<int64_t> multipliers, std::vector<int64_t> offsets,
                    std::vector<int64_t> sizes, PleWeights weights) {
        NgramEmbedding embedding(std::move(shards), rows, width, c.ngram, c.ngram_heads, c.eos,
                                 std::move(multipliers), std::move(offsets), std::move(sizes));
        return std::make_unique<Ple>(std::move(embedding), std::move(weights), c.hidden, c.streams,
                                     c.ngram, c.eps);
    }
    std::unique_ptr<Layer> layer(const Config &c, int layer, bool predictor);
};

std::unique_ptr<Layer> LayerLoader::layer(const Config &c, int layer, bool predictor) {
    auto result = std::make_unique<Layer>();
    result->device = placement.target;
    LayerLoader loader{*this, yarn, kv_quantization};
    loader.role = "qwen4_exp";
    tyloi::models::qwen4_exp::load_layer(*result, loader, c, layer, predictor);
    return result;
}

std::optional<Predictor> Predictor::load_if_present(const tyloi::weights::Loader& weights,
                                                      const tyloi::models::LoadOptions& options,
                                                      const tyloi::models::qwen4_exp::Config &main,
                                                      ::tyloi::Tensor embedding, ::tyloi::Tensor output,
                                                      double softcap) {
    Predictor result;
    result.embedding = std::move(embedding);
    result.output = std::move(output);
    result.softcap = softcap;
    result.device = weights.placement.target;
    LayerLoader loader{weights, options.yarn, options.kv_quantization};
    loader.role = "qwen4_exp";
    if (!tyloi::models::qwen4_exp::load_predictor(result, loader, main))
        return std::nullopt;
    return result;
}

using tyloi::models::require_model;

void read(Model& model, std::shared_ptr<const tyloi::ModelSource> model_source,
          std::string_view definition, std::int64_t context_size) {
    model.source = std::move(model_source);
    const auto& source = *model.source;
    ::tyloi::models::validate_model_source(source);
    model.graph = tyloi::models::graph(source);
    model.plan = ::tyloi::models::model_plan(model.graph);
    require_model(Model::accepts_backbone(model.graph.backbone),
                  "model backbone does not match the requested causal model");
    model.load_config(definition);
    require_model(model.num_hidden_layers() == model.graph.topology.text_layers,
                  "model graph/config text-layer topology mismatch");
    if (context_size > 0) {
        require_model(context_size <= model.max_position_embeddings(),
                      "--ctx-size exceeds max_position_embeddings");
        model.config.maximum = context_size;
        model.metadata.max_position_embeddings = context_size;
    }
}

void load(Model& model, const tyloi::weights::Loader& tensors, LayerPlacementConfig& placement,
          const tyloi::models::LoadOptions& options, bool load_layers) {
    validate_native_placement(tensors.placement.tensor, placement, options.gpu_layers);
    if (placement.devices.empty()) placement.devices = {tensors.placement.target.index};
    if (placement.primary_device() != tensors.placement.target.index)
        throw std::invalid_argument("Qwen4 weight and layer primary devices differ");
    if (options.expert_parallel && model.metadata.num_experts <= 0)
        throw std::runtime_error("--expert-parallel requires a model with routed experts");
    placement.prepare(model.num_hidden_layers());
    const int dense_cpu_layer_count = options.gpu_layers >= 0
        ? static_cast<int>(std::max<std::int64_t>(model.num_hidden_layers() - options.gpu_layers, 0)) : 0;
    if (dense_cpu_layer_count > 0) {
        if (tensors.placement.tensor.enabled() || options.expert_parallel || placement.enabled())
            throw std::runtime_error("--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
        std::cerr << "dense_layer_placement cpu=0-" << (dense_cpu_layer_count - 1)
                  << " gpu=" << dense_cpu_layer_count << '-' << (model.num_hidden_layers() - 1)
                  << std::endl;
    }
    tyloi::ops::DeviceScope model_guard(tensors.placement.target);
    if (model.plan.vision == ::tyloi::models::VisionAdapter::grid_vit && dense_cpu_layer_count > 0)
        throw std::runtime_error("CUDA grid-Vision currently requires GPU-resident text layers");
    model.device = tensors.placement.target;
    constexpr auto embedding = "model.token_embedding.weight";
    constexpr auto output = "model.output.weight";
    const bool independent = !model.tie_word_embeddings() && tensors.has(output);
    model.embed = tensors.embedding(embedding, independent);
    LayerLoader loader{tensors, options.yarn, options.kv_quantization};
    loader.role = "qwen4_exp";
    model.final_mixer = loader.final_mixer(
        tyloi::models::qwen4_exp::load_residual(loader, model.config, "model.mhc.pre", false));
    model.lm_head = independent ? tensors.weight(output, TensorParallelAxis::Output)
        : tensors.placement.tensor.enabled() ? tensors.weight(embedding, TensorParallelAxis::Output)
                                              : model.embed;
    if (load_layers) {
        model.layers.reserve(static_cast<std::size_t>(model.num_hidden_layers()));
        for (int layer = 0; layer < model.num_hidden_layers(); ++layer) {
            const Device device(tensors.placement.target.type, placement.device_for_layer(layer));
            const bool cpu = layer < dense_cpu_layer_count;
            tyloi::ops::DeviceScope layer_guard(cpu ? Device(kCPU) : device);
            std::cerr << "loading layer " << layer << ' '
                      << model.metadata.layer_types.at(static_cast<std::size_t>(layer)) << ' '
                      << (cpu ? "CPU" : device.is_cuda() ? "CUDA" : "Metal") << std::endl;
            LayerLoader placed{tensors.for_layer(cpu ? Device(kCPU) : device), options.yarn, options.kv_quantization};
            auto block = placed.layer(model.config, layer, false);
            model.layers.push_back(std::move(block));
        }
    }
}

void load(std::unique_ptr<tyloi::models::CausalModel>& owner,
          std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
          const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
          const tyloi::models::LoadOptions& options) {
    if (!options.cpu_offload_layers.empty())
        throw std::invalid_argument("Qwen4 does not support --cpu-offload-layers");
    owner = std::make_unique<Model>();
    auto& model = static_cast<Model&>(*owner);
    model.host_tokens = options.host_tokens;
    read(model, std::move(source), definition, options.context);
    load(model, weights, placement, options, true);
    model.parts = std::make_unique<Components>();
    load_components(*model.parts, model, weights, options.optional_components, options);
}

} // namespace tyloi::models::qwen4_exp

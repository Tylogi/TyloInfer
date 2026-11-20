#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/qwen4_exp/load.h"
#include "engine/models/qwen4_exp/components.h"
#include "engine/models/qwen4_exp/layer.h"
#include "engine/weights/loader.h"
#include "engine/weights/memory.h"
#include "engine/weights/source/read.h"
#include "engine/weights/group.h"
#include "engine/weights/moe/experts.h"
#include "engine/models/qwen4_exp/predictor.h"
#include "engine/ops/ffn.h"
#include "engine/ops/gdn.h"
#include "engine/ops/qsa.h"
#include "engine/ops/residual.h"

#include "engine/models/load.h"
#include "engine/models/common/load.h"
#include "engine/models/source/model_config.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/source/model.h"
#include "engine/weights/cache/cache.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::models::qwen4_exp {
namespace {

struct LayerLoader : tyloi::weights::Loader {
    const tyloi::ops::YarnScaling& yarn;
    const tyloi::state::KvQuantization& kv_quantization;

    LayerLoader(const weights::Loader& weights, const ops::YarnScaling& yarn,
                const state::KvQuantization& kv_quantization)
        : Loader(weights), yarn(yarn), kv_quantization(kv_quantization) {
        role = "qwen4_exp";
    }

    ops::Residual residual(const std::string& root, const Config& c, bool final = false) const {
        ops::ResidualWeights values;
        if (c.streams == 1) {
            values.norm = norm(root + (final ? ".output_norm.weight" : ".norm.weight"), c.hidden);
        } else {
            const auto prefix = root + ".mhc.pre.";
            values.norm = ops::contiguous(ops::cast(dense(prefix + "norm.weight"), kFloat32));
            values.down = weight(prefix + "down.weight");
            values.up = weight(prefix + "up.weight");
            if (!final) values.injection = weight(root + ".mhc.post.inject.weight");
        }
        return ops::Residual(std::move(values),
            {c.hidden, c.streams, c.eps, final, placement.target,
                placement.target.is_metal() && source.metadata().contains("gguf_metadata"),
                c.norm_weight_offset});
    }
    std::unique_ptr<ops::Gdn> gdn(const std::string& p, const Config& c) const {
        const bool independent = has(p + ".query.weight") || has(p + ".key.weight") || has(p + ".value.weight");
        if (independent && has(p + ".qkv.weight"))
            throw std::runtime_error("ambiguous Qwen4 GDN input projections");
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
        w.tiled_heads = c.linear_tiled_heads;
        w.input_projection = weights::pack_projections(
            std::move(w.input_projection), options.grouped, placement.target.is_cpu());
        return std::make_unique<ops::Gdn>(std::move(w), c.key_heads, c.value_heads,
            c.linear_width, c.kernel, c.eps, c.silu_gate, placement.target);
    }
    std::unique_ptr<ops::Qsa> qsa(const std::string& p, const Config& c) const {
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
        w.input_projection = weights::pack_projections(
            std::move(w.input_projection), options.grouped, placement.target.is_cpu());
        for (auto* norm : {&w.query_norm, &w.key_norm, &w.index_query_norm, &w.index_key_norm})
            if (norm->defined()) *norm = tyloi::ops::contiguous(tyloi::ops::cast(*norm, tyloi::kFloat32));
        tyloi::ops::QsaConfig geometry{c.heads, c.kv_heads, c.width, c.index_heads, c.index_width,
            c.pool, c.budget, c.maximum, c.eps, c.rotary, c.rope_base, c.sections, c.interleaved,
            yarn, placement.target, kv_quantization};
        return std::make_unique<ops::Qsa>(std::move(w), std::move(geometry));
    }
    Ffn ffn(const std::string& root, const Config& c, int layer) const {
        if (!c.experts) {
            Mlp result;
            result.down = weight(root + ".down.weight", TensorParallelAxis::Input);
            if (!(result.down.dim() == 2 && result.down.size(0) == c.hidden &&
                result.down.size(1) == c.moe_width))
                throw std::runtime_error("dense Qwen4 FFN down projection disagrees with its config");
            result.gate_up = gate_up({root + ".gate.weight", root + ".up.weight"}, result.down);
            if (!(result.gate_up.size() == 2 || (result.gate_up.size() == 1 &&
                c.moe_width <= std::numeric_limits<std::int64_t>::max() / 2)))
                throw std::runtime_error("invalid dense Qwen4 FFN projection group");
            const auto rows = result.gate_up.size() == 1 ? c.moe_width * 2 : c.moe_width;
            for (const auto& projection : result.gate_up)
                if (!(projection.dim() == 2 && projection.size(0) == rows && projection.size(1) == c.hidden))
                    throw std::runtime_error("dense Qwen4 FFN gate/up projections disagree with its config");
            return result;
        }
        tyloi::models::Moe f;
        f.gate_up = routed_gate_up(root, layer, c.experts, c.moe_width, c.hidden);
        f.down = routed(root + ".experts.down.weight", layer, c.experts, c.hidden, c.moe_width);
        f.router = weight(root + ".router.weight");
        f.shared_gate = weight(root + ".shared_expert.router.weight");
        f.shared.gate_up = {weight(root + ".shared_expert.gate.weight"), weight(root + ".shared_expert.up.weight")};
        f.shared.down = weight(root + ".shared_expert.down.weight");
        f.routing = c.routing();
        f.top_k = c.topk;
        f.layer = layer;
        if (!(f.router && f.router.dim() == 2 && f.router.size(0) == c.experts &&
            f.router.size(1) == c.hidden &&
            f.top_k > 0 && f.top_k <= f.router.size(0) && f.shared_gate &&
            f.shared.gate_up[0] && f.shared.gate_up[1] && f.shared.down &&
            f.down && f.down.dim() == 3 && f.down.size(0) == c.experts &&
            f.down.size(1) == c.hidden && f.down.size(2) == c.moe_width))
            throw std::runtime_error("Qwen4 shared and routed expert projections are incomplete");
        return f;
    }
    std::vector<int64_t> integers(const std::string &name) const {
        const auto &type = require_tensor(source, name).dtype;
        if (type != "I64" && type != "I32")
            throw std::runtime_error("Qwen4 PLE metadata must be a dense integer array: " + name);
        auto value = host(name);
        if (value.dim() != 1)
            throw std::runtime_error("Qwen4 PLE metadata must be a vector");
        return tyloi::ops::read_integers(value);
    }
    std::unique_ptr<Ple> ple(const std::string& root, const Config& c) const {
        std::vector<Tensor> shards;
        const auto width = c.hidden / ((c.ngram - 1) * c.ngram_heads);
        std::int64_t row_count = 0;
        for (std::int64_t i = 0; i < c.shards; ++i) {
            auto table = rows(root + ".ngram.shard." + std::to_string(i) + ".weight");
            if (!(table.size(1) == width && table.size(0) > 0 && (!row_count || table.size(0) == row_count)))
                throw std::runtime_error("Qwen4 PLE embedding shard dimensions disagree");
            row_count = table.size(0);
            shards.push_back(std::move(table));
        }
        auto multipliers = integers(root + ".ngram.layer_multipliers");
        auto offsets = integers(root + ".ngram.head_offsets");
        auto sizes = integers(root + ".ngram.head_vocab_sizes");
        PleWeights values{weight(root + ".key.weight"), weight(root + ".value.weight"),
            dense(root + ".key_norm.weight"), dense(root + ".query_norm.weight"),
            dense(root + ".conv_norm.weight"), dense(root + ".conv.weight")};
        NgramEmbedding embedding(std::move(shards), row_count, width, c.ngram, c.ngram_heads, c.eos,
            std::move(multipliers), std::move(offsets), std::move(sizes));
        return std::make_unique<Ple>(std::move(embedding), std::move(values), c.hidden, c.streams, c.ngram, c.eps);
    }
    std::unique_ptr<Layer> layer(const Config& c, int layer, bool predictor) const;
};

std::unique_ptr<Layer> LayerLoader::layer(const Config& c, int layer, bool predictor) const {
    const auto root = std::string(predictor ? "predictor" : "model") + ".block." + std::to_string(layer);
    auto result = std::make_unique<Layer>();
    result->device = placement.target;
    result->config = c;
    result->attention_gr = residual(root + ".attention", c);
    result->ffn_gr = residual(root + ".mlp", c);
    result->ffn = ffn(root + ".mlp", c, layer);
    if (c.linear_layer(layer, predictor)) result->gdn = gdn(root + ".linear_attention", c);
    else result->qsa = qsa(root + ".attention", c);
    if (c.position_embedding_layer(layer, predictor)) result->ple = ple(root + ".position_embedding", c);
    if (const auto* routed = std::get_if<Moe>(&result->ffn))
        tyloi::weights::prepare_experts(routed->gate_up, routed->down, routed->top_k, routed->swiglu_limit);
    return result;
}

} // namespace

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
    if (!loader.has("predictor.embedding_norm.weight") || main.predictor_layers <= 0) {
        if (loader.has_prefix("predictor"))
            throw std::runtime_error("Qwen4 model source contains an incomplete or undeclared MTP head");
        return std::nullopt;
    }
    if (main.dedicated_predictor_embeddings)
        throw std::runtime_error("Qwen4 dedicated predictor embeddings are not supported");
    result.config = main;
    result.embedding_norm = loader.norm("predictor.embedding_norm.weight", main.hidden);
    result.hidden_norm = loader.norm("predictor.hidden_norm.weight", main.hidden * main.streams);
    result.embedding_fusion = loader.weight("predictor.fusion.embedding.weight");
    result.hidden_fusion = loader.weight("predictor.fusion.hidden.weight");
    result.final_mixer = std::make_unique<ops::Residual>(loader.residual("predictor", main, true));
    for (std::int64_t i = 0; i < main.predictor_layers; ++i)
        result.layers.push_back(loader.layer(main, static_cast<int>(i), true));
    if (result.device.is_cuda()) {
        constexpr auto head = "predictor.output.weight";
        constexpr auto tokens = "predictor.output.token_ids";
        const bool has_head = loader.has(head), has_tokens = loader.has(tokens);
        if (!(has_head == has_tokens))
            throw std::runtime_error("MTP draft output requires both weights and vocabulary IDs");
        if (has_head) {
            const auto& metadata = require_tensor(loader.source, tokens);
            if (!(metadata.dtype == "I64" || metadata.dtype == "I32"))
                throw std::runtime_error("MTP draft vocabulary IDs must be integers");
            auto ids = loader.host(tokens);
            const auto& weight = require_tensor(loader.source, head);
            if (!(ids.dim() == 1 && ids.numel() > 0 && weight.shape &&
                *weight.shape == std::vector<int64_t>({ids.numel(), main.hidden})))
                throw std::runtime_error("MTP draft output geometry differs from target");
            auto ordered = tyloi::ops::read_integers(ids);
            std::sort(ordered.begin(), ordered.end());
            if (!(ordered.front() >= 0 && ordered.back() < main.vocab &&
                std::adjacent_find(ordered.begin(), ordered.end()) == ordered.end()))
                throw std::runtime_error("MTP draft vocabulary contains invalid or duplicate IDs");
            result.draft_output = loader.weight(head, TensorParallelAxis::Mirrored);
            result.draft_token_ids = tyloi::ops::contiguous(tyloi::ops::copy(
                tyloi::ops::cast(ids, tyloi::kInt64), result.device));
            tyloi::weights::observe_weight(result.draft_token_ids);
        }
    }
    return result;
}

void read(Model& model, std::shared_ptr<const tyloi::ModelSource> model_source,
          std::string_view definition, std::int64_t context_size) {
    model.source = std::move(model_source);
    const auto& source = *model.source;
    ::tyloi::models::validate_model_source(source);
    model.graph = tyloi::models::graph(source);
    if (!Model::accepts_backbone(model.graph.backbone))
        throw std::runtime_error("model backbone does not match the requested causal model");
    for (const auto& component : model.graph.components)
        if (component.kind != "text" && component.kind != "vision" && component.kind != "predictor")
            throw std::runtime_error("unsupported Qwen4 component: " + component.kind);
    if (const auto* vision = model.graph.component("vision"); vision &&
        (vision->tensor_root != "vision" || vision->implementation != "grid_vit" ||
         vision->input_contract != kGridVisionInputContract ||
         vision->position_policy != kGridMropePositionPolicy))
        throw std::runtime_error("Qwen4 vision requires grid_vit/grid_vision.v1/grid_mrope");
    if (const auto* predictor = model.graph.component("predictor"); predictor &&
        (predictor->tensor_root != "predictor" || predictor->implementation != "next_token_prediction" ||
         !predictor->input_contract.empty() || !predictor->position_policy.empty()))
        throw std::runtime_error("unsupported Qwen4 predictor declaration");
    model.load_config(definition);
    if (model.graph.backbone != model.metadata.model_type)
        throw std::runtime_error("Qwen4 model graph and FFN variant disagree");
    const auto layout = source.tensor_layout();
    model.config.norm_weight_offset = layout.norm_weight_offset;
    model.config.linear_tiled_heads |= layout.qwen_gdn_gguf_layout;
    if (model.num_hidden_layers() != model.graph.topology.text_layers)
        throw std::runtime_error("model graph/config text-layer topology mismatch");
    if (context_size > 0) {
        if (context_size > model.max_position_embeddings())
            throw std::runtime_error("--ctx-size exceeds max_position_embeddings");
        model.config.maximum = context_size;
        model.metadata.max_position_embeddings = context_size;
    }
}

void load(Model& model, const tyloi::weights::Loader& tensors, LayerPlacementConfig& placement,
          const tyloi::models::LoadOptions& options, bool load_layers) {
    tyloi::models::validate_native_placement(tensors.placement.tensor, placement, options.gpu_layers);
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
    if (model.graph.has_component("vision") && dense_cpu_layer_count > 0)
        throw std::runtime_error("CUDA grid-Vision currently requires GPU-resident text layers");
    model.device = tensors.placement.target;
    constexpr auto embedding = "model.token_embedding.weight";
    constexpr auto output = "model.output.weight";
    const bool independent = !model.tie_word_embeddings() && tensors.has(output);
    model.embed = tensors.embedding(embedding, independent);
    LayerLoader loader{tensors, options.yarn, options.kv_quantization};
    model.final_mixer = std::make_unique<ops::Residual>(loader.residual("model", model.config, true));
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

#include "engine/models/source/model.h"
#include "engine/ops/ops.h"
#include "engine/models/qwen35/load.h"

#include "engine/models/qwen35/metadata.h"
#include "engine/models/qwen35/mtp.h"
#include "engine/models/qwen35/weights.h"

#include "engine/models/common/load.h"
#include "engine/models/common/check.h"
#include "engine/models/load.h"
#include "engine/models/source/model_graph.h"
#include "engine/weights/loader.h"
#include "engine/weights/cache/cache.h"
#include "engine/weights/source/read.h"

#include <algorithm>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>

namespace tyloi::models::qwen35 {
namespace {

struct LayerLoader : TransformerWeightLoader {
    bool mirror_scalars = true;

    void linear_projections(::tyloi::models::LinearAttention& attention,
                            const tyloi::models::qwen35::LinearWeightNames& names,
                            bool split_qkv) const {
        const auto dense_group = [&](std::initializer_list<std::pair<std::string, int64_t>> names) {
            std::vector<::tyloi::Tensor> parts;
            for (const auto& [name, rows] : names) {
                auto weight = weights.tensor(name);
                if (weight.dim() != 2)
                    throw std::runtime_error("dense linear group expects 2D weights");
                if (weight.size(0) != rows)
                    throw std::runtime_error("dense linear group output width mismatch");
                if (!parts.empty() && weight.size(1) != parts.front().size(1))
                    throw std::runtime_error("cannot group dense tensors with different input width");
                parts.push_back(tyloi::ops::cast(weight, ::tyloi::kFloat32));
            }
            return tyloi::ops::contiguous(tyloi::ops::cat(parts, 0));
        };
        if (split_qkv) {
            attention.split_in_proj = true;
            attention.qkv_proj = weights.group({names.qk, names.value});
            if (is_quant_dtype(require_tensor(weights.source, names.gate).dtype)) {
                attention.z_proj = weights.weight(names.gate);
                const bool alpha_quant = is_quant_dtype(require_tensor(weights.source, names.alpha).dtype);
                const bool beta_quant = is_quant_dtype(require_tensor(weights.source, names.beta).dtype);
                if (alpha_quant != beta_quant)
                    throw std::runtime_error(
                        "linear_attn alpha/beta must use the same storage kind");
                attention.ab_is_nint = alpha_quant;
                if (attention.ab_is_nint) {
                    const auto scalar_axis =
                        mirror_scalars
                            ? std::optional<TensorParallelAxis>(TensorParallelAxis::Mirrored)
                            : std::nullopt;
                    if (mirror_scalars && weights.placement.tensor.enabled()) {
                        attention.ab_nint_proj = {
                            weights.weight(names.alpha, scalar_axis),
                            weights.weight(names.beta, scalar_axis)};
                    } else {
                        attention.ab_nint_proj = weights.group({names.alpha, names.beta});
                    }
                } else {
                    attention.ab_proj = dense_group({{names.alpha, attention.value_heads},
                                                     {names.beta, attention.value_heads}});
                }
            } else {
                attention.split_dense_zab = true;
                attention.zab_proj = dense_group({{names.gate, attention.value_heads * attention.value_dim},
                                                  {names.alpha, attention.value_heads},
                                                  {names.beta, attention.value_heads}});
            }
            return;
        }

        const bool alpha_quant = is_quant_dtype(require_tensor(weights.source, names.alpha).dtype);
        const bool beta_quant = is_quant_dtype(require_tensor(weights.source, names.beta).dtype);
        if (alpha_quant != beta_quant)
            throw std::runtime_error(
                "linear_attention alpha/beta must use the same storage kind");
        if (alpha_quant) {
            attention.in_proj = weights.group({names.qkv, names.gate, names.alpha, names.beta});
        } else {
            attention.dense_ab_tail = true;
            attention.in_proj = weights.group({names.qkv, names.gate});
            attention.ab_proj = dense_group({{names.alpha, attention.value_heads},
                                             {names.beta, attention.value_heads}});
        }
    }

    void linear_output(::tyloi::models::LinearAttention& attention, const std::string& name) const {
        if (is_quant_dtype(require_tensor(weights.source, name).dtype)) {
            attention.out_proj = weights.weight(name);
        } else {
            attention.dense_out_proj = true;
            attention.out_proj_dense = weights.tensor(name);
            if (require_tensor(weights.source, name).dtype == "F16")
                attention.out_proj_dense = tyloi::ops::contiguous(
                    tyloi::ops::cast(attention.out_proj_dense, ::tyloi::kFloat16));
            if (attention.out_proj_dense.dim() != 2)
                throw std::runtime_error(
                    "dense linear_attention output projection must be 2D");
        }
    }

    void load_linear_attention(::tyloi::models::LinearAttention& attention, const Config& config,
                               int layer, std::string_view tensor_root) const {
        const auto prefix = std::string(tensor_root) + ".block." + std::to_string(layer) + ".";
        const auto linear = prefix + "linear_attention.";
        tyloi::models::qwen35::LinearWeightNames names{
            linear + "qkv.weight", linear + "qk.weight", linear + "value.weight",
            linear + "gate.weight", linear + "alpha.weight", linear + "beta.weight"};
        linear_projections(attention, names, has(names.qk) && has(names.value));
        attention.conv_weight = weights.tensor(linear + "conv.weight");
        if (has(linear + "conv.bias")) attention.conv_bias = weights.tensor(linear + "conv.bias");
        attention.dt_bias = weights.tensor(linear + "dt_bias");
        auto a = weights.tensor(linear + "a");
        attention.a_log = config.legacy_tensor_layout.linear_attention_a_is_log
            ? a : ::tyloi::ops::log_negative(a);
        attention.linear_norm = weights.tensor(linear + "norm.weight");
        linear_output(attention, linear + "output.weight");
    }
};

std::optional<Predictor> load_predictor(LayerLoader&, const Model&, tyloi::state::KvQuantization);

} // namespace

void load_layer(Layer& result, const ::tyloi::weights::Loader& weights, const Config& config,
                 int index, const std::string& type, bool mirror_scalars,
                 std::string_view tensor_root, tyloi::state::KvQuantization quantization) {
    if (index < 0 || index >= config.num_hidden_layers)
        throw std::out_of_range("Qwen layer index is outside the configured model");
    if (static_cast<std::size_t>(index) < config.layer_types.size() &&
        config.layer_types[static_cast<std::size_t>(index)] != type)
        throw std::invalid_argument("Qwen layer type disagrees with model config");
    tyloi::ops::DeviceScope scope(weights.placement.target);
    result.device = weights.placement.target;
    LayerLoader loader{{weights}, mirror_scalars};
    if (Config::attention_kind(type) == Config::AttentionKind::full) {
        auto& full = result.weights.emplace<Full>();
        full.attention.kv_quantization = quantization;
        tyloi::models::load_attention_mixer(
            full.attention, loader, config, index,
            std::string(tensor_root) + ".block." + std::to_string(index) + ".",
            config.legacy_tensor_layout.norm_weight_offset,
            config.attention_output_gate);
        full.attention.attention_rotary_dim = config.rotary_dim;
        full.attention.no_rope = config.positional_encoding == tyloi::models::PositionalEncoding::none;
        full.ffn_norm = weights.tensor(std::string(tensor_root) + ".block." +
                                       std::to_string(index) + ".mlp.norm.weight");
        full.ffn = tyloi::models::qwen35::load_ffn(loader, config, index, tensor_root);
    } else {
        auto& linear = result.weights.emplace<Linear>();
        linear.attention.key_heads = config.linear_num_key_heads;
        linear.attention.value_heads = config.linear_num_value_heads;
        linear.attention.key_dim = config.linear_key_head_dim;
        linear.attention.value_dim = config.linear_value_head_dim;
        linear.attention.conv_kernel = config.linear_conv_kernel_dim;
        linear.attention.rms_norm_eps = config.rms_norm_eps;
        linear.attention.norm_weight_offset = config.legacy_tensor_layout.norm_weight_offset;
        linear.attention.tiled_v_heads = config.legacy_tensor_layout.qwen_gdn_gguf_layout;
        const auto prefix = std::string(tensor_root) + ".block." + std::to_string(index) + ".";
        linear.attention.attn_norm = weights.tensor(prefix + "attention.norm.weight");
        linear.ffn_norm = weights.tensor(prefix + "mlp.norm.weight");
        loader.load_linear_attention(linear.attention, config, index, tensor_root);
        linear.ffn = tyloi::models::qwen35::load_ffn(loader, config, index, tensor_root);
    }
}

Layer load_layer(const ::tyloi::weights::Loader& weights, const Config& config,
                 int index, const std::string& type, bool mirror_scalars,
                 std::string_view tensor_root, tyloi::state::KvQuantization quantization) {
    Layer result;
    load_layer(result, weights, config, index, type, mirror_scalars, tensor_root, quantization);
    return result;
}

Model read(std::shared_ptr<const tyloi::ModelSource> source,
           std::string_view definition, std::int64_t context) {
    tyloi::models::validate_model_source(*source);
    Model model;
    model.source = std::move(source);
    const auto& source_ref = *model.source;
    model.graph = tyloi::models::graph(source_ref);
    for (const auto& component : model.graph.components)
        if (component.kind != "text" && component.kind != "vision" && component.kind != "predictor")
            throw std::runtime_error("unsupported Qwen component: " + component.kind);
    if (const auto* vision = model.graph.component("vision"); vision &&
        (vision->tensor_root != "vision" || vision->implementation != "grid_vit" ||
         vision->input_contract != kGridVisionInputContract ||
         vision->position_policy != kGridMropePositionPolicy))
        throw std::runtime_error("Qwen vision requires grid_vit/grid_vision.v1/grid_mrope");
    if (const auto* predictor = model.graph.component("predictor"); predictor &&
        (predictor->tensor_root != "predictor" || predictor->implementation != "next_token_prediction" ||
         !predictor->input_contract.empty() || !predictor->position_policy.empty()))
        throw std::runtime_error("unsupported Qwen predictor declaration");
    if (model.graph.backbone != "qwen3_5" && model.graph.backbone != "generic_qwen")
        throw std::runtime_error("Qwen3.5 loading requires a Qwen-compatible model graph");
    if (model.graph.backbone != "qwen3_5" &&
            (model.graph.has_component("vision") || model.graph.has_component("predictor")))
        throw std::runtime_error("generic Qwen does not support vision or predictor components");
    model.config = Config::from_json(definition, model.graph);
    model.config.legacy_tensor_layout = source_ref.tensor_layout();
    if (context > 0) {
        if (context > model.config.max_position_embeddings)
            throw std::runtime_error("--ctx-size exceeds max_position_embeddings");
        model.config.max_position_embeddings = context;
    }
    if (model.config.num_hidden_layers != model.graph.topology.text_layers)
        throw std::runtime_error("model graph/config text-layer topology mismatch");
    model.metadata = tyloi::models::qwen35::metadata_from_config(model.config);
    return model;
}

std::unique_ptr<Model> load(std::shared_ptr<const tyloi::ModelSource> source,
                            std::string_view definition, const ::tyloi::weights::Loader& weights,
                            LayerPlacementConfig& placement, const tyloi::models::LoadOptions& options,
                            int& cpu_layers) {
    auto model = std::make_unique<Model>(read(std::move(source), definition, options.context));
    load(*model, weights, placement, options.yarn, cpu_layers,
         {.gpu_layers = options.gpu_layers, .expert_parallel = options.expert_parallel,
          .mirror_scalars = options.mirror_scalars,
          .recurrent_transposed = options.recurrent_transposed,
          .kv_quantization = options.kv_quantization});
    return model;
}

void load(Model& model, const ::tyloi::weights::Loader& weights,
          LayerPlacementConfig& placement, const tyloi::ops::YarnScaling& yarn,
          int& cpu_layers, const LoadOptions& options) {
    model.device = weights.placement.target;
    model.recurrent_transposed = options.recurrent_transposed;
    model.kv_quantization = options.kv_quantization;
    if (options.expert_parallel && model.config.num_experts <= 0)
        throw std::runtime_error("--expert-parallel requires a model with routed experts");

    placement.prepare(model.config.num_hidden_layers);
    const int placed_cpu_layers = options.gpu_layers < 0
        ? 0
        : static_cast<int>(std::max<int64_t>(model.config.num_hidden_layers - options.gpu_layers, 0));
    if (placed_cpu_layers > 0) {
        if (weights.placement.tensor.enabled() || options.expert_parallel || placement.enabled())
            throw std::runtime_error(
                "--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
        bool supported = model.config.num_experts <= 0;
        for (const auto& layer_type : model.config.layer_types)
            supported = supported &&
                        (layer_type == "full_attention" || layer_type == "linear_attention");
        if (!supported)
            throw std::runtime_error(
                "--n-gpu-layers currently supports dense Qwen-style blocks only");
        std::cerr << "dense_layer_placement cpu=0-" << placed_cpu_layers - 1
                  << " gpu=" << placed_cpu_layers << '-' << model.config.num_hidden_layers - 1
                  << std::endl;
    }
    if (model.graph.has_component("vision") && placed_cpu_layers > 0)
        throw std::runtime_error("CUDA grid-Vision currently requires GPU-resident text layers");

    cpu_layers = placed_cpu_layers;
    const auto& config = model.config;
    tyloi::ops::DeviceScope scope(model.device);
    if (config.positional_encoding != tyloi::models::PositionalEncoding::none) {
        const auto make_rope = [&](::tyloi::Device device) {
            return tyloi::ops::make_rope(config.max_position_embeddings, config.rotary_dim,
                config.rope_base, device, config.mrope_sections, config.mrope_interleaved,
                model.metadata.rope_interleaved, yarn);
        };
        model.rope = make_rope(model.device);
        if (cpu_layers > 0) model.cpu_rope = make_rope(::tyloi::Device(::tyloi::kCPU));
        if (placement.enabled()) {
            for (int device : placement.devices)
                model.device_ropes.emplace(device, make_rope(::tyloi::Device(model.device.type, device)));
        }
    }

    if (options.load_input_output) {
        const auto output_name = std::string("model.output.weight");
        const bool independent_output = !config.tie_word_embeddings && weights.has(output_name);
        model.embed = weights.embedding("model.token_embedding.weight", independent_output);
        model.output_norm = weights.tensor("model.output_norm.weight");
        if (config.tie_word_embeddings || !weights.has(output_name)) {
            model.lm_head = weights.placement.tensor.enabled()
                ? weights.weight("model.token_embedding.weight", TensorParallelAxis::Output)
                : model.embed;
        } else {
            model.lm_head = weights.weight(output_name, TensorParallelAxis::Output);
        }
        if (options.load_layers) {
            model.layers.reserve(static_cast<std::size_t>(config.num_hidden_layers));
            for (int index = 0; index < config.num_hidden_layers; ++index) {
                const int device = placement.device_for_layer(index);
                const bool cpu = index < cpu_layers;
                std::cerr << "loading layer " << index << ' '
                          << config.layer_types.at(static_cast<std::size_t>(index)) << ' '
                          << (cpu ? "CPU" : "device") << std::endl;
                const auto target = cpu ? ::tyloi::Device(::tyloi::kCPU) : ::tyloi::Device(model.device.type, device);
                model.layers.push_back(load_layer(
                    weights.for_layer(target), config, index,
                    config.layer_types.at(static_cast<std::size_t>(index)), options.mirror_scalars,
                    "model", model.kv_quantization));
            }
        }
    } else {
        if (options.load_layers)
            throw std::invalid_argument("streamed model setup requires deferred layers");
        model.output_norm = weights.tensor("model.output_norm.weight");
    }
}

void load_vision(Model& model, const ::tyloi::weights::Loader& weights, bool resident) {
    if (const auto* component = model.graph.component("vision")) {
        const auto& config = model.config;
        if (!config.grid_vision || !config.image_token_id || !config.video_token_id)
            throw std::runtime_error("grid-Vision configuration is incomplete");
        model.vision.emplace(tyloi::models::grid::Vision::load(
            weights, weights.cache(), *config.grid_vision,
            *config.image_token_id, *config.video_token_id,
            component->input_contract, component->position_policy, resident));
    }
}

void load_components(Model& model, const ::tyloi::weights::Loader& weights,
        const LayerPlacementConfig& placement, int cpu_layers, bool cpu_offload,
        bool resident, bool preallocate) {
    const bool mtp_allowed =
        !placement.enabled() && cpu_layers == 0 && !cpu_offload &&
        weights.cache().transfer() == ::tyloi::weights::Cache::Transfer::automatic;

    load_vision(model, weights, resident);
    if (model.graph.has_component("predictor")) {
        if (mtp_allowed && model.metadata.num_experts == 0 && model.supports_speculation()) {
            LayerLoader loader{{weights}};
            auto predictor = load_predictor(loader, model, model.kv_quantization);
            if (predictor) model.mtp = std::make_unique<Mtp>(std::move(*predictor), preallocate);
        } else {
            std::cerr
                << "qwen_mtp unavailable: requires dense device-resident Qwen blocks\n";
        }
    }
}

namespace {
std::optional<tyloi::models::qwen35::Predictor> load_predictor(LayerLoader& loader,
                                         const Model& target,
                                         tyloi::state::KvQuantization quantization) {
    const auto& config = target.config;
    const bool fusion_present = loader.has("predictor.fusion.weight");
    const bool any = fusion_present || loader.has("predictor.hidden_norm.weight") ||
        loader.has("predictor.embedding_norm.weight") ||
        loader.has("predictor.output_norm.weight") ||
        loader.has("predictor.block.0.attention.query.weight");
    if (!fusion_present) {
        tyloi::models::require_model(!any, "Qwen model source contains an incomplete MTP head");
        return {};
    }
    tyloi::models::require_model(config.mtp_num_hidden_layers > 0 && !config.mtp_use_dedicated_embeddings,
        "Qwen MTP requires declared predictor layers and shared embeddings");
    tyloi::models::qwen35::Predictor model;
    model.config = config;
    model.embedding = target.embed;
    model.rope = target.rope;
    model.device = target.device;
    model.hidden_norm = loader.weights.tensor("predictor.hidden_norm.weight");
    model.embedding_norm = loader.weights.tensor("predictor.embedding_norm.weight");
    model.output_norm = loader.weights.tensor("predictor.output_norm.weight");
    model.fusion = loader.linear("predictor.fusion.weight");
    tyloi::models::require_model(model.fusion.size(1) == 2 * config.hidden_size &&
        model.fusion.size(0) == config.hidden_size &&
        model.hidden_norm.numel() == config.hidden_size &&
        model.embedding_norm.numel() == config.hidden_size &&
        model.output_norm.numel() == config.hidden_size,
        "Qwen MTP component dimensions disagree with the backbone");
    model.layers.reserve(config.mtp_num_hidden_layers);
    for (int index = 0; index < config.mtp_num_hidden_layers; ++index) {
        const auto prefix = "predictor.block." + std::to_string(index) + ".";
        tyloi::models::qwen35::Predictor::Layer layer;
        layer.attention.kv_quantization = quantization;
        tyloi::models::load_attention_mixer(layer.attention, loader, config, index, prefix,
            config.legacy_tensor_layout.norm_weight_offset, config.attention_output_gate);
        layer.ffn_norm = loader.weights.tensor(prefix + "mlp.norm.weight");
        const auto mlp = prefix + "mlp.";
        if (loader.has(mlp + "experts.gate_up.weight") || loader.has(mlp + "experts.gate.weight") ||
            loader.has(mlp + "experts.up.weight") || loader.has(mlp + "experts.down.weight"))
            throw std::runtime_error("Qwen MTP requires dense predictor projections");
        layer.ffn = tyloi::models::load_dense_ffn(loader, config, mlp);
        model.layers.push_back(std::move(layer));
    }
    return model;
}

} // namespace
} // namespace tyloi::models::qwen35

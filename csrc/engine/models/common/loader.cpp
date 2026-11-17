#include "engine/models/common/load.h"
#include "engine/models/common/important_neurons.h"
#include "engine/models/common/mlp.h"
#include "engine/models/common/attention.h"
#include "engine/weights/loader.h"
#include "engine/weights/source/read.h"

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>

namespace tyloi::models {

std::string configure_yarn(tyloi::ops::YarnScaling& yarn, bool extend, std::string definition,
                           std::string_view backbone, int64_t context) {
    yarn = {};
    if(extend) {
        if(backbone!="qwen3_5" && backbone!="qwen4_exp")
            throw std::invalid_argument("YaRN extension requires a Qwen model");
        auto config=nlohmann::json::parse(definition);
        auto& text=config.contains("text_config") ? config["text_config"] : config.contains("language_config") ? config["language_config"] : config;
        auto rope=text.value("rope_parameters",text.value("rope_scaling",nlohmann::json::object()));
        if(rope.contains("full_attention"))rope=rope["full_attention"];
        auto native=text.value("max_position_embeddings",int64_t(0));
        const auto kind=rope.value("rope_type",rope.value("type",std::string("default")));
        if(kind=="yarn")native=rope.value("original_max_position_embeddings",native);
        const auto maximum=rope.value("factor",4.0);
        if((kind!="default" && kind!="yarn") || native<=0 || !std::isfinite(maximum) || maximum<1 ||
            context>std::floor(native*maximum))throw std::invalid_argument("context exceeds Qwen YaRN limit");
        yarn.original_context=native;
        yarn.factor=std::max(1.0,double(context)/native);
        yarn.beta_fast=rope.value("beta_fast",32.0);
        yarn.beta_slow=rope.value("beta_slow",1.0);
        if(yarn.enabled())text["max_position_embeddings"]=context;
        definition=config.dump();
    }
    return definition;
}

void validate_native_placement(const ParallelConfig& tensor, const LayerPlacementConfig& layers,
                               int gpu_layers) {
    if (tensor.enabled() || layers.enabled() || gpu_layers >= 0)
        throw std::runtime_error(
            "native attention adapter supports expert parallelism, but "
            "tensor/layer parallelism and offload require a different placement path");
}

} // namespace tyloi::models

::tyloi::Tensor TransformerWeightLoader::dense(const std::string& name) const {
    return weights.tensor(name);
}
::tyloi::Tensor TransformerWeightLoader::linear(const std::string& name) const {
    return weights.weight(name);
}
bool TransformerWeightLoader::has(const std::string& name) const { return tyloi::has_tensor(weights.source, name); }
::tyloi::Tensor TransformerWeightLoader::router_parameter(const std::string& name) const {
    if(keep_quantized_router && name.ends_with("router.weight") && tyloi::require_tensor(weights.source,name).dtype=="TQ")
        return linear(name);
    auto value = weights.tensor(name);
    auto converted = tyloi::ops::cast(value, ::tyloi::kFloat32);
    return tyloi::ops::contiguous(converted);
}
std::vector<::tyloi::Tensor> TransformerWeightLoader::projections(const std::vector<std::string>& names) const {
    return weights.group(names);
}
std::vector<::tyloi::Tensor> TransformerWeightLoader::gate_up(const std::vector<std::string>& names,
                                                 const ::tyloi::Tensor& down) const {
    return weights.gate_up(names, down, shared_gate_up_compatible_prefix,
                              preserve_projection_boundaries);
}
void TransformerWeightLoader::qkv(tyloi::models::FullAttention& block, const std::vector<std::string>& names) const {
    const bool mirror_kv = block.attention_output_gate &&
        weights.options.mirror_attention_kv &&
        is_quant_dtype(tyloi::require_tensor(weights.source, names[1]).dtype) &&
        is_quant_dtype(tyloi::require_tensor(weights.source, names[2]).dtype);
    if (mirror_kv) {
        block.split_q_kv_projections = true;
        block.q_projection = weights.weight(names[0]);
        block.k_projection = weights.weight(names[1], TensorParallelAxis::Mirrored);
        block.v_projection = weights.weight(names[2], TensorParallelAxis::Mirrored);
    } else {
        block.qkv = weights.group(names, 2, nullptr, preserve_projection_boundaries);
    }
}
void TransformerWeightLoader::output(tyloi::models::FullAttention& block, const std::string& name) const {
    block.o = weights.weight(name);
}
void TransformerWeightLoader::important_neurons(int64_t hidden, int64_t intermediate,
    tyloi::models::Mlp& ffn, const std::string& down,
    const std::string& gate, const std::string& up) const {
    tyloi::models::load_important_branch(weights, hidden, intermediate, ffn,
        down, gate, up);
}
::tyloi::Tensor TransformerWeightLoader::expert(tyloi::models::ExpertProjection projection,
                                     const std::string& name, int layer) const {
    using Role = tyloi::models::ExpertProjection;
    const auto& record = tyloi::require_tensor(weights.source, name);
    const char* role = nullptr;
    switch (projection) {
        case Role::gate: role = "gate"; break;
        case Role::up: role = "up"; break;
        case Role::gate_up: role = "gate_up"; break;
        case Role::down: role = "down"; break;
    }
    if(record.dtype=="TQ") {
        if(projection==Role::gate_up)
            throw std::invalid_argument("TQ MoE stores separate Gate, Up and Down tensors: "+name);
        return weights.experts(name, true, layer, role);
    }
    if (record.dtype == "F32" || record.dtype == "F16" || record.dtype == "BF16") {
        // Source-precision banks stay dense instead of entering an MFE cohort.
        if (cpu_offloaded)
            throw std::runtime_error(
                "CPU-offloaded MoE cannot materialize a dense expert bank at layer " +
                std::to_string(layer));
        if (projection == Role::gate_up)
            throw std::runtime_error(
                "dense fused Gate/Up expert banks require separate Gate and Up records at layer " +
                std::to_string(layer));
        auto bank = weights.native(name);
        if (bank.dim() != 3)
            throw std::runtime_error(
                "dense routed expert tensor must be [experts,out,in]: " + name);
        return bank;
    }

    if (record.dtype == "MLX-AFFINE" || record.dtype.starts_with("GGUF_"))
        return weights.experts(name, true, layer, role, cpu_offloaded);

#if TYLOINFER_ENABLE_MFQ_COMPAT
    return weights.experts(name, true, layer, role, cpu_offloaded);

#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}
::tyloi::Tensor TransformerWeightLoader::headwise(const std::string& name) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT

    return weights.experts(name);

#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

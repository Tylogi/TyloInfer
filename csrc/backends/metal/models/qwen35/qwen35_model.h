#pragma once

#include "engine/models/qwen35/config.h"
#include "mfq_container.h"
#include "mlx_grid_vision.h"
#include "mlx_tensor.h"
#include "mlx_transformer.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

inline constexpr std::string_view kMfqModelConfigAsset =
    "__mfq_asset__/model_config.json";

using tyloi::models::qwen35::Config;

Config load_qwen35_config(const MfqContainer& model);

struct Qwen35TensorMetadata {
    std::string dtype;
    std::vector<std::int64_t> shape;
    bool packed = false;
    std::int32_t axis = -1;
    std::int64_t neuron_len = 0;
    std::int32_t bits = 0;
    std::int32_t sub_bits = 0;
    std::int32_t group_size = 0;
    std::uint32_t output_size = 0;
    std::uint32_t groups = 0;
};

struct Qwen35ResolvedLayerNames {
    std::string attention_norm;
    std::string attention_query;
    std::string attention_key;
    std::string attention_value;
    std::string attention_output;
    std::string attention_query_norm;
    std::string attention_key_norm;
    std::string ffn_norm;
    std::string ffn_gate;
    std::string ffn_up;
    std::string ffn_down;
    std::string linear_qkv;
    std::optional<std::string> linear_qk;
    std::optional<std::string> linear_value;
    std::string linear_z;
    std::string linear_alpha;
    std::string linear_beta;
    std::string linear_conv;
    std::optional<std::string> linear_conv_bias;
    std::string linear_dt_bias;
    std::string linear_a;
    std::string linear_norm;
    std::string linear_output;
};

struct Qwen35TensorNames {
    std::string token_embedding = "model.token_embedding.weight";
    std::string attention_norm = "model.block.{i}.attention.norm.weight";
    std::string attention_query = "model.block.{i}.attention.query.weight";
    std::string attention_key = "model.block.{i}.attention.key.weight";
    std::string attention_value = "model.block.{i}.attention.value.weight";
    std::string attention_output = "model.block.{i}.attention.output.weight";
    std::string attention_query_norm =
        "model.block.{i}.attention.query_norm.weight";
    std::string attention_key_norm =
        "model.block.{i}.attention.key_norm.weight";
    std::string ffn_norm = "model.block.{i}.mlp.norm.weight";
    std::string ffn_gate = "model.block.{i}.mlp.gate.weight";
    std::string ffn_up = "model.block.{i}.mlp.up.weight";
    std::string ffn_down = "model.block.{i}.mlp.down.weight";
    std::string output_norm = "model.output_norm.weight";
    std::string output = "model.output.weight";
    std::string linear_qkv =
        "model.block.{i}.linear_attention.qkv.weight";
    std::optional<std::string> linear_qk =
        "model.block.{i}.linear_attention.qk.weight";
    std::optional<std::string> linear_value =
        "model.block.{i}.linear_attention.value.weight";
    std::string linear_z =
        "model.block.{i}.linear_attention.gate.weight";
    std::string linear_alpha =
        "model.block.{i}.linear_attention.alpha.weight";
    std::string linear_beta =
        "model.block.{i}.linear_attention.beta.weight";
    std::string linear_conv =
        "model.block.{i}.linear_attention.conv.weight";
    std::optional<std::string> linear_conv_bias =
        "model.block.{i}.linear_attention.conv.bias";
    std::string linear_dt_bias =
        "model.block.{i}.linear_attention.dt_bias";
    std::string linear_a = "model.block.{i}.linear_attention.a";
    std::string linear_norm =
        "model.block.{i}.linear_attention.norm.weight";
    std::string linear_output =
        "model.block.{i}.linear_attention.output.weight";

    static Qwen35TensorNames canonical();
    static Qwen35TensorNames predictor();

    Qwen35ResolvedLayerNames layer(std::size_t index) const;
};

Qwen35TensorMetadata inspect_qwen35_tensor_metadata(
    const MfqContainer& model,
    const std::string& name);

void validate_qwen35_model_bindings(
    const MfqContainer& model,
    const Config& config,
    const Qwen35TensorNames& names);

MlxRmsNorm load_qwen35_rms_norm(
    const MfqContainer& model,
    const std::string& name,
    double eps,
    double weight_offset = 1.0);

} // namespace mfq::metal

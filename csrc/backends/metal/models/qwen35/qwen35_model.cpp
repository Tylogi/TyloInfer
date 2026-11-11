#include "qwen35_model.h"

#include "engine/weights/encoding/quant/nint_blob.h"
#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "mlx_nint.h"
#include "mlx_nint8_zero.h"
#include "mlx_vq.h"
#endif

#include "mlx_legacy_tensor_compat.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::array;

class PrefixCursor {
public:
    explicit PrefixCursor(const std::vector<std::uint8_t>& bytes)
        : bytes_(bytes) {}

    template <typename T>
    T scalar(const char* what) {
        if (sizeof(T) > bytes_.size() - offset_) {
            throw std::runtime_error(
                std::string("truncated MFQ tensor ") + what);
        }
        T value{};
        std::memcpy(&value, bytes_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    std::string bytes(std::size_t count, const char* what) {
        if (count > bytes_.size() - offset_) {
            throw std::runtime_error(
                std::string("truncated MFQ tensor ") + what);
        }
        std::string value(
            reinterpret_cast<const char*>(bytes_.data() + offset_),
            count);
        offset_ += count;
        return value;
    }

    std::size_t offset() const noexcept {
        return offset_;
    }

private:
    const std::vector<std::uint8_t>& bytes_;
    std::size_t offset_ = 0;
};

std::vector<std::uint8_t> read_record_prefix(
    const MfqContainer& model,
    const std::string& name,
    std::size_t maximum = 128) {
    const auto& record = model.record(name);
    const auto count = static_cast<std::size_t>(
        std::min<std::uint64_t>(record.nbytes, maximum));
    return model.read_range(name, 0, count);
}

std::vector<std::int64_t> read_shape(
    PrefixCursor& cursor,
    std::uint32_t dimensions,
    const std::string& name) {
    if (dimensions == 0 || dimensions > 8) {
        throw std::runtime_error(
            "invalid MFQ tensor dimension count: " + name);
    }
    std::vector<std::int64_t> shape;
    shape.reserve(dimensions);
    for (std::uint32_t index = 0; index < dimensions; ++index) {
        const auto value = cursor.scalar<std::int64_t>("shape");
        if (value <= 0 ||
            value > std::numeric_limits<std::int32_t>::max()) {
            throw std::runtime_error(
                "invalid MFQ tensor shape: " + name);
        }
        shape.push_back(value);
    }
    return shape;
}

std::int64_t shape_product_except_axis(
    const std::vector<std::int64_t>& shape,
    std::int32_t axis,
    const std::string& name) {
    if (axis < 0 ||
        static_cast<std::size_t>(axis) >= shape.size()) {
        throw std::runtime_error(
            "invalid MFQ tensor axis: " + name);
    }
    std::int64_t product = 1;
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index == static_cast<std::size_t>(axis)) {
            continue;
        }
        if (product >
            std::numeric_limits<std::int64_t>::max() / shape[index]) {
            throw std::runtime_error(
                "MFQ tensor shape overflows: " + name);
        }
        product *= shape[index];
    }
    return product;
}

bool is_dense_float_dtype(const std::string& dtype) {
    return dtype == "BF16" || dtype == "F16" || dtype == "F32";
}

bool is_supported_linear_dtype(const std::string& dtype) {
    return is_dense_float_dtype(dtype) || dtype == "TQ" || is_mx_dtype(dtype) || is_gguf_dtype(dtype)
#if TYLOINFER_ENABLE_MFQ_COMPAT
        || is_nint_dtype(dtype) || is_nint8_zero_dtype(dtype) || is_vq_dtype(dtype)
#endif
        ;
}

std::string shape_text(const std::vector<std::int64_t>& shape) {
    std::string result = "[";
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            result += ",";
        }
        result += std::to_string(shape[index]);
    }
    result += "]";
    return result;
}

void require_tensor(
    const MfqContainer& model,
    const std::string& name,
    const std::vector<std::int64_t>& expected_shape,
    bool dense_only) {
    if (!model.contains(name)) {
        throw std::runtime_error(
            "Qwen3.5 model is missing tensor: " + name);
    }
    const auto metadata =
        inspect_qwen35_tensor_metadata(model, name);
    if (metadata.shape != expected_shape) {
        throw std::runtime_error(
            "Qwen3.5 tensor " + name + " has shape " +
            shape_text(metadata.shape) + ", expected " +
            shape_text(expected_shape));
    }
    const bool supported = dense_only
        ? is_dense_float_dtype(metadata.dtype)
        : is_supported_linear_dtype(metadata.dtype);
    if (!supported) {
        throw std::runtime_error(
            "Qwen3.5 tensor " + name +
            " has unsupported dtype " + metadata.dtype);
    }
}

void require_optional_tensor(
    const MfqContainer& model,
    const std::string& name,
    const std::vector<std::int64_t>& expected_shape,
    bool dense_only) {
    if (model.contains(name)) {
        require_tensor(model, name, expected_shape, dense_only);
    }
}

std::string format_layer_name(
    const std::string& pattern,
    std::size_t index) {
    auto result = pattern;
    const auto placeholder = result.find("{i}");
    if (placeholder == std::string::npos) {
        throw std::runtime_error(
            "Qwen3.5 tensor pattern lacks {i}: " + pattern);
    }
    result.replace(placeholder, 3, std::to_string(index));
    return result;
}

std::optional<std::string> format_optional_layer_name(
    const std::optional<std::string>& pattern,
    std::size_t index) {
    if (!pattern) {
        return std::nullopt;
    }
    return format_layer_name(*pattern, index);
}

array load_norm_weight(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    if (!is_dense_float_dtype(record.dtype)) {
        throw std::runtime_error(
            "RMSNorm requires a dense BF16/F16/F32 tensor: " + name);
    }
    auto weight = load_dense_array(record.dtype, model.read(name));
    if (weight.ndim() != 1) {
        throw std::runtime_error(
            "RMSNorm weight must be one-dimensional: " + name);
    }
    return weight.dtype() == mlx::core::float32
        ? weight
        : mlx::core::astype(weight, mlx::core::float32);
}

} // namespace

Config load_qwen35_config(const MfqContainer& model) {
    if (!model.contains(std::string(kMfqModelConfigAsset))) {
        throw std::runtime_error("MFQ has no embedded model config asset");
    }
    auto config = Config::from_json(
        model.read_text(std::string(kMfqModelConfigAsset)), effective_model_graph(model));
    config.legacy_tensor_layout = model.legacy_tensor_layout();
    return config.context(0);
}

Qwen35TensorNames Qwen35TensorNames::canonical() {
    return {};
}

Qwen35TensorNames Qwen35TensorNames::predictor() {
    Qwen35TensorNames names;
    names.attention_norm =
        "predictor.block.{i}.attention.norm.weight";
    names.attention_query =
        "predictor.block.{i}.attention.query.weight";
    names.attention_key =
        "predictor.block.{i}.attention.key.weight";
    names.attention_value =
        "predictor.block.{i}.attention.value.weight";
    names.attention_output =
        "predictor.block.{i}.attention.output.weight";
    names.attention_query_norm =
        "predictor.block.{i}.attention.query_norm.weight";
    names.attention_key_norm =
        "predictor.block.{i}.attention.key_norm.weight";
    names.ffn_norm =
        "predictor.block.{i}.mlp.norm.weight";
    names.ffn_gate = "predictor.block.{i}.mlp.gate.weight";
    names.ffn_up = "predictor.block.{i}.mlp.up.weight";
    names.ffn_down = "predictor.block.{i}.mlp.down.weight";
    return names;
}

Qwen35ResolvedLayerNames Qwen35TensorNames::layer(
    std::size_t index) const {
    return {
        format_layer_name(attention_norm, index),
        format_layer_name(attention_query, index),
        format_layer_name(attention_key, index),
        format_layer_name(attention_value, index),
        format_layer_name(attention_output, index),
        format_layer_name(attention_query_norm, index),
        format_layer_name(attention_key_norm, index),
        format_layer_name(ffn_norm, index),
        format_layer_name(ffn_gate, index),
        format_layer_name(ffn_up, index),
        format_layer_name(ffn_down, index),
        format_layer_name(linear_qkv, index),
        format_optional_layer_name(linear_qk, index),
        format_optional_layer_name(linear_value, index),
        format_layer_name(linear_z, index),
        format_layer_name(linear_alpha, index),
        format_layer_name(linear_beta, index),
        format_layer_name(linear_conv, index),
        format_optional_layer_name(linear_conv_bias, index),
        format_layer_name(linear_dt_bias, index),
        format_layer_name(linear_a, index),
        format_layer_name(linear_norm, index),
        format_layer_name(linear_output, index),
    };
}

Qwen35TensorMetadata inspect_qwen35_tensor_metadata(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    const auto prefix = read_record_prefix(model, name);
    PrefixCursor cursor(prefix);
    Qwen35TensorMetadata metadata;
    metadata.dtype = record.dtype;

    if (is_gguf_dtype(record.dtype)) {
        metadata.shape = model.tensor_shape(name);
        metadata.packed = true;
        metadata.axis = 0;
        metadata.neuron_len = metadata.shape.back();
        metadata.output_size = static_cast<std::uint32_t>(metadata.shape.front());
        return metadata;
    }

    if (record.dtype == "TQ") {
        metadata.shape = model.tq_shape(name);
        metadata.packed = true;
        metadata.axis = 0;
        metadata.neuron_len = metadata.shape.back();
        metadata.output_size = static_cast<std::uint32_t>(metadata.shape.front());
        return metadata;
    }
    if (is_mx_dtype(record.dtype)) {
        if (cursor.bytes(4, "MX magic") != "MXT1" ||
            cursor.scalar<std::uint8_t>("MX version") != 1) {
            throw std::runtime_error("invalid MX tensor header: " + name);
        }
        cursor.scalar<std::uint8_t>("MX kind");
        cursor.scalar<std::uint16_t>("MX flags");
        const auto outputs = cursor.scalar<std::uint64_t>("MX rows");
        const auto width = cursor.scalar<std::uint64_t>("MX columns");
        if (!outputs || !width || outputs > std::numeric_limits<std::int32_t>::max() ||
            width > std::numeric_limits<std::int64_t>::max())
            throw std::runtime_error("invalid MX tensor shape: " + name);
        metadata.shape = {static_cast<std::int64_t>(outputs), static_cast<std::int64_t>(width)};
        metadata.packed = true;
        metadata.axis = 0;
        metadata.neuron_len = static_cast<std::int64_t>(width);
        metadata.output_size = static_cast<std::uint32_t>(outputs);
        metadata.bits = record.dtype == "MXFP4" ? 4 : 8;
        return metadata;
    }

    if (is_dense_float_dtype(record.dtype) ||
        record.dtype == "I32" ||
        record.dtype == "I64") {
        metadata.shape = read_shape(
            cursor,
            cursor.scalar<std::uint32_t>("dimension count"),
            name);
        return metadata;
    }

#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (is_nint8_zero_dtype(record.dtype)) {
        if (cursor.bytes(4, "NINT8-0 magic") != "NI80") {
            throw std::runtime_error(
                "invalid NINT8-0 tensor magic: " + name);
        }
        metadata.packed = true;
        metadata.bits = 8;
        metadata.axis = cursor.scalar<std::int32_t>("axis");
        metadata.neuron_len =
            cursor.scalar<std::int32_t>("neuron length");
        metadata.shape = read_shape(
            cursor,
            cursor.scalar<std::uint32_t>("dimension count"),
            name);
        metadata.output_size =
            cursor.scalar<std::uint32_t>("output size");
        metadata.groups =
            cursor.scalar<std::uint32_t>("group count");
        if (metadata.neuron_len <= 0 ||
            metadata.neuron_len % 32 != 0 ||
            shape_product_except_axis(
                metadata.shape,
                metadata.axis,
                name) != metadata.neuron_len ||
            metadata.output_size !=
                static_cast<std::uint32_t>(
                    metadata.shape.at(
                        static_cast<std::size_t>(metadata.axis))) ||
            metadata.groups !=
                static_cast<std::uint32_t>(metadata.neuron_len / 32)) {
            throw std::runtime_error(
                "inconsistent NINT8-0 tensor header: " + name);
        }
        const auto expected = cursor.offset() +
            static_cast<std::uint64_t>(metadata.output_size) *
            metadata.groups * 34u;
        if (expected != record.nbytes) {
            throw std::runtime_error(
                "invalid NINT8-0 tensor length: " + name);
        }
        return metadata;
    }

    if (is_nint_dtype(record.dtype)) {
        metadata.packed = true;
        metadata.bits = mfq::nint_logical_bits(
            cursor.scalar<std::uint8_t>("NINT bits"));
        metadata.sub_bits = static_cast<std::int32_t>(
            cursor.scalar<std::uint8_t>("NINT sub bits") & 0x7fu);
        metadata.group_size =
            cursor.scalar<std::int32_t>("NINT group size");
        metadata.axis = cursor.scalar<std::int32_t>("NINT axis");
        metadata.neuron_len =
            cursor.scalar<std::int32_t>("NINT neuron length");
        metadata.shape = read_shape(
            cursor,
            cursor.scalar<std::uint32_t>("dimension count"),
            name);
        metadata.output_size =
            cursor.scalar<std::uint32_t>("output size");
        metadata.groups =
            cursor.scalar<std::uint32_t>("group count");
        if (metadata.bits <= 0 || metadata.bits > 8 ||
            metadata.sub_bits <= 0 || metadata.sub_bits > 8 ||
            metadata.group_size <= 0 ||
            metadata.neuron_len <= 0 ||
            shape_product_except_axis(
                metadata.shape,
                metadata.axis,
                name) != metadata.neuron_len ||
            metadata.output_size !=
                static_cast<std::uint32_t>(
                    metadata.shape.at(
                        static_cast<std::size_t>(metadata.axis))) ||
            metadata.groups !=
                static_cast<std::uint32_t>(
                    (metadata.neuron_len +
                     metadata.group_size - 1) /
                    metadata.group_size) ||
            (record.dtype.size() == 5 &&
             metadata.bits != record.dtype[4] - '0')) {
            throw std::runtime_error(
                "inconsistent NINT tensor header: " + name);
        }
        return metadata;
    }

    if (is_vq_dtype(record.dtype)) {
        const auto vq = inspect_vq_blob(record.dtype, prefix);
        metadata.packed = true;
        metadata.neuron_len = vq.input_size;
        metadata.output_size =
            static_cast<std::uint32_t>(vq.output_size);
        metadata.shape.reserve(vq.output_shape.size() + 1);
        for (const int dimension : vq.output_shape) {
            metadata.shape.push_back(dimension);
        }
        metadata.shape.push_back(vq.input_size);
        return metadata;
    }

#endif
    throw std::runtime_error(
        "unsupported Qwen3.5 tensor dtype " +
        record.dtype + ": " + name);
}

void validate_qwen35_model_bindings(
    const MfqContainer& model,
    const Config& config,
    const Qwen35TensorNames& names) {
    require_tensor(
        model,
        names.token_embedding,
        {config.vocab_size, config.hidden_size},
        false);
    require_tensor(
        model,
        names.output_norm,
        {config.hidden_size},
        true);
    if (model.contains(names.output)) {
        require_tensor(
            model,
            names.output,
            {config.vocab_size, config.hidden_size},
            false);
    } else if (!config.tie_word_embeddings) {
        throw std::runtime_error(
            "Qwen3.5 model is missing untied output tensor: " +
            names.output);
    }

    for (std::size_t index = 0;
         index < config.layer_types.size();
         ++index) {
        const auto layer = names.layer(index);
        require_tensor(
            model,
            layer.attention_norm,
            {config.hidden_size},
            true);
        require_tensor(
            model,
            layer.ffn_norm,
            {config.hidden_size},
            true);
        require_tensor(
            model,
            layer.ffn_gate,
            {config.intermediate_size, config.hidden_size},
            false);
        require_tensor(
            model,
            layer.ffn_up,
            {config.intermediate_size, config.hidden_size},
            false);
        require_tensor(
            model,
            layer.ffn_down,
            {config.hidden_size, config.intermediate_size},
            false);

        if (config.layer_types[index] == "full_attention") {
            require_tensor(
                model,
                layer.attention_query,
                {config.query_projection_size(), config.hidden_size},
                false);
            require_tensor(
                model,
                layer.attention_key,
                {config.kv_size(), config.hidden_size},
                false);
            require_tensor(
                model,
                layer.attention_value,
                {config.kv_size(), config.hidden_size},
                false);
            require_tensor(
                model,
                layer.attention_output,
                {config.hidden_size, config.attention_size()},
                false);
            require_optional_tensor(
                model,
                layer.attention_query_norm,
                {config.head_dim},
                true);
            require_optional_tensor(
                model,
                layer.attention_key_norm,
                {config.head_dim},
                true);
            continue;
        }

        const bool split_input =
            layer.linear_qk.has_value() &&
            layer.linear_value.has_value() &&
            model.contains(*layer.linear_qk) &&
            model.contains(*layer.linear_value);
        if (split_input) {
            require_tensor(
                model,
                *layer.linear_qk,
                {2 * config.linear_k_size(), config.hidden_size},
                false);
            require_tensor(
                model,
                *layer.linear_value,
                {config.linear_v_size(), config.hidden_size},
                false);
        } else {
            require_tensor(
                model,
                layer.linear_qkv,
                {config.linear_qkv_size(), config.hidden_size},
                false);
        }
        require_tensor(
            model,
            layer.linear_z,
            {config.linear_v_size(), config.hidden_size},
            false);
        require_tensor(
            model,
            layer.linear_alpha,
            {config.linear_value_heads(), config.hidden_size},
            false);
        require_tensor(
            model,
            layer.linear_beta,
            {config.linear_value_heads(), config.hidden_size},
            false);

        const auto conv = inspect_qwen35_tensor_metadata(
            model,
            layer.linear_conv);
        const std::vector<std::int64_t> conv_2d = {
            config.linear_qkv_size(),
            config.linear_conv_kernel_dim,
        };
        const std::vector<std::int64_t> conv_3d = {
            config.linear_qkv_size(),
            1,
            config.linear_conv_kernel_dim,
        };
        if (!is_dense_float_dtype(conv.dtype) ||
            (conv.shape != conv_2d && conv.shape != conv_3d)) {
            throw std::runtime_error(
                "Qwen3.5 tensor " + layer.linear_conv +
                " has incompatible convolution shape/dtype");
        }
        if (layer.linear_conv_bias.has_value()) {
            require_optional_tensor(
                model,
                *layer.linear_conv_bias,
                {config.linear_qkv_size()},
                true);
        }
        require_tensor(
            model,
            layer.linear_dt_bias,
            {config.linear_value_heads()},
            true);
        require_tensor(
            model,
            layer.linear_a,
            {config.linear_value_heads()},
            true);
        require_tensor(
            model,
            layer.linear_norm,
            {config.linear_value_head_dim},
            true);
        require_tensor(
            model,
            layer.linear_output,
            {config.hidden_size, config.linear_v_size()},
            false);
    }
}

MlxRmsNorm load_qwen35_rms_norm(
    const MfqContainer& model,
    const std::string& name,
    double eps,
    double weight_offset) {
    return MlxRmsNorm(
        load_norm_weight(model, name),
        static_cast<float>(eps),
        static_cast<float>(weight_offset));
}

} // namespace mfq::metal

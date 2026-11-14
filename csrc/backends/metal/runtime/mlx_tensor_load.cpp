#include "mlx_tensor.h"
#include "mfq_container.h"
#include "mlx_kernel_prepare.h"
#include "engine/backends/metal/native/allocation.h"
#include "engine/backends/metal/native/tensor/tensor.h"
#include "engine/weights/memory.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::metal {
namespace {
using mlx::core::array;
using mlx::core::Dtype;
using mlx::core::Shape;

class DenseCursor {
public:
    explicit DenseCursor(std::span<const std::uint8_t> blob)
        : blob_(blob) {}

    template <typename T>
    T scalar(const char* name) {
        if (sizeof(T) > blob_.size() - offset_) {
            throw std::runtime_error(
                std::string("truncated dense tensor ") + name);
        }
        T value{};
        std::memcpy(&value, blob_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    const std::uint8_t* data() const noexcept {
        return blob_.data() + offset_;
    }
    std::size_t remaining() const noexcept {
        return blob_.size() - offset_;
    }

private:
    std::span<const std::uint8_t> blob_;
    std::size_t offset_ = 0;
};

std::pair<Dtype, std::size_t> dense_dtype(const std::string& name) {
    if (name == "BF16") {
        return {mlx::core::bfloat16, 2};
    }
    if (name == "F16") {
        return {mlx::core::float16, 2};
    }
    if (name == "F32") {
        return {mlx::core::float32, 4};
    }
    if (name == "I32") {
        return {mlx::core::int32, 4};
    }
    if (name == "I64") {
        return {mlx::core::int64, 8};
    }
    throw std::runtime_error("unsupported dense MFQ dtype: " + name);
}

array load_weight(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    if (record.dtype != "BF16" &&
        record.dtype != "F16" &&
        record.dtype != "F32") {
        throw std::runtime_error(
            "dense linear/embedding requires BF16, F16, or F32 tensor: " + name);
    }
    const auto mapped = model.map_record(name);
    auto result = load_dense_array(record.dtype, mapped.view());
    if (result.ndim() != 2) {
        throw std::runtime_error(
            "linear/embedding weight must have rank two: " + name);
    }
    return result;
}

} // namespace

using mlx::core::array;
using mlx::core::Dtype;
using mlx::core::Shape;

array load_dense_array(
    const std::string& dtype_name,
    std::span<const std::uint8_t> blob) {
    DenseCursor cursor(blob);
    const auto dimensions =
        cursor.scalar<std::uint32_t>("dimension count");
    if (dimensions == 0 || dimensions > 8) {
        throw std::runtime_error("invalid dense MFQ dimension count");
    }
    std::vector<std::int64_t> shape;
    shape.reserve(dimensions);
    std::size_t elements = 1;
    for (std::uint32_t index = 0; index < dimensions; ++index) {
        const auto value = cursor.scalar<std::int64_t>("shape");
        if (value <= 0 ||
            value > std::numeric_limits<std::int32_t>::max() ||
            elements >
                std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(value)) {
            throw std::runtime_error("invalid dense MFQ shape");
        }
        shape.push_back(value);
        elements *= static_cast<std::size_t>(value);
    }
    const auto [dtype, item_size] = dense_dtype(dtype_name);
    if (elements >
            std::numeric_limits<std::size_t>::max() / item_size ||
        cursor.remaining() != elements * item_size) {
        throw std::runtime_error("dense MFQ payload length mismatch");
    }

    const auto tensor = tyloi::metal::copy_from_host(
        cursor.data(), shape, tyloi::metal::scalar_type(dtype));
    tyloi::weights::observe_weight(tensor);
    auto result = tyloi::metal::array(tensor);
    tyloi::metal::AllocationScope::record(result);
    return result;
}

MlxLinear MlxLinear::load(
    const MfqContainer& model,
    const std::string& name) {
    const auto finish = [](MlxLinear result) {
        if (mlx_predequantize_fp16_enabled()) {
            result.materialize_fp16();
        }
        if (auto* preparation = MlxKernelPreparation::current();
            preparation && !preparation->model_paths() && result.packed()) {
            preparation->collect([&] {
                for (const auto rows : preparation->row_buckets()) {
                    for (const auto dtype : {mlx::core::float16, mlx::core::float32}) {
                        const auto input = mlx::core::zeros(Shape{rows, result.input_size()}, dtype);
                        preparation->add(result(input));

                        if (rows == 1) {
                            if (auto greedy = result.greedy_argmax(input)) preparation->add(*greedy);
                        }
                    }
                }
            });
        }
        return result;
    };
    const auto& record = model.record(name);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (is_nint8_zero_dtype(record.dtype)) return finish(MlxLinear(MlxNint8ZeroWeight::from_blob(model.map_record(name).view())));
    if (is_nint_dtype(record.dtype)) return finish(MlxLinear(MlxNintWeight::from_blob(model.map_record(name).view())));
    if (is_vq_dtype(record.dtype)) return finish(MlxLinear(MlxVqWeight::from_blob(record.dtype, model.map_record(name).view())));
    if (is_fp8_sq_dtype(record.dtype)) return finish(MlxLinear(MlxFp8SqWeight::from_blob(record.dtype, model.map_record(name).view())));
    if (is_mxfp4_sq_dtype(record.dtype)) return finish(MlxLinear(MlxMxfp4SqWeight::from_blob(model.map_record(name).view())));
#endif
    if (record.dtype == "TQ") return finish(MlxLinear(MlxTqWeight::load(model, name)));
    if (is_gguf_dtype(record.dtype)) return finish(MlxLinear(MlxGgufWeight::load(model, name, true)));
    if (record.dtype == "MLX-AFFINE") return finish(MlxLinear(MlxAffineWeight::load(model, name)));
    if (is_mx_dtype(record.dtype)) {
        return finish(MlxLinear(
            MlxMxWeight::from_blob(record.dtype, model.read(name))));
    }
    return finish(MlxLinear(load_weight(model, name)));
}


MlxEmbedding MlxEmbedding::load(
    const MfqContainer& model,
    const std::string& name,
    std::optional<Dtype> activation_dtype) {
    const auto finish = [activation_dtype](MlxEmbedding result) {
        if (activation_dtype) result.set_activation_dtype(*activation_dtype);
        if (mlx_predequantize_fp16_enabled()) {
            result.materialize_fp16();
        }
        if (auto* preparation = MlxKernelPreparation::current()) {
            preparation->collect([&] {
                for (const auto rows : preparation->row_buckets()) {
                    preparation->add(result(mlx::core::zeros(Shape{rows}, mlx::core::int32),
                        result.preferred_activation_dtype()));
                    const std::vector<std::int32_t> ids(rows, 0);
                    preparation->add(result(array(ids.begin(), Shape{rows}), result.preferred_activation_dtype()));
                }
            });
        }
        return result;
    };
    const auto& record = model.record(name);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (is_nint8_zero_dtype(record.dtype)) return finish(MlxEmbedding(MlxNint8ZeroWeight::from_blob(model.map_record(name).view())));
    if (is_nint_dtype(record.dtype)) return finish(MlxEmbedding(MlxNintWeight::from_blob(model.map_record(name).view())));
    if (is_vq_dtype(record.dtype)) return finish(MlxEmbedding(MlxVqWeight::from_blob(record.dtype, model.map_record(name).view())));
    if (is_fp8_sq_dtype(record.dtype)) return finish(MlxEmbedding(MlxFp8SqWeight::from_blob(record.dtype, model.map_record(name).view())));
    if (is_mxfp4_sq_dtype(record.dtype)) return finish(MlxEmbedding(MlxMxfp4SqWeight::from_blob(model.map_record(name).view())));
#endif
    if (record.dtype == "TQ") return finish(MlxEmbedding(MlxTqWeight::load(model, name)));
    if (is_gguf_dtype(record.dtype)) return finish(MlxEmbedding(MlxGgufWeight::load(model, name)));
    if (record.dtype == "MLX-AFFINE") return finish(MlxEmbedding(MlxAffineWeight::load(model, name)));
    if (is_mx_dtype(record.dtype)) {
        return finish(MlxEmbedding(
            MlxMxWeight::from_blob(record.dtype, model.read(name))));
    }
    return finish(MlxEmbedding(load_weight(model, name)));
}


} // namespace tyloi::metal

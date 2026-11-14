#include "engine/weights/encoding/metal.h"

#include "engine/weights/encoding/mlx_affine.h"
#include "engine/weights/source/read.h"
#include "engine/weights/tq/source.h"
#include "mlx_tensor.h"

#include <climits>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::encoding {

MetalTensor::MetalTensor(std::shared_ptr<const metal::MlxLinear> linear,
        std::shared_ptr<const metal::MlxEmbedding> embedding,
        std::vector<std::int64_t> shape)
    : linear_(std::move(linear)), embedding_(std::move(embedding)),
      shape_(std::move(shape)) {}

std::vector<std::shared_ptr<TensorStorage>> MetalTensor::backings() const {
    return linear_->backings();
}
std::size_t MetalTensor::backing_nbytes() const { return backing_bytes(backings()); }

std::int64_t MetalTensor::numel() const {
    std::int64_t result = 1;
    for (const auto extent : shape_) {
        if (extent < 0 || (extent && result > std::numeric_limits<std::int64_t>::max() / extent))
            throw std::overflow_error("Metal weight element count overflow");
        result *= extent;
    }
    return result;
}
std::int64_t MetalTensor::size(std::int64_t dimension) const {
    if (dimension < 0) dimension += static_cast<std::int64_t>(shape_.size());
    return shape_.at(static_cast<std::size_t>(dimension));
}

namespace {
template<class Weight>
Tensor make_weight(Weight weight, std::vector<std::int64_t> shape = {}) {
    if (shape.empty()) shape = {weight.output_size(), weight.input_size()};
    if ((shape.size() != 2 && shape.size() != 3) ||
        shape[shape.size() - 2] != weight.output_size() || shape.back() != weight.input_size())
        throw std::invalid_argument("Metal encoded weight shape differs from payload");
    auto linear = std::make_shared<metal::MlxLinear>(weight);
    auto embedding = std::make_shared<metal::MlxEmbedding>(std::move(weight));
    return std::make_shared<MetalTensor>(std::move(linear), std::move(embedding),
        std::move(shape));
}

metal::MlxAffineWeight affine(const std::vector<std::uint8_t>& blob) {
    using namespace mlx::core;
    const auto layout = MlxAffineLayout::parse(
        reinterpret_cast<const std::byte*>(blob.data()), blob.size());
    if (layout.total_bytes() != blob.size()) throw std::invalid_argument("MLX affine payload size mismatch");
    Shape shape;
    for (auto d : layout.shape) {
        if (d <= 0 || d > INT_MAX) throw std::invalid_argument("MLX affine dimension out of range");
        shape.push_back(static_cast<int>(d));
    }
    auto packed_shape = shape;
    packed_shape.back() = shape.back() / 32 * layout.bits;
    auto scale_shape = shape;
    scale_shape.back() /= layout.group_size;
    const auto dtype = layout.scale_dtype == "BF16" ? bfloat16 :
        layout.scale_dtype == "F16" ? float16 : float32;
    auto copy = [&](Shape dimensions, Dtype type, std::size_t offset, std::size_t bytes) {
        array value(allocator::malloc(bytes), std::move(dimensions), type);
        std::memcpy(value.data<std::uint8_t>(), blob.data() + offset, bytes);
        return value;
    };
    return metal::MlxAffineWeight::from_arrays(
        copy(packed_shape, uint32, layout.prefix_bytes, layout.values_bytes),
        copy(scale_shape, dtype, layout.scales_offset(), layout.scales_bytes),
        copy(scale_shape, dtype, layout.biases_offset(), layout.scales_bytes),
        layout.bits, layout.group_size);
}
} // namespace

Tensor load_metal_encoded(const source::Source& source, const std::string& name) {
    const auto* record = source.find_tensor(name);
    const bool tq = (!record && is_tq_table(source, name)) || (record && record->dtype == "TQ");
    if (tq) {
        const auto shape = tq_shape(source, name);
        if (shape.back() > INT_MAX || shape.front() > INT_MAX)
            throw std::invalid_argument("TQ weight dimension out of range");
        std::vector<std::uint8_t> blob;
        if (record) blob = read_tensor(source, name);
        else {
            auto table = tq_table(source, name);
            blob = table->slice_rows(0, table->rows());
        }
        return make_weight(metal::MlxTqWeight::from_blob(blob, static_cast<int>(shape.back()),
            shape.size() == 3 ? static_cast<int>(shape.front()) : 1), shape);
    }
    const auto& entry = require_tensor(source, name);
    const auto blob = read_tensor(source, name);
    const auto& dtype = entry.dtype;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (metal::is_nint8_zero_dtype(dtype)) return make_weight(metal::MlxNint8ZeroWeight::from_blob(blob));
    if (metal::is_nint_dtype(dtype)) return make_weight(metal::MlxNintWeight::from_blob(blob));
    if (metal::is_vq_dtype(dtype)) return make_weight(metal::MlxVqWeight::from_blob(dtype, blob));
    if (metal::is_fp8_sq_dtype(dtype)) return make_weight(metal::MlxFp8SqWeight::from_blob(dtype, blob));
    if (metal::is_mxfp4_sq_dtype(dtype)) return make_weight(metal::MlxMxfp4SqWeight::from_blob(blob));
#endif
    if (dtype == "MLX-AFFINE") {
        auto weight = affine(blob);
        const auto layout = MlxAffineLayout::parse(reinterpret_cast<const std::byte*>(blob.data()), blob.size());
        return make_weight(std::move(weight), layout.shape);
    }
    if (metal::is_mx_dtype(dtype)) return make_weight(metal::MlxMxWeight::from_blob(dtype, blob));
    if (metal::is_gguf_dtype(dtype)) {
        if (!entry.shape) throw std::invalid_argument("GGUF weight has no shape: " + name);
        mlx::core::Shape shape;
        for (const auto extent : *entry.shape) {
            if (extent <= 0 || extent > INT_MAX) throw std::invalid_argument("GGUF weight dimension out of range");
            shape.push_back(static_cast<int>(extent));
        }
        auto weight = metal::MlxGgufWeight::from_bytes(dtype, blob, std::move(shape), source, name);
        return make_weight(std::move(weight), *entry.shape);
    }
    throw std::invalid_argument("unsupported Metal encoded weight: " + dtype);
}

} // namespace tyloi::encoding

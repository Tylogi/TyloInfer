#include "mlx_affine.h"
#include "mfq_container.h"
#include "engine/backends/metal/ops/gguf/native/mlx_native_gguf.h"
#include "engine/weights/encoding/mlx_affine.h"
#include <mlx/allocator.h>
#include <climits>
#include <stdexcept>

namespace tyloi::metal {
using namespace mlx::core;
namespace {
tyloi::MlxAffineLayout read_layout(const MfqContainer& model, const std::string& name) {
    auto prefix = model.read_range(name, 0, 8);
    const auto size = tyloi::MlxAffineLayout::prefix_size(reinterpret_cast<const std::byte*>(prefix.data()), prefix.size());
    prefix = model.read_range(name, 0, size);
    const auto layout = tyloi::MlxAffineLayout::parse(reinterpret_cast<const std::byte*>(prefix.data()), prefix.size());
    if (layout.total_bytes() != model.record(name).nbytes)
        throw std::invalid_argument("MLX affine tensor payload length mismatch");
    return layout;
}
} // namespace

MlxAffineWeight MlxAffineWeight::load(const MfqContainer& model, const std::string& name) {
    if (model.record(name).dtype != "MLX-AFFINE") throw std::invalid_argument("tensor is not MLX affine");
    const auto layout = read_layout(model, name);
    Shape shape;
    for (const auto dimension : layout.shape) {
        if (dimension > INT_MAX) throw std::invalid_argument("MLX affine tensor dimension exceeds runtime limits");
        shape.push_back(int(dimension));
    }
    const auto read = [&](Shape field_shape, Dtype dtype, std::uint64_t offset, std::size_t bytes) {
        array result(allocator::malloc(bytes), std::move(field_shape), dtype);
        model.read_range_into(name, offset, {reinterpret_cast<std::byte*>(result.data<std::uint8_t>()), bytes});
        return result;
    };
    auto packed_shape = shape; packed_shape.back() = shape.back() / 32 * layout.bits;
    auto scale_shape = shape; scale_shape.back() /= layout.group_size;
    const auto dtype = layout.scale_dtype == "BF16" ? bfloat16 : layout.scale_dtype == "F16" ? float16 : float32;
    auto result = from_arrays(read(packed_shape, uint32, layout.prefix_bytes, layout.values_bytes),
        read(scale_shape, dtype, layout.scales_offset(), layout.scales_bytes),
        read(scale_shape, dtype, layout.biases_offset(), layout.scales_bytes), layout.bits, layout.group_size);
    model.record_prepared(name);
    return result;
}
bool MlxAffineWeight::supports_native(const MfqContainer& model, const std::string& name) {
    if (model.record(name).dtype != "MLX-AFFINE") return false;
    const auto layout = read_layout(model, name);
    if (layout.shape.size() != 2 && layout.shape.size() != 3) return false;
    const auto rows = layout.shape[layout.shape.size() - 2], width = layout.shape.back();
    return layout.scale_dtype == "BF16" && rows <= INT_MAX && width <= INT_MAX &&
        (layout.shape.size() != 3 || rows % 64 == 0) &&
        mlx_native_affine_supported(layout.bits, layout.group_size, int(rows), int(width));
}

} // namespace tyloi::metal

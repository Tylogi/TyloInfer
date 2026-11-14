#include "mlx_native_gguf.h"
#include "mfq_container.h"
#include "engine/backends/metal/native/allocation.h"
#include "ggml.h"
#include <mlx/allocator.h>
#include <cctype>
#include <climits>
#include <stdexcept>
#include <string_view>

namespace tyloi::metal {
using namespace mlx::core;
MlxNativeGgufWeight mlx_native_gguf_load(const MfqContainer& model, const std::string& name) {
    const auto& record = model.record(name);
    const auto shape = model.tensor_shape(name);
    if (shape.size() != 2 && shape.size() != 3) throw std::invalid_argument("native GGUF weight rank differs");
    const int experts = shape.size() == 3 ? int(shape[0]) : 1;
    const int rows = int(shape[shape.size() - 2]), width = int(shape.back());
    int type = -1;
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const auto* base = ggml_type_name(ggml_type(i));
        if (!base) continue;
        std::string dtype = "GGUF_";
        for (const auto c : std::string_view(base)) dtype += char(std::toupper(static_cast<unsigned char>(c)));
        if (dtype == record.dtype) type = i;
    }
    if (type < 0 || record.nbytes > INT_MAX) throw std::invalid_argument("unsupported native GGUF source tensor");
    array packed(allocator::malloc(record.nbytes), {int(record.nbytes)}, uint8);
    model.read_range_into(name, 0, {reinterpret_cast<std::byte*>(packed.data<std::uint8_t>()), std::size_t(record.nbytes)});
    auto result = mlx_native_gguf_pack(packed, type, experts * rows, width, int(record.nbytes / (experts * rows)));
    eval(std::vector<array>(result.planes.begin(), result.planes.end()));
    for (const auto& plane : result.planes) tyloi::metal::AllocationScope::record(plane);
    result.outputs = rows; result.experts = experts;
    model.record_prepared(name);
    return result;
}


} // namespace tyloi::metal

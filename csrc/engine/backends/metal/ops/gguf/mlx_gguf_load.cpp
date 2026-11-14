#include "mlx_gguf.h"
#include "mlx_gguf_internal.h"
#include "mlx_gguf_staged.h"
#include "native/mlx_native_gguf.h"
#include "mfq_container.h"
#include "engine/backends/metal/native/allocation.h"
#include "engine/backends/metal/native/tensor/tensor.h"
#include "engine/weights/memory.h"
#include "nlohmann/json.hpp"
#include <mlx/allocator.h>
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace tyloi::metal {
using namespace mlx::core;
namespace {
int checked_int(std::size_t value) {
    if (value > INT_MAX) throw std::overflow_error("GGUF tensor dimension exceeds runtime limits");
    return static_cast<int>(value);
}
} // namespace

bool MlxGgufWeight::supports_native(const MfqContainer& model, const std::string& name) {
    const auto rotations = model.header().extra_json.find("gguf_hadamard");
    if (rotations != model.header().extra_json.end() && nlohmann::json::parse(rotations->second).contains(name)) return false;
    const auto& dtype = model.record(name).dtype;
    const auto format = gguf_format_id(dtype);
    if (format < 0) return false;
    const auto shape = model.tensor_shape(name);
    if ((shape.size() != 2 && shape.size() != 3) ||
        std::any_of(shape.begin(), shape.end(), [](auto d) { return d < 1 || d > INT_MAX; })) return false;
    return mlx_native_gguf_supported(format, int(shape[shape.size() - 2]), int(shape.back()));
}

MlxGgufWeight MlxGgufWeight::load(const MfqContainer& model, const std::string& name, bool native_projection) {
    Shape shape;
    for (const auto d : model.tensor_shape(name)) {
        if (d <= 0 || d > INT_MAX) throw std::invalid_argument("invalid GGUF tensor dimension");
        shape.push_back(static_cast<int>(d));
    }
    const auto& record = model.record(name);
    if (record.dtype == "GGUF_MIXED") {
        if (shape.size() != 3) throw std::invalid_argument("mixed GGUF weights require an expert bank");
        const auto layouts = nlohmann::json::parse(model.header().extra_json.at("gguf_mixed_experts"));
        const auto& entries = layouts.at(name);
        if (!entries.is_array() || entries.size() != std::size_t(shape[0]))
            throw std::invalid_argument("mixed GGUF expert layout length mismatch");
        std::vector<std::int32_t> ids;
        std::vector<std::string> dtypes;
        std::vector<array> storage;
        std::size_t next = 0;
        for (const auto& entry : entries) {
            const auto dtype = entry.at("dtype").get<std::string>();
            const auto offset = entry.at("offset").get<std::size_t>();
            const auto bytes = entry.at("nbytes").get<std::size_t>();
            if (offset != next || offset > record.nbytes || bytes > record.nbytes - offset ||
                bytes != std::size_t(shape[1]) * gguf_packed_row_bytes(dtype, shape[2]))
                throw std::invalid_argument("mixed GGUF expert byte ranges disagree");
            auto value = array(allocator::malloc(bytes), Shape{checked_int(bytes)}, uint8);
            model.read_range_into(name, offset, {reinterpret_cast<std::byte*>(value.data<std::uint8_t>()), bytes});
            ids.push_back(int(ids.size())); dtypes.push_back(dtype); storage.push_back(std::move(value));
            next += bytes;
        }
        if (next != record.nbytes) throw std::invalid_argument("mixed GGUF expert payload has trailing bytes");
        auto result = from_mixed_experts(std::move(shape), ids, dtypes, storage);
        result.configure_hadamard(model, name);
        model.record_prepared(name);
        return result;
    }
    auto result = allocate_packed(record.dtype, std::move(shape), record.nbytes);
    result.configure_hadamard(model, name);
    auto packed = tyloi::metal::array(result.bytes_);
    model.read_range_into(name, 0, {reinterpret_cast<std::byte*>(packed.data<std::uint8_t>()), record.nbytes});
    if (native_projection && result.shape_.size() == 2 && supports_native(model, name)) {
        auto prepared = mlx_native_gguf_pack(tyloi::metal::array(result.bytes_), result.format_, result.output_size(), result.input_size(), result.row_bytes_);
        eval(std::vector<array>(prepared.planes.begin(), prepared.planes.end()));
        for (const auto& plane : prepared.planes) {
            tyloi::metal::AllocationScope::record(plane);
            tyloi::weights::observe_weight(tyloi::metal::tensor(plane));
        }
        result.native_ = std::make_shared<MlxNativeGgufWeight>(std::move(prepared));
    }
    if (!result.native_ && gguf_staged_enabled() && result.expert_count() <= 512)
        MlxGgufStagedPrefill::prepare(result);
    tyloi::metal::AllocationScope::record(tyloi::metal::array(result.bytes_));
    if (result.planes_) for (const auto& plane : *result.planes_) {
        tyloi::metal::AllocationScope::record(tyloi::metal::array(plane));
        tyloi::weights::observe_weight(plane);
    }
    model.record_prepared(name);
    return result;
}

} // namespace tyloi::metal

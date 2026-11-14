#include "read.h"
#include "engine/weights/encoding/fp8_sq_blob.h"

#include <cstring>

bool is_quant_dtype(const std::string& dtype) {
    return dtype == "TQ" || dtype == "NINT" || dtype == "NINT8-0" ||
        dtype == "NVQ" || dtype == "NPQ" || dtype == "NEPQ" || dtype == "MXFP4-SQ" ||
        tyloi::fp8sq::is_dtype(dtype) || dtype == "MXFP4" || dtype == "MXFP8" ||
        dtype == "MLX-AFFINE" || dtype.compare(0, 5, "GGUF_") == 0;
}

namespace tyloi {

const tyloi::TensorMetadata& require_tensor(
        const tyloi::source::Source& source,
        std::string_view name) {
    const auto* tensor = source.find_tensor(name);
    if (tensor == nullptr) {
        throw std::runtime_error("missing tensor: " + std::string(name));
    }
    return *tensor;
}

bool has_tensor(
        const tyloi::source::Source& source,
        std::string_view name) noexcept {
    return source.find_tensor(name) != nullptr;
}

bool has_tensor_prefix(
        const tyloi::source::Source& source,
        std::string_view prefix) {
    for (const auto& tensor : source.tensors()) {
        if (tensor.name == prefix ||
            (tensor.name.size() > prefix.size() &&
             tensor.name.compare(0, prefix.size(), prefix) == 0 &&
             tensor.name[prefix.size()] == '.')) {
            return true;
        }
    }
    return false;
}

std::vector<std::uint8_t> read_tensor(
        const tyloi::source::Source& source,
        std::string_view name) {
    const auto& tensor = require_tensor(source, name);
    if (tensor.nbytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("model tensor is too large to read");
    }
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(tensor.nbytes));
    source.read_range_into(
        name, 0, reinterpret_cast<std::byte*>(result.data()), result.size());
    return result;
}

std::vector<std::uint8_t> read_asset(
        const tyloi::source::Source& source,
        std::string_view name) {
    const auto bytes = source.read_asset(name);
    std::vector<std::uint8_t> result(bytes.size());
    if (!bytes.empty()) {
        std::memcpy(result.data(), bytes.data(), bytes.size());
    }
    return result;
}

std::string read_asset_text(
        const tyloi::source::Source& source,
        std::string_view name) {
    const auto bytes = source.read_asset(name);
    return {
        reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

} // namespace tyloi

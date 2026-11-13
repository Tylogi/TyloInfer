#include "engine/tensor.h"

#include <algorithm>
#include <stdexcept>

namespace tyloi {
namespace {
std::size_t normalize_dimension(std::int64_t dimension, std::size_t rank) {
    const auto count = static_cast<std::int64_t>(rank);
    const auto normalized = dimension < 0 ? dimension + count : dimension;
    if (normalized < 0 || normalized >= count)
        throw std::out_of_range("tensor dimension is out of range");
    return static_cast<std::size_t>(normalized);
}
} // namespace

DenseTensor narrow(const DenseTensor& source,
    std::int64_t dimension,
    std::int64_t start,
    std::int64_t length) {
    const auto& view = source.view_descriptor();
    if (!source.defined()) {
        throw std::invalid_argument("cannot narrow an undefined tensor");
    }
    const auto selected = normalize_dimension(dimension, view.rank);
    const auto extent = view.sizes[selected];
    auto normalized_start = start < 0 ? start + extent : start;
    if (normalized_start < 0 || length < 0 || normalized_start + length > extent) {
        throw std::out_of_range("narrow range is out of bounds");
    }
    auto result = view;
    result.sizes[selected] = length;
    const auto offset_elements = normalized_start * view.strides[selected];
    const auto offset_bytes = static_cast<std::size_t>(offset_elements) * source.element_size();
    result.data = static_cast<std::byte*>(view.data) + offset_bytes;
    return source.alias(result, source.byte_offset() + offset_bytes);
}

DenseTensor slice(const DenseTensor& source,
    std::int64_t dimension,
    std::int64_t start,
    std::int64_t end,
    std::int64_t step) {
    const auto& view = source.view_descriptor();
    if (step <= 0) {
        throw std::invalid_argument("native slice requires a positive step");
    }
    const auto selected = normalize_dimension(dimension, view.rank);
    const auto extent = view.sizes[selected];
    auto first = start < 0 ? start + extent : start;
    auto last = end < 0 ? end + extent : end;
    first = std::clamp<std::int64_t>(first, 0, extent);
    last = std::clamp<std::int64_t>(last, 0, extent);
    const auto length = last <= first ? 0 : (last - first + step - 1) / step;
    auto narrowed = narrow(source, dimension, first, std::max<std::int64_t>(0, last - first));
    auto result = narrowed.view_descriptor();
    result.sizes[selected] = length;
    result.strides[selected] *= step;
    return narrowed.alias(result, narrowed.byte_offset());
}

} // namespace tyloi

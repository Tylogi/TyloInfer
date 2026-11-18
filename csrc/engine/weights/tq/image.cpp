#include "engine/weights/tq/image.h"
#include "engine/backends/cuda/native/tensor/backend.h"

#include <climits>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
std::pair<std::size_t, std::size_t> image_offsets(const tyloi::TqLayout& layout) {
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (layout.packed > limit - 3) throw std::overflow_error("TQ image offset overflow");
    const auto words = (layout.packed + 3) & ~std::size_t(3);
    if (layout.packed_bytes > limit - words || words + layout.packed_bytes > limit - 32)
        throw std::overflow_error("TQ image size overflow");
    return {words, words + layout.packed_bytes};
}
}

std::vector<std::uint8_t> tq_image(
        const std::vector<std::uint8_t>& blob, const tyloi::TqLayout& layout) {
    // Keep the packed words aligned in the single image consumed by CUDA.
    const auto [words_offset, relative_offset] = image_offsets(layout);
    if (layout.packed > blob.size() || layout.packed_bytes > blob.size() - layout.packed)
        throw std::invalid_argument("TQ image source is too short");
    std::vector<std::uint8_t> image(relative_offset + 32);
    std::memcpy(image.data(), blob.data(), layout.packed);
    std::memcpy(image.data() + words_offset,
                blob.data() + layout.packed, layout.packed_bytes);
    std::memcpy(image.data() + relative_offset, tyloi::kTqRelative, sizeof(tyloi::kTqRelative));
    return image;
}

::tyloi::encoding::TqTensor tq_view(::tyloi::DenseTensor image,
                 const tyloi::TqLayout& layout, std::int64_t columns) {
    if (columns < 1 || columns > layout.width || layout.rows > INT_MAX ||
        layout.width > INT_MAX) {
        throw std::invalid_argument("invalid TQ CUDA logical shape");
    }
    const auto [words_offset, relative_offset] = image_offsets(layout);
    int gslog = 0;
    for (auto gs = layout.gs; gs > 1; gs >>= 1) ++gslog;
    return ::tyloi::encoding::tq_tensor(image, layout.rows, columns,
        {words_offset, layout.packed_bytes, layout.scales, layout.scale_bytes,
         relative_offset, relative_offset + 32, layout.width, gslog, layout.columns});
}

::tyloi::encoding::TqTensor promote_tq(const ::tyloi::encoding::TqTensor& weight,
                   const ::tyloi::Device& device) {
    auto image = weight.image().to(device,false,false).contiguous();
    return ::tyloi::encoding::tq_tensor(image, weight.size(0), weight.size(1),
                                        weight.layout());
}

::tyloi::encoding::TqTensor to_device_tq(const std::vector<std::uint8_t>& blob,
                      std::int64_t columns, int device, bool cuda) {
    const tyloi::TqLayout layout(blob.data(), blob.size());
    const auto image = tq_image(blob, layout);
    const MfqCudaGuard guard(device);
    auto storage = ::tyloi::cuda::from_blob(
        const_cast<std::uint8_t*>(image.data()),
        {static_cast<std::int64_t>(image.size())},
        ::tyloi::TensorOptions().dtype(::tyloi::kUInt8)).clone();
    if (cuda) {
        storage = storage.to(
            ::tyloi::Device(::tyloi::kCUDA, device),
            false, false).contiguous();
    }
    return tq_view(std::move(storage), layout, columns);
}

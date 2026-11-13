#include "tensor.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <vector>

namespace tyloi::cpu::ops {
namespace {
const DenseTensor& checked(const Tensor& tensor) {
    const auto* dense = tensor.representation<DenseTensor>();
    if (!tensor.defined() || !dense || tensor.is_encoded() ||
        tensor.device() != Device{DeviceType::cpu, 0} || tensor.options().pinned())
        throw std::invalid_argument("CPU storage requires an unpinned dense CPU:0 tensor");
    const auto& view = dense->view_descriptor();
    const auto width = scalar_size(view.scalar_type);
    if (!width || view.rank > kMaximumTensorRank || !dense->storage() ||
        dense->storage()->device != view.device || dense->storage()->pinned)
        throw std::invalid_argument("invalid CPU tensor storage");
    const auto count = view.numel();
    std::size_t max_offset = 0;
    for (std::size_t i = 0; i < view.rank; ++i) {
        if (view.strides[i] < 0) throw std::invalid_argument("negative CPU tensor stride");
        if (count && view.sizes[i] > 1) {
            const auto stride = static_cast<std::uint64_t>(view.strides[i]);
            const auto extent = static_cast<std::uint64_t>(view.sizes[i] - 1);
            if (stride > (std::numeric_limits<std::size_t>::max() - max_offset) / extent)
                throw std::overflow_error("CPU tensor stride overflow");
            max_offset += static_cast<std::size_t>(stride * extent);
        }
    }
    const auto bytes = dense->storage()->bytes;
    const auto offset = dense->byte_offset();
    if (offset > bytes || (count && (!dense->data_ptr() || max_offset > (bytes - offset) / width ||
        width > bytes - offset - max_offset * width)))
        throw std::invalid_argument("CPU tensor view exceeds its storage");
    return *dense;
}

std::size_t element_offset(const TensorView& view, std::int64_t index) {
    std::size_t offset = 0;
    for (std::size_t i = view.rank; i > 0; --i) {
        const auto dim = i - 1;
        offset += static_cast<std::size_t>(index % view.sizes[dim]) *
                  static_cast<std::size_t>(view.strides[dim]);
        index /= view.sizes[dim];
    }
    return offset * scalar_size(view.scalar_type);
}

std::uint16_t encode_float16(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
    const auto exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    auto mantissa = bits & 0x007fffffu;
    if (((bits >> 23) & 0xffu) == 0xffu) {
        if (mantissa == 0) return static_cast<std::uint16_t>(sign | 0x7c00u);
        return static_cast<std::uint16_t>(sign | 0x7e00u | (mantissa >> 13));
    }
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (exponent <= 0) {
        if (exponent < -10) return sign;
        mantissa |= 0x00800000u;
        const auto shift = static_cast<unsigned>(14 - exponent);
        auto rounded = mantissa >> shift;
        const auto remainder = mantissa & ((std::uint32_t{1} << shift) - 1);
        const auto halfway = std::uint32_t{1} << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    auto rounded = mantissa >> 13;
    const auto remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        ++rounded;
        if (rounded == 0x0400u) {
            rounded = 0;
            if (exponent + 1 >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
            return static_cast<std::uint16_t>(sign | ((exponent + 1) << 10));
        }
    }
    return static_cast<std::uint16_t>(sign | (exponent << 10) | rounded);
}
}

Tensor empty(std::span<const std::int64_t> shape, const TensorOptions& options) {
    if (options.target_device() != Device{DeviceType::cpu, 0} || options.pinned() ||
        !scalar_size(options.scalar_type()))
        throw std::invalid_argument("CPU allocation requires a valid dtype and unpinned CPU:0");
    // A non-null sentinel makes a rank-zero view count as one element.
    auto view = make_contiguous_view(reinterpret_cast<void*>(1), shape, options.scalar_type(), options.target_device());
    const auto bytes = view.nbytes();
    auto owner = std::shared_ptr<void>(::operator new(std::max<std::size_t>(bytes, 1)),
                                       [](void* p) { ::operator delete(p); });
    auto storage = std::make_shared<TensorStorage>();
    storage->owner = owner;
    storage->base = owner.get();
    storage->bytes = bytes;
    storage->device = options.target_device();
    view.data = storage->base;
    return DenseTensor(std::move(storage), view);
}

Tensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options) {
    auto result = empty(shape, options);
    fill(result, 0.0);
    return result;
}

Tensor clone(const Tensor& source) {
    const auto& input = checked(source);
    auto result = empty(input.sizes(), input.options());
    copy_into(result, source);
    return result;
}

void copy_into(Tensor destination, const Tensor& source) {
    const auto& dst = checked(destination);
    const auto& src = checked(source);
    if (dst.sizes() != src.sizes() || dst.scalar_type() != src.scalar_type())
        throw std::invalid_argument("CPU copy requires matching shape and dtype");
    const auto count = src.numel();
    if (!count) return;
    const auto bytes = src.nbytes();
    if (src.is_contiguous() && dst.is_contiguous()) {
        std::memmove(dst.data_ptr(), src.data_ptr(), bytes);
        return;
    }
    const auto width = src.element_size();
    const auto* in = static_cast<const std::byte*>(src.data_ptr());
    auto* out = static_cast<std::byte*>(dst.data_ptr());
    const auto& src_storage = *src.storage();
    const auto& dst_storage = *dst.storage();
    const auto src_base = reinterpret_cast<std::uintptr_t>(src_storage.base);
    const auto dst_base = reinterpret_cast<std::uintptr_t>(dst_storage.base);
    const bool shared_owner = src_storage.owner && dst_storage.owner &&
        !src_storage.owner.owner_before(dst_storage.owner) &&
        !dst_storage.owner.owner_before(src_storage.owner);
    const bool overlapping_storage = src_base <= dst_base
        ? dst_base - src_base < src_storage.bytes
        : src_base - dst_base < dst_storage.bytes;
    if (src_storage.base && dst_storage.base && !shared_owner && !overlapping_storage &&
        src_storage.base != dst_storage.base) {
        for (std::int64_t i = 0; i < count; ++i)
            std::memcpy(out + element_offset(dst.view_descriptor(), i),
                        in + element_offset(src.view_descriptor(), i), width);
        return;
    }
    std::vector<std::byte> scratch(bytes);
    for (std::int64_t i = 0; i < count; ++i)
        std::memcpy(scratch.data() + static_cast<std::size_t>(i) * width,
                    in + element_offset(src.view_descriptor(), i), width);
    for (std::int64_t i = 0; i < count; ++i)
        std::memcpy(out + element_offset(dst.view_descriptor(), i),
                    scratch.data() + static_cast<std::size_t>(i) * width, width);
}

void fill(Tensor& value, double fill_value) {
    const auto& dense = checked(value);
    if (!dense.is_contiguous()) throw std::invalid_argument("CPU fill requires a contiguous tensor");
    if (!dense.numel()) return;
    if (fill_value == 0.0) {
        std::memset(dense.data_ptr(), 0, dense.nbytes());
        return;
    }
    const auto count = dense.numel();
    switch (dense.scalar_type()) {
        case ScalarType::boolean: std::fill_n(dense.data_ptr<bool>(), count, fill_value != 0.0); break;
        case ScalarType::uint8: std::fill_n(dense.data_ptr<std::uint8_t>(), count, static_cast<std::uint8_t>(fill_value)); break;
        case ScalarType::int8: std::fill_n(dense.data_ptr<std::int8_t>(), count, static_cast<std::int8_t>(fill_value)); break;
        case ScalarType::int16: std::fill_n(dense.data_ptr<std::int16_t>(), count, static_cast<std::int16_t>(fill_value)); break;
        case ScalarType::int32: std::fill_n(dense.data_ptr<std::int32_t>(), count, static_cast<std::int32_t>(fill_value)); break;
        case ScalarType::int64: std::fill_n(dense.data_ptr<std::int64_t>(), count, static_cast<std::int64_t>(fill_value)); break;
        case ScalarType::float16: std::fill_n(dense.data_ptr<std::uint16_t>(), count, encode_float16(static_cast<float>(fill_value))); break;
        case ScalarType::bfloat16: {
            auto bits = std::bit_cast<std::uint32_t>(static_cast<float>(fill_value));
            bits += 0x7fffu + ((bits >> 16) & 1u);
            std::fill_n(dense.data_ptr<std::uint16_t>(), count, static_cast<std::uint16_t>(bits >> 16));
            break;
        }
        case ScalarType::float32: std::fill_n(dense.data_ptr<float>(), count, static_cast<float>(fill_value)); break;
        case ScalarType::float64: std::fill_n(dense.data_ptr<double>(), count, fill_value); break;
        default: throw std::invalid_argument("CPU fill for this scalar type is not implemented yet");
    }
}
}

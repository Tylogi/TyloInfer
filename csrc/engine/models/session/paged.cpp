#include "engine/models/session/paged.h"
#include "engine/ops/ops.h"
#include <algorithm>
#include <iterator>
#include <span>

#include <cstring>
#include <limits>

namespace {
using WireType = std::pair<int32_t, tyloi::ScalarType>;
constexpr WireType tensor_types[]{
    {1,tyloi::kUInt8}, {2,tyloi::kInt8}, {3,tyloi::kInt16}, {4,tyloi::kInt32}, {5,tyloi::kInt64},
    {6,tyloi::kFloat16}, {7,tyloi::kBFloat16}, {8,tyloi::kFloat32}, {12,tyloi::ScalarType::uint32}
};
}

void SessionBlockWriter::raw(const void *data, size_t size) {
    const auto *begin = static_cast<const uint8_t *>(data);
    bytes_.insert(bytes_.end(), begin, begin + size);
}

void SessionBlockWriter::tensor(const ::tyloi::Tensor &tensor, int original_device) {
    this->tensor(tensor, original_device < 0 ? tensor.device() :
        ::tyloi::Device(::tyloi::DeviceType::cuda, original_device));
}

void SessionBlockWriter::tensor(const ::tyloi::Tensor &tensor, ::tyloi::Device original_device) {
    const auto& source = tensor;
    if (!source.defined() || (!original_device.is_cuda() && !original_device.is_metal()) ||
        (source.device() != original_device && !source.device().is_cpu()) ||
        source.is_encoded() || source.dim() <= 0 || source.dim() > 8) {
        throw std::runtime_error("invalid paged cache tensor");
    }
    const auto dtype = source.scalar_type();
    const auto type = std::ranges::find(tensor_types, dtype, &WireType::second);
    if (type == std::end(tensor_types) || (dtype == ::tyloi::ScalarType::uint32 && !original_device.is_metal())) {
        throw std::runtime_error("unsupported paged cache tensor dtype");
    }
    for (int64_t axis = 0; axis < source.dim(); ++axis)
        if (source.size(axis) <= 0) throw std::runtime_error("invalid paged cache tensor shape");
    const auto host = ::tyloi::ops::contiguous(::tyloi::ops::copy(source, ::tyloi::Device(::tyloi::kCPU)));
    const auto* dense = host.representation<::tyloi::DenseTensor>();
    if (!dense) throw std::runtime_error("paged cache tensor is not host-readable");
    const auto bytes = host.nbytes();
    scalar<int32_t>(type->first);
    scalar<int32_t>(original_device.index);
    scalar<uint32_t>(static_cast<uint32_t>(source.dim()));
    scalar<uint32_t>(original_device.is_metal() ? 1 : 0);
    for (int64_t axis = 0; axis < source.dim(); ++axis)
        scalar<int64_t>(source.size(axis));
    scalar<uint64_t>(static_cast<uint64_t>(bytes));
    raw(dense->data_ptr(), bytes);
}

SessionBlockPayload SessionBlockWriter::finish() && {
    return std::make_shared<const std::vector<uint8_t>>(std::move(bytes_));
}

void SessionBlockReader::raw(void *destination, size_t size, const char *name) {
    if (remaining() < size) {
        throw CudaSessionStateError(std::string("truncated CUDA paged cache ") + name);
    }
    std::memcpy(destination, cursor_, size);
    cursor_ += size;
}

std::pair<::tyloi::Tensor, int> SessionBlockReader::tensor(bool skip) {
    const auto dtype_value = scalar<int32_t>("dtype");
    const auto device = scalar<int32_t>("device");
    const auto rank = scalar<uint32_t>("rank");
    const auto flags = scalar<uint32_t>("tensor flags");
    if (rank == 0 || rank > 8 || device < 0 || flags > 1 || (flags == 1 && device != 0)) {
        throw CudaSessionStateError("invalid CUDA paged cache tensor header");
    }
    const auto type = std::ranges::find(tensor_types, dtype_value, &WireType::first);
    if (type == std::end(tensor_types) || (type->second == ::tyloi::ScalarType::uint32 && flags != 1)) {
        throw CudaSessionStateError("invalid CUDA paged cache tensor dtype");
    }
    const auto dtype = type->second;
    std::vector<int64_t> shape;
    shape.reserve(rank);
    uint64_t elements = 1;
    for (uint32_t index = 0; index < rank; ++index) {
        const auto dimension = scalar<int64_t>("shape");
        if (dimension <= 0 ||
            elements > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dimension)) {
            throw CudaSessionStateError("invalid CUDA paged cache tensor shape");
        }
        shape.push_back(dimension);
        elements *= static_cast<uint64_t>(dimension);
    }
    const auto bytes = scalar<uint64_t>("tensor size");
    const uint64_t element_size = ::tyloi::scalar_size(dtype);
    if (elements > std::numeric_limits<uint64_t>::max() / element_size ||
        bytes != elements * element_size || bytes > remaining()) {
        throw CudaSessionStateError("invalid CUDA paged cache tensor size");
    }
    const auto* contents = cursor_;
    cursor_ += static_cast<size_t>(bytes);
    if (skip) return {};
    const auto data = std::span<const std::byte>(reinterpret_cast<const std::byte*>(contents),
                                                  static_cast<size_t>(bytes));
    auto view = ::tyloi::ops::from_blob(const_cast<std::byte*>(data.data()), shape,
        ::tyloi::TensorOptions().device(::tyloi::kCPU).dtype(dtype));
    auto cpu = ::tyloi::ops::clone(view);
    tensor_device_ = ::tyloi::Device(flags == 1 ? ::tyloi::DeviceType::metal : ::tyloi::DeviceType::cuda, device);
    return {std::move(cpu), device};
}

void SessionBlockReader::expect_end() const {
    if (cursor_ != end_) {
        throw CudaSessionStateError("trailing CUDA paged cache payload bytes");
    }
}

size_t SessionBlockReader::remaining() const { return static_cast<size_t>(end_ - cursor_); }

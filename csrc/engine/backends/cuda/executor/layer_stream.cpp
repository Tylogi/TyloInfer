#include "engine/weights/layer_stream.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/device.h"
#include "engine/backends/cuda/native/stream.h"
#include "engine/backends/cuda/native/tensor/tensor.h"
#include "engine/tensor/encoding/ggml.h"
#include "engine/weights/encoding/tq/tensor.h"
#include "engine/weights/memory.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tyloi::weights {
namespace {

Tensor replace_buffers(const Tensor& tensor, const std::vector<DenseTensor>& buffers) {
    if (const auto* tq = tensor.representation<encoding::TqTensor>())
        return std::make_shared<encoding::TqTensor>(buffers.at(0), tq->size(0), tq->size(1), tq->layout());
    if (const auto* ggml = tensor.representation<GgmlQuantizedTensor>(); ggml && (ggml->rotation || ggml->correction))
        throw std::invalid_argument("streamed GGML layers require self-contained weights");
    if (const auto* encoded = tensor.representation<EncodedTensor>())
        return encoded->with_buffers(buffers);
    if (tensor.is_encoded()) throw std::invalid_argument("unsupported streamed layer weight encoding");
    return buffers.at(0);
}

DenseTensor view(const DenseTensor& arena, const DenseTensor& source, std::size_t offset) {
    return DenseTensor(arena.storage(),
        make_contiguous_view(nullptr, source.sizes(), source.scalar_type(), arena.device()), offset);
}

} // namespace

struct LayerStream::Impl {
    struct Slot {
        DenseTensor bytes;
        cuda::Event ready, used;
        std::size_t layer = std::numeric_limits<std::size_t>::max();
        bool consumed = false;
    };

    Device device;
    cuda::Stream copy;
    std::vector<DenseTensor> layers;
    std::array<Slot, 2> slots;
    std::size_t slot_count, maximum = 0, host = 0;
    Slot* active = nullptr;
    cuda::StreamHandle compute;

    Impl(Device target, bool prefetch) : device(target), copy(target.index), slot_count(prefetch ? 2 : 1) {}

    Slot& slot(std::size_t index) { return slots[index % slot_count]; }

    void queue(std::size_t index) {
        auto& target = slot(index);
        if (target.layer == index) return;
        if (target.consumed) copy.wait(target.used);
        target.layer = std::numeric_limits<std::size_t>::max();
        const auto& source = layers.at(index);
        constexpr std::size_t chunk = 4u << 20;
        for (std::size_t offset = 0; offset < source.nbytes(); offset += chunk) {
            TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(
                static_cast<std::byte*>(target.bytes.data_ptr()) + offset,
                static_cast<const std::byte*>(source.data_ptr()) + offset,
                std::min(chunk, source.nbytes() - offset), cudaMemcpyHostToDevice, copy.get()));
        }
        target.ready.record(copy.get());
        target.layer = index;
        target.consumed = false;
    }
};

LayerStream::LayerStream(Device device, bool prefetch) {
    if (!device.is_cuda()) throw std::invalid_argument("layer weight streaming requires CUDA");
    const cuda::DeviceGuard guard(device.index);
    impl_ = std::make_unique<Impl>(device, prefetch);
}

LayerStream::~LayerStream() {
    const cuda::DeviceGuard guard(impl_->device.index);
    try {
        impl_->copy.synchronize();
        if (impl_->active) TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(impl_->compute.stream()));
        for (auto& slot : impl_->slots)
            if (slot.consumed) slot.used.synchronize();
    } catch (...) { std::terminate(); }
}

namespace {

std::size_t next_offset(const DenseTensor& buffer, std::size_t& bytes) {
    if (!buffer.defined()) return 0;
    if (!buffer.device().is_cpu() || !buffer.is_contiguous())
        throw std::invalid_argument("streamed layer weights require contiguous host buffers");
    constexpr auto limit = std::size_t(std::numeric_limits<std::int64_t>::max());
    if (bytes > limit - 255) throw std::overflow_error("streamed layer weight size overflow");
    const auto offset = (bytes + 255) & ~std::size_t(255);
    if (buffer.nbytes() > limit - offset) throw std::overflow_error("streamed layer weight size overflow");
    bytes = offset + buffer.nbytes();
    return offset;
}

} // namespace

void LayerStream::measure(const Tensor& tensor, std::size_t& bytes) {
    for (const auto* buffer : tensor.buffers()) next_offset(*buffer, bytes);
}

void LayerStream::allocate(std::span<const std::size_t> sizes) {
    if (sizes.empty() || std::find(sizes.begin(), sizes.end(), 0) != sizes.end())
        throw std::invalid_argument("streamed layer has no immutable weights");
    impl_->layers.reserve(sizes.size());
    impl_->maximum = *std::max_element(sizes.begin(), sizes.end());
    const cuda::DeviceGuard guard(impl_->device.index);
    const auto allocation_stream = cuda::current_stream(impl_->device.index);
    for (std::size_t i = 0; i < impl_->slot_count; ++i) {
        auto& slot = impl_->slots[i];
        slot.bytes = cuda::empty({static_cast<std::int64_t>(impl_->maximum)},
            TensorOptions{}.device(impl_->device).dtype(kUInt8));
        slot.ready.record(allocation_stream.stream());
        impl_->copy.wait(slot.ready);
        observe_weight(slot.bytes);
    }
}

void LayerStream::allocate_host(std::size_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() - impl_->host)
        throw std::overflow_error("streamed model weight size overflow");
    auto image = cuda::empty({static_cast<std::int64_t>(bytes)},
        TensorOptions{}.device(kCPU).dtype(kUInt8).pinned_memory(true));
    observe_weight(image);
    impl_->layers.push_back(std::move(image));
    impl_->host += bytes;
}

Tensor LayerStream::pack(std::size_t index, const Tensor& tensor, std::size_t& bytes) {
    const auto& host = impl_->layers.at(index);
    const auto& target = impl_->slot(index).bytes;
    std::vector<DenseTensor> buffers;
    for (const auto* buffer : tensor.buffers()) {
        const auto offset = next_offset(*buffer, bytes);
        if (bytes > host.nbytes()) throw std::logic_error("streamed layer weight layout changed during preparation");
        if (buffer->defined()) {
            if (buffer->nbytes())
                std::memcpy(host.data_ptr<std::byte>() + offset, buffer->data_ptr(), buffer->nbytes());
            buffers.push_back(view(target, *buffer, offset));
        } else buffers.emplace_back();
    }
    return replace_buffers(tensor, buffers);
}

void LayerStream::bind(const std::shared_ptr<LayerStream>& stream) {
    cuda::Executor::current().bind_layer_stream(stream);
}

void LayerStream::stage(std::size_t index) {
    const auto device = impl_->device;
    const cuda::DeviceGuard guard(device.index);
    const auto stream = cuda::current_stream(device.index);
    cudaStreamCaptureStatus capture;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &capture));
    if (capture != cudaStreamCaptureStatusNone)
        throw std::logic_error("streamed layer weights require eager CUDA execution");
    if (impl_->active) {
        impl_->active->used.record(impl_->compute.stream());
        impl_->active->consumed = true;
    }
    impl_->queue(index);
    auto& slot = impl_->slot(index);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), slot.ready.get(), 0));
    impl_->compute = stream;
    impl_->active = &slot;
    if (impl_->slot_count == 2 && index + 1 < impl_->layers.size()) impl_->queue(index + 1);
}

std::size_t LayerStream::host_bytes() const noexcept { return impl_->host; }
std::size_t LayerStream::device_bytes() const noexcept { return impl_->maximum * impl_->slot_count; }

} // namespace tyloi::weights

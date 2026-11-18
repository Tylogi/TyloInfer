#include "engine/backends/cuda/ops/gguf/rows.h"
#include "backend.h"
#include "engine/backends/cuda/kernels/projection/gguf/linear.h"
#include "engine/backends/cuda/native/allocation.h"
#include "engine/backends/cuda/native/memory.h"
#include "engine/backends/cuda/native/resident_allocations.h"
#include "engine/backends/cuda/ops/row_stage.h"
#include "engine/weights/rows/gguf.h"
#include "engine/weights/rows/gguf_read.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tyloi::cuda::gguf {
namespace {
using Tables = std::vector<std::shared_ptr<weights::GgufRows>>;

struct Layout {
    std::int64_t count = 1, width = 0;
    std::size_t stride = 0, offset = 0, bytes = 0, payload = 0, output = 0;
};

Layout layout(const Tables& tables, std::int64_t rows,
              const std::vector<std::int64_t>& shape, Device device) {
    if (tables.empty() || !tables.front() || rows <= 0 || (!device.is_cpu() && !device.is_cuda()))
        throw std::invalid_argument("invalid GGUF row stage");
    Layout result;
    for (const auto extent : shape) {
        if (extent <= 0 || result.count > INT64_MAX / extent)
            throw std::overflow_error("invalid GGUF row stage shape");
        result.count *= extent;
    }
    result.width = tables.front()->width();
    if (result.width <= 0) throw std::invalid_argument("invalid GGUF row width");
    for (std::size_t i = 0; i < tables.size(); ++i) {
        const auto& table = tables[i];
        if (!table || table->width() != result.width || table->rows() <= 0 ||
            table->rows() > rows || (i + 1 < tables.size() && table->rows() != rows))
            throw std::invalid_argument("invalid GGUF row shard geometry");
        if (!table->staged_row_bytes() || table->staged_row_bytes() > SIZE_MAX - 7)
            throw std::overflow_error("GGUF staged row size exceeds bounds");
        result.stride = std::max(result.stride, std::size_t(table->staged_row_bytes()));
    }
    result.stride = (result.stride + 7) & ~std::size_t(7);
    const auto count = std::size_t(result.count);
    if (count > SIZE_MAX / sizeof(StagedRow) || count > SIZE_MAX / result.stride)
        throw std::overflow_error("GGUF row stage packet exceeds bounds");
    result.offset = count * sizeof(StagedRow);
    result.payload = count * result.stride;
    if (result.payload > SIZE_MAX - result.offset || result.offset + result.payload > INT64_MAX)
        throw std::overflow_error("GGUF row stage packet exceeds tensor limits");
    result.bytes = result.offset + result.payload;
    auto output_shape = shape;
    output_shape.push_back(result.width);
    result.output = make_contiguous_view(nullptr, output_shape,
        device.is_cuda() ? kFloat16 : kFloat32, device).nbytes();
    if (device.is_cuda() && result.count > (INT64_MAX - 255) / result.width)
        throw std::overflow_error("GGUF row stage kernel geometry exceeds bounds");
    return result;
}

void validate_ids(const Tables& tables, std::int64_t rows, const Layout& prepared,
                  const std::vector<std::int64_t>& ids) {
    if (ids.size() != std::size_t(prepared.count))
        throw std::invalid_argument("GGUF row stage upload shape differs");
    for (const auto id : ids) {
        if (id < 0 || std::uint64_t(id / rows) >= tables.size() ||
            id % rows >= tables[std::size_t(id / rows)]->rows())
            throw std::out_of_range("GGUF row stage shard id");
    }
}

bool mapped_packet(const Layout& prepared, Device device) {
    int can_map = 0;
    TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, device.index));
    return can_map && prepared.bytes <= 16 * 1024;
}

class ChargedHost {
public:
    ChargedHost(std::size_t bytes, bool mapped)
        : buffer_(std::make_unique<HostBuffer>(bytes, mapped)), bytes_(bytes) {
        charge_tensor_host_bytes(bytes_);
    }
    ~ChargedHost() { buffer_.reset(); tensor_host_bytes.fetch_sub(bytes_); }
    void* data() const { return buffer_->data(); }
private:
    std::unique_ptr<HostBuffer> buffer_;
    std::size_t bytes_;
};

class GgufRowStage final : public RowStage {
    Tables tables_;
    std::shared_ptr<weights::GgufRead> read_;
    std::vector<std::int64_t> shape_;
    std::int64_t rows_;
    Layout layout_;
    Device device_;
    std::vector<StreamHandle> streams_;
    RowStageFence fence_;
    std::unique_ptr<ChargedHost> host_;
    DenseTensor packet_;
    Tensor output_;
    void* packet_address_ = nullptr;
    bool mapped_ = false;

    StreamHandle retain_stream() {
        const auto stream = current_stream(device_.index);
        if (std::none_of(streams_.begin(), streams_.end(), [&](const auto& prior) {
                return prior.stream() == stream.stream(); })) streams_.push_back(stream);
        return stream;
    }
    void prepare(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape) {
        if (shape != shape_) throw std::invalid_argument("GGUF row stage upload shape differs");
        validate_ids(tables_, rows_, layout_, ids);
        auto payload = read_->collect(ids, shape);
        if (!payload.is_cpu() || payload.scalar_type() != kUInt8 || !payload.is_contiguous() ||
            payload.dim() != 2 || payload.size(0) != layout_.count ||
            payload.size(1) != std::int64_t(layout_.stride))
            throw std::runtime_error("GGUF row reader returned invalid payload");
        auto* descriptors = static_cast<StagedRow*>(host_->data());
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const auto& table = tables_[std::size_t(ids[i] / rows_)];
            descriptors[i] = {layout_.offset + i * layout_.stride, std::int32_t(table->staged_type()), 0};
        }
        std::memcpy(static_cast<std::byte*>(host_->data()) + layout_.offset,
                    payload.data_ptr(), layout_.payload);
    }

public:
    GgufRowStage(Tables tables, std::int64_t rows, std::vector<std::int64_t> shape,
                 Device device, std::shared_ptr<weights::GgufRead> read)
        : tables_(std::move(tables)), read_(std::move(read)), shape_(std::move(shape)),
          rows_(rows), layout_(layout(tables_, rows_, shape_, device)), device_(device) {
        if (!device_.is_cuda()) throw std::invalid_argument("GGUF row stage requires CUDA");
        if (!read_) read_ = std::make_shared<weights::GgufRead>(tables_, rows_);
        const ResidentAllocationScope resident;
        const AllocationScope allocation;
        const MfqCudaGuard guard(device_);
        const auto stream = retain_stream();
        mapped_ = mapped_packet(layout_, device_);
        host_ = std::make_unique<ChargedHost>(layout_.bytes, mapped_);
        if (mapped_) TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&packet_address_, host_->data(), 0));
        else {
            packet_ = cuda::empty({std::int64_t(layout_.bytes)},
                TensorOptions().device(device_).dtype(kUInt8));
            packet_address_ = packet_.data_ptr();
        }
        auto output_shape = shape_;
        output_shape.push_back(layout_.width);
        output_ = cuda::zeros(output_shape, TensorOptions().device(device_).dtype(kFloat16));
        std::memset(host_->data(), 0, layout_.bytes);
        auto* descriptors = static_cast<StagedRow*>(host_->data());
        for (std::int64_t i = 0; i < layout_.count; ++i)
            descriptors[i] = {layout_.offset + std::size_t(i) * layout_.stride,
                              std::int32_t(tables_.front()->staged_type()), 0};
        try {
            if (!mapped_) TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(packet_address_, host_->data(), layout_.bytes,
                cudaMemcpyHostToDevice, stream.stream()));
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        } catch (...) {
            if (cudaStreamSynchronize(stream.stream()) != cudaSuccess) std::terminate();
            throw;
        }
    }
    ~GgufRowStage() override {
        try {
            read_->clear();
            const MfqCudaGuard guard(device_);
            for (const auto& stream : streams_) TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
            output_ = {};
            packet_ = {};
            host_.reset();
        } catch (...) { std::terminate(); }
    }
    const Tensor& output() const override { return output_; }
    bool mapped_rows() const override { return mapped_; }
    void upload(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape) override {
        const MfqCudaGuard guard(device_);
        const auto stream = retain_stream();
        fence_.before_write();
        prepare(ids, shape);
        if (!mapped_) {
            packet_.record_stream(reinterpret_cast<std::uintptr_t>(stream.stream()), stream.lifetime());
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(packet_address_, host_->data(), layout_.bytes,
                cudaMemcpyHostToDevice, stream.stream()));
            fence_.record_transfer(stream.stream());
        }
    }
    void upload_graph(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape) override {
        if (!mapped_) { upload(ids, shape); return; }
        fence_.owner_fenced();
        prepare(ids, shape);
    }
    void decode() override {
        const MfqCudaGuard guard(device_);
        const auto stream = retain_stream();
        const auto capture = fence_.join(stream.stream());
        output_.record_stream(reinterpret_cast<std::uintptr_t>(stream.stream()), stream.lifetime());
        if (packet_.defined())
            packet_.record_stream(reinterpret_cast<std::uintptr_t>(stream.stream()), stream.lifetime());
        staged_rows(packet_address_, layout_.count, layout_.width, output_.dense().data_ptr(), true, stream.stream());
        fence_.record_consumer(stream.stream(), capture);
    }
};
} // namespace

std::shared_ptr<RowStage> make_row_stage(const Tables& tables, std::int64_t rows,
        const std::vector<std::int64_t>& shape, Device device, std::shared_ptr<weights::GgufRead> read) {
    return std::make_shared<GgufRowStage>(tables, rows, shape, device, std::move(read));
}

std::pair<std::size_t, std::size_t> row_stage_growth(const Tables& tables, std::int64_t rows,
        const std::vector<std::int64_t>& shape, Device device) {
    const auto prepared = layout(tables, rows, shape, device);
    if (device.is_cpu()) return {prepared.output, 0};
    const auto packet = mapped_packet(prepared, device) ? 0 : prepared.bytes;
    if (prepared.payload > SIZE_MAX - prepared.bytes || prepared.output > SIZE_MAX - packet)
        throw std::overflow_error("GGUF row stage growth exceeds bounds");
    return {prepared.bytes + prepared.payload, packet + prepared.output};
}

Tensor lookup_rows(const Tables& tables, std::int64_t rows, const std::vector<std::int64_t>& ids,
        const std::vector<std::int64_t>& shape, Device device, std::shared_ptr<weights::GgufRead> read) {
    if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
        if (std::any_of(shape.begin(), shape.end(), [](auto extent) { return extent < 0; }) || !ids.empty())
            throw std::invalid_argument("GGUF row lookup shape differs");
        const auto prepared = layout(tables, rows, {}, device);
        if (read) read->clear();
        auto output_shape = shape;
        output_shape.push_back(prepared.width);
        return cuda::empty(output_shape, TensorOptions().device(device).dtype(
            device.is_cuda() ? kFloat16 : kFloat32));
    }
    const auto prepared = layout(tables, rows, shape, device);
    validate_ids(tables, rows, prepared, ids);
    if (device.is_cpu()) {
        if (read) read->clear();
        auto output_shape = shape;
        output_shape.push_back(prepared.width);
        auto output = cuda::empty(output_shape, TensorOptions().device(device).dtype(kFloat32));
        for (std::size_t i = 0; i < ids.size(); ++i)
            tables[std::size_t(ids[i] / rows)]->decode_row(ids[i] % rows,
                output.data_ptr<float>() + i * std::size_t(prepared.width));
        return output;
    }
    auto stage = make_row_stage(tables, rows, shape, device, std::move(read));
    stage->upload(ids, shape);
    stage->decode();
    return stage->output();
}
} // namespace tyloi::cuda::gguf

#include "backend.h"
#include "engine/backends/cuda/kernels/embedding/raw.h"
#include "engine/backends/cuda/ops/row_stage.h"
#include "engine/weights/encoding/rows.h"
#include "engine/backends/cuda/native/memory.h"
#include <cstdint>
#include <limits>

namespace {
class DenseRowStage final : public RowStage {
public:
    DenseRowStage(std::vector<tyloi::Tensor> tables, std::int64_t rows,
                  std::vector<std::int64_t> shape, tyloi::Device device)
        : tables_(std::move(tables)), rows_(rows), shape_(std::move(shape)), device_(device) {
        if (!device.is_cuda() || tables_.empty() || rows_ <= 0)
            throw std::invalid_argument("invalid dense row stage");
        const MfqCudaGuard guard(device_);
        auto output_shape = shape_;
        output_shape.push_back(tables_.front().size(1));
        output_ = tyloi::cuda::zeros(output_shape,
            tyloi::TensorOptions().device(device_).dtype(tables_.front().scalar_type()));
        bytes_ = output_.nbytes();
        host_ = std::make_unique<tyloi::cuda::HostBuffer>(bytes_);
        tyloi::cuda::charge_tensor_host_bytes(bytes_);
    }
    ~DenseRowStage() override {
        if (pending_) {
            try {
                const MfqCudaGuard guard(device_);
                TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream_.stream()));
            } catch (...) { std::terminate(); }
        }
        tyloi::cuda::tensor_host_bytes.fetch_sub(bytes_);
    }
    const tyloi::Tensor& output() const override { return output_; }
    bool mapped_rows() const override { return false; }
    // Dense rows reach their final representation during upload.
    void decode() override {}
    void upload(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape) override {
        const auto width = tables_.front().size(1);
        if (shape != shape_ || ids.size() != static_cast<std::size_t>(output_.numel() / width))
            throw std::invalid_argument("dense row upload shape differs");
        const auto row_bytes = static_cast<std::size_t>(width) * output_.element_size();
        std::vector<std::vector<tyloi::weights::Rows::Row>> requests(tables_.size());
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const auto id = ids[i];
            if (id < 0 || static_cast<std::uint64_t>(id / rows_) >= tables_.size() ||
                id % rows_ >= tables_[id / rows_].size(0))
                throw std::out_of_range("dense row shard id is out of bounds");
            requests[id / rows_].emplace_back(id % rows_,
                static_cast<std::byte*>(host_->data()) + i * row_bytes);
        }
        const MfqCudaGuard guard(device_);
        if (pending_) {
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream_.stream()));
            pending_ = false;
        }
        for (std::size_t i = 0; i < requests.size(); ++i)
            if (!requests[i].empty()) tables_[i].representation<tyloi::weights::Rows>()->read_rows(requests[i]);
        stream_ = tyloi::cuda::current_stream(device_.index);
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(output_.dense().data_ptr(), host_->data(), bytes_,
            cudaMemcpyHostToDevice, stream_.stream()));
        pending_ = true;
    }
private:
    std::vector<tyloi::Tensor> tables_;
    std::int64_t rows_;
    std::vector<std::int64_t> shape_;
    tyloi::Device device_;
    tyloi::Tensor output_;
    std::size_t bytes_ = 0;
    std::unique_ptr<tyloi::cuda::HostBuffer> host_;
    tyloi::cuda::StreamHandle stream_;
    bool pending_ = false;
};
}

std::shared_ptr<RowStage> tyloi::cuda::make_dense_row_stage(const std::vector<Tensor>& tables,
        std::int64_t rows_per_shard, const std::vector<std::int64_t>& shape, Device device) {
    return std::make_shared<DenseRowStage>(tables, rows_per_shard, shape, device);
}

::tyloi::DenseTensor embedding_lookup_cuda(
        ::tyloi::DenseTensor weight,
        ::tyloi::DenseTensor token_ids) {
    TYLOINFER_RUNTIME_CHECK(
        weight.is_cuda() && weight.is_contiguous() && weight.dim() == 2,
        "embedding weight must be contiguous CUDA rank-2");
    TYLOINFER_RUNTIME_CHECK(
        weight.scalar_type() == ::tyloi::kFloat32 ||
        weight.scalar_type() == ::tyloi::kFloat16,
        "embedding weight must be fp16 or fp32");
    TYLOINFER_RUNTIME_CHECK(
        token_ids.is_cuda() && token_ids.is_contiguous() &&
        token_ids.scalar_type() == ::tyloi::kInt64 &&
        token_ids.device() == weight.device(),
        "embedding token IDs must be contiguous CUDA int64 on the weight device");
    const int vocabulary = static_cast<int>(weight.size(0));
    const int width = static_cast<int>(weight.size(1));
    const int token_count = static_cast<int>(token_ids.numel());
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = ::tyloi::cuda::empty(shape, weight.options());
    if (token_count == 0) {
        return output;
    }
    tyloi::cuda::embedding_raw::dense(weight.data_ptr(), token_ids.data_ptr<int64_t>(), output.data_ptr(),
        token_count, width, vocabulary, weight.scalar_type() == ::tyloi::kFloat16
            ? tyloi::cuda::NumericType::f16 : tyloi::cuda::NumericType::f32, mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


#if TYLOINFER_ENABLE_MFQ_COMPAT
::tyloi::DenseTensor nint_selected_rows_cuda(
        ::tyloi::DenseTensor packed,
        ::tyloi::DenseTensor descriptors, int64_t width) {
    TYLOINFER_RUNTIME_CHECK(packed.is_cuda() && packed.is_contiguous() &&
        packed.scalar_type() == ::tyloi::kUInt8 && packed.dim() == 1,
        "NINT selected payload must be CUDA uint8 rank-1");
    TYLOINFER_RUNTIME_CHECK(descriptors.is_cuda() && descriptors.is_contiguous() &&
        descriptors.scalar_type() == ::tyloi::kInt32 &&
        descriptors.dim() == 2 && descriptors.size(1) == 6 &&
        descriptors.device() == packed.device(),
        "NINT selected descriptors must be CUDA int32 [rows,6] on the payload device");
    TYLOINFER_RUNTIME_CHECK(width > 0 && width <= std::numeric_limits<int>::max() &&
        descriptors.size(0) <= std::numeric_limits<int>::max() / 6 &&
        static_cast<uint64_t>(descriptors.size(0)) * width <= std::numeric_limits<uint32_t>::max(),
        "NINT selected row dimensions exceed bounds");
    const MfqCudaGuard guard(packed.device());
    auto output = ::tyloi::cuda::empty({descriptors.size(0), width},
        packed.options().dtype(::tyloi::kFloat16));
    if (descriptors.size(0) == 0) return output;
    TYLOINFER_RUNTIME_CHECK(packed.numel() > 0, "NINT selected payload is empty");
    const size_t total = static_cast<size_t>(descriptors.size(0)) * width;
    tyloi::cuda::embedding_raw::selected(packed.data_ptr<uint8_t>(), descriptors.data_ptr<int32_t>(),
        nullptr, output.data_ptr(), total, int(width), mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


void nint_selected_rows_into_cuda(
        ::tyloi::DenseTensor packed, ::tyloi::DenseTensor descriptors,
        ::tyloi::DenseTensor inverse, ::tyloi::DenseTensor output) {
    TYLOINFER_RUNTIME_CHECK(packed.is_cuda() && packed.is_contiguous() &&
        packed.scalar_type() == ::tyloi::kUInt8 && packed.dim() == 1 && packed.numel() > 0,
        "NINT selected payload must be nonempty CUDA uint8 rank-1");
    TYLOINFER_RUNTIME_CHECK(descriptors.is_cuda() && descriptors.is_contiguous() &&
        descriptors.scalar_type() == ::tyloi::kInt32 && descriptors.dim() == 2 &&
        descriptors.size(1) == 6 && descriptors.device() == packed.device(),
        "NINT selected descriptors must be CUDA int32 [rows,6]");
    TYLOINFER_RUNTIME_CHECK(inverse.is_cuda() && inverse.is_contiguous() &&
        inverse.scalar_type() == ::tyloi::kInt64 && inverse.dim() == 1 &&
        inverse.device() == packed.device() && output.device() == packed.device() &&
        output.is_contiguous() && output.scalar_type() == ::tyloi::kFloat16 &&
        output.dim() == 2 && output.size(0) == inverse.numel() && output.size(1) > 0 &&
        output.size(1) <= std::numeric_limits<int>::max() &&
        output.numel() <= std::numeric_limits<uint32_t>::max(),
        "NINT selected inverse/output geometry differs");
    if (!output.numel()) return;
    const MfqCudaGuard guard(packed.device());
    tyloi::cuda::embedding_raw::selected(packed.data_ptr<uint8_t>(), descriptors.data_ptr<int32_t>(),
        inverse.data_ptr<int64_t>(), output.data_ptr(), output.numel(), int(output.size(1)), mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
}

::tyloi::DenseTensor nint_embedding_cuda(
        ::tyloi::DenseTensor bitstream,
        ::tyloi::DenseTensor row_q_bits,
        ::tyloi::DenseTensor row_q_bit_offsets,
        ::tyloi::DenseTensor subgroup_scale,
        ::tyloi::DenseTensor subgroup_minimum,
        ::tyloi::DenseTensor neuron_scale,
        ::tyloi::DenseTensor neuron_minimum,
        ::tyloi::DenseTensor token_ids,
        int64_t width,
        int64_t group_size) {
    TYLOINFER_RUNTIME_CHECK(
        bitstream.is_cuda() && bitstream.is_contiguous() &&
        bitstream.scalar_type() == ::tyloi::kUInt8 &&
        bitstream.dim() == 1,
        "NINT embedding bitstream must be contiguous CUDA uint8 rank-1");
    TYLOINFER_RUNTIME_CHECK(
        row_q_bits.is_cuda() && row_q_bits.is_contiguous() &&
        row_q_bits.scalar_type() == ::tyloi::kUInt8 &&
        row_q_bits.dim() == 1 && row_q_bit_offsets.is_cuda() &&
        row_q_bit_offsets.is_contiguous() &&
        row_q_bit_offsets.scalar_type() == ::tyloi::kInt64 &&
        row_q_bit_offsets.dim() == 1,
        "NINT embedding q metadata is invalid");
    TYLOINFER_RUNTIME_CHECK(
        subgroup_scale.is_cuda() && subgroup_scale.is_contiguous() &&
        subgroup_scale.scalar_type() == ::tyloi::kUInt8 &&
        subgroup_scale.dim() == 2 && subgroup_minimum.is_cuda() &&
        subgroup_minimum.is_contiguous() &&
        subgroup_minimum.scalar_type() == ::tyloi::kUInt8 &&
        subgroup_minimum.sizes() == subgroup_scale.sizes(),
        "NINT embedding subgroup metadata is invalid");
    TYLOINFER_RUNTIME_CHECK(
        neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == ::tyloi::kFloat32 &&
        neuron_minimum.is_cuda() && neuron_minimum.is_contiguous() &&
        neuron_minimum.scalar_type() == ::tyloi::kFloat32,
        "NINT embedding neuron metadata is invalid");
    TYLOINFER_RUNTIME_CHECK(
        token_ids.is_cuda() && token_ids.is_contiguous() &&
        token_ids.scalar_type() == ::tyloi::kInt64,
        "NINT embedding token IDs must be contiguous CUDA int64");
    const int vocabulary = static_cast<int>(subgroup_scale.size(0));
    const int groups = static_cast<int>(subgroup_scale.size(1));
    TYLOINFER_RUNTIME_CHECK(
        vocabulary > 0 && groups > 0 && group_size >= 4 && group_size <= 64 &&
        width > 0 && width <= static_cast<int64_t>(groups) * group_size,
        "NINT embedding dimensions are invalid");
    TYLOINFER_RUNTIME_CHECK(
        row_q_bits.numel() == vocabulary &&
        row_q_bit_offsets.numel() == vocabulary &&
        neuron_scale.numel() == vocabulary &&
        neuron_minimum.numel() == vocabulary,
        "NINT embedding row metadata shape mismatch");
    TYLOINFER_RUNTIME_CHECK(
        bitstream.device() == row_q_bits.device() &&
        bitstream.device() == row_q_bit_offsets.device() &&
        bitstream.device() == subgroup_scale.device() &&
        bitstream.device() == subgroup_minimum.device() &&
        bitstream.device() == neuron_scale.device() &&
        bitstream.device() == neuron_minimum.device() &&
        bitstream.device() == token_ids.device(),
        "NINT embedding tensors must share one CUDA device");
    const int token_count = static_cast<int>(token_ids.numel());
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = ::tyloi::cuda::empty(
        shape, bitstream.options().dtype(::tyloi::kFloat16));
    if (token_count == 0) {
        return output;
    }
    tyloi::cuda::embedding_raw::nint(bitstream.data_ptr<uint8_t>(), row_q_bits.data_ptr<uint8_t>(),
        row_q_bit_offsets.data_ptr<int64_t>(), subgroup_scale.data_ptr<uint8_t>(), subgroup_minimum.data_ptr<uint8_t>(),
        neuron_scale.data_ptr<float>(), neuron_minimum.data_ptr<float>(), token_ids.data_ptr<int64_t>(), output.data_ptr(),
        token_count, vocabulary, groups, int(group_size), int(width), mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


::tyloi::DenseTensor nint8_zero_embedding_lookup_cuda(
        ::tyloi::DenseTensor quantized,
        ::tyloi::DenseTensor scale,
        ::tyloi::DenseTensor token_ids,
        int64_t width) {
    TYLOINFER_RUNTIME_CHECK(
        quantized.is_cuda() && quantized.is_contiguous() &&
        quantized.scalar_type() == ::tyloi::kUInt8 &&
        quantized.dim() == 3 && quantized.size(2) == 32,
        "NINT8-0 embedding q must be contiguous CUDA uint8 [V,G,32]");
    TYLOINFER_RUNTIME_CHECK(
        scale.is_cuda() && scale.is_contiguous() &&
        scale.scalar_type() == ::tyloi::kFloat16 &&
        scale.dim() == 2 && scale.size(0) == quantized.size(0) &&
        scale.size(1) == quantized.size(1),
        "NINT8-0 embedding scale must be contiguous CUDA fp16 [V,G]");
    TYLOINFER_RUNTIME_CHECK(
        token_ids.is_cuda() && token_ids.is_contiguous() &&
        token_ids.scalar_type() == ::tyloi::kInt64 &&
        token_ids.device() == quantized.device() &&
        scale.device() == quantized.device(),
        "NINT8-0 embedding tensors must share one CUDA device");
    TYLOINFER_RUNTIME_CHECK(
        width > 0 && width <= quantized.size(1) * 32,
        "NINT8-0 embedding width is invalid");
    const int vocabulary = static_cast<int>(quantized.size(0));
    const int groups = static_cast<int>(quantized.size(1));
    const int token_count = static_cast<int>(token_ids.numel());
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = ::tyloi::cuda::empty(
        shape, quantized.options().dtype(::tyloi::kFloat16));
    if (token_count == 0) {
        return output;
    }
    tyloi::cuda::embedding_raw::nint8(quantized.data_ptr<uint8_t>(), scale.data_ptr(), token_ids.data_ptr<int64_t>(),
        output.data_ptr(), token_count, vocabulary, groups, int(width), mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

#endif

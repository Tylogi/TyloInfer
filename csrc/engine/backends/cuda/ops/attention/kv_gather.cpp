#include "engine/backends/cuda/ops/attention/kv_storage.h"

#include "engine/backends/cuda/kernels/moe/pipeline.h"
#include "engine/backends/cuda/native/tensor/backend.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tyloi::cuda::ops {
namespace {

std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("CUDA KV gather size overflow");
    return a * b;
}

std::size_t add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b)
        throw std::overflow_error("CUDA KV gather size overflow");
    return a + b;
}

const DenseTensor& dense(const Tensor& value) {
    const auto* tensor = value.representation<DenseTensor>();
    if (!tensor || !value.defined() || value.is_encoded() || !value.is_contiguous())
        throw std::invalid_argument("CUDA KV gather requires contiguous dense tensors");
    return *tensor;
}

} // namespace

std::size_t gather_kv(std::span<const tyloi::ops::KvGatherRow> selected,
                      const tyloi::ops::KvGatherWorkspace& work) {
    const auto& output = dense(work.rows);
    const auto& upload = dense(work.upload);
    const auto& host = dense(work.host);
    const auto& descriptors = dense(work.descriptors);
    const auto& device_descriptors = dense(work.device_descriptors);
    if (!output.is_cuda() || output.device().index < 0 || output.dim() != 2 ||
        output.size(0) <= 0 || output.size(1) <= 0 || !output.element_size() ||
        selected.empty() || selected.size() > static_cast<std::size_t>(output.size(0)) ||
        !output.data_ptr() || !upload.data_ptr() || !host.data_ptr() ||
        !descriptors.data_ptr() || !device_descriptors.data_ptr() ||
        upload.device() != output.device() || upload.dim() != 2 ||
        upload.sizes() != output.sizes() || upload.scalar_type() != output.scalar_type() ||
        !host.is_cpu() || host.device().index != 0 || !host.is_pinned() || host.dim() != 2 ||
        host.sizes() != output.sizes() || host.scalar_type() != output.scalar_type() ||
        !descriptors.is_cpu() || descriptors.device().index != 0 || !descriptors.is_pinned() ||
        descriptors.dim() != 1 || descriptors.scalar_type() != kInt64 ||
        device_descriptors.device() != output.device() || device_descriptors.dim() != 1 ||
        device_descriptors.scalar_type() != kInt64 || work.heads <= 0 || work.key_width < 0 ||
        output.size(1) % work.heads != 0 ||
        (work.key_width && work.key_width >= output.size(1) / work.heads))
        throw std::invalid_argument("invalid fixed CUDA KV gather workspace");

    const auto capacity = static_cast<std::size_t>(output.size(0));
    const auto width = static_cast<std::size_t>(output.size(1));
    const auto element = output.element_size();
    const auto row_bytes = multiply(width, element);
    const auto heads = static_cast<std::size_t>(work.heads);
    const auto parts = work.key_width ? multiply(heads, 2) : std::size_t(1);
    const auto copy_count = multiply(selected.size(), parts);
    const auto descriptor_bytes = add(4 * sizeof(std::int64_t), multiply(copy_count, sizeof(MoeMappedCopyDescriptor)));
    if (descriptor_bytes > descriptors.nbytes() || descriptor_bytes > device_descriptors.nbytes() ||
        copy_count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::invalid_argument("CUDA KV gather descriptor workspace is too small");

    std::size_t cold_bytes = 0;
    for (const auto& selected_row : selected) {
        const auto& source = dense(selected_row.source);
        if (!source.data_ptr() || (!source.is_cpu() && source.device() != output.device()) ||
            (source.is_cpu() && source.device().index != 0) || source.dim() != 2 ||
            source.size(0) <= 0 || source.size(1) != output.size(1) ||
            source.scalar_type() != output.scalar_type() || selected_row.row < 0 ||
            selected_row.row >= source.size(0) || selected_row.output < 0 ||
            static_cast<std::uint64_t>(selected_row.output) >= capacity)
            throw std::invalid_argument("invalid CUDA KV gather row");
        if (source.is_cpu()) cold_bytes = add(cold_bytes, row_bytes);
    }
    if (cold_bytes > host.nbytes() || cold_bytes > upload.nbytes())
        throw std::invalid_argument("CUDA KV gather staging workspace is too small");

    auto* host_data = static_cast<std::uint8_t*>(host.data_ptr());
    auto* upload_data = static_cast<std::uint8_t*>(upload.data_ptr());
    auto* destination = static_cast<std::uint8_t*>(output.data_ptr());
    auto* header = descriptors.data_ptr<std::int64_t>();
    auto* copies = reinterpret_cast<MoeMappedCopyDescriptor*>(header + 4);
    std::size_t cold_offset = 0, count = 0, total = 0;
    const auto append = [&](std::uint8_t* target, const std::uint8_t* source, std::size_t bytes) {
        const auto padded = add(bytes, 15) & ~std::size_t(15);
        copies[count++] = {reinterpret_cast<std::uint64_t>(target), reinterpret_cast<std::uint64_t>(source), total, bytes};
        total = add(total, padded);
    };
    for (const auto& selected_row : selected) {
        const auto& source = selected_row.source.dense();
        const auto* input = static_cast<const std::uint8_t*>(source.data_ptr()) +
            multiply(static_cast<std::size_t>(selected_row.row), row_bytes);
        if (source.is_cpu()) {
            std::memcpy(host_data + cold_offset, input, row_bytes);
            input = upload_data + cold_offset;
            cold_offset += row_bytes;
        }
        const auto row = static_cast<std::size_t>(selected_row.output);
        if (!work.key_width) {
            append(destination + row * row_bytes, input, row_bytes);
        } else {
            const auto keys = multiply(static_cast<std::size_t>(work.key_width), element);
            const auto values = row_bytes / heads - keys;
            for (std::size_t head = 0; head < heads; ++head) {
                append(destination + (head * capacity + row) * keys, input + head * keys, keys);
                append(destination + capacity * heads * keys + (head * capacity + row) * values,
                    input + heads * keys + head * values, values);
            }
        }
    }
    if (total > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("CUDA KV gather copy range overflow");
    header[0] = static_cast<std::int64_t>(count);
    header[1] = static_cast<std::int64_t>(total);
    header[2] = header[3] = 0;

    MfqCudaGuard guard(output.device().index);
    const auto stream = current_stream().stream();
    if (cold_bytes)
        TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(upload_data, host_data, cold_bytes, cudaMemcpyHostToDevice, stream));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(device_descriptors.data_ptr(), header, descriptor_bytes,
                                               cudaMemcpyHostToDevice, stream));
    const auto blocks = std::min<std::size_t>(256, total / 4096 + (total % 4096 != 0));
    moe_copy_mapped(device_descriptors.data_ptr(),
        reinterpret_cast<const std::uint32_t*>(device_descriptors.data_ptr<std::int64_t>() + 2),
        static_cast<int>(blocks), 256, stream);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetLastError());
    return cold_bytes;
}

} // namespace tyloi::cuda::ops

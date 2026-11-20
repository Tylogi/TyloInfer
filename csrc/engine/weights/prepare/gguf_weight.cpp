#include "gguf_weight.h"

#include "engine/backends/cuda/native/allocation.h"
#include "engine/backends/cuda/native/device.h"
#include "engine/backends/cuda/native/resident_allocations.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/kernels/projection/gguf/quad/storage.h"
#include "engine/weights/cache/ggml.h"
#include "engine/weights/cache/resources.h"
#include "engine/weights/encoding/gguf_quant.h"
#include "engine/weights/source/row_residual.h"
#include "engine/weights/source/transform.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tyloi::weights {
namespace {

DenseTensor read_bytes(const source::Source& source, const std::string& name,
        std::uint64_t offset, std::size_t bytes) {
    auto result = cuda::empty(
        {static_cast<std::int64_t>(std::max<std::size_t>(bytes, 1))},
        TensorOptions{}.device(kCPU).dtype(kUInt8));
    if (bytes) source.read_range_into(name, offset, result.data_ptr<std::byte>(), bytes);
    return result;
}

std::optional<GgmlInputRotation> rotation_for(const source::Source& source,
        const std::string& name, std::int64_t width, std::int64_t first,
        std::int64_t last, Device target) {
    const auto transform = source::hadamard(source, name);
    if (!transform) return std::nullopt;
    const auto block = transform->block_size;
    const auto& signs = transform->signs;
    if (first < 0 || last <= first || last > width || first % block ||
            (last - first) % block ||
            (!signs.empty() && static_cast<std::int64_t>(signs.size()) != width) ||
            std::any_of(signs.begin(), signs.end(), [](int sign) {
                return sign != 1 && sign != -1;
            }))
        throw std::invalid_argument("GGUF Hadamard basis is incompatible with the projection slice");
    GgmlInputRotation result;
    result.block = static_cast<int>(block);
    if (!signs.empty()) result.signs_cpu.assign(signs.begin() + first, signs.begin() + last);
    if (std::all_of(result.signs_cpu.begin(), result.signs_cpu.end(), [](int sign) {
            return sign == 1;
        })) result.signs_cpu.clear();
    if (result.block == 1 && result.signs_cpu.empty()) return std::nullopt;
    if (target.is_cuda()) {
        cuda::DeviceGuard guard(target.index);
        if (result.block > 128) {
            int regular = 0, optin = 0;
            TYLOINFER_NATIVE_CUDA_CHECK(cudaDeviceGetAttribute(
                &regular, cudaDevAttrMaxSharedMemoryPerBlock, target.index));
            TYLOINFER_NATIVE_CUDA_CHECK(cudaDeviceGetAttribute(
                &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, target.index));
            if (std::uint64_t(result.block) * sizeof(float) >
                    static_cast<std::uint64_t>(std::max(regular, optin)))
                throw std::invalid_argument(
                    "GGUF Hadamard block exceeds CUDA shared-memory capacity");
        }
    }
    if (!result.signs_cpu.empty()) {
        const cuda::AllocationScope allocation;
        auto host = cuda::empty({static_cast<std::int64_t>(result.signs_cpu.size())},
            TensorOptions{}.device(kCPU).dtype(kInt8));
        std::memcpy(host.data_ptr(), result.signs_cpu.data(), result.signs_cpu.size());
        result.signs_device = target.is_cuda() ? host.to(target) : std::move(host);
    }
    return result;
}

std::optional<GgmlRowCorrection> correction_for(const source::Source& source,
        const std::string& name, std::int64_t row_begin, std::int64_t row_end,
        std::int64_t column_begin, std::int64_t column_end, Device target) {
    const auto file = source.row_residual(name);
    if (!file || !file->payload_bytes()) return std::nullopt;
    if (file->row_count() > INT64_MAX || file->columns() > INT64_MAX ||
            file->payload_bytes() > INT64_MAX)
        throw std::invalid_argument("GGUF row correction extent is too large");
    if (row_begin < 0 || row_end > static_cast<std::int64_t>(file->row_count()) ||
            column_begin < 0 || column_end > static_cast<std::int64_t>(file->columns()))
        throw std::out_of_range("GGUF row correction slice is out of bounds");
    GgmlRowCorrection result;
    result.source_columns = static_cast<std::int64_t>(file->columns());
    result.first = row_begin;
    result.rows = row_end - row_begin;
    result.column_first = column_begin;
    result.columns = column_end - column_begin;
    const auto count = static_cast<std::int64_t>(file->row_count());
    result.offsets = cuda::empty({count},
        TensorOptions{}.device(kCPU).dtype(kInt64));
    result.types = cuda::empty({count},
        TensorOptions{}.device(kCPU).dtype(kInt32));
    for (std::int64_t row = 0; row < count; ++row) {
        result.offsets.data_ptr<std::int64_t>()[row] =
            static_cast<std::int64_t>(file->rows()[row].offset);
        result.types.data_ptr<std::int32_t>()[row] =
            static_cast<std::int32_t>(file->rows()[row].type);
    }
    result.packed = cuda::empty(
        {static_cast<std::int64_t>(file->payload_bytes())},
        TensorOptions{}.device(kCPU).dtype(kUInt8));
    file->read_payload(0, result.packed.data_ptr(),
        static_cast<std::size_t>(file->payload_bytes()));
    if (target.is_cuda()) {
        cuda::DeviceGuard guard(target.index);
        const auto device = Device(kCUDA, target.index);
        result.packed = result.packed.to(device);
        result.offsets = result.offsets.to(device);
        result.types = result.types.to(device);
    }
    return result;
}

} // namespace
} // namespace tyloi::weights

namespace tyloi {

std::shared_ptr<TensorImpl> GgmlQuantizedTensor::relocated(Device target) const {
    if (!target.is_cpu() && !target.is_cuda())
        throw std::invalid_argument("GGUF weights require CPU or CUDA");
    if (target.is_cpu() && target.index != 0)
        throw std::invalid_argument("CPU tensor device index must be zero");
    auto moved = std::static_pointer_cast<GgmlQuantizedTensor>(cloned());
    if (target.is_cpu() && moved->quad_storage_rows != 0) {
        if (!moved->packed.is_cuda() || !moved->packed.is_contiguous() ||
                moved->packed.scalar_type() != kUInt8 || moved->out <= 0 ||
                moved->row_bytes == 0 ||
                moved->row_bytes > std::numeric_limits<std::uint64_t>::max() /
                    static_cast<std::uint64_t>(moved->out))
            throw std::invalid_argument("invalid GGUF quad storage relocation");
        const auto bytes = moved->row_bytes * static_cast<std::uint64_t>(moved->out);
        if (bytes > static_cast<std::uint64_t>(INT64_MAX))
            throw std::invalid_argument("GGUF canonical storage is too large");
        cuda::DeviceGuard guard(moved->packed.get_device());
        const auto stream = mfq_get_current_cuda_stream(moved->packed.get_device());
        auto canonical = cuda::empty(
            {static_cast<std::int64_t>(bytes)}, moved->packed.options());
        mfq_cuda_record_stream(moved->packed, stream);
        mfq_cuda_record_stream(canonical, stream);
        cuda::gguf::unpack_quad_storage(
            moved->type, moved->packed.data_ptr(), moved->quad_storage_rows,
            moved->neuron_len, moved->quad_row_begin, moved->out, nullptr,
            moved->out, canonical.data_ptr(), moved->row_bytes, stream.stream());
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        moved->packed = std::move(canonical);
        moved->quad_storage_rows = 0;
        moved->quad_row_begin = 0;
    }
    for (const auto* buffer : moved->buffers()) {
        auto& writable = const_cast<DenseTensor&>(*buffer);
        if (writable.defined() && writable.device() != target)
            writable = writable.to(target);
    }
    return moved;
}

} // namespace tyloi

namespace tyloi::weights {

GgmlQuantizedTensor load_gguf(const source::Source& source, const std::string& name,
        Device target, std::int64_t row_begin, std::int64_t row_end,
        std::int64_t column_begin, std::int64_t column_end) {
    if (!target.is_cpu() && !target.is_cuda())
        throw std::invalid_argument("GGUF weights require CPU or CUDA");
    const auto* record = source.find_tensor(name);
    if (!record || !record->shape || record->shape->size() != 2)
        throw std::invalid_argument("GGUF projection requires a matrix: " + name);
    const auto type = gguf::quant_type(record->dtype);
    const auto rows = record->shape->at(0);
    const auto columns = record->shape->at(1);
    const auto source_stride = gguf::row_bytes(type, columns);
    if (rows <= 0 || columns <= 0 || (target.is_cpu() && target.index != 0) ||
            source_stride > UINT64_MAX / static_cast<std::uint64_t>(rows) ||
            source_stride > static_cast<std::uint64_t>(INT64_MAX) /
                static_cast<std::uint64_t>(rows) ||
            record->nbytes != source_stride * static_cast<std::uint64_t>(rows))
        throw std::invalid_argument("invalid GGUF matrix payload: " + name);
    if (row_end == -1) row_end = rows;
    if (column_end == -1) column_end = columns;
    const auto block = ggml_blck_size(type);
    if (row_begin < 0 || row_begin >= row_end || row_end > rows ||
            column_begin < 0 || column_begin >= column_end || column_end > columns ||
            column_begin % block || column_end % block)
        throw std::invalid_argument(
            "GGUF slices must contain complete quantization blocks: " + name);
    const auto out = row_end - row_begin;
    const auto width = column_end - column_begin;
    const auto bytes = gguf::row_bytes(type, width);
    cuda::AllocationScope allocation;
    DenseTensor packed;
    if (column_begin == 0 && column_end == columns) {
        packed = read_bytes(source, name,
            static_cast<std::uint64_t>(row_begin) * source_stride,
            static_cast<std::size_t>(source_stride * static_cast<std::uint64_t>(out)));
    } else {
        packed = cuda::empty(
            {static_cast<std::int64_t>(bytes * static_cast<std::uint64_t>(out))},
            TensorOptions{}.device(kCPU).dtype(kUInt8));
        const auto reader = source.tensor_reader(name);
        auto* destination = packed.data_ptr<std::byte>();
        for (std::int64_t row = 0; row < out; ++row) {
            reader(static_cast<std::uint64_t>(row_begin + row) * source_stride +
                    static_cast<std::uint64_t>(column_begin / block) * ggml_type_size(type),
                destination + row * bytes, static_cast<std::size_t>(bytes));
        }
    }
    if (target.is_cuda()) {
        cuda::DeviceGuard guard(target.index);
        packed = packed.to(target);
    }
    GgmlQuantizedTensor result;
    result.packed = std::move(packed);
    result.type = type;
    result.out = out;
    result.neuron_len = width;
    result.row_bytes = bytes;
    result.rotation = rotation_for(source, name, columns, column_begin, column_end, target);
    result.correction = correction_for(source, name, row_begin, row_end,
        column_begin, column_end, target);
    if (target.is_cuda() && !result.rotation && !result.correction && out >= 4 &&
            cuda::gguf::supports_quad_storage(type)) {
        cuda::DeviceGuard guard(target.index);
        const auto stream = mfq_get_current_cuda_stream(target.index);
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &capture));
        if (capture != cudaStreamCaptureStatusNone)
            throw std::invalid_argument(
                "GGUF storage preparation must precede graph capture");
        const auto bytes_total = cuda::gguf::quad_storage_bytes(type, out, width);
        if (result.packed.numel() != static_cast<std::int64_t>(bytes_total))
            throw std::invalid_argument("GGUF quad storage byte count differs");
        auto source_image = cuda::empty_like(result.packed);
        mfq_cuda_record_stream(source_image, stream);
        mfq_cuda_record_stream(result.packed, stream);
        TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(
            source_image.data_ptr(), result.packed.data_ptr(),
            result.packed.nbytes(), cudaMemcpyDeviceToDevice, stream.stream()));
        cuda::gguf::pack_quad_storage(type, source_image.data_ptr(), bytes,
            out, width, result.packed.data_ptr(), stream.stream());
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        result.quad_storage_rows = out;
        result.quad_row_begin = 0;
    }
    return result;
}

Tensor Cache::routed_gguf(const source::Source& source, const std::string& name, Device target) {
    if (!target.is_cuda()) throw std::invalid_argument("GGUF expert banks require CUDA execution");
    const auto* record = source.find_tensor(name);
    if (!record || !record->shape || record->shape->size() != 3)
        throw std::invalid_argument("GGUF expert bank must be [experts,out,in]: " + name);
    const auto& shape = *record->shape;
    const auto experts = shape[0], out = shape[1], input = shape[2];
    const auto type = gguf::quant_type(record->dtype);
    const auto stride = gguf::row_bytes(type, input);
    if (experts <= 0 || experts > INT_MAX || out <= 0 || out > INT_MAX || input > INT_MAX ||
        stride > std::uint64_t(INT64_MAX) / out / experts ||
        record->nbytes != stride * static_cast<std::uint64_t>(out) * experts)
        throw std::invalid_argument("invalid GGUF expert bank payload: " + name);

    auto bank = std::make_shared<GgmlExpertBank>();
    bank->target = target;
    bank->experts = experts;
    bank->expert_bytes = static_cast<std::int64_t>(stride * out);
    bank->image.type = type;
    bank->image.row_bytes = stride;
    bank->image.out = out;
    bank->image.neuron_len = input;
    const auto device_guard = cuda::DeviceGuard(target.index);
    const cuda::AllocationScope weight_memory;
    const cuda::ResidentAllocationScope permanent;
    bank->image.rotation = rotation_for(source, name, input, 0, input, target);
    bank->image.correction = correction_for(source, name, 0, experts * out, 0, input, target);
    auto reader = source.tensor_reader(name);
    if (!reader) throw std::invalid_argument("GGUF expert bank has no retained reader: " + name);
    const bool resident = name.starts_with("predictor.");
    if (resident) {
        const auto size = static_cast<std::int64_t>(record->nbytes);
        auto host = cuda::empty({std::min<std::int64_t>(size, 4 << 20)}, TensorOptions{}.device(kCPU).dtype(kUInt8));
        bank->image.packed = cuda::empty({size}, TensorOptions{}.device(target).dtype(kUInt8));
        const auto stream = mfq_get_current_cuda_stream(target.index);
        for (std::int64_t offset = 0; offset < size;) {
            const auto bytes = std::min<std::int64_t>(size - offset, host.numel());
            reader(offset, host.data_ptr<std::byte>(), bytes);
            TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(bank->image.packed.data_ptr<std::byte>() + offset,
                host.data_ptr(), bytes, cudaMemcpyHostToDevice, stream.stream()));
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
            offset += bytes;
        }
        if (!bank->image.rotation && !bank->image.correction &&
                !(type >= GGML_TYPE_TQ2_T && type <= GGML_TYPE_TQK4)) {
            auto addresses = cuda::empty({experts}, TensorOptions{}.device(kCPU).dtype(kInt64));
            const auto base = reinterpret_cast<std::uintptr_t>(bank->image.packed.data_ptr());
            for (std::int64_t expert = 0; expert < experts; ++expert)
                addresses.data_ptr<std::uint64_t>()[expert] = base + expert * bank->expert_bytes;
            bank->resident_addresses = addresses.to(target);
        }
    }
    if (!data_->ggml) {
        auto cache = std::make_shared<GgmlExpertCache>(data_->device_bytes.value_or(0),
            data_->host_bytes, data_->host_limit);
        register_host_cache(cache);
        data_->ggml = std::move(cache);
    }
    bank->cache = data_->ggml;
    bank->source_id = bank->cache->register_source(bank,
        resident ? source::Source::TensorReader{} : std::move(reader),
        resident ? source::Source::TensorBatchReader{} : source.tensor_batch_reader(name));
    return bank;
}

} // namespace tyloi::weights

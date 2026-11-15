#include "tensor.h"

#include "engine/backends/metal/native/tensor/tensor.h"
#include "engine/backends/metal/native/kernel.h"

#include <mlx/mlx.h>
#include <mlx/backend/metal/device.h>
#include <mlx/primitives.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tyloi::metal::ops {
namespace {

const mlx::core::array& input(const Tensor& value) {
    if (!value.defined() || value.device() != Device{DeviceType::metal, 0} || value.is_encoded()) {
        throw std::invalid_argument("expected a defined, unencoded Metal 0 tensor");
    }
    return metal::array(value);
}

int dimension(std::int64_t dim, std::size_t rank) {
    if (rank > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("Metal tensor rank exceeds int32 limit");
    const auto count = static_cast<std::int64_t>(rank);
    if (dim < 0) dim += count;
    if (dim < 0 || dim >= count) throw std::out_of_range("tensor dimension is out of range");
    return static_cast<int>(dim);
}

mlx::core::Stream gpu_stream() {
    return mlx::core::default_stream(mlx::core::Device::gpu);
}

mlx::core::array arithmetic_scalar(const mlx::core::array& value, double number);
void write_result(Tensor& value, mlx::core::array result);

mlx::core::Shape checked_shape(std::span<const std::int64_t> shape, const TensorOptions& options) {
    if (!options.has_device() || options.target_device() != Device{DeviceType::metal, 0} || options.pinned())
        throw std::invalid_argument("Metal allocation requires unpinned Metal 0 options");
    if (shape.size() > kMaximumTensorRank) throw std::invalid_argument("tensor rank exceeds maximum");
    mlx::core::Shape result;
    result.reserve(shape.size());
    for (const auto extent : shape) {
        if (extent < 0 || extent > std::numeric_limits<int>::max())
            throw std::invalid_argument("invalid Metal tensor extent");
        result.push_back(static_cast<int>(extent));
    }
    (void)metal::dtype(options.scalar_type());
    (void)make_contiguous_view(nullptr, shape, options.scalar_type(), options.target_device()).nbytes();
    return result;
}

template<class Data> auto allocation_buffer(Data& data) {
    if constexpr (requires { data.buffer(); }) return data.buffer();
    else return data.buffer;
}

class BufferWrite final : public mlx::core::UnaryPrimitive, public metal::PreparableKernel {
public:
    BufferWrite(mlx::core::Stream stream, bool zero) : UnaryPrimitive(stream), zero_(zero) {}
    std::string preparation_key() const override { return zero_ ? "tylo_zero_bytes" : "tylo_copy_bytes"; }
    void prepare_gpu() override { (void)kernel(); }
    void eval_cpu(const std::vector<mlx::core::array>&, mlx::core::array&) override {
        throw std::runtime_error("Metal buffer write requires the GPU");
    }
    void eval_gpu(const std::vector<mlx::core::array>& inputs, mlx::core::array& output) override {
        const auto& destination = inputs.at(0);
        const auto checked_buffer = [](const mlx::core::array& value) {
            const auto data = value.data_shared_ptr();
            if (!data || !allocation_buffer(*data).ptr())
                throw std::invalid_argument("Metal tensor has no writable backing");
            auto* buffer = static_cast<MTL::Buffer*>(allocation_buffer(*data).ptr());
            const auto available = std::min<std::size_t>(value.buffer_size(), buffer->allocatedSize());
            const auto offset = static_cast<std::size_t>(value.offset());
            if (offset > available || value.nbytes() > available - offset)
                throw std::invalid_argument("Metal tensor exceeds its backing buffer");
            return buffer;
        };
        auto* dest_buffer = checked_buffer(destination);
        if (!zero_) {
            const auto& source = inputs.at(1);
            auto* source_buffer = checked_buffer(source);
            if (dest_buffer == source_buffer) {
                const auto dest_offset = static_cast<std::size_t>(destination.offset());
                const auto source_offset = static_cast<std::size_t>(source.offset());
                if (dest_offset == source_offset) { output.copy_shared_buffer(destination); return; }
                if ((dest_offset < source_offset ? source_offset - dest_offset : dest_offset - source_offset) < destination.nbytes())
                    throw std::invalid_argument("overlapping Metal tensor copy is unsupported");
            }
        }
        output.copy_shared_buffer(destination);
        const auto bytes = destination.nbytes();
        if (!bytes) return;
        auto& selected_stream = stream();
        auto* selected_kernel = kernel();
        auto& encoder = mlx::core::metal::get_command_encoder(selected_stream);
        encoder.set_compute_pipeline_state(selected_kernel);
        encoder.set_output_array(output, 0);
        if (!zero_) encoder.set_input_array(inputs.at(1), 1);
        const auto count = static_cast<std::uint64_t>(bytes);
        const auto threads = static_cast<std::uint32_t>(std::min<std::size_t>(bytes, 65536));
        encoder.set_bytes(count, 2);
        encoder.set_bytes(threads, 3);
        encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(std::min<std::uint32_t>(threads, 256), 1, 1));
    }
    const char* name() const override { return "MetalBufferWrite"; }
    bool is_equivalent(const mlx::core::Primitive&) const override { return false; }
    std::vector<mlx::core::Shape> output_shapes(const std::vector<mlx::core::array>& inputs) override {
        return {inputs.at(0).shape()};
    }
private:
    MTL::ComputePipelineState* kernel() {
        auto& device = mlx::core::metal::device(stream().device);
        auto* library = device.get_library("tylo_tensor_buffer_write", [] {
            return std::string(R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void tylo_copy_bytes(device uchar* destination [[buffer(0)]],
    device const uchar* source [[buffer(1)]], constant ulong& count [[buffer(2)]],
    constant uint& stride [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    for (ulong i = index; i < count; i += stride) destination[i] = source[i];
}
kernel void tylo_zero_bytes(device uchar* destination [[buffer(0)]],
    constant ulong& count [[buffer(2)]], constant uint& stride [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    for (ulong i = index; i < count; i += stride) destination[i] = 0;
}
)METAL");
        });
        return device.get_kernel(zero_ ? "tylo_zero_bytes" : "tylo_copy_bytes", library);
    }
    bool zero_;
};

void write(Tensor& destination, const Tensor* source) {
    const auto& target = input(destination);
    if (!destination.is_contiguous()) throw std::invalid_argument("Metal buffer write requires contiguous destination");
    std::vector<mlx::core::array> inputs{target};
    if (source) {
        const auto& from = input(*source);
        if (!source->is_contiguous() || from.shape() != target.shape() || from.dtype() != target.dtype())
            throw std::invalid_argument("Metal tensor copy requires contiguous matching tensors");
        if (from.id() == target.id()) return;
        inputs.push_back(from);
    }
    if (!target.nbytes()) return;
    auto output = mlx::core::array(target.shape(), target.dtype(),
        std::make_shared<BufferWrite>(gpu_stream(), source == nullptr), std::move(inputs));
    metal::replace(destination, std::move(output));
}

}  // namespace

Tensor empty(std::span<const std::int64_t> shape, const TensorOptions& options) {
    (void)checked_shape(shape, options);
    return metal::allocate(shape, options.scalar_type());
}

template<class T>
Tensor tensor_values(std::span<const T> values, Device target, ScalarType type) {
    if (target != Device{DeviceType::metal, 0}) throw std::invalid_argument("expected Metal 0 target");
    if (values.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("Metal tensor extent exceeds int32 limit");
    const std::int64_t shape[]{static_cast<std::int64_t>(values.size())};
    return metal::copy_from_host(values.data(), shape, type);
}
Tensor tensor(std::span<const float> values, Device target) { return tensor_values(values, target, kFloat32); }
Tensor tensor(std::span<const std::int32_t> values, Device target) { return tensor_values(values, target, kInt32); }
Tensor tensor(std::span<const std::int64_t> values, Device target) { return tensor_values(values, target, kInt64); }
Tensor tensor(std::span<const std::uint8_t> values, Device target) { return tensor_values(values, target, kUInt8); }

Tensor arange(std::int64_t first, std::int64_t last, Device target) {
    if (target != Device{DeviceType::metal, 0}) throw std::invalid_argument("expected Metal 0 target");
    const auto count = last > first ? static_cast<__int128>(last) - first : 0;
    if (count > std::numeric_limits<int>::max()) throw std::overflow_error("Metal arange extent exceeds int32 limit");
    const auto stream = gpu_stream();
    auto offsets = mlx::core::arange(static_cast<int>(count), mlx::core::int64, stream);
    return metal::tensor(mlx::core::add(offsets, mlx::core::array(first, mlx::core::int64), stream));
}

Tensor full(std::span<const std::int64_t> shape, std::int64_t value, Device target) {
    auto sizes = checked_shape(shape, TensorOptions().dtype(kInt64).device(target));
    return metal::tensor(mlx::core::full(std::move(sizes), mlx::core::array(value, mlx::core::int64), mlx::core::int64, gpu_stream()));
}

Tensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options) {
    (void)checked_shape(shape, options);
    auto result = metal::allocate(shape, options.scalar_type());
    fill(result, 0);
    return result;
}

void copy_into(Tensor destination, const Tensor& source) {
    if (source.device() == Device{DeviceType::cpu, 0}) {
        auto staging = metal::copy(source, Device{DeviceType::metal, 0});
        write(destination, &staging);
    } else {
        write(destination, &source);
    }
}

void fill(Tensor& value, double number) {
    if (number == 0) {
        write(value, nullptr);
        return;
    }
    const auto& target = input(value); const auto stream = gpu_stream();
    auto result = mlx::core::full(target.shape(), arithmetic_scalar(target, number), target.dtype(), stream);
    write_result(value, std::move(result));
}

Tensor copy(const Tensor& value, Device target) {
    return metal::copy(value, target);
}

Tensor clone(const Tensor& value) {
    const auto& source = input(value);
    if (source.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("Metal clone tensor element count exceeds int32 limit");
    }
    // MLX may elide copy() when the result is otherwise identical to its input.
    // Cache snapshots must own a distinct allocation because live cache updates
    // can write through aliases. Appending and then slicing off one sentinel forces
    // a new backing allocation without applying arithmetic to the tensor values.
    auto flat = mlx::core::reshape(source, {-1});
    auto storage = mlx::core::concatenate(
        {flat, mlx::core::zeros({1}, source.dtype())},
        0);
    return metal::tensor(mlx::core::reshape(
        mlx::core::slice(storage, {0}, {static_cast<int>(source.size())}),
        source.shape()));
}

Tensor cast(const Tensor& value, ScalarType type) {
    const auto& source = input(value);
    if (value.scalar_type() == type) return value;
    const auto target = metal::dtype(type);
    return metal::tensor(mlx::core::astype(source, target));
}

Tensor reshape(const Tensor& value, std::span<const std::int64_t> shape) {
    const auto& source = input(value);
    if (shape.size() > kMaximumTensorRank) throw std::invalid_argument("tensor rank exceeds maximum");
    mlx::core::Shape target;
    target.reserve(shape.size());
    bool inferred = false;
    for (const auto extent : shape) {
        if (extent < -1 || extent > std::numeric_limits<int>::max() || (extent == -1 && inferred))
            throw std::invalid_argument("invalid reshape extent");
        inferred |= extent == -1;
        target.push_back(static_cast<int>(extent));
    }
    return metal::tensor(mlx::core::reshape(source, target));
}

Tensor transpose(const Tensor& value, std::int64_t first, std::int64_t second) {
    const auto& source = input(value);
    const auto a = dimension(first, source.ndim());
    const auto b = dimension(second, source.ndim());
    return metal::tensor(mlx::core::swapaxes(source, a, b));
}

Tensor slice(const Tensor& value, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step) {
    const auto& source = input(value);
    if (step <= 0 || step > std::numeric_limits<int>::max())
        throw std::invalid_argument("Metal slice requires a positive int32 step");
    const auto selected = dimension(dim, source.ndim());
    const std::int64_t extent = source.shape(selected);
    auto first = std::clamp(start < 0 ? start + extent : start, std::int64_t{0}, extent);
    auto last = std::clamp(end < 0 ? end + extent : end, std::int64_t{0}, extent);
    last = std::max(first, last);
    mlx::core::Shape starts(source.ndim(), 0);
    mlx::core::Shape stops = source.shape();
    mlx::core::Shape strides(source.ndim(), 1);
    starts[selected] = static_cast<int>(first);
    stops[selected] = static_cast<int>(last);
    strides[selected] = static_cast<int>(step);
    return metal::tensor(mlx::core::slice(source, starts, stops, strides));
}

Tensor contiguous(const Tensor& value) {
    const auto& source = input(value);
    if (value.is_contiguous()) return value;
    return metal::tensor(mlx::core::contiguous(source));
}

Tensor update_slice(const Tensor& destination, const Tensor& source,
        std::int64_t dim, std::int64_t start) {
    const auto& target = input(destination);
    const auto& values = input(source);
    if (target.ndim() != values.ndim() || target.dtype() != values.dtype())
        throw std::invalid_argument("Metal slice update requires matching rank and dtype");
    const auto axis = dimension(dim, target.ndim());
    for (int index = 0; index < target.ndim(); ++index)
        if (index != axis && target.shape(index) != values.shape(index))
            throw std::invalid_argument("Metal slice update shapes do not match");
    if (start < 0 || start > target.shape(axis) || values.shape(axis) > target.shape(axis) - start)
        throw std::out_of_range("Metal slice update exceeds its destination");
    if (!values.size()) return destination;
    mlx::core::Shape begin(target.ndim(), 0);
    auto end = target.shape();
    begin[axis] = static_cast<int>(start);
    end[axis] = static_cast<int>(start + values.shape(axis));
    return metal::tensor(mlx::core::slice_update(target, values, begin, end, gpu_stream()));
}


Tensor cat(std::span<const Tensor> values, std::int64_t dim) {
    if (values.empty()) throw std::invalid_argument("Metal cat requires at least one tensor");
    const auto& first = input(values.front());
    const auto axis = dimension(dim, first.ndim());
    std::vector<mlx::core::array> arrays;
    arrays.reserve(values.size());
    std::int64_t extent = 0;
    for (const auto& value : values) {
        const auto& item = input(value);
        if (item.ndim() != first.ndim() || item.dtype() != first.dtype())
            throw std::invalid_argument("Metal cat requires matching ranks and dtypes");
        for (int index = 0; index < first.ndim(); ++index)
            if (index != axis && item.shape(index) != first.shape(index))
                throw std::invalid_argument("Metal cat shapes do not match");
        if (item.shape(axis) > std::numeric_limits<int>::max() - extent)
            throw std::overflow_error("Metal cat extent exceeds int32 limit");
        extent += item.shape(axis);
        arrays.push_back(item);
    }
    return metal::tensor(mlx::core::concatenate(std::move(arrays), axis, gpu_stream()));
}

Tensor index_select(const Tensor& value, std::int64_t dim, const Tensor& indices) {
    const auto& source = input(value);
    const auto& selected = input(indices);
    const auto axis = dimension(dim, source.ndim());
    if (selected.ndim() != 1 || (selected.dtype() != mlx::core::int32 && selected.dtype() != mlx::core::int64))
        throw std::invalid_argument("Metal indices must be a one-dimensional integer tensor");
    return metal::tensor(mlx::core::take(source, selected, axis, gpu_stream()));
}

void index_copy(Tensor& value, std::int64_t dim, const Tensor& indices, const Tensor& source) {
    const auto& target = input(value);
    const auto& selected = input(indices);
    const auto& updates = input(source);
    if (!value.is_contiguous()) throw std::invalid_argument("Metal in-place update requires contiguous destination");
    const auto axis = dimension(dim, target.ndim());
    if (selected.ndim() != 1 || (selected.dtype() != mlx::core::int32 && selected.dtype() != mlx::core::int64))
        throw std::invalid_argument("Metal indices must be a one-dimensional integer tensor");
    if (target.ndim() != updates.ndim() || target.dtype() != updates.dtype() ||
        updates.shape(axis) != selected.size())
        throw std::invalid_argument("Metal index copy requires matching ranks and dtypes");
    for (int index = 0; index < target.ndim(); ++index)
        if (index != axis && target.shape(index) != updates.shape(index))
            throw std::invalid_argument("Metal index copy shapes do not match");
    if (!selected.size()) return;
    mlx::core::Shape index_shape(target.ndim(), 1);
    index_shape[axis] = updates.shape(axis);
    auto expanded = mlx::core::broadcast_to(
        mlx::core::reshape(selected, index_shape, gpu_stream()), updates.shape(), gpu_stream());
    auto result = mlx::core::put_along_axis(target, expanded, updates, axis, gpu_stream());
    write_result(value, std::move(result));
}


Tensor add(const Tensor& left, const Tensor& right) {
    return metal::tensor(mlx::core::add(input(left), input(right), gpu_stream()));
}
Tensor multiply(const Tensor& left, const Tensor& right) {
    return metal::tensor(mlx::core::multiply(input(left), input(right), gpu_stream()));
}
Tensor matmul(const Tensor& left, const Tensor& right) {
    return metal::tensor(mlx::core::matmul(input(left), input(right), gpu_stream()));
}
Tensor add(const Tensor& value, std::int64_t offset) {
    const auto& x = input(value);
    return metal::tensor(mlx::core::astype(mlx::core::add(x, mlx::core::array(offset, mlx::core::int64), gpu_stream()), x.dtype(), gpu_stream()));
}
// MLX scalar arithmetic uses FP32; results retain the activation dtype.
Tensor divide(const Tensor& value, double number) {
    const auto& x = input(value);
    return metal::tensor(mlx::core::astype(mlx::core::divide(x, mlx::core::array(static_cast<float>(number)), gpu_stream()), x.dtype(), gpu_stream()));
}
Tensor scale(const Tensor& value, double number) {
    const auto& x = input(value);
    return metal::tensor(mlx::core::astype(mlx::core::multiply(x, mlx::core::array(static_cast<float>(number)), gpu_stream()), x.dtype(), gpu_stream()));
}
Tensor softcap(const Tensor& value, double number) {
    const auto& x = input(value);
    if (!(number > 0)) return value;
    auto scaled = mlx::core::astype(mlx::core::divide(x, mlx::core::array(static_cast<float>(number)), gpu_stream()), x.dtype(), gpu_stream());
    auto capped = mlx::core::tanh(scaled, gpu_stream());
    return metal::tensor(mlx::core::astype(mlx::core::multiply(capped, mlx::core::array(static_cast<float>(number)), gpu_stream()), x.dtype(), gpu_stream()));
}
Tensor gelu(const Tensor& value, bool approximate) {
    const auto& x = input(value);
    const auto stream = gpu_stream();
    const auto xf = mlx::core::astype(x, mlx::core::float32, stream);
    mlx::core::array factor = approximate
        ? mlx::core::tanh(mlx::core::multiply(mlx::core::array(0.7978845608028654f),
              mlx::core::add(xf, mlx::core::multiply(mlx::core::array(0.044715f),
                  mlx::core::multiply(mlx::core::square(xf, stream), xf, stream), stream), stream), stream), stream)
        : mlx::core::erf(mlx::core::multiply(xf, mlx::core::array(0.7071067811865476f), stream), stream);
    return metal::tensor(mlx::core::astype(mlx::core::multiply(mlx::core::multiply(xf, mlx::core::array(0.5f), stream),
        mlx::core::add(mlx::core::array(1.0f), factor, stream), stream), x.dtype(), stream));
}
Tensor silu(const Tensor& value) {
    const auto& x = input(value); const auto stream = gpu_stream();
    return metal::tensor(mlx::core::astype(mlx::core::multiply(x, mlx::core::sigmoid(x, stream), stream), x.dtype(), stream));
}
Tensor relu_squared(const Tensor& value) {
    const auto& x = input(value); const auto stream = gpu_stream();
    return metal::tensor(mlx::core::astype(mlx::core::square(mlx::core::maximum(x, mlx::core::array(0), stream), stream), x.dtype(), stream));
}
Tensor abs(const Tensor& value) {
    const auto& x = input(value); const auto stream = gpu_stream();
    return metal::tensor(mlx::core::astype(mlx::core::abs(x, stream), x.dtype(), stream));
}
Tensor square(const Tensor& value) {
    const auto& x = input(value); const auto stream = gpu_stream();
    return metal::tensor(mlx::core::astype(mlx::core::square(x, stream), x.dtype(), stream));
}
Tensor exp(const Tensor& value) {
    const auto& x = input(value); const auto stream = gpu_stream();
    return metal::tensor(mlx::core::astype(mlx::core::exp(x, stream), x.dtype(), stream));
}
Tensor sqrt(const Tensor& value) {
    const auto& x = input(value); const auto stream = gpu_stream();
    return metal::tensor(mlx::core::astype(mlx::core::sqrt(x, stream), x.dtype(), stream));
}
Tensor logical_not(const Tensor& value) {
    return metal::tensor(mlx::core::logical_not(input(value), gpu_stream()));
}
Tensor clamp(const Tensor& value, double minimum, double maximum) {
    const auto& x = input(value); const auto stream = gpu_stream();
    return metal::tensor(mlx::core::astype(mlx::core::clip(x, mlx::core::array(static_cast<float>(minimum)), mlx::core::array(static_cast<float>(maximum)), stream), x.dtype(), stream));
}
Tensor sum(const Tensor& value, std::int64_t dim, bool keepdim) {
    const auto& x = input(value); const auto axis = dimension(dim, x.ndim()); const auto stream = gpu_stream();
    if (x.shape(axis) == 0) throw std::invalid_argument("cannot reduce an empty dimension");
    const bool floating = x.dtype() == mlx::core::float16 || x.dtype() == mlx::core::bfloat16 || x.dtype() == mlx::core::float32;
    return metal::tensor(mlx::core::astype(mlx::core::sum(floating ? x : mlx::core::astype(x, mlx::core::int64, stream), axis, keepdim, stream), floating ? x.dtype() : mlx::core::int64, stream));
}
Tensor mean(const Tensor& value, std::int64_t dim, bool keepdim) {
    const auto& x = input(value); const auto axis = dimension(dim, x.ndim()); const auto stream = gpu_stream();
    if (x.shape(axis) == 0) throw std::invalid_argument("cannot reduce an empty dimension");
    if (x.dtype() != mlx::core::float16 && x.dtype() != mlx::core::bfloat16 && x.dtype() != mlx::core::float32)
        throw std::invalid_argument("native mean requires a floating dtype");
    return metal::tensor(mlx::core::astype(mlx::core::mean(x, axis, keepdim, stream), x.dtype(), stream));
}
Tensor amax(const Tensor& value, std::int64_t dim, bool keepdim) {
    const auto& x = input(value); const auto axis = dimension(dim, x.ndim()); const auto stream = gpu_stream();
    if (x.shape(axis) == 0) throw std::invalid_argument("cannot reduce an empty dimension");
    return metal::tensor(mlx::core::max(x, axis, keepdim, stream));
}
Tensor argmax(const Tensor& value, std::int64_t dim, bool keepdim) {
    const auto& x = input(value); const auto axis = dimension(dim, x.ndim()); const auto stream = gpu_stream();
    if (x.shape(axis) == 0) throw std::invalid_argument("cannot reduce an empty dimension");
    return metal::tensor(mlx::core::astype(mlx::core::argmax(x, axis, keepdim, stream), mlx::core::int64, stream));
}
Tensor all(const Tensor& value, std::int64_t dim, bool keepdim) {
    const auto& x = input(value); const auto axis = dimension(dim, x.ndim()); const auto stream = gpu_stream();
    if (x.shape(axis) == 0) throw std::invalid_argument("cannot reduce an empty dimension");
    return metal::tensor(mlx::core::all(x, axis, keepdim, stream));
}
Tensor any(const Tensor& value, std::int64_t dim, bool keepdim) {
    const auto& x = input(value); const auto axis = dimension(dim, x.ndim()); const auto stream = gpu_stream();
    if (x.shape(axis) == 0) throw std::invalid_argument("cannot reduce an empty dimension");
    return metal::tensor(mlx::core::any(x, axis, keepdim, stream));
}
Tensor expand(const Tensor& value, std::span<const std::int64_t> shape) {
    const auto& x = input(value);
    if (shape.size() < static_cast<std::size_t>(x.ndim()) || shape.size() > kMaximumTensorRank)
        throw std::invalid_argument("expand rank is incompatible with tensor rank");
    mlx::core::Shape sizes; sizes.reserve(shape.size());
    const auto offset = shape.size() - x.ndim();
    for (std::size_t i = 0; i < shape.size(); ++i) {
        auto requested = shape[i];
        if (i < offset) {
            if (requested < 0) throw std::invalid_argument("new expand dimensions must be explicit");
        } else {
            const auto extent = x.shape(static_cast<int>(i - offset));
            if (requested == -1) requested = extent;
            if (requested < 0 || (requested != extent && extent != 1))
                throw std::invalid_argument("cannot expand a non-singleton dimension");
        }
        if (requested > std::numeric_limits<int>::max()) throw std::overflow_error("Metal tensor extent exceeds int32 limit");
        sizes.push_back(static_cast<int>(requested));
    }
    return metal::tensor(mlx::core::broadcast_to(x, sizes, gpu_stream()));
}
Tensor unsqueeze(const Tensor& value, std::int64_t dim) {
    const auto& x = input(value);
    if (static_cast<std::size_t>(x.ndim()) >= kMaximumTensorRank) throw std::invalid_argument("cannot unsqueeze this tensor");
    const auto axis = dimension(dim, x.ndim() + 1);
    return metal::tensor(mlx::core::expand_dims(x, axis, gpu_stream()));
}
Tensor select(const Tensor& value, std::int64_t dim, std::int64_t index) {
    const auto& x = input(value); const auto axis = dimension(dim, x.ndim());
    const auto extent = x.shape(axis);
    if (index < 0) index += extent;
    if (index < 0 || index >= extent) throw std::out_of_range("tensor index is out of range");
    mlx::core::Shape begin(x.ndim(), 0), end = x.shape();
    begin[axis] = static_cast<int>(index); end[axis] = static_cast<int>(index + 1);
    return metal::tensor(mlx::core::squeeze(mlx::core::slice(x, begin, end, gpu_stream()), axis, gpu_stream()));
}

namespace {
mlx::core::array scalar(double number) {
    return mlx::core::array(static_cast<float>(number));
}

mlx::core::array arithmetic_scalar(const mlx::core::array& value, double number) {
    const auto type = value.dtype();
    if (type == mlx::core::float16 || type == mlx::core::bfloat16 || type == mlx::core::float32)
        return scalar(number);
    if (!std::isfinite(number) || std::trunc(number) != number)
        throw std::invalid_argument("Metal integer scalar must be finite and integral");
    const auto n = static_cast<long double>(number);
    if (type == mlx::core::uint8 || type == mlx::core::uint16 || type == mlx::core::uint32 || type == mlx::core::uint64) {
        if (type == mlx::core::uint64 && (number < 0 || number >= 0x1p64))
            throw std::invalid_argument("Metal integer scalar is out of range");
        const long double maximum = type == mlx::core::uint8 ? UINT8_MAX : type == mlx::core::uint16 ? UINT16_MAX :
            type == mlx::core::uint32 ? UINT32_MAX : static_cast<long double>(UINT64_MAX);
        if (n < 0 || n > maximum) throw std::invalid_argument("Metal integer scalar is out of range");
        return mlx::core::array(static_cast<std::uint64_t>(number), mlx::core::uint64);
    }
    const long double minimum = type == mlx::core::bool_ ? 0 : type == mlx::core::int8 ? INT8_MIN :
        type == mlx::core::int16 ? INT16_MIN : type == mlx::core::int32 ? INT32_MIN : static_cast<long double>(INT64_MIN);
    const long double maximum = type == mlx::core::bool_ ? 1 : type == mlx::core::int8 ? INT8_MAX :
        type == mlx::core::int16 ? INT16_MAX : type == mlx::core::int32 ? INT32_MAX : static_cast<long double>(INT64_MAX);
    if (type == mlx::core::int64 && (number < -0x1p63 || number >= 0x1p63))
        throw std::invalid_argument("Metal integer scalar is out of range");
    if (n < minimum || n > maximum) throw std::invalid_argument("Metal integer scalar is out of range");
    return mlx::core::array(static_cast<std::int64_t>(number), mlx::core::int64);
}

void index_type(const mlx::core::array& indices) {
    if (indices.dtype() != mlx::core::int32 && indices.dtype() != mlx::core::int64)
        throw std::invalid_argument("Metal indices must be int32 or int64");
}

void write_result(Tensor& value, mlx::core::array result) {
    if (!value.is_contiguous()) throw std::invalid_argument("Metal in-place update requires contiguous destination");
    auto independent = clone(metal::tensor(std::move(result)));
    copy_into(value, independent);
}

void scatter_impl(Tensor& value, std::int64_t dim, const Tensor& indices, const Tensor& source, bool add) {
    const auto& target = input(value); const auto& selected = input(indices); const auto& updates = input(source);
    const auto axis = dimension(dim, target.ndim());
    index_type(selected);
    if (selected.ndim() != target.ndim() || selected.shape() != updates.shape() || target.dtype() != updates.dtype())
        throw std::invalid_argument("Metal scatter requires matching index and source shapes, rank and dtype");
    auto end = target.shape();
    for (int i = 0; i < target.ndim(); ++i) {
        if (i != axis && selected.shape(i) > target.shape(i))
            throw std::invalid_argument("Metal scatter index exceeds destination extent");
        if (i != axis) end[i] = selected.shape(i);
    }
    if (add && target.dtype() != mlx::core::int32 && target.dtype() != mlx::core::int64 &&
        target.dtype() != mlx::core::float16 && target.dtype() != mlx::core::bfloat16 && target.dtype() != mlx::core::float32)
        throw std::invalid_argument("unsupported Metal scatter_add dtype");
    const auto stream = gpu_stream();
    mlx::core::Shape begin(target.ndim(), 0);
    auto region = mlx::core::slice(target, begin, end, stream);
    auto updated = add ? mlx::core::scatter_add_axis(region, selected, updates, axis, stream)
                       : mlx::core::put_along_axis(region, selected, updates, axis, stream);
    auto result = mlx::core::slice_update(target, updated, begin, end, stream);
    write_result(value, std::move(result));
}
}  // namespace

Tensor clamp_min(const Tensor& value, double minimum) {
    const auto& x = input(value); const auto stream = gpu_stream();
    auto bound = arithmetic_scalar(x, minimum);
    if (std::isnan(minimum)) return value;
    return metal::tensor(mlx::core::astype(mlx::core::where(mlx::core::isnan(x, stream), bound,
        mlx::core::maximum(x, bound, stream), stream), x.dtype(), stream));
}
Tensor clamp_max(const Tensor& value, double maximum) {
    const auto& x = input(value); const auto stream = gpu_stream();
    auto bound = arithmetic_scalar(x, maximum);
    if (std::isnan(maximum)) return value;
    return metal::tensor(mlx::core::astype(mlx::core::where(mlx::core::isnan(x, stream), bound,
        mlx::core::minimum(x, bound, stream), stream), x.dtype(), stream));
}
Tensor remainder(const Tensor& value, double divisor) {
    const auto& x = input(value); const auto stream = gpu_stream();
    return metal::tensor(mlx::core::astype(mlx::core::remainder(x, arithmetic_scalar(x, divisor), stream), x.dtype(), stream));
}
Tensor eq(const Tensor& left, const Tensor& right) {
    const auto& x = input(left); const auto& y = input(right);
    if (x.dtype() != y.dtype()) throw std::invalid_argument("Metal comparison requires matching dtypes");
    return metal::tensor(mlx::core::equal(x, y, gpu_stream()));
}
Tensor ne(const Tensor& left, const Tensor& right) {
    const auto& x = input(left); const auto& y = input(right);
    if (x.dtype() != y.dtype()) throw std::invalid_argument("Metal comparison requires matching dtypes");
    return metal::tensor(mlx::core::not_equal(x, y, gpu_stream()));
}
Tensor eq(const Tensor& value, double other) {
    const auto& x = input(value);
    return metal::tensor(mlx::core::equal(x, arithmetic_scalar(x, other), gpu_stream()));
}
Tensor ne(const Tensor& value, double other) {
    const auto& x = input(value);
    return metal::tensor(mlx::core::not_equal(x, arithmetic_scalar(x, other), gpu_stream()));
}
bool equal(const Tensor& left, const Tensor& right) {
    const auto& x = input(left); const auto& y = input(right);
    if (x.shape() != y.shape() || x.dtype() != y.dtype()) return false;
    if (!x.size()) return true;
    auto result = mlx::core::array_equal(x, y, gpu_stream());
    metal::eval(result);
    return result.item<bool>();
}
Tensor masked_select(const Tensor& value, const Tensor& mask) {
    (void)input(value); (void)input(mask);
    throw std::invalid_argument("Metal masked_select with dynamic output size is unsupported");
}
Tensor gather(const Tensor& value, std::int64_t dim, const Tensor& indices) {
    const auto& x = input(value); const auto& selected = input(indices); const auto axis = dimension(dim, x.ndim());
    index_type(selected);
    if (selected.ndim() != x.ndim()) throw std::invalid_argument("Metal gather requires matching ranks");
    auto end = x.shape();
    for (int i = 0; i < x.ndim(); ++i) {
        if (i != axis && selected.shape(i) > x.shape(i))
            throw std::invalid_argument("Metal gather index exceeds source extent");
        if (i != axis) end[i] = selected.shape(i);
    }
    const auto stream = gpu_stream();
    auto region = mlx::core::slice(x, mlx::core::Shape(x.ndim(), 0), end, stream);
    return metal::tensor(mlx::core::take_along_axis(region, selected, axis, stream));
}
Tensor repeat(const Tensor& value, std::span<const std::int64_t> repeats) {
    const auto& x = input(value);
    if (repeats.size() < static_cast<std::size_t>(x.ndim()) || repeats.size() > kMaximumTensorRank)
        throw std::invalid_argument("invalid Metal repeat rank");
    std::vector<int> counts; counts.reserve(repeats.size());
    const auto offset = repeats.size() - x.ndim();
    for (std::size_t i = 0; i < repeats.size(); ++i) {
        const auto extent = i < offset ? 1 : x.shape(static_cast<int>(i - offset));
        if (repeats[i] < 0 || repeats[i] > std::numeric_limits<int>::max() ||
            (extent && repeats[i] > std::numeric_limits<int>::max() / extent))
            throw std::invalid_argument("invalid Metal repeat extent");
        counts.push_back(static_cast<int>(repeats[i]));
    }
    return metal::tensor(mlx::core::tile(x, std::move(counts), gpu_stream()));
}
Tensor repeat_interleave(const Tensor& value, std::int64_t repeats, std::int64_t dim) {
    const auto& x = input(value); const auto axis = dimension(dim, x.ndim());
    if (repeats < 0 || repeats > std::numeric_limits<int>::max() ||
        (x.shape(axis) && repeats > std::numeric_limits<int>::max() / x.shape(axis)))
        throw std::invalid_argument("invalid Metal repeat extent");
    return metal::tensor(mlx::core::repeat(x, static_cast<int>(repeats), axis, gpu_stream()));
}
Tensor tril(const Tensor& value, std::int64_t diagonal) {
    const auto& x = input(value);
    if (x.ndim() < 2 || diagonal < std::numeric_limits<int>::min() || diagonal > std::numeric_limits<int>::max())
        throw std::invalid_argument("invalid Metal tril geometry");
    return metal::tensor(mlx::core::tril(x, static_cast<int>(diagonal), gpu_stream()));
}
Tensor masked_fill(const Tensor& value, const Tensor& mask, double number) {
    const auto& x = input(value); const auto& flags = input(mask); const auto stream = gpu_stream();
    auto condition = mlx::core::broadcast_to(mlx::core::astype(flags, mlx::core::bool_, stream), x.shape(), stream);
    auto fill = mlx::core::astype(scalar(number), x.dtype(), stream);
    return metal::tensor(mlx::core::where(condition, fill, x, stream));
}
void masked_fill_inplace(Tensor& value, const Tensor& mask, double number) {
    write_result(value, input(masked_fill(value, mask, number)));
}
void scatter_add(Tensor& value, std::int64_t dim, const Tensor& indices, const Tensor& source) {
    scatter_impl(value, dim, indices, source, true);
}
void scatter(Tensor& value, std::int64_t dim, const Tensor& indices, const Tensor& source) {
    scatter_impl(value, dim, indices, source, false);
}
void index_fill(Tensor& value, std::int64_t dim, const Tensor& indices, double number) {
    const auto& x = input(value); const auto& selected = input(indices); const auto axis = dimension(dim, x.ndim());
    index_type(selected);
    if (selected.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("Metal index extent exceeds int32 limit");
    const auto stream = gpu_stream();
    mlx::core::Shape shape(x.ndim(), 1); shape[axis] = static_cast<int>(selected.size());
    auto output_shape = x.shape(); output_shape[axis] = shape[axis];
    auto expanded = mlx::core::broadcast_to(mlx::core::reshape(selected, shape, stream), output_shape, stream);
    auto fill = mlx::core::astype(scalar(number), x.dtype(), stream);
    auto updates = mlx::core::broadcast_to(fill, output_shape, stream);
    write_result(value, mlx::core::put_along_axis(x, expanded, updates, axis, stream));
}

Tensor log_negative(const Tensor& value) {
    return metal::tensor(mlx::core::log(-input(value), gpu_stream()));
}

}  // namespace tyloi::metal::ops

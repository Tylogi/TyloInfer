#include "engine/weights/loader.h"
#include "engine/weights/cache/cache.h"
#include "engine/weights/encoding/dense/blob.h"
#include "engine/weights/encoding/metal.h"
#include "engine/weights/encoding/rows.h"
#include "engine/weights/memory.h"
#include "engine/weights/rows/source.h"
#include "engine/weights/tq/source.h"
#include "engine/backends/metal/native/tensor/tensor.h"
#include "engine/backends/metal/native/stream.h"
#include "engine/ops/ops.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace tyloi::weights {

namespace {
void require_target(Device target) {
    if (target != Device{} && target != Device(DeviceType::metal))
        throw std::invalid_argument("Metal loader requires CPU or Metal device 0");
}
Tensor materialize_weight(Tensor value) {
    if (value.is_metal()) {
        auto array = metal::array(value);
        metal::eval(array);
    }
    observe_weight(value);
    return value;
}
}

struct Loader::Impl { Cache& cache; };

Loader::Loader(const source::Source& source, Options options, Placement placement,
        std::shared_ptr<Impl> impl)
    : source(source), options(std::move(options)), placement(std::move(placement)),
      impl_(std::move(impl)) {}

Loader Cache::loader(const source::Source& source, Loader::Options options,
        Loader::Placement placement) {
    require_target(placement.target);
    return Loader(source, std::move(options), std::move(placement),
                  std::make_shared<Loader::Impl>(Loader::Impl{*this}));
}

Cache& Loader::cache() const noexcept { return impl_->cache; }

DenseTensor Loader::host_value(const std::string& name) const {
    const auto& record = require_tensor(source, name);
    ScalarType type;
    if (record.dtype == "F32") type = kFloat32;
    else if (record.dtype == "F16") type = kFloat16;
    else if (record.dtype == "BF16") type = kBFloat16;
    else if (record.dtype == "I32") type = kInt32;
    else if (record.dtype == "I64") type = kInt64;
    else throw std::invalid_argument("expected dense weight: " + name);
    const auto blob = read_tensor(source, name);
    if (placement.drop_cache) source.drop_file_cache();
    const auto layout = dense::parse(blob, scalar_size(type), record.shape);
    const auto bytes = blob.size() - layout.offset;
    auto payload = std::make_shared<std::vector<std::uint8_t>>(bytes);
    if (bytes) std::memcpy(payload->data(), blob.data() + layout.offset, bytes);
    auto storage = std::make_shared<TensorStorage>();
    storage->owner = payload;
    storage->base = payload->data();
    storage->bytes = bytes;
    storage->device = Device{};
    return DenseTensor(storage, make_contiguous_view(storage->base, layout.shape, type, Device{}));
}

Tensor Loader::host(const std::string& name) const {
    Tensor result(host_value(name));
    observe_weight(result);
    return result;
}

Tensor Loader::native(const std::string& name) const {
    require_target(placement.target);
    auto cpu = host_value(name);
    if (placement.target.is_cpu()) {
        Tensor result(std::move(cpu));
        observe_weight(result);
        return result;
    }
    auto result = metal::copy_from_host(cpu.data_ptr(), cpu.sizes().vec(), cpu.scalar_type());
    observe_weight(result);
    return result;
}

Tensor Loader::weight(const std::string& name, std::optional<TensorParallelAxis> axis,
                      const std::vector<TensorParallelSlice>* slices) const {
    require_target(placement.target);
    if ((placement.tensor.enabled() && axis != TensorParallelAxis::Mirrored) || slices)
        throw std::invalid_argument("Metal weight loading does not support tensor-parallel slices");
    const auto* record = source.find_tensor(name);
    const bool encoded = (!record && is_tq_table(source, name)) ||
        (record && is_quant_dtype(record->dtype));
    if (!encoded) return native(name);
    if (placement.target.is_cpu())
        throw std::invalid_argument("Metal CPU-offloaded encoded weights are not available");
    auto value = encoding::load_metal_encoded(source, name);
    if (placement.drop_cache) source.drop_file_cache();
    return value;
}

Tensor Loader::tensor(const std::string& name) const {
    const auto* record = source.find_tensor(name);
    const bool encoded = (!record && is_tq_table(source, name)) ||
        (record && is_quant_dtype(record->dtype));
    if (!encoded) {
        auto value = native(name);
        if (value.scalar_type() == kFloat16 || value.scalar_type() == kBFloat16)
            value = ops::contiguous(ops::cast(value, kFloat32));
        return materialize_weight(std::move(value));
    }
    const auto value = weight(name);
    if (value.dim() != 2) throw std::invalid_argument("dense parameter requires a matrix: " + name);
    const auto ids = ops::arange(0, value.size(0), placement.target);
    return materialize_weight(ops::contiguous(ops::cast(ops::embedding(ids, value), kFloat32)));
}

Tensor Loader::norm(const std::string& name, std::int64_t width) const {
    auto value = tensor(name);
    if (value.dim() != 1 || value.size(0) != width)
        throw std::invalid_argument("normalization width mismatch: " + name);
    return materialize_weight(ops::contiguous(ops::cast(value, kFloat32)));
}

BiasWeight Loader::biased(const std::string& weight_name, const std::string& bias_name) const {
    BiasWeight result{weight(weight_name), {}};
    if (!bias_name.empty() && has(bias_name)) result.bias = tensor(bias_name);
    return result;
}
BiasWeight Loader::affine(const std::string& weight_name, const std::string& bias_name) const {
    auto result = biased(weight_name, bias_name);
    if (result.bias && (result.bias.dim() != 1 || result.bias.size(0) != result.weight.size(0)))
        throw std::invalid_argument("affine bias shape mismatch: " + bias_name);
    return result;
}
BiasWeight bias_rows(const BiasWeight& value, std::int64_t begin, std::int64_t count) {
    if (begin < 0 || count < 0 || begin > value.weight.size(0) || count > value.weight.size(0) - begin)
        throw std::out_of_range("affine row range is invalid");
    BiasWeight result{ops::contiguous(ops::slice(value.weight, 0, begin, begin + count)), {}};
    if (value.bias) result.bias = ops::contiguous(ops::slice(value.bias, 0, begin, begin + count));
    return result;
}

std::vector<Tensor> Loader::group(const std::vector<std::string>& names, std::size_t prefix,
        const std::vector<TensorParallelSlice>* slices, bool) const {
    if (names.empty() || prefix > names.size()) throw std::invalid_argument("invalid projection group");
    std::vector<Tensor> result;
    result.reserve(names.size());
    for (const auto& name : names) result.push_back(weight(name, std::nullopt, slices));
    return result;
}
std::vector<Tensor> Loader::gate_up(const std::vector<std::string>& names,
        const Tensor&, std::size_t prefix, bool preserve_boundaries) const {
    return group(names, std::min(prefix, names.size()), nullptr, preserve_boundaries);
}
Tensor Loader::embedding(const std::string& name, bool) const { return weight(name); }
Tensor Loader::rows(const std::string& name) const {
    if (is_tq_table(source, name)) return std::make_shared<Rows>(tq_table(source, name));
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (require_tensor(source, name).dtype == "NINT")
        return std::make_shared<Rows>(load_nint_rows(source, name));
#endif
    const auto& dtype = require_tensor(source, name).dtype;
    if (dtype == "F32" || dtype == "F16" || dtype == "BF16")
        return load_dense_rows(source, name);
    return weight(name);
}
Tensor Loader::dense(const std::string& name) const {
    const auto& dtype = require_tensor(source, name).dtype;
    if (dtype == "F32" || dtype == "F16" || dtype == "BF16") return native(name);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (dtype == "NINT" || dtype == "NINT8-0") return tensor(name);
#endif
    throw std::invalid_argument("expected a dense parameter: " + name);
}
Tensor Loader::experts(const std::string& name, bool, int layer, const std::string& role,
                       bool cpu_offloaded) const {
    require_target(placement.target);
    if (cpu_offloaded || placement.target.is_cpu())
        throw std::invalid_argument("Metal CPU-offloaded expert banks are not available");
    const auto& dtype = require_tensor(source, name).dtype;
    if (dtype == "F32" || dtype == "F16" || dtype == "BF16") return native(name);
    return impl_->cache.experts(source, name, layer, role);
}
Tensor Loader::routed(const std::string& name, int layer, std::int64_t experts,
        std::int64_t output, std::int64_t input, const std::string& role) const {
    auto value = this->experts(name, true, layer, role);
    if (value.dim() != 3 || value.size(0) != experts || value.size(1) != output || value.size(2) != input)
        throw std::invalid_argument("routed expert bank shape mismatch: " + name);
    return value;
}
std::vector<Tensor> Loader::routed_gate_up(const std::string& prefix, int layer,
        std::int64_t experts, std::int64_t width, std::int64_t input, const std::string&) const {
    const auto base = prefix + ".experts";
    const auto gate = base + ".gate.weight", up = base + ".up.weight";
    if (has(gate) != has(up)) throw std::invalid_argument("incomplete routed gate/up pair: " + base);
    if (!has(gate)) return {routed(base + ".gate_up.weight", layer, experts, width * 2, input, "gate_up")};
    return {routed(gate, layer, experts, width, input, "gate"),
            routed(up, layer, experts, width, input, "up")};
}

} // namespace tyloi::weights

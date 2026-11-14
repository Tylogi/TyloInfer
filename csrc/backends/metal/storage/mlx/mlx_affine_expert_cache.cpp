#include "mlx_affine_expert_cache.h"
#include "mlx_affine_indirect.h"
#include "mfq_container.h"
#include "mlx_expert_segment_cache.h"
#include "mlx_kernel_prepare.h"
#include "mlx_moe_ops.h"
#include "engine/backends/metal/native/resident_allocations.h"
#include "engine/weights/encoding/mlx_affine.h"
#include "engine/weights/source/source.h"
#include "engine/weights/source/read.h"

#include <climits>
#include <algorithm>

namespace tyloi::metal {
namespace {
using namespace mlx::core;
using Layer = std::array<tyloi::MlxAffineLayout, 3>;

const std::string& dtype(const MfqContainer& source, const std::string& name) { return source.record(name).dtype; }
const std::string& dtype(const tyloi::source::Source& source, const std::string& name) { return tyloi::require_tensor(source, name).dtype; }
std::uint64_t extent(const MfqContainer& source, const std::string& name) { return source.record(name).nbytes; }
std::uint64_t extent(const tyloi::source::Source& source, const std::string& name) { return tyloi::require_tensor(source, name).nbytes; }
std::vector<std::byte> read_prefix(const MfqContainer& source, const std::string& name, std::size_t size) {
    std::vector<std::byte> bytes(size);
    source.read_range_into(name, 0, bytes);
    return bytes;
}
std::vector<std::byte> read_prefix(const tyloi::source::Source& source, const std::string& name, std::size_t size) {
    return source.read(name, 0, size);
}
void prepared(const MfqContainer& source, const std::string& name) { source.record_prepared(name); }
void prepared(const tyloi::source::Source&, const std::string&) {}

array ffn(const std::array<MlxAffineIndirectWeight, 3>& weights, const array& input,
    const array& ids, const array& scores, bool partial = false) {
    std::optional<MlxExpertRouteTiles> plan;
    if (ids.shape(0) > MlxAffineIndirectWeight::kVectorRows) plan = mlx_expert_route_tiles(ids, weights[0].expert_count());
    const auto* routes = plan ? &*plan : nullptr;
    auto intermediate = weights[0].routed_swiglu(weights[1], input, ids, routes);
    auto pairs = weights[2].routed_matmul(intermediate, ids, false, routes);
    return moe_weighted_reduce(partial ? astype(pairs, float32) : pairs, scores);
}

std::array<MlxAffineIndirectWeight, 3> bind_affine_layer(const Layer& layer, const std::vector<array>& tables,
    const std::shared_ptr<tyloi::metal::ResidentAllocations>& residency) {
    const auto make = [&](int projection) {
        const auto& layout = layer[projection];
        Shape shape; for (const auto dim : layout.shape) shape.push_back(int(dim));
        const auto dtype = layout.scale_dtype == "BF16" ? bfloat16 : layout.scale_dtype == "F16" ? float16 : float32;
        return MlxAffineIndirectWeight::from_address_tables(std::move(shape), layout.bits, layout.group_size, dtype,
            {tables[projection * 3], tables[projection * 3 + 1], tables[projection * 3 + 2]}, residency);
    };
    return {make(0), make(1), make(2)};
}
}

namespace {
struct Description { std::vector<Layer> layers; std::vector<tyloi::weights::Segments> segments; };
template<class Source>
Description describe(const Source& model, const std::vector<std::string>& prefixes) {
        Description result;
        for (const auto& prefix : prefixes) {
            Layer layer;
            tyloi::weights::Segments definition;
            for (int p = 0; p < 3; ++p) {
                const auto name = prefix + ".mlp.experts." + std::array{"gate", "up", "down"}[p] + ".weight";
                if (dtype(model, name) != "MLX-AFFINE")
                    throw std::invalid_argument("MLX expert cache requires native affine tensors");
                auto bytes = read_prefix(model, name, 8);
                const auto length = tyloi::MlxAffineLayout::prefix_size(bytes.data(), bytes.size());
                bytes = read_prefix(model, name, length);
                auto& layout = layer[p];
                layout = tyloi::MlxAffineLayout::parse(bytes.data(), bytes.size());
                if (layout.shape.size() != 3 || layout.total_bytes() != extent(model, name) ||
                    std::any_of(layout.shape.begin(), layout.shape.end(), [](auto dim) { return dim < 1 || dim > INT_MAX; }))
                    throw std::invalid_argument("invalid MLX expert tensor geometry");
                definition.experts = int(layout.shape[0]);
                definition.fields.push_back({name, layout.prefix_bytes, std::size_t(layout.values_bytes / definition.experts)});
                definition.fields.push_back({name, layout.scales_offset(), std::size_t(layout.scales_bytes / definition.experts)});
                definition.fields.push_back({name, layout.biases_offset(), std::size_t(layout.scales_bytes / definition.experts)});
                prepared(model, name);
            }
            if (layer[0].shape != layer[1].shape || layer[2].shape !=
                std::vector<std::int64_t>{layer[0].shape[0], layer[0].shape[2], layer[0].shape[1]})
                throw std::invalid_argument("MLX expert Gate/Up/Down geometry differs");
            result.layers.push_back(std::move(layer)); result.segments.push_back(std::move(definition));
        }
        return result;
}
} // namespace

struct MlxAffineExpertCache::Plan::Impl {
    std::vector<Layer> layers;
    MlxExpertSegmentCache::Plan segments;
    Impl(std::vector<Layer> value, MlxExpertSegmentCache::Plan plan)
        : layers(std::move(value)), segments(std::move(plan)) {}
};

struct MlxAffineExpertCache::Impl {
    std::vector<Layer> layers;
    std::vector<std::optional<std::array<MlxAffineIndirectWeight, 3>>> weights;
    std::unique_ptr<MlxExpertSegmentCache> cache;

    Impl(std::vector<Layer> value, MlxExpertSegmentCache::Plan plan, std::size_t budget, int workers)
        : layers(std::move(value)), cache(std::make_unique<MlxExpertSegmentCache>(std::move(plan), budget, workers)) {
        weights.resize(layers.size());
    }
    Impl(const MfqContainer& model, const std::vector<std::string>& prefixes, std::size_t budget, int workers) {
        auto description = describe(model, prefixes);
        layers = std::move(description.layers);
        cache = std::make_unique<MlxExpertSegmentCache>(model, std::move(description.segments), budget, workers);
        weights.resize(layers.size());
    }
    const std::array<MlxAffineIndirectWeight, 3>& layer_weights(std::size_t index) {
        if (!weights[index]) weights[index] = bind_affine_layer(layers[index], cache->address_tables(index), cache->residency());
        return *weights[index];
    }
};

MlxAffineExpertCache::Plan::Plan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MlxAffineExpertCache::Plan::Plan(Plan&&) noexcept = default;
MlxAffineExpertCache::Plan& MlxAffineExpertCache::Plan::operator=(Plan&&) noexcept = default;
MlxAffineExpertCache::Plan::~Plan() = default;
std::size_t MlxAffineExpertCache::Plan::minimum() const noexcept { return impl_->segments.minimum(); }
std::size_t MlxAffineExpertCache::Plan::capacity() const noexcept { return impl_->segments.capacity(); }
MlxAffineExpertCache::Plan MlxAffineExpertCache::plan(
        const tyloi::source::Source& source, const std::vector<std::string>& prefixes) {
    auto description = describe(source, prefixes);
    auto segments = MlxExpertSegmentCache::plan(source, std::move(description.segments));
    return Plan(std::make_unique<Plan::Impl>(std::move(description.layers), std::move(segments)));
}
MlxAffineExpertCache::MlxAffineExpertCache(Plan plan, std::size_t budget, int workers)
    : impl_(std::make_shared<Impl>(std::move(plan.impl_->layers), std::move(plan.impl_->segments), budget, workers)) {}
MlxAffineExpertCache::MlxAffineExpertCache(const MfqContainer& model, const std::vector<std::string>& prefixes,
    std::size_t budget, int workers) : impl_(std::make_shared<Impl>(model, prefixes, budget, workers)) {}
MlxAffineExpertCache::MlxAffineExpertCache(const tyloi::source::Source& source, const std::vector<std::string>& prefixes,
    std::size_t budget, int workers) : MlxAffineExpertCache(plan(source, prefixes), budget, workers) {}
void MlxAffineExpertCache::validate_layer(std::size_t index, int experts, int input, int intermediate) const {
    if (impl_->layers.at(index)[0].shape != std::vector<std::int64_t>{experts, intermediate, input})
        throw std::invalid_argument("MLX expert cache layer differs from model geometry");
}
void MlxAffineExpertCache::prepare_layer(std::size_t index, int top_k) const {
    auto* preparation = MlxKernelPreparation::current();
    if (!preparation) return;
    const auto& layer = impl_->layers.at(index);
    if (top_k < 1 || top_k > layer[0].shape[0]) throw std::invalid_argument("invalid MLX preparation route count");
    std::vector<array> tables;
    for (int field = 0; field < 9; ++field) {
        const std::vector<std::uint64_t> empty(layer[0].shape[0], 0);
        tables.emplace_back(empty.begin(), Shape{int(layer[0].shape[0])}, uint64);
    }
    const auto residency = std::shared_ptr<tyloi::metal::ResidentAllocations>(tyloi::metal::ResidentAllocations::retain_arrays(tables, 1u << 20, true));
    const auto weights = bind_affine_layer(layer, tables, residency);
    preparation->collect([&] {
        for (const auto rows : preparation->row_buckets())
            preparation->add(ffn(weights, zeros({rows, int(layer[0].shape[2])}, preparation->activation_dtype()),
                zeros({rows, top_k}, int32), zeros({rows, top_k}, float32)));
    });
}
array MlxAffineExpertCache::apply_ffn(std::size_t index, const array& input, const array& ids, const array& weights,
    const array* parallel) {
    const auto& shape = impl_->layers.at(index)[0].shape;
    if (ids.ndim() != 2 || weights.shape() != ids.shape() || input.ndim() < 1 || input.shape(-1) != shape[2] ||
        input.size() / shape[2] != std::size_t(ids.shape(0)))
        throw std::invalid_argument("MLX expert FFN inputs disagree");
    std::optional<MlxExpertRouteTiles> plan;
    auto result = impl_->cache->apply(index, ids, 6, [&] {
        const auto& layer = impl_->layer_weights(index);
        if (ids.shape(0) > MlxAffineIndirectWeight::kVectorRows && !plan) plan = mlx_expert_route_tiles(ids, layer[0].expert_count());
        return layer[0].routed_swiglu(layer[1], input, ids, plan ? &*plan : nullptr);
    }, [&](const array& hidden, bool partial) {
        auto pairs = impl_->layer_weights(index)[2].routed_matmul(hidden, ids, false, plan ? &*plan : nullptr);
        return moe_weighted_reduce(partial ? astype(pairs, float32) : pairs, weights);
    }, parallel);
    return result.dtype() == input.dtype() ? result : astype(result, input.dtype());
}
std::size_t MlxAffineExpertCache::limit_bytes() const noexcept { return impl_->cache->limit_bytes(); }
void MlxAffineExpertCache::preload() { impl_->cache->preload(); }
std::size_t MlxAffineExpertCache::backing_bytes() const noexcept { return impl_->cache->backing_bytes(); }
std::size_t MlxAffineExpertCache::resident_bytes() const noexcept { return impl_->cache->resident_bytes(); }
std::size_t MlxAffineExpertCache::set_limit(std::size_t bytes) { return impl_->cache->set_limit(bytes); }
MlxSsdExpertCacheStats MlxAffineExpertCache::stats() const { return impl_->cache->stats(); }
void MlxAffineExpertCache::clear() { impl_->cache->clear(); }
} // namespace tyloi::metal

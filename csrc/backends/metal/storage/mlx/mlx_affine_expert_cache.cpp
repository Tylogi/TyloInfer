#include "mlx_affine_expert_cache.h"
#include "mlx_affine_indirect.h"
#include "mlx_affine_expert_cache_internal.h"
#include "mlx_expert_segment_cache.h"
#include "mlx_kernel_prepare.h"
#include "mlx_moe_ops.h"
#include "engine/backends/metal/native/resident_allocations.h"
#include "engine/weights/encoding/mlx_affine.h"
#include "engine/weights/source/source.h"
#include "engine/weights/source/read.h"

#include <climits>
#include <algorithm>
#include <cmath>

namespace tyloi::metal {
namespace affine_cache_detail {
const std::string& dtype(const tyloi::source::Source& source, const std::string& name) { return tyloi::require_tensor(source, name).dtype; }
std::uint64_t extent(const tyloi::source::Source& source, const std::string& name) { return tyloi::require_tensor(source, name).nbytes; }
std::vector<std::byte> read_prefix(const tyloi::source::Source& source, const std::string& name, std::size_t size) {
    return source.read(name, 0, size);
}
void prepared(const tyloi::source::Source&, const std::string&) {}

} // namespace affine_cache_detail
using affine_cache_detail::Layer;
using affine_cache_detail::describe;

namespace {
using namespace mlx::core;
using namespace affine_cache_detail;

array ffn(const std::array<MlxAffineIndirectWeight, 3>& weights, const array& input,
    const array& ids, const array& scores, float limit, bool partial = false) {
    std::optional<MlxExpertRouteTiles> plan;
    if (ids.shape(0) > MlxAffineIndirectWeight::kVectorRows) plan = mlx_expert_route_tiles(ids, weights[0].expert_count());
    const auto* routes = plan ? &*plan : nullptr;
    auto intermediate = weights[0].routed_swiglu(weights[1], input, ids, routes, limit);
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




struct MlxAffineExpertCache::Impl {
    std::vector<Layer> layers;
    std::vector<std::optional<std::array<MlxAffineIndirectWeight, 3>>> weights;
    std::unique_ptr<MlxExpertSegmentCache> cache;

    Impl(std::vector<Layer> value, MlxExpertSegmentCache::Plan plan, std::size_t budget, int workers)
        : layers(std::move(value)), cache(std::make_unique<MlxExpertSegmentCache>(std::move(plan), budget, workers)) {
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
MlxAffineExpertCache::MlxAffineExpertCache(const tyloi::source::Source& source, const std::vector<std::string>& prefixes,
    std::size_t budget, int workers) : MlxAffineExpertCache(plan(source, prefixes), budget, workers) {}
void MlxAffineExpertCache::validate_layer(std::size_t index, int experts, int input, int intermediate) const {
    if (impl_->layers.at(index)[0].shape != std::vector<std::int64_t>{experts, intermediate, input})
        throw std::invalid_argument("MLX expert cache layer differs from model geometry");
}
void MlxAffineExpertCache::prepare_layer(std::size_t index, int top_k, float limit) const {
    if (!std::isfinite(limit) || limit < 0.f) throw std::invalid_argument("invalid MLX SwiGLU limit");
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
                zeros({rows, top_k}, int32), zeros({rows, top_k}, float32), limit));
    });
}
array MlxAffineExpertCache::apply_ffn(std::size_t index, const array& input, const array& ids, const array& weights,
    const array* parallel, float swiglu_limit) {
    if (!std::isfinite(swiglu_limit) || swiglu_limit < 0.f) throw std::invalid_argument("invalid MLX SwiGLU limit");
    const auto& shape = impl_->layers.at(index)[0].shape;
    if (ids.ndim() != 2 || weights.shape() != ids.shape() || input.ndim() < 1 || input.shape(-1) != shape[2] ||
        input.size() / shape[2] != std::size_t(ids.shape(0)))
        throw std::invalid_argument("MLX expert FFN inputs disagree");
    std::optional<MlxExpertRouteTiles> plan;
    auto result = impl_->cache->apply(index, ids, 6, [&] {
        const auto& layer = impl_->layer_weights(index);
        if (ids.shape(0) > MlxAffineIndirectWeight::kVectorRows && !plan) plan = mlx_expert_route_tiles(ids, layer[0].expert_count());
        return layer[0].routed_swiglu(layer[1], input, ids, plan ? &*plan : nullptr, swiglu_limit);
    }, [&](const array& hidden, bool partial) {
        auto pairs = impl_->layer_weights(index)[2].routed_matmul(hidden, ids, false, plan ? &*plan : nullptr);
        return moe_weighted_reduce(partial ? astype(pairs, float32) : pairs, weights);
    }, parallel);
    return result.dtype() == input.dtype() ? result : astype(result, input.dtype());
}
std::size_t MlxAffineExpertCache::limit_bytes() const noexcept { return impl_->cache->limit_bytes(); }
void MlxAffineExpertCache::preload() { impl_->cache->preload(); }
void MlxAffineExpertCache::drain() { impl_->cache->drain(); }
std::size_t MlxAffineExpertCache::backing_bytes() const noexcept { return impl_->cache->backing_bytes(); }
std::size_t MlxAffineExpertCache::resident_bytes() const noexcept { return impl_->cache->resident_bytes(); }
std::size_t MlxAffineExpertCache::set_limit(std::size_t bytes) { return impl_->cache->set_limit(bytes); }
MlxSsdExpertCacheStats MlxAffineExpertCache::stats() const { return impl_->cache->stats(); }
void MlxAffineExpertCache::clear() { impl_->cache->clear(); }
} // namespace tyloi::metal

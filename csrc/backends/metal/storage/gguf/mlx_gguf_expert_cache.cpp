#include "mlx_gguf_expert_cache.h"
#include "mlx_gguf_prefill.h"
#include "mlx_gguf_expert_cache_internal.h"
#include "mlx_expert_segment_cache.h"
#include "mlx_expert_segment_cache_internal.h"
#include "mlx_kernel_prepare.h"
#include "mlx_moe_ops.h"
#include "engine/backends/metal/native/resident_allocations.h"
#include "engine/weights/source/source.h"
#include "engine/weights/source/read.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>

namespace tyloi::metal {
namespace gguf_cache_detail {
std::size_t metadata_reservation(const std::vector<Layer>& layers) {
    std::size_t total = 0;
    const auto add = [&](std::size_t bytes) {
        const auto reserved = segment_metadata_reservation(bytes);
        if (reserved > SIZE_MAX - total) throw std::overflow_error("GGUF metadata extent is too large");
        total += reserved;
    };
    for (const auto& layer : layers) for (const auto& projection : layer) {
        if (!projection.expert_dtypes.empty()) add(std::size_t(projection.shape[0]) * 2 * sizeof(std::uint32_t));
        if (projection.rotation) add(projection.rotation->signs.empty() ? sizeof(float) :
                                      projection.rotation->signs.size() * sizeof(float));
    }
    return total;
}
const std::unordered_map<std::string, std::string>& metadata(const tyloi::source::Source& source) { return source.metadata(); }
const std::string& dtype(const tyloi::source::Source& source, const std::string& name) { return tyloi::require_tensor(source, name).dtype; }
std::uint64_t extent(const tyloi::source::Source& source, const std::string& name) { return tyloi::require_tensor(source, name).nbytes; }
std::vector<std::int64_t> shape(const tyloi::source::Source& source, const std::string& name, nlohmann::json& shapes) {
    const auto& tensor = tyloi::require_tensor(source, name);
    if (tensor.shape) return *tensor.shape;
    if (shapes.is_null()) {
        const auto found = source.metadata().find("tensor_shapes");
        if (found == source.metadata().end()) throw std::runtime_error("missing tensor shape: " + name);
        shapes = nlohmann::json::parse(found->second);
    }
    if (!shapes.contains(name)) throw std::runtime_error("missing tensor shape: " + name);
    return shapes.at(name).get<std::vector<std::int64_t>>();
}
void prepared(const tyloi::source::Source&, const std::string&) {}

std::optional<Projection::Rotation> rotation(const tyloi::source::Source& source,
        const nlohmann::json& layouts, const std::string& name) {
    if (const auto* transforms = source.hadamard_transforms()) {
        const auto found = transforms->find(name);
        if (found == transforms->end()) return std::nullopt;
        return Projection::Rotation{found->second.block_size,
            std::vector<float>(found->second.signs.begin(), found->second.signs.end())};
    }
    if (!source.metadata().contains("gguf_hadamard")) {
        const auto required = source.metadata().find("gguf_requires_hadamard");
        if (required != source.metadata().end() && required->second == "true")
            throw std::invalid_argument("GGUF Hadamard descriptors are missing");
    }
    if (!layouts.contains(name)) return std::nullopt;
    const auto& entry = layouts.at(name);
    return Projection::Rotation{entry.at("block_size").get<std::int64_t>(), entry.at("signs").get<std::vector<float>>()};
}
bool normalized_rotations(const tyloi::source::Source& source) { return source.hadamard_transforms() != nullptr; }
} // namespace gguf_cache_detail
using gguf_cache_detail::Layer;
using gguf_cache_detail::describe;

namespace {
using namespace mlx::core;
using namespace gguf_cache_detail;

void configure(MlxGgufWeight& weight, const Projection& projection) {
    if (projection.rotation) weight.set_hadamard(projection.rotation->block, projection.rotation->signs);
}

std::optional<MlxGgufPrefillPlan> ffn_plan(const std::array<MlxGgufWeight, 3>& w,
    const array& input, const array& ids) {
    if (w[0].has_mixed_formats() || w[1].has_mixed_formats() || w[2].has_mixed_formats()) return std::nullopt;
    if (!gguf_compact_ffn_available(w[0], w[1], w[2], ids) &&
        !(gguf_expert_prefill_available(input) && !gguf_short_sparse_routes(ids, w[0].expert_count()))) return std::nullopt;
    const int tile = gguf_prefill_tile_rows(w[0], w[1], w[2], ids);
    if (tile == 16 && (w[0].has_staged_planes() || w[1].has_staged_planes() || w[2].has_staged_planes())) return std::nullopt;
    return MlxGgufPrefillPlan(input, ids, w[0].expert_count(), tile);
}

array ffn(const std::array<MlxGgufWeight, 3>& w, const array& input,
    const array& ids, const array& weights, bool float_accumulation = false, float limit = 0.f) {
    if (const auto plan = ffn_plan(w, input, ids))
        return plan->down(w[2], plan->gate_up(w[0], w[1], nullptr, nullptr, limit), weights, float_accumulation);
    std::optional<MlxGgufRoutePlan> plan;
    if (ids.shape(0) > 6) plan = gguf_route_plan(ids, w[0].expert_count(),
        w[0].has_mixed_formats() && w[1].has_mixed_formats() && w[2].has_mixed_formats());
    const auto* routes = plan ? &*plan : nullptr;
    auto hidden = w[0].routed_swiglu(w[1], input, ids, routes, limit);
    if (!float_accumulation) return w[2].routed_weighted_matmul(hidden, ids, weights, routes);
    auto pairs = w[2].routed_matmul(hidden, ids, false, routes);
    return moe_weighted_reduce(float_accumulation ? astype(pairs, float32) : pairs, weights);
}
}




struct MlxGgufExpertCache::Impl {
    std::vector<Layer> layers;
    std::vector<std::optional<std::array<MlxGgufWeight, 3>>> weights;
    std::unique_ptr<MlxExpertSegmentCache> cache;

    Impl(std::vector<Layer> value, MlxExpertSegmentCache::Plan plan, std::size_t budget, int workers)
        : layers(std::move(value)), cache(std::make_unique<MlxExpertSegmentCache>(std::move(plan), budget, workers, true)) {
        weights.resize(layers.size());
    }
    const std::array<MlxGgufWeight, 3>& layer_weights(std::size_t index) {
        if (!weights[index]) {
            const auto& layer = layers[index]; const auto& tables = cache->address_tables(index);
            const auto load = [&](int p) {
                auto weight = layer[p].expert_dtypes.empty()
                    ? MlxGgufWeight::from_address_table(layer[p].dtype, layer[p].shape, tables[p], cache->residency())
                    : MlxGgufWeight::from_mixed_address_table(layer[p].shape, layer[p].expert_dtypes, tables[p], cache->residency());
                configure(weight, layer[p]);
                return weight;
            };
            auto loaded = std::array{load(0), load(1), load(2)};
            std::vector<std::shared_ptr<tyloi::TensorStorage>> storage;
            for (const auto& weight : loaded) {
                auto backing = weight.backings();
                storage.insert(storage.end(), backing.begin(), backing.end());
            }
            cache->record_metadata(storage);
            weights[index] = std::move(loaded);
        }
        return *weights[index];
    }
};

MlxGgufExpertCache::Plan::Plan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MlxGgufExpertCache::Plan::Plan(Plan&&) noexcept = default;
MlxGgufExpertCache::Plan& MlxGgufExpertCache::Plan::operator=(Plan&&) noexcept = default;
MlxGgufExpertCache::Plan::~Plan() = default;
std::size_t MlxGgufExpertCache::Plan::minimum() const noexcept { return impl_->segments.minimum(); }
std::size_t MlxGgufExpertCache::Plan::capacity() const noexcept { return impl_->segments.capacity(); }
MlxGgufExpertCache::Plan MlxGgufExpertCache::plan(
        const tyloi::source::Source& source, const std::vector<std::string>& prefixes) {
    auto description = describe(source, prefixes);
    auto segments = MlxExpertSegmentCache::plan(source, std::move(description.segments));
    segments.reserve_metadata(gguf_cache_detail::metadata_reservation(description.layers));
    return Plan(std::make_unique<Plan::Impl>(std::move(description.layers), std::move(segments)));
}
MlxGgufExpertCache::MlxGgufExpertCache(Plan plan, std::size_t budget, int workers)
    : impl_(std::make_shared<Impl>(std::move(plan.impl_->layers), std::move(plan.impl_->segments), budget, workers)) {}
MlxGgufExpertCache::MlxGgufExpertCache(const tyloi::source::Source& source,
    const std::vector<std::string>& prefixes, std::size_t budget, int workers)
    : MlxGgufExpertCache(plan(source, prefixes), budget, workers) {}

void MlxGgufExpertCache::validate_layer(std::size_t layer, int experts, int input, int intermediate) const {
    if (impl_->layers.at(layer)[0].shape != Shape{experts, intermediate, input})
        throw std::invalid_argument("GGUF expert cache layer does not match model geometry");
}
void MlxGgufExpertCache::prepare_layer(std::size_t index, int top_k, float limit) const {
    if (!std::isfinite(limit) || limit < 0.f) throw std::invalid_argument("invalid GGUF SwiGLU limit");
    auto* preparation = MlxKernelPreparation::current();
    if (!preparation) return;
    const auto& layer = impl_->layers.at(index);
    if (top_k < 1 || top_k > layer[0].shape[0]) throw std::invalid_argument("invalid GGUF preparation route count");
    if (std::any_of(layer.begin(), layer.end(), [](const Projection& p) { return !p.expert_dtypes.empty(); })) {
        const auto metadata_weight = [&](int p) {
            auto table = zeros({layer[p].shape[0]}, uint64);
            table.eval();
            const auto bytes = static_cast<MTL::Buffer*>(table.buffer().ptr())->allocatedSize();
            std::shared_ptr<tyloi::metal::ResidentAllocations> residency = tyloi::metal::ResidentAllocations::retain_arrays({table}, bytes, true);
            auto weight = layer[p].expert_dtypes.empty()
                ? MlxGgufWeight::from_address_table(layer[p].dtype, layer[p].shape, table, std::move(residency))
                : MlxGgufWeight::from_mixed_address_table(layer[p].shape, layer[p].expert_dtypes, table, std::move(residency));
            configure(weight, layer[p]);
            return weight;
        };
        const std::array weights{metadata_weight(0), metadata_weight(1), metadata_weight(2)};
        preparation->collect([&] {
            for (const auto rows : preparation->row_buckets())
                for (const bool partial : {false, true})
                    preparation->add(ffn(weights, zeros({rows, layer[0].shape[2]}, float16),
                        zeros({rows, top_k}, int32), zeros({rows, top_k}, float32), partial, limit));
        });
        return;
    }
    std::array<std::vector<array>, 3> values;
    for (int p = 0; p < 3; ++p) values[p].push_back(zeros({layer[p].shape[1], layer[p].stride}, uint8));
    eval(values[0][0], values[1][0], values[2][0]);
    const std::vector<std::int32_t> selected{0};
    std::array<MlxGgufWeight, 3> weights{
        MlxGgufWeight::from_experts(layer[0].dtype, layer[0].shape, selected, values[0]),
        MlxGgufWeight::from_experts(layer[1].dtype, layer[1].shape, selected, values[1]),
        MlxGgufWeight::from_experts(layer[2].dtype, layer[2].shape, selected, values[2])};
    for (int p = 0; p < 3; ++p) configure(weights[p], layer[p]);
    preparation->collect([&] {
        for (const auto rows : preparation->row_buckets())
            for (const bool partial : {false, true})
                preparation->add(ffn(weights, zeros({rows, layer[0].shape[2]}, float16),
                    zeros({rows, top_k}, int32), zeros({rows, top_k}, float32), partial, limit));
    });
}
array MlxGgufExpertCache::apply_ffn(std::size_t layer_index, const array& input,
    const array& ids, const array& route_weights, const array* parallel, float limit) {
    if (!std::isfinite(limit) || limit < 0.f) throw std::invalid_argument("invalid GGUF SwiGLU limit");
    const auto& layer = impl_->layers.at(layer_index);
    if (ids.ndim() != 2 || route_weights.shape() != ids.shape() || input.ndim() == 0 ||
        input.shape(-1) != layer[0].shape[2] || input.size() / layer[0].shape[2] != std::size_t(ids.shape(0)))
        throw std::invalid_argument("GGUF expert FFN inputs and routes disagree");
    std::optional<MlxGgufRoutePlan> plan;
    std::optional<MlxGgufPrefillPlan> prefill;
    auto output = impl_->cache->apply(layer_index, ids, 2, [&] {
        const auto& weights = impl_->layer_weights(layer_index);
        if (!prefill) prefill = ffn_plan(weights, input, ids);
        if (prefill) {
            return prefill->gate_up(weights[0], weights[1], nullptr, nullptr, limit);
        }
        if (!plan && ids.shape(0) > 6) plan = gguf_route_plan(ids, weights[0].expert_count(),
            weights[0].has_mixed_formats() && weights[1].has_mixed_formats() && weights[2].has_mixed_formats());
        return weights[0].routed_swiglu(weights[1], input, ids, plan ? &*plan : nullptr, limit);
    }, [&](const array& hidden, bool partial) {
        if (prefill) return prefill->down(impl_->layer_weights(layer_index)[2], hidden, route_weights, partial);
        if (!partial) return impl_->layer_weights(layer_index)[2].routed_weighted_matmul(hidden, ids, route_weights,
            plan ? &*plan : nullptr);
        auto pairs = impl_->layer_weights(layer_index)[2].routed_matmul(hidden, ids, false, plan ? &*plan : nullptr);
        return moe_weighted_reduce(partial ? astype(pairs, float32) : pairs, route_weights);
    }, parallel);
    return output.dtype() == input.dtype() ? output : astype(output, input.dtype());
}
void MlxGgufExpertCache::set_prefill_next_rows(std::size_t layer, std::size_t rows) { impl_->cache->set_prefill_next_rows(layer, rows); }
void MlxGgufExpertCache::preload() { impl_->cache->preload(); }
void MlxGgufExpertCache::drain() { impl_->cache->drain(); }
std::size_t MlxGgufExpertCache::limit_bytes() const noexcept { return impl_->cache->limit_bytes(); }
std::size_t MlxGgufExpertCache::backing_bytes() const noexcept { return impl_->cache->backing_bytes(); }
std::size_t MlxGgufExpertCache::resident_bytes() const noexcept { return impl_->cache->resident_bytes(); }
std::size_t MlxGgufExpertCache::set_limit(std::size_t bytes) { return impl_->cache->set_limit(bytes); }
MlxSsdExpertCacheStats MlxGgufExpertCache::stats() const { return impl_->cache->stats(); }
void MlxGgufExpertCache::clear() { impl_->cache->clear(); }
} // namespace tyloi::metal

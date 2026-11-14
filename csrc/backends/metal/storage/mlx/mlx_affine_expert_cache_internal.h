#pragma once
#include "mlx_affine_expert_cache.h"
#include "mlx_expert_segment_cache.h"
#include "engine/weights/encoding/mlx_affine.h"
#include <algorithm>
#include <array>
#include <climits>

namespace tyloi::metal {
namespace affine_cache_detail {
using Layer = std::array<tyloi::MlxAffineLayout, 3>;
const std::string& dtype(const MfqContainer&, const std::string&);
const std::string& dtype(const tyloi::source::Source&, const std::string&);
std::uint64_t extent(const MfqContainer&, const std::string&);
std::uint64_t extent(const tyloi::source::Source&, const std::string&);
std::vector<std::byte> read_prefix(const MfqContainer&, const std::string&, std::size_t);
std::vector<std::byte> read_prefix(const tyloi::source::Source&, const std::string&, std::size_t);
void prepared(const MfqContainer&, const std::string&);
void prepared(const tyloi::source::Source&, const std::string&);
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
} // namespace affine_cache_detail
struct MlxAffineExpertCache::Plan::Impl {
    std::vector<affine_cache_detail::Layer> layers;
    MlxExpertSegmentCache::Plan segments;
    Impl(std::vector<affine_cache_detail::Layer> value, MlxExpertSegmentCache::Plan plan)
        : layers(std::move(value)), segments(std::move(plan)) {}
};
} // namespace tyloi::metal

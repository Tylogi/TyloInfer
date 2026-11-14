#pragma once
#include "mlx_gguf_expert_cache.h"
#include "mlx_expert_segment_cache.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <unordered_map>

namespace tyloi::metal {
namespace gguf_cache_detail {
using mlx::core::Shape;
struct Projection {
    struct Rotation { std::int64_t block; std::vector<float> signs; };
    std::string dtype;
    Shape shape;
    int stride = 0;
    std::vector<std::string> expert_dtypes;
    std::optional<Rotation> rotation;
};
using Layer = std::array<Projection, 3>;
std::size_t metadata_reservation(const std::vector<Layer>& layers);


using Metadata = std::unordered_map<std::string, std::string>;
const Metadata& metadata(const MfqContainer&);
const Metadata& metadata(const tyloi::source::Source&);
const std::string& dtype(const MfqContainer&, const std::string&);
const std::string& dtype(const tyloi::source::Source&, const std::string&);
std::uint64_t extent(const MfqContainer&, const std::string&);
std::uint64_t extent(const tyloi::source::Source&, const std::string&);
std::vector<std::int64_t> shape(const MfqContainer&, const std::string&, nlohmann::json&);
std::vector<std::int64_t> shape(const tyloi::source::Source&, const std::string&, nlohmann::json&);
void prepared(const MfqContainer&, const std::string&);
void prepared(const tyloi::source::Source&, const std::string&);
std::optional<Projection::Rotation> rotation(const MfqContainer&, const nlohmann::json&, const std::string&);
std::optional<Projection::Rotation> rotation(const tyloi::source::Source&, const nlohmann::json&, const std::string&);
bool normalized_rotations(const MfqContainer&);
bool normalized_rotations(const tyloi::source::Source&);
struct Description { std::vector<Layer> layers; std::vector<tyloi::weights::Segments> segments; };
template<class Source>
Description describe(const Source& model, const std::vector<std::string>& prefixes) {
        Description result;
        const Metadata& meta = metadata(model);
        const auto mixed = meta.find("gguf_mixed_experts");
        const nlohmann::json mixed_layouts = mixed == meta.end() ? nlohmann::json::object()
            : nlohmann::json::parse(mixed->second);
        const auto rotations = meta.find("gguf_hadamard");
        const nlohmann::json rotation_layouts = rotations == meta.end() || normalized_rotations(model) ? nlohmann::json::object()
            : nlohmann::json::parse(rotations->second);
        nlohmann::json shapes;
        for (const auto& prefix : prefixes) {
            Layer layer;
            tyloi::weights::Segments definition;
            for (int p = 0; p < 3; ++p) {
                const auto name = prefix + ".mlp.experts." + std::array{"gate", "up", "down"}[p] + ".weight";
                auto& projection = layer[p];
                projection.dtype = dtype(model, name);
                const auto shape = gguf_cache_detail::shape(model, name, shapes);
                projection.rotation = rotation(model, rotation_layouts, name);
                if (!is_gguf_dtype(projection.dtype) || shape.size() != 3 ||
                    std::any_of(shape.begin(), shape.end(), [](auto d) { return d < 1 || d > INT_MAX; }))
                    throw std::invalid_argument("GGUF expert cache requires native rank-three tensors");
                for (const auto d : shape) projection.shape.push_back(int(d));
                if (projection.dtype == "GGUF_MIXED") {
                    const auto& entries = mixed_layouts.at(name);
                    if (!entries.is_array() || entries.size() != std::size_t(shape[0]))
                        throw std::invalid_argument("GGUF mixed cache expert count differs");
                    tyloi::weights::Segment field{name};
                    std::size_t next = 0;
                    for (const auto& entry : entries) {
                        const auto dtype = entry.at("dtype").get<std::string>();
                        const auto offset = entry.at("offset").get<std::uint64_t>();
                        const auto bytes = entry.at("nbytes").get<std::size_t>();
                        const auto stride = gguf_packed_row_bytes(dtype, projection.shape[2]);
                        if (offset != next || offset > extent(model, name) || bytes > extent(model, name) - offset ||
                            stride > INT_MAX || bytes != std::size_t(shape[1]) * stride)
                            throw std::invalid_argument("GGUF mixed cache expert byte range differs");
                        field.expert_offsets.push_back(offset); field.expert_sizes.push_back(bytes);
                        projection.expert_dtypes.push_back(dtype); next += bytes;
                    }
                    if (next != extent(model, name)) throw std::invalid_argument("GGUF mixed cache has trailing bytes");
                    definition.experts = int(shape[0]); definition.fields.push_back(std::move(field));
                    prepared(model, name);
                    continue;
                }
                const auto stride = gguf_packed_row_bytes(projection.dtype, projection.shape[2]);
                if (stride > INT_MAX || std::size_t(shape[1]) > SIZE_MAX / stride)
                    throw std::overflow_error("GGUF expert extent is too large");
                projection.stride = int(stride);
                const auto bytes = std::size_t(shape[1]) * stride;
                if (bytes > SIZE_MAX / shape[0] || bytes * shape[0] != extent(model, name))
                    throw std::invalid_argument("GGUF expert extent differs from tensor geometry");
                definition.experts = int(shape[0]); definition.fields.push_back({name, 0, bytes});
                prepared(model, name);
            }
            if (layer[0].shape != layer[1].shape || layer[2].shape !=
                Shape{layer[0].shape[0], layer[0].shape[2], layer[0].shape[1]})
                throw std::invalid_argument("GGUF expert Gate/Up/Down geometry differs");
            result.layers.push_back(std::move(layer)); result.segments.push_back(std::move(definition));
        }
        return result;
}
} // namespace gguf_cache_detail
struct MlxGgufExpertCache::Plan::Impl {
    std::vector<gguf_cache_detail::Layer> layers;
    MlxExpertSegmentCache::Plan segments;
    Impl(std::vector<gguf_cache_detail::Layer> value, MlxExpertSegmentCache::Plan plan)
        : layers(std::move(value)), segments(std::move(plan)) {}
};
} // namespace tyloi::metal

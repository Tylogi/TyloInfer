#pragma once
#include "mlx_expert_segment_cache.h"
#include <cstdint>
#include <functional>
#include <span>
#include <unordered_map>
#include <utility>

namespace tyloi::metal {
using mlx::core::array;
struct MlxExpertSegmentCache::Plan::Impl {
    struct Reader {
        std::uint64_t extent;
        std::function<void(std::uint64_t, std::span<const std::span<std::byte>>)> read;
    };
    struct Layer {
        tyloi::weights::Segments source;
        tyloi::weights::SegmentLayout layout;
        std::vector<array> tables;
        Layer(tyloi::weights::Segments definition, std::span<const std::size_t> extents, std::size_t page)
            : source(std::move(definition)), layout(source, extents, page) {}
    };
    std::unordered_map<std::string, Reader> readers;
    std::vector<Layer> layers;
    std::size_t backing = 0, maximum_entry = 0, capacity = 0;
    std::size_t metadata_reserve = 0, table_reserve = 0;
    long double total_entry_bytes = 0, maximum_layer_bytes = 0;
};
using SegmentPlan = MlxExpertSegmentCache::Plan::Impl;
std::size_t segment_metadata_reservation(std::size_t bytes);
std::unique_ptr<SegmentPlan> segment_make_plan(std::vector<tyloi::weights::Segments> definitions,
    const std::function<SegmentPlan::Reader(const std::string&)>& open);
} // namespace tyloi::metal

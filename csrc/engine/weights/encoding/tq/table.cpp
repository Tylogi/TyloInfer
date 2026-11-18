#include "engine/weights/encoding/tq/table.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tyloi {
std::vector<std::uint8_t> join_rows(
        const std::vector<std::vector<std::uint8_t>>& blobs) {
    if (blobs.empty()) throw std::invalid_argument("empty TQ row selection");
    const TqLayout first(blobs.front().data(), blobs.front().size());
    std::uint64_t rows = 0, packed_bytes = 0;
    for (const auto& blob : blobs) {
        const TqLayout layout(blob.data(), blob.size());
        if (layout.width != first.width || layout.columns != first.columns ||
            layout.gs != first.gs) {
            throw std::invalid_argument("incompatible TQ row segments");
        }
        if (first.columns && std::memcmp(
                blobs.front().data() + first.scales,
                blob.data() + layout.scales, first.scale_bytes)) {
            throw std::invalid_argument("TQ row segments have different column scales");
        }
        if (rows > UINT32_MAX - layout.rows ||
            packed_bytes > UINT32_MAX - layout.packed_bytes) {
            throw std::overflow_error("selected TQ rows exceed RTRELL05 limits");
        }
        rows += layout.rows;
        packed_bytes += layout.packed_bytes;
    }
    const auto scale_bytes = first.columns
        ? first.scale_bytes : (rows * first.groups + 1) / 2;
    const auto scale_offset = 64 + rows * 8;
    const auto packed_offset = scale_offset + scale_bytes;
    if (packed_offset > SIZE_MAX - packed_bytes) {
        throw std::overflow_error("selected TQ rows exceed host size limits");
    }
    std::vector<std::uint8_t> result(packed_offset + packed_bytes);
    std::memcpy(result.data(), blobs.front().data(), 64);
    TqLayout::integer(result.data() + 12, rows, 4);
    TqLayout::integer(result.data() + 28, packed_bytes, 4);
    std::uint64_t row_offset = 0, byte_offset = 0;
    if (first.columns) {
        std::memcpy(result.data() + scale_offset,
                    blobs.front().data() + first.scales, scale_bytes);
    }
    for (const auto& blob : blobs) {
        const TqLayout layout(blob.data(), blob.size());
        for (std::size_t row = 0; row < layout.rows; ++row) {
            auto descriptor = TqLayout::integer(blob.data() + 64 + row * 8, 8);
            descriptor += byte_offset / 4;
            TqLayout::integer(result.data() + 64 + (row_offset + row) * 8,
                              descriptor, 8);
        }
        if (!layout.columns) {
            for (std::size_t i = 0; i < std::size_t(layout.rows) * layout.groups; ++i) {
                const auto selector =
                    (blob[layout.scales + i / 2] >> ((i & 1) * 4)) & 15;
                const auto index = row_offset * layout.groups + i;
                result[scale_offset + index / 2] |=
                    selector << ((index & 1) * 4);
            }
        }
        std::memcpy(result.data() + packed_offset + byte_offset,
                    blob.data() + layout.packed, layout.packed_bytes);
        row_offset += layout.rows;
        byte_offset += layout.packed_bytes;
    }
    return result;
}

TqTable::TqTable(std::vector<std::shared_ptr<TqRows>> segments,
                 std::int64_t columns, std::uint64_t expected_rows,
                 std::uint64_t storage_bytes, std::vector<TqOrigin> readers,
                 std::vector<std::size_t> origins)
    : segments_(std::move(segments)), readers_(std::move(readers)), origins_(std::move(origins)),
      columns_(columns), storage_bytes_(storage_bytes) {
    if (segments_.empty() || columns_ < 1 || expected_rows < 1) {
        throw std::invalid_argument("invalid TQ row table geometry");
    }
    if ((!readers_.empty() || !origins_.empty()) &&
        (origins_.size() != segments_.size() ||
         std::any_of(origins_.begin(), origins_.end(), [&](auto origin) {
             return origin >= readers_.size() || !readers_[origin].read;
         }))) throw std::invalid_argument("invalid TQ row source readers");
    std::uint64_t end = 0;
    ends_.reserve(segments_.size());
    for (const auto& segment : segments_) {
        if (!segment || segment->layout().width < static_cast<std::uint64_t>(columns_)) {
            throw std::invalid_argument("TQ row segment width differs");
        }
        if (end > UINT64_MAX - segment->layout().rows) {
            throw std::overflow_error("TQ row count exceeds bounds");
        }
        end += segment->layout().rows;
        ends_.push_back(end);
    }
    if (end != expected_rows) {
        throw std::invalid_argument("TQ table row count differs");
    }
}

std::vector<std::uint8_t> TqTable::slice_rows(
        std::uint64_t begin, std::uint64_t end) const {
    if (begin >= end || end > rows()) {
        throw std::out_of_range("invalid TQ table row range");
    }
    std::uint64_t start = 0;
    std::vector<std::vector<std::uint8_t>> pieces;
    for (std::size_t index = 0; index < segments_.size(); ++index) {
        const auto first = std::max(begin, start);
        const auto last = std::min(end, ends_[index]);
        if (first < last) {
            pieces.push_back(segments_[index]->slice_rows(first - start, last - start));
        }
        start = ends_[index];
        if (start >= end) break;
    }
    if (pieces.size() == 1) return std::move(pieces.front());
    return join_rows(pieces);
}

std::size_t TqTable::expert_image_bytes(std::uint64_t expert, std::uint64_t count) const {
    if (!count || rows() % count || expert >= rows() / count)
        throw std::out_of_range("invalid TQ expert image range");
    const auto begin = expert * count, end = begin + count;
    auto segment = std::upper_bound(ends_.begin(), ends_.end(), begin) - ends_.begin();
    const auto& layout = segments_.at(segment)->layout();
    std::size_t packed = 0;
    while (segment < segments_.size()) {
        const auto start = segment ? ends_[segment - 1] : 0;
        const auto first = std::max(begin, start), last = std::min(end, ends_[segment]);
        if (first < last) {
            const auto& part = segments_[segment]->layout();
            if (part.width != layout.width || part.columns != layout.columns || part.gs != layout.gs)
                throw std::invalid_argument("incompatible TQ expert row segments");
            const auto bytes = segments_[segment]->packed_row_bytes(first - start, last - start);
            if (bytes > SIZE_MAX - packed) throw std::overflow_error("TQ expert packed bytes");
            packed += bytes;
        }
        if (ends_[segment] >= end) break;
        ++segment;
    }
    const auto scales = layout.columns ? layout.scale_bytes : (count * layout.groups + 1) / 2;
    const auto metadata = (64 + count * 8 + scales + 3) & ~std::uint64_t(3);
    if (metadata > SIZE_MAX - 32 || packed > SIZE_MAX - 32 - metadata)
        throw std::overflow_error("TQ expert image bytes");
    return std::size_t(metadata) + packed + 32;
}

std::vector<std::vector<std::uint8_t>> TqTable::slice_experts(
        const std::vector<std::int64_t>& experts, std::uint64_t rows_per_expert) const {
    if (!rows_per_expert || rows() % rows_per_expert)
        throw std::invalid_argument("invalid TQ expert row count");
    std::vector<std::vector<std::vector<std::uint8_t>>> pieces(experts.size());
    std::vector<std::vector<ReadSpan>> spans(readers_.size());
    for (std::size_t index = 0; index < experts.size(); ++index) {
        const auto expert = experts[index];
        if (expert < 0 || std::uint64_t(expert) >= rows() / rows_per_expert)
            throw std::out_of_range("TQ expert ID");
        const auto begin = std::uint64_t(expert) * rows_per_expert, end = begin + rows_per_expert;
        auto segment = std::upper_bound(ends_.begin(), ends_.end(), begin) - ends_.begin();
        while (segment < segments_.size()) {
            const auto start = segment ? ends_[segment - 1] : 0;
            const auto first = std::max(begin, start), last = std::min(end, ends_[segment]);
            if (first < last) pieces[index].push_back(segments_[segment]->slice_rows(
                first - start, last - start, readers_.empty() ? nullptr : &spans[origins_[segment]]));
            if (ends_[segment] >= end) break;
            ++segment;
        }
    }
    for (std::size_t origin = 0; origin < readers_.size(); ++origin) if (!spans[origin].empty()) {
        const auto& reader = readers_[origin];
        if (reader.batch) reader.batch(spans[origin]);
        else for (const auto& span : spans[origin]) reader.read(span.offset, span.destination, span.size);
    }
    std::vector<std::vector<std::uint8_t>> result;
    result.reserve(experts.size());
    for (auto& parts : pieces)
        result.push_back(parts.size() == 1 ? std::move(parts.front()) : join_rows(parts));
    return result;
}

std::size_t TqTable::row_image_bytes_bound() const {
    std::size_t bytes = 0;
    for (const auto& segment : segments_) {
        const auto& layout = segment->layout();
        if (layout.width > INT_MAX) {
            throw std::overflow_error("TQ row image width exceeds limits");
        }
        const auto scales = layout.columns ? layout.scale_bytes : (layout.groups + 1) / 2;
        bytes = std::max(bytes,
            ((64 + 8 + scales + 3) & ~std::size_t(3)) + layout.width + 32);
    }
    return (bytes + 7) & ~std::size_t(7);
}

} // namespace tyloi

#pragma once

#include "engine/weights/encoding/tq/rows.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tyloi {

struct TqOrigin {
    TqRead read;
    std::function<void(const std::vector<ReadSpan>&)> batch;
};

std::vector<std::uint8_t> join_rows(
    const std::vector<std::vector<std::uint8_t>>& blobs);

// A logical row table assembled from independently readable TQ segments.
// It owns only segment readers and contains no model-source or device state.
class TqTable {
    std::vector<std::shared_ptr<TqRows>> segments_;
    std::vector<std::uint64_t> ends_;
    std::vector<TqOrigin> readers_;
    std::vector<std::size_t> origins_;
    std::int64_t columns_ = 0;
    std::uint64_t storage_bytes_ = 0;
    bool parallel_reads_ = false;

public:
    TqTable(std::vector<std::shared_ptr<TqRows>> segments,
            std::int64_t columns, std::uint64_t expected_rows,
            std::uint64_t storage_bytes, std::vector<TqOrigin> readers = {},
            std::vector<std::size_t> origins = {}, bool parallel_reads = false);

    std::uint64_t rows() const noexcept { return ends_.back(); }
    std::int64_t width() const noexcept { return columns_; }
    std::uint64_t storage_bytes() const noexcept { return storage_bytes_; }
    bool parallel_reads() const noexcept { return parallel_reads_; }
    // Prepare before publishing the table to concurrent readers.
    void prepare_routed_metadata();
    std::size_t routed_image_storage_bound(std::uint64_t experts) const;
    std::size_t row_image_bytes_bound() const;
    std::vector<std::uint8_t> slice_rows(std::uint64_t begin,
                                         std::uint64_t end) const;
    std::size_t expert_image_bytes(std::uint64_t expert, std::uint64_t count) const;
    std::vector<std::vector<std::uint8_t>> slice_experts(
        const std::vector<std::int64_t>& experts, std::uint64_t rows_per_expert) const;

    std::size_t segment_count() const noexcept { return segments_.size(); }
    std::uint64_t segment_end(std::size_t index) const { return ends_.at(index); }
    const TqRows& segment(std::size_t index) const { return *segments_.at(index); }
};

} // namespace tyloi

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace tyloi {

class FileRangeReader;

struct ResidualRow {
    std::uint64_t offset = 0;
    std::uint32_t type = 0;
    std::uint32_t bytes = 0;
};

class RowResidualFile {
    std::shared_ptr<FileRangeReader> reader_;
    std::uint64_t columns_ = 0;
    std::uint64_t payload_offset_ = 0;
    std::uint64_t payload_bytes_ = 0;
    std::vector<ResidualRow> rows_;

public:
    explicit RowResidualFile(const std::filesystem::path& path);
    std::uint64_t columns() const noexcept { return columns_; }
    std::uint64_t row_count() const noexcept { return rows_.size(); }
    std::uint64_t payload_bytes() const noexcept { return payload_bytes_; }
    const std::vector<ResidualRow>& rows() const noexcept { return rows_; }
    void read_payload(std::uint64_t offset, void* output, std::size_t bytes) const;
    std::vector<std::uint8_t> read_row(std::uint64_t index) const;
};

} // namespace tyloi

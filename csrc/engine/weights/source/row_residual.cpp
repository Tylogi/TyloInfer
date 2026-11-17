#include "row_residual.h"

#include "engine/io/file_range_reader.h"
#include "engine/weights/encoding/gguf_quant.h"

#include <climits>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tyloi {
namespace {

std::uint64_t integer(const std::uint8_t* bytes, unsigned count) {
    std::uint64_t value = 0;
    for (unsigned index = 0; index < count; ++index)
        value |= std::uint64_t(bytes[index]) << (8 * index);
    return value;
}

} // namespace

RowResidualFile::RowResidualFile(const std::filesystem::path& path) {
    reader_ = std::make_shared<FileRangeReader>(path, FileReadMode::Buffered);
    const auto size = reader_->size();
    std::uint8_t header[48];
    reader_->read(0, reinterpret_cast<std::byte*>(header), sizeof(header));
    const auto count = integer(header + 16, 8);
    const auto padded = integer(header + 32, 8);
    columns_ = integer(header + 24, 8);
    payload_bytes_ = integer(header + 40, 8);
    if (std::memcmp(header, "TQRES001", 8) || integer(header + 8, 4) != 1 ||
            integer(header + 12, 4) || !count || !columns_ ||
            columns_ > std::uint64_t(INT64_MAX) - 127 ||
            padded != (columns_ + 127) / 128 * 128 || size < sizeof(header) ||
            count > (size - sizeof(header)) / 16) {
        throw std::invalid_argument("invalid row residual header");
    }
    payload_offset_ = sizeof(header) + count * 16;
    if (payload_bytes_ != size - payload_offset_ ||
            count > std::uint64_t(SIZE_MAX) / sizeof(ResidualRow)) {
        throw std::invalid_argument("row residual extent differs");
    }
    rows_.reserve(static_cast<std::size_t>(count));
    std::uint64_t cursor = 0;
    for (std::uint64_t index = 0; index < count; ++index) {
        std::uint8_t descriptor[16];
        reader_->read(sizeof(header) + index * 16,
            reinterpret_cast<std::byte*>(descriptor), sizeof(descriptor));
        ResidualRow row{
            integer(descriptor, 8),
            static_cast<std::uint32_t>(integer(descriptor + 8, 4)),
            static_cast<std::uint32_t>(integer(descriptor + 12, 4))};
        const bool valid = row.type >= 144 && row.type <= 151;
        const auto expected = row.type
            ? valid ? gguf::row_bytes(ggml_type(row.type), static_cast<std::int64_t>(padded))
                    : std::numeric_limits<std::uint64_t>::max()
            : 0;
        if (row.offset != cursor || row.bytes != expected ||
                cursor > payload_bytes_ || row.bytes > payload_bytes_ - cursor) {
            throw std::invalid_argument("invalid row residual descriptor");
        }
        cursor += row.bytes;
        rows_.push_back(row);
    }
    if (cursor != payload_bytes_)
        throw std::invalid_argument("unreferenced residual data");
}

void RowResidualFile::read_payload(
        std::uint64_t offset, void* output, std::size_t bytes) const {
    if (offset > payload_bytes_ || bytes > payload_bytes_ - offset ||
            (bytes && !output)) {
        throw std::out_of_range("residual payload read is out of bounds");
    }
    reader_->read(payload_offset_ + offset,
        reinterpret_cast<std::byte*>(output), bytes);
}

std::vector<std::uint8_t> RowResidualFile::read_row(std::uint64_t index) const {
    const auto& row = rows_.at(static_cast<std::size_t>(index));
    std::vector<std::uint8_t> data(row.bytes);
    read_payload(row.offset, data.data(), data.size());
    return data;
}

} // namespace tyloi

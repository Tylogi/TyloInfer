#include "hf.h"

#include "safetensors.h"
#include "components/ggml/include/ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tyloi::hf {

namespace {

std::uint32_t packed_u32(const std::byte* bytes) {
    std::uint32_t result = 0;
    for (unsigned i = 0; i < 4; ++i)
        result |= std::uint32_t(std::to_integer<unsigned char>(bytes[i])) << (8 * i);
    return result;
}

std::uint16_t packed_u16(const std::byte* bytes) {
    return std::uint16_t(std::to_integer<unsigned char>(bytes[0])) |
        (std::uint16_t(std::to_integer<unsigned char>(bytes[1])) << 8);
}

float packed_scale(const std::byte* bytes, std::string_view dtype) {
    if (dtype == "F16") return ggml_fp16_to_fp32(packed_u16(bytes));
    const auto bits = dtype == "BF16"
        ? std::uint32_t(packed_u16(bytes)) << 16 : packed_u32(bytes);
    float value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void read_packed_source(const Safetensors& files, const std::string& name,
        std::uint64_t offset, std::byte* destination, std::size_t size) {
    const auto found = files.records().find(name);
    if (found == files.records().end() || offset > found->second.metadata.nbytes ||
            size > found->second.metadata.nbytes - offset)
        throw std::out_of_range("packed INT4 source range is out of bounds: " + name);
    files.read(found->second.location.shard,
        found->second.location.offset + offset, destination, size);
}

void read_packed_int4(const Safetensors& files, const PackedInt4& packed,
        std::uint64_t offset, std::byte* destination, std::size_t count) {
    if (offset > packed.bytes || count > packed.bytes - offset || (count && !destination))
        throw std::out_of_range("packed INT4 execution read is out of bounds");
    if (!count) return;
    const auto first = offset / 20;
    const auto last = (offset + count + 19) / 20;
    const auto tile_rows = std::max<std::uint64_t>(1,
        (std::uint64_t{8} << 20) /
            (std::uint64_t(packed.width / 2) +
             std::uint64_t(packed.groups) * packed.scale_bytes));
    const auto row_first = first / std::uint64_t(packed.blocks);
    const auto row_last = (last - 1) / std::uint64_t(packed.blocks) + 1;
    for (auto row = row_first; row < row_last;) {
        const auto end = std::min<std::uint64_t>(row_last, row + tile_rows);
        const auto rows = end - row;
        std::vector<std::byte> values(rows * std::uint64_t(packed.width / 2));
        std::vector<std::byte> scales(rows * std::uint64_t(packed.groups) * packed.scale_bytes);
        read_packed_source(files, packed.values, row * std::uint64_t(packed.width / 2),
            values.data(), values.size());
        read_packed_source(files, packed.scales,
            row * std::uint64_t(packed.groups) * packed.scale_bytes,
            scales.data(), scales.size());
        const auto zero_row = row / 8;
        const auto zero_end = (end + 7) / 8;
        std::vector<std::byte> zeros;
        if (packed.has_zero_points) {
            zeros.resize((zero_end - zero_row) * std::uint64_t(packed.groups) * 4);
            read_packed_source(files, packed.zero_points,
                zero_row * std::uint64_t(packed.groups) * 4, zeros.data(), zeros.size());
        }
        for (auto r = row; r < end; ++r) {
            const auto begin_block = std::max<std::uint64_t>(first, r * std::uint64_t(packed.blocks));
            const auto end_block = std::min<std::uint64_t>(last, (r + 1) * std::uint64_t(packed.blocks));
            for (auto block = begin_block; block < end_block; ++block) {
                const auto column = (block - r * std::uint64_t(packed.blocks)) * 32;
                const auto group = column / std::uint64_t(packed.group_size);
                const auto scale_offset =
                    ((r - row) * std::uint64_t(packed.groups) + group) * packed.scale_bytes;
                const auto d = packed_scale(scales.data() + scale_offset, packed.scale_dtype);
                if (!(d >= 0) || !std::isfinite(d))
                    throw std::invalid_argument("packed INT4 scale must be finite and nonnegative");
                const auto z = packed.has_zero_points
                    ? (packed_u32(zeros.data() +
                        ((r / 8 - zero_row) * std::uint64_t(packed.groups) + group) * 4) >>
                        (4 * (r % 8))) & 15u : 8u;
                std::array<std::byte, 20> image{};
                const auto scale_bits = ggml_fp32_to_fp16(d);
                const auto bias_bits = ggml_fp32_to_fp16(-d * float(z));
                if (!std::isfinite(ggml_fp16_to_fp32(scale_bits)) ||
                        !std::isfinite(ggml_fp16_to_fp32(bias_bits)))
                    throw std::runtime_error("packed INT4 scale exceeds GGUF execution range");
                image[0] = std::byte(scale_bits); image[1] = std::byte(scale_bits >> 8);
                image[2] = std::byte(bias_bits); image[3] = std::byte(bias_bits >> 8);
                const auto* source = values.data() +
                    (r - row) * std::uint64_t(packed.width / 2) + column / 2;
                for (unsigned i = 0; i < 16; ++i) {
                    const auto q0 = (std::to_integer<unsigned char>(source[i / 2]) >> (4 * (i % 2))) & 15u;
                    const auto q1 = (std::to_integer<unsigned char>(source[8 + i / 2]) >> (4 * (i % 2))) & 15u;
                    image[4 + i] = std::byte(q0 | (q1 << 4));
                }
                const auto begin = std::max<std::uint64_t>(offset, block * 20);
                const auto stop = std::min<std::uint64_t>(offset + count, (block + 1) * 20);
                std::copy_n(image.data() + begin - block * 20,
                    std::size_t(stop - begin), destination + begin - offset);
            }
        }
        row = end;
    }
}

} // namespace

void Record::read(
    const Safetensors& files,
    std::uint64_t relative_offset,
    std::byte* destination,
    std::size_t size) const {
    if (relative_offset > metadata.nbytes ||
        size > metadata.nbytes - relative_offset) {
        throw std::out_of_range("model tensor byte range is out of bounds: " +
                                metadata.name);
    }
    const auto request_end = relative_offset + size;
    for (const auto& segment : segments) {
        const auto segment_size = segment.inline_bytes.empty()
            ? segment.nbytes : segment.inline_bytes.size();
        if (segment_size > std::numeric_limits<std::uint64_t>::max() -
                           segment.offset) {
            throw std::overflow_error(
                "HF virtual tensor segment byte size overflow");
        }
        const auto segment_end = segment.offset + segment_size;
        const auto begin = std::max(relative_offset, segment.offset);
        const auto end = std::min<std::uint64_t>(request_end, segment_end);
        if (begin >= end) continue;
        const auto target_offset = static_cast<std::size_t>(begin - relative_offset);
        const auto count = static_cast<std::size_t>(end - begin);
        if (!segment.inline_bytes.empty()) {
            std::copy_n(
                segment.inline_bytes.data() + (begin - segment.offset), count,
                destination + target_offset);
        } else if (segment.packed) {
            read_packed_int4(files, *segment.packed,
                segment.source_offset + begin - segment.offset,
                destination + target_offset, count);
        } else {
            const auto found = files.records().find(segment.source_name);
            if (found == files.records().end()) {
                throw std::runtime_error(
                    "model tensor not found: " + segment.source_name);
            }
            const auto& source = found->second;
            const auto source_offset =
                segment.source_offset + begin - segment.offset;
            if (source_offset > source.metadata.nbytes ||
                count > source.metadata.nbytes - source_offset) {
                throw std::out_of_range(
                    "model tensor byte range is out of bounds: " +
                    segment.source_name);
            }
            files.read(
                source.location.shard,
                source.location.offset + source_offset,
                destination + target_offset, count);
        }
    }
}

std::optional<TensorFileRange> Record::file_range(
    const Safetensors& files,
    std::uint64_t offset,
    std::size_t size) const {
    if (offset > metadata.nbytes || size > metadata.nbytes - offset) {
        throw std::out_of_range("HF tensor file range is out of bounds");
    }
    for (const auto& segment : segments) {
        if (!segment.inline_bytes.empty() || segment.packed || offset < segment.offset ||
            offset - segment.offset > segment.nbytes ||
            size > segment.nbytes - (offset - segment.offset)) continue;
        const auto found = files.records().find(segment.source_name);
        if (found == files.records().end()) {
            throw std::runtime_error(
                "model tensor not found: " + segment.source_name);
        }
        const auto& location = found->second.location;
        return TensorFileRange{
            files.paths().at(location.shard),
            location.offset + segment.source_offset + offset - segment.offset,
            size};
    }
    return std::nullopt;
}

} // namespace tyloi::hf

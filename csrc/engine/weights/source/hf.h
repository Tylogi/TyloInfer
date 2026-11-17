#pragma once

#include "engine/weights/source/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tyloi {

class Safetensors;

namespace hf {

// Virtual GGUF_Q4_1 image synthesized from compressed-tensors INT4 fields.
// Reads transcode only the requested blocks; the packed source stays external.
struct PackedInt4 {
    std::string values;
    std::string shape;
    std::string scales;
    std::string zero_points;
    std::string scale_dtype;
    std::int64_t rows = 0;
    std::int64_t width = 0;
    std::int64_t group_size = 0;
    std::int64_t groups = 0;
    std::int64_t blocks = 0;
    std::size_t scale_bytes = 0;
    bool has_zero_points = false;
    std::uint64_t bytes = 0;
};

struct Segment {
    std::uint64_t offset = 0;
    std::vector<std::byte> inline_bytes;
    std::string source_name;
    std::uint64_t source_offset = 0;
    std::uint64_t nbytes = 0;
    std::shared_ptr<const PackedInt4> packed;
};

struct Record {
    TensorMetadata metadata;
    std::vector<Segment> segments;

    void read(
        const Safetensors& files,
        std::uint64_t relative_offset,
        std::byte* destination,
        std::size_t size) const;
    std::optional<TensorFileRange> file_range(
        const Safetensors& files,
        std::uint64_t offset,
        std::size_t size) const;
};

} // namespace hf
} // namespace tyloi

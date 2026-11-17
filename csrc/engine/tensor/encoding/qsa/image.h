#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tyloinfer::qsa {

struct Precision {
    std::uint8_t key_bits, value_bits;
};

inline constexpr std::uint8_t affine8_tag = 254;
enum class RowPrecisionAllocation { Energy, Uniform };

struct MicroblockImage {
    std::vector<std::uint8_t> key, value;
    // 254: FP16 scale + INT8 zero per 256-element H4 mode, then INT8 codes.
    // 1..8: uniform TQ; 255: row precision allocated from H4 mode energy.
    std::uint8_t key_bits = affine8_tag, value_bits = affine8_tag;
    // Trellis address width, independent of the payload's bits per value.
    // Canonical Affine8 has no trellis; original TQ8 images use 16-bit states.
    std::uint8_t key_state_bits = 16, value_state_bits = 16;
};

// Storage is supplied by the backend. A CUDA backend can own one mapped,
// pinned image without retaining an additional pageable copy of its bytes.
struct RamKvImage {
    std::shared_ptr<const void> owner;
    const std::uint8_t* key = nullptr;
    const std::uint8_t* value = nullptr;
    std::size_t key_bytes = 0, value_bytes = 0;
    std::uint8_t key_bits = 0, value_bits = 0, key_state_bits = 16, value_state_bits = 16;
    std::size_t bytes() const { return key_bytes + value_bytes; }
};
using RamKvImagePtr = std::shared_ptr<const RamKvImage>;

} // namespace tyloinfer::qsa

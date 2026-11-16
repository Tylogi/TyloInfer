#pragma once

#include "engine/tensor/tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace tyloi::state {

// Packed TurboQuant rows hold one FP32 norm and indices for the padded rotation width.
inline int kv_packed_width(int dimension, int bits) {
    if (bits < 2 || bits > 8)
        throw std::runtime_error("invalid TurboQuant component bits");
    if (dimension <= 0 || dimension > 1024)
        throw std::runtime_error("unsupported TurboQuant KV head dimension");
    int width = 1;
    while (width < dimension) width *= 2;
    return 1 + (width * bits + 31) / 32;
}

struct KvQuantization {
    double bits = 0;
    bool enabled() const noexcept { return bits != 0; }
    int key_bits() const noexcept { return int(std::floor(bits)); }
    int value_bits() const noexcept { return int(std::ceil(bits)); }
    bool operator==(const KvQuantization&) const = default;
};

inline bool kv_bits_valid(double bits) noexcept {
    return bits == 0 || bits == 2 || bits == 2.5 || bits == 3 || bits == 3.5 || bits == 4 || bits == 6 || bits == 8;
}

// QSA index rows are stored as FP16 independently of raw KV dtype.
struct QsaLayout {
    std::int64_t maximum = 0, heads = 0, key_width = 0, value_width = 0, index_width = 0, pool = 0;
    ScalarType dtype = kFloat16;

    std::size_t row_bytes() const;
    std::size_t raw_bytes() const;
    std::size_t index_bytes() const;
    std::int64_t pooled_rows() const;
};

struct KvLayout {
    std::int64_t key_width, value_width;
    ScalarType storage_dtype;
    std::size_t bytes;
};

std::int64_t kv_capacity(std::int64_t current, std::int64_t required, std::int64_t maximum);

// Zero capacity plans an empty initial cache without allocating device storage.
KvLayout kv_layout(std::int64_t batch, std::int64_t heads, std::int64_t capacity,
    std::int64_t head_dim, bool ring, ScalarType dtype, KvQuantization quantization = {});

} // namespace tyloi::state

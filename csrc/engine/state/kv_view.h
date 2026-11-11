#pragma once

#include "engine/tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
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
};

struct KvView {
    Tensor visible_key, visible_value;
    Tensor physical_key, physical_value;
    Tensor key_chunk_ptrs, value_chunk_ptrs, page_table;
    KvQuantization quantization;
    std::int64_t page_size = 0;
    std::int64_t pages_per_chunk = 0;
    std::int64_t heads = 0;
    bool ring = false;
    std::int64_t batch = 0;
    std::int64_t head_dim = 0;
    ScalarType dtype = ScalarType::float16;

    bool paged() const noexcept { return page_size > 0; }
    bool defined() const noexcept {
        return paged()
            ? key_chunk_ptrs.defined() && value_chunk_ptrs.defined() && page_table.defined()
            : physical_key.defined() && physical_value.defined();
    }
};

std::size_t kv_bytes(std::int64_t batch, std::int64_t heads, std::int64_t capacity,
                     std::int64_t head_dim, bool ring, ScalarType dtype,
                     KvQuantization settings = {});

KvView allocate_kv(std::int64_t batch, std::int64_t heads, std::int64_t capacity,
                   std::int64_t head_dim, bool ring, Device device, ScalarType dtype,
                   KvQuantization settings = {});

} // namespace tyloi::state

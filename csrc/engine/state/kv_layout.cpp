#include "engine/state/kv_layout.h"

#include <limits>

namespace tyloi::state {
namespace {
std::size_t checked_multiply(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("KV tensor size overflow");
    return a * b;
}

} // namespace

KvLayout kv_layout(int64_t batch, int64_t heads, int64_t capacity, int64_t head_dim,
                   bool ring, ScalarType dtype, KvQuantization quantization) {
    if (batch <= 0 || heads <= 0 || capacity < 0 || head_dim <= 0)
        throw std::invalid_argument("invalid KV tensor dimensions");
    for (auto dimension : {batch, heads, capacity, head_dim})
        if (static_cast<std::uint64_t>(dimension) > std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("KV tensor size overflow");
    if (ring && quantization.enabled())
        throw std::runtime_error("TurboQuant ring cache is unsupported");
    int64_t key_width = head_dim, value_width = head_dim;
    auto storage_dtype = dtype;
    if (quantization.enabled()) {
        if (!std::isfinite(quantization.bits) || quantization.bits < 2 || quantization.bits > 8 ||
            head_dim > std::numeric_limits<int>::max())
            throw std::invalid_argument("invalid TurboQuant KV layout");
        key_width = tyloi::state::kv_packed_width(int(head_dim), quantization.key_bits());
        value_width = tyloi::state::kv_packed_width(int(head_dim), quantization.value_bits());
        storage_dtype = tyloi::kInt32;
    }
    const auto item_size = tyloi::scalar_size(storage_dtype);
    if (item_size == 0) throw std::invalid_argument("invalid KV tensor dtype");
    const auto rows = checked_multiply(checked_multiply(static_cast<std::size_t>(batch),
        static_cast<std::size_t>(heads)), static_cast<std::size_t>(capacity));
    const auto key_bytes = checked_multiply(checked_multiply(rows, static_cast<std::size_t>(key_width)), item_size);
    const auto value_bytes = checked_multiply(checked_multiply(rows, static_cast<std::size_t>(value_width)), item_size);
    if (key_bytes > std::numeric_limits<std::size_t>::max() - value_bytes)
        throw std::overflow_error("KV tensor size overflow");
    return {key_width, value_width, storage_dtype, key_bytes + value_bytes};
}

std::size_t kv_bytes(int64_t batch, int64_t heads, int64_t capacity,
        int64_t head_dim, bool ring, ScalarType dtype, KvQuantization quantization) {
    if (capacity <= 0) throw std::invalid_argument("invalid KV tensor dimensions");
    return kv_layout(batch, heads, capacity, head_dim, ring, dtype, quantization).bytes;
}

} // namespace tyloi::state

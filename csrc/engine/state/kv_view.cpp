#include "engine/state/kv_view.h"
#include "engine/ops.h"

#include <stdexcept>
#include <utility>

tyloi::state::KvView tyloi::state::allocate_kv(
        int64_t B, int64_t H, int64_t max_seq, int64_t D, bool ring,
        tyloi::Device device, tyloi::ScalarType dtype, KvQuantization quantization) {
    if (max_seq < 0 || (max_seq == 0 && !device.is_metal()))
        throw std::invalid_argument("invalid KV tensor dimensions");
    const auto layout = kv_layout(B, H, max_seq, D, ring, dtype, quantization);
    // Metal TurboQuant kernels interpret packed words as unsigned integers.
    const auto storage_dtype = device.is_metal() && quantization.enabled()
        ? ScalarType::uint32 : layout.storage_dtype;
    auto opts = ::tyloi::TensorOptions().device(device).dtype(storage_dtype);
    KvView result;
    result.physical_key = tyloi::ops::zeros({B,H,max_seq,layout.key_width},opts);
    result.physical_value = tyloi::ops::zeros({B,H,max_seq,layout.value_width},opts);
    result.quantization = quantization;
    result.ring = ring;
    result.batch = B;
    result.heads = H;
    result.head_dim = D;
    result.dtype = dtype;
    return result;
}

tyloi::state::KvView tyloi::state::grow_kv(
        const KvView& cache, std::int64_t capacity, std::int64_t keep) {
    if (!cache.defined() || cache.paged() || cache.ring ||
        cache.physical_key.is_encoded() || cache.physical_value.is_encoded())
        throw std::invalid_argument("KV growth requires an unpaged dense cache");
    const auto& key = cache.physical_key;
    const auto& value = cache.physical_value;
    if (key.dim() != 4 || value.dim() != 4 || key.device() != value.device() ||
        key.size(0) != cache.batch || value.size(0) != cache.batch ||
        key.size(1) != cache.heads || value.size(1) != cache.heads ||
        key.size(2) != value.size(2))
        throw std::invalid_argument("invalid KV cache geometry");
    const auto old_capacity = key.size(2);
    const auto layout = kv_layout(cache.batch, cache.heads, old_capacity, cache.head_dim,
                                  false, cache.dtype, cache.quantization);
    const auto storage_dtype = key.device().is_metal() && cache.quantization.enabled()
        ? ScalarType::uint32 : layout.storage_dtype;
    if (key.size(3) != layout.key_width || value.size(3) != layout.value_width ||
        key.scalar_type() != storage_dtype || value.scalar_type() != storage_dtype ||
        capacity <= old_capacity || keep < 0 || keep > old_capacity)
        throw std::invalid_argument("invalid KV cache growth");
    auto result = allocate_kv(cache.batch, cache.heads, capacity, cache.head_dim,
                              false, key.device(), cache.dtype, cache.quantization);
    if (keep) {
        auto next_key = ops::update_slice(result.physical_key, ops::slice(key, 2, 0, keep), 2, 0);
        auto next_value = ops::update_slice(result.physical_value, ops::slice(value, 2, 0, keep), 2, 0);
        result.physical_key = std::move(next_key);
        result.physical_value = std::move(next_value);
    }
    return result;
}

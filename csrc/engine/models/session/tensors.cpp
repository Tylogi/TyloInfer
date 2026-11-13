#include "engine/models/session/state.h"
#include "engine/ops.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

size_t session_tensor_bytes(const ::tyloi::Tensor &tensor) {
    if (!tensor.defined())
        return 0;
    return static_cast<size_t>(tensor.numel()) * tensor.element_size();
}

bool session_tensor_layout_matches(const ::tyloi::Tensor &target,
                                   const ::tyloi::Tensor &source) {
    if (!target.defined() || !source.defined() || target.dim() != source.dim() ||
        target.scalar_type() != source.scalar_type() || target.device() != source.device()) return false;
    for (int64_t i = 0; i < target.dim(); ++i)
        if (target.size(i) != source.size(i)) return false;
    return true;
}

::tyloi::Tensor restore_session_tensor(::tyloi::Tensor target,
                                       const ::tyloi::Tensor &source) {
    if (!source.defined()) {
        return {};
    }
    if (source.device().is_metal()) {
        if (source.is_encoded()) throw CudaSessionStateError("session tensor is not dense");
        return tyloi::ops::clone(source);
    }
    const auto* dense = source.representation<::tyloi::DenseTensor>();
    if (!dense) throw CudaSessionStateError("session tensor is not dense");
    const auto* current = target.representation<::tyloi::DenseTensor>();
    if (!current || !session_tensor_layout_matches(target, source)) {
        target = tyloi::ops::empty(dense->sizes(), source.options());
    }
    tyloi::ops::copy_into(target, source);
    return target;
}

::tyloi::Tensor restore_session_prefix_tensor(::tyloi::Tensor target,
                                               const ::tyloi::Tensor &tensor, int64_t dimension,
                                               int64_t capacity) {
    if (tensor.device().is_metal()) {
        if (!tensor.defined() || tensor.is_encoded() || dimension < 0 || dimension >= tensor.dim() ||
            capacity <= 0 || tensor.size(dimension) > capacity)
            throw CudaSessionStateError("session prefix tensor layout is invalid");
        std::vector<int64_t> shape;
        shape.reserve(static_cast<size_t>(tensor.dim()));
        for (int64_t i = 0; i < tensor.dim(); ++i) shape.push_back(tensor.size(i));
        shape[static_cast<size_t>(dimension)] = capacity;
        auto result = tyloi::ops::zeros(shape, tensor.options());
        if (tensor.size(dimension) > 0)
            result = tyloi::ops::update_slice(result, tyloi::ops::clone(tensor), dimension, 0);
        return result;
    }
    const auto* dense = tensor.representation<::tyloi::DenseTensor>();
    if (!dense) throw CudaSessionStateError("session prefix tensor is not dense");
    const auto& source = tensor;
    if (!source.defined() || dimension < 0 || dimension >= source.dim() || capacity <= 0 ||
        source.size(dimension) > capacity) {
        throw CudaSessionStateError("session prefix tensor layout is invalid");
    }
    auto target_shape = dense->sizes().vec();
    target_shape.at(static_cast<size_t>(dimension)) = capacity;
    const auto* current = target.representation<::tyloi::DenseTensor>();
    const bool compatible =
        current && current->defined() && current->sizes() == ::tyloi::IntArrayRef(target_shape) &&
        current->scalar_type() == source.scalar_type() && current->device() == source.device();
    if (!compatible) {
        target = tyloi::ops::empty(target_shape, source.options());
    }
    if (source.size(dimension) > 0) {
        auto prefix = tyloi::ops::slice(target, dimension, 0, source.size(dimension));
        tyloi::ops::copy_into(prefix, source);
    }
    return target;
}

KvSnapshot capture_kv(const tyloi::state::KvView &cache, int64_t cache_pos, size_t &bytes) {
    if (cache.physical_key.device().is_metal() || cache.physical_value.device().is_metal()) {
        const auto& key = cache.physical_key;
        const auto& value = cache.physical_value;
        const tyloi::Device device{tyloi::DeviceType::metal, 0};
        if (cache.paged() || !key.defined() || !value.defined() || key.is_encoded() || value.is_encoded() ||
            key.device() != device || value.device() != device || key.dim() != 4 || value.dim() != 4 ||
            key.size(0) != cache.batch || value.size(0) != cache.batch || cache.batch <= 0 ||
            key.size(1) != cache.heads || value.size(1) != cache.heads ||
            key.size(2) != value.size(2) || key.size(2) <= 0 || cache_pos <= 0 ||
            (!cache.ring && cache_pos > key.size(2)))
            throw std::runtime_error("full-attention KV cache is unavailable");
        const auto capacity = key.size(2);
        if (cache.dtype != tyloi::kFloat16 && cache.dtype != tyloi::kBFloat16 && cache.dtype != tyloi::kFloat32)
            throw std::runtime_error("full-attention KV cache is unavailable");
        const auto layout = tyloi::state::kv_layout(cache.batch, cache.heads, capacity, cache.head_dim,
                                                    cache.ring, cache.dtype, cache.quantization);
        const auto physical_dtype = cache.quantization.enabled() ? tyloi::ScalarType::uint32 : layout.storage_dtype;
        if (key.scalar_type() != physical_dtype || value.scalar_type() != physical_dtype ||
            key.size(3) != layout.key_width || value.size(3) != layout.value_width)
            throw std::runtime_error("full-attention KV cache is unavailable");
        const auto saved_tokens = cache.ring ? capacity : std::min(cache_pos, capacity);
        KvSnapshot state{tyloi::ops::clone(tyloi::ops::slice(key, 2, 0, saved_tokens)),
                         tyloi::ops::clone(tyloi::ops::slice(value, 2, 0, saved_tokens)),
                         capacity, cache.ring};
        bytes += session_tensor_bytes(state.k);
        bytes += session_tensor_bytes(state.v);
        return state;
    }
    if (!cache.physical_key.representation<::tyloi::DenseTensor>() ||
        !cache.physical_value.representation<::tyloi::DenseTensor>() ||
        !cache.physical_key.defined() || !cache.physical_value.defined() || cache.physical_key.dim() != 4 ||
        !session_kv_geometry_matches(cache.physical_key,cache.physical_value) || cache.physical_key.size(0) != 1) {
        throw std::runtime_error("full-attention KV cache is unavailable");
    }
    KvSnapshot state;
    state.capacity = cache.physical_key.size(2);
    state.ring = cache.ring;
    const int64_t saved_tokens =
        state.ring ? state.capacity : std::min<int64_t>(cache_pos, state.capacity);
    state.k = tyloi::ops::clone(tyloi::ops::slice(cache.physical_key, 2, 0, saved_tokens));
    state.v = tyloi::ops::clone(tyloi::ops::slice(cache.physical_value, 2, 0, saved_tokens));
    bytes += session_tensor_bytes(state.k);
    bytes += session_tensor_bytes(state.v);
    return state;
}

void validate_kv(const KvSnapshot& state) {
    if (state.k.device().is_metal() || state.v.device().is_metal()) {
        const tyloi::Device device{tyloi::DeviceType::metal, 0};
        const auto dtype = state.k.scalar_type();
        if (!state.k.defined() || !state.v.defined() || state.k.is_encoded() || state.v.is_encoded() ||
            state.k.device() != device || state.v.device() != device ||
            !session_kv_geometry_matches(state.k, state.v) || state.k.size(0) <= 0 ||
            state.k.size(1) <= 0 || state.k.size(3) <= 0 || state.v.size(3) <= 0 ||
            state.capacity <= 0 || state.k.size(2) <= 0 || state.k.size(2) > state.capacity ||
            (state.ring && state.k.size(2) != state.capacity) ||
            (dtype != tyloi::kFloat16 && dtype != tyloi::kBFloat16 && dtype != tyloi::kFloat32 &&
             dtype != tyloi::ScalarType::uint32) || (state.ring && dtype == tyloi::ScalarType::uint32))
            throw CudaSessionStateError("full-attention session KV layout is invalid");
        return;
    }
    if (!state.k.representation<::tyloi::DenseTensor>() ||
        !state.v.representation<::tyloi::DenseTensor>() ||
        state.capacity <= 0 || state.k.dim() != 4 ||
        !session_kv_geometry_matches(state.k,state.v) || state.k.size(0) != 1 ||
        state.k.size(2) > state.capacity) {
        throw CudaSessionStateError("full-attention session KV layout is invalid");
    }
}

void restore_kv(tyloi::state::KvView &cache, const KvSnapshot &state, int64_t head_dim,
                ::tyloi::ScalarType dtype, tyloi::state::KvQuantization quantization) {
    validate_kv(state);
    if (state.k.device().is_metal()) {
        const auto layout = tyloi::state::kv_layout(state.k.size(0), state.k.size(1), state.capacity,
                                                    head_dim, state.ring, dtype, quantization);
        const auto physical_dtype = quantization.enabled() ? tyloi::ScalarType::uint32 : layout.storage_dtype;
        if ((dtype != tyloi::kFloat16 && dtype != tyloi::kBFloat16 && dtype != tyloi::kFloat32) ||
            state.k.scalar_type() != physical_dtype || state.v.scalar_type() != physical_dtype ||
            state.k.size(3) != layout.key_width || state.v.size(3) != layout.value_width)
            throw CudaSessionStateError("full-attention session KV layout is invalid");
        auto key = restore_session_prefix_tensor({}, state.k, 2, state.capacity);
        auto value = restore_session_prefix_tensor({}, state.v, 2, state.capacity);
        tyloi::state::KvView restored;
        restored.physical_key = std::move(key);
        restored.physical_value = std::move(value);
        restored.ring = state.ring;
        restored.quantization = quantization;
        restored.batch = state.k.size(0);
        restored.heads = state.k.size(1);
        restored.head_dim = head_dim;
        restored.dtype = dtype;
        cache = std::move(restored);
        return;
    }
    cache.physical_key = restore_session_prefix_tensor(cache.physical_key, state.k, 2, state.capacity);
    cache.physical_value = restore_session_prefix_tensor(cache.physical_value, state.v, 2, state.capacity);
    cache.visible_key = {};
    cache.visible_value = {};
    cache.ring = state.ring;
    cache.quantization = quantization;
    cache.batch = state.k.size(0);
    cache.heads = state.k.size(1);
    cache.head_dim = head_dim;
    cache.dtype = dtype;
    cache.key_chunk_ptrs = {};
    cache.value_chunk_ptrs = {};
    cache.page_table = {};
    cache.page_size = 0;
    cache.pages_per_chunk = 0;
}

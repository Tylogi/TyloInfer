#include "engine/models/session/state.h"
#include "engine/models/session/snapshot.h"
#include "engine/ops.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

::tyloi::Tensor concatenate_session_tensors(std::span<const ::tyloi::Tensor> tensors, std::int64_t dimension) {
    if (tensors.empty() || !tensors.front() || !tensors.front().device().is_cpu() || tensors.front().is_encoded() ||
        dimension < 0 || dimension >= tensors.front().dim())
        throw CudaSessionStateError("invalid host session fragments");
    std::vector<std::int64_t> shape;
    for (int axis = 0; axis < tensors.front().dim(); ++axis) shape.push_back(tensors.front().size(axis));
    shape[dimension] = 0;
    for (const auto& tensor : tensors) {
        if (!tensor || tensor.is_encoded() || tensor.device() != tensors.front().device() ||
            tensor.scalar_type() != tensors.front().scalar_type() || tensor.dim() != tensors.front().dim())
            throw CudaSessionStateError("incompatible host session fragments");
        for (int axis = 0; axis < tensor.dim(); ++axis)
            if (axis != dimension && tensor.size(axis) != shape[axis])
                throw CudaSessionStateError("incompatible host session fragment shape");
        if (tensor.size(dimension) < 0 || shape[dimension] > INT64_MAX - tensor.size(dimension))
            throw CudaSessionStateError("host session fragment extent overflow");
        shape[dimension] += tensor.size(dimension);
    }
    auto result = tyloi::ops::empty(shape, tensors.front().options());
    std::int64_t offset = 0;
    for (const auto& tensor : tensors) {
        const auto end = offset + tensor.size(dimension);
        tyloi::ops::copy_into(tyloi::ops::slice(result, dimension, offset, end), tensor);
        offset = end;
    }
    return result;
}

std::vector<SessionBlockPayload> encode_paged_session(const TextSessionState &state,
                                                        size_t block_size, size_t first_block,
                                                        size_t maximum_blocks) {
    static constexpr std::array<uint8_t, 8> magic{'M', 'F', 'Q', 'C', 'U', 'D', '1', 0};
    const auto *layers = std::get_if<std::vector<KvSnapshot>>(&state.payload);
    // Paged prefixes are text-only; media positions are not block-sliceable.
    if (state.kind() != TextSessionStateKind::FullAttention || state.decode_position_delta != 0 ||
        !state.input_key.empty() || state.cache_pos <= 0 || layers == nullptr || layers->empty() ||
        state.tokens.size() != static_cast<size_t>(state.cache_pos)) {
        throw std::runtime_error("CUDA session state is not block-sliceable");
    }
    const size_t full_blocks = state.tokens.size() / block_size;
    if (first_block > full_blocks) {
        throw std::runtime_error("invalid CUDA cache first block");
    }
    const size_t end_block =
        maximum_blocks > full_blocks - first_block ? full_blocks : first_block + maximum_blocks;
    std::vector<SessionBlockPayload> result;
    result.reserve(end_block - first_block);
    for (size_t block_index = first_block; block_index < end_block; ++block_index) {
        const int64_t start = static_cast<int64_t>(block_index * block_size);
        SessionBlockWriter writer;
        writer.raw(magic.data(), magic.size());
        writer.scalar<uint32_t>(1);
        writer.scalar<uint32_t>(static_cast<uint32_t>(start));
        writer.scalar<uint32_t>(static_cast<uint32_t>(block_size));
        writer.scalar<uint32_t>(static_cast<uint32_t>(layers->size()));
        for (const auto &layer : *layers) {
            const auto& k = layer.k;
            const auto& v = layer.v;
            if (layer.ring || layer.capacity <= 0 || !k || !v || k.is_encoded() || v.is_encoded() ||
                k.device() != v.device() || !session_kv_geometry_matches(k,v) ||
                k.size(2) < start + static_cast<int64_t>(block_size)) {
                throw std::runtime_error("CUDA session layer cannot be block-sliced");
            }
            writer.scalar<int64_t>(layer.capacity);
            writer.tensor(tyloi::ops::slice(k, 2, start, start + static_cast<int64_t>(block_size)), layer.target_device());
            writer.tensor(tyloi::ops::slice(v, 2, start, start + static_cast<int64_t>(block_size)), layer.target_device());
        }
        result.push_back(std::move(writer).finish());
    }
    return result;
}

SessionBlockPayload encode_paged_block(const TextSessionState &state, size_t block_size,
                                         size_t block_index) {
    auto result = encode_paged_session(state, block_size, block_index, 1);
    if (result.size() != 1) {
        throw std::runtime_error("invalid CUDA cache block index");
    }
    return std::move(result.front());
}

DecodedSessionBlock decode_paged_block(const std::vector<uint8_t> &payload) {
    static constexpr std::array<uint8_t, 8> magic{'M', 'F', 'Q', 'C', 'U', 'D', '1', 0};
    SessionBlockReader reader(payload);
    std::array<uint8_t, 8> found_magic{};
    reader.raw(found_magic.data(), found_magic.size(), "magic");
    if (found_magic != magic || reader.scalar<uint32_t>("version") != 1) {
        throw CudaSessionStateError("unsupported CUDA paged cache payload");
    }
    DecodedSessionBlock block;
    block.start = reader.scalar<uint32_t>("start");
    block.count = reader.scalar<uint32_t>("count");
    const auto layer_count = reader.scalar<uint32_t>("layer count");
    if (block.count == 0 || layer_count == 0 || layer_count > 4096) {
        throw CudaSessionStateError("invalid CUDA paged cache geometry");
    }
    block.layers.reserve(layer_count);
    for (uint32_t index = 0; index < layer_count; ++index) {
        DecodedSessionBlockLayer layer;
        layer.capacity = reader.scalar<int64_t>("capacity");
        auto key = reader.tensor();
        const auto key_device = reader.tensor_device();
        auto value = reader.tensor();
        if (key_device != reader.tensor_device() || !session_kv_geometry_matches(key.first,value.first) ||
            key.first.size(2) != static_cast<int64_t>(block.count)) {
            throw CudaSessionStateError("invalid CUDA paged cache layer");
        }
        layer.device = key_device;
        layer.k = std::move(key.first);
        layer.v = std::move(value.first);
        block.layers.push_back(std::move(layer));
    }
    reader.expect_end();
    return block;
}

TextSessionState decode_paged_session(const std::vector<SessionBlockPayload> &payloads,
                                           const std::vector<int64_t> &tokens, size_t block_size) {
    if (payloads.empty() || tokens.size() != payloads.size() * block_size) {
        throw CudaSessionStateError("CUDA paged cache chain length mismatch");
    }
    std::vector<DecodedSessionBlock> blocks;
    blocks.reserve(payloads.size());
    size_t expected_start = 0;
    size_t layer_count = 0;
    for (const auto &payload : payloads) {
        if (!payload) {
            throw CudaSessionStateError("CUDA paged cache block payload is null");
        }
        auto block = decode_paged_block(*payload);
        if (block.start != expected_start || block.count != block_size ||
            (layer_count != 0 && block.layers.size() != layer_count)) {
            throw CudaSessionStateError("incompatible CUDA paged cache chain");
        }
        expected_start += block.count;
        layer_count = block.layers.size();
        blocks.push_back(std::move(block));
    }
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = static_cast<int64_t>(tokens.size());
    state.payload = std::vector<KvSnapshot>{};
    auto &layers = std::get<std::vector<KvSnapshot>>(state.payload);
    layers.reserve(layer_count);
    for (size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
        const auto &final = blocks.back().layers[layer_index];
        std::vector<int64_t> key_shape, value_shape;
        for (int axis = 0; axis < 4; ++axis) {
            key_shape.push_back(axis == 2 ? static_cast<int64_t>(tokens.size()) : final.k.size(axis));
            value_shape.push_back(axis == 2 ? static_cast<int64_t>(tokens.size()) : final.v.size(axis));
        }
        auto key = tyloi::ops::empty(key_shape, final.k.options());
        auto value = tyloi::ops::empty(value_shape, final.v.options());
        int64_t offset = 0;
        for (const auto &block : blocks) {
            const auto &layer = block.layers[layer_index];
            if (layer.capacity != final.capacity || layer.device != final.device ||
                layer.k.scalar_type() != final.k.scalar_type() ||
                layer.k.size(0) != final.k.size(0) || layer.k.size(1) != final.k.size(1) ||
                layer.k.size(3) != final.k.size(3) || layer.v.size(3)!=final.v.size(3)) {
                throw CudaSessionStateError("inconsistent CUDA paged cache topology");
            }
            tyloi::ops::copy_into(tyloi::ops::slice(key, 2, offset, offset + block.count), layer.k);
            tyloi::ops::copy_into(tyloi::ops::slice(value, 2, offset, offset + block.count), layer.v);
            offset += block.count;
        }
        state.bytes += static_cast<size_t>(key.numel() * key.element_size() +
                                           value.numel() * value.element_size());
        layers.push_back(KvSnapshot{
            std::move(key),
            std::move(value),
            final.capacity,
            false,
            final.device,
        });
    }
    return state;
}

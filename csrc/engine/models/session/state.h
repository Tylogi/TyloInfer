#pragma once

#include "engine/tensor/tensor.h"
#include "engine/models/session/paged.h"
#include "engine/state/kv/view.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

struct TextSessionState;
enum class TextSessionStateKind : int;

struct KvSnapshot {
    ::tyloi::Tensor k;
    ::tyloi::Tensor v;
    int64_t capacity = 0;
    bool ring = false;
    // Disk snapshots retain their destination while their tensors stay on the host.
    std::optional<::tyloi::Device> destination;

    ::tyloi::Device target_device() const { return destination.value_or(k.device()); }

    template <class Visit>
    void tensors(const Visit& visit) const {
        if (k) visit(k);
        if (v) visit(v);
    }
};

/** Check that key and value tensors have compatible session geometry. */
inline bool session_kv_geometry_matches(const ::tyloi::Tensor& k,
        const ::tyloi::Tensor& v) {
    if(!k.defined()||!v.defined()||k.dim()!=4||v.dim()!=4||k.scalar_type()!=v.scalar_type())return false;
    for(int axis=0;axis<3;++axis)if(k.size(axis)!=v.size(axis))return false;
    return k.scalar_type()==::tyloi::kInt32 ||
        k.scalar_type()==::tyloi::ScalarType::uint32 || k.size(3)==v.size(3);
}

enum class HybridBlockSessionStateKind {
    FullAttention,
    Recurrent,
    // The block keeps no cache state, so a checkpoint only records position.
    Stateless,
};

struct HybridBlockSessionState {
    HybridBlockSessionStateKind kind =
        HybridBlockSessionStateKind::FullAttention;
    KvSnapshot full_attention;
    ::tyloi::Tensor convolution;
    ::tyloi::Tensor recurrent;
    bool transposed = true;

    template <class Visit>
    void tensors(const Visit& visit) const {
        full_attention.tensors(visit);
        if (convolution) visit(convolution);
        if (recurrent) visit(recurrent);
    }
};

struct DecodedSessionBlockLayer {
    int64_t capacity = 0;
    ::tyloi::Device device{::tyloi::DeviceType::cpu, 0};
    ::tyloi::Tensor k;
    ::tyloi::Tensor v;
};

struct DecodedSessionBlock {
    uint32_t start = 0;
    uint32_t count = 0;
    std::vector<DecodedSessionBlockLayer> layers;
};

/** Join host snapshot fragments without allocating accelerator storage. */
::tyloi::Tensor concatenate_session_tensors(std::span<const ::tyloi::Tensor> tensors, std::int64_t dimension);

/** Encode a bounded range of complete blocks from a text session. */
std::vector<SessionBlockPayload> encode_paged_session(
    const TextSessionState& state,
    size_t block_size,
    size_t first_block,
    size_t maximum_blocks = std::numeric_limits<size_t>::max());

/** Encode one complete text session block. */
SessionBlockPayload encode_paged_block(
    const TextSessionState& state,
    size_t block_size,
    size_t block_index);

/** Decode and validate one paged session payload. */
DecodedSessionBlock decode_paged_block(
    const std::vector<uint8_t>& payload);

/** Reconstruct a text session from a validated sequence of paged payloads. */
TextSessionState decode_paged_session(
    const std::vector<SessionBlockPayload>& payloads,
    const std::vector<int64_t>& tokens,
    size_t block_size);

/** Return the logical bytes held by a defined session tensor. */
size_t session_tensor_bytes(const ::tyloi::Tensor& tensor);

/** Compare tensor shape, dtype and device for backing reuse. */
bool session_tensor_layout_matches(
    const ::tyloi::Tensor& target,
    const ::tyloi::Tensor& source);

/** Return the new backing bytes needed to restore a complete tensor. */
std::size_t restore_tensor_bytes(const ::tyloi::Tensor& target, const ::tyloi::Tensor& source);
/** Return the new backing bytes needed to restore a prefix at the given capacity. */
std::size_t restore_prefix_bytes(const ::tyloi::Tensor& target, const ::tyloi::Tensor& source,
    std::int64_t dimension, std::int64_t capacity);

/** Add the device backing and any host staging needed by a fresh KV restore. */
void add_kv_restore_growth(std::vector<std::pair<::tyloi::Device, std::size_t>>& growth,
                           const KvSnapshot& state);

/** Restore a tensor into compatible backing or newly allocated storage. */
::tyloi::Tensor restore_session_tensor(
    ::tyloi::Tensor target,
    const ::tyloi::Tensor& source);

/** Restore a prefix into compatible or newly allocated capacity. */
::tyloi::Tensor restore_session_prefix_tensor(
    ::tyloi::Tensor target,
    const ::tyloi::Tensor& source,
    int64_t dimension,
    int64_t capacity);

/** Capture the committed KV prefix and add its logical bytes to the total. */
KvSnapshot capture_kv(
    const tyloi::state::KvView& cache,
    int64_t cache_pos,
    size_t& bytes);

/** Reject a KV snapshot whose layout cannot be restored. */
void validate_kv(const KvSnapshot& state);

/** Restore validated KV backing and cache metadata. */
void restore_kv(
    tyloi::state::KvView& cache,
    const KvSnapshot& state,
    int64_t head_dim,
    ::tyloi::ScalarType dtype,
    tyloi::state::KvQuantization quantization);

/** Stage KV into independent targets retained before any asynchronous copy. */
void stage_kv(tyloi::state::KvView& cache, const KvSnapshot& state,
              int64_t head_dim, ::tyloi::ScalarType dtype,
              tyloi::state::KvQuantization quantization);

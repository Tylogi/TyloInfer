#include "mlx_moe.h"
#include "mlx_moe_cache_internal.h"
#include "mlx_moe_ops.h"

#include "mlx_kernel_prepare.h"
#include "engine/backends/metal/native/kernel.h"
#include "engine/backends/metal/kernels/source.h"
#include "engine/backends/metal/native/allocation.h"
#include "mlx_nvq_decode.h"
#include "mfq_mfe_prefill_embedded.h"
#include "mlx_nint.h"
#include "mlx_nint_metadata.h"
#include "mlx_nint8_zero.h"
#include "mlx_mx.h"
#include "mlx_fp8_sq.h"
#include "engine/backends/metal/native/profile.h"
#include "mlx_mxfp4_sq.h"
#include "engine/backends/metal/native/device.h"
#include "engine/backends/metal/native/env.h"
#include "engine/backends/metal/native/memory.h"
#include "engine/backends/metal/native/tensor/tensor.h"
#include "mlx_vq.h"

#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <mlx/memory.h>
#include <mlx/primitives.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dispatch/dispatch.h>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

namespace tyloi::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;
using mfe_internal::kDescriptorSize;
using mfe_internal::kMfeSmallMReadWorkers;
using mfe_internal::kMfeDirectRangeMaxExperts;
using mfe_internal::kFamilyNint;
using mfe_internal::kFamilyMxfp4;
using mfe_internal::kFamilyMxfp8;
using mfe_internal::kFamilyVq;
using mfe_internal::kVqProfileGeneric;
using mfe_internal::kVqProfileJsc4;
using mfe_internal::kVqProfileNpqS;
using mfe_internal::kVqProfileNvq1L;
using mfe_internal::kVqProfileJsc8;
using mfe_internal::kVqProfileNpqL;
using mfe_internal::kVqProfileNvq1S;
using mfe_internal::kVqProfileJscExtended8;
using mfe_internal::kVqProfileJscExtended4;
using mfe_internal::MfeDirectReadTarget;
using mfe_internal::direct_read_targets;
using mfe_internal::pooled_expert_staging;


std::atomic<std::size_t> uncached_packed_storage_bytes{0};
constexpr int kMfeFoldedGateSimdPairs = 4;

constexpr int kFamily = 0;
constexpr int kLocalExpert = 1;
constexpr int kOut = 2;
constexpr int kInput = 3;

constexpr int kNintBits = 4;
constexpr int kNintGroupSize = 5;
constexpr int kNintGroups = 6;
constexpr int kNintQOffset = 7;
constexpr int kNintSubOffset = 8;
constexpr int kNintAnchorOffset = 9;
constexpr int kNintExecution = 10;
constexpr int kNintRowMetadataOffset = 11;
constexpr int kNintQDispatch = 12;
constexpr int kNintV2 = 13;

// Keep the same family value and fields as the Python Metal descriptor.  The
// unused value 1 remains available for the VQ-family extension.
constexpr int kFamilyNint8Zero = 2;
constexpr int kQ8Groups = 4;
constexpr int kQ8QOffset = 5;
constexpr int kQ8ScaleOffset = 6;

constexpr int kFamilyBf16 = 5;
constexpr int kFamilyF16 = 6;
constexpr int kMxGroups = 4;
constexpr int kMxValueOffset = 5;
constexpr int kMxScaleOffset = 6;
constexpr int kDenseValueOffset = 4;

bool apple_m5_family() noexcept {
    return !mlx_metal_nax_disabled() && mlx_apple_chip_starts_with("Apple M5");
}

bool apple_m3_ultra() noexcept {
    return mlx_apple_chip_is("Apple M3 Ultra");
}

bool mixed_grouped_nax_enabled(int route_count) noexcept {
    if (mlx_metal_nax_disabled()) return false;
    // The heterogeneous NAX path is a large-M prefill kernel.  Keep small-M
    // and short-tail shapes on the compatibility kernel, whose boundary
    // handling is both cheaper and already exhaustive.
    constexpr int kMinRoutes = 512;
    if (route_count < kMinRoutes) {
        return false;
    }
    const char* value = std::getenv(
        "TYLOINFER_METAL_MIXED_PREFILL_NAX");
    if (value == nullptr) {
        return apple_m5_family();
    }
    return std::string_view(value) != "0"
        && std::string_view(value) != "false"
        && std::string_view(value) != "off";
}

int grouped_mmq_tile_columns(
    int block_rows,
    int output_width,
    bool fused_swiglu) noexcept {
    const char* value = std::getenv(
        "TYLOINFER_METAL_GROUPED_MMQ_TILE_COLUMNS");
    if (value != nullptr) {
        if (block_rows == 32 && std::string_view(value) == "128") {
            return 128;
        }
        if (std::string_view(value) == "96") {
            if (block_rows == 48 || block_rows == 64) {
                return 96;
            }
        }
        if (std::string_view(value) == "64") {
            return 64;
        }
    }
    if (block_rows == 128 && !fused_swiglu) {
        return 32;
    }
    // Use the largest column tile that remains within the 32-KiB
    // threadgroup-memory budget for this row tile. Wider column tiles reduce
    // repeated activation staging; retain BN64 for genuinely narrow outputs.
    if (block_rows == 32 && output_width >= 512) {
        return 128;
    }
    if (block_rows == 48 && output_width >= 512) {
        return 96;
    }
    return block_rows == 64 && output_width >= 1024 ? 96 : 64;
}

bool mxfp4_nax_prefill_enabled(
    int route_count,
    int experts,
    int input_width,
    int output_width,
    bool allow_automatic) noexcept {
    constexpr int kDefaultMinRoutes = 1024;
    const bool native_256_expert_geometry = experts == 256 && (
        (input_width == 4096 &&
         (output_width == 2048 || output_width == 4096)) ||
        (input_width == 2048 && output_width == 4096));
    const bool native_384_expert_geometry = experts == 384 && (
        (input_width == 5120 &&
         (output_width == 2304 || output_width == 4608)) ||
        (input_width == 2304 && output_width == 5120));
    const bool native_optimized_geometry =
        native_256_expert_geometry || native_384_expert_geometry;
    // M5 uses MFQ's unified grouped-NAX path below.  MLX gather_qmm is a
    // win only for the measured M3 Ultra native expert geometries; selecting
    // it ahead of grouped-NAX on M5 more than doubles routed-projection time
    // at the 256-expert, 4096<->2048, top-6 production geometry.
    const bool automatic = route_count >= kDefaultMinRoutes
        && apple_m3_ultra() && native_optimized_geometry;
    const char* value = std::getenv(
        "TYLOINFER_METAL_MFE_PREFILL_NAX");
    if (value == nullptr) {
        return allow_automatic && automatic;
    }
    const auto setting = std::string_view(value);
    if (
        setting == "1"
        || setting == "true"
        || setting == "on"
    ) {
        return true;
    }
    return setting == "auto" && allow_automatic && automatic;
}

bool mxfp4_nax_smallm_preferred(
    const array& expert_ids,
    int tokens,
    int addressable_experts,
    int logical_experts,
    int input_width,
    int output_width) noexcept {
    // Historical naming: this policy selects MLX gather_qmm, not an
    // unconditional hardware-NAX kernel. MLX dispatches the ordinary
    // fp_gather_qmm kernels on pre-NAX devices (including M3 Ultra) and may
    // select its NAX implementation only on hardware that supports it.
    const char* value = std::getenv(
        "TYLOINFER_METAL_MFE_SMALLM_NAX");
    const auto setting = value == nullptr
        ? std::string_view("auto")
        : std::string_view(value);
    const bool force =
        setting == "1"
        || setting == "true"
        || setting == "on";
    const bool m3_ultra_native_geometry =
        apple_m3_ultra() && logical_experts == 256 && (
            (input_width == 4096 &&
             (output_width == 2048 || output_width == 4096)) ||
            (input_width == 2048 && output_width == 4096));
    if (
        tokens < 4
        || tokens > 6
        || (!force && setting != "auto" && setting != "adaptive")
        || (!force && !apple_m5_family() && !m3_ultra_native_geometry)
        || expert_ids.dtype() != mlx::core::int32
        || !expert_ids.flags().row_contiguous
        || !expert_ids.is_available()
    ) {
        return false;
    }
    const auto* ids = expert_ids.data<std::int32_t>();
    bool has_duplicate = false;
    for (std::size_t index = 0; index < expert_ids.size(); ++index) {
        if (ids[index] < 0 || ids[index] >= addressable_experts) {
            return false;
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (ids[index] == ids[previous]) {
                has_duplicate = true;
                break;
            }
        }
    }
    return has_duplicate;
}

bool mxfp4_decode_down_reduce_enabled() noexcept {
    const char* value = std::getenv(
        "TYLOINFER_METAL_MFE_DECODE_DOWN_REDUCE");
    if (value == nullptr) {
        return true;
    }
    const auto setting = std::string_view(value);
    return setting != "0"
        && setting != "false"
        && setting != "off";
}

int mxfp4_decode_down_reduce_rows() noexcept {
    const char* value = std::getenv(
        "TYLOINFER_METAL_MFE_DECODE_DOWN_REDUCE_ROWS");
    if (value != nullptr) {
        const auto setting = std::string_view(value);
        if (setting == "1") return 1;
        if (setting == "4") return 4;
    }
    return 2;
}

int mfe_decode_rows_per_simd(int automatic_rows) noexcept {
    const char* value = std::getenv(
        "TYLOINFER_METAL_MFE_DECODE_ROWS_PER_SIMD");
    if (value != nullptr) {
        const auto setting = std::string_view(value);
        if (setting == "1") return 1;
        if (setting == "2") return 2;
        if (setting == "4") return 4;
    }
    return automatic_rows;
}

bool mfe_narrow_bindings_enabled() noexcept {
    const char* value = std::getenv(
        "TYLOINFER_METAL_MFE_NARROW_BINDINGS");
    if (value == nullptr) {
        return true;
    }
    const auto setting = std::string_view(value);
    return setting != "0"
        && setting != "false"
        && setting != "off";
}

bool mfe_split_swiglu_enabled() noexcept {
    const char* value = std::getenv(
        "TYLOINFER_METAL_MFE_SPLIT_SWIGLU");
    if (value == nullptr) {
        return true;
    }
    const auto setting = std::string_view(value);
    return setting != "0"
        && setting != "false"
        && setting != "off";
}

bool routed_sort_enabled(int tokens) noexcept {
    if (tokens <= 1) {
        return false;
    }
    const char* value = std::getenv(
        "TYLOINFER_METAL_MFE_SORT_ROUTES");
    if (value == nullptr) {
        return tokens > 6;
    }
    const auto setting = std::string_view(value);
    return !(
        setting == "0"
        || setting == "false"
        || setting == "off");
}

constexpr int kVqGroupSize = 4;
constexpr int kVqGroups = 5;
constexpr int kVqVectorSize = 6;
constexpr int kVqVectors = 7;
constexpr int kVqIndexBits = 8;
constexpr int kVqStateBits = 9;
constexpr int kVqStates = 10;
constexpr int kVqEntries = 11;
constexpr int kVqCodeBanks = 12;
constexpr int kVqAuxMode = 13;
constexpr int kVqCodeBankMode = 14;
constexpr int kVqHasTableBanks = 15;
constexpr int kVqGroupsPerSuper = 16;
constexpr int kVqSupergroups = 17;
constexpr int kVqIndicesOffset = 18;
constexpr int kVqStateOffset = 19;
constexpr int kVqAuxOffset = 20;
constexpr int kVqAnchorOffset = 21;
constexpr int kVqCodebookOffset = 22;
constexpr int kVqScaleOffset = 23;
constexpr int kVqStateBankOffset = 24;
constexpr int kVqBankOffset = 25;
constexpr int kVqParameterOffset = 26;
constexpr int kVqRotationVariant = 27;
constexpr int kVqProfile = 28;
constexpr int kVqExecutionLayout = 29;
constexpr int kVqResidualCodebookOffset = 30;
constexpr int kVqResidualRecordOffset = 31;
constexpr std::uint32_t kGroupedVqVectorProfileMask =
    (std::uint32_t{1} << kVqProfileGeneric)
    | (std::uint32_t{1} << kVqProfileNpqS)
    | (std::uint32_t{1} << kVqProfileNvq1L)
    | (std::uint32_t{1} << kVqProfileNpqL)
    | (std::uint32_t{1} << kVqProfileNvq1S)
    | (std::uint32_t{1} << kVqProfileJscExtended4);
constexpr std::uint32_t kGroupedJscExtendedProfileMask =
    (std::uint32_t{1} << kVqProfileJscExtended8)
    | (std::uint32_t{1} << kVqProfileJscExtended4);

const std::string kMoeHeader = std::string(R"METAL(
#ifndef TYLOINFER_MOE_ROUTE_GROUPS
#define TYLOINFER_MOE_ROUTE_GROUPS 0
#endif
#ifndef TYLOINFER_MOE_VQ_LAYOUTS
#define TYLOINFER_MOE_VQ_LAYOUTS 0xfffffffffffffffful
#endif
template <uint K, typename T>
struct MfqMoeVqInputs {
    device const T* values;
    thread const uint* offsets;
    T operator[](uint index) const {
        return values[offsets[index / K] + index % K];
    }
};

template <uint ROUTED_WIDTH, uint TOKEN_STRIDE, typename T>
struct MfqMoeVqOutput {
    device T* values;
    device T& operator[](uint index) const {
        return values[(index / ROUTED_WIDTH) * TOKEN_STRIDE + index % ROUTED_WIDTH];
    }
};

template <uint K, typename T>
inline device const T* mfq_nvq_input_row(MfqMoeVqInputs<K, T> x, uint row) {
    return x.values + x.offsets[row];
}
)METAL") + detail::kNintMetadataSource + kNvqDecodeHeader + R"METAL(
#ifndef TYLOINFER_NINT_Q_DISPATCH
#define TYLOINFER_NINT_Q_DISPATCH 0
#endif

#define TYLOINFER_MFE_NINT_PROFILE(GS, EXPANSION, TAIL, Q) \
    mfq_moe_nint_profile<GS, MATRIX_ROWS, uint(K), K_LANES, EXPANSION, TAIL, 1u, Q>( \
        x, nint_q, nint_sub_scale, nint_sub_min, x_offset, q_offset, sub_offset, \
        metadata_rows, q_widths, q_row_byte_offsets, q_row_bit_shifts, neuron_scales, \
        neuron_minimums, accumulators, groups, k_lane)
#define TYLOINFER_MFE_NINT_CALL_Q(GS, Q) \
    if constexpr (OUT > K) { TYLOINFER_MFE_NINT_PROFILE(GS, true, true, Q); } \
    else { \
        TYLOINFER_MFE_NINT_PROFILE(GS, false, false, Q); \
        if constexpr (uint(K) % GS != 0u) { TYLOINFER_MFE_NINT_PROFILE(GS, false, true, Q); } \
    }
#define TYLOINFER_MFE_NINT_CALL_WITH(GS, ENABLED) \
    if constexpr (ENABLED && MATRIX_ROWS == 1u) { \
        switch (metadata_rows[0].q_bits) { \
            case 4u: { TYLOINFER_MFE_NINT_CALL_Q(GS, 4u); break; } \
            case 5u: { TYLOINFER_MFE_NINT_CALL_Q(GS, 5u); break; } \
            case 6u: { TYLOINFER_MFE_NINT_CALL_Q(GS, 6u); break; } \
            case 8u: { TYLOINFER_MFE_NINT_CALL_Q(GS, 8u); break; } \
            default: { TYLOINFER_MFE_NINT_CALL_Q(GS, 0u); break; } \
        } \
    } else { TYLOINFER_MFE_NINT_CALL_Q(GS, 0u); }
#define TYLOINFER_MFE_NINT_CALL(GS) TYLOINFER_MFE_NINT_CALL_WITH(GS, TYLOINFER_NINT_Q_DISPATCH)
#ifndef TYLOINFER_MFE_NINT_DISPATCH_WITH
#define TYLOINFER_MFE_NINT_DISPATCH_WITH(CALL) \
    if (group_size == 24u) { CALL(24u); } \
    else if (group_size == 28u) { CALL(28u); } else
#endif
#define TYLOINFER_MFE_NINT_DISPATCH TYLOINFER_MFE_NINT_DISPATCH_WITH(TYLOINFER_MFE_NINT_CALL)

template <typename Stream>
inline uint mfq_moe_read_bits(
    Stream stream,
    uint value_index,
    uint bits
) {
    uint residual_bits = (value_index & 7u) * bits;
    uint byte_index =
        (value_index >> 3) * bits + (residual_bits >> 3);
    uint shift = residual_bits & 7u;
    uint packed = uint(stream[byte_index]);
    if (shift + bits > 8u) {
        packed |= uint(stream[byte_index + 1u]) << 8;
    }
    if (shift + bits > 16u) {
        packed |= uint(stream[byte_index + 2u]) << 16;
    }
    return (packed >> shift)
        & ((1u << bits) - 1u);
}

template <bool ALIGNED_GROUP = true, bool WORD_QUAD = false>
inline ushort4 mfq_moe_decode_nint_quad_at(
    device const uchar* stream,
    uint byte_index,
    uint shift,
    uint bits
) {
    uint required_bits = shift + 4u * bits;
    packed_uchar4 bytes =
        *reinterpret_cast<device const packed_uchar4*>(stream + byte_index);
    uint packed = as_type<uint>(bytes);
    if constexpr (WORD_QUAD) {
        if constexpr (ALIGNED_GROUP) {
            if (shift != 0u) {
                packed >>= shift;
            }
        } else {
            packed >>= shift;
        }
    } else {
        if (shift != 0u) {
            packed = (packed >> shift)
                | (required_bits > 32u
                    ? uint(stream[byte_index + 4u]) << (32u - shift)
                    : 0u);
        }
    }
    uint mask = (1u << bits) - 1u;
    return ushort4(
        packed & mask,
        (packed >> bits) & mask,
        (packed >> (2u * bits)) & mask,
        (packed >> (3u * bits)) & mask);
}

inline ushort4 mfq_moe_read_nint_row_quad(
    device const uchar* stream,
    uint row_byte_offset,
    uint row_bit_shift,
    uint value_index,
    uint bits
) {
    uint row_relative_bits = row_bit_shift + value_index * bits;
    return mfq_moe_decode_nint_quad_at(
        stream,
        row_byte_offset + (row_relative_bits >> 3u),
        row_relative_bits & 7u,
        bits);
}

struct MfqMoeNintOctet {
    ushort4 low;
    ushort4 high;
};

template <bool ALIGNED_EIGHT>
inline MfqMoeNintOctet mfq_moe_decode_nint_octet_at(
    device const uchar* stream, uint byte_index, uint shift, uint bits
) {
    if constexpr (ALIGNED_EIGHT) shift = 0u;
    const uint word0 = as_type<uint>(
        *reinterpret_cast<device const packed_uchar4*>(stream + byte_index));
    const uint word1 = bits > 4u ? as_type<uint>(
        *reinterpret_cast<device const packed_uchar4*>(stream + byte_index + 4u)) : 0u;
    const uint first = shift == 0u ? word0 : (word0 >> 4u) | (word1 << 28u);
    const uint bit = 4u * bits + shift;
    const uint second = bit == 32u ? word1
        : (word0 >> bit) | (word1 << (32u - bit));
    const uint mask = (1u << bits) - 1u;
    return {
        ushort4(first & mask, (first >> bits) & mask,
            (first >> (2u * bits)) & mask, (first >> (3u * bits)) & mask),
        ushort4(second & mask, (second >> bits) & mask,
            (second >> (2u * bits)) & mask, (second >> (3u * bits)) & mask)
    };
}

template <
    uint GROUP_SIZE,
    uint MATRIX_ROWS,
    uint K,
    uint K_LANES,
    bool EXPANSION,
    bool TAIL,
    uint INPUT_ROWS = 1u,
    uint FIXED_Q = 0u,
    typename XStream,
    typename SubScaleStream,
    typename SubMinStream
>
inline void mfq_moe_nint_profile(
    XStream x,
    device const uchar* q_stream,
    SubScaleStream sub_scale_stream,
    SubMinStream sub_min_stream,
    uint x_offset,
    uint q_offset,
    uint sub_offset,
    thread const MfqNintRow* metadata_rows,
    thread const uint* q_widths,
    thread const uint* q_row_byte_offsets,
    thread const uint* q_row_bit_shifts,
    thread const float* neuron_scales,
    thread const float* neuron_minimums,
    thread float* accumulators,
    uint groups,
    uint k_lane,
    thread const uint* input_offsets = nullptr
) {
    constexpr uint OCTETS = (!EXPANSION || GROUP_SIZE == 24u || GROUP_SIZE == 28u)
        && GROUP_SIZE % 4u == 0u ? GROUP_SIZE / 8u : 0u;
    const uint first_group = !EXPANSION && TAIL
        ? K / GROUP_SIZE + (k_lane + K_LANES - (K / GROUP_SIZE) % K_LANES) % K_LANES
        : k_lane;
    constexpr uint END_GROUP = TAIL ? (K + GROUP_SIZE - 1u) / GROUP_SIZE : K / GROUP_SIZE;
    static_assert(K_LANES % 8u == 0u);
    uint4 sub_cursors[MATRIX_ROWS];
    uint3 q_group_cursors[MATRIX_ROWS];
    for (uint row = 0u; row < MATRIX_ROWS; ++row) {
        const MfqNintRow metadata = metadata_rows[row];
        const uint bit = metadata.sub_shift + (first_group & 7u) * metadata.sub_bits;
        sub_cursors[row] = uint4(
            sub_offset + metadata.sub_offset
                + (first_group >> 3u) * metadata.sub_bits + (bit >> 3u),
            bit & 7u,
            (1u << metadata.sub_bits) - 1u,
            (K_LANES / 8u) * metadata.sub_bits);
        const uint column = first_group * GROUP_SIZE;
        const uint bits = FIXED_Q != 0u ? FIXED_Q : q_widths[row];
        const uint q_bit = q_row_bit_shifts[row] + (column & 7u) * bits;
        q_group_cursors[row] = uint3(
            q_row_byte_offsets[row] + (column >> 3u) * bits + (q_bit >> 3u),
            q_bit & 7u,
            GROUP_SIZE * (K_LANES / 8u) * bits);
    }
    for (
        uint group = first_group;
        group < END_GROUP;
        group += K_LANES
    ) {
        float scales[MATRIX_ROWS];
        float minimums[MATRIX_ROWS];
        uint q_byte_cursors[MATRIX_ROWS];
        uint q_bit_cursors[MATRIX_ROWS];
        uint column_base = group * GROUP_SIZE;
        for (uint row = 0u; row < MATRIX_ROWS; ++row) {
            const uint4 sub = sub_cursors[row];
            const uint scale = mfq_nint_sub_word(sub_scale_stream, sub.x);
            const uint minimum = mfq_nint_sub_word(sub_min_stream, sub.x);
            scales[row] = neuron_scales[row]
                * float((scale >> sub.y) & sub.z);
            minimums[row] = neuron_minimums[row]
                * float((minimum >> sub.y) & sub.z);
            sub_cursors[row].x += sub.w;
            q_byte_cursors[row] = q_group_cursors[row].x;
            q_bit_cursors[row] = q_group_cursors[row].y;
            q_group_cursors[row].x += q_group_cursors[row].z;
        }
        float activation_sum[INPUT_ROWS] = {0.0f};
        float quantized_dots[MATRIX_ROWS * INPUT_ROWS] = {0.0f};
        for (uint block = 0u; block < OCTETS; ++block) {
            const uint column = column_base + block * 8u;
            if (TAIL && column >= K) break;
            float4 activation0[INPUT_ROWS];
            float4 activation1[INPUT_ROWS];
            for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
            const uint offset = INPUT_ROWS == 1u ? x_offset : input_offsets[input_row];
            activation0[input_row] = float4(0.0f);
            activation1[input_row] = float4(0.0f);
            if (!TAIL || column + 7u < K) {
                activation0[input_row] = float4(x[offset + column], x[offset + column + 1u],
                    x[offset + column + 2u], x[offset + column + 3u]);
                activation1[input_row] = float4(x[offset + column + 4u], x[offset + column + 5u],
                    x[offset + column + 6u], x[offset + column + 7u]);
            } else {
                for (uint item = 0u; item < 4u; ++item) {
                    if (column + item < K) activation0[input_row][item] = float(x[offset + column + item]);
                    if (column + 4u + item < K) activation1[input_row][item] = float(x[offset + column + 4u + item]);
                }
            }
            activation_sum[input_row] += activation0[input_row].x + activation0[input_row].y
                + activation0[input_row].z + activation0[input_row].w
                + activation1[input_row].x + activation1[input_row].y
                + activation1[input_row].z + activation1[input_row].w;
            }
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                const auto quantized = mfq_moe_decode_nint_octet_at<GROUP_SIZE % 8u == 0u>(
                    q_stream + q_offset, q_byte_cursors[row], q_bit_cursors[row], FIXED_Q != 0u ? FIXED_Q : q_widths[row]);
                for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row)
                    quantized_dots[row * INPUT_ROWS + input_row] +=
                        dot(activation0[input_row], float4(quantized.low))
                        + dot(activation1[input_row], float4(quantized.high));
                q_byte_cursors[row] += FIXED_Q != 0u ? FIXED_Q : q_widths[row];
            }
        }
        for (
            uint element = OCTETS * 8u;
            element < GROUP_SIZE;
            element += 4u
        ) {
            uint column = column_base + element;
            if (TAIL && column >= K) {
                break;
            }
            float4 activations[INPUT_ROWS];
            for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
            const uint offset = INPUT_ROWS == 1u ? x_offset : input_offsets[input_row];
            if (element + 3u < GROUP_SIZE && (!TAIL || column + 3u < K)) {
                activations[input_row] = float4(
                    x[offset + column],
                    x[offset + column + 1u],
                    x[offset + column + 2u],
                    x[offset + column + 3u]);
            } else {
                activations[input_row] = float4(
                    !TAIL || column < K
                        ? float(x[offset + column]) : 0.0f,
                    element + 1u < GROUP_SIZE && (!TAIL || column + 1u < K)
                        ? float(x[offset + column + 1u]) : 0.0f,
                    element + 2u < GROUP_SIZE && (!TAIL || column + 2u < K)
                        ? float(x[offset + column + 2u]) : 0.0f,
                    element + 3u < GROUP_SIZE && column + 3u < K
                        ? float(x[offset + column + 3u]) : 0.0f);
            }
            activation_sum[input_row] += activations[input_row].x + activations[input_row].y
                + activations[input_row].z + activations[input_row].w;
            }
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                ushort4 quantized = mfq_moe_decode_nint_quad_at<
                    GROUP_SIZE % 8u == 0u, GROUP_SIZE % 4u == 0u>(
                    q_stream + q_offset,
                    q_byte_cursors[row],
                    q_bit_cursors[row],
                    FIXED_Q != 0u ? FIXED_Q : q_widths[row]);
                for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row)
                    quantized_dots[row * INPUT_ROWS + input_row] +=
                        dot(activations[input_row], float4(quantized));
                uint next_bit = q_bit_cursors[row]
                    + 4u * (FIXED_Q != 0u ? FIXED_Q : q_widths[row]);
                q_byte_cursors[row] += next_bit >> 3u;
                q_bit_cursors[row] = next_bit & 7u;
            }
        }
        for (uint row = 0u; row < MATRIX_ROWS; ++row) {
            for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
            const uint index = row * INPUT_ROWS + input_row;
            accumulators[index] = fma(
                scales[row],
                quantized_dots[index],
                fma(
                    -minimums[row],
                    activation_sum[input_row],
                    accumulators[index]));
            }
        }
    }
}

#define TYLOINFER_MFE_NINT_PAIR_PROFILE(GS, EXPANSION, TAIL, Q) \
    mfq_moe_nint_profile<GS, MATRIX_ROWS, K, K_LANES, EXPANSION, TAIL, 2u, Q>( \
        x, q_stream, sub_scale_stream, sub_min_stream, 0u, uint(d[7]), uint(d[8]), \
        metadata_rows, q_widths, q_row_byte_offsets, q_row_bit_shifts, neuron_scales, \
        neuron_minimums, values, uint(d[6]), k_lane, input_offsets)
#define TYLOINFER_MFE_NINT_PAIR_CALL_Q(GS, Q) \
    if constexpr (OUT > K) { TYLOINFER_MFE_NINT_PAIR_PROFILE(GS, true, true, Q); } \
    else { \
        TYLOINFER_MFE_NINT_PAIR_PROFILE(GS, false, false, Q); \
        if constexpr (K % GS != 0u) { TYLOINFER_MFE_NINT_PAIR_PROFILE(GS, false, true, Q); } \
    }
#define TYLOINFER_MFE_NINT_PAIR_CALL(GS) \
    if (!leader) return true; \
    if constexpr (TYLOINFER_NINT_Q_DISPATCH && MATRIX_ROWS == 1u) { \
        switch (metadata_rows[0].q_bits) { \
            case 4u: { TYLOINFER_MFE_NINT_PAIR_CALL_Q(GS, 4u); break; } \
            case 5u: { TYLOINFER_MFE_NINT_PAIR_CALL_Q(GS, 5u); break; } \
            case 6u: { TYLOINFER_MFE_NINT_PAIR_CALL_Q(GS, 6u); break; } \
            case 8u: { TYLOINFER_MFE_NINT_PAIR_CALL_Q(GS, 8u); break; } \
            default: { TYLOINFER_MFE_NINT_PAIR_CALL_Q(GS, 0u); break; } \
        } \
    } else { TYLOINFER_MFE_NINT_PAIR_CALL_Q(GS, 0u); }
#define TYLOINFER_MFE_NINT_SUPPORTED(GS) return true
inline bool mfq_moe_nint_supported_gs(uint group_size) {
    TYLOINFER_MFE_NINT_DISPATCH_WITH(TYLOINFER_MFE_NINT_SUPPORTED) { return false; }
}

template <uint MATRIX_ROWS, uint ROWS_PER_SIMD, uint K, uint K_LANES,
    uint OUT, uint MATRIX_OUT, uint PROJECTIONS, uint ROUTES,
    bool SHARED_INPUT, bool SPLIT_SWIGLU, bool PACKED_SWIGLU,
    typename RouteStream, typename YStream, typename T>
inline bool mfq_moe_nint_pair(
    device const T* x, YStream y, device const int* d,
    device const uchar* q_stream, device const uchar* sub_scale_stream,
    device const uchar* sub_min_stream, RouteStream route_rows,
    threadgroup T* split_values, uint output_base, uint lane_group,
    uint k_lane, uint projection, uint variant_stride, float swiglu_limit,
    bool leader
) {
    uint input_offsets[2];
    for (uint row = 0u; row < 2u; ++row)
        input_offsets[row] = (uint(d[27]) * variant_stride
            + (SHARED_INPUT ? uint(route_rows[row]) / ROUTES : uint(route_rows[row]))) * K;
    device const uint* row_metadata = reinterpret_cast<device const uint*>(q_stream + uint(d[11]));
    MfqNintRow metadata_rows[MATRIX_ROWS];
    uint q_widths[MATRIX_ROWS];
    uint q_row_byte_offsets[MATRIX_ROWS];
    uint q_row_bit_shifts[MATRIX_ROWS];
    float neuron_scales[MATRIX_ROWS];
    float neuron_minimums[MATRIX_ROWS];
    for (uint row = 0u; row < MATRIX_ROWS; ++row) {
        const uint output = min(output_base + (PACKED_SWIGLU
            ? (row / ROWS_PER_SIMD) * OUT + row % ROWS_PER_SIMD : row), MATRIX_OUT - 1u);
        const MfqNintRow metadata = mfq_nint_row(row_metadata, uint(d[1]) * MATRIX_OUT + output);
        metadata_rows[row] = metadata;
        q_widths[row] = metadata.q_bits;
        q_row_byte_offsets[row] = metadata.q_offset;
        q_row_bit_shifts[row] = metadata.q_shift;
        neuron_scales[row] = metadata.scale;
        neuron_minimums[row] = metadata.minimum;
    }
    float values[MATRIX_ROWS * 2u] = {0.0f};
    const uint group_size = uint(d[5]);
    TYLOINFER_MFE_NINT_DISPATCH_WITH(TYLOINFER_MFE_NINT_PAIR_CALL) { return false; }
    constexpr uint PHYSICAL_ROWS = (32u / K_LANES) * ROWS_PER_SIMD;
    for (uint row = 0u; row < MATRIX_ROWS; ++row) {
        for (uint input_row = 0u; input_row < 2u; ++input_row) {
            const uint index = row * 2u + input_row;
            for (uint offset = K_LANES >> 1u; offset > 0u; offset >>= 1u)
                values[index] += simd_shuffle_down(values[index], offset);
            if constexpr (SPLIT_SWIGLU) {
                if (k_lane == 0u)
                    split_values[(input_row * 2u + projection) * PHYSICAL_ROWS
                        + lane_group * ROWS_PER_SIMD + row] = T(values[index]);
            }
        }
    }
    if constexpr (SPLIT_SWIGLU) threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
        const uint output = output_base + row;
        for (uint input_row = 0u; input_row < 2u; ++input_row) {
            if (k_lane != 0u || output >= OUT || (SPLIT_SWIGLU && projection != 0u)) continue;
            float value = values[row * 2u + input_row];
            if constexpr (SPLIT_SWIGLU || PACKED_SWIGLU) {
                const uint local = lane_group * ROWS_PER_SIMD + row;
                float gate = SPLIT_SWIGLU
                    ? float(split_values[input_row * 2u * PHYSICAL_ROWS + local]) : float(T(value));
                float up = SPLIT_SWIGLU
                    ? float(split_values[(input_row * 2u + 1u) * PHYSICAL_ROWS + local])
                    : float(T(values[(row + ROWS_PER_SIMD) * 2u + input_row]));
                if (swiglu_limit > 0.0f) {
                    gate = min(gate, swiglu_limit);
                    up = clamp(up, -swiglu_limit, swiglu_limit);
                }
                value = gate / (1.0f + exp(-gate)) * up;
            }
            const uint destination = SPLIT_SWIGLU || PACKED_SWIGLU
                ? uint(route_rows[input_row]) * OUT + output
                : (uint(route_rows[input_row]) * PROJECTIONS + projection) * OUT + output;
            y[destination] = T(value);
        }
    }
    return true;
}

template <uint BITS, typename Stream>
inline uint3 mfq_moe_read_npq_group_indices(
    Stream stream,
    uint row,
    uint group,
    uint vectors
) {
    const uint first = group * 3u;
    if (first + 2u >= vectors) {
        return uint3(
            first < vectors
                ? mfq_moe_read_bits(
                      stream,
                      row * vectors + first,
                      BITS)
                : 0u,
            first + 1u < vectors
                ? mfq_moe_read_bits(
                      stream,
                      row * vectors + first + 1u,
                      BITS)
                : 0u,
            0u);
    }
    const uint linear = row * vectors + first;
    const uint bit = linear * BITS;
    const uint byte = bit >> 3;
    const uint shift = bit & 7u;
    uint packed = uint(stream[byte])
        | (uint(stream[byte + 1u]) << 8)
        | (uint(stream[byte + 2u]) << 16);
    if (shift + 3u * BITS > 24u) {
        packed |= uint(stream[byte + 3u]) << 24;
    }
    const uint mask = (1u << BITS) - 1u;
    return uint3(
        (packed >> shift) & mask,
        (packed >> (shift + BITS)) & mask,
        (packed >> (shift + 2u * BITS)) & mask);
}

constant constexpr float mfq_moe_mxfp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

inline float mfq_moe_mxfp4_value(uchar code) {
    return mfq_moe_mxfp4_lut[uint(code & 15u)];
}

inline float mfq_moe_e8m0(uchar raw) {
    if (raw == 255u) {
        return NAN;
    }
    uint bits = raw == 0u ? 0x00400000u : uint(raw) << 23u;
    return as_type<float>(bits);
}

inline float mfq_moe_mxfp8_value(uchar raw) {
    uint magnitude = uint(raw & 0x7fu);
    uint exponent = magnitude >> 3u;
    uint mantissa = magnitude & 7u;
    if (exponent == 15u && mantissa == 7u) {
        return NAN;
    }
    float value = exponent == 0u
        ? float(mantissa) * 0.001953125f
        : as_type<float>((exponent + 120u) << 23u)
            * (1.0f + float(mantissa) * 0.125f);
    return (raw & 0x80u) == 0u ? value : -value;
}

inline float4 mfq_moe_load_code4(
    device const int8_t* stream,
    uint offset
) {
    return float4(
        *(device const char4*)(stream + offset));
}

inline float4 mfq_moe_load_code4(
    constant const int8_t* stream,
    uint offset
) {
    return float4(
        *(constant const char4*)(stream + offset));
}

inline bool mfq_moe_reusable_vq_profile(uint profile) {
    return profile == 1u || profile == 3u || profile == 4u
        || profile == 6u || profile == 7u || profile == 8u;
}

template <uint PROFILE, uint EXCLUDED = 0u>
inline uint mfq_moe_vq_layout(device const int* descriptor) {
    constexpr uint layouts = uint((TYLOINFER_MOE_VQ_LAYOUTS >> (PROFILE * 7u)) & 127u) & ~EXCLUDED;
    if constexpr (layouts != 0u && (layouts & (layouts - 1u)) == 0u) return ctz(layouts);
    else return uint(descriptor[29]);
}

template <uint INPUT_ROWS, uint MATRIX_ROWS, uint ROWS_PER_SIMD,
    uint K, uint K_LANES, uint OUT, uint MATRIX_OUT, uint PROJECTIONS,
    uint ROUTES, uint PROFILE_MASK, bool SHARED_INPUT, bool SPLIT_SWIGLU, bool PACKED_SWIGLU, bool MXFP4,
    typename RouteStream, typename YStream, typename T>
inline void mfq_moe_reused_rows(
    device const T* x, YStream y, device const int* d,
    device const uchar* indices, device const uchar* states,
    device const uchar* aux, device const float* anchors,
    device const int8_t* codebooks, device const float* scales,
    device const uchar* state_banks, device const float* parameters,
    device const uchar* mx_values, device const uchar* mx_scales,
    RouteStream route_rows, threadgroup T* split_values,
    uint output_base, uint lane_group, uint k_lane, uint projection,
    uint variant_stride, float swiglu_limit
) {
    uint input_offsets[INPUT_ROWS];
    for (uint row = 0u; row < INPUT_ROWS; ++row) {
        uint route = route_rows[row];
        input_offsets[row] = (uint(d[27]) * variant_stride
            + (SHARED_INPUT ? route / ROUTES : route)) * K;
    }
    MfqMoeVqInputs<K, T> inputs{x, input_offsets};
    uint outputs[MATRIX_ROWS];
    float row_anchors[MATRIX_ROWS];
    for (uint row = 0u; row < MATRIX_ROWS; ++row) {
        uint output = min(output_base + (PACKED_SWIGLU
            ? (row / ROWS_PER_SIMD) * OUT + row % ROWS_PER_SIMD : row), MATRIX_OUT - 1u);
        outputs[row] = uint(d[1]) * MATRIX_OUT + output;
        row_anchors[row] = MXFP4 ? 0.0f : anchors[uint(d[21]) + outputs[row]];
    }
    float values[MATRIX_ROWS * INPUT_ROWS] = {0.0f};
    uint profile = uint(d[28]) & 255u;
    if constexpr (MXFP4) {
        const uint groups = uint(d[4]);
        for (uint group = k_lane; group < groups; group += K_LANES) {
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                const float scale = mfq_moe_e8m0(mx_scales[ulong(uint(d[6]))
                    + ulong(outputs[row]) * groups + group]);
                for (uint component = 0u; component < 32u; component += 16u) {
                    const uint column = group * 32u + component;
                    const ulong offset = ulong(uint(d[5])) + ulong(outputs[row]) * (K / 2u) + column / 2u;
                    const uint2 packed = *reinterpret_cast<device const uint2*>(mx_values + offset);
                    for (uint pair = 0u; pair < 8u; ++pair) {
                        const uint word = pair < 4u ? packed.x : packed.y;
                        const uchar code = uchar(word >> ((pair & 3u) * 8u));
                        const float first = scale * mfq_moe_mxfp4_value(code & 15u);
                        const float second = scale * mfq_moe_mxfp4_value(code >> 4u);
                        for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
                            const uint xbase = input_offsets[input_row] + column + pair * 2u;
                            const uint index = row * INPUT_ROWS + input_row;
                            values[index] = fma(float(x[xbase]), first, values[index]);
                            values[index] = fma(float(x[xbase + 1u]), second, values[index]);
                        }
                    }
                }
            }
        }
    } else if ((PROFILE_MASK & 8u) != 0u && profile == 3u) {
        mfq_nvq1_profile<MATRIX_ROWS, K, K_LANES, 11u, INPUT_ROWS>(
            inputs, indices, states, aux, scales, codebooks, outputs, row_anchors, values,
            parameters[uint(d[26])], 0u, uint(d[18]), uint(d[19]), uint(d[20]), uint(d[23]), uint(d[22]), k_lane);
    } else if ((PROFILE_MASK & 64u) != 0u && profile == 6u) {
        mfq_nvq1_profile<MATRIX_ROWS, K, K_LANES, 9u, INPUT_ROWS>(
            inputs, indices, states, aux, scales, codebooks, outputs, row_anchors, values,
            parameters[uint(d[26])], 0u, uint(d[18]), uint(d[19]), uint(d[20]), uint(d[23]), uint(d[22]), k_lane);
    } else if ((PROFILE_MASK & 128u) != 0u && profile == 7u && mfq_moe_vq_layout<7u>(d) == 0u) {
        mfq_nvq_banked_profile<MATRIX_ROWS, K, K_LANES, (OUT > K), 8u, INPUT_ROWS>(
            inputs, indices, states, aux, state_banks, scales, codebooks, outputs, row_anchors, values,
            0u, uint(d[18]), uint(d[19]), uint(d[20]), uint(d[24]), uint(d[23]), uint(d[22]), k_lane, uint(d[8]));
    } else if ((PROFILE_MASK & 256u) != 0u && profile == 8u) {
        mfq_nvq_banked_profile<MATRIX_ROWS, K, K_LANES, (OUT > K), 4u, INPUT_ROWS>(
            inputs, indices, states, aux, state_banks, scales, codebooks, outputs, row_anchors, values,
            0u, uint(d[18]), uint(d[19]), uint(d[20]), uint(d[24]), uint(d[23]), uint(d[22]), k_lane, uint(d[8]));
    } else if ((PROFILE_MASK & 2u) != 0u && profile == 1u) {
        mfq_nvq_banked_profile<MATRIX_ROWS, K, K_LANES, (OUT > K), 4u, INPUT_ROWS, 8u>(
            inputs, indices, states, aux, state_banks, scales, codebooks, outputs, row_anchors, values,
            0u, uint(d[18]), uint(d[19]), uint(d[20]), uint(d[24]), uint(d[23]), uint(d[22]),
            k_lane, 8u, uint(d[13]));
    } else if ((PROFILE_MASK & 16u) != 0u && profile == 4u) {
        mfq_nvq_banked_profile<MATRIX_ROWS, K, K_LANES, (OUT > K), 8u, INPUT_ROWS, 8u>(
            inputs, indices, states, aux, state_banks, scales, codebooks, outputs, row_anchors, values,
            0u, uint(d[18]), uint(d[19]), uint(d[20]), uint(d[24]), uint(d[23]), uint(d[22]),
            k_lane, 8u, uint(d[13]));
    } else if ((PROFILE_MASK & 128u) != 0u && profile == 7u) {
        mfq_nvq_jsc_profile<8u, K, MATRIX_ROWS, 1u, false, 0u, INPUT_ROWS>(
            inputs, indices, states, aux, scales, state_banks, codebooks,
            0u, outputs, row_anchors, values, uint(d[8]), uint(d[11]), uint(d[18]),
            uint(d[19]), uint(d[20]), uint(d[22]), uint(d[23]), uint(d[24]),
            k_lane, K_LANES, mfq_moe_vq_layout<7u, 65u>(d));
    }
    constexpr uint PHYSICAL_ROWS = (32u / K_LANES) * ROWS_PER_SIMD;
    for (uint row = 0u; row < MATRIX_ROWS; ++row) {
        for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
            uint index = row * INPUT_ROWS + input_row;
            for (uint offset = K_LANES >> 1u; offset > 0u; offset >>= 1u)
                values[index] += simd_shuffle_down(values[index], offset);
            if constexpr (SPLIT_SWIGLU) {
                if (k_lane == 0u)
                    split_values[(input_row * 2u + projection) * PHYSICAL_ROWS
                        + lane_group * ROWS_PER_SIMD + row] = T(values[index]);
            }
        }
    }
    if constexpr (SPLIT_SWIGLU) return;
    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
        uint output = output_base + row;
        for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
            if (k_lane != 0u || output >= OUT || (SPLIT_SWIGLU && projection != 0u)) continue;
            float value = values[row * INPUT_ROWS + input_row];
            if constexpr (SPLIT_SWIGLU || PACKED_SWIGLU) {
                uint local = lane_group * ROWS_PER_SIMD + row;
                float gate = SPLIT_SWIGLU
                    ? float(split_values[input_row * 2u * PHYSICAL_ROWS + local]) : float(T(value));
                float up = SPLIT_SWIGLU
                    ? float(split_values[(input_row * 2u + 1u) * PHYSICAL_ROWS + local])
                    : float(T(values[(row + ROWS_PER_SIMD) * INPUT_ROWS + input_row]));
                if (swiglu_limit > 0.0f) {
                    gate = min(gate, swiglu_limit);
                    up = clamp(up, -swiglu_limit, swiglu_limit);
                }
                value = gate / (1.0f + exp(-gate)) * up;
            }
            uint destination = SPLIT_SWIGLU || PACKED_SWIGLU
                ? route_rows[input_row] * OUT + output
                : (route_rows[input_row] * PROJECTIONS + projection) * OUT + output;
            y[destination] = T(value);
        }
    }
}

template <uint INPUT_ROWS, uint ROWS_PER_SIMD, uint K_LANES, uint OUT,
    typename RouteStream, typename YStream, typename T>
inline void mfq_moe_finish_reused_swiglu(
    YStream y, RouteStream route_rows, threadgroup T* values,
    uint output_base, uint lane_group, uint k_lane, uint projection, float limit
) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint PHYSICAL_ROWS = (32u / K_LANES) * ROWS_PER_SIMD;
    if (projection != 0u || k_lane != 0u) return;
    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
        const uint output = output_base + row;
        if (output >= OUT) continue;
        const uint local = lane_group * ROWS_PER_SIMD + row;
        for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
            float gate = float(values[input_row * 2u * PHYSICAL_ROWS + local]);
            float up = float(values[(input_row * 2u + 1u) * PHYSICAL_ROWS + local]);
            if (limit > 0.0f) {
                gate = min(gate, limit);
                up = clamp(up, -limit, limit);
            }
            y[route_rows[input_row] * OUT + output] = T(gate / (1.0f + exp(-gate)) * up);
        }
    }
}

#define TYLOINFER_MOE_FORMAT_ROWS(N, LIMIT, ROWS, MX) \
    mfq_moe_reused_rows<N, MATRIX_ROWS, ROWS_PER_SIMD, uint(K), K_LANES, uint(OUT), \
        uint(MATRIX_OUT), uint(PROJECTIONS), uint(ROUTES), uint(VQ_PROFILE_MASK), SHARED_INPUT != 0, \
        SPLIT_SWIGLU, PACKED_SWIGLU, MX>(x, y, descriptors + descriptor_base, \
        vq_indices, vq_state, vq_aux, vq_anchors, vq_codebooks, vq_scales, \
        vq_state_to_codebank, vq_parameters, mx_values, mx_scales, ROWS, split_values, \
        output_base, lane_group, k_lane, projection, uint(VARIANT_STRIDE), LIMIT)

#define TYLOINFER_MOE_SHARED_ROWS(N, LIMIT, ROWS) \
    do { \
    if (family == 3u) { TYLOINFER_MOE_FORMAT_ROWS(N, LIMIT, ROWS, true); } \
    else { TYLOINFER_MOE_FORMAT_ROWS(N, LIMIT, ROWS, false); } \
    if constexpr (SPLIT_SWIGLU) mfq_moe_finish_reused_swiglu<N, ROWS_PER_SIMD, K_LANES, uint(OUT)>( \
        y, ROWS, split_values, output_base, lane_group, k_lane, projection, LIMIT); \
    } while (false)

)METAL";

constexpr const char* kMoeSource = R"METAL(
#ifndef TYLOINFER_MOE_DUAL_PAGES
#define TYLOINFER_MOE_DUAL_PAGES 0
#endif
#ifndef TYLOINFER_MOE_OTHER_ROW
#define TYLOINFER_MOE_OTHER_ROW (uint(expert) * 2u + (projection ^ 1u))
#define TYLOINFER_MOE_OTHER_DESCRIPTORS descriptors
#endif
    constexpr bool HAS_ROUTE_GROUPS = TYLOINFER_MOE_ROUTE_GROUPS != 0;
#if !TYLOINFER_MOE_ROUTE_GROUPS
    device const int* route_groups = nullptr;
#endif
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint K_LANES = uint(K_LANES_VALUE);
    constexpr uint LANE_GROUPS = 32u / K_LANES;
    constexpr uint ROWS_PER_SIMD = uint(ROWS_PER_SIMD_VALUE);
    // A split Gate/Up MFE stores the two projections under independent
    // descriptors, and the descriptors may name completely different
    // decoder families.  Pair the two physical SIMD groups instead of
    // pretending the Up rows are the second half of the Gate descriptor.
    constexpr bool SPLIT_SWIGLU =
        uint(FUSED_SWIGLU) != 0u && uint(PROJECTIONS) == 2u;
    constexpr bool PACKED_SWIGLU =
        uint(FUSED_SWIGLU) != 0u && uint(PROJECTIONS) == 1u;
    constexpr uint MATRIX_ROWS =
        PACKED_SWIGLU
            ? 2u * ROWS_PER_SIMD
            : ROWS_PER_SIMD;
    constexpr uint ROWS_PER_PHYSICAL_SIMD =
        LANE_GROUPS * ROWS_PER_SIMD;
    constexpr uint ROWS_PER_TG =
        SPLIT_SWIGLU
            ? ROWS_PER_PHYSICAL_SIMD
            : SIMD_GROUPS * ROWS_PER_PHYSICAL_SIMD;
    constexpr uint OUTPUT_TILES =
        (uint(OUT) + ROWS_PER_TG - 1u) / ROWS_PER_TG;

    uint lane = thread_index_in_simdgroup;
    uint k_lane = lane & (K_LANES - 1u);
    uint lane_group = lane / K_LANES;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint output_tile = workgroup % OUTPUT_TILES;
    uint projection_index = workgroup / OUTPUT_TILES;
    uint projection = SPLIT_SWIGLU
        ? simd_group
        : projection_index % uint(PROJECTIONS);
    uint sorted_slot = SPLIT_SWIGLU
        ? projection_index
        : projection_index / uint(PROJECTIONS);
    uint route_index = uint(SORTED_ROUTES) != 0u
        ? uint(route_order[sorted_slot])
        : sorted_slot;
    uint route = route_index % uint(ROUTES);
    uint token = route_index / uint(ROUTES);
    if (token >= uint(TOKENS)) {
        return;
    }

    uint output_base =
        output_tile * ROWS_PER_TG
        + (SPLIT_SWIGLU ? 0u : simd_group * ROWS_PER_PHYSICAL_SIMD)
        + lane_group * ROWS_PER_SIMD;
    uint2 route_group = HAS_ROUTE_GROUPS
        ? uint2(*(device const packed_uint2*)(route_groups + route_index * 8u))
        : uint2(uint(expert_ids[route_index]), 0u);
    int logical_expert = int(route_group.x);
#if TYLOINFER_MOE_DUAL_PAGES
    const bool valid_id = logical_expert >= 0 && logical_expert < EXPERT_MAP_SIZE;
    const int gate_expert = EXPERT_MAP_SIZE != 0
        ? (valid_id ? gate_map[logical_expert] : -1) : logical_expert;
    const int up_expert = EXPERT_MAP_SIZE != 0
        ? (valid_id ? up_map[logical_expert] : -1) : logical_expert;
    int expert = projection == 0u ? gate_expert : up_expert;
    int other_expert = projection == 0u ? up_expert : gate_expert;
    const bool invalid_expert = gate_expert < 0 || gate_expert >= EXPERTS
        || up_expert < 0 || up_expert >= geometry[4];
#else
    int expert = logical_expert;
    if (PACKED_EXPERT_IDS != 0) {
        expert = (logical_expert >> 8) - 1;
    } else if (EXPERT_MAP_SIZE != 0) {
        expert = logical_expert >= 0 &&
                logical_expert < int(EXPERT_MAP_SIZE)
            ? expert_map[logical_expert]
            : -1;
    }
    const bool invalid_expert = expert < 0 || expert >= int(EXPERTS);
#endif
    if (invalid_expert) {
        for (
            uint row = 0u;
            row < ROWS_PER_SIMD;
            ++row
        ) {
            uint output = output_base + row;
            if (
                k_lane == 0u
                && output < uint(OUT)
                && (!SPLIT_SWIGLU || projection == 0u)
            ) {
                uint destination = SPLIT_SWIGLU
                    ? (token * uint(ROUTES) + route) * uint(OUT) + output
                    : (
                        (
                            (token * uint(ROUTES) + route)
                                * uint(PROJECTIONS)
                            + projection
                        ) * uint(OUT)
                        + output
                    );
                y[destination] = T(0.0f);
            }
        }
        return;
    }

    uint descriptor_base = (
        uint(expert) * uint(PROJECTIONS)
        + projection
    ) * uint(DESCRIPTOR_SIZE);
    uint family =
        uint(descriptors[descriptor_base]);
    uint local_expert =
        uint(descriptors[descriptor_base + 1u]);
    uint rotation_variant =
        uint(descriptors[descriptor_base + 27u]);
    uint x_offset = (
        rotation_variant * uint(VARIANT_STRIDE)
        + (
            uint(SHARED_INPUT) != 0u
                ? token
                : token * uint(ROUTES) + route
        )
    ) * uint(K);
    float accumulators[MATRIX_ROWS] = {0.0f};
    threadgroup T split_values[
        12u * ROWS_PER_PHYSICAL_SIMD];
    threadgroup uint grouped_route_rows[6];
    if constexpr (HAS_ROUTE_GROUPS) {
        if (family == 0u && route_group.y > 1u && route_group.y <= 6u) {
            bool reusable = mfq_moe_nint_supported_gs(uint(descriptors[descriptor_base + 5u]));
            if constexpr (SPLIT_SWIGLU) {
                const uint other = TYLOINFER_MOE_OTHER_ROW * uint(DESCRIPTOR_SIZE);
                reusable = reusable && TYLOINFER_MOE_OTHER_DESCRIPTORS[other] == 0
                    && mfq_moe_nint_supported_gs(uint(TYLOINFER_MOE_OTHER_DESCRIPTORS[other + 5u]));
            }
            if (reusable) {
                device const int* rows = route_groups + route_index * 8u + 2u;
                uint position = 0u;
                while (position < route_group.y && uint(rows[position]) != route_index) ++position;
                const uint first = position & ~1u;
                if (first + 1u < route_group.y) {
                    mfq_moe_nint_pair<MATRIX_ROWS, ROWS_PER_SIMD, uint(K), K_LANES,
                        uint(OUT), uint(MATRIX_OUT), uint(PROJECTIONS), uint(ROUTES),
                        SHARED_INPUT != 0, SPLIT_SWIGLU, PACKED_SWIGLU>(
                        x, y, descriptors + descriptor_base, nint_q, nint_sub_scale, nint_sub_min,
                        rows + first, split_values, output_base, lane_group, k_lane, projection,
                        uint(VARIANT_STRIDE), params[0], position == first);
                    return;
                }
            }
        }
    }
    if (((uint(VQ_PROFILE_MASK) & 474u) != 0u || (uint(FAMILY_MASK) & 8u) != 0u)
            && (!HAS_ROUTE_GROUPS || route_group.y > 1u)
            && (HAS_ROUTE_GROUPS || (uint(TOKENS) > 1u && uint(TOKENS) <= 6u))
            && (family == 1u || family == 3u)) {
        uint route_count = uint(TOKENS) * uint(ROUTES);
        uint preceding = 0u;
        uint matches = 0u;
        if constexpr (HAS_ROUTE_GROUPS) {
            matches = route_group.y;
        } else {
            for (uint index = lane; index < route_count; index += 32u) {
                uint match = uint(expert_ids[index] == logical_expert);
                preceding += uint(index < route_index) * match;
                matches += match;
            }
            preceding = simd_sum(preceding);
            matches = simd_sum(matches);
        }
        if (matches > 1u) {
            bool reusable_vq = family == 3u || mfq_moe_reusable_vq_profile(
                uint(descriptors[descriptor_base + 28u]) & 255u);
            if constexpr (SPLIT_SWIGLU) {
                uint other = TYLOINFER_MOE_OTHER_ROW * uint(DESCRIPTOR_SIZE);
                const uint other_family = uint(TYLOINFER_MOE_OTHER_DESCRIPTORS[other]);
                reusable_vq = reusable_vq && (other_family == 3u
                    || (other_family == 1u && mfq_moe_reusable_vq_profile(uint(TYLOINFER_MOE_OTHER_DESCRIPTORS[other + 28u]) & 255u)));
            }
            if (reusable_vq) {
                if constexpr (HAS_ROUTE_GROUPS) {
                    device const int* rows = route_groups + route_index * 8u + 2u;
                    if (uint(rows[0]) != route_index) return;
                    switch (matches) {
                        case 2u: TYLOINFER_MOE_SHARED_ROWS(2u, params[0], rows); return;
                        case 3u: TYLOINFER_MOE_SHARED_ROWS(3u, params[0], rows); return;
                        case 4u: TYLOINFER_MOE_SHARED_ROWS(4u, params[0], rows); return;
                        case 5u: TYLOINFER_MOE_SHARED_ROWS(5u, params[0], rows); return;
                        case 6u: TYLOINFER_MOE_SHARED_ROWS(6u, params[0], rows); return;
                    }
                } else {
                if (preceding % 6u != 0u) return;
                if (simd_group == 0u) {
                    uint matched = 0u;
                    for (uint base = 0u; base < route_count; base += 32u) {
                        uint index = base + lane;
                        uint match = uint(index < route_count && expert_ids[index] == logical_expert);
                        uint position = matched + simd_prefix_exclusive_sum(match);
                        if (match != 0u && position >= preceding && position < preceding + 6u)
                            grouped_route_rows[position - preceding] = index;
                        matched += simd_sum(match);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                switch (min(matches - preceding, 6u)) {
                    case 2u: TYLOINFER_MOE_SHARED_ROWS(2u, params[0], grouped_route_rows); return;
                    case 3u: TYLOINFER_MOE_SHARED_ROWS(3u, params[0], grouped_route_rows); return;
                    case 4u: TYLOINFER_MOE_SHARED_ROWS(4u, params[0], grouped_route_rows); return;
                    case 5u: TYLOINFER_MOE_SHARED_ROWS(5u, params[0], grouped_route_rows); return;
                    case 6u: TYLOINFER_MOE_SHARED_ROWS(6u, params[0], grouped_route_rows); return;
                }
                }
            }
        }
    }

    if (
        (uint(FAMILY_MASK) & 1u) != 0u
        && family == 0u
    ) {
        uint group_size =
            uint(descriptors[descriptor_base + 5u]);
        uint groups =
            uint(descriptors[descriptor_base + 6u]);
        uint q_offset =
            uint(descriptors[descriptor_base + 7u]);
        uint sub_offset =
            uint(descriptors[descriptor_base + 8u]);
        uint row_metadata_offset =
            uint(descriptors[descriptor_base + 11u]);
        device const uint* row_metadata =
            reinterpret_cast<device const uint*>(
                nint_q + row_metadata_offset);

        MfqNintRow metadata_rows[MATRIX_ROWS];
        uint q_widths[MATRIX_ROWS];
        uint q_row_byte_offsets[MATRIX_ROWS];
        uint q_row_bit_shifts[MATRIX_ROWS];
        float neuron_scales[MATRIX_ROWS];
        float neuron_minimums[MATRIX_ROWS];
        for (uint row = 0u; row < MATRIX_ROWS; ++row) {
            uint output = min(
                output_base + (
                    PACKED_SWIGLU
                        ? (row / ROWS_PER_SIMD) * uint(OUT)
                            + row % ROWS_PER_SIMD
                        : row
                ),
                uint(MATRIX_OUT) - 1u);
            uint pool_output =
                local_expert * uint(MATRIX_OUT) + output;
            const MfqNintRow metadata = mfq_nint_row(row_metadata, pool_output);
            metadata_rows[row] = metadata;
            q_widths[row] = metadata.q_bits;
            q_row_byte_offsets[row] = metadata.q_offset;
            q_row_bit_shifts[row] = metadata.q_shift;
            neuron_scales[row] = metadata.scale;
            neuron_minimums[row] = metadata.minimum;
        }

        TYLOINFER_MFE_NINT_DISPATCH {
        for (uint group = k_lane; group < groups; group += K_LANES) {
            float scales[MATRIX_ROWS];
            float minimums[MATRIX_ROWS];
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                const MfqNintRow metadata = metadata_rows[row];
                scales[row] = neuron_scales[row]
                    * float(mfq_nint_sub_value(nint_sub_scale,
                        sub_offset + metadata.sub_offset, metadata.sub_shift,
                        metadata.sub_bits, group));
                minimums[row] = neuron_minimums[row]
                    * float(mfq_nint_sub_value(nint_sub_min,
                        sub_offset + metadata.sub_offset, metadata.sub_shift,
                        metadata.sub_bits, group));
            }
            uint column_base = group * group_size;
            for (uint element = 0u; element < group_size; element += 4u) {
                uint column = column_base + element;
                float4 activations;
                if (element + 3u < group_size && column + 3u < uint(K)) {
                    activations = float4(
                        x[x_offset + column],
                        x[x_offset + column + 1u],
                        x[x_offset + column + 2u],
                        x[x_offset + column + 3u]);
                } else {
                    activations = float4(
                        element < group_size && column < uint(K)
                            ? float(x[x_offset + column]) : 0.0f,
                        element + 1u < group_size && column + 1u < uint(K)
                            ? float(x[x_offset + column + 1u]) : 0.0f,
                        element + 2u < group_size && column + 2u < uint(K)
                            ? float(x[x_offset + column + 2u]) : 0.0f,
                        element + 3u < group_size && column + 3u < uint(K)
                            ? float(x[x_offset + column + 3u]) : 0.0f);
                }
                for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                    ushort4 quantized = mfq_moe_read_nint_row_quad(
                        nint_q + q_offset,
                        q_row_byte_offsets[row],
                        q_row_bit_shifts[row],
                        column,
                        q_widths[row]);
                    float4 decoded = scales[row] * float4(quantized)
                        - minimums[row];
                    accumulators[row] += dot(activations, decoded);
                }
            }
        }
        }
    } else if (
        (uint(FAMILY_MASK) & 2u) != 0u
        && family == 1u
    ) {
        uint group_size =
            uint(descriptors[descriptor_base + 4u]);
        uint groups =
            uint(descriptors[descriptor_base + 5u]);
        uint vector_size =
            uint(descriptors[descriptor_base + 6u]);
        uint vectors =
            uint(descriptors[descriptor_base + 7u]);
        uint index_bits =
            uint(descriptors[descriptor_base + 8u]);
        uint state_bits =
            uint(descriptors[descriptor_base + 9u]);
        uint states =
            uint(descriptors[descriptor_base + 10u]);
        uint entries =
            uint(descriptors[descriptor_base + 11u]);
        uint code_banks =
            uint(descriptors[descriptor_base + 12u]);
        uint aux_mode =
            uint(descriptors[descriptor_base + 13u]);
        uint code_bank_mode =
            uint(descriptors[descriptor_base + 14u]);
        uint has_table_banks =
            uint(descriptors[descriptor_base + 15u]);
        uint groups_per_super =
            uint(descriptors[descriptor_base + 16u]);
        uint supergroups =
            uint(descriptors[descriptor_base + 17u]);
        uint indices_offset =
            uint(descriptors[descriptor_base + 18u]);
        uint state_offset =
            uint(descriptors[descriptor_base + 19u]);
        uint aux_offset =
            uint(descriptors[descriptor_base + 20u]);
        uint anchor_offset =
            uint(descriptors[descriptor_base + 21u]);
        uint codebook_offset =
            uint(descriptors[descriptor_base + 22u]);
        uint scale_offset =
            uint(descriptors[descriptor_base + 23u]);
        uint state_bank_offset =
            uint(descriptors[descriptor_base + 24u]);
        uint bank_offset =
            uint(descriptors[descriptor_base + 25u]);
        uint parameter_offset =
            uint(descriptors[descriptor_base + 26u]);
        uint vectors_per_group =
            (
                group_size + vector_size - 1u
            ) / vector_size;
        uint signs = (uint(K) + 7u) / 8u;

        uint outputs[MATRIX_ROWS];
        float row_anchors[MATRIX_ROWS];
        for (
            uint row = 0u;
            row < MATRIX_ROWS;
            ++row
        ) {
            uint output = min(
                output_base + (
                    PACKED_SWIGLU
                        ? (row / ROWS_PER_SIMD) * uint(OUT)
                            + row % ROWS_PER_SIMD
                        : row
                ),
                uint(MATRIX_OUT) - 1u);
            uint pool_output =
                local_expert * uint(MATRIX_OUT) + output;
            outputs[row] = pool_output;
            row_anchors[row] =
                vq_anchors[anchor_offset + pool_output];
        }

        uint profile =
            uint(descriptors[descriptor_base + 28u]) & 255u;
        if (
            (uint(VQ_PROFILE_MASK) & 2u) != 0u
            && profile == 1u
        ) {
            mfq_nvq_banked_profile<MATRIX_ROWS, uint(K), K_LANES, (OUT > K), 4u, 1u, 8u>(
                x, vq_indices, vq_state, vq_aux, vq_state_to_codebank, vq_scales, vq_codebooks,
                outputs, row_anchors, accumulators, x_offset,
                indices_offset, state_offset, aux_offset, state_bank_offset,
                scale_offset, codebook_offset, k_lane, 8u,
                uint(descriptors[descriptor_base + 13u]));
        } else if (
            (uint(VQ_PROFILE_MASK) & 16u) != 0u
            && profile == 4u
        ) {
            mfq_nvq_banked_profile<MATRIX_ROWS, uint(K), K_LANES, (OUT > K), 8u, 1u, 8u>(
                x, vq_indices, vq_state, vq_aux, vq_state_to_codebank, vq_scales, vq_codebooks,
                outputs, row_anchors, accumulators, x_offset,
                indices_offset, state_offset, aux_offset, state_bank_offset,
                scale_offset, codebook_offset, k_lane, 8u,
                uint(descriptors[descriptor_base + 13u]));
        } else if (
            (uint(VQ_PROFILE_MASK) & 128u) != 0u
            && profile == 7u
        ) {
            uint cohort_execution_layout = mfq_moe_vq_layout<7u>(descriptors + descriptor_base);
            if (cohort_execution_layout == 0u) {
                mfq_nvq_banked_profile<MATRIX_ROWS, uint(K), K_LANES, (OUT > K), 8u>(
                    x, vq_indices, vq_state, vq_aux, vq_state_to_codebank, vq_scales, vq_codebooks,
                    outputs, row_anchors, accumulators, x_offset,
                    indices_offset, state_offset, aux_offset, state_bank_offset,
                    scale_offset, codebook_offset, k_lane, index_bits);
            } else {
                mfq_nvq_jsc_profile<8u, uint(K), MATRIX_ROWS, VQ_EXECUTION_LAYOUT, false, 0u>(
                    x, vq_indices, vq_state, vq_aux, vq_scales,
                    vq_state_to_codebank, vq_codebooks, x_offset,
                    outputs, row_anchors, accumulators,
                    index_bits, entries, indices_offset, state_offset,
                    aux_offset, codebook_offset, scale_offset,
                    state_bank_offset, k_lane, K_LANES,
                    mfq_moe_vq_layout<7u, 65u>(descriptors + descriptor_base));
            }
        } else if (
            (uint(VQ_PROFILE_MASK) & 256u) != 0u
            && profile == 8u
        ) {
            mfq_nvq_banked_profile<MATRIX_ROWS, uint(K), K_LANES, (OUT > K)>(
                x, vq_indices, vq_state, vq_aux, vq_state_to_codebank, vq_scales, vq_codebooks,
                outputs, row_anchors, accumulators, x_offset,
                indices_offset, state_offset, aux_offset, state_bank_offset,
                scale_offset, codebook_offset, k_lane, index_bits);
        } else if (
            (uint(VQ_PROFILE_MASK) & 4u) != 0u
            && profile == 2u
        ) {
            for (
                uint group = k_lane;
                group < groups;
                group += K_LANES
            ) {
                uint states_by_row[MATRIX_ROWS];
                float weight_scales[MATRIX_ROWS];
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    uint state_index =
                        outputs[row] * groups + group;
                    uint state = mfq_moe_read_bits(
                        vq_state + state_offset,
                        state_index,
                        2u);
                    states_by_row[row] = state;
                    weight_scales[row] =
                        row_anchors[row]
                        * vq_scales[scale_offset + state];
                }
                uint3 indices_by_row[MATRIX_ROWS];
                if (uint(NPQ_GROUPED_INDICES) != 0u) {
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        indices_by_row[row] =
                            mfq_moe_read_npq_group_indices<6u>(
                                vq_indices + indices_offset,
                                outputs[row],
                                group,
                                vectors);
                    }
                }

                for (
                    uint local_vector = 0u;
                    local_vector < 3u;
                    ++local_vector
                ) {
                    uint column_base =
                        group * 24u + local_vector * 8u;
                    if (column_base >= uint(K)) {
                        break;
                    }
                    float4 activation0 = float4(
                        float(x[x_offset + column_base]),
                        float(x[x_offset + column_base + 1u]),
                        float(x[x_offset + column_base + 2u]),
                        float(x[x_offset + column_base + 3u]));
                    float4 activation1 = float4(
                        float(x[x_offset + column_base + 4u]),
                        float(x[x_offset + column_base + 5u]),
                        float(x[x_offset + column_base + 6u]),
                        float(x[x_offset + column_base + 7u]));
                    uint vector = column_base >> 3;
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        uint index =
                            uint(NPQ_GROUPED_INDICES) != 0u
                            ? indices_by_row[row][local_vector]
                            : mfq_moe_read_bits(
                                  vq_indices + indices_offset,
                                  outputs[row] * vectors + vector,
                                  6u);
                        uint code_base = codebook_offset + (
                            states_by_row[row] * 64u + index
                        ) * 8u;
                        float4 code0 = mfq_moe_load_code4(
                            vq_codebooks, code_base);
                        float4 code1 = mfq_moe_load_code4(
                            vq_codebooks, code_base + 4u);
                        float code_dot =
                            dot(activation0, code0)
                            + dot(activation1, code1);
                        accumulators[row] = fma(
                            weight_scales[row],
                            code_dot,
                            accumulators[row]);
                    }
                }
            }
        } else if (
            (uint(VQ_PROFILE_MASK) & 32u) != 0u
            && profile == 5u
        ) {
            for (
                uint group = k_lane;
                group < groups;
                group += K_LANES
            ) {
                uint states_by_row[MATRIX_ROWS];
                float weight_scales[MATRIX_ROWS];
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    uint state_index =
                        outputs[row] * groups + group;
                    uint state = mfq_moe_read_bits(
                        vq_state + state_offset,
                        state_index,
                        3u);
                    states_by_row[row] = state;
                    weight_scales[row] =
                        row_anchors[row]
                        * vq_scales[scale_offset + state];
                }
                uint3 indices_by_row[MATRIX_ROWS];
                if (uint(NPQ_GROUPED_INDICES) != 0u) {
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        indices_by_row[row] =
                            mfq_moe_read_npq_group_indices<7u>(
                                vq_indices + indices_offset,
                                outputs[row],
                                group,
                                vectors);
                    }
                }

                for (
                    uint local_vector = 0u;
                    local_vector < 3u;
                    ++local_vector
                ) {
                    uint column_base =
                        group * 24u + local_vector * 8u;
                    if (column_base >= uint(K)) {
                        break;
                    }
                    float4 activation0 = float4(
                        float(x[x_offset + column_base]),
                        float(x[x_offset + column_base + 1u]),
                        float(x[x_offset + column_base + 2u]),
                        float(x[x_offset + column_base + 3u]));
                    float4 activation1 = float4(
                        float(x[x_offset + column_base + 4u]),
                        float(x[x_offset + column_base + 5u]),
                        float(x[x_offset + column_base + 6u]),
                        float(x[x_offset + column_base + 7u]));
                    uint vector = column_base >> 3;
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        uint index =
                            uint(NPQ_GROUPED_INDICES) != 0u
                            ? indices_by_row[row][local_vector]
                            : mfq_moe_read_bits(
                                  vq_indices + indices_offset,
                                  outputs[row] * vectors + vector,
                                  7u);
                        uint code_base = codebook_offset + (
                            states_by_row[row] * 128u + index
                        ) * 8u;
                        float4 code0 = mfq_moe_load_code4(
                            vq_codebooks, code_base);
                        float4 code1 = mfq_moe_load_code4(
                            vq_codebooks, code_base + 4u);
                        float code_dot =
                            dot(activation0, code0)
                            + dot(activation1, code1);
                        accumulators[row] = fma(
                            weight_scales[row],
                            code_dot,
                            accumulators[row]);
                    }
                }
            }
        } else if (
            (uint(VQ_PROFILE_MASK) & 64u) != 0u && profile == 6u
        ) {
            mfq_nvq1_profile<MATRIX_ROWS, uint(K), K_LANES, 9u>(
                x, vq_indices, vq_state, vq_aux, vq_scales, vq_codebooks, outputs, row_anchors,
                accumulators, vq_parameters[parameter_offset], x_offset,
                indices_offset, state_offset, aux_offset, scale_offset, codebook_offset, k_lane);
        } else if (
            (uint(VQ_PROFILE_MASK) & 8u) != 0u && profile == 3u
        ) {
            mfq_nvq1_profile<MATRIX_ROWS, uint(K), K_LANES, 11u>(
                x, vq_indices, vq_state, vq_aux, vq_scales, vq_codebooks, outputs, row_anchors,
                accumulators, vq_parameters[parameter_offset], x_offset,
                indices_offset, state_offset, aux_offset, scale_offset, codebook_offset, k_lane);
        } else if (
            (uint(VQ_PROFILE_MASK) & 1u) != 0u
        ) {
        for (
            uint group = k_lane;
            group < groups;
            group += K_LANES
        ) {
            uint table_banks[MATRIX_ROWS];
            uint selected_code_banks[MATRIX_ROWS];
            uint delta_values[MATRIX_ROWS];
            float weight_scales[MATRIX_ROWS];
            for (
                uint row = 0u;
                row < MATRIX_ROWS;
                ++row
            ) {
                uint pool_output = outputs[row];
                uint state_index =
                    pool_output * groups + group;
                uint state = mfq_moe_read_bits(
                    vq_state + state_offset,
                    state_index,
                    state_bits);
                uint table_bank =
                    has_table_banks != 0u
                    ? uint(vq_banks[
                          bank_offset
                          + pool_output * supergroups
                          + group / groups_per_super
                      ])
                    : 0u;
                uint delta_value =
                    aux_mode == 3u
                    ? mfq_moe_read_bits(
                          vq_aux + aux_offset,
                          state_index,
                          1u)
                    : 0u;
                uint selected_code_bank = 0u;
                if (code_bank_mode == 1u) {
                    selected_code_bank = uint(
                        vq_state_to_codebank[
                            state_bank_offset + state
                        ]);
                } else if (code_bank_mode == 2u) {
                    selected_code_bank = delta_value;
                }
                table_banks[row] = table_bank;
                selected_code_banks[row] =
                    selected_code_bank;
                delta_values[row] = delta_value;
                weight_scales[row] =
                    row_anchors[row] * vq_scales[
                        scale_offset
                        + table_bank * states + state
                    ];
            }

            for (
                uint local_vector = 0u;
                local_vector < vectors_per_group;
                ++local_vector
            ) {
                uint column_base =
                    group * group_size
                    + local_vector * vector_size;
                if (column_base >= uint(K)) {
                    break;
                }
                float4 activations[2] = {
                    mfq_nvq_load_input4<uint(K)>(x, x_offset + column_base, column_base),
                    vector_size == 8u && column_base + 4u < uint(K)
                        ? mfq_nvq_load_input4<uint(K)>(x, x_offset + column_base + 4u, column_base + 4u)
                        : float4(0.0f),
                };
                uint vector = column_base >> (vector_size == 8u ? 3u : 2u);
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    uint index = mfq_moe_read_bits(
                        vq_indices + indices_offset,
                        outputs[row] * vectors + vector,
                        index_bits);
                    uint aux_value = 0u;
                    if (
                        aux_mode == 1u
                        || aux_mode == 2u
                    ) {
                        aux_value =
                            mfq_moe_read_bits(
                                vq_aux + aux_offset,
                                outputs[row] * signs
                                    + column_base / 8u,
                                7u);
                    }
                    uint code_base = codebook_offset + ((table_banks[row] * code_banks
                        + selected_code_banks[row]) * entries + index) * vector_size;
                    uint parity = (popcount(aux_value) & 1u)
                        ^ (aux_mode == 2u ? (index >> 7u) & 1u : 0u);
                    uint full_sign = aux_value | (parity << 7u);
                    float value = 0.0f;
                    #pragma clang loop unroll(full)
                    for (uint quad = 0u; quad < 8u; quad += 4u) {
                        if (quad >= vector_size) break;
                        float4 codes = mfq_moe_load_code4(vq_codebooks, code_base + quad);
                        if (aux_mode == 1u || aux_mode == 2u) {
                            uint sign = full_sign >> ((column_base + quad) & 7u);
                            codes = select(codes, -codes, bool4((sign & 1u) != 0u,
                                (sign & 2u) != 0u, (sign & 4u) != 0u, (sign & 8u) != 0u));
                        } else if (aux_mode == 3u) {
                            float delta = vq_parameters[parameter_offset];
                            codes += delta_values[row] != 0u ? -delta : delta;
                        }
                        value += dot(activations[quad / 4u], codes);
                    }
                    accumulators[row] = fma(weight_scales[row], value, accumulators[row]);
                }
            }
        }
        }
    } else if (
        (uint(FAMILY_MASK) & 4u) != 0u
        && family == 2u
    ) {
        uint groups =
            uint(descriptors[descriptor_base + 4u]);
        uint q_offset =
            uint(descriptors[descriptor_base + 5u]);
        uint scale_offset =
            uint(descriptors[descriptor_base + 6u]);
        for (
            uint group = k_lane;
            group < groups;
            group += K_LANES
        ) {
            uint column_base = group * 32u;
            uint outputs[MATRIX_ROWS];
            float scales[MATRIX_ROWS];
            for (
                uint row = 0u;
                row < MATRIX_ROWS;
                ++row
            ) {
                uint output = min(
                    output_base + (
                        PACKED_SWIGLU
                            ? (row / ROWS_PER_SIMD) * uint(OUT)
                                + row % ROWS_PER_SIMD
                            : row
                    ),
                    uint(MATRIX_OUT) - 1u);
                uint pool_output =
                    local_expert * uint(MATRIX_OUT) + output;
                outputs[row] = pool_output;
                scales[row] = float(q8_scales[
                    scale_offset
                    + pool_output * groups + group
                ]);
            }
            for (
                uint component = 0u;
                component < 32u;
                ++component
            ) {
                uint column =
                    column_base + component;
                float activation =
                    column < uint(K)
                        ? float(x[x_offset + column])
                        : 0.0f;
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    uint quantized_index = (
                        outputs[row] * groups + group
                    ) * 32u + component;
                    accumulators[row] = fma(
                        activation,
                        scales[row]
                            * float(q8_q[
                                q_offset
                                + quantized_index
                            ]),
                        accumulators[row]);
                }
            }
        }
    } else if (
        (uint(FAMILY_MASK) & 8u) != 0u
        && family == 3u
    ) {
        uint groups =
            uint(descriptors[descriptor_base + 4u]);
        uint value_offset =
            uint(descriptors[descriptor_base + 5u]);
        uint scale_offset =
            uint(descriptors[descriptor_base + 6u]);
        for (
            uint group = k_lane;
            group < groups;
            group += K_LANES
        ) {
            uint column_base = group * 32u;
            uint outputs[MATRIX_ROWS];
            float scales[MATRIX_ROWS];
            for (
                uint row = 0u;
                row < MATRIX_ROWS;
                ++row
            ) {
                uint output = min(
                    output_base + (
                        PACKED_SWIGLU
                            ? (row / ROWS_PER_SIMD) * uint(OUT)
                                + row % ROWS_PER_SIMD
                            : row
                    ),
                    uint(MATRIX_OUT) - 1u);
                ulong pool_output =
                    ulong(local_expert) * ulong(MATRIX_OUT) + ulong(output);
                outputs[row] = pool_output;
                scales[row] = mfq_moe_e8m0(mx_scales[
                    ulong(scale_offset)
                    + pool_output * ulong(groups) + ulong(group)
                ]);
            }
            for (
                uint component = 0u;
                component < 32u;
                component += 16u
            ) {
                uint column = column_base + component;
                vec<T, 4> source0 = *reinterpret_cast<
                    device const vec<T, 4>*>(
                        x + x_offset + column);
                vec<T, 4> source1 = *reinterpret_cast<
                    device const vec<T, 4>*>(
                        x + x_offset + column + 4u);
                vec<T, 4> source2 = *reinterpret_cast<
                    device const vec<T, 4>*>(
                        x + x_offset + column + 8u);
                vec<T, 4> source3 = *reinterpret_cast<
                    device const vec<T, 4>*>(
                        x + x_offset + column + 12u);
                float activations[16] = {
                    float(source0.x),
                    float(source0.y),
                    float(source0.z),
                    float(source0.w),
                    float(source1.x),
                    float(source1.y),
                    float(source1.z),
                    float(source1.w),
                    float(source2.x),
                    float(source2.y),
                    float(source2.z),
                    float(source2.w),
                    float(source3.x),
                    float(source3.y),
                    float(source3.z),
                    float(source3.w),
                };
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    ulong packed_offset = ulong(value_offset)
                        + outputs[row] * (ulong(K) >> 1u)
                        + (ulong(column) >> 1u);
                    uint2 packed = *reinterpret_cast<device const uint2*>(
                        mx_values + packed_offset);
                    for (uint pair = 0u; pair < 8u; ++pair) {
                        uint word = pair < 4u ? packed.x : packed.y;
                        uchar codes = uchar(
                            word >> ((pair & 3u) * 8u));
                        accumulators[row] = fma(
                            activations[pair * 2u],
                            scales[row]
                                * mfq_moe_mxfp4_value(codes & 15u),
                            accumulators[row]);
                        accumulators[row] = fma(
                            activations[pair * 2u + 1u],
                            scales[row]
                                * mfq_moe_mxfp4_value(codes >> 4u),
                            accumulators[row]);
                    }
                }
            }
        }
    } else if (
        (uint(FAMILY_MASK) & 16u) != 0u
        && family == 4u
    ) {
        uint groups =
            uint(descriptors[descriptor_base + 4u]);
        uint value_offset =
            uint(descriptors[descriptor_base + 5u]);
        uint scale_offset =
            uint(descriptors[descriptor_base + 6u]);
        for (
            uint group = k_lane;
            group < groups;
            group += K_LANES
        ) {
            uint column_base = group * 128u;
            ulong outputs[MATRIX_ROWS];
            float scales[MATRIX_ROWS];
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                uint output = min(
                    output_base + (
                        PACKED_SWIGLU
                            ? (row / ROWS_PER_SIMD) * uint(OUT)
                                + row % ROWS_PER_SIMD
                            : row
                    ),
                    uint(MATRIX_OUT) - 1u);
                ulong pool_output =
                    ulong(local_expert) * ulong(MATRIX_OUT) + ulong(output);
                outputs[row] = pool_output;
                scales[row] = mfq_moe_e8m0(mx_scales[
                    ulong(scale_offset)
                    + (pool_output >> 7u) * ulong(groups)
                    + ulong(group)
                ]);
            }
            for (uint component = 0u; component < 128u; ++component) {
                uint column = column_base + component;
                float activation = column < uint(K)
                    ? float(x[x_offset + column])
                    : 0.0f;
                for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                    uchar code = mx_values[
                        ulong(value_offset)
                        + outputs[row] * ulong(K)
                        + ulong(column)];
                    accumulators[row] = fma(
                        activation,
                        scales[row] * mfq_moe_mxfp8_value(code),
                        accumulators[row]);
                }
            }
        }
    } else if (
        (uint(FAMILY_MASK) & 96u) != 0u
        && (family == 5u || family == 6u)
    ) {
        uint value_offset =
            uint(descriptors[descriptor_base + 4u]);
        for (
            uint column = k_lane;
            column < uint(K);
            column += K_LANES
        ) {
            float activation = float(x[x_offset + column]);
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                uint output = min(
                    output_base + (
                        PACKED_SWIGLU
                            ? (row / ROWS_PER_SIMD) * uint(OUT)
                                + row % ROWS_PER_SIMD
                            : row
                    ),
                    uint(MATRIX_OUT) - 1u);
                ulong pool_output =
                    ulong(local_expert) * ulong(MATRIX_OUT) + ulong(output);
                ulong byte_offset = ulong(value_offset)
                    + (pool_output * ulong(K) + ulong(column)) * 2u;
                ushort raw = *reinterpret_cast<device const ushort*>(
                    q8_q + byte_offset);
                float weight = family == 5u
                    ? as_type<float>(uint(raw) << 16u)
                    : float(as_type<half>(raw));
                accumulators[row] = fma(
                    activation,
                    weight,
                    accumulators[row]);
            }
        }
    }

    if (
        uint(HAS_NEPQ_RESIDUAL) != 0u
        && family == 1u
    ) {
        uint residual_profile =
            uint(descriptors[descriptor_base + 28u]);
        uint position_bits = (residual_profile >> 8u) & 255u;
        uint block_vectors = (residual_profile >> 16u) & 255u;
        if (position_bits != 0u && block_vectors != 0u) {
            uint vectors =
                uint(descriptors[descriptor_base + 7u]);
            uint residual_blocks =
                (vectors + block_vectors - 1u) / block_vectors;
            uint residual_codebook_offset =
                uint(descriptors[descriptor_base + 30u]);
            uint residual_record_offset =
                uint(descriptors[descriptor_base + 31u]);
            uint position_mask = (1u << position_bits) - 1u;
            for (
                uint block = k_lane;
                block < residual_blocks;
                block += K_LANES
            ) {
                for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                    uint output = min(
                        output_base + (
                            PACKED_SWIGLU
                                ? (row / ROWS_PER_SIMD) * uint(OUT)
                                    + row % ROWS_PER_SIMD
                                : row
                        ),
                        uint(MATRIX_OUT) - 1u);
                    uint pool_output =
                        local_expert * uint(MATRIX_OUT) + output;
                    uint record_index = residual_record_offset
                        + pool_output * residual_blocks + block;
                    short records[2] = {
                        vq_residual_first[record_index],
                        vq_residual_second[record_index],
                    };
                    for (uint stream = 0u; stream < 2u; ++stream) {
                        int record = int(records[stream]);
                        if (record < 0) {
                            continue;
                        }
                        uint position = uint(record) & position_mask;
                        uint dictionary_id = uint(record) >> position_bits;
                        uint vector = block * block_vectors + position;
                        if (dictionary_id >= 1024u || vector >= vectors) {
                            continue;
                        }
                        uint input_offset = x_offset + vector * 8u;
                        uint dictionary_offset = residual_codebook_offset
                            + dictionary_id * 8u;
                        for (uint component = 0u; component < 8u; ++component) {
                            accumulators[row] = fma(
                                float(x[input_offset + component]),
                                vq_residual_codebooks[
                                    dictionary_offset + component],
                                accumulators[row]);
                        }
                    }
                }
            }
        }
    }

    if (SPLIT_SWIGLU) {
        for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
            float total = accumulators[row];
            for (
                uint offset = K_LANES >> 1;
                offset > 0u;
                offset >>= 1
            ) {
                total += simd_shuffle_down(total, offset);
            }
            if (k_lane == 0u) {
                split_values[
                    projection * ROWS_PER_PHYSICAL_SIMD
                    + lane_group * ROWS_PER_SIMD
                    + row
                ] = T(total);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0u && k_lane == 0u) {
            for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                uint local_output =
                    lane_group * ROWS_PER_SIMD + row;
                uint output = output_tile * ROWS_PER_TG + local_output;
                if (output < uint(OUT)) {
                    // Preserve the exact unfused graph boundary: both
                    // projections round before the elementwise activation.
                    float gate = float(split_values[local_output]);
                    float up = float(split_values[
                        ROWS_PER_PHYSICAL_SIMD + local_output]);
                    if (params[0] > 0.0f) {
                        gate = min(gate, params[0]);
                        up = clamp(up, -params[0], params[0]);
                    }
                    float activated = gate / (1.0f + exp(-gate));
                    y[
                        (token * uint(ROUTES) + route) * uint(OUT)
                        + output
                    ] = T(activated * up);
                }
            }
        }
    } else if (uint(FUSED_SWIGLU) != 0u) {
        for (
            uint row = 0u;
            row < ROWS_PER_SIMD;
            ++row
        ) {
            float gate = accumulators[row];
            float up = accumulators[ROWS_PER_SIMD + row];
            for (
                uint offset = K_LANES >> 1;
                offset > 0u;
                offset >>= 1
            ) {
                gate += simd_shuffle_down(gate, offset);
                up += simd_shuffle_down(up, offset);
            }
            uint output = output_base + row;
            if (k_lane == 0u && output < uint(OUT)) {
                // Match the unfused graph: both projections round to the
                // activation dtype before the elementwise SwiGLU.
                gate = float(T(gate));
                up = float(T(up));
                if (params[0] > 0.0f) {
                    gate = min(gate, params[0]);
                    up = clamp(up, -params[0], params[0]);
                }
                float activated = gate / (1.0f + exp(-gate));
                y[
                    (token * uint(ROUTES) + route)
                        * uint(OUT)
                    + output
                ] = T(activated * up);
            }
        }
    } else {
        for (
            uint row = 0u;
            row < ROWS_PER_SIMD;
            ++row
        ) {
            float total = accumulators[row];
            for (
                uint offset = K_LANES >> 1;
                offset > 0u;
                offset >>= 1
            ) {
                total += simd_shuffle_down(total, offset);
            }
            uint output = output_base + row;
            if (k_lane == 0u && output < uint(OUT)) {
                y[
                    (
                        (
                            token * uint(ROUTES) + route
                        ) * uint(PROJECTIONS)
                        + projection
                    ) * uint(OUT)
                    + output
                ] = T(total);
            }
        }
    }
)METAL";

constexpr const char* kMoeHadamardSource = R"METAL(
    const uint M = uint(x_shape[0]);
    uint row = thread_position_in_grid.x / 256u;
    uint lane = thread_index_in_threadgroup;
    if (row >= uint(M)) {
        return;
    }

    threadgroup float values[BLOCK];
    for (
        uint local_block = 0u;
        local_block < uint(K) / uint(BLOCK);
        ++local_block
    ) {
        uint column_base =
            local_block * uint(BLOCK);
        for (
            uint index = lane;
            index < uint(BLOCK);
            index += 256u
        ) {
            uint column = column_base + index;
            values[index] =
                float(x[row * uint(K) + column])
                * float(signs[column]);
        }
        threadgroup_barrier(
            mem_flags::mem_threadgroup);
        for (
            uint stride = 1u;
            stride < uint(BLOCK);
            stride <<= 1u
        ) {
            for (
                uint pair = lane;
                pair < uint(BLOCK) / 2u;
                pair += 256u
            ) {
                uint pair_block = pair / stride;
                uint within =
                    pair - pair_block * stride;
                uint first =
                    pair_block * (stride << 1u)
                    + within;
                uint second = first + stride;
                float first_value = values[first];
                float second_value = values[second];
                values[first] =
                    first_value + second_value;
                values[second] =
                    first_value - second_value;
            }
            threadgroup_barrier(
                mem_flags::mem_threadgroup);
        }
        float inverse = rsqrt(float(BLOCK));
        for (
            uint index = lane;
            index < uint(BLOCK);
            index += 256u
        ) {
            uint column = column_base + index;
            y[row * uint(K) + column] =
                T(values[index] * inverse);
        }
        threadgroup_barrier(
            mem_flags::mem_threadgroup);
    }
)METAL";

class BlobCursor {
public:
    explicit BlobCursor(
        std::span<const std::uint8_t> blob)
        : blob_(blob) {}

    template <typename T>
    T scalar(const char* name) {
        require(sizeof(T), name);
        T value{};
        std::memcpy(
            &value,
            blob_.data() + offset_,
            sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    std::span<const std::uint8_t> bytes(
        std::size_t count,
        const char* name) {
        require(count, name);
        auto result = blob_.subspan(offset_, count);
        offset_ += count;
        return result;
    }

    std::size_t remaining() const noexcept {
        return blob_.size() - offset_;
    }

private:
    void require(
        std::size_t count,
        const char* name) const {
        if (count > blob_.size() - offset_) {
            throw std::runtime_error(
                std::string("truncated MFE ") + name);
        }
    }

    std::span<const std::uint8_t> blob_;
    std::size_t offset_ = 0;
};

std::size_t checked_size(
    std::uint64_t value,
    const char* name) {
    if (
        value
        > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())
    ) {
        throw std::runtime_error(
            std::string("MFE ") + name
            + " exceeds addressable memory");
    }
    return static_cast<std::size_t>(value);
}

std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* name) {
    if (
        right != 0
        && left
            > std::numeric_limits<std::size_t>::max()
                / right
    ) {
        throw std::runtime_error(
            std::string("MFE ") + name
            + " overflows");
    }
    return left * right;
}

std::size_t checked_add(
    std::size_t left,
    std::size_t right,
    const char* name) {
    if (
        right
        > std::numeric_limits<std::size_t>::max()
            - left
    ) {
        throw std::runtime_error(
            std::string("MFE ") + name
            + " overflows");
    }
    return left + right;
}

std::size_t checked_packed_size(
    std::size_t count,
    int bits,
    const char* name) {
    const auto bit_count = checked_product(
        count,
        static_cast<std::size_t>(bits),
        name);
    return checked_add(bit_count, 7, name) / 8;
}

std::int32_t checked_int(
    std::size_t value,
    const char* name) {
    if (
        value
        > static_cast<std::size_t>(
            std::numeric_limits<std::int32_t>::max())
    ) {
        throw std::runtime_error(
            std::string("MFE ") + name
            + " exceeds int32 range");
    }
    return static_cast<std::int32_t>(value);
}

constexpr std::int32_t descriptor_u32_bits(
    std::uint32_t value) noexcept {
    return std::bit_cast<std::int32_t>(value);
}

constexpr std::uint32_t descriptor_u32_value(
    std::int32_t value) noexcept {
    return std::bit_cast<std::uint32_t>(value);
}

static_assert(
    descriptor_u32_value(
        descriptor_u32_bits(std::uint32_t{0x80000000u}))
    == std::uint32_t{0x80000000u});
static_assert(
    descriptor_u32_value(
        descriptor_u32_bits(std::uint32_t{0xffffffffu}))
    == std::uint32_t{0xffffffffu});

std::int32_t checked_descriptor_u32(
    std::size_t value,
    const char* name) {
    if (
        value
        > static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max())
    ) {
        throw std::runtime_error(
            std::string("MFE ") + name
            + " exceeds uint32 descriptor range");
    }
    return descriptor_u32_bits(
        static_cast<std::uint32_t>(value));
}

int checked_positive(
    std::uint32_t value,
    const char* name) {
    if (
        value == 0
        || value
            > static_cast<std::uint32_t>(
                std::numeric_limits<std::int32_t>::max())
    ) {
        throw std::runtime_error(
            std::string("invalid MFE ") + name);
    }
    return static_cast<int>(value);
}

array allocate_packed_array(std::size_t bytes, Dtype dtype) {
    if (bytes == 0 || bytes % dtype.size() != 0) {
        throw std::runtime_error("MFE packed stream is misaligned");
    }
    const auto layout = tyloi::metal::packed_layout(bytes / dtype.size());
    const Shape shape = layout.is_matrix()
        ? Shape{layout.rows, layout.columns} : Shape{layout.columns};
    auto result = array(mlx::core::allocator::malloc(bytes), shape, dtype);
    tyloi::metal::AllocationScope::record(result);
    return result;
}

template <typename Array = array>
Array wrap_wire_buffer(mlx::core::allocator::Buffer buffer, const Shape& shape,
    std::function<void(mlx::core::allocator::Buffer)> deleter) {
    if constexpr (requires { Array(buffer, shape, mlx::core::uint8, deleter); })
        return Array(buffer, shape, mlx::core::uint8, std::move(deleter));
    else
        return Array(typename Array::Data(buffer, std::move(deleter)), shape, mlx::core::uint8);
}

array allocate_wire_array(std::size_t bytes, bool account = false) {
    struct Storage {
        std::uint8_t* data = nullptr;
        std::size_t bytes = 0;
        bool accounted = false;
        mlx::core::allocator::Buffer buffer{nullptr};
        ~Storage() {
            if (buffer.ptr()) mlx::core::allocator::release(buffer);
            tyloi::metal::StagingAllocator<std::uint8_t>{}.deallocate(data, bytes);
            if (accounted) uncached_packed_storage_bytes.fetch_sub(bytes, std::memory_order_relaxed);
        }
    };
    const auto layout = tyloi::metal::packed_layout(bytes);
    const Shape shape = layout.is_matrix()
        ? Shape{layout.rows, layout.columns} : Shape{layout.columns};
    const auto page = static_cast<std::size_t>(getpagesize());
    auto storage = std::make_shared<Storage>();
    storage->bytes = checked_add(bytes, page - 1, "MFE wire allocation") / page * page;
    storage->data = tyloi::metal::StagingAllocator<std::uint8_t>{}.allocate(storage->bytes);
    storage->buffer = mlx::core::allocator::make_buffer(storage->data, storage->bytes);
    if (!storage->buffer.ptr()) return allocate_packed_array(bytes, mlx::core::uint8);
    if (account) {
        uncached_packed_storage_bytes.fetch_add(storage->bytes, std::memory_order_relaxed);
        storage->accounted = true;
    }
    const auto buffer = storage->buffer;
    return wrap_wire_buffer(buffer, shape,
        [storage = std::move(storage)](mlx::core::allocator::Buffer) {});
}

array empty_packed_array(Dtype dtype) {
    struct Entry {
        std::once_flag initialized;
        std::optional<array> value;
    };
    static std::array<Entry, static_cast<std::size_t>(Dtype::Val::complex64) + 1> values;
    auto& entry = values.at(static_cast<std::size_t>(dtype.val()));
    std::call_once(entry.initialized, [&] {
        auto value = allocate_packed_array(dtype.size(), dtype);
        std::memset(value.data<std::uint8_t>(), 0, dtype.size());
        entry.value.emplace(std::move(value));
    });
    return *entry.value;
}

bool redirect_direct_read(std::span<const std::uint8_t> source, std::uint8_t* destination) {
    if (!direct_read_targets) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(source.data());
    std::size_t copied = 0;
    bool redirected = false;
    for (auto& range : *direct_read_targets) {
        const auto address = reinterpret_cast<std::uintptr_t>(range.wire);
        if (address < begin || address - begin > source.size()
            || range.bytes > source.size() - (address - begin)) continue;
        const auto offset = static_cast<std::size_t>(address - begin);
        if (offset < copied || range.destination) throw std::runtime_error("overlapping direct MFE read");
        std::memcpy(destination + copied, source.data() + copied, offset - copied);
        range.destination = destination + offset;
        copied = offset + range.bytes;
        redirected = true;
    }
    if (redirected) std::memcpy(destination + copied, source.data() + copied, source.size() - copied);
    return redirected;
}

// Keep source ownership, not a second model-sized byte vector. Once all chunks
// are known, write them directly into one exact-sized Metal allocation.
class PackedStream {
public:
    PackedStream() = default;
    PackedStream(PackedStream&&) = default;
    PackedStream& operator=(PackedStream&&) = default;
    PackedStream(const PackedStream&) = delete;
    PackedStream& operator=(const PackedStream&) = delete;

    std::size_t size() const noexcept { return size_; }

    void retain_source(const array& source) { source_ = source; }

    void append(array source) {
        const auto bytes = source.nbytes();
        append_chunk(std::move(source), bytes);
    }

    void append(std::span<const std::uint8_t> source) {
        // Borrowed wire views live until from_blob finishes materialization.
        append_chunk(source, source.size());
    }

    template <typename T>
    void append(tyloi::metal::StagingVector<T> source) {
        const auto bytes = checked_product(source.size(), sizeof(T), "packed stream size");
        append_chunk(std::move(source), bytes);
    }

    void append(float value) {
        std::array<std::uint8_t, sizeof(float)> bytes;
        std::memcpy(bytes.data(), &value, sizeof(value));
        append_chunk(bytes, bytes.size());
    }

    void push_back(std::uint8_t value) {
        append_chunk(value, 1);
    }

    array materialize(Dtype dtype, const array* arena = nullptr, std::size_t arena_offset = 0) && {
        if (size_ == 0) return empty_packed_array(dtype);
        const auto bytes = size_;
        if (bytes % dtype.size() != 0) {
            throw std::runtime_error("MFE packed stream is misaligned");
        }
        const auto layout = tyloi::metal::packed_layout(bytes / dtype.size());
        const Shape shape = layout.is_matrix()
            ? Shape{layout.rows, layout.columns} : Shape{layout.columns};
        if (chunks_.size() == 1) {
            if (const auto* source = std::get_if<array>(&chunks_.front());
                source != nullptr && source->dtype() == dtype) {
                auto result = mlx::core::reshape(*source, shape);
                result.eval();
                tyloi::metal::AllocationScope::record(result);
                return result;
            }
        }
        if (source_ && bytes >= (std::size_t{8} << 20)
            && source_->nbytes() <= std::size_t(std::numeric_limits<int>::max())
            && bytes <= std::size_t(std::numeric_limits<int>::max())) {
            const auto wire = mlx::core::reshape(*source_, Shape{static_cast<int>(source_->nbytes())});
            const auto base = reinterpret_cast<std::uintptr_t>(source_->data<std::uint8_t>());
            std::vector<array> parts;
            for (const auto& chunk : chunks_) {
                std::visit([&](const auto& source) {
                    using T = std::decay_t<decltype(source)>;
                    if constexpr (std::is_same_v<T, std::uint8_t>) {
                        parts.emplace_back(std::initializer_list<std::uint8_t>{source}, mlx::core::uint8);
                    } else if constexpr (std::is_same_v<T, array>) {
                        parts.push_back(mlx::core::reshape(mlx::core::view(source, mlx::core::uint8),
                            Shape{checked_int(source.nbytes(), "packed source bytes")}));
                    } else {
                        const auto count = source.size() * sizeof(typename T::value_type);
                        const auto* data = reinterpret_cast<const std::uint8_t*>(source.data());
                        if constexpr (std::is_same_v<T, std::span<const std::uint8_t>>) {
                            const auto address = reinterpret_cast<std::uintptr_t>(data);
                            if (address >= base && address - base <= source_->nbytes()
                                && count <= source_->nbytes() - (address - base)) {
                                const auto start = static_cast<int>(address - base);
                                parts.push_back(mlx::core::slice(wire, {start},
                                    {start + static_cast<int>(count)}));
                                return;
                            }
                        }
                        parts.emplace_back(data, Shape{checked_int(count, "packed source bytes")}, mlx::core::uint8);
                    }
                }, chunk);
            }
            auto result = mlx::core::reshape(mlx::core::view(
                mlx::core::concatenate(std::move(parts), 0, mlx::core::Device::gpu), dtype), shape);
            result.eval();
            tyloi::metal::AllocationScope::record(result);
            return result;
        }
        auto result = arena ? array(shape, dtype, nullptr, {}) : allocate_packed_array(bytes, dtype);
        if (arena) {
            if (arena_offset > arena->nbytes() || bytes > arena->nbytes() - arena_offset)
                throw std::out_of_range("MFE packed arena is too small");
            result.copy_shared_buffer(*arena, result.strides(), result.flags(), bytes / dtype.size(), arena_offset / dtype.size());
            result.set_status(array::Status::available);
        }
        auto* destination = result.data<std::uint8_t>();
        std::size_t offset = 0;
        for (const auto& chunk : chunks_) {
            std::visit([&](const auto& source) {
                using T = std::decay_t<decltype(source)>;
                if constexpr (std::is_same_v<T, std::uint8_t>) {
                    destination[offset++] = source;
                } else {
                    const auto count = [&] {
                        if constexpr (std::is_same_v<T, array>) return source.nbytes();
                        else return source.size() * sizeof(typename T::value_type);
                    }();
                    const auto* data = [&] {
                        if constexpr (std::is_same_v<T, array>) return source.template data<std::uint8_t>();
                        else return source.data();
                    }();
                    if constexpr (std::is_same_v<T, std::span<const std::uint8_t>>) {
                        if (redirect_direct_read(source, destination + offset)) {
                            offset += count;
                            return;
                        }
                    }
                    std::memcpy(destination + offset, data, count);
                    offset += count;
                }
            }, chunk);
        }
        return result;
    }

private:
    using Chunk = std::variant<array, std::span<const std::uint8_t>, std::uint8_t,
        tyloi::metal::StagingVector<std::uint8_t>, tyloi::metal::StagingVector<std::uint32_t>,
        tyloi::metal::StagingVector<float>, tyloi::metal::StagingVector<std::int8_t>,
        tyloi::metal::StagingVector<std::int16_t>, std::array<std::uint8_t, sizeof(float)>>;

    template <typename T>
    void append_chunk(T source, std::size_t bytes) {
        if (bytes != 0) {
            const auto total = checked_add(size_, bytes, "packed stream size");
            chunks_.emplace_back(std::move(source));
            size_ = total;
        }
    }

    std::vector<Chunk> chunks_;
    std::optional<array> source_;
    std::size_t size_ = 0;
};

void append_raw(
    PackedStream& target,
    const array& source,
    Dtype expected,
    const char* name) {
    if (
        source.dtype() != expected
        || !source.flags().row_contiguous
    ) {
        throw std::runtime_error(
            std::string("invalid MFE packed ") + name);
    }
    auto evaluated = source;
    evaluated.eval();
    target.append(std::move(evaluated));
}

void append_bytes(
    PackedStream& target,
    std::span<const std::uint8_t> source,
    const char* name) {
    if (
        source.size()
        > std::numeric_limits<std::size_t>::max()
            - target.size()
    ) {
        throw std::runtime_error(
            std::string("MFE packed ") + name
            + " stream size overflows");
    }
    target.append(source);
}

array make_raw_array(PackedStream bytes, Dtype dtype) {
    return std::move(bytes).materialize(dtype);
}

template <typename Allocator>
array make_raw_array(
    std::vector<std::uint8_t, Allocator> bytes,
    Dtype dtype) {
    if (bytes.empty()) {
        return empty_packed_array(dtype);
    }
    auto result = allocate_packed_array(bytes.size(), dtype);
    std::memcpy(
        result.template data<std::uint8_t>(),
        bytes.data(),
        bytes.size());
    return result;
}

void align_packed_stream(
    std::vector<array>& chunks, std::size_t& offset, std::size_t alignment) {
    const auto padding = (alignment - offset % alignment) % alignment;
    if (padding != 0) {
        chunks.push_back(make_raw_array(std::vector<std::uint8_t>(padding, 0), mlx::core::uint8));
        offset = checked_add(offset, padding, "packed stream alignment");
    }
}

array make_int32_array(
    const std::vector<std::int32_t>& values,
    Shape shape,
    const array* arena = nullptr) {
    if (values.size() == 1 && values.front() == 0)
        return mlx::core::reshape(empty_packed_array(mlx::core::int32), std::move(shape));
    if (arena) {
        const auto bytes = checked_product(values.size(), sizeof(std::int32_t), "MFE metadata bytes");
        if (bytes > arena->nbytes()) throw std::out_of_range("MFE metadata arena is too small");
        auto result = array(std::move(shape), mlx::core::int32, nullptr, {});
        result.copy_shared_buffer(*arena, result.strides(), result.flags(), values.size());
        result.set_status(array::Status::available);
        std::memcpy(result.data<std::int32_t>(), values.data(), bytes);
        return result;
    }
    auto result = array(values.begin(), std::move(shape));
    tyloi::metal::AllocationScope::record(result);
    return result;
}

constexpr std::size_t kPackedCopyInputs = 24;
std::atomic<std::size_t> packed_copy_submissions{0};
std::atomic<std::size_t> indirect_expert_submissions{0};
std::atomic<std::size_t> packed_pinned_bytes{0}, packed_pin_errors{0};
using PackedCopySpec = std::array<std::uint32_t, 4>;

struct PackedCopyBatch {
    std::uint32_t first;
    std::uint32_t count;
    std::uint32_t source;
    bool operator==(const PackedCopyBatch&) const = default;
};

class PackedCopyPrimitive final : public mlx::core::UnaryPrimitive, public tyloi::metal::PreparableKernel {
public:
    PackedCopyPrimitive(mlx::core::Stream stream, Shape shape, std::vector<PackedCopyBatch> batches)
        : UnaryPrimitive(stream), shape_(std::move(shape)), batches_(std::move(batches)) {}
    std::string preparation_key() const override { return "mfq_mfe_packed_copy_v2"; }
    void prepare_gpu() override { (void)kernel(); }
    void eval_cpu(const std::vector<array>&, array&) override {
        throw std::runtime_error("packed expert copy requires Metal");
    }
    void eval_gpu(const std::vector<array>& inputs, array& output) override {
        if (inputs.size() < 2) throw std::logic_error("packed expert copy has no sources");
        if (output.nbytes() >= (std::size_t{64} << 10)) {
            auto storage = allocate_wire_array(output.nbytes(), true);
            output.copy_shared_buffer(storage, output.strides(), output.flags(), output.size());
        } else output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        auto& encoder = mlx::core::metal::get_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel());
        encoder.set_input_array(inputs.back(), kPackedCopyInputs);
        encoder.set_output_array(output, kPackedCopyInputs + 1);
        const auto empty = empty_packed_array(mlx::core::uint8);
        for (const auto& batch : batches_) {
            for (std::size_t index = 0; index < kPackedCopyInputs; ++index) {
                const auto source = batch.source + index;
                encoder.set_input_array(source < inputs.size() - 1 ? inputs[source] : empty, index);
            }
            const std::array<std::uint32_t, 2> geometry{batch.first, batch.source};
            encoder.set_bytes(geometry, kPackedCopyInputs + 2);
            encoder.dispatch_threadgroups(MTL::Size(batch.count, 1, 1), MTL::Size(256, 1, 1));
        }
        packed_copy_submissions.fetch_add(1, std::memory_order_relaxed);
    }
    const char* name() const override { return "PackedCopyPrimitive"; }
    bool is_equivalent(const mlx::core::Primitive& other) const override {
        const auto* value = dynamic_cast<const PackedCopyPrimitive*>(&other);
        return value && value->shape_ == shape_ && value->batches_ == batches_;
    }
    std::vector<Shape> output_shapes(const std::vector<array>&) override { return {shape_}; }
private:
    MTL::ComputePipelineState* kernel() {
        auto& device = tyloi::metal::kernel_device(stream().device);
        auto* library = device.get_library(preparation_key(), [] {
            std::ostringstream source;
            source << "#include <metal_stdlib>\nusing namespace metal;\n"
                "kernel void mfq_mfe_packed_copy_v2(\n";
            for (std::size_t index = 0; index < kPackedCopyInputs; ++index)
                source << "device const uchar* p" << index << " [[buffer(" << index << ")]],\n";
            source << R"(device const uint4* moves [[buffer(24)]],
device uchar* output [[buffer(25)]], constant uint* geometry [[buffer(26)]],
uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint4 move = moves[geometry[0] + group.x];
    const uint bytes = move.w & 0x0fffffffu;
    if (move.x == 0xffffffffu) {
        for (uint index = lane; index < bytes; index += 256) output[move.z + index] = 0;
        return;
    }
    device const uchar* input = nullptr;
    switch (move.x - geometry[1]) {
)";
            for (std::size_t index = 0; index < kPackedCopyInputs; ++index)
                source << "case " << index << ": input = p" << index << "; break;\n";
            source << R"(default: return;
    }
    input += move.y;
    device uchar* target = output + move.z;
    const uint alignment = move.y | move.z | (move.w >> 28);
    if ((alignment & 3u) == 0) {
        device const uint* words = reinterpret_cast<device const uint*>(input);
        device uint* destination = reinterpret_cast<device uint*>(target);
        for (uint index = lane; index < bytes / 4; index += 256) destination[index] = words[index];
        for (uint index = (bytes & ~3u) + lane; index < bytes; index += 256) target[index] = input[index];
    } else {
        for (uint index = lane; index < bytes; index += 256) target[index] = input[index];
    }
}
)";
            return source.str();
        });
        return device.get_kernel(preparation_key(), library);
    }
    Shape shape_;
    std::vector<PackedCopyBatch> batches_;
};

array packed_expert_copy(std::vector<array> sources,
    const std::vector<std::vector<PackedCopySpec>>& copies, std::size_t bytes) {
    if (sources.empty() || bytes == 0 || bytes > std::size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid packed expert copy extent");
    std::vector<std::uint32_t> moves;
    std::vector<PackedCopyBatch> batches;
    for (std::size_t index = 0; index < copies.size(); ++index) {
        if (copies[index].empty()) continue;
        batches.push_back({static_cast<std::uint32_t>(moves.size() / 4),
            static_cast<std::uint32_t>(copies[index].size()),
            static_cast<std::uint32_t>(index * kPackedCopyInputs)});
        for (const auto& copy : copies[index]) moves.insert(moves.end(), copy.begin(), copy.end());
    }
    if (moves.empty()) throw std::invalid_argument("packed expert copy has no ranges");
    sources.emplace_back(moves.begin(), Shape{checked_int(moves.size() / 4, "packed copy ranges"), 4});
    const Shape shape{static_cast<int>(bytes)};
    return array(shape, mlx::core::uint8,
        std::make_shared<PackedCopyPrimitive>(mlx::core::default_stream(mlx::core::Device::gpu), shape, std::move(batches)),
        std::move(sources));
}

struct PackedExpertArena {
    array storage;
    struct Stream {
        const std::vector<array>* values;
        std::size_t offset;
        std::size_t bytes;
    };
    std::vector<Stream> streams;
};

std::optional<PackedExpertArena> pack_expert_streams(std::vector<array> sources,
    std::span<const std::vector<array>* const> streams) {
    std::unordered_map<const void*, std::uint32_t> owners;
    std::unordered_map<std::uintptr_t, std::uint32_t> external;
    std::vector<array> unique;
    for (const auto& source : sources) {
        if (!source.is_available() || !source.flags().row_contiguous)
            continue;
        if (owners.emplace(source.data_shared_ptr().get(), static_cast<std::uint32_t>(unique.size())).second) {
            if (source.dtype() == mlx::core::uint8
                && source.buffer_size() <= std::size_t(std::numeric_limits<int>::max())) {
                auto backing = array(Shape{static_cast<int>(source.buffer_size())}, mlx::core::uint8, nullptr, {});
                backing.copy_shared_buffer(source, backing.strides(), backing.flags(), backing.size(), -source.offset());
                backing.set_status(array::Status::available);
                unique.push_back(std::move(backing));
            } else unique.push_back(source);
        }
    }
    std::vector<std::vector<PackedCopySpec>> copies(std::max<std::size_t>(1,
        (unique.size() + kPackedCopyInputs - 1) / kPackedCopyInputs));
    std::vector<PackedExpertArena::Stream> views;
    std::size_t bytes = 0;
    constexpr std::size_t tile = std::size_t{64} << 10;
    for (const auto* values : streams) {
        if (values->empty()) continue;
        const auto empty = empty_packed_array(values->front().dtype()).data_shared_ptr();
        if (std::all_of(values->begin(), values->end(), [&](const auto& value) {
                return value.data_shared_ptr() == empty;
            })) continue;
        bytes = checked_add(bytes, 15, "packed copy alignment") & ~std::size_t{15};
        const auto first = bytes;
        for (const auto& value : *values) {
            if (!value.is_available() || !value.flags().row_contiguous) return std::nullopt;
            const auto next = checked_add(bytes, value.nbytes(), "packed copy output");
            if (next > std::size_t(std::numeric_limits<int>::max())) return std::nullopt;
            const auto found = owners.find(value.data_shared_ptr().get());
            std::optional<std::pair<std::uint32_t, std::size_t>> range;
            if (found != owners.end()) {
                const auto& source = unique[found->second];
                const auto base = reinterpret_cast<std::uintptr_t>(source.data<std::uint8_t>());
                const auto address = reinterpret_cast<std::uintptr_t>(value.data<std::uint8_t>());
                if (address >= base && address - base <= source.nbytes()
                    && value.nbytes() <= source.nbytes() - (address - base))
                    range.emplace(found->second, address - base);
            }
            if (!range) {
                const auto* data = value.data<std::uint8_t>();
                if (value.nbytes() <= 16 && std::all_of(data, data + value.nbytes(), [](auto v) { return v == 0; })) {
                    copies.front().push_back({std::numeric_limits<std::uint32_t>::max(), 0,
                        static_cast<std::uint32_t>(bytes), static_cast<std::uint32_t>(value.nbytes())});
                } else {
                    const auto address = reinterpret_cast<std::uintptr_t>(data);
                    const auto found = external.find(address);
                    if (found != external.end() && unique[found->second].nbytes() >= value.nbytes())
                        range.emplace(found->second, 0);
                    else {
                        const auto index = static_cast<std::uint32_t>(unique.size());
                        unique.push_back(value);
                        external[address] = index;
                        copies.resize((unique.size() + kPackedCopyInputs - 1) / kPackedCopyInputs);
                        range.emplace(index, 0);
                    }
                }
            }
            if (range) {
                const auto alignment = static_cast<std::uint32_t>(
                    reinterpret_cast<std::uintptr_t>(unique[range->first].data<std::uint8_t>()) & 15u) << 28;
                for (std::size_t offset = 0; offset < value.nbytes(); offset += tile)
                    copies[range->first / kPackedCopyInputs].push_back({range->first,
                        static_cast<std::uint32_t>(range->second + offset),
                        static_cast<std::uint32_t>(bytes + offset),
                        static_cast<std::uint32_t>(std::min(tile, value.nbytes() - offset)) | alignment});
            }
            bytes = next;
        }
        views.push_back({values, first, bytes - first});
    }
    if (!bytes) return std::nullopt;
    if (unique.empty()) unique.push_back(empty_packed_array(mlx::core::uint8));
    return PackedExpertArena{packed_expert_copy(std::move(unique), copies, bytes), std::move(views)};
}

struct MfePageResources {
    NS::SharedPtr<MTL::ResidencySet> residency;
    std::vector<array> owners;
};

struct NativeMoeConfig {
    Dtype dtype;
    Shape output_shape;
    int tokens = 0;
    int routes = 0;
    int experts = 0;
    int output_width = 0;
    int matrix_output_width = 0;
    int projections = 0;
    int fused_swiglu = 0;
    int input_width = 0;
    int k_lanes = 0;
    int rows_per_simd = 1;
    std::shared_ptr<const std::vector<int>> nint_group_sizes;
    int descriptor_size = 0;
    int variant_stride = 0;
    int shared_input = 0;
    int vq_execution_layout = 0;
    std::uint64_t vq_layouts = UINT64_MAX;
    int family_mask = 0;
    int vq_profile_mask = 0;
    int has_nepq_residual = 0;
    int npq_grouped_indices = 0;
    int sorted_routes = 0;
    int expert_map_size = 0;
    int packed_expert_ids = 0;
    bool route_groups = false;
    bool nint_q_dispatch = false;
    bool indirect_pages = false;
    bool dual_pages = false;
    bool weighted_reduce = false;
    int direct_page_mask = 0;
    int other_experts = 0;
    int workgroups = 0;
    std::shared_ptr<const MfePageResources> page_resources;
    std::shared_ptr<const MfePageResources> other_page_resources;
};

constexpr std::array<std::pair<const char*, const char*>, 21> kExpertStreams{{
    {"nint_q", "uchar"}, {"nint_sub_scale", "uchar"}, {"nint_sub_min", "uchar"},
    {"nint_anchor_scale", "float"}, {"nint_anchor_min", "float"},
    {"q8_q", "int8_t"}, {"q8_scales", "half"},
    {"vq_indices", "uchar"}, {"vq_state", "uchar"}, {"vq_aux", "uchar"},
    {"vq_anchors", "float"}, {"vq_codebooks", "int8_t"}, {"vq_scales", "float"},
    {"vq_state_to_codebank", "uchar"}, {"vq_banks", "uchar"}, {"vq_parameters", "float"},
    {"vq_residual_codebooks", "float"}, {"vq_residual_first", "short"},
    {"vq_residual_second", "short"}, {"mx_values", "uchar"}, {"mx_scales", "uchar"}
}};

bool uses_indirect_expert_pages(std::uint32_t families) {
    if (families == (1u << kFamilyMxfp4) || mlx_reference_enabled()) return false;
    if (__builtin_available(macOS 15, *))
        return mlx::core::metal::device(mlx::core::Device::gpu).mtl_device()->supportsFamily(MTL::GPUFamilyMetal3);
    return false;
}

void retain_indirect_resources(mlx::core::metal::CommandEncoder& encoder, std::span<const array> sources) {
    struct Residency {
        NS::SharedPtr<MTL::ResidencySet> set;
        std::unordered_set<const MTL::Buffer*> buffers;
    };
    static thread_local std::unordered_map<MTL::CommandBuffer*, std::weak_ptr<Residency>> pending;
    auto* command = encoder.get_command_buffer();
    auto state = pending[command].lock();
    if (!state) {
        for (auto it = pending.begin(); it != pending.end();)
            if (it->second.expired()) it = pending.erase(it); else ++it;
        state = std::make_shared<Residency>();
        auto descriptor = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
        descriptor->setInitialCapacity(sources.size());
        NS::Error* error = nullptr;
        auto& device = mlx::core::metal::device(mlx::core::Device::gpu);
        state->set = NS::TransferPtr(device.mtl_device()->newResidencySet(descriptor.get(), &error));
        if (!state->set) throw std::runtime_error(error
            ? error->localizedDescription()->utf8String() : "expert residency set unavailable");
        pending[command] = state;
        command->useResidencySet(state->set.get());
        command->addCompletedHandler([state](MTL::CommandBuffer*) {});
    }
    bool changed = false;
    for (const auto& source : sources) {
        auto* buffer = static_cast<const MTL::Buffer*>(source.buffer().ptr());
        if (state->buffers.insert(buffer).second) {
            state->set->addAllocation(buffer);
            encoder.add_temporary(source);
            changed = true;
        }
    }
    if (changed) state->set->commit();
}

struct Mxfp4DecodeReduceConfig {
    Dtype dtype;
    Shape output_shape;
    int tokens = 0;
    int routes = 0;
    int experts = 0;
    int output_width = 0;
    int input_width = 0;
    int descriptor_size = 0;
    int rows_per_lane_group = 2;
    int workgroups = 0;
};

struct Mxfp4PairSwiGluConfig {
    Dtype dtype;
    Shape output_shape;
    int tokens = 0;
    int routes = 0;
    int experts = 0;
    int output_width = 0;
    int input_width = 0;
    int descriptor_size = 0;
    int workgroups = 0;
};

struct GroupedMmqConfig {
    Shape output_shape;
    int route_count = 0;
    int max_blocks = 0;
    int block_rows = 32;
    int tile_columns = 64;
    int tokens = 0;
    int routes = 0;
    int experts = 0;
    int output_width = 0;
    int matrix_output_width = 0;
    int projections = 1;
    int input_width = 0;
    int descriptor_size = 0;
    int variant_stride = 0;
    int shared_input = 0;
    int input_sorted = 0;
    int fused_swiglu = 0;
    int has_nepq_residual = 0;
    int family_mask = 127;
    int vq_profile_mask = 0;
    std::shared_ptr<const std::vector<int>> nint_group_sizes;
    std::shared_ptr<const std::vector<std::tuple<int, int, int>>> cohort_populations;
    bool use_nax = false;
    bool direct_nax = false;
    bool indirect_pages = false;
    float swiglu_limit = 0.0f;
};

struct GroupedMmqParameters {
    int route_count = 0;
    int tokens = 0;
    int routes = 0;
    int experts = 0;
    int output_width = 0;
    int matrix_output_width = 0;
    int projections = 1;
    int input_width = 0;
    int descriptor_size = 0;
    int variant_stride = 0;
    int shared_input = 0;
    int input_sorted = 0;
    float swiglu_limit = 0.0f;
};

static_assert(sizeof(GroupedMmqParameters) == 52);

struct Mxfp4BlocksConfig {
    Shape output_shape;
    int route_count = 0;
    int max_blocks = 0;
    int experts = 0;
    int output_width = 0;
    int input_width = 0;
    int block_rows = 32;
    bool paired = false;
};

// Adapted from oMLX's DeepSeek-V4 block-list builder.  The expert IDs have
// already been sorted, so one GPU thread can find each expert's contiguous
// range and split it into independently schedulable BM-row blocks.
const mlx::core::fast::CustomKernelFunction&
grouped_mmq_block_builder(bool cohort_ordered) {
    const auto make_builder = [](bool ordered) {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        std::vector<std::string> inputs{"indices"};
        if (ordered) inputs.push_back("expert_order");
        std::string source = ordered
            ? "#define TYLOINFER_SCHEDULED_EXPERT(index) expert_order[index]\n"
            : "#define TYLOINFER_SCHEDULED_EXPERT(index) int(index)\n";
        source +=
            R"METAL(
                const int M = indices_shape[0];
                const int NUM_EXPERTS = int(threads_per_threadgroup.x);
                const int MAX_BLOCKS = min(max(M, 1), M / BM + int(M % BM != 0) + NUM_EXPERTS);
                const uint schedule_index = thread_index_in_threadgroup;
                const uint expert = uint(
                    TYLOINFER_SCHEDULED_EXPERT(schedule_index));

                threadgroup atomic_int local_count;
                threadgroup int expert_starts[1024];
                threadgroup int expert_rows[1024];
                threadgroup int use_block_chunks;
                if constexpr (BLOCK_CHUNK == 0) {
                    if (expert == 0) {
                        atomic_store_explicit(
                            &local_count,
                            0,
                            memory_order_relaxed);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                int lo = 0;
                int hi = M;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (indices[mid] < int(expert)) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                const int start = lo;

                hi = M;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (indices[mid] <= int(expert)) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                const int end = lo;

                if constexpr (BLOCK_CHUNK == 0) {
                    for (int row = start; row < end; row += BM) {
                        const int rows = min(BM, end - row);
                        const int slot = atomic_fetch_add_explicit(
                            &local_count,
                            1,
                            memory_order_relaxed);
                        if (slot < MAX_BLOCKS) {
                            block_meta[slot * 3 + 0] = row;
                            block_meta[slot * 3 + 1] = int(expert);
                            block_meta[slot * 3 + 2] = rows;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (expert == 0) {
                        block_count[0] = atomic_load_explicit(
                            &local_count,
                            memory_order_relaxed);
                    }
                } else {
                    expert_starts[schedule_index] = start;
                    expert_rows[schedule_index] = end - start;
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    if (expert == 0) {
                        int active_experts = 0;
                        for (uint item = 0; item < NUM_EXPERTS; ++item) {
                            active_experts += expert_rows[item] > 0 ? 1 : 0;
                        }
                        // A heavily skewed route set already reuses a small
                        // working set and benefits from maximum interleave.
                        // Chunk only when at least half the pool is active.
                        use_block_chunks =
                            active_experts * 2 >= int(NUM_EXPERTS) ? 1 : 0;
                        if (use_block_chunks == 0) {
                            atomic_store_explicit(
                                &local_count,
                                0,
                                memory_order_relaxed);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    if (use_block_chunks == 0) {
                        for (int row = start; row < end; row += BM) {
                            const int rows = min(BM, end - row);
                            const int slot = atomic_fetch_add_explicit(
                                &local_count,
                                1,
                                memory_order_relaxed);
                            if (slot < MAX_BLOCKS) {
                                block_meta[slot * 3 + 0] = row;
                                block_meta[slot * 3 + 1] = int(expert);
                                block_meta[slot * 3 + 2] = rows;
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        if (expert == 0) {
                            block_count[0] = atomic_load_explicit(
                                &local_count,
                                memory_order_relaxed);
                        }
                    } else if (expert == 0) {
                        int maximum_blocks = 0;
                        for (uint item = 0; item < NUM_EXPERTS; ++item) {
                            maximum_blocks = max(
                                maximum_blocks,
                                (expert_rows[item] + BM - 1) / BM);
                        }
                        int slot = 0;
                        for (int first = 0;
                             first < maximum_blocks;
                             first += BLOCK_CHUNK) {
                            for (uint item = 0; item < NUM_EXPERTS; ++item) {
                                const int scheduled_expert = int(
                                    TYLOINFER_SCHEDULED_EXPERT(item));
                                int count =
                                    (expert_rows[item] + BM - 1) / BM;
                                for (int local = 0;
                                     local < BLOCK_CHUNK
                                         && first + local < count;
                                     ++local) {
                                    int row = expert_starts[item]
                                        + (first + local) * BM;
                                    if (slot < MAX_BLOCKS) {
                                        block_meta[slot * 3 + 0] = row;
                                        block_meta[slot * 3 + 1] =
                                            scheduled_expert;
                                        block_meta[slot * 3 + 2] = min(
                                            BM,
                                            expert_starts[item]
                                                + expert_rows[item] - row);
                                    }
                                    ++slot;
                                }
                            }
                        }
                        block_count[0] = slot;
                    }
                }
            )METAL";
        return mlx::core::fast::metal_kernel(
            ordered
                ? "mfq_grouped_mmq_block_builder_cohort"
                : "mfq_grouped_mmq_block_builder",
            std::move(inputs),
            {"block_meta", "block_count"},
            std::move(source),
            "",
            true,
            false,
            options);
    };
    static const auto unordered_builder = make_builder(false);
    static const auto ordered_builder = make_builder(true);
    return cohort_ordered ? ordered_builder : unordered_builder;
}

MlxGroupedMmqPlan make_grouped_mmq_plan(
    const array& expert_ids,
    const array& route_order_value,
    const array& expert_order,
    int experts,
    int block_rows = 32) {
    if (block_rows != 32 && block_rows != 48 && block_rows != 64
        && block_rows != 80 && block_rows != 96 && block_rows != 128) {
        throw std::invalid_argument(
            "grouped MMQ block rows must be 32, 48, 64, 80, 96, or 128");
    }
    if (experts <= 0 || experts > 1024) {
        throw std::invalid_argument(
            "grouped MMQ block builder requires 1..1024 experts; received " +
            std::to_string(experts));
    }
    if (
        expert_order.dtype() != mlx::core::int32
        || expert_order.ndim() != 1
        || expert_order.size() != static_cast<std::size_t>(experts)
    ) {
        throw std::invalid_argument(
            "grouped MMQ expert order must contain every expert");
    }
    auto ids = mlx::core::contiguous(
        mlx::core::astype(expert_ids, mlx::core::int32));
    if (ids.ndim() != 2) {
        throw std::invalid_argument(
            "routed expert IDs must have [tokens,routes] shape");
    }
    const int route_count = checked_int(
        checked_product(
            static_cast<std::size_t>(ids.shape(0)),
            static_cast<std::size_t>(ids.shape(1)),
            "route count"),
        "route count");
    auto route_order = mlx::core::contiguous(
        mlx::core::astype(route_order_value, mlx::core::int32));
    if (
        route_order.ndim() != 1
        || route_order.size()
            != static_cast<std::size_t>(route_count)
    ) {
        throw std::invalid_argument(
            "route order must contain one index per routed row");
    }
    const auto population_bound =
        (static_cast<std::size_t>(route_count) + block_rows - 1) / block_rows
        + static_cast<std::size_t>(experts);
    const int max_blocks = checked_int(
        std::min(static_cast<std::size_t>(std::max(route_count, 1)), population_bound),
        "grouped MMQ max blocks");
    auto sorted_ids = mlx::core::contiguous(
        mlx::core::take(
            mlx::core::reshape(ids, Shape{route_count}),
            route_order,
            0));
    const bool cohort_ordered = experts >= 128
        && route_count > experts * 96;
    const int block_chunk = cohort_ordered ? 2 : 0;
    std::vector<array> builder_inputs{std::move(sorted_ids)};
    if (cohort_ordered) builder_inputs.push_back(expert_order);
    auto outputs = grouped_mmq_block_builder(cohort_ordered)(
        std::move(builder_inputs),
        {
            Shape{max_blocks, 3},
            Shape{1},
        },
        {
            mlx::core::int32,
            mlx::core::int32,
        },
        {experts, 1, 1},
        {experts, 1, 1},
        {
            {"BM", block_rows},
            {
                "BLOCK_CHUNK",
                // Above 96 mean routes, pair adjacent blocks in deterministic
                // cohort order to retain packed-weight locality without
                // serializing an entire expert. Smaller pools and route sets
                // keep the original parallel atomic path.
                block_chunk,
            },
        },
        std::nullopt,
        false,
        {});
    return MlxGroupedMmqPlan{
        .block_meta = std::move(outputs.at(0)),
        .block_count = std::move(outputs.at(1)),
        .max_blocks = max_blocks,
        .block_rows = block_rows,
        .route_count = route_count,
        .experts = experts,
    };
}

std::string nint_group_specialization(
    const std::shared_ptr<const std::vector<int>>& group_sizes) {
    if (!group_sizes || group_sizes->empty()) return {};
    std::ostringstream source;
    source << "#define TYLOINFER_MFE_NINT_DISPATCH_WITH(CALL) ";
    for (std::size_t index = 0; index < group_sizes->size(); ++index) {
        const auto group_size = (*group_sizes)[index];
        source << (index == 0 ? "if" : "else if")
            << " (group_size == " << group_size << "u) { CALL("
            << group_size << "u); } ";
    }
    source << "else\n";
    return source.str();
}

void append_nint_group_key(
    std::ostringstream& name,
    const std::shared_ptr<const std::vector<int>>& group_sizes) {
    if (group_sizes)
        for (const auto group_size : *group_sizes) name << '_' << group_size;
}

std::string native_moe_kernel_name(
    const NativeMoeConfig& config) {
    std::ostringstream name;
    name
        << "mfq_native_mfe_v16_"
        << (config.dtype == mlx::core::float16 ? "f16" : "f32")
        << (config.tokens == 1 ? "_decode" : "_batch")
        << "_r" << config.routes
        << "_o" << config.output_width
        << "_mo" << config.matrix_output_width
        << "_p" << config.projections
        << "_sw" << config.fused_swiglu
        << "_k" << config.input_width
        << "_kl" << config.k_lanes
        << "_rs" << config.rows_per_simd
        << "_si" << config.shared_input
        << "_vl" << config.vq_execution_layout
        << "_vls" << config.vq_layouts
        << "_fm" << config.family_mask
        << "_vm" << config.vq_profile_mask
        << "_nr" << config.has_nepq_residual
        << "_ni" << config.npq_grouped_indices;
    name << "_sr" << config.sorted_routes;
    name << "_em" << (config.expert_map_size != 0);
    name << "_pe" << config.packed_expert_ids;
    name << "_rg" << config.route_groups;
    name << "_qd" << config.nint_q_dispatch;
    if (config.indirect_pages) name << "_page_addresses";
    if (config.dual_pages) name << "_dual" << config.direct_page_mask;
    if (config.weighted_reduce) name << "_reduce_v1";
    if (config.projections == 2 && config.fused_swiglu
        && (config.family_mask & (1 << kFamilyMxfp4)) && (config.family_mask & (1 << kFamilyVq)))
        name << "_mixed_pair_v2";
    if (config.route_groups) name << "_pairq";
    name << "_ng";
    append_nint_group_key(name, config.nint_group_sizes);
    return name.str();
}

std::string make_mfe_reduce_body();

std::string make_native_moe_source(
    const NativeMoeConfig& config,
    const std::string& kernel_name) {
    const bool has_expert_map = config.expert_map_size != 0 && !config.dual_pages;
    std::ostringstream source;
    source
        << "#include <metal_stdlib>\n"
        << "using namespace metal;\n"
        << "using T = "
        << (config.dtype == mlx::core::float16 ? "half" : "float")
        << ";\n"
        << nint_group_specialization(config.nint_group_sizes)
        << "#define TYLOINFER_MOE_ROUTE_GROUPS " << config.route_groups << "\n"
        << "#define TYLOINFER_MOE_DUAL_PAGES " << config.dual_pages << "\n"
        << "#define TYLOINFER_NINT_Q_DISPATCH " << config.nint_q_dispatch << "\n"
        << "#define TYLOINFER_MOE_VQ_LAYOUTS " << config.vq_layouts << "ul\n"
        << kMoeHeader;
    if (config.indirect_pages) {
        source << "struct MfqExpertStreams {\n";
        for (const auto& [name, type] : kExpertStreams)
            source << "device const " << type << "* " << name << ";\n";
        source << "};\n";
    }
    source << "\nkernel void " << kernel_name << "(\n"
        << (config.dual_pages ? "device const int* gate_descriptors [[buffer(0)]],\n"
            : "device const int* descriptors [[buffer(0)]],\n")
        << (config.dual_pages ? "device const MfqExpertStreams* gate_streams [[buffer(1)]],\n"
            : config.indirect_pages ? "device const MfqExpertStreams* expert_streams [[buffer(1)]],\n"
            : "device const uchar* nint_q [[buffer(1)]],\n")
        << (config.dual_pages ? "device const int* up_descriptors [[buffer(2)]],\n"
            : "device const uchar* nint_sub_scale [[buffer(2)]],\n")
        << (config.dual_pages ? "device const MfqExpertStreams* up_streams [[buffer(3)]],\n"
            : "device const uchar* nint_sub_min [[buffer(3)]],\n")
        << (config.dual_pages ? "device const int* gate_map [[buffer(4)]],\n"
            : "device const float* nint_anchor_scale [[buffer(4)]],\n")
        << (config.dual_pages ? "device const int* up_map [[buffer(5)]],\n"
            : "device const float* nint_anchor_min [[buffer(5)]],\n")
        << (config.dual_pages ? "device const T* up_x [[buffer(6)]],\n"
            : "device const int8_t* q8_q [[buffer(6)]],\n")
        << "device const half* q8_scales [[buffer(7)]],\n"
        << "device const uchar* vq_indices [[buffer(8)]],\n"
        << "device const uchar* vq_state [[buffer(9)]],\n"
        << "device const uchar* vq_aux [[buffer(10)]],\n"
        << "device const float* vq_anchors [[buffer(11)]],\n"
        << "device const int8_t* vq_codebooks [[buffer(12)]],\n"
        << "device const float* vq_scales [[buffer(13)]],\n"
        << "device const uchar* vq_state_to_codebank [[buffer(14)]],\n"
        << "device const uchar* vq_banks [[buffer(15)]],\n"
        << "device const float* vq_parameters [[buffer(16)]],\n"
        << "device const float* vq_residual_codebooks [[buffer(17)]],\n"
        << "device const short* vq_residual_first [[buffer(18)]],\n"
        << "device const short* vq_residual_second [[buffer(19)]],\n"
        << "device const uchar* mx_values [[buffer(20)]],\n"
        << "device const uchar* mx_scales [[buffer(21)]],\n"
        << (config.dual_pages ? "device const T* gate_x [[buffer(22)]],\n"
            : "device const T* x [[buffer(22)]],\n")
        << "device const int* expert_ids [[buffer(23)]],\n"
        << "device const int* route_order [[buffer(24)]],\n"
        << "device const float* params [[buffer(25)]],\n";
    if (has_expert_map) {
        source << "device const int* expert_map [[buffer(26)]],\n";
    }
    const int input_count = 26 + static_cast<int>(has_expert_map) + static_cast<int>(config.route_groups)
        + static_cast<int>(config.weighted_reduce);
    if (config.route_groups)
        source << "device const int* route_groups [[buffer(" << input_count - 1 << ")]],\n";
    if (config.weighted_reduce)
        source << "device const float* route_weights [[buffer(" << input_count - 1 << ")]],\n";
    source
        << "device T* y [[buffer("
        << input_count
        << ")]],\n"
        << "constant int* geometry [[buffer("
        << input_count + 1
        << ")]],\n"
        << "uint thread_index_in_simdgroup "
           "[[thread_index_in_simdgroup]],\n"
        << "uint simdgroup_index_in_threadgroup "
           "[[simdgroup_index_in_threadgroup]],\n"
        << "uint3 threadgroup_position_in_grid "
           "[[threadgroup_position_in_grid]]) {\n"
        << (config.tokens == 1
            ? "constexpr int TOKENS = 1;\n"
            : "const int TOKENS = geometry[0];\n")
        << "constexpr int ROUTES = " << config.routes << ";\n"
        << "const int EXPERTS = geometry[2];\n"
        << "constexpr int OUT = " << config.output_width << ";\n"
        << "constexpr int MATRIX_OUT = "
        << config.matrix_output_width << ";\n"
        << "constexpr int PROJECTIONS = "
        << config.projections << ";\n"
        << "constexpr int FUSED_SWIGLU = "
        << config.fused_swiglu << ";\n"
        << "constexpr int K = " << config.input_width << ";\n"
        << "constexpr int K_LANES_VALUE = "
        << config.k_lanes << ";\n"
        << "constexpr int ROWS_PER_SIMD_VALUE = "
        << config.rows_per_simd << ";\n"
        << "constexpr int DESCRIPTOR_SIZE = "
        << config.descriptor_size << ";\n"
        << (config.tokens == 1
            ? "constexpr int VARIANT_STRIDE = " + std::to_string(config.variant_stride) + ";\n"
            : "const int VARIANT_STRIDE = geometry[1];\n")
        << "constexpr int SHARED_INPUT = "
        << config.shared_input << ";\n"
        << "constexpr int VQ_EXECUTION_LAYOUT = "
        << config.vq_execution_layout << ";\n"
        << "constexpr int FAMILY_MASK = "
        << config.family_mask << ";\n"
        << "constexpr int VQ_PROFILE_MASK = "
        << config.vq_profile_mask << ";\n"
        << "constexpr int HAS_NEPQ_RESIDUAL = "
        << config.has_nepq_residual << ";\n"
        << "constexpr int NPQ_GROUPED_INDICES = "
        << config.npq_grouped_indices << ";\n"
        << "constexpr int SORTED_ROUTES = "
        << config.sorted_routes << ";\n"
        << "const int EXPERT_MAP_SIZE = geometry[3];\n"
        << "constexpr int PACKED_EXPERT_IDS = "
        << config.packed_expert_ids << ";\n";
    source << "#define TYLOINFER_MOE_OTHER_ROW "
        << (config.dual_pages ? "uint(other_expert)" : "(uint(expert) * 2u + (projection ^ 1u))") << "\n"
        << "#define TYLOINFER_MOE_OTHER_DESCRIPTORS "
        << (config.dual_pages ? "(projection == 0u ? up_descriptors : gate_descriptors)" : "descriptors") << "\n";
    if (!has_expert_map) {
        source << "device const int* expert_map = expert_ids;\n";
    }
    std::string body = config.weighted_reduce ? make_mfe_reduce_body() : kMoeSource;
    if (config.indirect_pages) {
        const std::string marker = "    uint descriptor_base = (\n"
            "        uint(expert) * uint(PROJECTIONS)\n"
            "        + projection\n"
            "    ) * uint(DESCRIPTOR_SIZE);";
        const auto position = body.find(marker);
        if (position == std::string::npos) throw std::logic_error("MFE page binding marker is missing");
        if (config.dual_pages) body.replace(position, marker.size(),
            "    uint descriptor_base = uint(expert) * uint(DESCRIPTOR_SIZE);");
        std::ostringstream bindings;
        if (config.dual_pages) {
            bindings << "device const int* descriptors = projection == 0u ? gate_descriptors : up_descriptors;\n"
                << "device const T* x = projection == 0u ? gate_x : up_x;\n"
                << "device const MfqExpertStreams* selected_stream = projection == 0u ? gate_streams + "
                << ((config.direct_page_mask & 1) ? "0u" : "uint(expert)") << " : up_streams + "
                << ((config.direct_page_mask & 2) ? "0u" : "uint(expert)") << ";\n";
            for (const auto& [name, type] : kExpertStreams)
                bindings << "#define " << name << " selected_stream->" << name << "\n";
        } else for (const auto& [name, type] : kExpertStreams)
            bindings << "#define " << name << " expert_streams[uint(expert)]." << name << "\n";
        body.insert(position, bindings.str());
    }
    source << body << "\n}\n";
    return source.str();
}

class NativeMfePrimitive final
    : public mlx::core::UnaryPrimitive, public tyloi::metal::PreparableKernel {
public:
    NativeMfePrimitive(
        mlx::core::Stream stream,
        NativeMoeConfig config)
        : UnaryPrimitive(stream),
          config_(std::move(config)),
          kernel_name_(native_moe_kernel_name(config_)) {}

    std::string preparation_key() const override { return kernel_name_; }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& selected_stream = stream();
        auto& device = tyloi::metal::kernel_device(
            selected_stream.device);
        CompileOptions compile_options;
        compile_options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            kernel_name_,
            compile_options,
            [config = config_, name = kernel_name_] {
                return make_native_moe_source(
                    config,
                    name);
            });
        return device.get_kernel(kernel_name_, library);
    }

    void eval_cpu(
        const std::vector<array>&,
        array&) override {
        throw std::runtime_error(
            "native MFE primitive has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        const int input_count = 26 + static_cast<int>(config_.expert_map_size != 0 && !config_.dual_pages)
            + static_cast<int>(config_.route_groups) + static_cast<int>(config_.weighted_reduce);
        if (inputs.size() < static_cast<std::size_t>(input_count)
            || (!config_.indirect_pages && inputs.size() != static_cast<std::size_t>(input_count))) {
            throw std::logic_error(
                "native MFE primitive input count mismatch");
        }
        output.set_data(
            mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto* kernel = prepared_kernel();
        auto& encoder =
            mlx::core::metal::get_command_encoder(
                selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        if (config_.indirect_pages) {
            if (config_.page_resources && (!config_.dual_pages || config_.other_page_resources)) {
                auto* command = encoder.get_command_buffer();
                command->useResidencySet(config_.page_resources->residency.get());
                command->addCompletedHandler([resources = config_.page_resources](MTL::CommandBuffer*) {});
                if (config_.other_page_resources) {
                    command->useResidencySet(config_.other_page_resources->residency.get());
                    command->addCompletedHandler([resources = config_.other_page_resources](MTL::CommandBuffer*) {});
                }
            } else {
                for (std::size_t index = input_count; index < inputs.size(); ++index)
                    encoder.set_input_array(inputs[index], 1);
                retain_indirect_resources(encoder, std::span(inputs).subspan(input_count));
            }
        }
        for (int index = 0; index < input_count; ++index) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(index)],
                index);
        }
        encoder.set_output_array(output, input_count);
        const std::array<int, 5> geometry{
            config_.tokens, config_.variant_stride, config_.experts, config_.expert_map_size, config_.other_experts};
        encoder.set_bytes(geometry, input_count + 1);
        encoder.dispatch_threadgroups(
            MTL::Size(config_.workgroups, 1, 1),
            MTL::Size(config_.weighted_reduce ? config_.routes * 32 : 64, 1, 1));
        if (config_.indirect_pages) indirect_expert_submissions.fetch_add(1, std::memory_order_relaxed);
    }

    const char* name() const override {
        return "NativeMfePrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const NativeMfePrimitive*>(&other);
        return primitive != nullptr
            && primitive->kernel_name_ == kernel_name_
            && primitive->config_.tokens == config_.tokens
            && primitive->config_.variant_stride == config_.variant_stride
            && primitive->config_.experts == config_.experts
            && primitive->config_.other_experts == config_.other_experts
            && primitive->config_.expert_map_size == config_.expert_map_size
            && primitive->config_.workgroups == config_.workgroups
            && primitive->config_.page_resources == config_.page_resources
            && primitive->config_.other_page_resources == config_.other_page_resources;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {config_.output_shape};
    }

private:
    NativeMoeConfig config_;
    std::string kernel_name_;
};

array native_moe_dispatch(
    std::vector<array> inputs,
    NativeMoeConfig config) {
    auto stream = mlx::core::default_stream(
        mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "native MFE primitive requires the Metal device");
    }
    auto shape = config.output_shape;
    auto dtype = config.dtype;
    return array(
        std::move(shape),
        dtype,
        std::make_shared<NativeMfePrimitive>(
            stream,
            std::move(config)),
        std::move(inputs));
}

// Adapted from oMLX's two-launch Qwen routed-decode topology (Apache-2.0):
// the first expert launch runs routed and shared Gate/Up with their SwiGLU
// epilogues plus the shared gate; the second assigns one SIMD group to every
// routed expert and one to the shared expert, then combines in-threadgroup.
// The one-token launch uses oMLX's three-candidate lane cache for folded top-k.

struct MfeNintDecodeConfig {
    Dtype dtype;
    Dtype shared_gate_dtype;
    bool shared_gate_is_logits = false;
    int shared_router_group_size = 0;
    bool packed_gate_up = false;
    bool route_groups = false;
    bool folded_topk = false;
    bool gate_q_dispatch = false;
    bool down_q_dispatch = false;
    bool shared_gate_q_dispatch = false;
    bool shared_down_q_dispatch = false;
    Shape stage_one_shape;
    Shape output_shape;
    int tokens = 1;
    int routes = 0;
    int experts = 0;
    int hidden = 0;
    int intermediate = 0;
    int descriptor_size = 0;
    int gate_family_mask = 0;
    int gate_vq_profile_mask = 0;
    int gate_vq_execution_layout = 0;
    std::uint64_t gate_vq_layouts = UINT64_MAX;
    int gate_has_nepq_residual = 0;
    int gate_npq_grouped_indices = 0;
    std::shared_ptr<const std::vector<int>> gate_nint_group_sizes;
    int down_family_mask = 0;
    int down_vq_profile_mask = 0;
    int down_vq_execution_layout = 0;
    std::uint64_t down_vq_layouts = UINT64_MAX;
    int down_has_nepq_residual = 0;
    int down_npq_grouped_indices = 0;
    std::shared_ptr<const std::vector<int>> down_nint_group_sizes;
    int shared_gate_group_size = 0;
    int shared_gate_groups = 0;
    int shared_up_group_size = 0;
    int shared_up_groups = 0;
    int shared_down_group_size = 0;
    int shared_down_groups = 0;
    int k_lanes = 16;
    int down_k_lanes = 16;
    int down_rows_per_simd = 1;
    int stage_one_workgroups = 0;
    int stage_two_workgroups = 0;
};

bool mfe_paired_nint_down(const MfeNintDecodeConfig& config) {
    return config.tokens == 1 && config.down_k_lanes == 16 &&
        config.down_rows_per_simd == 2 && config.hidden > config.intermediate;
}

std::string metal_activation_type(Dtype dtype) {
    if (dtype == mlx::core::float16) return "half";
    if (dtype == mlx::core::bfloat16) return "bfloat";
    if (dtype == mlx::core::float32) return "float";
    throw std::invalid_argument("unsupported fused MoE activation dtype");
}

std::string mfe_decode_kernel_name(
    const MfeNintDecodeConfig& config,
    int stage) {
    std::string name;
    name.reserve(256);
    name.append("mfq_mfe_nint_decode_stage");
    const auto integer = [&](int value) {
        char buffer[12];
        const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
        name.append(buffer, result.ptr);
    };
    integer(stage);
    if (stage == 1 && config.folded_topk) {
        name.append("_gp");
        integer(kMfeFoldedGateSimdPairs);
    }
    for (const auto& field : std::array<std::pair<const char*, int>, 15>{{
             {"_r", config.routes}, {"_e", config.experts},
             {"_h", config.hidden}, {"_i", config.intermediate},
             {"_gf", config.gate_family_mask}, {"_gv", config.gate_vq_profile_mask},
             {"_df", config.down_family_mask}, {"_dv", config.down_vq_profile_mask},
             {"_ggs", config.shared_gate_group_size}, {"_gg", config.shared_gate_groups},
             {"_dgs", config.shared_down_group_size}, {"_dg", config.shared_down_groups},
             {"_kl", stage == 1 ? config.k_lanes : config.down_k_lanes}, {"_dr", config.down_rows_per_simd},
             {"_gt", static_cast<int>((stage == 2 && config.shared_router_group_size
                 ? mlx::core::float16 : config.shared_gate_dtype).val())}}}) {
        name.append(field.first);
        integer(field.second);
    }
    name.append("_gng");
    if (config.gate_nint_group_sizes)
        for (const auto value : *config.gate_nint_group_sizes) {
            name.push_back('_'); integer(value);
        }
    name.append("_dng");
    if (config.down_nint_group_sizes)
        for (const auto value : *config.down_nint_group_sizes) {
            name.push_back('_'); integer(value);
        }
    if (config.tokens > 1) name.append("_batch");
    name.append("_vls").append(std::to_string(stage == 1 ? config.gate_vq_layouts : config.down_vq_layouts));
    if (stage == 1 && config.route_groups) name.append("_rg1");
    if (config.shared_gate_is_logits) name.append("_gate_logits");
    if (stage == 1 && config.shared_router_group_size)
        name.append("_srg").append(std::to_string(config.shared_router_group_size));
    if (stage == 1 && config.shared_up_group_size != config.shared_gate_group_size)
        name.append("_ugs").append(std::to_string(config.shared_up_group_size))
            .append("_ug").append(std::to_string(config.shared_up_groups));
    if (config.packed_gate_up) name.append("_packed_gu");
    if (config.folded_topk) name.append("_folded_topk");
    if (stage == 2 && mfe_paired_nint_down(config)) name.append("_nint_pair");
    name.append("_rowq_auto");
    name.append(stage == 1 ? (config.gate_q_dispatch ? "1" : "0")
                           : (config.down_q_dispatch ? "1" : "0"));
    name.append(stage == 1 ? (config.shared_gate_q_dispatch ? "s1" : "s0")
                           : (config.shared_down_q_dispatch ? "s1" : "s0"));
    if (stage == 1 && config.route_groups) name.append("_pairq");
    return name;
}

void append_mfe_pool_arguments(std::ostringstream& source) {
    source
        << "device const int* descriptors [[buffer(0)]],\n"
        << "device const uchar* nint_q [[buffer(1)]],\n"
        << "device const uchar* nint_sub_scale [[buffer(2)]],\n"
        << "device const uchar* nint_sub_min [[buffer(3)]],\n"
        << "device const float* nint_anchor_scale [[buffer(4)]],\n"
        << "device const float* nint_anchor_min [[buffer(5)]],\n"
        << "device const int8_t* q8_q [[buffer(6)]],\n"
        << "device const half* q8_scales [[buffer(7)]],\n"
        << "device const uchar* vq_indices [[buffer(8)]],\n"
        << "device const uchar* vq_state [[buffer(9)]],\n"
        << "device const uchar* vq_aux [[buffer(10)]],\n"
        << "device const float* vq_anchors [[buffer(11)]],\n"
        << "device const int8_t* vq_codebooks [[buffer(12)]],\n"
        << "device const float* vq_scales [[buffer(13)]],\n"
        << "device const uchar* vq_state_to_codebank [[buffer(14)]],\n"
        << "device const uchar* vq_banks [[buffer(15)]],\n"
        << "device const float* vq_parameters [[buffer(16)]],\n"
        << "device const float* vq_residual_codebooks [[buffer(17)]],\n"
        << "device const short* vq_residual_first [[buffer(18)]],\n"
        << "device const short* vq_residual_second [[buffer(19)]],\n"
        << "device const uchar* mx_values [[buffer(20)]],\n"
        << "device const uchar* mx_scales [[buffer(21)]],\n";
}

std::string_view mfe_decode_body() {
    static const std::string body = [] {
        const std::string_view source(kMoeSource);
        const auto begin = source.find("    uint descriptor_base = (\n");
        const auto end = source.find("    if (SPLIT_SWIGLU) {\n", begin);
        if (begin == std::string_view::npos || end == std::string_view::npos) {
            throw std::logic_error("MFE decoder body markers are missing");
        }
        std::string body(source.substr(begin, end - begin));
        const auto routes_begin = body.find("    threadgroup uint grouped_route_rows[6];\n");
        const auto routes_end = body.find("    if (\n        (uint(FAMILY_MASK) & 1u)", routes_begin);
        if (routes_begin == std::string::npos || routes_end == std::string::npos)
            throw std::logic_error("MFE route grouping markers are missing");
        body.erase(routes_begin, routes_end - routes_begin);
        return body;
    }();
    return body;
}

std::string make_mfe_reduce_body() {
    std::string body = R"METAL(
    constexpr uint K_LANES = uint(K_LANES_VALUE);
    constexpr uint ROWS_PER_SIMD = uint(ROWS_PER_SIMD_VALUE);
    constexpr uint LANE_GROUPS = 32u / K_LANES;
    constexpr uint ROWS_PER_PHYSICAL_SIMD = LANE_GROUPS * ROWS_PER_SIMD;
    constexpr uint MATRIX_ROWS = ROWS_PER_SIMD;
    constexpr bool SPLIT_SWIGLU = false;
    constexpr bool PACKED_SWIGLU = false;
    constexpr uint OUTPUT_TILES = (uint(OUT) + ROWS_PER_PHYSICAL_SIMD - 1u) / ROWS_PER_PHYSICAL_SIMD;
    const uint lane = thread_index_in_simdgroup;
    const uint k_lane = lane & (K_LANES - 1u);
    const uint lane_group = lane / K_LANES;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint workgroup = threadgroup_position_in_grid.x;
    const uint output_tile = workgroup % OUTPUT_TILES;
    const uint token = workgroup / OUTPUT_TILES;
    const uint route = simd_group;
    const uint route_index = token * uint(ROUTES) + route;
    const uint projection = 0u;
    const uint output_base = output_tile * ROWS_PER_PHYSICAL_SIMD + lane_group * ROWS_PER_SIMD;
    threadgroup T partials[uint(ROUTES) * ROWS_PER_PHYSICAL_SIMD];
    const int logical_expert = expert_ids[route_index];
    int expert = logical_expert;
    if (PACKED_EXPERT_IDS != 0) expert = (logical_expert >> 8) - 1;
    else if (EXPERT_MAP_SIZE != 0) expert = logical_expert >= 0 && logical_expert < EXPERT_MAP_SIZE
        ? expert_map[logical_expert] : -1;
    if (expert >= 0 && expert < EXPERTS) {
)METAL";
    body += mfe_decode_body();
    body += R"METAL(
        for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
            float total = accumulators[row];
            for (uint offset = K_LANES >> 1u; offset > 0u; offset >>= 1u)
                total += simd_shuffle_down(total, offset);
            if (k_lane == 0u) partials[route * ROWS_PER_PHYSICAL_SIMD
                + lane_group * ROWS_PER_SIMD + row] = T(total);
        }
    } else if (k_lane == 0u) {
        for (uint row = 0u; row < ROWS_PER_SIMD; ++row)
            partials[route * ROWS_PER_PHYSICAL_SIMD + lane_group * ROWS_PER_SIMD + row] = T(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u && lane < ROWS_PER_PHYSICAL_SIMD) {
        float total = 0.0f;
        for (uint selected = 0u; selected < uint(ROUTES); ++selected)
            total += float(partials[selected * ROWS_PER_PHYSICAL_SIMD + lane])
                * route_weights[token * uint(ROUTES) + selected];
        const uint output = output_tile * ROWS_PER_PHYSICAL_SIMD + lane;
        if (output < uint(OUT)) y[token * uint(OUT) + output] = T(total);
    }
)METAL";
    return body;
}

std::string mfe_swiglu_tail() {
    const std::string_view source(kMoeSource);
    const auto begin = source.find("    if (SPLIT_SWIGLU) {\n");
    if (begin == std::string_view::npos) {
        throw std::logic_error("MFE SwiGLU tail marker is missing");
    }
    std::string tail(source.substr(begin));
    const std::string needle = "params[0]";
    std::size_t offset = 0;
    while ((offset = tail.find(needle, offset)) != std::string::npos) {
        tail.replace(offset, needle.size(), "SWIGLU_LIMIT");
        offset += std::string_view("SWIGLU_LIMIT").size();
    }
    return tail;
}

void append_mfe_decode_constants(
    std::ostringstream& source,
    const MfeNintDecodeConfig& config,
    bool gate_up) {
    source
        << "constexpr int TOKENS = 1;\n"
        << "constexpr int ROUTES = " << config.routes << ";\n"
        << "constexpr int EXPERTS = " << config.experts << ";\n"
        << "constexpr int OUT = "
        << (gate_up ? config.intermediate : config.hidden) << ";\n"
        << "constexpr int MATRIX_OUT = OUT" << (gate_up && config.packed_gate_up ? " * 2" : "") << ";\n"
        << "constexpr int PROJECTIONS = " << (gate_up && !config.packed_gate_up ? 2 : 1) << ";\n"
        << "constexpr int FUSED_SWIGLU = " << (gate_up ? 1 : 0) << ";\n"
        << "constexpr int K = "
        << (gate_up ? config.hidden : config.intermediate) << ";\n"
        << "constexpr int K_LANES_VALUE = "
        << (gate_up ? config.k_lanes : config.down_k_lanes) << ";\n"
        << "constexpr int ROWS_PER_SIMD_VALUE = "
        << (gate_up ? 1 : config.down_rows_per_simd) << ";\n"
        << "constexpr int DESCRIPTOR_SIZE = "
        << config.descriptor_size << ";\n"
        << "constexpr int VARIANT_STRIDE = ROUTES;\n"
        << "constexpr int SHARED_INPUT = " << (gate_up ? 1 : 0) << ";\n"
        << "constexpr int VQ_EXECUTION_LAYOUT = "
        << (gate_up ? config.gate_vq_execution_layout
                    : config.down_vq_execution_layout) << ";\n"
        << "constexpr int FAMILY_MASK = "
        << (gate_up ? config.gate_family_mask
                    : config.down_family_mask) << ";\n"
        << "constexpr int VQ_PROFILE_MASK = "
        << (gate_up ? config.gate_vq_profile_mask
                    : config.down_vq_profile_mask) << ";\n"
        << "constexpr int HAS_NEPQ_RESIDUAL = "
        << (gate_up ? config.gate_has_nepq_residual
                    : config.down_has_nepq_residual) << ";\n"
        << "constexpr int NPQ_GROUPED_INDICES = "
        << (gate_up ? config.gate_npq_grouped_indices
                    : config.down_npq_grouped_indices) << ";\n"
        << "constexpr int SORTED_ROUTES = 0;\n"
        << "constexpr int EXPERT_MAP_SIZE = 0;\n"
        << "constexpr int PACKED_EXPERT_IDS = 0;\n";
}

void append_mfe_shared_nint_decoder(
    std::ostringstream& source,
    const MfeNintDecodeConfig& config,
    bool gate_up) {
    const int group_size = gate_up
        ? config.shared_gate_group_size : config.shared_down_group_size;
    const int groups = gate_up
        ? config.shared_gate_groups : config.shared_down_groups;
    source
        << "  MfqNintRow metadata_rows[ROWS_PER_SIMD];\n"
        << "  uint q_widths[ROWS_PER_SIMD];\n"
        << "  uint q_row_byte_offsets[ROWS_PER_SIMD], q_row_bit_shifts[ROWS_PER_SIMD];\n"
        << "  float neuron_scales[ROWS_PER_SIMD], neuron_minimums[ROWS_PER_SIMD];\n"
        << "  for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {\n"
        << "    uint output = min(output_base + row, uint(OUT) - 1u);\n"
        << "    uint physical = " << (gate_up ? "projection * uint(OUT) + output" : "output") << ";\n"
        << "    MfqNintRow metadata = mfq_nint_row(shared_row_metadata, physical);\n"
        << "    metadata_rows[row] = metadata; q_widths[row] = metadata.q_bits;\n"
        << "    q_row_byte_offsets[row] = metadata.q_offset;\n"
        << "    q_row_bit_shifts[row] = metadata.q_shift;\n"
        << "    neuron_scales[row] = metadata.scale;\n"
        << "    neuron_minimums[row] = metadata.minimum;\n"
        << "  }\n"
        << "  device const uchar* nint_q = shared_q;\n"
        << "  device const uchar* nint_sub_scale = shared_sub_scale;\n"
        << "  device const uchar* nint_sub_min = shared_sub_min;\n"
        << "  uint x_offset = " << (gate_up ? "0u" : "SHARED_OFFSET") << ";\n";
    if (gate_up && config.shared_up_group_size != group_size) {
        source << "  uint q_offset = 0u, sub_offset = 0u;\n"
            << "  if (projection == 0u) { uint groups = " << groups << "u;\n"
            << "    TYLOINFER_MFE_NINT_CALL_WITH(" << group_size << "u, " << config.shared_gate_q_dispatch << ");\n"
            << "  } else { uint groups = " << config.shared_up_groups << "u;\n"
            << "    TYLOINFER_MFE_NINT_CALL_WITH(" << config.shared_up_group_size << "u, " << config.shared_gate_q_dispatch << ");\n"
            << "  }\n";
    } else {
        source << "  uint q_offset = 0u, sub_offset = 0u, groups = " << groups << "u;\n"
            << "  TYLOINFER_MFE_NINT_CALL_WITH(" << group_size << "u, "
            << (gate_up ? config.shared_gate_q_dispatch : config.shared_down_q_dispatch) << ");\n";
    }
}

constexpr const char* kFoldedTopKHeader = R"METAL(
inline void mfq_topk_keys(device const half* logits, uint lane, thread uint* keys, uint experts) {
    for (uint local = 0u; local < experts / 32u; ++local) {
        uint expert = lane + local * 32u;
        float value = float(logits[expert]);
        value = isnan(value) ? -FLT_MAX : value;
        if (value == 0.0f) value = 0.0f;
        uint bits = as_type<uint>(value);
        uint ordered = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
        keys[local] = ((ordered >> 13u) << 12u) | (4095u - expert);
    }
}
template <uint PER>
inline uint mfq_topk_step(thread const uint* keys, thread uint* head, thread uint& below, thread uint& left) {
    if (left == 0u) {
        uint a = 0u, b = 0u, c = 0u;
        for (uint local = 0u; local < PER; ++local) {
            uint v = keys[local] < below ? keys[local] : 0u;
            bool ga = v > a, gb = v > b, gc = v > c;
            c = gb ? b : (gc ? v : c);
            b = ga ? a : (gb ? v : b);
            a = ga ? v : a;
        }
        head[0] = a; head[1] = b; head[2] = c; left = 3u;
    }
    uint selected = simd_max(head[0]);
    if (head[0] == selected) {
        head[0] = head[1]; head[1] = head[2]; head[2] = 0u;
        below = selected; --left;
    }
    return 4095u - (selected & 4095u);
}
template <uint EXPERTS>
inline uint mfq_topk_nth(device const half* logits, uint lane, uint rank) {
    uint keys[EXPERTS / 32u], head[3] = {0u};
    mfq_topk_keys(logits, lane, keys, EXPERTS);
    uint below = 0xffffffffu, left = 0u, result = 0u;
    for (uint i = 0u; i <= rank; ++i)
        result = mfq_topk_step<EXPERTS / 32u>(keys, head, below, left);
    return result;
}
template <uint EXPERTS, uint ROUTES>
inline void mfq_topk_write(device const half* logits, uint lane, device float* routing) {
    uint keys[EXPERTS / 32u], head[3] = {0u}, ids[ROUTES];
    mfq_topk_keys(logits, lane, keys, EXPERTS);
    uint below = 0xffffffffu, left = 0u;
    for (uint i = 0u; i < ROUTES; ++i)
        ids[i] = mfq_topk_step<EXPERTS / 32u>(keys, head, below, left);
    if (lane == 0u) {
        float selected[ROUTES], denominator = 0.0f;
        for (uint i = 0u; i < ROUTES; ++i) {
            float value = float(logits[ids[i]]);
            selected[i] = isnan(value) ? -FLT_MAX : value;
        }
        float maximum = selected[0];
        for (uint i = 0u; i < ROUTES; ++i) {
            selected[i] = exp(selected[i] - maximum);
            denominator += selected[i];
        }
        for (uint i = 0u; i < ROUTES; ++i) {
            routing[i] = float(ids[i]);
            routing[ROUTES + i] = selected[i] / denominator;
        }
    }
}
)METAL";

std::string make_mfe_nint_gate_up_source(
    const MfeNintDecodeConfig& config,
    const std::string& kernel_name) {
    std::ostringstream source;
    source
        << "#include <metal_stdlib>\nusing namespace metal;\n"
        << "using T = " << metal_activation_type(config.dtype) << ";\n"
        << "using SharedGateT = "
        << (config.shared_router_group_size ? "uchar" : metal_activation_type(config.shared_gate_dtype)) << ";\n"
        << nint_group_specialization(config.gate_nint_group_sizes)
        << "#define TYLOINFER_NINT_Q_DISPATCH " << config.gate_q_dispatch << "\n"
        << "#define TYLOINFER_MOE_VQ_LAYOUTS " << config.gate_vq_layouts << "ul\n"
        << kMoeHeader
        << (config.folded_topk ? kFoldedTopKHeader : "");
    if (config.shared_router_group_size) {
        source << detail::nint_matmul_metal_header().substr(sizeof(detail::kNintMetadataSource) - 1)
            << "struct MfqSharedRouterInput { device const T* base; uint offset; T operator[](uint index) const { return base[offset + index]; } };\n"
            << "inline float4 mfq_nint_load_input4(MfqSharedRouterInput input, uint offset) { return mfq_nint_load_input4(input.base, input.offset + offset); }\n"
            << "template <bool SMALL> inline void mfq_shared_router(device const uchar* packed, device const T* values, device T* y, uint thread_index_in_simdgroup, uint input_offset) {\n"
            << "MfqSharedRouterInput x{values - input_offset, input_offset};\n"
            << "device const uint* header = reinterpret_cast<device const uint*>(packed);\n"
            << "device const uchar* q_packed = packed + header[0];\n"
            << "device const uchar* sub_scale = packed + header[1];\n"
            << "device const uchar* sub_min = packed + header[2];\n"
            << "device const uint* row_metadata = header + 4u;\n"
            << "device const int* expert_ids = reinterpret_cast<device const int*>(row_metadata);\n"
            << "device const int* expert_map = expert_ids;\n"
            << "uint simdgroup_index_in_threadgroup = 0u; uint3 threadgroup_position_in_grid = uint3(0u);\n";
        for (const auto& [name, value] : std::array<std::pair<const char*, int>, 14>{{
                 {"GS", config.shared_router_group_size},
                 {"NG", (config.hidden + config.shared_router_group_size - 1) / config.shared_router_group_size},
                 {"K", config.hidden}, {"OUT", 1}, {"M", 1}, {"TILE_M", 1}, {"OPS_PER_SIMD", 1},
                 {"ROUTED", 0}, {"ROUTES", 1}, {"SHARED_INPUT", 0}, {"EXPERT_MAP_SIZE", 1},
                 {"LOCAL_EXPERTS", 1}, {"OUT_PER_EXPERT", 1}, {"LOGICAL_OUT", 1}}})
            source << "#define " << name << " " << value << "\n";
        source << "#define SMALL_M SMALL\n#define TYLOINFER_NINT_STORE_OUTPUT(index, value) y[index] = T(value)\n"
            << detail::nint_matmul_metal_body()
            << "\n#undef TYLOINFER_NINT_STORE_OUTPUT\n#undef SMALL_M\n#undef GS\n#undef NG\n#undef K\n#undef OUT\n#undef M\n#undef TILE_M\n#undef OPS_PER_SIMD\n#undef ROUTED\n#undef ROUTES\n#undef SHARED_INPUT\n#undef EXPERT_MAP_SIZE\n#undef LOCAL_EXPERTS\n#undef OUT_PER_EXPERT\n#undef LOGICAL_OUT\n}\n";
    }
    source << "kernel void " << kernel_name << "(\n";
    append_mfe_pool_arguments(source);
    source
        << "device const T* x [[buffer(22)]],\n"
        << (config.folded_topk ? "device const half* router_logits [[buffer(23)]],\n"
            : "device const int* expert_ids [[buffer(23)]],\n")
        << "device const uchar* shared_q [[buffer(24)]],\n"
        << "device const uint* shared_row_metadata [[buffer(25)]],\n"
        << "device const uchar* shared_sub_scale [[buffer(26)]],\n"
        << "device const uchar* shared_sub_min [[buffer(27)]],\n"
        << "device const SharedGateT* shared_gate_weight [[buffer(28)]],\n"
        << (config.route_groups ? "device const int* route_groups [[buffer(29)]],\n" : "")
        << "device T* y [[buffer(" << (config.route_groups ? 30 : 29) << ")]],\n"
        << (config.folded_topk ? "device float* routing [[buffer(30)]],\n" : "")
        << "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],\n"
        << "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],\n"
        << "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],\n"
        << "uint3 group_count [[threadgroups_per_grid]]) {\n";
    append_mfe_decode_constants(source, config, true);
    if (config.route_groups)
        source << "device const T* batch_x = x; device T* batch_y = y;\n";
    if (config.tokens > 1) {
        source
            << "uint token_row = threadgroup_position_in_grid.x;\n"
            << "x += token_row * uint(" << config.hidden << ");\n"
            << "expert_ids += token_row * uint(" << config.routes << ");\n"
            << "y += token_row * uint(" << (config.routes + 1) * config.intermediate + 1 << ");\n";
    }
    source
        << "constexpr float SWIGLU_LIMIT = 0.0f;\n"
        << "constexpr uint SIMD_GROUPS = 2u;\n"
        << "constexpr uint K_LANES = uint(K_LANES_VALUE);\n"
        << "constexpr uint LANE_GROUPS = 32u / K_LANES;\n"
        << "constexpr uint ROWS_PER_SIMD = uint(ROWS_PER_SIMD_VALUE);\n"
        << "constexpr bool SPLIT_SWIGLU = true;\n"
        << "constexpr bool PACKED_SWIGLU = false;\n"
        << "constexpr uint MATRIX_ROWS = ROWS_PER_SIMD;\n"
        << "constexpr uint ROWS_PER_PHYSICAL_SIMD = "
        << (config.folded_topk ? kMfeFoldedGateSimdPairs : 1)
        << "u * LANE_GROUPS * ROWS_PER_SIMD;\n"
        << "constexpr uint ROWS_PER_TG = ROWS_PER_PHYSICAL_SIMD;\n"
        << "constexpr uint OUTPUT_TILES = (uint(OUT) + ROWS_PER_TG - 1u) / ROWS_PER_TG;\n"
        << "constexpr uint ROUTED_SIZE = uint(ROUTES) * uint(OUT);\n"
        << "constexpr uint SHARED_OFFSET = ROUTED_SIZE;\n"
        << "constexpr uint GATE_OFFSET = SHARED_OFFSET + uint(OUT);\n"
        << "uint lane = thread_index_in_simdgroup;\n"
        << "uint k_lane = lane & (K_LANES - 1u);\n"
        << "uint lane_group = lane / K_LANES + (simdgroup_index_in_threadgroup >> 1u) * LANE_GROUPS;\n"
        << "uint simd_group = simdgroup_index_in_threadgroup & 1u;\n"
        << "uint workgroup = threadgroup_position_in_grid."
        << (config.tokens > 1 ? "y" : "x") << ";\n"
        << "if (workgroup == 0u) {\n"
        << (config.folded_topk ? "  if (simdgroup_index_in_threadgroup == 1u) mfq_topk_write<uint(EXPERTS), uint(ROUTES)>(router_logits, lane, routing);\n" : "")
        << "  if (simdgroup_index_in_threadgroup == 0u) {\n";
    if (config.shared_router_group_size) {
        if (config.tokens == 1)
            source << "    mfq_shared_router<true>(shared_gate_weight, x, y + GATE_OFFSET, lane, 0u);\n";
        else source << "    if (group_count.x <= 5u) mfq_shared_router<true>(shared_gate_weight, x, y + GATE_OFFSET, lane, token_row * uint(K));\n"
                    << "    else mfq_shared_router<false>(shared_gate_weight, x, y + GATE_OFFSET, lane, token_row * uint(K));\n";
    } else if (config.shared_gate_is_logits) {
        source << "    if (lane == 0u) y[GATE_OFFSET] = T(shared_gate_weight["
               << (config.tokens > 1 ? "token_row" : "0u") << "]);\n";
    } else {
        source
            << "    float gate = 0.0f;\n"
            << "    for (uint column = lane; column < uint(K); column += 32u)\n"
            << "      gate = fma(float(x[column]), float(shared_gate_weight[column]), gate);\n"
            << "    gate = simd_sum(gate);\n"
            << "    if (lane == 0u) y[GATE_OFFSET] = T(gate);\n";
    }
    source
        << "  }\n"
        << "  return;\n"
        << "}\n"
        << "if (workgroup <= OUTPUT_TILES) {\n"
        << "  uint output_tile = workgroup - 1u;\n"
        << "  uint output_base = output_tile * ROWS_PER_TG + lane_group * ROWS_PER_SIMD;\n"
        << "  float accumulators[ROWS_PER_SIMD] = {0.0f};\n"
        << "  uint projection = simd_group;\n";
    append_mfe_shared_nint_decoder(source, config, true);
    source
        << "  threadgroup T values[2u * ROWS_PER_PHYSICAL_SIMD];\n"
        << "  for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {\n"
        << "    float total = accumulators[row];\n"
        << "    for (uint offset = K_LANES >> 1; offset > 0u; offset >>= 1u)\n"
        << "      total += simd_shuffle_down(total, offset);\n"
        << "    if (k_lane == 0u) values[projection * ROWS_PER_PHYSICAL_SIMD + lane_group * ROWS_PER_SIMD + row] = T(total);\n"
        << "  }\n"
        << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        << "  if (simd_group == 0u && k_lane == 0u) {\n"
        << "    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {\n"
        << "      uint local = lane_group * ROWS_PER_SIMD + row;\n"
        << "      uint output = output_tile * ROWS_PER_TG + local;\n"
        << "      if (output < uint(OUT)) {\n"
        << "        T gate = values[local]; T up = values[ROWS_PER_PHYSICAL_SIMD + local];\n"
        << "        y[SHARED_OFFSET + output] = T((float(gate) / (1.0f + exp(-float(gate)))) * float(up));\n"
        << "      }\n"
        << "    }\n"
        << "  }\n"
        << "  return;\n"
        << "}\n"
        << "workgroup -= 1u + OUTPUT_TILES;\n"
        << "uint output_tile = workgroup % OUTPUT_TILES;\n"
        << "uint projection_index = workgroup / OUTPUT_TILES;\n"
        << "uint projection = simd_group;\n"
        << "uint sorted_slot = projection_index;\n"
        << "uint route_index = sorted_slot;\n"
        << "uint route = route_index;\n"
        << "uint token = 0u;\n"
        << "uint output_base = output_tile * ROWS_PER_TG + lane_group * ROWS_PER_SIMD;\n";
    if (config.route_groups)
        source << "uint grouped_route = token_row * uint(ROUTES) + route;\n"
               << "uint2 route_group = uint2(*(device const packed_uint2*)(route_groups + grouped_route * 8u));\n"
               << "int logical_expert = int(route_group.x);\n";
    else if (config.folded_topk)
        source << "threadgroup uint selected_expert;\n"
               << "if (simdgroup_index_in_threadgroup == 0u) {\n"
               << "  uint selected = mfq_topk_nth<uint(EXPERTS)>(router_logits, lane, route);\n"
               << "  if (lane == 0u) selected_expert = selected;\n"
               << "}\n"
               << "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
               << "int logical_expert = int(selected_expert);\n";
    else source << "int logical_expert = expert_ids[route];\n";
    source << "int expert = logical_expert;\n";
    if (config.packed_gate_up) {
        source
            << "output_base += projection * uint(OUT);\n"
            << "projection = 0u;\n";
    }
    if (config.route_groups) {
        if (!config.packed_gate_up)
            source
                << "if (route_group.y > 1u && route_group.y <= 6u) {\n"
                << "  uint descriptor = (uint(expert) * 2u + projection) * uint(DESCRIPTOR_SIZE);\n"
                << "  uint other = (uint(expert) * 2u + (projection ^ 1u)) * uint(DESCRIPTOR_SIZE);\n"
                << "  if (descriptors[descriptor] == 0 && descriptors[other] == 0\n"
                << "      && mfq_moe_nint_supported_gs(uint(descriptors[descriptor + 5u]))\n"
                << "      && mfq_moe_nint_supported_gs(uint(descriptors[other + 5u]))) {\n"
                << "    device const int* rows = route_groups + grouped_route * 8u + 2u;\n"
                << "    uint position = 0u;\n"
                << "    while (position < route_group.y && uint(rows[position]) != grouped_route) ++position;\n"
                << "    uint first = position & ~1u;\n"
                << "    if (first + 1u < route_group.y) {\n"
                << "      MfqMoeVqOutput<uint(ROUTES) * uint(OUT), (uint(ROUTES) + 1u) * uint(OUT) + 1u, T> targets{batch_y};\n"
                << "      threadgroup T grouped_values[4u * ROWS_PER_PHYSICAL_SIMD];\n"
                << "      mfq_moe_nint_pair<1u, 1u, uint(K), K_LANES, uint(OUT), uint(MATRIX_OUT),\n"
                << "          2u, uint(ROUTES), true, true, false>(batch_x, targets, descriptors + descriptor,\n"
                << "          nint_q, nint_sub_scale, nint_sub_min, rows + first, grouped_values,\n"
                << "          output_base, lane_group, k_lane, simd_group, uint(VARIANT_STRIDE), 0.0f, position == first);\n"
                << "      return;\n"
                << "    }\n"
                << "  }\n"
                << "}\n";
        source
            << "uint grouped_count = route_group.y;\n"
            << "if (grouped_count > 1u && grouped_count <= 6u) {\n"
            << "  uint grouped_descriptor = (uint(expert) * uint(PROJECTIONS) + projection) * uint(DESCRIPTOR_SIZE);\n"
            << "  bool reusable = descriptors[grouped_descriptor] == 1 && mfq_moe_reusable_vq_profile(uint(descriptors[grouped_descriptor + 28u]) & 255u);\n";
        if (!config.packed_gate_up)
            source
                << "  uint other = (uint(expert) * 2u + (projection ^ 1u)) * uint(DESCRIPTOR_SIZE);\n"
                << "  reusable = reusable && descriptors[other] == 1 && mfq_moe_reusable_vq_profile(uint(descriptors[other + 28u]) & 255u);\n";
        source
            << "  if (reusable) {\n"
            << "    device const int* rows = route_groups + grouped_route * 8u + 2u;\n"
            << "    if (uint(rows[0]) != grouped_route) return;\n"
            << "    MfqMoeVqOutput<uint(ROUTES) * uint(OUT), (uint(ROUTES) + 1u) * uint(OUT) + 1u, T> targets{batch_y};\n"
            << "    threadgroup T grouped_values[12u * ROWS_PER_PHYSICAL_SIMD];\n"
            << "    switch (grouped_count) {\n";
        for (int rows = 2; rows <= 6; ++rows)
            source << "      case " << rows << "u: { mfq_moe_reused_rows<" << rows
                << "u, 1u, 1u, uint(K), K_LANES, uint(OUT), uint(MATRIX_OUT), uint(PROJECTIONS), uint(ROUTES), uint(VQ_PROFILE_MASK), true, true, false, false>("
                << "batch_x, targets, descriptors + grouped_descriptor, vq_indices, vq_state, vq_aux, vq_anchors, vq_codebooks, vq_scales, vq_state_to_codebank, vq_parameters, mx_values, mx_scales, rows, grouped_values, output_base, lane_group, k_lane, simd_group, uint(VARIANT_STRIDE), 0.0f);\n"
                << "mfq_moe_finish_reused_swiglu<" << rows << "u, 1u, K_LANES, uint(OUT)>("
                << "targets, rows, grouped_values, output_base, lane_group, k_lane, simd_group, 0.0f); return; }\n";
        source << "    }\n  }\n}\n";
    }
    source << mfe_decode_body();
    if (config.packed_gate_up) source << "projection = simd_group;\n";
    source << mfe_swiglu_tail();
    source << "\n}\n";
    return source.str();
}

std::string make_mfe_nint_down_source(
    const MfeNintDecodeConfig& config,
    const std::string& kernel_name) {
    const bool paired_nint = mfe_paired_nint_down(config);
    std::ostringstream source;
    source
        << "#include <metal_stdlib>\nusing namespace metal;\n"
        << "using T = " << metal_activation_type(config.dtype) << ";\n"
        << nint_group_specialization(config.down_nint_group_sizes)
        << "#define TYLOINFER_NINT_Q_DISPATCH " << config.down_q_dispatch << "\n"
        << "#define TYLOINFER_MOE_VQ_LAYOUTS " << config.down_vq_layouts << "ul\n"
        << kMoeHeader
        << "kernel void " << kernel_name << "(\n";
    append_mfe_pool_arguments(source);
    source
        << "device const T* x [[buffer(22)]],\n"
        << "device const uchar* shared_q [[buffer(23)]],\n"
        << "device const uint* shared_row_metadata [[buffer(24)]],\n"
        << "device const uchar* shared_sub_scale [[buffer(25)]],\n"
        << "device const uchar* shared_sub_min [[buffer(26)]],\n"
        << (config.folded_topk ? "device const float* expert_ids [[buffer(27)]],\n"
            : "device const int* expert_ids [[buffer(27)]],\n")
        << "device const float* route_weights [[buffer(28)]],\n"
        << "device T* y [[buffer(29)]],\n"
        << "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],\n"
        << "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],\n"
        << "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {\n";
    if (config.folded_topk) source << "route_weights += " << config.routes << "u;\n";
    append_mfe_decode_constants(source, config, false);
    if (config.tokens > 1) {
        source
            << "uint token_row = threadgroup_position_in_grid.x;\n"
            << "x += token_row * uint(" << (config.routes + 1) * config.intermediate + 1 << ");\n"
            << "expert_ids += token_row * uint(" << config.routes << ");\n"
            << "route_weights += token_row * uint(" << config.routes << ");\n"
            << "y += token_row * uint(" << config.hidden << ");\n";
    }
    source
        << "constexpr uint K_LANES = uint(K_LANES_VALUE);\n"
        << "constexpr uint LANE_GROUPS = 32u / K_LANES;\n"
        << "constexpr uint ROWS_PER_SIMD = uint(ROWS_PER_SIMD_VALUE);\n"
        << "constexpr bool SPLIT_SWIGLU = false;\n"
        << "constexpr bool PACKED_SWIGLU = false;\n"
        << "constexpr uint MATRIX_ROWS = ROWS_PER_SIMD;\n"
        << "constexpr uint ROWS_PER_PHYSICAL_SIMD = LANE_GROUPS * ROWS_PER_SIMD;\n"
        << "constexpr uint OUTPUT_TILES = (uint(OUT) + ROWS_PER_PHYSICAL_SIMD - 1u) / ROWS_PER_PHYSICAL_SIMD;\n"
        << "constexpr uint ROUTED_SIZE = uint(ROUTES) * uint(K);\n"
        << "constexpr uint SHARED_OFFSET = ROUTED_SIZE;\n"
        << "constexpr uint GATE_OFFSET = SHARED_OFFSET + uint(K);\n"
        << "uint lane = thread_index_in_simdgroup;\n"
        << "uint k_lane = lane & (K_LANES - 1u);\n"
        << "uint lane_group = lane / K_LANES;\n"
        << "uint route = simdgroup_index_in_threadgroup;\n"
        << "uint output_tile = threadgroup_position_in_grid."
        << (config.tokens > 1 ? "y" : "x") << ";\n"
        << "uint output_base = output_tile * ROWS_PER_PHYSICAL_SIMD + lane_group * ROWS_PER_SIMD;\n"
        << "threadgroup T partials[(ROUTES + 1) * ROWS_PER_PHYSICAL_SIMD];\n"
        << "if (route < uint(ROUTES)) {\n"
        << "  int expert = expert_ids[route];\n"
        << "  uint projection = 0u; uint token = 0u;\n"
        << "  if (expert >= 0 && expert < EXPERTS) {\n";
    if (paired_nint) source << R"METAL(
    device const int* selected = descriptors + uint(expert) * uint(DESCRIPTOR_SIZE);
    if (selected[0] == 0 && mfq_moe_nint_supported_gs(uint(selected[5]))) {
        uint narrow_lane = lane & 7u;
        uint narrow_group = lane >> 3u;
        uint output = min(output_tile * 4u + narrow_group, uint(OUT) - 1u);
        uint physical = uint(selected[1]) * uint(OUT) + output;
        device const uint* metadata = reinterpret_cast<device const uint*>(nint_q + uint(selected[11]));
        MfqNintRow metadata_rows[1] = {mfq_nint_row(metadata, physical)};
        uint q_widths[1] = {metadata_rows[0].q_bits};
        uint q_row_byte_offsets[1] = {metadata_rows[0].q_offset};
        uint q_row_bit_shifts[1] = {metadata_rows[0].q_shift};
        float neuron_scales[1] = {metadata_rows[0].scale};
        float neuron_minimums[1] = {metadata_rows[0].minimum};
        float low[1] = {0.0f}, high[1] = {0.0f};
        uint x_offset = route * uint(K), q_offset = uint(selected[7]), sub_offset = uint(selected[8]);
        uint group_size = uint(selected[5]), groups = uint(selected[6]);
        #define TYLOINFER_PAIRED_DOWN_Q(GS, Q) \
            mfq_moe_nint_profile<GS, 1u, uint(K), 16u, true, true, 1u, Q>( \
                x, nint_q, nint_sub_scale, nint_sub_min, x_offset, q_offset, sub_offset, \
                metadata_rows, q_widths, q_row_byte_offsets, q_row_bit_shifts, neuron_scales, \
                neuron_minimums, low, groups, narrow_lane); \
            mfq_moe_nint_profile<GS, 1u, uint(K), 16u, true, true, 1u, Q>( \
                x, nint_q, nint_sub_scale, nint_sub_min, x_offset, q_offset, sub_offset, \
                metadata_rows, q_widths, q_row_byte_offsets, q_row_bit_shifts, neuron_scales, \
                neuron_minimums, high, groups, narrow_lane + 8u)
        #define TYLOINFER_PAIRED_DOWN_GS(GS) \
            if constexpr (TYLOINFER_NINT_Q_DISPATCH) { \
            switch (metadata_rows[0].q_bits) { \
                case 4u: { TYLOINFER_PAIRED_DOWN_Q(GS, 4u); break; } \
                case 5u: { TYLOINFER_PAIRED_DOWN_Q(GS, 5u); break; } \
                case 6u: { TYLOINFER_PAIRED_DOWN_Q(GS, 6u); break; } \
                case 8u: { TYLOINFER_PAIRED_DOWN_Q(GS, 8u); break; } \
                default: { TYLOINFER_PAIRED_DOWN_Q(GS, 0u); break; } \
            } \
            } else { TYLOINFER_PAIRED_DOWN_Q(GS, 0u); }
        TYLOINFER_MFE_NINT_DISPATCH_WITH(TYLOINFER_PAIRED_DOWN_GS) { }
        #undef TYLOINFER_PAIRED_DOWN_GS
        #undef TYLOINFER_PAIRED_DOWN_Q
        float total = low[0] + high[0];
        for (uint offset = 4u; offset > 0u; offset >>= 1u)
            total += simd_shuffle_down(total, offset);
        if (narrow_lane == 0u) partials[route * 4u + narrow_group] = T(total);
    } else {
)METAL";
    source << mfe_decode_body();
    source
        << "    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {\n"
        << "      float total = accumulators[row];\n"
        << "      for (uint offset = K_LANES >> 1; offset > 0u; offset >>= 1u) total += simd_shuffle_down(total, offset);\n"
        << "      if (k_lane == 0u) partials[route * ROWS_PER_PHYSICAL_SIMD + lane_group * ROWS_PER_SIMD + row] = T(total);\n"
        << "    }\n";
    if (paired_nint) source << "    }\n";
    source
        << "  } else if (k_lane == 0u) {\n"
        << "    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) partials[route * ROWS_PER_PHYSICAL_SIMD + lane_group * ROWS_PER_SIMD + row] = T(0.0f);\n"
        << "  }\n"
        << "} else {\n"
        << "  float accumulators[ROWS_PER_SIMD] = {0.0f};\n";
    append_mfe_shared_nint_decoder(source, config, false);
    source
        << "  for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {\n"
        << "    float total = accumulators[row];\n"
        << "    for (uint offset = K_LANES >> 1; offset > 0u; offset >>= 1u) total += simd_shuffle_down(total, offset);\n"
        << "    if (k_lane == 0u) partials[uint(ROUTES) * ROWS_PER_PHYSICAL_SIMD + lane_group * ROWS_PER_SIMD + row] = T(total);\n"
        << "  }\n"
        << "}\n"
        << "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        << "if (route == 0u && k_lane == 0u) {\n"
        << "  float gate = float(x[GATE_OFFSET]); float shared_scale = 1.0f / (1.0f + exp(-gate));\n"
        << "  for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {\n"
        << "    uint local = lane_group * ROWS_PER_SIMD + row; uint output = output_tile * ROWS_PER_PHYSICAL_SIMD + local;\n"
        << "    if (output < uint(OUT)) {\n"
        << "      float total = 0.0f;\n"
        << "      for (uint selected = 0u; selected < uint(ROUTES); ++selected)\n"
        << "        total = fma(float(partials[selected * ROWS_PER_PHYSICAL_SIMD + local]), route_weights[selected], total);\n"
        << "      total += shared_scale * float(partials[uint(ROUTES) * ROWS_PER_PHYSICAL_SIMD + local]);\n"
        << "      y[output] = T(total);\n"
        << "    }\n"
        << "  }\n"
        << "}\n}\n";
    return source.str();
}

class MfeNintDecodePrimitive final : public mlx::core::Primitive, public tyloi::metal::PreparableKernel {
public:
    MfeNintDecodePrimitive(
        mlx::core::Stream stream,
        MfeNintDecodeConfig config,
        int stage)
        : Primitive(stream),
          config_(std::move(config)),
          stage_(stage),
          kernel_name_(mfe_decode_kernel_name(config_, stage_)) {}

    std::string preparation_key() const override { return kernel_name_; }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& selected_stream = stream();
        auto& device = tyloi::metal::kernel_device(selected_stream.device);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            kernel_name_,
            options,
            [config = config_, name = kernel_name_, stage = stage_] {
                return stage == 1
                    ? make_mfe_nint_gate_up_source(config, name)
                    : make_mfe_nint_down_source(config, name);
            });
        return device.get_kernel(kernel_name_, library);
    }

    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error(
            "two-stage MFE/NINT decode has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        const int input_count = 29 + static_cast<int>(stage_ == 1 && config_.route_groups);
        const bool routing_output = stage_ == 1 && config_.folded_topk;
        if (inputs.size() != static_cast<std::size_t>(input_count) ||
            outputs.size() != (routing_output ? 2u : 1u)) {
            throw std::logic_error(
                "two-stage MFE/NINT decode input count mismatch");
        }
        for (auto& output : outputs)
            output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto* kernel = prepared_kernel();
        auto& encoder = mlx::core::metal::get_command_encoder(
            selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        for (int index = 0; index < input_count; ++index) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(index)], index);
        }
        encoder.set_output_array(outputs[0], input_count);
        if (routing_output) encoder.set_output_array(outputs[1], input_count + 1);
        const int workgroups = stage_ == 1
            ? config_.stage_one_workgroups : config_.stage_two_workgroups;
        encoder.dispatch_threadgroups(
            MTL::Size(
                config_.tokens > 1 ? config_.tokens : workgroups,
                config_.tokens > 1 ? workgroups : 1,
                1),
            MTL::Size(
                stage_ == 1
                    ? 64 * (config_.folded_topk ? kMfeFoldedGateSimdPairs : 1)
                    : (config_.routes + 1) * 32,
                1,
                1));
    }

    const char* name() const override {
        return "MfeNintDecodePrimitive";
    }

    bool is_equivalent(const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const MfeNintDecodePrimitive*>(&other);
        return primitive != nullptr &&
            primitive->kernel_name_ == kernel_name_ &&
            primitive->config_.tokens == config_.tokens;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        std::vector<Shape> shapes{stage_ == 1
            ? config_.stage_one_shape
            : config_.output_shape};
        if (stage_ == 1 && config_.folded_topk)
            shapes.emplace_back(Shape{2, config_.routes});
        return shapes;
    }

private:
    MfeNintDecodeConfig config_;
    int stage_ = 0;
    std::string kernel_name_;
};

std::vector<array> mfe_nint_decode_dispatch(
    std::vector<array> inputs,
    const MfeNintDecodeConfig& config,
    int stage) {
    auto stream = mlx::core::default_stream(
        mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "two-stage MFE/NINT decode requires the Metal device");
    }
    std::vector<Shape> shapes{stage == 1 ? config.stage_one_shape : config.output_shape};
    std::vector<Dtype> dtypes{config.dtype};
    if (stage == 1 && config.folded_topk) {
        shapes.emplace_back(Shape{2, config.routes});
        dtypes.push_back(mlx::core::float32);
    }
    return array::make_arrays(
        shapes, dtypes,
        std::make_shared<MfeNintDecodePrimitive>(
            stream, config, stage),
        std::move(inputs));
}

std::string mxfp4_decode_reduce_kernel_name(
    const Mxfp4DecodeReduceConfig& config) {
    std::ostringstream name;
    name
        << "mfq_mxfp4_decode_down_reduce_"
        << (config.dtype == mlx::core::float16 ? "f16" : "f32")
        << "_t" << config.tokens
        << "_r" << config.routes
        << "_e" << config.experts
        << "_o" << config.output_width
        << "_k" << config.input_width
        << "_rr" << config.rows_per_lane_group;
    return name.str();
}

std::string make_mxfp4_decode_reduce_source(
    const Mxfp4DecodeReduceConfig& config,
    const std::string& kernel_name) {
    std::ostringstream source;
    source
        << "#include <metal_stdlib>\n"
        << "using namespace metal;\n"
        << "using T = "
        << (config.dtype == mlx::core::float16 ? "half" : "float")
        << ";\n"
        << "constant constexpr float MXFP4_LUT[16] = {\n"
           "    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,\n"
           "    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,\n"
           "};\n"
        << "kernel void " << kernel_name << "(\n"
        << "device const int* descriptors [[buffer(0)]],\n"
        << "device const uchar* mx_values [[buffer(1)]],\n"
        << "device const uchar* mx_scales [[buffer(2)]],\n"
        << "device const T* x [[buffer(3)]],\n"
        << "device const int* expert_ids [[buffer(4)]],\n"
        << "device const float* route_weights [[buffer(5)]],\n"
        << "device T* y [[buffer(6)]],\n"
        << "uint thread_index_in_simdgroup "
           "[[thread_index_in_simdgroup]],\n"
        << "uint simdgroup_index_in_threadgroup "
           "[[simdgroup_index_in_threadgroup]],\n"
        << "uint3 threadgroup_position_in_grid "
           "[[threadgroup_position_in_grid]]) {\n"
        << "constexpr uint TOKENS = " << config.tokens << "u;\n"
        << "constexpr uint ROUTES = " << config.routes << "u;\n"
        << "constexpr uint EXPERTS = " << config.experts << "u;\n"
        << "constexpr uint OUT = " << config.output_width << "u;\n"
        << "constexpr uint K = " << config.input_width << "u;\n"
        << "constexpr uint DESCRIPTOR_SIZE = "
        << config.descriptor_size << "u;\n"
        << "constexpr uint ROWS_PER_LANE_GROUP = "
        << config.rows_per_lane_group << "u;\n"
        << R"METAL(
    constexpr uint K_LANES = 16u;
    constexpr uint LANE_GROUPS = 32u / K_LANES;
    constexpr uint OUTPUTS_PER_TG = LANE_GROUPS * ROWS_PER_LANE_GROUP;
    threadgroup float route_outputs[ROUTES * OUTPUTS_PER_TG];

    const uint lane = thread_index_in_simdgroup;
    const uint k_lane = lane & (K_LANES - 1u);
    const uint lane_group = lane / K_LANES;
    const uint route = simdgroup_index_in_threadgroup;
    const uint output_tiles = (OUT + OUTPUTS_PER_TG - 1u)
        / OUTPUTS_PER_TG;
    const uint token = threadgroup_position_in_grid.x / output_tiles;
    const uint output_tile = threadgroup_position_in_grid.x
        - token * output_tiles;
    if (token >= TOKENS) return;
    const uint output_base = output_tile * OUTPUTS_PER_TG
        + lane_group * ROWS_PER_LANE_GROUP;
    uint bounded_outputs[ROWS_PER_LANE_GROUP];
    float accumulators[ROWS_PER_LANE_GROUP] = {0.0f};
    for (uint row = 0u; row < ROWS_PER_LANE_GROUP; ++row) {
        bounded_outputs[row] = min(output_base + row, OUT - 1u);
    }

    const int expert = expert_ids[token * ROUTES + route];
    if (expert >= 0 && expert < int(EXPERTS)) {
        const uint descriptor_base = uint(expert) * DESCRIPTOR_SIZE;
        // This primitive is selected only for a pure MXFP4 projection.
        const uint local_expert = uint(descriptors[descriptor_base + 1u]);
        const uint groups = uint(descriptors[descriptor_base + 4u]);
        const uint value_offset = uint(descriptors[descriptor_base + 5u]);
        const uint scale_offset = uint(descriptors[descriptor_base + 6u]);

        for (uint group = k_lane; group < groups; group += K_LANES) {
            const uint column_base = group * 32u;
            ulong pool_outputs[ROWS_PER_LANE_GROUP];
            float scales[ROWS_PER_LANE_GROUP];
            for (uint row = 0u; row < ROWS_PER_LANE_GROUP; ++row) {
                pool_outputs[row] = ulong(local_expert) * ulong(OUT)
                    + ulong(bounded_outputs[row]);
                const uchar raw_scale = mx_scales[
                    ulong(scale_offset)
                        + pool_outputs[row] * ulong(groups)
                        + ulong(group)];
                const uint scale_bits = raw_scale == 0u
                    ? 0x00400000u
                    : uint(raw_scale) << 23u;
                scales[row] = raw_scale == 255u
                    ? NAN
                    : as_type<float>(scale_bits);
            }

            for (uint component = 0u; component < 32u; component += 16u) {
                const uint column = column_base + component;
                const vec<T, 4> source0 = *reinterpret_cast<
                    device const vec<T, 4>*>(
                        x + (token * ROUTES + route) * K + column);
                const vec<T, 4> source1 = *reinterpret_cast<
                    device const vec<T, 4>*>(
                        x + (token * ROUTES + route) * K + column + 4u);
                const vec<T, 4> source2 = *reinterpret_cast<
                    device const vec<T, 4>*>(
                        x + (token * ROUTES + route) * K + column + 8u);
                const vec<T, 4> source3 = *reinterpret_cast<
                    device const vec<T, 4>*>(
                        x + (token * ROUTES + route) * K + column + 12u);
                const float activations[16] = {
                    float(source0.x), float(source0.y),
                    float(source0.z), float(source0.w),
                    float(source1.x), float(source1.y),
                    float(source1.z), float(source1.w),
                    float(source2.x), float(source2.y),
                    float(source2.z), float(source2.w),
                    float(source3.x), float(source3.y),
                    float(source3.z), float(source3.w),
                };
                for (uint row = 0u; row < ROWS_PER_LANE_GROUP; ++row) {
                    const ulong packed_offset = ulong(value_offset)
                        + pool_outputs[row] * (ulong(K) >> 1u)
                        + (ulong(column) >> 1u);
                    const uint2 packed = *reinterpret_cast<
                        device const uint2*>(mx_values + packed_offset);
                    for (uint pair = 0u; pair < 8u; ++pair) {
                        const uint word = pair < 4u ? packed.x : packed.y;
                        const uchar codes = uchar(
                            word >> ((pair & 3u) * 8u));
                        accumulators[row] = fma(
                            activations[pair * 2u],
                            scales[row] * MXFP4_LUT[uint(codes & 15u)],
                            accumulators[row]);
                        accumulators[row] = fma(
                            activations[pair * 2u + 1u],
                            scales[row] * MXFP4_LUT[uint(codes >> 4u)],
                            accumulators[row]);
                    }
                }
            }
        }
        for (uint offset = K_LANES >> 1u; offset > 0u; offset >>= 1u) {
            for (uint row = 0u; row < ROWS_PER_LANE_GROUP; ++row) {
                accumulators[row] += simd_shuffle_down(
                    accumulators[row], offset);
            }
        }
    }
    if (k_lane == 0u) {
        for (uint row = 0u; row < ROWS_PER_LANE_GROUP; ++row) {
            // Preserve the original two-dispatch contract: each down result
            // first rounds to the activation dtype before route weighting.
            const float down = expert >= 0 && expert < int(EXPERTS)
                ? float(T(accumulators[row]))
                : 0.0f;
            route_outputs[
                route * OUTPUTS_PER_TG
                    + lane_group * ROWS_PER_LANE_GROUP + row
            ] = down * route_weights[token * ROUTES + route];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One SIMD group performs the tiny deterministic Top-6 reduction while
    // the other route groups finish together at the same barrier.
    if (simdgroup_index_in_threadgroup == 0u && lane < OUTPUTS_PER_TG) {
        const uint output = output_tile * OUTPUTS_PER_TG + lane;
        float total = 0.0f;
        for (uint selected = 0u; selected < ROUTES; ++selected) {
            total += route_outputs[selected * OUTPUTS_PER_TG + lane];
        }
        if (output < OUT) {
            y[token * OUT + output] = T(total);
        }
    }
}
)METAL";
    return source.str();
}

class Mxfp4DecodeReducePrimitive final
    : public mlx::core::UnaryPrimitive, public tyloi::metal::PreparableKernel {
public:
    Mxfp4DecodeReducePrimitive(
        mlx::core::Stream stream,
        Mxfp4DecodeReduceConfig config)
        : UnaryPrimitive(stream),
          config_(std::move(config)),
          kernel_name_(mxfp4_decode_reduce_kernel_name(config_)) {}

    std::string preparation_key() const override { return kernel_name_; }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& selected_stream = stream();
        auto& device = tyloi::metal::kernel_device(
            selected_stream.device);
        CompileOptions compile_options;
        compile_options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            kernel_name_,
            compile_options,
            [config = config_, name = kernel_name_] {
                return make_mxfp4_decode_reduce_source(config, name);
            });
        return device.get_kernel(kernel_name_, library);
    }

    void eval_cpu(
        const std::vector<array>&,
        array&) override {
        throw std::runtime_error(
            "MXFP4 decode down-reduce has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() != 6) {
            throw std::logic_error(
                "MXFP4 decode down-reduce input count mismatch");
        }
        output.set_data(
            mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto* kernel = prepared_kernel();
        auto& encoder = mlx::core::metal::get_command_encoder(
            selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        for (int index = 0; index < 6; ++index) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(index)],
                index);
        }
        encoder.set_output_array(output, 6);
        encoder.dispatch_threadgroups(
            MTL::Size(config_.workgroups, 1, 1),
            MTL::Size(config_.routes * 32, 1, 1));
    }

    const char* name() const override {
        return "Mxfp4DecodeReducePrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const Mxfp4DecodeReducePrimitive*>(&other);
        return primitive != nullptr
            && primitive->kernel_name_ == kernel_name_;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {config_.output_shape};
    }

private:
    Mxfp4DecodeReduceConfig config_;
    std::string kernel_name_;
};

array mxfp4_decode_reduce_dispatch(
    std::vector<array> inputs,
    Mxfp4DecodeReduceConfig config) {
    auto stream = mlx::core::default_stream(
        mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "MXFP4 decode down-reduce requires the Metal device");
    }
    auto shape = config.output_shape;
    const auto dtype = config.dtype;
    return array(
        std::move(shape),
        dtype,
        std::make_shared<Mxfp4DecodeReducePrimitive>(
            stream,
            std::move(config)),
        std::move(inputs));
}

std::string mxfp4_pair_swiglu_kernel_name(
    const Mxfp4PairSwiGluConfig& config) {
    std::ostringstream name;
    name
        << "mfq_mxfp4_pair_swiglu_"
        << (config.dtype == mlx::core::float16 ? "f16" : "f32")
        << "_t" << config.tokens
        << "_r" << config.routes
        << "_e" << config.experts
        << "_o" << config.output_width
        << "_k" << config.input_width;
    return name.str();
}

std::string make_mxfp4_pair_swiglu_source(
    const Mxfp4PairSwiGluConfig& config,
    const std::string& kernel_name) {
    std::ostringstream source;
    source
        << "#include <metal_stdlib>\n"
        << "using namespace metal;\n"
        << "using T = "
        << (config.dtype == mlx::core::float16 ? "half" : "float")
        << ";\n"
        << "constant constexpr float MXFP4_LUT[16] = {\n"
           "    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,\n"
           "    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,\n"
           "};\n"
        << "inline float decode_e8m0(uchar raw) {\n"
           "    if (raw == 255u) return NAN;\n"
           "    uint bits = raw == 0u ? 0x00400000u : uint(raw) << 23u;\n"
           "    return as_type<float>(bits);\n"
           "}\n"
        << "kernel void " << kernel_name << "(\n"
        << "device const int* gate_descriptors [[buffer(0)]],\n"
        << "device const uchar* gate_values [[buffer(1)]],\n"
        << "device const uchar* gate_scales [[buffer(2)]],\n"
        << "device const int* up_descriptors [[buffer(3)]],\n"
        << "device const uchar* up_values [[buffer(4)]],\n"
        << "device const uchar* up_scales [[buffer(5)]],\n"
        << "device const T* x [[buffer(6)]],\n"
        << "device const int* expert_ids [[buffer(7)]],\n"
        << "device const float* params [[buffer(8)]],\n"
        << "device T* y [[buffer(9)]],\n"
        << "uint thread_index_in_simdgroup "
           "[[thread_index_in_simdgroup]],\n"
        << "uint simdgroup_index_in_threadgroup "
           "[[simdgroup_index_in_threadgroup]],\n"
        << "uint3 threadgroup_position_in_grid "
           "[[threadgroup_position_in_grid]]) {\n"
        << "constexpr uint TOKENS = " << config.tokens << "u;\n"
        << "constexpr uint ROUTES = " << config.routes << "u;\n"
        << "constexpr uint EXPERTS = " << config.experts << "u;\n"
        << "constexpr uint OUT = " << config.output_width << "u;\n"
        << "constexpr uint K = " << config.input_width << "u;\n"
        << "constexpr uint DESCRIPTOR_SIZE = "
        << config.descriptor_size << "u;\n"
        << R"METAL(
    constexpr uint K_LANES = 16u;
    constexpr uint LANE_GROUPS = 32u / K_LANES;
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint OUTPUTS_PER_TG = LANE_GROUPS * SIMD_GROUPS;
    constexpr uint OUTPUT_TILES = (OUT + OUTPUTS_PER_TG - 1u)
        / OUTPUTS_PER_TG;

    const uint lane = thread_index_in_simdgroup;
    const uint k_lane = lane & (K_LANES - 1u);
    const uint lane_group = lane / K_LANES;
    const uint workgroup = threadgroup_position_in_grid.x;
    const uint output_tile = workgroup % OUTPUT_TILES;
    const uint routed_row = workgroup / OUTPUT_TILES;
    const uint route = routed_row % ROUTES;
    const uint token = routed_row / ROUTES;
    if (token >= TOKENS) return;

    const uint output = output_tile * OUTPUTS_PER_TG
        + simdgroup_index_in_threadgroup * LANE_GROUPS
        + lane_group;
    const uint bounded_output = min(output, OUT - 1u);
    const int expert = expert_ids[token * ROUTES + route];
    float gate_accumulator = 0.0f;
    float up_accumulator = 0.0f;
    if (expert >= 0 && expert < int(EXPERTS)) {
        const uint descriptor_base = uint(expert) * DESCRIPTOR_SIZE;
        const uint gate_local = uint(gate_descriptors[descriptor_base + 1u]);
        const uint up_local = uint(up_descriptors[descriptor_base + 1u]);
        const uint groups = uint(gate_descriptors[descriptor_base + 4u]);
        const uint gate_value_offset = uint(
            gate_descriptors[descriptor_base + 5u]);
        const uint gate_scale_offset = uint(
            gate_descriptors[descriptor_base + 6u]);
        const uint up_value_offset = uint(
            up_descriptors[descriptor_base + 5u]);
        const uint up_scale_offset = uint(
            up_descriptors[descriptor_base + 6u]);
        const ulong gate_pool_output = ulong(gate_local) * ulong(OUT)
            + ulong(bounded_output);
        const ulong up_pool_output = ulong(up_local) * ulong(OUT)
            + ulong(bounded_output);

        for (uint group = 0u; group < groups; ++group) {
            const uint column = group * 32u + k_lane * 2u;
            const vec<T, 2> raw_activations = *reinterpret_cast<
                device const vec<T, 2>*>(x + token * K + column);
            const float2 activations = float2(raw_activations);
            const ulong gate_packed_offset = ulong(gate_value_offset)
                + gate_pool_output * (ulong(K) >> 1u)
                + (ulong(column) >> 1u);
            const ulong up_packed_offset = ulong(up_value_offset)
                + up_pool_output * (ulong(K) >> 1u)
                + (ulong(column) >> 1u);
            const uchar gate_codes = gate_values[gate_packed_offset];
            const uchar up_codes = up_values[up_packed_offset];
            const float gate_scale = decode_e8m0(gate_scales[
                ulong(gate_scale_offset)
                    + gate_pool_output * ulong(groups) + ulong(group)]);
            const float up_scale = decode_e8m0(up_scales[
                ulong(up_scale_offset)
                    + up_pool_output * ulong(groups) + ulong(group)]);
            const float2 gate_weights = gate_scale * float2(
                MXFP4_LUT[uint(gate_codes & 15u)],
                MXFP4_LUT[uint(gate_codes >> 4u)]);
            const float2 up_weights = up_scale * float2(
                MXFP4_LUT[uint(up_codes & 15u)],
                MXFP4_LUT[uint(up_codes >> 4u)]);
            gate_accumulator += dot(activations, gate_weights);
            up_accumulator += dot(activations, up_weights);
        }
        for (uint offset = K_LANES >> 1u; offset > 0u; offset >>= 1u) {
            gate_accumulator += simd_shuffle_down(
                gate_accumulator, offset);
            up_accumulator += simd_shuffle_down(
                up_accumulator, offset);
        }
    }

    if (k_lane == 0u && output < OUT) {
        if (expert < 0 || expert >= int(EXPERTS)) {
            y[routed_row * OUT + output] = T(0.0f);
            return;
        }
        // Preserve the unfused graph's projection-boundary rounding.
        float gate = float(T(gate_accumulator));
        float up = float(T(up_accumulator));
        if (params[0] > 0.0f) {
            gate = min(gate, params[0]);
            up = clamp(up, -params[0], params[0]);
        }
        const float activated = gate / (1.0f + exp(-gate));
        y[routed_row * OUT + output] = T(activated * up);
    }
}
)METAL";
    return source.str();
}

class Mxfp4PairSwiGluPrimitive final
    : public mlx::core::UnaryPrimitive, public tyloi::metal::PreparableKernel {
public:
    Mxfp4PairSwiGluPrimitive(
        mlx::core::Stream stream,
        Mxfp4PairSwiGluConfig config)
        : UnaryPrimitive(stream),
          config_(std::move(config)),
          kernel_name_(mxfp4_pair_swiglu_kernel_name(config_)) {}

    std::string preparation_key() const override { return kernel_name_; }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& selected_stream = stream();
        auto& device = tyloi::metal::kernel_device(
            selected_stream.device);
        CompileOptions compile_options;
        compile_options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            kernel_name_,
            compile_options,
            [config = config_, name = kernel_name_] {
                return make_mxfp4_pair_swiglu_source(config, name);
            });
        return device.get_kernel(kernel_name_, library);
    }

    void eval_cpu(
        const std::vector<array>&,
        array&) override {
        throw std::runtime_error(
            "MXFP4 pair SwiGLU has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() != 9) {
            throw std::logic_error(
                "MXFP4 pair SwiGLU input count mismatch");
        }
        output.set_data(
            mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto* kernel = prepared_kernel();
        auto& encoder = mlx::core::metal::get_command_encoder(
            selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        for (int index = 0; index < 9; ++index) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(index)],
                index);
        }
        encoder.set_output_array(output, 9);
        encoder.dispatch_threadgroups(
            MTL::Size(config_.workgroups, 1, 1),
            MTL::Size(64, 1, 1));
    }

    const char* name() const override {
        return "Mxfp4PairSwiGluPrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const Mxfp4PairSwiGluPrimitive*>(&other);
        return primitive != nullptr
            && primitive->kernel_name_ == kernel_name_;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {config_.output_shape};
    }

private:
    Mxfp4PairSwiGluConfig config_;
    std::string kernel_name_;
};

array mxfp4_pair_swiglu_dispatch(
    std::vector<array> inputs,
    Mxfp4PairSwiGluConfig config) {
    auto stream = mlx::core::default_stream(
        mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "MXFP4 pair SwiGLU requires the Metal device");
    }
    auto shape = config.output_shape;
    const auto dtype = config.dtype;
    return array(
        std::move(shape),
        dtype,
        std::make_shared<Mxfp4PairSwiGluPrimitive>(
            stream,
            std::move(config)),
        std::move(inputs));
}

class Mxfp4BlocksPrimitive final
    : public mlx::core::UnaryPrimitive, public tyloi::metal::PreparableKernel {
public:
    Mxfp4BlocksPrimitive(
        mlx::core::Stream stream,
        Mxfp4BlocksConfig config)
        : UnaryPrimitive(stream),
          config_(std::move(config)) {}

    std::string preparation_key() const override { return config_.paired ? "mfq_mxfp4_blocks_pair" : "mfq_mxfp4_blocks_single"; }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& device = tyloi::metal::kernel_device(stream().device);
        CompileOptions compile_options;
        compile_options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            "mfq_dsv4_mxfp4_blocks_v5",
            compile_options,
            dsv4_mxfp4_blocks_source);
        const char* kernel_name = config_.paired
            ? "mfq_dsv4_mxfp4_pair_concat_f16_bm32_bn32_bk32"
            : "mfq_dsv4_mxfp4_single_f16_bm32_bn32_bk32";
        return device.get_kernel(kernel_name, library);
    }

    void eval_cpu(
        const std::vector<array>&,
        array&) override {
        throw std::runtime_error(
            "MXFP4 block primitive has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() != 9) {
            throw std::logic_error(
                "MXFP4 block primitive input count mismatch");
        }
        output.set_data(
            mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto& device = tyloi::metal::kernel_device(
            selected_stream.device);
        auto* kernel = prepared_kernel();
        auto& encoder =
            mlx::core::metal::get_command_encoder(
                selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        for (int index = 0; index < 9; ++index) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(index)],
                index);
        }
        encoder.set_output_array(output, 9);
        encoder.set_bytes(config_.max_blocks, 10);
        encoder.set_bytes(config_.output_width, 11);
        encoder.set_bytes(config_.input_width, 12);
        encoder.dispatch_threadgroups(
            MTL::Size(
                ((config_.paired ? 2 : 1) * config_.output_width + 31) / 32,
                config_.max_blocks,
                1),
            MTL::Size(64, 1, 1));
    }

    const char* name() const override {
        return "Mxfp4BlocksPrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const Mxfp4BlocksPrimitive*>(&other);
        return primitive != nullptr
            && primitive->config_.route_count == config_.route_count
            && primitive->config_.max_blocks == config_.max_blocks
            && primitive->config_.experts == config_.experts
            && primitive->config_.output_width == config_.output_width
            && primitive->config_.input_width == config_.input_width
            && primitive->config_.block_rows == config_.block_rows
            && primitive->config_.paired == config_.paired;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {config_.output_shape};
    }

private:
    Mxfp4BlocksConfig config_;
};

array mxfp4_blocks_dispatch(
    std::vector<array> inputs,
    Mxfp4BlocksConfig config) {
    auto stream = mlx::core::default_stream(
        mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "MXFP4 blocks require the Metal device");
    }
    auto shape = config.output_shape;
    return array(
        std::move(shape),
        mlx::core::float16,
        std::make_shared<Mxfp4BlocksPrimitive>(
            stream,
            std::move(config)),
        std::move(inputs));
}

class GroupedMmqPrimitive final
    : public mlx::core::UnaryPrimitive, public tyloi::metal::PreparableKernel {
public:
    GroupedMmqPrimitive(
        mlx::core::Stream stream,
        GroupedMmqConfig config)
        : UnaryPrimitive(stream),
          config_(std::move(config)) {
        if (config_.use_nax && !config_.direct_nax && config_.projections == 1
            && config_.fused_swiglu == 0 && config_.cohort_populations
            && config_.cohort_populations->size() > 1) {
            int mask = 0;
            int experts = 0;
            for (const auto& [family, group, count] : *config_.cohort_populations) {
                mask |= family;
                experts += count;
            }
            compact_blocks_ = mask == config_.family_mask && experts == config_.experts;
        }
    }

    std::string preparation_key() const override {
        std::string key = "grouped_mmq";
        for (const int value : {config_.block_rows, config_.tile_columns, config_.projections,
                config_.input_width, config_.output_width, config_.matrix_output_width,
                config_.fused_swiglu, config_.has_nepq_residual, config_.family_mask,
                config_.vq_profile_mask, static_cast<int>(config_.use_nax), static_cast<int>(config_.direct_nax),
                static_cast<int>(config_.indirect_pages)})
            key += "_" + std::to_string(value);
        if (config_.nint_group_sizes)
            for (const int group : *config_.nint_group_sizes) key += "_gs" + std::to_string(group);
        if (compact_blocks()) key += "_compact_blocks_p"
            + std::to_string(config_.cohort_populations->size());
        return key;
    }
    void prepare_gpu() override {
        (void)prepared_kernels();
        if (compact_blocks()) (void)prepared_partition();
    }

    bool compact_blocks() const {
        return compact_blocks_;
    }

    MTL::ComputePipelineState* prepared_partition() {
        auto& device = tyloi::metal::kernel_device(stream().device);
        const int passes = static_cast<int>(config_.cohort_populations->size());
        const std::string name = "mfq_grouped_mfe_partition_v1_p" + std::to_string(passes);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        auto* library = device.get_library(name, options, [passes] {
            return std::string("#include <metal_stdlib>\nusing namespace metal;\n#define PASSES ")
                + std::to_string(passes) + R"METAL(
kernel void mfq_grouped_mfe_partition(
    device const int* descriptors [[buffer(0)]],
    device const int* source [[buffer(1)]],
    device const int* count [[buffer(2)]],
    device int* destination [[buffer(3)]],
    device int* counts [[buffer(4)]],
    constant int4& params [[buffer(5)]],
    constant int2* keys [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_int local_counts[PASSES];
    for (int pass = int(tid); pass < PASSES; pass += 256)
        atomic_store_explicit(local_counts + pass, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int block = int(tid); block < count[0]; block += 256) {
        const int expert = source[block * 3 + 1];
        if (expert < 0 || expert >= params.y || source[block * 3 + 2] <= 0) continue;
        device const int* descriptor = descriptors + expert * params.z;
        const int family = descriptor[0];
        for (int pass = 0; pass < PASSES; ++pass) {
            if ((keys[pass].x & (1 << family)) == 0
                || (keys[pass].y != 0 && descriptor[5] != keys[pass].y)) continue;
            const int slot = atomic_fetch_add_explicit(local_counts + pass, 1, memory_order_relaxed);
            const int offset = (pass * params.x + slot) * 3;
            destination[offset] = source[block * 3];
            destination[offset + 1] = expert;
            destination[offset + 2] = source[block * 3 + 2];
            break;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int pass = int(tid); pass < PASSES; pass += 256)
        counts[pass] = atomic_load_explicit(local_counts + pass, memory_order_relaxed);
}
)METAL";
        });
        return device.get_kernel("mfq_grouped_mfe_partition", library);
    }

    std::vector<std::tuple<MTL::ComputePipelineState*, int, int, int, int>> prepared_kernels() {
        auto& selected_stream = stream();
        auto& device = tyloi::metal::kernel_device(
            selected_stream.device);
        CompileOptions compile_options;
        compile_options.math_mode = MathMode::Fast;
        const bool vector_vq =
            (static_cast<std::uint32_t>(config_.vq_profile_mask)
                & kGroupedVqVectorProfileMask) != 0;
        const bool vector_jsc_extended =
            (static_cast<std::uint32_t>(config_.vq_profile_mask)
                & kGroupedJscExtendedProfileMask) != 0;
        const bool aligned_input = config_.tile_columns <= 64
            && !config_.direct_nax && config_.input_width % 32 == 0;
        const auto get_library = [&](int family_mask, int group_size, int simd_rows) {
            const bool uses_vq = (family_mask & (1 << kFamilyVq)) != 0;
            const bool selected_vector_vq = vector_vq && uses_vq;
            const bool selected_extended = vector_jsc_extended && uses_vq;
            std::string library_name;
            if (config_.use_nax) {
                library_name = selected_vector_vq
                    ? (selected_extended
                        ? "mfq_grouped_mfe_nax_v34_legacy_vq_jsc_extended"
                        : "mfq_grouped_mfe_nax_v34_legacy_vq_vector")
                    : (selected_extended
                        ? "mfq_grouped_mfe_nax_v34_jsc_extended"
                        : "mfq_grouped_mfe_nax_v34");
            } else {
                library_name = selected_vector_vq
                    ? (selected_extended
                        ? "mfq_grouped_mmq_v28_legacy_vq_jsc_extended"
                        : "mfq_grouped_mmq_v28_legacy_vq_vector")
                    : (selected_extended
                        ? "mfq_grouped_mmq_v28_jsc_extended"
                        : "mfq_grouped_mmq_v28");
            }
            library_name += "_fm" + std::to_string(family_mask)
                + "_gs" + std::to_string(group_size);
            if (compact_blocks()) library_name += "_compact";
            if (config_.indirect_pages) library_name += "_pages";
            if (!config_.use_nax)
                library_name += "_bm" + std::to_string(config_.block_rows);
            if (config_.use_nax) {
                library_name += "_bm" + std::to_string(config_.block_rows)
                    + "_bn" + std::to_string(config_.tile_columns)
                    + "_ak" + std::to_string(aligned_input)
                    + "_sm" + std::to_string(simd_rows)
                    + "_sw" + std::to_string(config_.fused_swiglu)
                    + "_di" + std::to_string(config_.direct_nax)
                    + "_iw" + std::to_string(config_.input_width)
                    + "_ow" + std::to_string(config_.output_width)
                    + "_mw" + std::to_string(config_.matrix_output_width);
            }
            return device.get_library(
                library_name,
                compile_options,
                [
                    use_nax = config_.use_nax,
                    indirect_pages = config_.indirect_pages,
                    vector_vq = selected_vector_vq,
                    vector_jsc_extended = selected_extended,
                    family_mask,
                    group_size,
                    block_rows = config_.block_rows,
                    tile_columns = config_.tile_columns,
                    aligned_input,
                    simd_rows,
                    fused = config_.fused_swiglu,
                    direct = config_.direct_nax,
                    input_width = config_.input_width,
                    output_width = config_.output_width,
                    matrix_output_width = config_.matrix_output_width
                ] {
                    std::string source;
                    source.reserve(
                        (use_nax
                            ? sizeof(detail::kSteelNaxSource)
                            : sizeof(detail::kSteelMmaSource))
                        + sizeof(detail::kMfePrefillSource)
                        + 256);
                    source += "#include <metal_stdlib>\n";
                    source += "#include <metal_simdgroup>\n";
                    source += "#include <metal_simdgroup_matrix>\n";
                    if (use_nax) {
                        source += "#include <MetalPerformancePrimitives/"
                            "MetalPerformancePrimitives.h>\n";
                        source += "#define TYLOINFER_ENABLE_NAX 1\n";
                    }
                    if (vector_vq) {
                        source +=
                            "#define TYLOINFER_ENABLE_LEGACY_VQ_VECTOR 1\n";
                    }
                    if (vector_jsc_extended) {
                        source +=
                            "#define TYLOINFER_ENABLE_JSC_EXTENDED_VECTOR 1\n";
                    }
                    source += "#define TYLOINFER_GROUPED_FAMILY_MASK "
                        + std::to_string(family_mask) + "\n";
                    source += "#define TYLOINFER_GROUPED_NINT_GROUP_SIZE "
                        + std::to_string(group_size) + "\n";
                    source += "#define TYLOINFER_GROUPED_MMQ_BM "
                        + std::to_string(block_rows) + "\n";
                    if (use_nax) {
                        source += "#define TYLOINFER_GROUPED_NAX_BM "
                            + std::to_string(block_rows) + "\n";
                        source += "#define TYLOINFER_GROUPED_NAX_BN "
                            + std::to_string(tile_columns) + "\n";
                        source += "#define TYLOINFER_GROUPED_NAX_ALIGNED_INPUT "
                            + std::to_string(aligned_input) + "\n";
                        source += "#define TYLOINFER_GROUPED_NAX_SIMD_ROWS "
                            + std::to_string(simd_rows) + "\n";
                        source += "#define TYLOINFER_GROUPED_NAX_INPUT_WIDTH "
                            + std::to_string(input_width) + "\n";
                        source += "#define TYLOINFER_GROUPED_NAX_OUTPUT_WIDTH "
                            + std::to_string(output_width) + "\n";
                        source += "#define TYLOINFER_GROUPED_NAX_MATRIX_OUTPUT_WIDTH "
                            + std::to_string(matrix_output_width) + "\n";
                        const bool aligned_nint = family_mask == (1 << kFamilyNint)
                            && group_size == 28 && tile_columns <= 64
                            && !fused && !direct;
                        const bool balanced_loader = tile_columns <= 64
                            && block_rows == 128 && !fused && !direct
                            && ((family_mask == (1 << kFamilyNint)
                                    && group_size == 24 && simd_rows == 16)
                                || (family_mask == (1 << kFamilyVq)
                                    && simd_rows == 16));
                        source += "#define TYLOINFER_GROUPED_NAX_BK "
                            + std::to_string(aligned_nint ? 224
                                : balanced_loader ? 192 : 96) + "\n";
                        const int x_stride = aligned_nint ? 228
                            : balanced_loader ? 196 : tile_columns == 128
                            || (tile_columns <= 64 && block_rows != 32
                                && block_rows != 64) ? 100 : 104;
                        const int w_stride = tile_columns <= 64 ? x_stride : 100;
                        source += "#define TYLOINFER_GROUPED_NAX_X_STRIDE "
                            + std::to_string(x_stride) + "\n";
                        source += "#define TYLOINFER_GROUPED_NAX_W_STRIDE "
                            + std::to_string(w_stride) + "\n";
                        source += "#define TYLOINFER_GROUPED_NAX_FUSED "
                            + std::to_string(fused != 0) + "\n";
                        source += "#define TYLOINFER_GROUPED_NAX_DIRECT "
                            + std::to_string(direct) + "\n";
                    }
                    source += "using namespace metal;\n";
                    source += "using bfloat16_t = bfloat;\n";
                    if (indirect_pages) {
                        source += "#define TYLOINFER_GROUPED_INDIRECT_PAGES 1\nstruct MfqExpertStreams {\n";
                        for (const auto& [name, type] : kExpertStreams)
                            source += "device const " + std::string(type) + "* " + name + ";\n";
                        source += "};\n";
                    }
                    source += detail::kNintMetadataSource;
                    source += use_nax
                        ? detail::kSteelNaxSource
                        : detail::kSteelMmaSource;
                    source += detail::kMfePrefillSource;
                    return source;
                });
        };
        const char* kernel_name = config_.use_nax
            ? "mfq_grouped_mfe_nax_f16_specialized"
            : (config_.fused_swiglu != 0
                ? (config_.has_nepq_residual != 0
                    ? "mfq_grouped_mmq_swiglu_f16_specialized_nr"
                    : "mfq_grouped_mmq_swiglu_f16_specialized")
                : (config_.has_nepq_residual != 0
                    ? "mfq_grouped_mmq_f16_specialized_nr"
                    : "mfq_grouped_mmq_f16_specialized"));
        std::vector<std::tuple<MTL::ComputePipelineState*, int, int, int, int>> kernels;
        std::vector<std::pair<int, int>> passes;
        if (!config_.direct_nax
            && config_.projections == 1 && config_.fused_swiglu == 0) {
            for (int family = 0; family < 7; ++family) {
                const int mask = 1 << family;
                if ((config_.family_mask & mask) == 0) continue;
                if (family == kFamilyNint && config_.nint_group_sizes
                    && !config_.nint_group_sizes->empty()) {
                    for (const auto group_size : *config_.nint_group_sizes) {
                        passes.emplace_back(mask, group_size);
                    }
                } else {
                    passes.emplace_back(mask, 0);
                }
            }
        } else {
            const int group_size = !config_.use_nax && config_.nint_group_sizes
                && config_.nint_group_sizes->size() == 1 ? config_.nint_group_sizes->front() : 0;
            passes.emplace_back(config_.family_mask, group_size);
        }
        for (const auto& [family_mask, group_size] : passes) {
            const int simd_rows = config_.use_nax
                    && (family_mask == (1 << kFamilyVq)
                        || (family_mask == (1 << kFamilyNint) && group_size == 24))
                    && config_.block_rows == 128 && config_.tile_columns == 32
                    && config_.output_width > config_.input_width
                    && aligned_input && !config_.direct_nax
                    && config_.fused_swiglu == 0
                ? 32 : 16;
            auto* library = get_library(family_mask, group_size, simd_rows);
            auto* kernel = device.get_kernel(kernel_name, library);
            const int tile_columns = config_.use_nax ? config_.tile_columns
                : config_.block_rows >= 80
                    && !(family_mask == (1 << kFamilyVq)
                        || (family_mask == (1 << kFamilyNint)
                            && (group_size == 24 || group_size == 28))) ? 32 : 64;
            kernels.emplace_back(kernel, simd_rows, tile_columns, family_mask, group_size);
            if (!config_.use_nax && config_.block_rows >= 64) {
                const auto mid_name = std::string(kernel_name) + "_mid";
                kernels.emplace_back(device.get_kernel(mid_name, library), simd_rows, 64, family_mask, group_size);
                if (config_.block_rows > 80) {
                    const auto mid80_name = std::string(kernel_name) + "_mid80";
                    kernels.emplace_back(device.get_kernel(mid80_name, library), simd_rows, tile_columns, family_mask, group_size);
                }
                const auto tail_name = std::string(kernel_name) + "_tail";
                kernels.emplace_back(device.get_kernel(tail_name, library), simd_rows, 64, family_mask, group_size);
            }
        }
        return kernels;
    }

    void eval_cpu(
        const std::vector<array>&,
        array&) override {
        throw std::runtime_error(
            "grouped MMQ primitive has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() < (config_.indirect_pages ? 29u : 28u)
            || (!config_.indirect_pages && inputs.size() != 28)) {
            throw std::logic_error(
                "grouped MMQ primitive input count mismatch");
        }
        output.set_data(
            mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto& encoder =
            mlx::core::metal::get_command_encoder(
                selected_stream);
        if (config_.indirect_pages) {
            for (std::size_t index = 29; index < inputs.size(); ++index)
                encoder.set_input_array(inputs[index], 29);
            retain_indirect_resources(encoder, std::span(inputs).subspan(29));
            encoder.set_input_array(inputs[28], 29);
        }
        const auto kernels = prepared_kernels();
        std::optional<array> compact_meta;
        std::optional<array> compact_count;
        if (compact_blocks()) {
            const int passes = static_cast<int>(config_.cohort_populations->size());
            compact_meta.emplace(Shape{passes, config_.max_blocks, 3}, mlx::core::int32, nullptr, std::vector<array>{});
            compact_count.emplace(Shape{passes}, mlx::core::int32, nullptr, std::vector<array>{});
            for (auto* temporary : {&*compact_meta, &*compact_count}) {
                temporary->set_data(mlx::core::allocator::malloc(temporary->nbytes()));
                encoder.add_temporary(*temporary);
            }
            std::vector<std::array<int, 2>> keys;
            for (const auto& [mask, group, population] : *config_.cohort_populations)
                keys.push_back({mask, group});
            encoder.set_compute_pipeline_state(prepared_partition());
            encoder.set_input_array(inputs[0], 0);
            encoder.set_input_array(inputs[26], 1);
            encoder.set_input_array(inputs[27], 2);
            encoder.set_output_array(*compact_meta, 3);
            encoder.set_output_array(*compact_count, 4);
            encoder.set_bytes(std::array<int, 4>{config_.max_blocks, config_.experts,
                config_.descriptor_size, 0}, 5);
            encoder.set_vector_bytes(keys, 6);
            encoder.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(256, 1, 1));
        }
        encoder.set_input_array(inputs[0], 0);
        for (int source = 8; source <= 16; ++source) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(source)],
                source - 7);
        }
        for (int source = 22; source <= 24; ++source) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(source)],
                source - 12);
        }
        const GroupedMmqParameters parameters{
            .route_count = config_.route_count,
            .tokens = config_.tokens,
            .routes = config_.routes,
            .experts = config_.experts,
            .output_width = config_.output_width,
            .matrix_output_width = config_.matrix_output_width,
            .projections = config_.projections,
            .input_width = config_.input_width,
            .descriptor_size = config_.descriptor_size,
            .variant_stride = config_.variant_stride,
            .shared_input = config_.shared_input,
            .input_sorted = config_.input_sorted,
            .swiglu_limit = config_.swiglu_limit,
        };
        encoder.set_output_array(output, 13);
        encoder.set_bytes(parameters, 14);
        encoder.set_input_array(inputs[17], 15);
        encoder.set_input_array(inputs[18], 16);
        encoder.set_input_array(inputs[19], 17);
        encoder.set_input_array(inputs[26], 18);
        encoder.set_input_array(inputs[27], 19);
        encoder.set_input_array(inputs[20], 20);
        encoder.set_input_array(inputs[21], 21);
        for (int source = 1; source <= 5; ++source) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(source)],
                source + 21);
        }
        encoder.set_input_array(inputs[6], 27);
        encoder.set_input_array(inputs[7], 28);
        const int rows_x = config_.use_nax && config_.block_rows == 128
            ? 1 : std::min(32, config_.max_blocks);
        for (const auto& [kernel, simd_rows, tile_columns, family_mask, group_size] : kernels) {
            const int columns = (config_.output_width + tile_columns - 1) / tile_columns;
            int pass_blocks = config_.max_blocks;
            if (compact_blocks()) {
                const auto& populations = *config_.cohort_populations;
                const auto found = std::find_if(populations.begin(), populations.end(),
                    [family = family_mask, group = group_size](const auto& population) {
                        return std::get<0>(population) == family && std::get<1>(population) == group;
                    });
                if (found == populations.end()) throw std::logic_error("grouped MFE cohort is missing");
                const auto pass = static_cast<std::size_t>(found - populations.begin());
                pass_blocks = std::min(config_.max_blocks,
                    (config_.route_count + config_.block_rows - 1) / config_.block_rows + std::get<2>(*found));
                encoder.set_input_array(*compact_meta, 18,
                    pass * static_cast<std::size_t>(config_.max_blocks) * 3 * sizeof(int));
                encoder.set_input_array(*compact_count, 19, pass * sizeof(int));
            }
            encoder.set_compute_pipeline_state(kernel);
            encoder.dispatch_threadgroups(
                config_.use_nax ? MTL::Size(compact_blocks() ? 1 : rows_x,
                    std::size_t(compact_blocks() ? pass_blocks : (config_.max_blocks + rows_x - 1) / rows_x) * columns, 1)
                    : MTL::Size(columns, config_.max_blocks, 1),
                MTL::Size(
                    config_.use_nax
                        ? config_.block_rows * config_.tile_columns / simd_rows
                        : 256,
                    1,
                    1));
        }
    }

    const char* name() const override {
        return "GroupedMmqPrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const GroupedMmqPrimitive*>(&other);
        return primitive != nullptr
            && primitive->config_.route_count == config_.route_count
            && primitive->config_.max_blocks == config_.max_blocks
            && primitive->config_.block_rows == config_.block_rows
            && primitive->config_.tile_columns == config_.tile_columns
            && primitive->config_.experts == config_.experts
            && primitive->config_.output_width == config_.output_width
            && primitive->config_.projections == config_.projections
            && primitive->config_.input_width == config_.input_width
            && primitive->config_.variant_stride == config_.variant_stride
            && primitive->config_.shared_input == config_.shared_input
            && primitive->config_.input_sorted == config_.input_sorted
            && primitive->config_.fused_swiglu == config_.fused_swiglu
            && primitive->config_.has_nepq_residual
                == config_.has_nepq_residual
            && primitive->config_.family_mask == config_.family_mask
            && primitive->config_.vq_profile_mask == config_.vq_profile_mask
            && primitive->config_.nint_group_sizes == config_.nint_group_sizes
            && primitive->config_.cohort_populations == config_.cohort_populations
            && primitive->config_.use_nax == config_.use_nax
            && primitive->config_.direct_nax == config_.direct_nax
            && primitive->config_.indirect_pages == config_.indirect_pages
            && primitive->config_.swiglu_limit == config_.swiglu_limit;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {config_.output_shape};
    }

private:
    GroupedMmqConfig config_;
    bool compact_blocks_ = false;
};

array grouped_mmq_dispatch(
    std::vector<array> inputs,
    GroupedMmqConfig config) {
    auto stream = mlx::core::default_stream(
        mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "grouped MMQ requires the Metal device");
    }
    auto shape = config.output_shape;
    return array(
        std::move(shape),
        mlx::core::float16,
        std::make_shared<GroupedMmqPrimitive>(
            stream,
            std::move(config)),
        std::move(inputs));
}

mlx::core::fast::CustomKernelFunction
make_moe_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_heterogeneous_mfe",
        {
            "descriptors",
            "nint_q",
            "nint_sub_scale",
            "nint_sub_min",
            "nint_anchor_scale",
            "nint_anchor_min",
            "q8_q",
            "q8_scales",
            "vq_indices",
            "vq_state",
            "vq_aux",
            "vq_anchors",
            "vq_codebooks",
            "vq_scales",
            "vq_state_to_codebank",
            "vq_banks",
            "vq_parameters",
            "vq_residual_codebooks",
            "vq_residual_first",
            "vq_residual_second",
            "mx_values",
            "mx_scales",
            "x",
            "expert_ids",
            "route_order",
            "params",
            "expert_map",
        },
        {"y"},
        std::string(
            "const int TOKENS = expert_ids_shape[0];\n"
            "const int VARIANT_STRIDE = SHARED_INPUT != 0 ? TOKENS : TOKENS * ROUTES;\n") + kMoeSource,
        kMoeHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
moe_kernel() {
    static const auto kernel = make_moe_kernel();
    return kernel;
}

mlx::core::fast::CustomKernelFunction
make_moe_hadamard_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_moe_signed_hadamard",
        {"x", "signs"},
        {"y"},
        kMoeHadamardSource,
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
moe_hadamard_kernel() {
    static const auto kernel =
        make_moe_hadamard_kernel();
    return kernel;
}

struct RotationSpec {
    array signs;
    std::vector<std::int8_t> sign_values;
    int block = 0;
    std::uint64_t seed = 0;
};

struct PackedStreams {
    PackedStream nint_q;
    PackedStream nint_sub_scale;
    PackedStream nint_sub_min;
    PackedStream nint_anchor_scale;
    PackedStream nint_anchor_min;
    PackedStream q8_q;
    PackedStream q8_scales;
    PackedStream vq_indices;
    PackedStream vq_state;
    PackedStream vq_aux;
    PackedStream vq_anchors;
    PackedStream vq_codebooks;
    PackedStream vq_scales;
    PackedStream vq_state_to_codebank;
    PackedStream vq_banks;
    PackedStream vq_parameters;
    PackedStream vq_residual_codebooks;
    PackedStream vq_residual_first;
    PackedStream vq_residual_second;
    PackedStream mx_values;
    PackedStream mx_scales;

    void retain_source(const array& source) {
        for (auto* stream : {&nint_q, &nint_sub_scale, &nint_sub_min,
            &nint_anchor_scale, &nint_anchor_min, &q8_q, &q8_scales, &vq_indices,
            &vq_state, &vq_aux, &vq_anchors, &vq_codebooks, &vq_scales,
            &vq_state_to_codebank, &vq_banks, &vq_parameters, &vq_residual_codebooks,
            &vq_residual_first, &vq_residual_second, &mx_values, &mx_scales}) {
            stream->retain_source(source);
        }
    }

    std::size_t size() const {
        std::size_t bytes = 0;
        for (const auto count : {
                 nint_q.size(), nint_sub_scale.size(), nint_sub_min.size(),
                 nint_anchor_scale.size(), nint_anchor_min.size(), q8_q.size(),
                 q8_scales.size(), vq_indices.size(), vq_state.size(), vq_aux.size(),
                 vq_anchors.size(), vq_codebooks.size(), vq_scales.size(),
                 vq_state_to_codebank.size(), vq_banks.size(), vq_parameters.size(),
                 vq_residual_codebooks.size(), vq_residual_first.size(),
                 vq_residual_second.size(), mx_values.size(), mx_scales.size()}) {
            bytes = checked_add(bytes, count, "packed streams size");
        }
        return bytes;
    }
};

struct DenseReferenceMoeWeight {
    array values;

    array embedding(
        const array& rows,
        Dtype dtype) const {
        return mlx::core::astype(
            mlx::core::take(values, rows, 0),
            dtype);
    }

    std::size_t packed_nbytes() const noexcept {
        return values.nbytes();
    }
};

using ReferenceMoeWeight = std::variant<
    MlxNintWeight,
    MlxNint8ZeroWeight,
    MlxVqWeight,
    MlxMxWeight,
    MlxFp8SqWeight,
    MlxMxfp4SqWeight,
    DenseReferenceMoeWeight>;

struct ReferenceMoeCohort {
    std::vector<std::int32_t> expert_ids;
    ReferenceMoeWeight weight;
};

struct Mxfp4SqMoeCohort {
    array expert_map;
    MlxMxfp4SqWeight weight;
};

struct Fp8SqMoeCohort {
    array expert_map;
    MlxFp8SqWeight weight;
};

void validate_nint_payload_shape(
    std::span<const std::uint8_t> payload,
    int expected_rows,
    int expected_columns) {
    BlobCursor cursor(payload);
    const int raw_bits = static_cast<int>(
        cursor.scalar<std::uint8_t>("NINT bits"));
    const bool is_nint_v2 = (raw_bits & 0x80) != 0;
    const int bits = raw_bits & 0x7f;
    const int sub_bits = static_cast<int>(
        cursor.scalar<std::uint8_t>(
            "NINT sub bits"));
    const int group_size =
        cursor.scalar<std::int32_t>(
            "NINT group size");
    const int axis =
        cursor.scalar<std::int32_t>("NINT axis");
    const int columns =
        cursor.scalar<std::int32_t>(
            "NINT neuron length");
    const auto dimensions =
        cursor.scalar<std::uint32_t>(
            "NINT dimension count");
    if (
        bits < 1
        || bits > 8
        || sub_bits < 1
        || sub_bits > 8
        || group_size <= 0
        || axis != 0
        || columns != expected_columns
        || (dimensions != 2 && dimensions != 3)
    ) {
        throw std::runtime_error(
            "MFE NINT cohort shape is inconsistent");
    }
    auto rows =
        cursor.scalar<std::int64_t>(
            "NINT output shape");
    if (dimensions == 3) {
        const auto output = cursor.scalar<std::int64_t>("NINT expert output shape");
        if (rows <= 0 || output <= 0 || rows > expected_rows / output)
            throw std::runtime_error("MFE NINT cohort row dimensions are inconsistent");
        rows *= output;
    }
    const auto shape_columns =
        cursor.scalar<std::int64_t>(
            "NINT input shape");
    const auto output_size =
        cursor.scalar<std::uint32_t>(
            "NINT output size");
    const auto groups =
        cursor.scalar<std::uint32_t>(
            "NINT group count");
    const auto expected_groups =
        (
            static_cast<std::uint64_t>(
                expected_columns)
            + static_cast<std::uint64_t>(
                group_size)
            - 1
        ) / static_cast<std::uint64_t>(
            group_size);
    if (
        rows != expected_rows
        || shape_columns != expected_columns
        || output_size
            != static_cast<std::uint32_t>(
                expected_rows)
        || groups != expected_groups
    ) {
        throw std::runtime_error(
            "MFE NINT cohort shape is inconsistent");
    }

    const auto metadata_count = checked_product(
        static_cast<std::size_t>(output_size),
        static_cast<std::size_t>(groups),
        "NINT metadata count");
    const auto value_count = checked_product(
        metadata_count,
        static_cast<std::size_t>(group_size),
        "NINT value count");
    const auto anchor_bytes = checked_product(
        static_cast<std::size_t>(output_size),
        2 * sizeof(std::uint16_t),
        "NINT anchor bytes");
    const auto packed_metadata =
        checked_packed_size(
            metadata_count,
            sub_bits,
            "NINT metadata bytes");
    const auto packed_values =
        checked_packed_size(
            value_count,
            bits,
            "NINT value bytes");
    const auto packed_tail = checked_add(
        checked_product(
            packed_metadata,
            2,
            "NINT metadata bytes"),
        packed_values,
        "NINT packed tail");
    const auto old_tail = checked_add(
        checked_product(
            metadata_count,
            2,
            "legacy NINT metadata bytes"),
        value_count,
        "legacy NINT tail");
    const auto remaining = cursor.remaining();
    if (is_nint_v2) {
        const auto k_selector_bytes = checked_packed_size(
            static_cast<std::size_t>(output_size),
            2,
            "NINTv2 k-selector bytes");
        const auto q_selector_bytes = checked_packed_size(
            static_cast<std::size_t>(output_size),
            3,
            "NINTv2 q-selector bytes");
        const auto minimum_v2 = checked_add(
            anchor_bytes,
            checked_add(
                k_selector_bytes,
                q_selector_bytes,
                "NINTv2 selector bytes"),
            "NINTv2 minimum payload");
        if (remaining < minimum_v2) {
            throw std::runtime_error(
                "invalid MFE NINTv2 cohort payload length");
        }
        return;
    }
    if (
        remaining
            != checked_add(
                anchor_bytes,
                packed_tail,
                "NINT payload bytes")
        && remaining
            != checked_add(
                anchor_bytes,
                old_tail,
                "legacy NINT payload bytes")
    ) {
        throw std::runtime_error(
            "invalid MFE NINT cohort payload length");
    }
}

void claim_experts(
    const std::vector<std::int32_t>& expert_ids,
    std::vector<int>& owners,
    int pool) {
    for (const auto expert : expert_ids) {
        if (
            expert < 0
            || expert
                >= static_cast<std::int32_t>(
                    owners.size())
        ) {
            throw std::runtime_error(
                "MFE pool contains an invalid expert id");
        }
        if (owners[static_cast<std::size_t>(expert)] >= 0) {
            throw std::runtime_error(
                "an expert belongs to multiple MFE pools");
        }
        owners[static_cast<std::size_t>(expert)] = pool;
    }
}

std::optional<MlxNintWeight> add_nint_pool(
    std::span<const std::uint8_t> payload,
    const std::vector<std::int32_t>& expert_ids,
    int out_per_expert,
    int neuron_len,
    PackedStreams& streams,
    std::vector<std::int32_t>& descriptors,
    bool pack_execution = true) {
    const auto expected_rows = checked_product(
        expert_ids.size(),
        static_cast<std::size_t>(out_per_expert),
        "NINT cohort row count");
    const int checked_rows = checked_int(
        expected_rows,
        "NINT cohort row count");
    validate_nint_payload_shape(
        payload,
        checked_rows,
        neuron_len);
    if (!pack_execution) {
        return MlxNintWeight::from_blob(payload);
    }
    auto storage = detail::parse_nint_blob(payload, true);
    if (
        storage.output_size != checked_rows
        || storage.input_size != neuron_len
        || storage.groups
            != (
                neuron_len + storage.group_size - 1
            ) / storage.group_size
    ) {
        throw std::runtime_error(
            "MFE NINT cohort shape is inconsistent");
    }

    while ((streams.nint_q.size() & 15u) != 0u) {
        streams.nint_q.push_back(0);
    }
    const int row_metadata_offset = checked_int(
        streams.nint_q.size(),
        "NINT row-metadata offset");
    const int q_offset = checked_int(
        checked_add(streams.nint_q.size(),
            checked_product(storage.row_metadata.size(), sizeof(std::uint32_t), "NINT metadata bytes"),
            "NINT q offset"),
        "NINT q offset");
    const int sub_offset = checked_int(
        streams.nint_sub_scale.size(),
        "NINT sub offset");

    for (std::size_t local_expert = 0;
         local_expert < expert_ids.size(); ++local_expert) {
        const int expert = expert_ids[local_expert];
        const auto base = checked_product(
            static_cast<std::size_t>(expert),
            static_cast<std::size_t>(kDescriptorSize),
            "descriptor offset");
        descriptors[base + kFamily] = kFamilyNint;
        descriptors[base + kLocalExpert] =
            checked_int(local_expert, "local expert");
        descriptors[base + kOut] = out_per_expert;
        descriptors[base + kInput] = neuron_len;
        descriptors[base + kNintBits] = storage.bits;
        descriptors[base + kNintGroupSize] = storage.group_size;
        descriptors[base + kNintGroups] = storage.groups;
        descriptors[base + kNintQOffset] = q_offset;
        descriptors[base + kNintSubOffset] = sub_offset;
        descriptors[base + kNintAnchorOffset] = 0;
        descriptors[base + kNintExecution] = 0;
        descriptors[base + kNintRowMetadataOffset] = row_metadata_offset;
        const auto first_row = local_expert * static_cast<std::size_t>(out_per_expert);
        const auto first_q = storage.row_metadata[first_row * 4u] & 15u;
        bool uniform_q = first_q == 4u || first_q == 5u || first_q == 6u || first_q == 8u;
        for (int row = 1; uniform_q && row < out_per_expert; ++row)
            uniform_q = (storage.row_metadata[(first_row + row) * 4u] & 15u) == first_q;
        descriptors[base + kNintQDispatch] = static_cast<int>(uniform_q);
        descriptors[base + kNintV2] = 1;
    }

    streams.nint_q.append(std::move(storage.row_metadata));
    streams.nint_q.append(storage.borrowed_q);
    streams.nint_q.append(std::move(storage.q_packed));
    while ((streams.nint_q.size() & 15u) != 0u) {
        streams.nint_q.push_back(0);
    }
    for (const auto view : storage.borrowed_sub_scale) streams.nint_sub_scale.append(view);
    for (const auto view : storage.borrowed_sub_min) streams.nint_sub_min.append(view);
    streams.nint_sub_scale.append(std::move(storage.sub_scale));
    streams.nint_sub_min.append(std::move(storage.sub_min));
    return std::nullopt;
}

MlxNint8ZeroWeight add_q8_pool(
    std::span<const std::uint8_t> payload,
    const std::vector<std::int32_t>& expert_ids,
    int out_per_expert,
    int neuron_len,
    PackedStreams& streams,
    std::vector<std::int32_t>& descriptors,
    bool pack_execution = true) {
    auto weight =
        MlxNint8ZeroWeight::from_blob(payload);
    const auto expected_rows = checked_product(
        expert_ids.size(),
        static_cast<std::size_t>(out_per_expert),
        "NINT8-0 cohort row count");
    if (
        weight.output_size()
            != checked_int(
                expected_rows,
                "NINT8-0 cohort row count")
        || weight.input_size() != neuron_len
        || weight.groups() != neuron_len / 32
    ) {
        throw std::runtime_error(
            "MFE NINT8-0 cohort shape is inconsistent");
    }

    if (!pack_execution) {
        return weight;
    }

    const int q_offset =
        checked_int(
            streams.q8_q.size(),
            "NINT8-0 q offset");
    const int scale_offset =
        checked_int(
            streams.q8_scales.size()
                / sizeof(std::uint16_t),
            "NINT8-0 scale offset");
    for (
        std::size_t local_expert = 0;
        local_expert < expert_ids.size();
        ++local_expert
    ) {
        const int expert =
            expert_ids[local_expert];
        const auto base =
            checked_product(
                static_cast<std::size_t>(expert),
                static_cast<std::size_t>(
                    kDescriptorSize),
                "descriptor offset");
        descriptors[base + kFamily] =
            kFamilyNint8Zero;
        descriptors[base + kLocalExpert] =
            checked_int(local_expert, "local expert");
        descriptors[base + kOut] = out_per_expert;
        descriptors[base + kInput] = neuron_len;
        descriptors[base + kQ8Groups] =
            weight.groups();
        descriptors[base + kQ8QOffset] = q_offset;
        descriptors[base + kQ8ScaleOffset] =
            scale_offset;
    }

    append_raw(
        streams.q8_q,
        weight.quantized_values(),
        mlx::core::int8,
        "NINT8-0 values");
    append_raw(
        streams.q8_scales,
        weight.scales(),
        mlx::core::float16,
        "NINT8-0 scales");
    return weight;
}

std::optional<MlxMxWeight> add_mx_pool(
    std::string_view dtype,
    std::span<const std::uint8_t> payload,
    const std::vector<std::int32_t>& expert_ids,
    int out_per_expert,
    int neuron_len,
    PackedStreams& streams,
    std::vector<std::int32_t>& descriptors,
    bool pack_execution = true) {
    const bool mxfp4 = dtype == "MXFP4";
    const bool mxfp8 = dtype == "MXFP8";
    if (!mxfp4 && !mxfp8) {
        throw std::invalid_argument("unsupported MFE MX cohort dtype");
    }
    const std::string label(dtype);
    const auto field = [&](const char* suffix) {
        return label + " " + suffix;
    };
    BlobCursor cursor(payload);
    const auto magic_name = field("magic");
    const auto magic = cursor.bytes(4, magic_name.c_str());
    const auto version_name = field("version");
    const auto kind_name = field("kind");
    const auto reserved_name = field("reserved");
    if (
        magic.size() != 4
        || std::memcmp(magic.data(), "MXT1", 4) != 0
        || cursor.scalar<std::uint8_t>(version_name.c_str()) != 1
        || cursor.scalar<std::uint8_t>(kind_name.c_str())
            != static_cast<std::uint8_t>(mxfp4 ? 4 : 8)
        || cursor.scalar<std::uint16_t>(reserved_name.c_str()) != 0
    ) {
        throw std::runtime_error(
            "invalid MFE " + label + " payload header");
    }
    const auto rows_name = field("rows");
    const auto columns_name = field("columns");
    const auto storage_rows_name = field("storage rows");
    const auto storage_columns_name = field("storage columns");
    const auto scale_rows_name = field("scale rows");
    const auto scale_columns_name = field("scale columns");
    const auto rows = cursor.scalar<std::uint64_t>(rows_name.c_str());
    const auto columns = cursor.scalar<std::uint64_t>(columns_name.c_str());
    const auto storage_rows =
        cursor.scalar<std::uint64_t>(storage_rows_name.c_str());
    const auto storage_columns =
        cursor.scalar<std::uint64_t>(storage_columns_name.c_str());
    const auto scale_rows =
        cursor.scalar<std::uint64_t>(scale_rows_name.c_str());
    const auto scale_columns =
        cursor.scalar<std::uint64_t>(scale_columns_name.c_str());
    const auto expected_rows = checked_product(
        expert_ids.size(),
        static_cast<std::size_t>(out_per_expert),
        "MX cohort row count");
    const auto block = static_cast<std::uint64_t>(mxfp4 ? 32 : 128);
    if (
        neuron_len % static_cast<int>(block) != 0
        || rows != expected_rows
        || columns != static_cast<std::uint64_t>(neuron_len)
        || storage_rows != rows
        || storage_columns != (mxfp4 ? columns / 2 : columns)
        || scale_rows != (mxfp4 ? rows : (rows + 127) / 128)
        || scale_columns != columns / block
    ) {
        throw std::runtime_error(
            "MFE " + label + " cohort shape is inconsistent");
    }
    const auto value_count = checked_product(
        checked_size(storage_rows, "MX storage rows"),
        checked_size(storage_columns, "MX storage columns"),
        "MX value bytes");
    const auto scale_count = checked_product(
        checked_size(scale_rows, "MX scale rows"),
        checked_size(scale_columns, "MX scale columns"),
        "MX scale bytes");
    const auto values_name = field("values");
    const auto scales_name = field("scales");
    const auto values = cursor.bytes(value_count, values_name.c_str());
    const auto scales = cursor.bytes(scale_count, scales_name.c_str());
    if (std::find(scales.begin(), scales.end(), std::uint8_t{255})
        != scales.end()) {
        throw std::runtime_error(
            "MFE " + label + " contains an E8M0 NaN scale");
    }
    if (
        mxfp8
        && std::any_of(
            values.begin(),
            values.end(),
            [](std::uint8_t value) {
                return (value & 0x7fu) == 0x7fu;
            })
    ) {
        throw std::runtime_error(
            "MFE MXFP8 contains an E4M3 NaN code");
    }
    if (cursor.remaining() != 0) {
        throw std::runtime_error(
            "trailing bytes in MFE " + label + " cohort");
    }
    if (!pack_execution) {
        return MlxMxWeight::from_blob(label, payload);
    }

    const std::int32_t value_offset = checked_descriptor_u32(
        streams.mx_values.size(),
        "MX value offset");
    const std::int32_t scale_offset = checked_descriptor_u32(
        streams.mx_scales.size(),
        "MX scale offset");
    const int groups = neuron_len / static_cast<int>(block);
    for (
        std::size_t local_expert = 0;
        local_expert < expert_ids.size();
        ++local_expert
    ) {
        const int expert = expert_ids[local_expert];
        const auto base = checked_product(
            static_cast<std::size_t>(expert),
            static_cast<std::size_t>(kDescriptorSize),
            "descriptor offset");
        descriptors[base + kFamily] =
            mxfp4 ? kFamilyMxfp4 : kFamilyMxfp8;
        descriptors[base + kLocalExpert] =
            checked_int(local_expert, "local expert");
        descriptors[base + kOut] = out_per_expert;
        descriptors[base + kInput] = neuron_len;
        descriptors[base + kMxGroups] = groups;
        descriptors[base + kMxValueOffset] = value_offset;
        descriptors[base + kMxScaleOffset] = scale_offset;
    }
    append_bytes(streams.mx_values, values, "MX values");
    append_bytes(streams.mx_scales, scales, "MX scales");
    return std::nullopt;
}

std::optional<DenseReferenceMoeWeight> add_dense_pool(
    std::string_view dtype,
    std::span<const std::uint8_t> payload,
    const std::vector<std::int32_t>& expert_ids,
    int out_per_expert,
    int neuron_len,
    PackedStreams& streams,
    std::vector<std::int32_t>& descriptors,
    bool pack_execution = true) {
    const bool bf16 = dtype == "BF16";
    if (!bf16 && dtype != "F16") {
        throw std::invalid_argument("unsupported MFE dense cohort dtype");
    }
    BlobCursor cursor(payload);
    if (cursor.scalar<std::uint32_t>("dense dimension count") != 2) {
        throw std::runtime_error(
            "MFE dense expert cohort must be rank 2");
    }
    const auto rows = cursor.scalar<std::int64_t>("dense rows");
    const auto columns = cursor.scalar<std::int64_t>("dense columns");
    const auto expected_rows = checked_product(
        expert_ids.size(),
        static_cast<std::size_t>(out_per_expert),
        "dense cohort row count");
    const auto value_count = checked_product(
        expected_rows,
        static_cast<std::size_t>(neuron_len),
        "dense cohort value count");
    const auto value_bytes = checked_product(
        value_count,
        sizeof(std::uint16_t),
        "dense cohort value bytes");
    if (
        rows != static_cast<std::int64_t>(expected_rows)
        || columns != neuron_len
        || cursor.remaining() != value_bytes
    ) {
        throw std::runtime_error(
            "MFE dense expert cohort shape is inconsistent");
    }
    const auto values = cursor.bytes(value_bytes, "dense values");
    if (!pack_execution) {
        std::vector<std::uint8_t> copied(values.begin(), values.end());
        auto array_value = make_raw_array(
            std::move(copied),
            bf16 ? mlx::core::bfloat16 : mlx::core::float16);
        return DenseReferenceMoeWeight{
            mlx::core::reshape(
                std::move(array_value),
                Shape{
                    checked_int(expected_rows, "dense cohort rows"),
                    neuron_len,
                }),
        };
    }

    if ((streams.q8_q.size() & 1u) != 0u) {
        streams.q8_q.push_back(0);
    }
    const int value_offset = checked_int(
        streams.q8_q.size(),
        "dense value offset");
    for (
        std::size_t local_expert = 0;
        local_expert < expert_ids.size();
        ++local_expert
    ) {
        const int expert = expert_ids[local_expert];
        const auto base = checked_product(
            static_cast<std::size_t>(expert),
            static_cast<std::size_t>(kDescriptorSize),
            "descriptor offset");
        descriptors[base + kFamily] = bf16 ? kFamilyBf16 : kFamilyF16;
        descriptors[base + kLocalExpert] =
            checked_int(local_expert, "local expert");
        descriptors[base + kOut] = out_per_expert;
        descriptors[base + kInput] = neuron_len;
        descriptors[base + kDenseValueOffset] = value_offset;
    }
    append_bytes(streams.q8_q, values, "dense values");
    return std::nullopt;
}

std::vector<std::int8_t> int8_values(
    const array& source,
    const char* name) {
    if (
        source.dtype() != mlx::core::int8
        || !source.flags().row_contiguous
    ) {
        throw std::runtime_error(
            std::string("invalid MFE packed ") + name);
    }
    auto evaluated = source;
    evaluated.eval();
    return {
        evaluated.data<std::int8_t>(),
        evaluated.data<std::int8_t>()
            + static_cast<std::ptrdiff_t>(
                evaluated.size()),
    };
}

int rotation_variant(
    const detail::ParsedVqStorage& weight,
    std::vector<RotationSpec>& rotations) {
    if (weight.rotation_block == 0) {
        return 0;
    }
    std::vector<std::int8_t> values(weight.rotation_signs.begin(), weight.rotation_signs.end());
    if (
        values.size()
        != static_cast<std::size_t>(
            weight.input_size)
    ) {
        throw std::runtime_error(
            "rotated MFE VQ sign width mismatch");
    }
    for (
        std::size_t index = 0;
        index < rotations.size();
        ++index
    ) {
        const auto& rotation = rotations[index];
        if (
            rotation.block == weight.rotation_block
            && rotation.seed == weight.rotation_seed
        ) {
            if (rotation.sign_values != values) {
                throw std::runtime_error(
                    "conflicting MFE HSG1 sign "
                    "vectors share one rotation key");
            }
            return checked_int(
                index + 1,
                "rotation variant");
        }
    }
    rotations.push_back({
        make_raw_array(std::vector<std::uint8_t>(values.begin(), values.end()), mlx::core::int8),
        std::move(values),
        weight.rotation_block,
        weight.rotation_seed,
    });
    return checked_int(
        rotations.size(),
        "rotation variant");
}

int rotation_variant(
    const RotationSpec& source,
    std::vector<RotationSpec>& rotations) {
    for (
        std::size_t index = 0;
        index < rotations.size();
        ++index
    ) {
        const auto& rotation = rotations[index];
        if (
            rotation.block == source.block
            && rotation.seed == source.seed
        ) {
            if (
                rotation.sign_values
                != source.sign_values
            ) {
                throw std::runtime_error(
                    "conflicting MFE HSG1 sign "
                    "vectors share one rotation key");
            }
            return checked_int(
                index + 1,
                "rotation variant");
        }
    }
    rotations.push_back(source);
    return checked_int(
        rotations.size(),
        "rotation variant");
}

std::optional<MlxVqWeight> add_vq_pool(
    std::string_view dtype,
    std::span<const std::uint8_t> payload,
    std::span<const std::uint8_t> runtime,
    const std::vector<std::int32_t>& expert_ids,
    int out_per_expert,
    int neuron_len,
    PackedStreams& streams,
    std::vector<RotationSpec>& rotations,
    std::vector<std::int32_t>& descriptors,
    bool pack_execution = true) {
    const auto expected_rows = checked_product(
        expert_ids.size(),
        static_cast<std::size_t>(out_per_expert),
        "VQ cohort row count");
    const int checked_rows = checked_int(
        expected_rows,
        "VQ cohort row count");
    const auto metadata = inspect_vq_blob(
        dtype,
        payload,
        runtime);
    const auto& output_shape = metadata.output_shape;
    const bool cross_expert =
        output_shape.size() == 2;
    if (
        metadata.output_size != checked_rows
        || metadata.input_size != neuron_len
        || (
            cross_expert
            && (
                output_shape[0]
                    != static_cast<int>(
                        expert_ids.size())
                || output_shape[1]
                    != out_per_expert
            )
        )
        || (
            !cross_expert
            && (
                output_shape.size() != 1
                || output_shape.front()
                    != checked_rows
            )
        )
    ) {
        throw std::runtime_error(
            "MFE VQ cohort shape is inconsistent");
    }
    // Every supported VQ profile stores one FP16 anchor per flattened output
    // row.  Reject impossible dimensions before the full parser allocates
    // canonical execution arrays from an attacker-controlled header.
    if (
        expected_rows
        > payload.size() / sizeof(std::uint16_t)
    ) {
        throw std::runtime_error(
            "MFE VQ cohort dimensions exceed "
            "its payload");
    }
    if (!pack_execution) {
        return MlxVqWeight::from_blob(dtype, payload, runtime);
    }
    auto weight = detail::parse_vq_blob(dtype, payload, runtime, true);
    if (
        weight.output_size != metadata.output_size
        || weight.input_size != metadata.input_size
        || weight.output_shape != metadata.output_shape
        || weight.rotation_block
            != metadata.rotation_block
        || weight.rotation_seed
            != metadata.rotation_seed
    ) {
        throw std::runtime_error(
            "MFE VQ header/full parse mismatch");
    }
    const auto& profile = weight.label;
    const bool group64_execution =
        profile == "NVQ2J-XL"
        && weight.execution_layout == 1;
    const bool native_jsc = detail::vq_native_jsc(weight.execution_layout, weight.group_size, weight.state_bits,
        weight.table_banks, weight.index_bits, weight.vector_size, weight.input_size,
        weight.aux_mode, weight.code_bank_mode);
    const bool native_nvq1 = detail::vq_native_nvq1(weight.execution_layout, weight.group_size,
        weight.vector_size, weight.label, weight.index_bits, weight.state_bits,
        weight.code_bank_mode, weight.aux_mode, weight.table_banks);
    const bool packed_nvq1s_execution =
        profile == "NVQ1-S" && native_nvq1;
    const bool packed_nvq1l_execution =
        profile == "NVQ1-L" && native_nvq1;

    if (group64_execution) {
        while ((streams.vq_indices.size() & 7u) != 0u) {
            streams.vq_indices.push_back(0);
        }
    }
    const int indices_offset = checked_int(
        streams.vq_indices.size(),
        "VQ index offset");
    const int state_offset = checked_int(
        streams.vq_state.size(),
        "VQ state offset");
    const int aux_offset = checked_int(
        streams.vq_aux.size(),
        "VQ auxiliary offset");
    const int anchor_offset = checked_int(
        streams.vq_anchors.size() / sizeof(float),
        "VQ anchor offset");
    const int codebook_offset = checked_int(
        streams.vq_codebooks.size(),
        "VQ codebook offset");
    const int scale_offset = checked_int(
        streams.vq_scales.size() / sizeof(float),
        "VQ scale offset");
    const int state_bank_offset = checked_int(
        streams.vq_state_to_codebank.size(),
        "VQ state-bank offset");
    const int bank_offset = checked_int(
        streams.vq_banks.size(),
        "VQ bank offset");
    const int parameter_offset = checked_int(
        streams.vq_parameters.size()
            / sizeof(float),
        "VQ parameter offset");
    const int residual_codebook_offset = checked_int(
        streams.vq_residual_codebooks.size()
            / sizeof(float),
        "VQ residual codebook offset");
    const int residual_record_offset = checked_int(
        streams.vq_residual_first.size()
            / sizeof(std::int16_t),
        "VQ residual record offset");
    const int rotation =
        rotation_variant(weight, rotations);

    for (
        std::size_t local_expert = 0;
        local_expert < expert_ids.size();
        ++local_expert
    ) {
        const int expert = expert_ids[local_expert];
        const auto base = checked_product(
            static_cast<std::size_t>(expert),
            static_cast<std::size_t>(kDescriptorSize),
            "descriptor offset");
        descriptors[base + kFamily] = kFamilyVq;
        descriptors[base + kLocalExpert] =
            checked_int(local_expert, "local expert");
        descriptors[base + kOut] = out_per_expert;
        descriptors[base + kInput] = neuron_len;
        descriptors[base + kVqGroupSize] =
            weight.group_size;
        descriptors[base + kVqGroups] =
            weight.groups;
        descriptors[base + kVqVectorSize] =
            weight.vector_size;
        descriptors[base + kVqVectors] =
            weight.vectors;
        descriptors[base + kVqIndexBits] =
            weight.index_bits;
        descriptors[base + kVqStateBits] =
            weight.state_bits;
        descriptors[base + kVqStates] =
            weight.states;
        descriptors[base + kVqEntries] =
            weight.entries;
        descriptors[base + kVqCodeBanks] =
            weight.code_banks;
        descriptors[base + kVqAuxMode] =
            weight.aux_mode;
        descriptors[base + kVqCodeBankMode] =
            weight.code_bank_mode;
        descriptors[base + kVqHasTableBanks] =
            static_cast<int>(
                weight.table_banks > 1);
        descriptors[base + kVqGroupsPerSuper] =
            weight.groups_per_supergroup;
        descriptors[base + kVqSupergroups] =
            weight.supergroups;
        descriptors[base + kVqIndicesOffset] =
            indices_offset;
        descriptors[base + kVqStateOffset] =
            state_offset;
        descriptors[base + kVqAuxOffset] =
            aux_offset;
        descriptors[base + kVqAnchorOffset] =
            anchor_offset;
        descriptors[base + kVqCodebookOffset] =
            codebook_offset;
        descriptors[base + kVqScaleOffset] =
            scale_offset;
        descriptors[base + kVqStateBankOffset] =
            state_bank_offset;
        descriptors[base + kVqBankOffset] =
            bank_offset;
        descriptors[base + kVqParameterOffset] =
            parameter_offset;
        descriptors[base + kVqRotationVariant] =
            rotation;
        const auto& profile = weight.label;
        const int execution_profile =
            profile == "NVQ2J-L" || profile == "NVQ2J-XL"
                ? kVqProfileJscExtended8
                : profile == "NVQ3J-512" || profile == "NVQ3J-L"
                ? kVqProfileJscExtended4
                : profile == "NVQ2J" || profile == "NVQ3J"
                    || (native_jsc && weight.index_bits == 8)
                ? (
                    weight.vector_size == 4
                        ? kVqProfileJsc4
                        : kVqProfileJsc8
                )
                : (
                    profile == "NPQ0-S" || profile == "NPQ0-L"
                        ? (
                            profile == "NPQ0-S"
                                ? kVqProfileNpqS
                                : kVqProfileNpqL
                        )
                        : (
                            packed_nvq1l_execution
                                ? kVqProfileNvq1L
                                : (
                                    packed_nvq1s_execution
                                        ? kVqProfileNvq1S
                                        : kVqProfileGeneric
                                )
                        )
                );
        descriptors[base + kVqProfile] =
            execution_profile
            | (weight.residual_position_bits << 8)
            | (weight.residual_block_vectors << 16);
        descriptors[base + kVqExecutionLayout] =
            group64_execution
                ? 2 : 0;
        descriptors[base + kVqResidualCodebookOffset] =
            residual_codebook_offset;
        descriptors[base + kVqResidualRecordOffset] =
            residual_record_offset;
    }

    streams.vq_indices.append(weight.borrowed_indices);
    streams.vq_indices.append(std::move(weight.indices));
    streams.vq_state.append(weight.borrowed_state);
    streams.vq_state.append(std::move(weight.states_packed));
    if (!group64_execution) {
        streams.vq_aux.append(weight.borrowed_auxiliary);
        streams.vq_aux.append(std::move(weight.auxiliary));
    }
    streams.vq_anchors.append(std::move(weight.anchors));
    streams.vq_codebooks.append(std::span(
        reinterpret_cast<const std::uint8_t*>(weight.borrowed_codebooks.data()),
        weight.borrowed_codebooks.size()));
    streams.vq_codebooks.append(std::move(weight.codebooks));
    streams.vq_scales.append(std::move(weight.scales));
    streams.vq_state_to_codebank.append(std::move(weight.state_to_codebank));
    streams.vq_banks.append(std::move(weight.bank_ids));
    streams.vq_parameters.append(weight.parameter);
    if (weight.residual_position_bits != 0) {
        streams.vq_residual_codebooks.append(std::move(weight.residual_codebook));
        streams.vq_residual_first.append(std::move(weight.residual_first));
        streams.vq_residual_second.append(std::move(weight.residual_second));
    }
    return std::nullopt;
}

bool is_ascii(
    std::span<const std::uint8_t> value) {
    return std::all_of(
        value.begin(),
        value.end(),
        [](std::uint8_t byte) {
            return byte <= 0x7fu;
        });
}

struct MfePoolView {
    std::vector<std::int32_t> expert_ids;
    std::string dtype;
    std::span<const std::uint8_t> runtime;
    std::span<const std::uint8_t> payload;
};

struct MfeProjectionView {
    int experts;
    int output;
    int input;
    std::vector<MfePoolView> pools;
};

MfeProjectionView inspect_nint_projection(std::span<const std::uint8_t> blob) {
    BlobCursor cursor(blob);
    const auto bits = cursor.scalar<std::uint8_t>("NINT bits") & 127;
    const auto sub_bits = cursor.scalar<std::uint8_t>("NINT sub bits");
    const auto group_size = cursor.scalar<std::int32_t>("NINT group size");
    const auto axis = cursor.scalar<std::int32_t>("NINT axis");
    const auto input = cursor.scalar<std::int32_t>("NINT input width");
    const auto rank = cursor.scalar<std::uint32_t>("NINT rank");
    if (bits < 1 || bits > 8 || sub_bits < 1 || sub_bits > 8 ||
        group_size <= 0 || rank != 3 || axis != 0 || input <= 0)
        throw std::runtime_error("routed NINT tensor must use [experts,out,in]");
    const int experts = checked_int(checked_size(cursor.scalar<std::int64_t>("NINT experts"), "NINT experts"), "NINT experts");
    const int output = checked_int(checked_size(cursor.scalar<std::int64_t>("NINT output"), "NINT output"), "NINT output");
    if (experts <= 0 || output <= 0 || cursor.scalar<std::int64_t>("NINT input") != input)
        throw std::runtime_error("routed NINT tensor shape disagrees");
    const auto rows = checked_int(checked_product(static_cast<std::size_t>(experts),
        static_cast<std::size_t>(output), "routed NINT rows"), "routed NINT rows");
    if (cursor.scalar<std::uint32_t>("NINT rows") != rows ||
        cursor.scalar<std::uint32_t>("NINT groups") != (static_cast<std::int64_t>(input) + group_size - 1) / group_size ||
        static_cast<std::size_t>(rows) > cursor.remaining() / 4)
        throw std::runtime_error("routed NINT tensor row metadata disagrees");
    MfePoolView pool;
    pool.expert_ids.resize(experts);
    std::iota(pool.expert_ids.begin(), pool.expert_ids.end(), 0);
    pool.dtype = "NINT";
    pool.payload = blob;
    return {experts, output, input, {std::move(pool)}};
}

MfeProjectionView inspect_mfe_projection(std::span<const std::uint8_t> blob) {
    BlobCursor cursor(blob);
    const auto magic_bytes = cursor.bytes(4, "header");
    const std::string_view magic(
        reinterpret_cast<const char*>(magic_bytes.data()), magic_bytes.size());
    if (magic != "MFE1" && magic != "NIM1" && magic != "NIM2") {
        throw std::runtime_error("invalid MFE magic");
    }
    MfeProjectionView result{
        checked_positive(cursor.scalar<std::uint32_t>("expert count"), "expert count"),
        checked_positive(cursor.scalar<std::uint32_t>("output width"), "output width"),
        checked_positive(cursor.scalar<std::uint32_t>("neuron length"), "neuron length"),
        {},
    };
    const auto pool_count = cursor.scalar<std::uint32_t>("pool count");
    if (pool_count == 0 || pool_count > static_cast<std::uint32_t>(result.experts)) {
        throw std::runtime_error("invalid MFE pool count");
    }
    if (static_cast<std::size_t>(result.experts) > cursor.remaining() / sizeof(std::int32_t)) {
        throw std::runtime_error("MFE expert count exceeds its payload");
    }
    std::vector<int> owners(static_cast<std::size_t>(result.experts), -1);
    for (std::uint32_t pool = 0; pool < pool_count; ++pool) {
        const auto count = cursor.scalar<std::uint32_t>("pool header");
        const auto dtype_bytes = magic == "NIM1" ? 0u
            : cursor.scalar<std::uint32_t>("v2 pool header");
        const auto payload_bytes = cursor.scalar<std::uint64_t>("pool header");
        const auto runtime_bytes = magic == "NIM1" ? std::uint64_t{0}
            : cursor.scalar<std::uint64_t>("v2 pool header");
        if (count == 0 || count > static_cast<std::uint32_t>(result.experts) ||
            (magic != "NIM1" && (dtype_bytes == 0 || dtype_bytes > 32))) {
            throw std::runtime_error("invalid MFE pool metadata");
        }
        MfePoolView current;
        current.expert_ids.resize(count);
        for (auto& expert : current.expert_ids) {
            expert = cursor.scalar<std::int32_t>("expert IDs");
        }
        claim_experts(current.expert_ids, owners, static_cast<int>(pool));
        current.dtype = "NINT";
        if (magic != "NIM1") {
            const auto raw_dtype = cursor.bytes(dtype_bytes, "cohort dtype");
            if (!is_ascii(raw_dtype)) {
                throw std::runtime_error("MFE cohort dtype must be ASCII");
            }
            current.dtype = std::string(tyloi::canonical_format_dtype(std::string_view(
                reinterpret_cast<const char*>(raw_dtype.data()), raw_dtype.size())));
        }
        current.runtime = cursor.bytes(
            checked_size(runtime_bytes, "cohort runtime metadata"), "cohort runtime metadata");
        current.payload = cursor.bytes(
            checked_size(payload_bytes, "cohort payload"), "cohort payload");
        result.pools.push_back(std::move(current));
    }
    if (cursor.remaining() != 0) {
        throw std::runtime_error("trailing bytes in MFE tensor");
    }
    if (std::find(owners.begin(), owners.end(), -1) != owners.end()) {
        throw std::runtime_error("MFE pools do not cover every expert");
    }
    return result;
}

struct ProjectionStorageInfo {
    std::size_t packed_bytes;
    bool automatic_mxfp4_nax_prefill;
    int logical_experts;
};

array concatenate_1d(
    std::vector<array> values) {
    if (values.empty()) {
        throw std::invalid_argument("cannot concatenate an empty packed stream list");
    }
    const auto empty = empty_packed_array(values.front().dtype());
    if (std::all_of(values.begin(), values.end(), [&](const array& value) {
            return value.data_shared_ptr() == empty.data_shared_ptr();
        })) {
        return empty;
    }
    const bool all_vectors = std::all_of(
        values.begin(),
        values.end(),
        [](const array& value) {
            return value.ndim() == 1;
        });
    if (!all_vectors) {
        const auto dtype = values.front().dtype();
        std::size_t total_bytes = 0;
        for (auto& value : values) {
            if (value.dtype() != dtype || !value.flags().row_contiguous) {
                throw std::invalid_argument(
                    "packed stream concatenation requires contiguous matching dtypes");
            }
            value.eval();
            total_bytes = checked_add(
                total_bytes,
                value.nbytes(),
                "packed concatenation bytes");
        }
        tyloi::metal::StagingVector<std::uint8_t> bytes(total_bytes);
        std::size_t offset = 0;
        for (const auto& value : values) {
            std::memcpy(
                bytes.data() + offset,
                value.data<std::uint8_t>(),
                value.nbytes());
            offset += value.nbytes();
        }
        return make_raw_array(std::move(bytes), dtype);
    }
    return mlx::core::contiguous(
        mlx::core::concatenate(
            std::move(values),
            0));
}

array apply_rotation(
    const array& source,
    const RotationSpec& rotation,
    int rows,
    int width) {
    if (
        source.ndim() != 2
        || source.shape(0) != rows
        || source.shape(1) != width
        || rotation.block <= 0
        || rotation.block > 8192
        || (
            rotation.block
            & (rotation.block - 1)
        ) != 0
        || width % rotation.block != 0
        || rotation.signs.dtype()
            != mlx::core::int8
        || rotation.signs.size()
            != static_cast<std::size_t>(width)
    ) {
        throw std::runtime_error(
            "invalid MFE HSG1 rotation layout");
    }
    const auto grid = checked_product(
        static_cast<std::size_t>(rows),
        256,
        "HSG1 Metal grid");
    if (
        grid
        > static_cast<std::size_t>(
            std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error(
            "MFE HSG1 Metal grid exceeds MLX limits");
    }
    auto outputs = moe_hadamard_kernel()(
        {source, rotation.signs},
        {Shape{rows, width}},
        {source.dtype()},
        {
            static_cast<int>(grid),
            1,
            1,
        },
        {256, 1, 1},
        {
            {"T", source.dtype()},
            {"K", width},
            {"BLOCK", rotation.block},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

int descriptor_with_offset(
    int value,
    std::size_t offset,
    const char* name) {
    if (value < 0) {
        throw std::runtime_error(
            std::string("invalid MFE ") + name);
    }
    return checked_int(
        checked_add(
            static_cast<std::size_t>(value),
            offset,
            name),
        name);
}

std::int32_t descriptor_u32_with_offset(
    std::int32_t value,
    std::size_t offset,
    const char* name) {
    return checked_descriptor_u32(
        checked_add(
            static_cast<std::size_t>(
                descriptor_u32_value(value)),
            offset,
            name),
        name);
}


} // namespace

namespace mfe_internal {
std::size_t checked_product(std::size_t a, std::size_t b, const char* n) { return ::tyloi::metal::checked_product(a, b, n); }
std::size_t checked_add(std::size_t a, std::size_t b, const char* n) { return ::tyloi::metal::checked_add(a, b, n); }
std::size_t checked_packed_size(std::size_t a, int b, const char* n) { return ::tyloi::metal::checked_packed_size(a, b, n); }
std::int32_t checked_int(std::size_t a, const char* n) { return ::tyloi::metal::checked_int(a, n); }
array allocate_packed_array(std::size_t n, Dtype t) { return ::tyloi::metal::allocate_packed_array(n, t); }
array allocate_wire_array(std::size_t n, bool account) { return ::tyloi::metal::allocate_wire_array(n, account); }
bool uses_indirect_expert_pages(std::uint32_t mask) { return ::tyloi::metal::uses_indirect_expert_pages(mask); }
void prepare_packed_copy() {
    PackedCopyPrimitive(mlx::core::default_stream(mlx::core::Device::gpu), Shape{1}, {}).prepare_gpu();
}
void collect_packed_copy() {
    MlxKernelPreparation::current()->add(packed_expert_copy({empty_packed_array(mlx::core::uint8)},
        {{PackedCopySpec{0, 0, 0, 1}}}, 1));
}
}

struct MlxMfeWeight::Impl {
    array descriptors;
    std::shared_ptr<const MfePageResources> page_resources;
    std::optional<array> indirect_addresses;
    std::vector<array> indirect_owners;
    std::shared_ptr<const std::vector<MlxMfeWeight>> indirect_sources;
    std::optional<array> packed_arena;
    array nint_q;
    array nint_sub_scale;
    array nint_sub_min;
    array nint_anchor_scale;
    array nint_anchor_min;
    array q8_q;
    array q8_scales;
    array vq_indices;
    array vq_state;
    array vq_aux;
    array vq_anchors;
    array vq_codebooks;
    array vq_scales;
    array vq_state_to_codebank;
    array vq_banks;
    array vq_parameters;
    array vq_residual_codebooks;
    array vq_residual_first;
    array vq_residual_second;
    array mx_values;
    array mx_scales;
    // Metal kernels address packed pools through descriptor offsets, so their
    // logical MLX shape is not part of the ABI. Prefix views retain the same
    // backing buffer and base address while preventing MLX's command-buffer
    // heuristic from counting every resident expert as touched by each
    // decode launch.
    std::vector<array> narrow_bindings;
    std::vector<RotationSpec> rotations;
    std::vector<std::int32_t> descriptor_values;
    std::vector<std::shared_ptr<const Impl>> projection_views;
    std::optional<array> expert_order;
    std::vector<ReferenceMoeCohort> reference_cohorts;
    std::vector<Mxfp4SqMoeCohort> mxfp4_sq_cohorts;
    std::vector<Fp8SqMoeCohort> fp8_sq_cohorts;
    std::optional<array> standalone_owner;
    bool has_generic_cohorts = false;
    int experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    int projections = 0;
    // Physical SSD arenas may contain many more rows than one model layer.
    // Keep the logical expert geometry separately so hardware policy does not
    // confuse cache capacity with the routed operator it is executing.
    int logical_experts = 0;
    std::uint32_t family_mask = 0;
    std::uint32_t vq_profile_mask = 0;
    std::shared_ptr<const std::vector<int>> nint_group_sizes;
    std::shared_ptr<const MlxMfeKernelProfile> kernel_profile;
    std::shared_ptr<const std::vector<std::tuple<int, int, int>>> cohort_populations;
    bool vq_execution_layout = false;
    std::uint64_t vq_layouts = 0;
    bool npq_grouped_indices = true;
    bool native_primitive = true;
    bool grouped_mmq = false;
    bool has_nepq_residual = false;
    std::optional<array> mxfp4_slot_ids;
    bool mxfp4_slot_ids_sorted = false;
    bool automatic_mxfp4_nax_prefill = true;
    int k_lanes_override = 0;
    bool nint_q_dispatch = true;
    std::size_t packed_bytes = 0;

    void collect_cache_storage(
        std::vector<array>& roots,
        std::vector<std::shared_ptr<tyloi::TensorStorage>>& nested) const;

    MlxMfeKernelProfile compilation_profile() const {
        return kernel_profile ? *kernel_profile
            : MlxMfeKernelProfile{family_mask, vq_profile_mask, vq_layouts,
                nint_group_sizes, vq_execution_layout, nint_q_dispatch};
    }

    Impl(
        array descriptor_array,
        array nint_q_array,
        array nint_sub_scale_array,
        array nint_sub_min_array,
        array nint_anchor_scale_array,
        array nint_anchor_min_array,
        array q8_q_array,
        array q8_scale_array,
        array vq_indices_array,
        array vq_state_array,
        array vq_aux_array,
        array vq_anchor_array,
        array vq_codebook_array,
        array vq_scale_array,
        array vq_state_bank_array,
        array vq_bank_array,
        array vq_parameter_array,
        array vq_residual_codebook_array,
        array vq_residual_first_array,
        array vq_residual_second_array,
        array mx_value_array,
        array mx_scale_array,
        std::vector<RotationSpec> rotation_values,
        std::vector<std::int32_t> values,
        int expert_count,
        int output_width,
        int input_width,
        int projection_count)
        : descriptors(std::move(descriptor_array)),
          nint_q(std::move(nint_q_array)),
          nint_sub_scale(
              std::move(nint_sub_scale_array)),
          nint_sub_min(
              std::move(nint_sub_min_array)),
          nint_anchor_scale(
              std::move(nint_anchor_scale_array)),
          nint_anchor_min(
              std::move(nint_anchor_min_array)),
          q8_q(std::move(q8_q_array)),
          q8_scales(std::move(q8_scale_array)),
          vq_indices(std::move(vq_indices_array)),
          vq_state(std::move(vq_state_array)),
          vq_aux(std::move(vq_aux_array)),
          vq_anchors(std::move(vq_anchor_array)),
          vq_codebooks(std::move(vq_codebook_array)),
          vq_scales(std::move(vq_scale_array)),
          vq_state_to_codebank(
              std::move(vq_state_bank_array)),
          vq_banks(std::move(vq_bank_array)),
          vq_parameters(
              std::move(vq_parameter_array)),
          vq_residual_codebooks(
              std::move(vq_residual_codebook_array)),
          vq_residual_first(
              std::move(vq_residual_first_array)),
          vq_residual_second(
              std::move(vq_residual_second_array)),
          mx_values(std::move(mx_value_array)),
          mx_scales(std::move(mx_scale_array)),
          rotations(std::move(rotation_values)),
          descriptor_values(std::move(values)),
          experts(expert_count),
          out_per_expert(output_width),
          neuron_len(input_width),
          projections(projection_count) {
        grouped_mmq = !descriptor_values.empty();
        std::vector<int> group_sizes;
        for (
            std::size_t base = 0;
            base + kDescriptorSize
                <= descriptor_values.size();
            base += kDescriptorSize
        ) {
            const auto family = descriptor_values[
                base + kFamily];
            if (family >= 0 && family < 7) {
                family_mask |= std::uint32_t{1}
                    << static_cast<unsigned>(family);
            }
            if (family == kFamilyNint) {
                nint_q_dispatch &= descriptor_values[base + kNintQDispatch] != 0;
                const auto group_size = descriptor_values[base + kNintGroupSize];
                if (group_size > 0 && std::find(group_sizes.begin(), group_sizes.end(), group_size)
                    == group_sizes.end()) group_sizes.push_back(group_size);
            }
            if (family == kFamilyVq) {
                const auto profile = descriptor_values[
                    base + kVqProfile] & 255;
                if (profile >= 0 && profile < 9) {
                    const auto shift = static_cast<unsigned>(profile) * 7u;
                    const auto layout = descriptor_values[base + kVqExecutionLayout];
                    vq_layouts |= (layout >= 0 && layout < 7
                        ? std::uint64_t{1} << layout : std::uint64_t{127}) << shift;
                    vq_profile_mask |= std::uint32_t{1}
                        << static_cast<unsigned>(profile);
                }
                if ((descriptor_values[base + kVqProfile] >> 8) != 0) {
                    has_nepq_residual = true;
                }
            }
            const bool supported_nint =
                family == kFamilyNint
                && descriptor_values[base + kNintGroupSize] > 0
                && descriptor_values[base + kNintGroups]
                    == (neuron_len
                        + descriptor_values[base + kNintGroupSize] - 1)
                        / descriptor_values[base + kNintGroupSize]
                && descriptor_values[base + kNintV2] != 0;
            const bool supported_q8 =
                family == kFamilyNint8Zero
                && descriptor_values[base + kQ8Groups]
                    == neuron_len / 32;
            const bool supported_vq =
                family == kFamilyVq
                && descriptor_values[base + kVqGroupSize] == 24
                && (
                    descriptor_values[base + kVqVectorSize] == 4
                    || descriptor_values[base + kVqVectorSize] == 8
                );
            const bool supported_mxfp4 =
                family == kFamilyMxfp4
                && descriptor_values[base + kMxGroups]
                    == neuron_len / 32;
            const bool supported_mxfp8 =
                family == kFamilyMxfp8
                && neuron_len % 128 == 0
                && descriptor_values[base + kMxGroups]
                    == neuron_len / 128;
            const bool supported_dense =
                family == kFamilyBf16 || family == kFamilyF16;
            if (!supported_nint && !supported_q8
                && !supported_vq && !supported_mxfp4
                && !supported_mxfp8 && !supported_dense) {
                grouped_mmq = false;
            }
            if (
                descriptor_values[
                    base + kVqExecutionLayout] != 0
            ) {
                vq_execution_layout = true;
            }
        }
        std::sort(group_sizes.begin(), group_sizes.end());
        nint_group_sizes = std::make_shared<const std::vector<int>>(std::move(group_sizes));
        std::vector<std::tuple<int, int, int>> populations;
        if (projections == 1) {
            for (int family = 0; family < 7; ++family) {
                if ((family_mask & (std::uint32_t{1} << family)) == 0) continue;
                const auto groups = family == kFamilyNint ? *nint_group_sizes : std::vector<int>{0};
                for (const int group : groups) {
                    int count = 0;
                    for (std::size_t base = 0; base + kDescriptorSize <= descriptor_values.size(); base += kDescriptorSize)
                        if (descriptor_values[base + kFamily] == family
                            && (group == 0 || descriptor_values[base + kNintGroupSize] == group)) ++count;
                    populations.emplace_back(1 << family, group, count);
                }
            }
        }
        cohort_populations = std::make_shared<const std::vector<std::tuple<int, int, int>>>(std::move(populations));
        if (
            descriptor_values.size()
                == static_cast<std::size_t>(experts) * kDescriptorSize
        ) {
            std::vector<std::int32_t> schedule(
                static_cast<std::size_t>(experts));
            std::iota(schedule.begin(), schedule.end(), 0);
            // Keep adjacent threadgroups on one exact decoder cohort and walk
            // each cohort's packed rows monotonically. Quantization assigns
            // formats independently of global expert ID, so the default ID
            // order otherwise jumps among unrelated packed streams on every
            // block. Row destinations remain route-order based; this only
            // changes execution order.
            std::stable_sort(
                schedule.begin(),
                schedule.end(),
                [this](std::int32_t left, std::int32_t right) {
                    const auto left_base = static_cast<std::size_t>(left)
                        * kDescriptorSize;
                    const auto right_base = static_cast<std::size_t>(right)
                        * kDescriptorSize;
                    const int left_family =
                        descriptor_values[left_base + kFamily];
                    const int right_family =
                        descriptor_values[right_base + kFamily];
                    const auto compare_field = [&](int field) {
                        if (
                            left_family == right_family
                            && (
                                left_family == kFamilyMxfp4
                                || left_family == kFamilyMxfp8
                            )
                            && (
                                field == kMxValueOffset
                                || field == kMxScaleOffset
                            )
                        ) {
                            const auto left_value = descriptor_u32_value(
                                descriptor_values[left_base + field]);
                            const auto right_value = descriptor_u32_value(
                                descriptor_values[right_base + field]);
                            return left_value < right_value
                                ? -1
                                : left_value > right_value ? 1 : 0;
                        }
                        return descriptor_values[left_base + field]
                            < descriptor_values[right_base + field]
                            ? -1
                            : descriptor_values[left_base + field]
                                > descriptor_values[right_base + field]
                            ? 1
                            : 0;
                    };
                    if (const int family = compare_field(kFamily);
                        family != 0) {
                        return family < 0;
                    }
                    for (int field = 4; field < kDescriptorSize; ++field) {
                        if (const int value = compare_field(field);
                            value != 0) {
                            return value < 0;
                        }
                    }
                    return descriptor_values[left_base + kLocalExpert]
                        < descriptor_values[right_base + kLocalExpert];
                });
            expert_order.emplace(make_int32_array(
                schedule,
                Shape{experts}));
        }
        grouped_mmq = grouped_mmq && expert_order.has_value();
        if (
            family_mask == (std::uint32_t{1} << kFamilyMxfp4)
            && descriptor_values.size()
                == static_cast<std::size_t>(experts) * kDescriptorSize
        ) {
            const auto value_stride = checked_product(
                static_cast<std::size_t>(out_per_expert),
                static_cast<std::size_t>(neuron_len / 2),
                "MXFP4 slot value stride");
            const auto scale_stride = checked_product(
                static_cast<std::size_t>(out_per_expert),
                static_cast<std::size_t>(neuron_len / 32),
                "MXFP4 slot scale stride");
            std::vector<std::int32_t> slots(
                static_cast<std::size_t>(experts));
            bool valid_slots = value_stride != 0 && scale_stride != 0
                && mx_values.size() % value_stride == 0
                && mx_scales.size() % scale_stride == 0
                && mx_values.size() / value_stride
                    == mx_scales.size() / scale_stride;
            for (int expert = 0; expert < experts; ++expert) {
                const auto base = static_cast<std::size_t>(expert)
                    * kDescriptorSize;
                const int local = descriptor_values[base + kLocalExpert];
                const auto value_offset = descriptor_u32_value(
                    descriptor_values[base + kMxValueOffset]);
                const auto scale_offset = descriptor_u32_value(
                    descriptor_values[base + kMxScaleOffset]);
                if (!valid_slots || local < 0
                    || static_cast<std::size_t>(value_offset) % value_stride
                        != 0
                    || static_cast<std::size_t>(scale_offset) % scale_stride
                        != 0) {
                    valid_slots = false;
                    break;
                }
                const auto value_slot =
                    static_cast<std::size_t>(value_offset) / value_stride
                    + static_cast<std::size_t>(local);
                const auto scale_slot =
                    static_cast<std::size_t>(scale_offset) / scale_stride
                    + static_cast<std::size_t>(local);
                if (value_slot != scale_slot
                    || value_slot >= mx_values.size() / value_stride) {
                    valid_slots = false;
                    break;
                }
                slots[static_cast<std::size_t>(expert)] = checked_int(
                    value_slot,
                    "MXFP4 physical expert slot");
            }
            if (valid_slots) {
                mxfp4_slot_ids_sorted =
                    std::is_sorted(slots.begin(), slots.end());
                mxfp4_slot_ids.emplace(make_int32_array(
                    slots,
                    Shape{experts}));
            }
        }
        const char* specialize_env = std::getenv(
            "TYLOINFER_METAL_MFE_SPECIALIZE");
        if (
            specialize_env != nullptr
            && std::string_view(specialize_env) == "0"
        ) {
            family_mask = 126;
            vq_profile_mask = 511;
        }
        const char* npq_indices_env = std::getenv(
            "TYLOINFER_METAL_MFE_NPQ_INDICES");
        npq_grouped_indices =
            npq_indices_env == nullptr
            || std::string_view(npq_indices_env) != "0";
        const char* native_env = std::getenv(
            "TYLOINFER_METAL_MFE_NATIVE_PRIMITIVE");
        native_primitive = native_env == nullptr
            || std::string_view(native_env) != "0";
        const char* k_lanes_env = std::getenv(
            "TYLOINFER_METAL_MFE_K_LANES");
        if (k_lanes_env != nullptr) {
            const auto value = std::string_view(k_lanes_env);
            if (value == "16") {
                k_lanes_override = 16;
            } else if (value == "32") {
                k_lanes_override = 32;
            }
        }
        packed_bytes =
            descriptors.nbytes()
            + nint_q.nbytes()
            + nint_sub_scale.nbytes()
            + nint_sub_min.nbytes()
            + nint_anchor_scale.nbytes()
            + nint_anchor_min.nbytes()
            + q8_q.nbytes()
            + q8_scales.nbytes()
            + vq_indices.nbytes()
            + vq_state.nbytes()
            + vq_aux.nbytes()
            + vq_anchors.nbytes()
            + vq_codebooks.nbytes()
            + vq_scales.nbytes()
            + vq_state_to_codebank.nbytes()
            + vq_banks.nbytes()
            + vq_parameters.nbytes()
            + vq_residual_codebooks.nbytes()
            + vq_residual_first.nbytes()
            + vq_residual_second.nbytes()
            + mx_values.nbytes()
            + mx_scales.nbytes();
        for (const auto& rotation : rotations) {
            packed_bytes += rotation.signs.nbytes();
        }

        const auto narrow = [](const array& source) {
            if (source.ndim() == 0 || source.size() <= 1) {
                return source;
            }
            Shape begin(static_cast<std::size_t>(source.ndim()), 0);
            Shape end = source.shape();
            end.front() = 1;
            return mlx::core::slice(source, std::move(begin), std::move(end));
        };
        narrow_bindings = {
            descriptors,
            narrow(nint_q),
            narrow(nint_sub_scale),
            narrow(nint_sub_min),
            narrow(nint_anchor_scale),
            narrow(nint_anchor_min),
            narrow(q8_q),
            narrow(q8_scales),
            narrow(vq_indices),
            narrow(vq_state),
            narrow(vq_aux),
            narrow(vq_anchors),
            narrow(vq_codebooks),
            narrow(vq_scales),
            narrow(vq_state_to_codebank),
            narrow(vq_banks),
            narrow(vq_parameters),
            narrow(vq_residual_codebooks),
            narrow(vq_residual_first),
            narrow(vq_residual_second),
            narrow(mx_values),
            narrow(mx_scales),
        };
    }

    void bind_projection_views(const std::vector<ProjectionStorageInfo>& sources) {
        projection_views.clear();
        projection_views.reserve(sources.size());
        for (std::size_t projection = 0; projection < sources.size(); ++projection) {
            std::vector<std::int32_t> values;
            values.reserve(static_cast<std::size_t>(experts) * kDescriptorSize);
            for (int expert = 0; expert < experts; ++expert) {
                const auto begin = descriptor_values.begin() +
                    (static_cast<std::size_t>(expert) * sources.size() + projection) * kDescriptorSize;
                values.insert(values.end(), begin, begin + kDescriptorSize);
            }
            auto view = std::make_shared<Impl>(
                make_int32_array(values, Shape{experts, kDescriptorSize}),
                nint_q, nint_sub_scale, nint_sub_min, nint_anchor_scale, nint_anchor_min,
                q8_q, q8_scales, vq_indices, vq_state, vq_aux, vq_anchors,
                vq_codebooks, vq_scales, vq_state_to_codebank, vq_banks, vq_parameters,
                vq_residual_codebooks, vq_residual_first, vq_residual_second,
                mx_values, mx_scales, rotations, std::move(values),
                experts, out_per_expert, neuron_len, 1);
            view->automatic_mxfp4_nax_prefill = sources[projection].automatic_mxfp4_nax_prefill;
            view->logical_experts = sources[projection].logical_experts;
            view->packed_bytes = sources[projection].packed_bytes;
            projection_views.push_back(std::move(view));
        }
    }
};

void MlxMfeWeight::Impl::collect_cache_storage(
    std::vector<array>& roots,
    std::vector<std::shared_ptr<tyloi::TensorStorage>>& nested) const {
    std::vector<const Impl*> pending{this};
    std::unordered_set<const Impl*> visited;
    while (!pending.empty()) {
        const auto* impl = pending.back();
        pending.pop_back();
        if (!visited.insert(impl).second) continue;
        for (const auto* value : {&impl->descriptors, &impl->nint_q, &impl->nint_sub_scale,
                &impl->nint_sub_min, &impl->nint_anchor_scale, &impl->nint_anchor_min,
                &impl->q8_q, &impl->q8_scales, &impl->vq_indices, &impl->vq_state,
                &impl->vq_aux, &impl->vq_anchors, &impl->vq_codebooks, &impl->vq_scales,
                &impl->vq_state_to_codebank, &impl->vq_banks, &impl->vq_parameters,
                &impl->vq_residual_codebooks, &impl->vq_residual_first,
                &impl->vq_residual_second, &impl->mx_values, &impl->mx_scales}) {
            roots.push_back(*value);
        }
        for (const auto* value : {&impl->indirect_addresses, &impl->packed_arena,
                &impl->expert_order, &impl->standalone_owner, &impl->mxfp4_slot_ids}) {
            if (*value) roots.push_back(**value);
        }
        for (const auto& value : impl->indirect_owners) roots.push_back(value);
        if (impl->page_resources)
            for (const auto& value : impl->page_resources->owners)
                roots.push_back(value);
        for (const auto& value : impl->narrow_bindings) roots.push_back(value);
        for (const auto& rotation : impl->rotations) roots.push_back(rotation.signs);
        for (const auto& cohort : impl->reference_cohorts) {
            std::visit([&](const auto& weight) {
                using Weight = std::decay_t<decltype(weight)>;
                if constexpr (std::is_same_v<Weight, DenseReferenceMoeWeight>) {
                    roots.push_back(weight.values);
                } else {
                    auto values = weight.backings();
                    nested.insert(nested.end(), values.begin(), values.end());
                }
            }, cohort.weight);
        }
        for (const auto& cohort : impl->mxfp4_sq_cohorts) {
            roots.push_back(cohort.expert_map);
            auto values = cohort.weight.backings();
            nested.insert(nested.end(), values.begin(), values.end());
        }
        for (const auto& cohort : impl->fp8_sq_cohorts) {
            roots.push_back(cohort.expert_map);
            auto values = cohort.weight.backings();
            nested.insert(nested.end(), values.begin(), values.end());
        }
        for (const auto& view : impl->projection_views)
            pending.push_back(view.get());
        if (impl->indirect_sources)
            for (const auto& source : *impl->indirect_sources)
                pending.push_back(source.impl_.get());
    }
}

MlxMfeWeight::MlxMfeWeight(
    std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl)) {
    if (!impl_) {
        throw std::invalid_argument(
            "MFE implementation cannot be null");
    }
    if (auto* preparation = MlxKernelPreparation::current(); preparation && !preparation->model_paths() && preparation->routes() > 0
        && impl_->reference_cohorts.empty() && !mlx_reference_enabled()) {
        preparation->collect([&] {
            const int routes = preparation->routes();
            const auto prepare = [&](const MlxMfeWeight& weight, bool projection_view) {
                for (const auto rows : preparation->row_buckets()) {
                    if (projection_view && rows > 8) continue;
                    const auto ids = mlx::core::zeros(Shape{rows, routes}, mlx::core::int32);
                    const auto shared = mlx::core::zeros(Shape{rows, weight.neuron_len()}, mlx::core::float16);
                    const auto routed = mlx::core::zeros(Shape{rows, routes, weight.neuron_len()}, mlx::core::float16);
                    preparation->add(weight.routed_matmul(shared, ids));
                    if (!projection_view) preparation->add(weight.routed_matmul(routed, ids));
                    if (weight.projections() == 2) preparation->add(weight.routed_swiglu(shared, ids));
                    if (weight.projections() == 1 && rows > 1 && rows <= 24) {
                        const auto groups = mlx::core::zeros(Shape{rows, routes, 8}, mlx::core::int32);
                        preparation->add(weight.routed_matmul(shared, ids, &groups));
                        if (!projection_view) preparation->add(weight.routed_matmul(routed, ids, &groups));
                    }
                    if (!projection_view && weight.supports_fused_routed_reduce()) preparation->add(weight.routed_matmul_reduce(routed, ids,
                        mlx::core::zeros(Shape{rows, routes}, mlx::core::float32)));
                }
            };
            prepare(*this, false);
            for (const auto& view : impl_->projection_views) prepare(MlxMfeWeight(view), true);
        });
    }
}

MlxMfeWeight MlxMfeWeight::from_blob(
    std::span<const std::uint8_t> blob) {
    const std::array blobs{blob};
    return from_projection_blobs(blobs);
}

MlxMfeWeight MlxMfeWeight::from_blob(const array& blob) {
    if (blob.dtype() != mlx::core::uint8 || !blob.flags().row_contiguous)
        throw std::invalid_argument("MFE wire storage must be contiguous bytes");
    auto storage = blob;
    storage.eval();
    const std::array blobs{std::span<const std::uint8_t>(storage.data<std::uint8_t>(), storage.nbytes())};
    return from_projection_blobs_impl(blobs, false, &storage);
}

MlxMfeWeight MlxMfeWeight::from_projection_blobs(
    std::span<const std::span<const std::uint8_t>> blobs) {
    return from_projection_blobs_impl(blobs, false);
}

MlxMfeWeight MlxMfeWeight::from_nint_blob(
    std::span<const std::uint8_t> blob) {
    const std::array blobs{blob};
    return from_projection_blobs_impl(blobs, true);
}

MlxMfeWeight MlxMfeWeight::from_nint_blob(const array& blob) {
    if (blob.dtype() != mlx::core::uint8 || !blob.flags().row_contiguous)
        throw std::invalid_argument("NINT wire storage must be contiguous bytes");
    auto storage = blob;
    storage.eval();
    const std::array blobs{std::span<const std::uint8_t>(storage.data<std::uint8_t>(), storage.nbytes())};
    return from_projection_blobs_impl(blobs, true, &storage);
}

MlxMfeWeight MlxMfeWeight::from_projection_blobs_impl(
    std::span<const std::span<const std::uint8_t>> blobs,
    bool nint_tensor,
    const array* storage,
    const array* destination) {
    if (blobs.empty()) {
        throw std::invalid_argument("at least one MFE projection is required");
    }
    const int projection_count = checked_int(blobs.size(), "projection count");
    std::vector<MfeProjectionView> projections;
    projections.reserve(blobs.size());
    for (const auto blob : blobs) {
        projections.push_back(nint_tensor
            ? inspect_nint_projection(blob) : inspect_mfe_projection(blob));
    }
    const int expert_count = projections.front().experts;
    const int output_width = projections.front().output;
    const int input_width = projections.front().input;
    for (const auto& projection : projections) {
        if (projection.experts != expert_count || projection.output != output_width ||
            projection.input != input_width) {
            throw std::invalid_argument("MFE projections have incompatible shapes");
        }
    }
    const bool reference = mlx_reference_enabled();
    if (projection_count > 1 && (reference || std::any_of(
        projections.begin(), projections.end(), [](const auto& projection) {
            return std::any_of(projection.pools.begin(), projection.pools.end(),
                [](const auto& pool) {
                    return !pool.runtime.empty() || is_mxfp4_sq_dtype(pool.dtype) ||
                        is_fp8_sq_dtype(pool.dtype);
                });
        }))) {
        // Standalone SQ/reference and rotated cohorts keep their established
        // ownership/dispatch path, without parsing or repacking weights twice.
        std::vector<MlxMfeWeight> weights;
        weights.reserve(blobs.size());
        for (const auto blob : blobs) weights.push_back(from_blob(blob));
        return concatenate_projections(weights).materialize_packed_projections();
    }

    const int descriptor_rows = checked_int(checked_product(
        static_cast<std::size_t>(expert_count), blobs.size(), "projection rows"),
        "projection rows");
    std::vector<std::int32_t> descriptors(checked_product(
        static_cast<std::size_t>(descriptor_rows),
        static_cast<std::size_t>(kDescriptorSize), "descriptor count"), 0);
    PackedStreams streams;
    if (storage) streams.retain_source(*storage);
    std::vector<RotationSpec> rotations;
    std::vector<ReferenceMoeCohort> reference_cohorts;
    std::vector<Mxfp4SqMoeCohort> mxfp4_sq_cohorts;
    std::vector<Fp8SqMoeCohort> fp8_sq_cohorts;
    std::vector<std::int32_t> standalone_owner(
        static_cast<std::size_t>(expert_count), -1);
    std::vector<ProjectionStorageInfo> projection_info;
    bool has_generic_cohorts = true;

    for (std::size_t projection = 0; projection < projections.size(); ++projection) {
        if ((streams.q8_q.size() & 1u) != 0u) streams.q8_q.push_back(0);
        const auto before = streams.size();
        std::vector<std::int32_t> local_descriptors(checked_product(
            static_cast<std::size_t>(expert_count), static_cast<std::size_t>(kDescriptorSize),
            "descriptor count"), 0);
        bool projection_generic = false;
        for (const auto& pool : projections[projection].pools) {
            const auto& expert_ids = pool.expert_ids;
            const auto count = expert_ids.size();
            const auto& dtype = pool.dtype;
            const auto runtime = pool.runtime;
            const auto payload = pool.payload;
            if (is_nint_dtype(dtype)) {
                projection_generic = true;
                if (!runtime.empty()) {
                    throw std::runtime_error(
                        "unexpected MFE NINT "
                        "runtime metadata");
                }
                auto weight = add_nint_pool(
                    payload,
                    expert_ids,
                    output_width,
                    input_width,
                    streams,
                    local_descriptors,
                    !reference);
                if (reference) {
                    reference_cohorts.push_back({
                        expert_ids,
                        std::move(*weight),
                    });
                }
            } else if (is_nint8_zero_dtype(dtype)) {
                projection_generic = true;
                if (!runtime.empty()) {
                    throw std::runtime_error(
                        "unexpected MFE NINT8-0 "
                        "runtime metadata");
                }
                auto weight = add_q8_pool(
                    payload,
                    expert_ids,
                    output_width,
                    input_width,
                    streams,
                    local_descriptors,
                    !reference);
                if (reference) {
                    reference_cohorts.push_back({
                        expert_ids,
                        std::move(weight),
                    });
                }
            } else if (dtype == "MXFP4" || dtype == "MXFP8") {
                projection_generic = true;
                if (!runtime.empty()) {
                    throw std::runtime_error(
                        "unexpected MFE MX runtime metadata");
                }
                auto weight = add_mx_pool(
                    dtype,
                    payload,
                    expert_ids,
                    output_width,
                    input_width,
                    streams,
                    local_descriptors,
                    !reference);
                if (reference) {
                    reference_cohorts.push_back({
                        expert_ids,
                        std::move(*weight),
                    });
                }
            } else if (dtype == "BF16" || dtype == "F16") {
                projection_generic = true;
                if (!runtime.empty()) {
                    throw std::runtime_error(
                        "unexpected MFE dense runtime metadata");
                }
                auto weight = add_dense_pool(
                    dtype,
                    payload,
                    expert_ids,
                    output_width,
                    input_width,
                    streams,
                    local_descriptors,
                    !reference);
                if (reference) {
                    reference_cohorts.push_back({
                        expert_ids,
                        std::move(*weight),
                    });
                }
            } else if (is_vq_dtype(dtype)) {
                projection_generic = true;
                auto weight = add_vq_pool(
                    dtype,
                    payload,
                    runtime,
                    expert_ids,
                    output_width,
                    input_width,
                    streams,
                    rotations,
                    local_descriptors,
                    !reference);
                if (reference) {
                    reference_cohorts.push_back({
                        expert_ids,
                        std::move(*weight),
                    });
                }
            } else if (is_mxfp4_sq_dtype(dtype)) {
                if (!runtime.empty()) {
                    throw std::runtime_error(
                        "unexpected MFE MXFP4-SQ runtime metadata");
                }
                auto weight = MlxMxfp4SqWeight::from_blob(payload);
                if (weight.input_size() != input_width ||
                    weight.output_size() != checked_int(
                        checked_product(
                            static_cast<std::size_t>(count),
                            static_cast<std::size_t>(output_width),
                            "MXFP4-SQ cohort output width"),
                        "MXFP4-SQ cohort output width")) {
                    throw std::runtime_error(
                        "MFE MXFP4-SQ cohort shape is inconsistent");
                }
                std::vector<std::int32_t> local_map(
                    static_cast<std::size_t>(expert_count), -1);
                const int cohort_index = checked_int(
                    mxfp4_sq_cohorts.size(),
                    "MXFP4-SQ cohort count");
                for (std::size_t local = 0; local < expert_ids.size(); ++local) {
                    const auto expert = expert_ids[local];
                    local_map[static_cast<std::size_t>(expert)] =
                        checked_int(local, "MXFP4-SQ local expert");
                    standalone_owner[static_cast<std::size_t>(expert)] =
                        cohort_index;
                }
                if (reference) {
                    reference_cohorts.push_back({expert_ids, weight});
                }
                mxfp4_sq_cohorts.push_back({
                    make_int32_array(local_map, Shape{expert_count}),
                    std::move(weight),
                });
            } else if (is_fp8_sq_dtype(dtype)) {
                if (!runtime.empty()) {
                    throw std::runtime_error(
                        "unexpected MFE native-FP8 SQ runtime metadata");
                }
                auto weight = MlxFp8SqWeight::from_blob(dtype, payload);
                if (weight.input_size() != input_width ||
                    weight.output_size() != checked_int(
                        checked_product(
                            static_cast<std::size_t>(count),
                            static_cast<std::size_t>(output_width),
                            "native-FP8 SQ cohort output width"),
                        "native-FP8 SQ cohort output width")) {
                    throw std::runtime_error(
                        "MFE native-FP8 SQ cohort shape is inconsistent");
                }
                std::vector<std::int32_t> local_map(
                    static_cast<std::size_t>(expert_count), -1);
                const int cohort_index = checked_int(
                    fp8_sq_cohorts.size(),
                    "native-FP8 SQ cohort count");
                for (std::size_t local = 0; local < expert_ids.size(); ++local) {
                    const auto expert = expert_ids[local];
                    local_map[static_cast<std::size_t>(expert)] =
                        checked_int(local, "native-FP8 SQ local expert");
                    standalone_owner[static_cast<std::size_t>(expert)] =
                        cohort_index;
                }
                if (reference) {
                    reference_cohorts.push_back({expert_ids, weight});
                }
                fp8_sq_cohorts.push_back({
                    make_int32_array(local_map, Shape{expert_count}),
                    std::move(weight),
                });
            } else {
                throw std::runtime_error(
                    "unsupported nested MFE cohort dtype: "
                    + dtype);
            }
        }
        has_generic_cohorts = has_generic_cohorts && projection_generic;
        projection_info.push_back({
            checked_add(streams.size() - before,
                local_descriptors.size() * sizeof(std::int32_t), "projection bytes"),
            true, 0,
        });
        for (int expert = 0; expert < expert_count; ++expert) {
            const auto source = static_cast<std::size_t>(expert) * kDescriptorSize;
            const auto target = (static_cast<std::size_t>(expert) * projections.size()
                + projection) * kDescriptorSize;
            std::copy_n(local_descriptors.begin() + source, kDescriptorSize,
                descriptors.begin() + target);
        }
    }

    const Shape descriptor_shape{
        descriptor_rows,
        kDescriptorSize,
    };
    std::optional<array> arena = destination ? std::optional<array>(*destination) : std::nullopt;
    const auto stream_bytes = streams.size();
    const auto descriptor_bytes = checked_product(descriptors.size(), sizeof(std::int32_t), "MFE descriptor bytes");
    if (destination && (expert_count != 1 || reference || destination->dtype() != mlx::core::uint8
        || !destination->is_available() || !destination->flags().row_contiguous
        || destination->nbytes() < stream_bytes + 21 * 15 + descriptor_bytes + 15))
        throw std::invalid_argument("MFE expert staging arena is too small or incompatible");
    if (!destination && expert_count == 1 && !reference && stream_bytes > 0 && stream_bytes <= (std::size_t{8} << 20))
        arena = allocate_packed_array(stream_bytes + 21 * 15 + descriptor_bytes + 15, mlx::core::uint8);
    std::size_t arena_offset = arena ? descriptor_bytes : 0;
    const auto materialize = [&](PackedStream stream, Dtype dtype) {
        arena_offset = (arena_offset + 15) & ~std::size_t{15};
        const auto begin = arena_offset;
        arena_offset += stream.size();
        return std::move(stream).materialize(dtype, arena ? &*arena : nullptr, begin);
    };
    auto impl = std::make_shared<Impl>(
        make_int32_array(
            descriptors,
            descriptor_shape, arena ? &*arena : nullptr),
        materialize(
            std::move(streams.nint_q),
            mlx::core::uint8),
        materialize(
            std::move(streams.nint_sub_scale),
            mlx::core::uint8),
        materialize(
            std::move(streams.nint_sub_min),
            mlx::core::uint8),
        materialize(
            std::move(streams.nint_anchor_scale),
            mlx::core::float32),
        materialize(
            std::move(streams.nint_anchor_min),
            mlx::core::float32),
        materialize(
            std::move(streams.q8_q),
            mlx::core::int8),
        materialize(
            std::move(streams.q8_scales),
            mlx::core::float16),
        materialize(
            std::move(streams.vq_indices),
            mlx::core::uint8),
        materialize(
            std::move(streams.vq_state),
            mlx::core::uint8),
        materialize(
            std::move(streams.vq_aux),
            mlx::core::uint8),
        materialize(
            std::move(streams.vq_anchors),
            mlx::core::float32),
        materialize(
            std::move(streams.vq_codebooks),
            mlx::core::int8),
        materialize(
            std::move(streams.vq_scales),
            mlx::core::float32),
        materialize(
            std::move(
                streams.vq_state_to_codebank),
            mlx::core::uint8),
        materialize(
            std::move(streams.vq_banks),
            mlx::core::uint8),
        materialize(
            std::move(streams.vq_parameters),
            mlx::core::float32),
        materialize(
            std::move(streams.vq_residual_codebooks),
            mlx::core::float32),
        materialize(
            std::move(streams.vq_residual_first),
            mlx::core::int16),
        materialize(
            std::move(streams.vq_residual_second),
            mlx::core::int16),
        materialize(
            std::move(streams.mx_values),
            mlx::core::uint8),
        materialize(
            std::move(streams.mx_scales),
            mlx::core::uint8),
        std::move(rotations),
        std::move(descriptors),
        expert_count,
        output_width,
        input_width,
        projection_count);
    impl->packed_arena = arena;
    impl->reference_cohorts =
        std::move(reference_cohorts);
    impl->mxfp4_sq_cohorts =
        std::move(mxfp4_sq_cohorts);
    impl->fp8_sq_cohorts =
        std::move(fp8_sq_cohorts);
    impl->has_generic_cohorts = has_generic_cohorts;
    if (!impl->mxfp4_sq_cohorts.empty()
        || !impl->fp8_sq_cohorts.empty()) {
        // These cohorts reuse their standalone packed kernels; none is
        // decoded by the heterogeneous MFE kernels.
        impl->grouped_mmq = false;
        impl->standalone_owner.emplace(make_int32_array(
            standalone_owner,
            Shape{expert_count}));
        if (!reference) {
            for (const auto& cohort : impl->mxfp4_sq_cohorts) {
                impl->packed_bytes += cohort.weight.packed_nbytes();
            }
            for (const auto& cohort : impl->fp8_sq_cohorts) {
                impl->packed_bytes += cohort.weight.packed_nbytes();
            }
        }
    }
    if (reference) {
        impl->grouped_mmq = false;
        impl->native_primitive = false;
        impl->packed_bytes = 0;
        for (const auto& cohort :
             impl->reference_cohorts) {
            impl->packed_bytes += std::visit(
                [](const auto& weight) {
                    return weight.packed_nbytes();
                },
                cohort.weight);
        }
    }
    if (projection_count > 1) {
        impl->bind_projection_views(projection_info);
    }
    return MlxMfeWeight(std::move(impl));
}

MlxMfeWeight MlxMfeWeight::from_mxfp4_slots(
    int experts,
    int out_per_expert,
    int neuron_len,
    const std::vector<std::int32_t>& slot_for_expert,
    array packed_values,
    array block_scales,
    int logical_experts) {
    if (experts <= 0 || out_per_expert <= 0 || neuron_len <= 0 ||
        neuron_len % 32 != 0 ||
        slot_for_expert.size() != static_cast<std::size_t>(experts) ||
        logical_experts < 0) {
        throw std::invalid_argument("invalid MXFP4 slot-view dimensions");
    }
    if (packed_values.dtype() != mlx::core::uint8 ||
        block_scales.dtype() != mlx::core::uint8 ||
        !packed_values.flags().row_contiguous ||
        !block_scales.flags().row_contiguous) {
        throw std::invalid_argument(
            "MXFP4 slot-view banks must be contiguous uint8 arrays");
    }
    const auto value_stride = checked_product(
        static_cast<std::size_t>(out_per_expert),
        static_cast<std::size_t>(neuron_len / 2),
        "MXFP4 slot value stride");
    const auto scale_stride = checked_product(
        static_cast<std::size_t>(out_per_expert),
        static_cast<std::size_t>(neuron_len / 32),
        "MXFP4 slot scale stride");
    if (packed_values.size() % value_stride != 0 ||
        block_scales.size() % scale_stride != 0 ||
        packed_values.size() / value_stride !=
            block_scales.size() / scale_stride) {
        throw std::invalid_argument(
            "MXFP4 slot-view bank geometry mismatch");
    }
    const auto slots = packed_values.size() / value_stride;
    if (slots == 0 || slots > static_cast<std::size_t>(
        std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("invalid MXFP4 arena slot count");
    }
    std::vector<std::int32_t> descriptors(
        checked_product(
            static_cast<std::size_t>(experts),
            static_cast<std::size_t>(kDescriptorSize),
            "MXFP4 slot descriptors"),
        0);
    for (int expert = 0; expert < experts; ++expert) {
        const auto slot = slot_for_expert[static_cast<std::size_t>(expert)];
        if (slot < 0 || static_cast<std::size_t>(slot) >= slots) {
            throw std::invalid_argument(
                "MXFP4 slot map contains an invalid arena slot");
        }
        const auto base = static_cast<std::size_t>(expert) * kDescriptorSize;
        descriptors[base + kFamily] = kFamilyMxfp4;
        descriptors[base + kLocalExpert] = slot;
        descriptors[base + kOut] = out_per_expert;
        descriptors[base + kInput] = neuron_len;
        descriptors[base + kMxGroups] = neuron_len / 32;
        descriptors[base + kMxValueOffset] = 0;
        descriptors[base + kMxScaleOffset] = 0;
    }

    auto empty_u8 = [] {
        return make_raw_array(std::vector<std::uint8_t>{}, mlx::core::uint8);
    };
    auto empty_i8 = [] {
        return make_raw_array(std::vector<std::uint8_t>{}, mlx::core::int8);
    };
    auto empty_i16 = [] {
        return make_raw_array(std::vector<std::uint8_t>{}, mlx::core::int16);
    };
    auto empty_f16 = [] {
        return make_raw_array(std::vector<std::uint8_t>{}, mlx::core::float16);
    };
    auto empty_f32 = [] {
        return make_raw_array(std::vector<std::uint8_t>{}, mlx::core::float32);
    };
    auto impl = std::make_shared<Impl>(
        make_int32_array(descriptors, Shape{experts, kDescriptorSize}),
        empty_u8(),
        empty_u8(),
        empty_u8(),
        empty_f32(),
        empty_f32(),
        empty_i8(),
        empty_f16(),
        empty_u8(),
        empty_u8(),
        empty_u8(),
        empty_f32(),
        empty_i8(),
        empty_f32(),
        empty_u8(),
        empty_u8(),
        empty_f32(),
        empty_f32(),
        empty_i16(),
        empty_i16(),
        std::move(packed_values),
        std::move(block_scales),
        std::vector<RotationSpec>{},
        std::move(descriptors),
        experts,
        out_per_expert,
        neuron_len,
        1);
    impl->logical_experts = logical_experts > 0
        ? logical_experts
        : experts;
    return MlxMfeWeight(std::move(impl));
}

MlxMfeWeight MlxMfeWeight::concatenate_projections(
    const std::vector<MlxMfeWeight>& weights) {
    if (std::any_of(weights.begin(), weights.end(), [](const auto& weight) { return weight.impl_->indirect_sources != nullptr; })) {
        std::vector<MlxMfeWeight> packed;
        for (const auto& weight : weights)
            packed.push_back(weight.impl_->indirect_sources
                ? concatenate_experts(*weight.impl_->indirect_sources).with_kernel_profile(weight.impl_->kernel_profile) : weight);
        return concatenate_projections(packed);
    }
    if (weights.empty()) {
        throw std::invalid_argument(
            "at least one MFE projection is required");
    }
    const auto& first = *weights.front().impl_;
    for (const auto& weight : weights) {
        const auto& current = *weight.impl_;
        if (
            current.projections != 1
            || current.experts != first.experts
            || current.out_per_expert
                != first.out_per_expert
            || current.neuron_len != first.neuron_len
        ) {
            throw std::invalid_argument(
                "MFE projections have incompatible shapes");
        }
    }

    const auto descriptor_count = checked_product(
        checked_product(
            static_cast<std::size_t>(first.experts),
            weights.size(),
            "projection descriptor count"),
        static_cast<std::size_t>(kDescriptorSize),
        "projection descriptor count");
    std::vector<std::int32_t> descriptors(
        descriptor_count,
        0);

    std::size_t nint_q_offset = 0;
    std::size_t nint_sub_offset = 0;
    std::size_t nint_anchor_offset = 0;
    std::size_t q8_q_offset = 0;
    std::size_t q8_scale_offset = 0;
    std::size_t vq_indices_offset = 0;
    std::size_t vq_state_offset = 0;
    std::size_t vq_aux_offset = 0;
    std::size_t vq_anchor_offset = 0;
    std::size_t vq_codebook_offset = 0;
    std::size_t vq_scale_offset = 0;
    std::size_t vq_state_bank_offset = 0;
    std::size_t vq_bank_offset = 0;
    std::size_t vq_parameter_offset = 0;
    std::size_t vq_residual_codebook_offset = 0;
    std::size_t vq_residual_record_offset = 0;
    std::size_t mx_value_offset = 0;
    std::size_t mx_scale_offset = 0;

    std::vector<array> nint_q_arrays;
    std::vector<array> nint_sub_scale_arrays;
    std::vector<array> nint_sub_min_arrays;
    std::vector<array> nint_anchor_scale_arrays;
    std::vector<array> nint_anchor_min_arrays;
    std::vector<array> q8_q_arrays;
    std::vector<array> q8_scale_arrays;
    std::vector<array> vq_indices_arrays;
    std::vector<array> vq_state_arrays;
    std::vector<array> vq_aux_arrays;
    std::vector<array> vq_anchor_arrays;
    std::vector<array> vq_codebook_arrays;
    std::vector<array> vq_scale_arrays;
    std::vector<array> vq_state_bank_arrays;
    std::vector<array> vq_bank_arrays;
    std::vector<array> vq_parameter_arrays;
    std::vector<array> vq_residual_codebook_arrays;
    std::vector<array> vq_residual_first_arrays;
    std::vector<array> vq_residual_second_arrays;
    std::vector<array> mx_value_arrays;
    std::vector<array> mx_scale_arrays;
    std::vector<RotationSpec> combined_rotations;
    for (
        std::size_t projection = 0;
        projection < weights.size();
        ++projection
    ) {
        const auto& source = *weights[projection].impl_;
        std::vector<int> rotation_map(
            source.rotations.size() + 1,
            0);
        for (
            std::size_t index = 0;
            index < source.rotations.size();
            ++index
        ) {
            rotation_map[index + 1] =
                rotation_variant(
                    source.rotations[index],
                    combined_rotations);
        }
        // Preserve typed row-offset/group64 alignment when rebasing arenas.
        if ((source.family_mask & (1u << kFamilyNint)) != 0u) {
            align_packed_stream(nint_q_arrays, nint_q_offset, 16);
        }
        if ((source.family_mask & (1u << kFamilyVq)) != 0u) {
            align_packed_stream(vq_indices_arrays, vq_indices_offset, 8);
        }
        if ((q8_q_offset & 1u) != 0u) {
            q8_q_arrays.push_back(make_raw_array(
                std::vector<std::uint8_t>{0},
                mlx::core::int8));
            ++q8_q_offset;
        }
        for (
            int expert = 0;
            expert < first.experts;
            ++expert
        ) {
            const auto source_base =
                static_cast<std::size_t>(expert)
                * kDescriptorSize;
            const auto target_base = (
                static_cast<std::size_t>(expert)
                    * weights.size()
                + projection
            ) * kDescriptorSize;
            std::copy_n(
                source.descriptor_values.begin()
                    + static_cast<std::ptrdiff_t>(
                        source_base),
                kDescriptorSize,
                descriptors.begin()
                    + static_cast<std::ptrdiff_t>(
                        target_base));
            auto* descriptor =
                descriptors.data() + target_base;
            if (descriptor[kFamily] == kFamilyNint) {
                descriptor[kNintQOffset] =
                    descriptor_with_offset(
                        descriptor[kNintQOffset],
                        nint_q_offset,
                        "NINT q offset");
                descriptor[kNintSubOffset] =
                    descriptor_with_offset(
                        descriptor[kNintSubOffset],
                        nint_sub_offset,
                        "NINT sub offset");
                descriptor[kNintAnchorOffset] =
                    descriptor_with_offset(
                        descriptor[
                            kNintAnchorOffset],
                        nint_anchor_offset,
                        "NINT anchor offset");
                if (descriptor[kNintV2] != 0) {
                    descriptor[kNintRowMetadataOffset] =
                        descriptor_with_offset(
                            descriptor[kNintRowMetadataOffset],
                            nint_q_offset,
                            "NINTv2 row-metadata offset");
                }
            } else if (
                descriptor[kFamily]
                == kFamilyNint8Zero
            ) {
                descriptor[kQ8QOffset] =
                    descriptor_with_offset(
                        descriptor[kQ8QOffset],
                        q8_q_offset,
                        "NINT8-0 q offset");
                descriptor[kQ8ScaleOffset] =
                    descriptor_with_offset(
                        descriptor[kQ8ScaleOffset],
                        q8_scale_offset,
                        "NINT8-0 scale offset");
            } else if (
                descriptor[kFamily] == kFamilyBf16
                || descriptor[kFamily] == kFamilyF16
            ) {
                descriptor[kDenseValueOffset] =
                    descriptor_with_offset(
                        descriptor[kDenseValueOffset],
                        q8_q_offset,
                        "dense value offset");
            } else if (
                descriptor[kFamily] == kFamilyVq
            ) {
                descriptor[kVqIndicesOffset] =
                    descriptor_with_offset(
                        descriptor[kVqIndicesOffset],
                        vq_indices_offset,
                        "VQ index offset");
                descriptor[kVqStateOffset] =
                    descriptor_with_offset(
                        descriptor[kVqStateOffset],
                        vq_state_offset,
                        "VQ state offset");
                descriptor[kVqAuxOffset] =
                    descriptor_with_offset(
                        descriptor[kVqAuxOffset],
                        vq_aux_offset,
                        "VQ auxiliary offset");
                descriptor[kVqAnchorOffset] =
                    descriptor_with_offset(
                        descriptor[kVqAnchorOffset],
                        vq_anchor_offset,
                        "VQ anchor offset");
                descriptor[kVqCodebookOffset] =
                    descriptor_with_offset(
                        descriptor[kVqCodebookOffset],
                        vq_codebook_offset,
                        "VQ codebook offset");
                descriptor[kVqScaleOffset] =
                    descriptor_with_offset(
                        descriptor[kVqScaleOffset],
                        vq_scale_offset,
                        "VQ scale offset");
                descriptor[kVqStateBankOffset] =
                    descriptor_with_offset(
                        descriptor[
                            kVqStateBankOffset],
                        vq_state_bank_offset,
                        "VQ state-bank offset");
                descriptor[kVqBankOffset] =
                    descriptor_with_offset(
                        descriptor[kVqBankOffset],
                        vq_bank_offset,
                        "VQ bank offset");
                descriptor[kVqParameterOffset] =
                    descriptor_with_offset(
                        descriptor[
                            kVqParameterOffset],
                        vq_parameter_offset,
                        "VQ parameter offset");
                descriptor[kVqResidualCodebookOffset] =
                    descriptor_with_offset(
                        descriptor[kVqResidualCodebookOffset],
                        vq_residual_codebook_offset,
                        "VQ residual codebook offset");
                descriptor[kVqResidualRecordOffset] =
                    descriptor_with_offset(
                        descriptor[kVqResidualRecordOffset],
                        vq_residual_record_offset,
                        "VQ residual record offset");
                const int local_rotation =
                    descriptor[
                        kVqRotationVariant];
                if (
                    local_rotation < 0
                    || static_cast<std::size_t>(
                           local_rotation)
                        >= rotation_map.size()
                ) {
                    throw std::runtime_error(
                        "invalid MFE VQ rotation "
                        "variant");
                }
                descriptor[kVqRotationVariant] =
                    rotation_map[
                        static_cast<std::size_t>(
                            local_rotation)];
            } else if (
                descriptor[kFamily] == kFamilyMxfp4
                || descriptor[kFamily] == kFamilyMxfp8
            ) {
                descriptor[kMxValueOffset] =
                    descriptor_u32_with_offset(
                        descriptor[kMxValueOffset],
                        mx_value_offset,
                        "MX value offset");
                descriptor[kMxScaleOffset] =
                    descriptor_u32_with_offset(
                        descriptor[kMxScaleOffset],
                        mx_scale_offset,
                        "MX scale offset");
            } else {
                throw std::runtime_error(
                    "unsupported MFE descriptor family");
            }
        }

        nint_q_arrays.push_back(source.nint_q);
        nint_sub_scale_arrays.push_back(
            source.nint_sub_scale);
        nint_sub_min_arrays.push_back(
            source.nint_sub_min);
        nint_anchor_scale_arrays.push_back(
            source.nint_anchor_scale);
        nint_anchor_min_arrays.push_back(
            source.nint_anchor_min);
        q8_q_arrays.push_back(source.q8_q);
        q8_scale_arrays.push_back(source.q8_scales);
        vq_indices_arrays.push_back(
            source.vq_indices);
        vq_state_arrays.push_back(source.vq_state);
        vq_aux_arrays.push_back(source.vq_aux);
        vq_anchor_arrays.push_back(
            source.vq_anchors);
        vq_codebook_arrays.push_back(
            source.vq_codebooks);
        vq_scale_arrays.push_back(source.vq_scales);
        vq_state_bank_arrays.push_back(
            source.vq_state_to_codebank);
        vq_bank_arrays.push_back(source.vq_banks);
        vq_parameter_arrays.push_back(
            source.vq_parameters);
        vq_residual_codebook_arrays.push_back(
            source.vq_residual_codebooks);
        vq_residual_first_arrays.push_back(
            source.vq_residual_first);
        vq_residual_second_arrays.push_back(
            source.vq_residual_second);
        mx_value_arrays.push_back(source.mx_values);
        mx_scale_arrays.push_back(source.mx_scales);

        nint_q_offset = checked_add(
            nint_q_offset,
            source.nint_q.size(),
            "NINT q stream size");
        nint_sub_offset = checked_add(
            nint_sub_offset,
            source.nint_sub_scale.size(),
            "NINT sub stream size");
        nint_anchor_offset = checked_add(
            nint_anchor_offset,
            source.nint_anchor_scale.size(),
            "NINT anchor stream size");
        q8_q_offset = checked_add(
            q8_q_offset,
            source.q8_q.size(),
            "NINT8-0 q stream size");
        q8_scale_offset = checked_add(
            q8_scale_offset,
            source.q8_scales.size(),
            "NINT8-0 scale stream size");
        vq_indices_offset = checked_add(
            vq_indices_offset,
            source.vq_indices.size(),
            "VQ index stream size");
        vq_state_offset = checked_add(
            vq_state_offset,
            source.vq_state.size(),
            "VQ state stream size");
        vq_aux_offset = checked_add(
            vq_aux_offset,
            source.vq_aux.size(),
            "VQ auxiliary stream size");
        vq_anchor_offset = checked_add(
            vq_anchor_offset,
            source.vq_anchors.size(),
            "VQ anchor stream size");
        vq_codebook_offset = checked_add(
            vq_codebook_offset,
            source.vq_codebooks.size(),
            "VQ codebook stream size");
        vq_scale_offset = checked_add(
            vq_scale_offset,
            source.vq_scales.size(),
            "VQ scale stream size");
        vq_state_bank_offset = checked_add(
            vq_state_bank_offset,
            source.vq_state_to_codebank.size(),
            "VQ state-bank stream size");
        vq_bank_offset = checked_add(
            vq_bank_offset,
            source.vq_banks.size(),
            "VQ bank stream size");
        vq_parameter_offset = checked_add(
            vq_parameter_offset,
            source.vq_parameters.size(),
            "VQ parameter stream size");
        vq_residual_codebook_offset = checked_add(
            vq_residual_codebook_offset,
            source.vq_residual_codebooks.size(),
            "VQ residual codebook stream size");
        vq_residual_record_offset = checked_add(
            vq_residual_record_offset,
            source.vq_residual_first.size(),
            "VQ residual record stream size");
        mx_value_offset = checked_add(
            mx_value_offset,
            source.mx_values.size(),
            "MXFP4 value stream size");
        mx_scale_offset = checked_add(
            mx_scale_offset,
            source.mx_scales.size(),
            "MXFP4 scale stream size");
    }

    const int projection_count =
        checked_int(weights.size(), "projection count");
    auto impl = std::make_shared<Impl>(
        make_int32_array(
            descriptors,
            Shape{
                first.experts * projection_count,
                kDescriptorSize,
            }),
        concatenate_1d(std::move(nint_q_arrays)),
        concatenate_1d(
            std::move(nint_sub_scale_arrays)),
        concatenate_1d(
            std::move(nint_sub_min_arrays)),
        concatenate_1d(
            std::move(nint_anchor_scale_arrays)),
        concatenate_1d(
            std::move(nint_anchor_min_arrays)),
        concatenate_1d(std::move(q8_q_arrays)),
        concatenate_1d(std::move(q8_scale_arrays)),
        concatenate_1d(
            std::move(vq_indices_arrays)),
        concatenate_1d(
            std::move(vq_state_arrays)),
        concatenate_1d(std::move(vq_aux_arrays)),
        concatenate_1d(
            std::move(vq_anchor_arrays)),
        concatenate_1d(
            std::move(vq_codebook_arrays)),
        concatenate_1d(
            std::move(vq_scale_arrays)),
        concatenate_1d(
            std::move(vq_state_bank_arrays)),
        concatenate_1d(
            std::move(vq_bank_arrays)),
        concatenate_1d(
            std::move(vq_parameter_arrays)),
        concatenate_1d(
            std::move(vq_residual_codebook_arrays)),
        concatenate_1d(
            std::move(vq_residual_first_arrays)),
        concatenate_1d(
            std::move(vq_residual_second_arrays)),
        concatenate_1d(std::move(mx_value_arrays)),
        concatenate_1d(std::move(mx_scale_arrays)),
        std::move(combined_rotations),
        std::move(descriptors),
        first.experts,
        first.out_per_expert,
        first.neuron_len,
        projection_count);
    impl->automatic_mxfp4_nax_prefill = std::all_of(
        weights.begin(),
        weights.end(),
        [](const MlxMfeWeight& weight) {
            return weight.impl_->automatic_mxfp4_nax_prefill;
        });
    impl->has_generic_cohorts = std::all_of(
        weights.begin(),
        weights.end(),
        [](const MlxMfeWeight& weight) {
            return weight.impl_->has_generic_cohorts;
        });
    impl->logical_experts = first.logical_experts > 0
        ? first.logical_experts
        : first.experts;
    if (std::any_of(weights.begin(), weights.end(),
            [](const auto& weight) { return weight.impl_->kernel_profile != nullptr; })) {
        auto profile = std::make_shared<MlxMfeKernelProfile>();
        profile->nint_q_dispatch = true;
        std::vector<int> groups;
        for (const auto& weight : weights) {
            const auto source = weight.impl_->compilation_profile();
            profile->family_mask |= source.family_mask;
            profile->vq_profile_mask |= source.vq_profile_mask;
            profile->vq_layouts |= source.vq_layouts;
            profile->vq_execution_layout |= source.vq_execution_layout;
            profile->nint_q_dispatch &= source.nint_q_dispatch;
            groups.insert(groups.end(), source.nint_group_sizes->begin(), source.nint_group_sizes->end());
        }
        std::sort(groups.begin(), groups.end());
        groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
        profile->nint_group_sizes = std::make_shared<const std::vector<int>>(std::move(groups));
        impl->kernel_profile = std::move(profile);
    }
    impl->projection_views.reserve(weights.size());
    impl->packed_bytes = 0;
    for (const auto& weight : weights) {
        // Keep each projection on its original packed pool.  Evaluating the
        // combined stream would otherwise copy every expert's Gate and Up
        // payload before a one-token decode can touch its selected experts.
        impl->projection_views.push_back(weight.impl_);
        impl->packed_bytes = checked_add(
            impl->packed_bytes,
            weight.impl_->packed_bytes,
            "split MFE projection bytes");
    }
    return MlxMfeWeight(std::move(impl));
}

MlxMfeWeight MlxMfeWeight::materialize_packed_projections() const {
    if (impl_->projections != 2 || impl_->projection_views.empty() ||
        !impl_->rotations.empty() ||
        std::any_of(impl_->projection_views.begin(), impl_->projection_views.end(),
            [](const auto& source) {
                return !source->reference_cohorts.empty() ||
                    !source->mxfp4_sq_cohorts.empty() || !source->fp8_sq_cohorts.empty();
            })) {
        return *this;
    }
    auto impl = std::make_shared<Impl>(*impl_);
    mlx::core::eval(std::vector<array>{
        impl->descriptors, impl->nint_q, impl->nint_sub_scale, impl->nint_sub_min,
        impl->nint_anchor_scale, impl->nint_anchor_min, impl->q8_q, impl->q8_scales,
        impl->vq_indices, impl->vq_state, impl->vq_aux, impl->vq_anchors,
        impl->vq_codebooks, impl->vq_scales, impl->vq_state_to_codebank, impl->vq_banks,
        impl->vq_parameters, impl->vq_residual_codebooks, impl->vq_residual_first,
        impl->vq_residual_second, impl->mx_values, impl->mx_scales,
    });
    // Preserve split dispatch and its arithmetic while releasing source pools.
    std::vector<ProjectionStorageInfo> sources;
    for (const auto& source : impl->projection_views) {
        sources.push_back({source->packed_bytes, source->automatic_mxfp4_nax_prefill,
            source->logical_experts});
    }
    impl->bind_projection_views(sources);
    return MlxMfeWeight(std::move(impl));
}

MlxMfeWeight MlxMfeWeight::concatenate_experts(
    const std::vector<MlxMfeWeight>& weights) {
    if (weights.empty()) {
        throw std::invalid_argument(
            "at least one MFE expert page is required");
    }
    const auto& first = *weights.front().impl_;
    for (const auto& weight : weights) {
        const auto& current = *weight.impl_;
        if (
            current.projections != 1
            || current.experts != 1
            || current.out_per_expert != first.out_per_expert
            || current.neuron_len != first.neuron_len
            || !current.reference_cohorts.empty()
            || !current.mxfp4_sq_cohorts.empty()
            || !current.fp8_sq_cohorts.empty()
            || current.standalone_owner.has_value()
        ) {
            throw std::invalid_argument(
                "MFE expert pages have incompatible layouts");
        }
    }

    if (weights.size() == 1) return weights.front();
    const int expert_count = checked_int(
        weights.size(),
        "resident expert count");
    std::vector<std::int32_t> descriptors(
        checked_product(
            weights.size(),
            static_cast<std::size_t>(kDescriptorSize),
            "resident expert descriptor count"),
        0);

    std::size_t nint_q_offset = 0;
    std::size_t nint_sub_offset = 0;
    std::size_t nint_anchor_offset = 0;
    std::size_t q8_q_offset = 0;
    std::size_t q8_scale_offset = 0;
    std::size_t vq_indices_offset = 0;
    std::size_t vq_state_offset = 0;
    std::size_t vq_aux_offset = 0;
    std::size_t vq_anchor_offset = 0;
    std::size_t vq_codebook_offset = 0;
    std::size_t vq_scale_offset = 0;
    std::size_t vq_state_bank_offset = 0;
    std::size_t vq_bank_offset = 0;
    std::size_t vq_parameter_offset = 0;
    std::size_t vq_residual_codebook_offset = 0;
    std::size_t vq_residual_record_offset = 0;
    std::size_t mx_value_offset = 0;
    std::size_t mx_scale_offset = 0;

    std::vector<array> nint_q_arrays;
    std::vector<array> nint_sub_scale_arrays;
    std::vector<array> nint_sub_min_arrays;
    std::vector<array> nint_anchor_scale_arrays;
    std::vector<array> nint_anchor_min_arrays;
    std::vector<array> q8_q_arrays;
    std::vector<array> q8_scale_arrays;
    std::vector<array> vq_indices_arrays;
    std::vector<array> vq_state_arrays;
    std::vector<array> vq_aux_arrays;
    std::vector<array> vq_anchor_arrays;
    std::vector<array> vq_codebook_arrays;
    std::vector<array> vq_scale_arrays;
    std::vector<array> vq_state_bank_arrays;
    std::vector<array> vq_bank_arrays;
    std::vector<array> vq_parameter_arrays;
    std::vector<array> vq_residual_codebook_arrays;
    std::vector<array> vq_residual_first_arrays;
    std::vector<array> vq_residual_second_arrays;
    std::vector<array> mx_value_arrays;
    std::vector<array> mx_scale_arrays;
    std::vector<RotationSpec> combined_rotations;

    for (std::size_t expert = 0; expert < weights.size(); ++expert) {
        const auto& source = *weights[expert].impl_;
        std::vector<int> rotation_map(
            source.rotations.size() + 1,
            0);
        for (std::size_t index = 0;
             index < source.rotations.size();
             ++index) {
            rotation_map[index + 1] = rotation_variant(
                source.rotations[index],
                combined_rotations);
        }
        if ((source.family_mask & (1u << kFamilyNint)) != 0u) {
            align_packed_stream(nint_q_arrays, nint_q_offset, 16);
        }
        if ((source.family_mask & (1u << kFamilyVq)) != 0u) {
            align_packed_stream(vq_indices_arrays, vq_indices_offset, 8);
        }
        if ((q8_q_offset & 1u) != 0u) {
            q8_q_arrays.push_back(make_raw_array(
                std::vector<std::uint8_t>{0},
                mlx::core::int8));
            ++q8_q_offset;
        }

        const auto target_base = expert * kDescriptorSize;
        std::copy_n(
            source.descriptor_values.begin(),
            kDescriptorSize,
            descriptors.begin()
                + static_cast<std::ptrdiff_t>(target_base));
        auto* descriptor = descriptors.data() + target_base;
        if (descriptor[kFamily] == kFamilyNint) {
            descriptor[kNintQOffset] = descriptor_with_offset(
                descriptor[kNintQOffset], nint_q_offset,
                "NINT q offset");
            descriptor[kNintSubOffset] = descriptor_with_offset(
                descriptor[kNintSubOffset], nint_sub_offset,
                "NINT sub offset");
            descriptor[kNintAnchorOffset] = descriptor_with_offset(
                descriptor[kNintAnchorOffset], nint_anchor_offset,
                "NINT anchor offset");
            if (descriptor[kNintV2] != 0) {
                descriptor[kNintRowMetadataOffset] = descriptor_with_offset(
                    descriptor[kNintRowMetadataOffset], nint_q_offset,
                    "NINTv2 row-metadata offset");
            }
        } else if (descriptor[kFamily] == kFamilyNint8Zero) {
            descriptor[kQ8QOffset] = descriptor_with_offset(
                descriptor[kQ8QOffset], q8_q_offset,
                "NINT8-0 q offset");
            descriptor[kQ8ScaleOffset] = descriptor_with_offset(
                descriptor[kQ8ScaleOffset], q8_scale_offset,
                "NINT8-0 scale offset");
        } else if (
            descriptor[kFamily] == kFamilyBf16
            || descriptor[kFamily] == kFamilyF16
        ) {
            descriptor[kDenseValueOffset] = descriptor_with_offset(
                descriptor[kDenseValueOffset], q8_q_offset,
                "dense value offset");
        } else if (descriptor[kFamily] == kFamilyVq) {
            descriptor[kVqIndicesOffset] = descriptor_with_offset(
                descriptor[kVqIndicesOffset], vq_indices_offset,
                "VQ index offset");
            descriptor[kVqStateOffset] = descriptor_with_offset(
                descriptor[kVqStateOffset], vq_state_offset,
                "VQ state offset");
            descriptor[kVqAuxOffset] = descriptor_with_offset(
                descriptor[kVqAuxOffset], vq_aux_offset,
                "VQ auxiliary offset");
            descriptor[kVqAnchorOffset] = descriptor_with_offset(
                descriptor[kVqAnchorOffset], vq_anchor_offset,
                "VQ anchor offset");
            descriptor[kVqCodebookOffset] = descriptor_with_offset(
                descriptor[kVqCodebookOffset], vq_codebook_offset,
                "VQ codebook offset");
            descriptor[kVqScaleOffset] = descriptor_with_offset(
                descriptor[kVqScaleOffset], vq_scale_offset,
                "VQ scale offset");
            descriptor[kVqStateBankOffset] = descriptor_with_offset(
                descriptor[kVqStateBankOffset], vq_state_bank_offset,
                "VQ state-bank offset");
            descriptor[kVqBankOffset] = descriptor_with_offset(
                descriptor[kVqBankOffset], vq_bank_offset,
                "VQ bank offset");
            descriptor[kVqParameterOffset] = descriptor_with_offset(
                descriptor[kVqParameterOffset], vq_parameter_offset,
                "VQ parameter offset");
            descriptor[kVqResidualCodebookOffset] = descriptor_with_offset(
                descriptor[kVqResidualCodebookOffset],
                vq_residual_codebook_offset,
                "VQ residual codebook offset");
            descriptor[kVqResidualRecordOffset] = descriptor_with_offset(
                descriptor[kVqResidualRecordOffset],
                vq_residual_record_offset,
                "VQ residual record offset");
            const int local_rotation =
                descriptor[kVqRotationVariant];
            if (local_rotation < 0
                || static_cast<std::size_t>(local_rotation)
                    >= rotation_map.size()) {
                throw std::runtime_error(
                    "invalid MFE VQ rotation variant");
            }
            descriptor[kVqRotationVariant] = rotation_map[
                static_cast<std::size_t>(local_rotation)];
        } else if (
            descriptor[kFamily] == kFamilyMxfp4
            || descriptor[kFamily] == kFamilyMxfp8
        ) {
            descriptor[kMxValueOffset] = descriptor_u32_with_offset(
                descriptor[kMxValueOffset], mx_value_offset,
                "MX value offset");
            descriptor[kMxScaleOffset] = descriptor_u32_with_offset(
                descriptor[kMxScaleOffset], mx_scale_offset,
                "MX scale offset");
        } else {
            throw std::runtime_error(
                "unsupported MFE resident expert family");
        }

        const int family = descriptor[kFamily];
        if (family == kFamilyNint) {
            nint_q_arrays.push_back(source.nint_q);
            nint_sub_scale_arrays.push_back(source.nint_sub_scale);
            nint_sub_min_arrays.push_back(source.nint_sub_min);
            nint_anchor_scale_arrays.push_back(source.nint_anchor_scale);
            nint_anchor_min_arrays.push_back(source.nint_anchor_min);
            nint_q_offset = checked_add(
                nint_q_offset, source.nint_q.size(),
                "NINT q stream size");
            nint_sub_offset = checked_add(
                nint_sub_offset, source.nint_sub_scale.size(),
                "NINT sub stream size");
            nint_anchor_offset = checked_add(
                nint_anchor_offset, source.nint_anchor_scale.size(),
                "NINT anchor stream size");
        } else if (family == kFamilyNint8Zero) {
            q8_q_arrays.push_back(source.q8_q);
            q8_scale_arrays.push_back(source.q8_scales);
            q8_q_offset = checked_add(
                q8_q_offset, source.q8_q.size(),
                "NINT8-0 q stream size");
            q8_scale_offset = checked_add(
                q8_scale_offset, source.q8_scales.size(),
                "NINT8-0 scale stream size");
        } else if (family == kFamilyBf16 || family == kFamilyF16) {
            q8_q_arrays.push_back(source.q8_q);
            q8_q_offset = checked_add(
                q8_q_offset, source.q8_q.size(),
                "dense value stream size");
        } else if (family == kFamilyVq) {
            vq_indices_arrays.push_back(source.vq_indices);
            vq_state_arrays.push_back(source.vq_state);
            vq_aux_arrays.push_back(source.vq_aux);
            vq_anchor_arrays.push_back(source.vq_anchors);
            vq_codebook_arrays.push_back(source.vq_codebooks);
            vq_scale_arrays.push_back(source.vq_scales);
            vq_state_bank_arrays.push_back(source.vq_state_to_codebank);
            vq_bank_arrays.push_back(source.vq_banks);
            vq_parameter_arrays.push_back(source.vq_parameters);
            vq_residual_codebook_arrays.push_back(
                source.vq_residual_codebooks);
            vq_residual_first_arrays.push_back(source.vq_residual_first);
            vq_residual_second_arrays.push_back(source.vq_residual_second);
            vq_indices_offset = checked_add(
                vq_indices_offset, source.vq_indices.size(),
                "VQ index stream size");
            vq_state_offset = checked_add(
                vq_state_offset, source.vq_state.size(),
                "VQ state stream size");
            vq_aux_offset = checked_add(
                vq_aux_offset, source.vq_aux.size(),
                "VQ auxiliary stream size");
            vq_anchor_offset = checked_add(
                vq_anchor_offset, source.vq_anchors.size(),
                "VQ anchor stream size");
            vq_codebook_offset = checked_add(
                vq_codebook_offset, source.vq_codebooks.size(),
                "VQ codebook stream size");
            vq_scale_offset = checked_add(
                vq_scale_offset, source.vq_scales.size(),
                "VQ scale stream size");
            vq_state_bank_offset = checked_add(
                vq_state_bank_offset, source.vq_state_to_codebank.size(),
                "VQ state-bank stream size");
            vq_bank_offset = checked_add(
                vq_bank_offset, source.vq_banks.size(),
                "VQ bank stream size");
            vq_parameter_offset = checked_add(
                vq_parameter_offset, source.vq_parameters.size(),
                "VQ parameter stream size");
            vq_residual_codebook_offset = checked_add(
                vq_residual_codebook_offset,
                source.vq_residual_codebooks.size(),
                "VQ residual codebook stream size");
            vq_residual_record_offset = checked_add(
                vq_residual_record_offset,
                source.vq_residual_first.size(),
                "VQ residual record stream size");
        } else {
            mx_value_arrays.push_back(source.mx_values);
            mx_scale_arrays.push_back(source.mx_scales);
            mx_value_offset = checked_add(
                mx_value_offset, source.mx_values.size(),
                "MX value stream size");
            mx_scale_offset = checked_add(
                mx_scale_offset, source.mx_scales.size(),
                "MX scale stream size");
        }
    }

    const std::array<const std::vector<array>*, 21> stream_arrays{&nint_q_arrays, &nint_sub_scale_arrays, &nint_sub_min_arrays,
        &nint_anchor_scale_arrays, &nint_anchor_min_arrays, &q8_q_arrays, &q8_scale_arrays,
        &vq_indices_arrays, &vq_state_arrays, &vq_aux_arrays, &vq_anchor_arrays,
        &vq_codebook_arrays, &vq_scale_arrays, &vq_state_bank_arrays, &vq_bank_arrays,
        &vq_parameter_arrays, &vq_residual_codebook_arrays, &vq_residual_first_arrays,
        &vq_residual_second_arrays, &mx_value_arrays, &mx_scale_arrays};
    std::vector<array> sources;
    for (const auto& weight : weights) {
        if (weight.impl_->packed_arena) sources.push_back(*weight.impl_->packed_arena);
    }
    const auto execution_arena = pack_expert_streams(std::move(sources), stream_arrays);
    const auto host_stream_bytes = [](const std::vector<array>& values) {
        if (values.size() < 2) return std::size_t{0};
        const auto dtype = values.front().dtype();
        const auto empty = empty_packed_array(dtype).data_shared_ptr();
        bool all_empty = true;
        std::size_t bytes = 0;
        for (const auto& value : values) {
            if (value.dtype() != dtype || !value.flags().row_contiguous || !value.is_available())
                return std::size_t{0};
            bytes = checked_add(bytes, value.nbytes(), "expert metadata bytes");
            if (bytes > (std::size_t{128} << 10)) return std::size_t{0};
            all_empty &= value.data_shared_ptr() == empty;
        }
        return all_empty ? std::size_t{0} : bytes;
    };
    std::size_t metadata_bytes = 0;
    std::vector<std::pair<const std::vector<array>*, std::size_t>> metadata_streams;
    for (const auto* values : stream_arrays) {
        if (execution_arena) break;
        const auto bytes = host_stream_bytes(*values);
        if (!bytes) continue;
        metadata_streams.emplace_back(values, bytes);
        metadata_bytes = checked_add(metadata_bytes, bytes, "expert metadata arena");
    }
    const auto descriptor_bytes = checked_product(descriptors.size(), sizeof(std::int32_t), "expert descriptors");
    std::optional<array> metadata_arena;
    if (metadata_bytes)
        metadata_arena = allocate_packed_array(metadata_bytes + descriptor_bytes + 22 * 15, mlx::core::uint8);
    std::size_t metadata_offset = descriptor_bytes;
    const auto concatenate_or_empty = [&](
        std::vector<array>&& values,
        Dtype dtype) {
        if (execution_arena) {
            const auto found = std::find_if(execution_arena->streams.begin(), execution_arena->streams.end(),
                [&](const auto& entry) { return entry.values == &values; });
            if (found == execution_arena->streams.end()) return empty_packed_array(dtype);
            return mlx::core::view(mlx::core::slice(execution_arena->storage,
                {static_cast<int>(found->offset)}, {static_cast<int>(found->offset + found->bytes)}), dtype);
        }
        if (metadata_arena) {
            const auto found = std::find_if(metadata_streams.begin(), metadata_streams.end(),
                [&](const auto& entry) { return entry.first == &values; });
            if (found != metadata_streams.end()) {
                PackedStream stream;
                for (auto& value : values) stream.append(std::move(value));
                metadata_offset = (metadata_offset + 15) & ~std::size_t{15};
                const auto begin = metadata_offset;
                metadata_offset += found->second;
                return std::move(stream).materialize(dtype, &*metadata_arena, begin);
            }
        }
        return values.empty()
            ? make_raw_array(
                  std::vector<std::uint8_t>{},
                  dtype)
            : concatenate_1d(std::move(values));
    };
    auto impl = std::make_shared<Impl>(
        make_int32_array(
            descriptors,
            Shape{expert_count, kDescriptorSize}, metadata_arena ? &*metadata_arena : nullptr),
        concatenate_or_empty(std::move(nint_q_arrays), mlx::core::uint8),
        concatenate_or_empty(
            std::move(nint_sub_scale_arrays), mlx::core::uint8),
        concatenate_or_empty(
            std::move(nint_sub_min_arrays), mlx::core::uint8),
        concatenate_or_empty(
            std::move(nint_anchor_scale_arrays), mlx::core::float32),
        concatenate_or_empty(
            std::move(nint_anchor_min_arrays), mlx::core::float32),
        concatenate_or_empty(std::move(q8_q_arrays), mlx::core::int8),
        concatenate_or_empty(
            std::move(q8_scale_arrays), mlx::core::float16),
        concatenate_or_empty(
            std::move(vq_indices_arrays), mlx::core::uint8),
        concatenate_or_empty(std::move(vq_state_arrays), mlx::core::uint8),
        concatenate_or_empty(std::move(vq_aux_arrays), mlx::core::uint8),
        concatenate_or_empty(
            std::move(vq_anchor_arrays), mlx::core::float32),
        concatenate_or_empty(
            std::move(vq_codebook_arrays), mlx::core::int8),
        concatenate_or_empty(
            std::move(vq_scale_arrays), mlx::core::float32),
        concatenate_or_empty(
            std::move(vq_state_bank_arrays), mlx::core::uint8),
        concatenate_or_empty(std::move(vq_bank_arrays), mlx::core::uint8),
        concatenate_or_empty(
            std::move(vq_parameter_arrays), mlx::core::float32),
        concatenate_or_empty(
            std::move(vq_residual_codebook_arrays), mlx::core::float32),
        concatenate_or_empty(
            std::move(vq_residual_first_arrays), mlx::core::int16),
        concatenate_or_empty(
            std::move(vq_residual_second_arrays), mlx::core::int16),
        concatenate_or_empty(std::move(mx_value_arrays), mlx::core::uint8),
        concatenate_or_empty(std::move(mx_scale_arrays), mlx::core::uint8),
        std::move(combined_rotations),
        std::move(descriptors),
        expert_count,
        first.out_per_expert,
        first.neuron_len,
        1);
    impl->has_generic_cohorts = std::all_of(weights.begin(), weights.end(),
        [](const auto& weight) { return weight.impl_->has_generic_cohorts; });
    if (execution_arena) impl->packed_arena = execution_arena->storage;
    impl->automatic_mxfp4_nax_prefill = std::all_of(
        weights.begin(),
        weights.end(),
        [](const MlxMfeWeight& weight) {
            return weight.impl_->automatic_mxfp4_nax_prefill;
        });
    return MlxMfeWeight(std::move(impl));
}

MlxMfeWeight MlxMfeWeight::concatenate_expert_pages(const std::vector<MlxMfeWeight>& weights,
    std::shared_ptr<const MlxMfeKernelProfile> profile) {
    if (weights.empty() || mlx_reference_enabled()) return concatenate_experts(weights);
    const auto streams = [](const Impl& value) {
        return std::array<const array*, 21>{&value.nint_q, &value.nint_sub_scale, &value.nint_sub_min,
            &value.nint_anchor_scale, &value.nint_anchor_min, &value.q8_q, &value.q8_scales,
            &value.vq_indices, &value.vq_state, &value.vq_aux, &value.vq_anchors,
            &value.vq_codebooks, &value.vq_scales, &value.vq_state_to_codebank, &value.vq_banks,
            &value.vq_parameters, &value.vq_residual_codebooks, &value.vq_residual_first,
            &value.vq_residual_second, &value.mx_values, &value.mx_scales};
    };
    const auto& first = *weights.front().impl_;
    std::uint32_t families = 0;
    for (const auto& weight : weights) {
        const auto& value = *weight.impl_;
        if (!value.native_primitive || value.projections != 1 || value.experts != 1
            || value.out_per_expert != first.out_per_expert || value.neuron_len != first.neuron_len
            || !value.reference_cohorts.empty() || !value.mxfp4_sq_cohorts.empty()
            || !value.fp8_sq_cohorts.empty() || value.standalone_owner || value.indirect_sources)
            return concatenate_experts(weights);
        for (const auto* stream : streams(value))
            if (!stream->is_available() || !stream->flags().row_contiguous) return concatenate_experts(weights);
        families |= value.family_mask;
    }
    if (!uses_indirect_expert_pages(profile ? profile->family_mask : families)) return concatenate_experts(weights);
    std::vector<std::int32_t> descriptors;
    std::vector<std::uint64_t> addresses;
    std::vector<array> owners;
    std::unordered_set<const MTL::Buffer*> buffers;
    std::vector<RotationSpec> rotations;
    std::size_t bytes = 0;
    for (const auto& weight : weights) {
        const auto& value = *weight.impl_;
        const auto begin = descriptors.size();
        descriptors.insert(descriptors.end(), value.descriptor_values.begin(), value.descriptor_values.end());
        const int variant = descriptors[begin + kVqRotationVariant];
        if (variant) descriptors[begin + kVqRotationVariant] = rotation_variant(value.rotations.at(variant - 1), rotations);
        for (const auto* stream : streams(value)) {
            auto* buffer = static_cast<const MTL::Buffer*>(stream->buffer().ptr());
            addresses.push_back(buffer->gpuAddress() + stream->offset());
            if (buffers.insert(buffer).second) owners.push_back(*stream);
        }
        bytes = checked_add(bytes, weight.packed_nbytes(), "indirect expert bytes");
    }
    const int experts = checked_int(weights.size(), "indirect expert count");
    auto impl = std::make_shared<Impl>(make_int32_array(descriptors, Shape{experts, kDescriptorSize}),
        first.nint_q, first.nint_sub_scale, first.nint_sub_min, first.nint_anchor_scale, first.nint_anchor_min,
        first.q8_q, first.q8_scales, first.vq_indices, first.vq_state, first.vq_aux, first.vq_anchors,
        first.vq_codebooks, first.vq_scales, first.vq_state_to_codebank, first.vq_banks, first.vq_parameters,
        first.vq_residual_codebooks, first.vq_residual_first, first.vq_residual_second,
        first.mx_values, first.mx_scales, std::move(rotations), std::move(descriptors),
        experts, first.out_per_expert, first.neuron_len, 1);
    impl->indirect_addresses.emplace(addresses.begin(), Shape{experts, 21}, mlx::core::uint64);
    impl->mxfp4_slot_ids.reset();
    impl->mxfp4_slot_ids_sorted = false;
    impl->indirect_owners = std::move(owners);
    impl->indirect_sources = std::make_shared<const std::vector<MlxMfeWeight>>(weights);
    impl->has_generic_cohorts = true;
    impl->kernel_profile = std::move(profile);
    impl->packed_bytes = checked_add(bytes, checked_add(impl->descriptors.nbytes(),
        impl->indirect_addresses->nbytes(), "indirect expert metadata"), "indirect expert storage");
    return MlxMfeWeight(std::move(impl));
}

MlxMfeWeight MlxMfeWeight::select_experts(
    std::span<const std::int32_t> experts) const {
    if (experts.empty() || impl_->projections != 1 || !impl_->projection_views.empty()
        || !impl_->reference_cohorts.empty() || !impl_->mxfp4_sq_cohorts.empty()
        || !impl_->fp8_sq_cohorts.empty() || impl_->standalone_owner)
        throw std::invalid_argument("MFE expert views require one native packed projection");
    std::vector<std::int32_t> values;
    values.reserve(experts.size() * kDescriptorSize);
    bool identity = experts.size() == static_cast<std::size_t>(impl_->experts);
    for (std::size_t index = 0; index < experts.size(); ++index) {
        const auto expert = experts[index];
        if (expert < 0 || expert >= impl_->experts)
            throw std::out_of_range("MFE view expert ID is out of range");
        identity &= expert == static_cast<std::int32_t>(index);
        const auto first = impl_->descriptor_values.begin() + expert * kDescriptorSize;
        values.insert(values.end(), first, first + kDescriptorSize);
    }
    if (identity) return *this;
    if (impl_->indirect_sources) {
        std::vector<MlxMfeWeight> pages;
        for (const auto expert : experts) pages.push_back(impl_->indirect_sources->at(expert));
        return concatenate_expert_pages(pages, impl_->kernel_profile);
    }
    auto descriptors = make_int32_array(values,
        Shape{checked_int(experts.size(), "MFE view experts"), kDescriptorSize});
    auto impl = std::make_shared<Impl>(std::move(descriptors),
        impl_->nint_q, impl_->nint_sub_scale, impl_->nint_sub_min,
        impl_->nint_anchor_scale, impl_->nint_anchor_min, impl_->q8_q, impl_->q8_scales,
        impl_->vq_indices, impl_->vq_state, impl_->vq_aux, impl_->vq_anchors,
        impl_->vq_codebooks, impl_->vq_scales, impl_->vq_state_to_codebank,
        impl_->vq_banks, impl_->vq_parameters, impl_->vq_residual_codebooks,
        impl_->vq_residual_first, impl_->vq_residual_second,
        impl_->mx_values, impl_->mx_scales, impl_->rotations, std::move(values),
        checked_int(experts.size(), "MFE view experts"), impl_->out_per_expert,
        impl_->neuron_len, 1);
    impl->automatic_mxfp4_nax_prefill = impl_->automatic_mxfp4_nax_prefill;
    impl->packed_arena = impl_->packed_arena;
    impl->has_generic_cohorts = impl_->has_generic_cohorts;
    impl->kernel_profile = impl_->kernel_profile;
    return MlxMfeWeight(std::move(impl));
}

MlxMfeWeight MlxMfeWeight::with_kernel_profile(
    std::shared_ptr<const MlxMfeKernelProfile> profile) const {
    if (impl_->kernel_profile == profile) return *this;
    auto impl = std::make_shared<Impl>(*impl_);
    impl->kernel_profile = std::move(profile);
    return MlxMfeWeight(std::move(impl));
}

MlxMfeWeight MlxMfeWeight::with_prepared_page_resources() const {
    if (!impl_->indirect_addresses || impl_->page_resources) return *this;
    auto resources = std::make_shared<MfePageResources>();
    resources->owners = impl_->indirect_owners;
    auto descriptor = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
    descriptor->setInitialCapacity(resources->owners.size());
    NS::Error* error = nullptr;
    auto& device = mlx::core::metal::device(mlx::core::Device::gpu);
    resources->residency = NS::TransferPtr(device.mtl_device()->newResidencySet(descriptor.get(), &error));
    if (!resources->residency) throw std::runtime_error(error
        ? error->localizedDescription()->utf8String() : "expert residency set unavailable");
    for (const auto& owner : resources->owners) {
        if (!owner.is_available()) throw std::logic_error("prepared expert page is not ready");
        resources->residency->addAllocation(static_cast<const MTL::Buffer*>(owner.buffer().ptr()));
    }
    resources->residency->commit();
    auto impl = std::make_shared<Impl>(*impl_);
    impl->page_resources = std::move(resources);
    return MlxMfeWeight(std::move(impl));
}

MlxMfeWeight MlxMfeWeight::prepare_cache_storage() const {
    std::vector<array> roots;
    std::vector<std::shared_ptr<tyloi::TensorStorage>> nested;
    impl_->collect_cache_storage(roots, nested);
    if (!roots.empty()) mlx::core::eval(std::move(roots));
    return *this;
}

void MlxMfeWeight::collect_cache_preparation(int experts, bool shared_input, bool fused_swiglu) const {
    auto* preparation = MlxKernelPreparation::current();
    const auto& weight = *this;
    const auto indirect = concatenate_expert_pages({weight, weight}, impl_->kernel_profile)
        .with_kernel_profile(impl_->kernel_profile);
    const int routes = preparation->routes();
    for (const int rows : {1, 2, 3, 4, 5, 6, 8}) {
        const auto input = mlx::core::zeros(shared_input
            ? Shape{rows, neuron_len()}
            : Shape{rows, routes, neuron_len()}, mlx::core::float16);
        const auto ids = mlx::core::zeros(Shape{rows, routes}, mlx::core::int32);
        const auto add = [&](const array* groups) {
            preparation->add(fused_swiglu
                ? weight.routed_swiglu(input, ids, 0.0f, groups)
                : weight.routed_matmul(input, ids, groups));
            preparation->add(fused_swiglu
                ? indirect.routed_swiglu(input, ids, 0.0f, groups)
                : indirect.routed_matmul(input, ids, groups));
        };
        add(nullptr);
        if (rows > 1 && rows <= 24) {
            const auto groups = mlx::core::zeros(Shape{rows, routes, 8}, mlx::core::int32);
            add(&groups);
        }
    }
    const auto maximum_rows = *std::max_element(preparation->row_buckets().begin(),
        preparation->row_buckets().end());
    if (maximum_rows < 32 || !indirect.supports_grouped_mmq()
        || !uses_indirect_expert_pages(impl_->kernel_profile->family_mask)) return;
    for (const int block_rows : {32, 48, 64, 80, 96, 128}) {
        for (const int rows : {32, maximum_rows}) {
            const auto ids = mlx::core::zeros(Shape{rows, routes}, mlx::core::int32);
            const auto order = mlx::core::arange(rows * routes, mlx::core::int32);
            const auto expert_ids = mlx::core::arange(experts, mlx::core::int32);
            preparation->add(make_grouped_mmq_plan(ids, order, expert_ids,
                experts, block_rows).block_meta);
        }
    }
    const auto& profile = *impl_->kernel_profile;
    std::vector<std::tuple<int, int, int>> cohorts;
    for (int family = 0; family < 7; ++family) {
        if (!(profile.family_mask & (1u << family))) continue;
        if (family == kFamilyNint && profile.nint_group_sizes && !profile.nint_group_sizes->empty()) {
            for (const int group : *profile.nint_group_sizes) cohorts.emplace_back(1 << family, group, 1);
        } else cohorts.emplace_back(1 << family, 0, 1);
    }
    const int width = fused_swiglu ? out_per_expert() / 2 : out_per_expert();
    for (const bool nax : {false, true}) {
        if (nax && !mixed_grouped_nax_enabled(checked_int(checked_product(
                maximum_rows, routes, "prefill preparation routes"), "prefill preparation routes"))) continue;
        for (const int block_rows : {32, 48, 64, 80, 96, 128}) {
            if (!nax && (block_rows == 48 || block_rows == 128)) continue;
            GroupedMmqConfig config{
                .output_shape = Shape{1, width},
                .block_rows = block_rows,
                .tile_columns = nax ? grouped_mmq_tile_columns(block_rows, width, fused_swiglu) : 64,
                .experts = 1,
                .output_width = width,
                .matrix_output_width = out_per_expert(),
                .input_width = neuron_len(),
                .fused_swiglu = static_cast<int>(fused_swiglu),
                .family_mask = static_cast<int>(profile.family_mask),
                .vq_profile_mask = static_cast<int>(profile.vq_profile_mask),
                .nint_group_sizes = profile.nint_group_sizes,
                .use_nax = nax,
                .indirect_pages = true,
            };
            preparation->add(grouped_mmq_dispatch({}, config));
            if (!nax || fused_swiglu) continue;
            for (std::size_t count = 2; count <= cohorts.size(); ++count) {
                auto passes = std::make_shared<const std::vector<std::tuple<int, int, int>>>(
                    cohorts.begin(), cohorts.begin() + count);
                config.family_mask = 0;
                for (const auto& [family, group, population] : *passes) config.family_mask |= family;
                config.experts = static_cast<int>(count);
                config.cohort_populations = std::move(passes);
                preparation->add(grouped_mmq_dispatch({}, config));
            }
        }
    }
}

MlxMfeWeight MlxMfeWeight::with_automatic_mxfp4_nax_prefill(
    bool enabled) const {
    auto impl = std::make_shared<Impl>(*impl_);
    impl->automatic_mxfp4_nax_prefill = enabled;
    for (auto& projection : impl->projection_views) {
        projection = MlxMfeWeight(projection)
            .with_automatic_mxfp4_nax_prefill(enabled)
            .impl_;
    }
    return MlxMfeWeight(std::move(impl));
}


array MlxMfeWeight::routed_matmul(
    const array& input,
    const array& expert_ids,
    const array* route_groups) const {
    return routed_matmul_impl(
        input,
        expert_ids,
        false,
        0.0f, nullptr, false, route_groups);
}

array MlxMfeWeight::routed_matmul_mapped(
    const array& input,
    const array& expert_ids,
    const array& expert_map) const {
    return routed_matmul_impl(
        input,
        expert_ids,
        false,
        0.0f,
        &expert_map);
}

array MlxMfeWeight::routed_matmul_packed(
    const array& input,
    const array& packed_expert_ids) const {
    return routed_matmul_impl(
        input,
        packed_expert_ids,
        false,
        0.0f,
        nullptr,
        true);
}

array MlxMfeWeight::routed_swiglu(
    const array& input,
    const array& expert_ids,
    float limit,
    const array* route_groups) const {
    const bool split_gate_up = impl_->projections == 2;
    if (
        (!split_gate_up && impl_->projections != 1)
        || impl_->out_per_expert <= 0
        || (!split_gate_up && (impl_->out_per_expert & 1) != 0)
    ) {
        throw std::invalid_argument(
            "MFE SwiGLU requires one fused or two split gate/up projections");
    }
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "MFE SwiGLU limit must be finite and non-negative");
    }
    if (split_gate_up) {
        if (
            input.ndim() >= 2
            && input.shape(0) >= 32
            && impl_->grouped_mmq
            && impl_->rotations.empty()
        ) {
            return routed_matmul_impl(
                input,
                expert_ids,
                true,
                limit, nullptr, false, route_groups);
        }
        const bool has_standalone_projection = std::any_of(
            impl_->projection_views.begin(),
            impl_->projection_views.end(),
            [](const auto& projection) {
                return !projection->reference_cohorts.empty()
                    || !projection->mxfp4_sq_cohorts.empty()
                    || !projection->fp8_sq_cohorts.empty();
            });
        if (mfe_split_swiglu_enabled()
            && impl_->native_primitive
            && !has_standalone_projection) {
            return routed_matmul_impl(
                input,
                expert_ids,
                true,
                limit, nullptr, false, route_groups);
        }
        auto gate_up = routed_matmul_impl(
            input,
            expert_ids,
            false,
            0.0f, nullptr, false, route_groups);
        return limit > 0.0f
            ? moe_limited_swiglu_split(gate_up, limit)
            : moe_swiglu_split(gate_up);
    }
    return routed_matmul_impl(
        input,
        expert_ids,
        true,
        limit, nullptr, false, route_groups);
}

array MlxMfeWeight::routed_swiglu_mapped(
    const array& input,
    const array& expert_ids,
    const array& expert_map,
    float limit) const {
    const bool split_gate_up = impl_->projections == 2;
    if (
        (!split_gate_up && impl_->projections != 1)
        || impl_->out_per_expert <= 0
        || (!split_gate_up && (impl_->out_per_expert & 1) != 0)
    ) {
        throw std::invalid_argument(
            "MFE SwiGLU requires one fused or two split gate/up projections");
    }
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "MFE SwiGLU limit must be finite and non-negative");
    }
    if (split_gate_up) {
        auto gate_up = routed_matmul_impl(
            input,
            expert_ids,
            false,
            0.0f,
            &expert_map);
        return limit > 0.0f
            ? moe_limited_swiglu_split(gate_up, limit)
            : moe_swiglu_split(gate_up);
    }
    return routed_matmul_impl(
        input,
        expert_ids,
        true,
        limit,
        &expert_map);
}

array MlxMfeWeight::routed_swiglu_packed(
    const array& input,
    const array& packed_expert_ids,
    float limit) const {
    const bool split_gate_up = impl_->projections == 2;
    if (
        (!split_gate_up && impl_->projections != 1)
        || impl_->out_per_expert <= 0
        || (!split_gate_up && (impl_->out_per_expert & 1) != 0)
    ) {
        throw std::invalid_argument(
            "MFE SwiGLU requires one fused or two split gate/up projections");
    }
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "MFE SwiGLU limit must be finite and non-negative");
    }
    if (split_gate_up) {
        auto gate_up = routed_matmul_impl(
            input,
            packed_expert_ids,
            false,
            0.0f,
            nullptr,
            true);
        return limit > 0.0f
            ? moe_limited_swiglu_split(gate_up, limit)
            : moe_swiglu_split(gate_up);
    }
    return routed_matmul_impl(
        input,
        packed_expert_ids,
        true,
        limit,
        nullptr,
        true);
}

array MlxMfeWeight::routed_swiglu_pair(
    const MlxMfeWeight& up,
    const array& input,
    const array& expert_ids,
    float limit,
    const array* route_groups,
    const array* gate_map,
    const array* up_map) const {
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "paired MFE SwiGLU limit must be finite and non-negative");
    }
    if ((gate_map == nullptr) != (up_map == nullptr)
        || (!gate_map && impl_->experts != up.impl_->experts)
        || impl_->out_per_expert != up.impl_->out_per_expert
        || impl_->neuron_len != up.impl_->neuron_len
        || impl_->projections != 1
        || up.impl_->projections != 1) {
        throw std::invalid_argument(
            "paired MFE Gate and Up weights are incompatible");
    }
    if (gate_map && (gate_map->ndim() != 1 || up_map->ndim() != 1
            || gate_map->size() != up_map->size() || gate_map->size() == 0
            || gate_map->dtype() != mlx::core::int32 || up_map->dtype() != mlx::core::int32
            || !gate_map->flags().row_contiguous || !up_map->flags().row_contiguous)) {
        throw std::invalid_argument("paired MFE maps must be equal-sized contiguous int32 vectors");
    }
    if (route_groups && (expert_ids.ndim() != 2 || expert_ids.shape(0) < 2 || expert_ids.shape(0) > 24
            || route_groups->dtype() != mlx::core::int32 || !route_groups->flags().row_contiguous
            || route_groups->shape() != Shape{expert_ids.shape(0), expert_ids.shape(1), 8})) {
        throw std::invalid_argument("paired MFE route groups must be contiguous int32 [tokens,routes,8]");
    }
    const auto fallback = [&]() {
        auto gate_output = gate_map ? routed_matmul_mapped(input, expert_ids, *gate_map)
            : routed_matmul(input, expert_ids, route_groups);
        auto up_output = up_map ? up.routed_matmul_mapped(input, expert_ids, *up_map)
            : up.routed_matmul(input, expert_ids, route_groups);
        if (limit > 0.0f) {
            return moe_limited_swiglu_pair(
                std::move(gate_output),
                std::move(up_output),
                limit);
        }
        return moe_swiglu_split(mlx::core::concatenate(
            {std::move(gate_output), std::move(up_output)},
            -1));
    };
    const bool native_pair =
        !gate_map &&
        !impl_->indirect_sources && !up.impl_->indirect_sources &&
        impl_->native_primitive
        && up.impl_->native_primitive
        && impl_->family_mask
            == (std::uint32_t{1} << kFamilyMxfp4)
        && up.impl_->family_mask
            == (std::uint32_t{1} << kFamilyMxfp4)
        && impl_->rotations.empty()
        && up.impl_->rotations.empty()
        && (impl_->k_lanes_override == 0
            || impl_->k_lanes_override == 16)
        && (up.impl_->k_lanes_override == 0
            || up.impl_->k_lanes_override == 16)
        && input.ndim() == 2
        && expert_ids.ndim() == 2
        && input.shape(0) >= 1
        && input.shape(0) <= 6
        && expert_ids.shape(0) == input.shape(0)
        && expert_ids.shape(1) >= 1
        && expert_ids.shape(1) <= 16
        && input.shape(1) == impl_->neuron_len
        && impl_->out_per_expert > 0
        && impl_->neuron_len > 0
        && impl_->neuron_len % 32 == 0
        && impl_->descriptor_values.size()
            == static_cast<std::size_t>(impl_->experts)
                * kDescriptorSize
        && up.impl_->descriptor_values.size()
            == static_cast<std::size_t>(up.impl_->experts)
                * kDescriptorSize;
    if (!native_pair || mlx_reference_enabled()) {
        bool reusable_routes = input.ndim() == 2 && input.shape(0) == 1;
        if (!reusable_routes && route_groups && route_groups->is_available()) {
            const auto* groups = route_groups->data<std::int32_t>();
            reusable_routes = true;
            bool any = false;
            for (std::size_t index = 0; index < expert_ids.size(); ++index) {
                if (groups[index * 8] < 0) continue;
                any = true;
                reusable_routes &= groups[index * 8 + 1] > 1;
            }
            reusable_routes &= any;
        }
        const auto eligible = [](const Impl& value) {
            return value.native_primitive && value.reference_cohorts.empty()
                && value.mxfp4_sq_cohorts.empty() && value.fp8_sq_cohorts.empty()
                && !value.standalone_owner && value.k_lanes_override != 32;
        };
        if (reusable_routes && !mlx_reference_enabled() && mfe_split_swiglu_enabled()
            && input.ndim() == 2 && input.shape(0) >= 1 && input.shape(0) <= 6
            && expert_ids.ndim() == 2 && expert_ids.shape(0) == input.shape(0)
            && expert_ids.shape(1) > 0 && expert_ids.shape(1) <= 16
            && input.shape(1) == impl_->neuron_len
            && eligible(*impl_) && eligible(*up.impl_)
            && uses_indirect_expert_pages(impl_->compilation_profile().family_mask
                | up.impl_->compilation_profile().family_mask)) {
            const auto streams = [](const Impl& value) {
                return std::array<const array*, 21>{&value.nint_q, &value.nint_sub_scale, &value.nint_sub_min,
                    &value.nint_anchor_scale, &value.nint_anchor_min, &value.q8_q, &value.q8_scales,
                    &value.vq_indices, &value.vq_state, &value.vq_aux, &value.vq_anchors,
                    &value.vq_codebooks, &value.vq_scales, &value.vq_state_to_codebank, &value.vq_banks,
                    &value.vq_parameters, &value.vq_residual_codebooks, &value.vq_residual_first,
                    &value.vq_residual_second, &value.mx_values, &value.mx_scales};
            };
            const std::array sources{impl_, up.impl_};
            bool available = true;
            for (const auto& source : sources) {
                available &= source->descriptor_values.size() == std::size_t(source->experts) * kDescriptorSize;
                if (!source->indirect_addresses) for (const auto* value : streams(*source))
                    available &= value->is_available() && value->flags().row_contiguous;
            }
            if (available && input.shape(0) > 1 && !gate_map) {
                for (int expert = 0; expert < impl_->experts; ++expert) {
                    const auto* gate = impl_->descriptor_values.data() + expert * kDescriptorSize;
                    const auto* other = up.impl_->descriptor_values.data() + expert * kDescriptorSize;
                    const auto reusable = [](const auto* descriptor) {
                        const auto profile = descriptor[kVqProfile] & 255;
                        return descriptor[kFamily] == kFamilyMxfp4
                            || (descriptor[kFamily] == kFamilyVq
                                && profile < 32 && (474u & (1u << profile)) != 0u);
                    };
                    if ((gate[kFamily] != kFamilyMxfp4 && gate[kFamily] != kFamilyVq)
                        || (other[kFamily] != kFamilyMxfp4 && other[kFamily] != kFamilyVq)
                        || reusable(gate) != reusable(other)) {
                        available = false;
                        break;
                    }
                }
            }
            if (available && gate_map && input.shape(0) > 1) {
                for (const auto& source : sources) {
                    const auto profile = source->compilation_profile();
                    available &= (profile.family_mask & ~((1u << kFamilyMxfp4) | (1u << kFamilyVq))) == 0
                        && (profile.vq_profile_mask & ~474u) == 0;
                }
            }
            if (available) {
                const auto page_addresses = [&](const auto& source) {
                    if (source->indirect_addresses) return *source->indirect_addresses;
                    std::vector<std::uint64_t> addresses;
                    addresses.reserve(21);
                    for (const auto* value : streams(*source)) {
                        const auto* buffer = static_cast<const MTL::Buffer*>(value->buffer().ptr());
                        addresses.push_back(buffer->gpuAddress() + value->offset());
                    }
                    return array(addresses.begin(), Shape{1, 21}, mlx::core::uint64);
                };
                auto source = mlx::core::contiguous(input.dtype() == mlx::core::float16
                    || input.dtype() == mlx::core::float32 ? input : mlx::core::astype(input, mlx::core::float16));
                const int tokens = input.shape(0), routes = expert_ids.shape(1);
                const auto rotated_source = [&](const auto& weight) {
                    if (weight->rotations.empty()) return source;
                    std::vector<array> variants{source};
                    for (const auto& rotation : weight->rotations)
                        variants.push_back(apply_rotation(source, rotation, tokens, impl_->neuron_len));
                    return mlx::core::contiguous(mlx::core::concatenate(variants, 0));
                };
                auto inputs = impl_->narrow_bindings;
                inputs[1] = page_addresses(impl_);
                inputs[2] = up.impl_->descriptors;
                inputs[3] = page_addresses(up.impl_);
                inputs[4] = gate_map ? *gate_map : empty_packed_array(mlx::core::int32);
                inputs[5] = up_map ? *up_map : empty_packed_array(mlx::core::int32);
                inputs[6] = rotated_source(up.impl_);
                inputs.push_back(rotated_source(impl_));
                inputs.push_back(mlx::core::contiguous(mlx::core::astype(expert_ids, mlx::core::int32)));
                inputs.push_back(empty_packed_array(mlx::core::int32));
                inputs.emplace_back(std::initializer_list<float>{limit}, mlx::core::float32);
                if (route_groups) inputs.push_back(*route_groups);
                const bool prepared = impl_->page_resources && up.impl_->page_resources;
                if (!prepared) for (const auto& weight : sources) {
                    if (weight->indirect_addresses)
                        inputs.insert(inputs.end(), weight->indirect_owners.begin(), weight->indirect_owners.end());
                    else for (const auto* value : streams(*weight)) inputs.push_back(*value);
                }
                const auto gate_profile = impl_->compilation_profile();
                const auto up_profile = up.impl_->compilation_profile();
                std::vector<int> groups = *gate_profile.nint_group_sizes;
                groups.insert(groups.end(), up_profile.nint_group_sizes->begin(), up_profile.nint_group_sizes->end());
                std::sort(groups.begin(), groups.end());
                groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
                const auto families = gate_profile.family_mask | up_profile.family_mask;
                const int rows_per_simd = tokens == 1
                    && (families & ~((1u << kFamilyMxfp4) | (1u << kFamilyVq))) == 0 ? 2 : 1;
                const auto output_rows = std::size_t{2} * rows_per_simd;
                const int workgroups = checked_int(checked_product(checked_product(tokens, routes, "paired MFE routes"),
                    (std::size_t(impl_->out_per_expert) + output_rows - 1) / output_rows,
                    "paired MFE workgroups"), "paired MFE workgroups");
                tyloi::profiling::mark("mfe.dispatch.indirect_pair_native");
                return native_moe_dispatch(std::move(inputs), NativeMoeConfig{
                    .dtype = source.dtype(),
                    .output_shape = Shape{tokens, routes, impl_->out_per_expert},
                    .tokens = tokens,
                    .routes = routes,
                    .experts = impl_->experts,
                    .output_width = impl_->out_per_expert,
                    .matrix_output_width = impl_->out_per_expert,
                    .projections = 2,
                    .fused_swiglu = 1,
                    .input_width = impl_->neuron_len,
                    .k_lanes = 16,
                    .rows_per_simd = rows_per_simd,
                    .nint_group_sizes = std::make_shared<const std::vector<int>>(std::move(groups)),
                    .descriptor_size = kDescriptorSize,
                    .variant_stride = tokens,
                    .shared_input = 1,
                    .vq_execution_layout = gate_profile.vq_execution_layout || up_profile.vq_execution_layout,
                    .vq_layouts = gate_profile.vq_layouts | up_profile.vq_layouts,
                    .family_mask = static_cast<int>(gate_profile.family_mask | up_profile.family_mask),
                    .vq_profile_mask = static_cast<int>(gate_profile.vq_profile_mask | up_profile.vq_profile_mask),
                    .has_nepq_residual = impl_->has_nepq_residual || up.impl_->has_nepq_residual,
                    .npq_grouped_indices = impl_->npq_grouped_indices && up.impl_->npq_grouped_indices,
                    .expert_map_size = gate_map ? checked_int(gate_map->size(), "paired MFE expert map") : 0,
                    .route_groups = route_groups != nullptr,
                    .nint_q_dispatch = gate_profile.nint_q_dispatch && up_profile.nint_q_dispatch,
                    .indirect_pages = true,
                    .dual_pages = true,
                    .direct_page_mask = (impl_->indirect_addresses ? 0 : 1) | (up.impl_->indirect_addresses ? 0 : 2),
                    .other_experts = up.impl_->experts,
                    .workgroups = workgroups,
                    .page_resources = prepared ? impl_->page_resources : nullptr,
                    .other_page_resources = prepared ? up.impl_->page_resources : nullptr,
                });
            }
        }
        tyloi::profiling::mark("mfe.dispatch.mxfp4_pair_fallback");
        return fallback();
    }

    tyloi::profiling::mark("mfe.dispatch.mxfp4_pair_native");

    auto source = input;
    if (source.dtype() != mlx::core::float16
        && source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::contiguous(std::move(source));
    auto ids = mlx::core::contiguous(
        mlx::core::astype(expert_ids, mlx::core::int32));
    const int tokens = ids.shape(0);
    const int routes = ids.shape(1);
    constexpr int kOutputsPerWorkgroup = 4;
    const auto output_tiles =
        (static_cast<std::size_t>(impl_->out_per_expert)
             + kOutputsPerWorkgroup - 1u)
        / kOutputsPerWorkgroup;
    const auto routed_rows = checked_product(
        static_cast<std::size_t>(tokens),
        static_cast<std::size_t>(routes),
        "MXFP4 pair routed rows");
    const int workgroups = checked_int(
        checked_product(
            routed_rows,
            output_tiles,
            "MXFP4 pair workgroups"),
        "MXFP4 pair workgroups");
    return mxfp4_pair_swiglu_dispatch(
        {
            impl_->descriptors,
            impl_->mx_values,
            impl_->mx_scales,
            up.impl_->descriptors,
            up.impl_->mx_values,
            up.impl_->mx_scales,
            std::move(source),
            std::move(ids),
            array({limit}, mlx::core::float32),
        },
        Mxfp4PairSwiGluConfig{
            .dtype = input.dtype() == mlx::core::float32
                ? mlx::core::float32
                : mlx::core::float16,
            .output_shape = Shape{
                tokens,
                routes,
                impl_->out_per_expert,
            },
            .tokens = tokens,
            .routes = routes,
            .experts = impl_->experts,
            .output_width = impl_->out_per_expert,
            .input_width = impl_->neuron_len,
            .descriptor_size = kDescriptorSize,
            .workgroups = workgroups,
        });
}

std::optional<array> MlxMfeWeight::decode_nint_shared(
    const MlxMfeWeight& down,
    const MlxNintSwiGluPair& shared_gate_up,
    const MlxNintWeight& shared_down,
    const array& shared_gate_weight,
    const array& input,
    const array& expert_ids,
    const array& route_weights,
    bool shared_gate_is_logits,
    const array* route_groups,
    const array* router_logits) const {
    const auto supported_impl = [](const Impl& impl, int projections) {
        return impl.native_primitive && !impl.indirect_sources &&
            impl.projections == projections &&
            impl.rotations.empty() &&
            impl.reference_cohorts.empty() &&
            impl.mxfp4_sq_cohorts.empty() &&
            impl.fp8_sq_cohorts.empty() &&
            !impl.standalone_owner.has_value() &&
            impl.has_generic_cohorts &&
            impl.descriptor_values.size() ==
                static_cast<std::size_t>(impl.experts) *
                    static_cast<std::size_t>(projections) *
                    kDescriptorSize &&
            impl.narrow_bindings.size() == 22;
    };
    const bool activation16 =
        input.dtype() == mlx::core::float16 ||
        input.dtype() == mlx::core::bfloat16;
    const bool packed_shared_router = shared_gate_weight.dtype() == mlx::core::uint8 &&
        shared_gate_weight.ndim() == 1 && shared_gate_weight.size() >= 36;
    const bool shared_gate_dtype_supported = packed_shared_router ||
        shared_gate_weight.dtype() == mlx::core::float16 ||
        shared_gate_weight.dtype() == mlx::core::bfloat16 ||
        (shared_gate_is_logits && shared_gate_weight.dtype() == mlx::core::float32);
    const int routes = expert_ids.ndim() == 2
        ? expert_ids.shape(1) : 0;
    const int tokens = input.ndim() == 2 ? input.shape(0) : 0;
    const bool shared_gate_shape_supported = packed_shared_router || (shared_gate_is_logits
        ? shared_gate_weight.size() == static_cast<std::size_t>(tokens)
        : shared_gate_weight.ndim() == 2 && shared_gate_weight.shape(0) == 1 &&
            shared_gate_weight.shape(1) == impl_->neuron_len);
    const bool packed_gate_up = impl_->projections == 1;
    const int intermediate = down.impl_->neuron_len;
    const bool supported = !mlx_reference_enabled() &&
        routes > 0 && routes <= 16 &&
        (routes + 1) * 32 <= 1024 &&
        impl_->experts > 0 && impl_->experts % 32 == 0 &&
        routes <= impl_->experts &&
        supported_impl(*impl_, packed_gate_up ? 1 : 2) &&
        supported_impl(*down.impl_, 1) &&
        impl_->experts == down.impl_->experts &&
        impl_->out_per_expert > 0 &&
        impl_->neuron_len > 0 &&
        down.impl_->out_per_expert == impl_->neuron_len &&
        intermediate > 0 &&
        impl_->out_per_expert == intermediate * (packed_gate_up ? 2 : 1) &&
        tokens > 0 &&
        input.shape(1) == impl_->neuron_len && activation16 &&
        expert_ids.dtype() == mlx::core::int32 &&
        expert_ids.shape() == Shape{tokens, routes} &&
        route_weights.dtype() == mlx::core::float32 &&
        route_weights.shape() == Shape{tokens, routes} &&
        shared_gate_up.input_size() == impl_->neuron_len &&
        shared_gate_up.output_size() == intermediate &&
        shared_down.input_size() == intermediate &&
        shared_down.output_size() == impl_->neuron_len &&
        shared_gate_shape_supported &&
        shared_gate_weight.flags().row_contiguous && shared_gate_dtype_supported;
    if (!supported) {
        return std::nullopt;
    }
    int shared_router_group_size = 0;
    if (packed_shared_router) {
        auto packed = shared_gate_weight;
        packed.eval();
        std::array<std::uint32_t, 9> header{};
        std::memcpy(header.data(), packed.data<std::uint8_t>(), sizeof(header));
        shared_router_group_size = static_cast<int>(header[3]);
        const auto q = header[4] & 15u, k = (header[4] >> 8u) & 15u;
        if (shared_router_group_size <= 0 || header[8] != std::uint32_t(impl_->neuron_len) ||
            q < 1 || q > 8 || k < 1 || k > 8 ||
            header[0] != sizeof(header) || header[1] < header[0] ||
            header[2] < header[1] || header[2] > packed.size() ||
            (header[6] & 0x7c00u) == 0x7c00u || ((header[6] >> 16u) & 0x7c00u) == 0x7c00u)
            throw std::invalid_argument("invalid packed NINT router header");
        const auto groups = (std::uint64_t(impl_->neuron_len) + header[3] - 1) / header[3];
        const auto q_bytes = header[5] + (((header[4] >> 4u) & 7u) + groups * header[3] * q + 7) / 8;
        const auto sub_bytes = header[7] + (((header[4] >> 12u) & 7u) + groups * k + 7) / 8;
        if (q_bytes >= header[1] - header[0] || sub_bytes >= header[2] - header[1] ||
            sub_bytes >= packed.size() - header[2])
            throw std::invalid_argument("truncated packed NINT router streams");
    }
    const bool folded = router_logits != nullptr && tokens == 1 && route_groups == nullptr &&
        impl_->experts <= 4096 && router_logits->dtype() == mlx::core::float16 &&
        router_logits->shape() == Shape{1, impl_->experts};
    if (route_groups != nullptr && (tokens < 2 || tokens > 24 || route_groups->dtype() != mlx::core::int32 ||
            route_groups->shape() != Shape{tokens, routes, 8} || !route_groups->flags().row_contiguous))
        throw std::invalid_argument("two-stage MFE route groups must be contiguous int32 [tokens,routes,8] for M=2..24");

    auto source = mlx::core::contiguous(mlx::core::reshape(
        input.dtype() == mlx::core::float16
            ? input
            : mlx::core::astype(input, mlx::core::float16),
        Shape{tokens, impl_->neuron_len}));
    auto ids = mlx::core::contiguous(mlx::core::reshape(
        expert_ids, Shape{tokens * routes}));
    auto weights = mlx::core::contiguous(mlx::core::reshape(
        route_weights, Shape{tokens * routes}));
    auto shared_gate = mlx::core::contiguous(shared_gate_weight);

    const int decode_k_lanes = impl_->k_lanes_override != 0
        ? impl_->k_lanes_override
        : 16;
    constexpr int kGateRowsPerSimd = 1;
    const int gate_rows_per_workgroup =
        (folded ? kMfeFoldedGateSimdPairs : 1) *
        (32 / decode_k_lanes) * kGateRowsPerSimd;
    const int gate_tiles =
        (intermediate + gate_rows_per_workgroup - 1) /
        gate_rows_per_workgroup;
    const bool mixed_decode_two_rows = apple_m5_family() &&
        down.impl_->family_mask ==
            ((std::uint32_t{1} << kFamilyNint) |
             (std::uint32_t{1} << kFamilyVq));
    const int down_rows_per_simd = mixed_decode_two_rows ? 2 : 1;
    const int down_k_lanes = down.impl_->k_lanes_override != 0
        ? down.impl_->k_lanes_override
        : (tokens == 1 && impl_->neuron_len > intermediate && down_rows_per_simd > 1
            ? 32 : decode_k_lanes);
    const int down_rows_per_workgroup =
        (32 / down_k_lanes) * down_rows_per_simd;
    const int down_tiles =
        (down.impl_->out_per_expert + down_rows_per_workgroup - 1) /
        down_rows_per_workgroup;
    const int stage_one_width =
        (routes + 1) * intermediate + 1;

    MfeNintDecodeConfig config{
        .dtype = mlx::core::float16,
        .shared_gate_dtype = shared_gate.dtype(),
        .shared_gate_is_logits = shared_gate_is_logits,
        .shared_router_group_size = shared_router_group_size,
        .packed_gate_up = packed_gate_up,
        .route_groups = route_groups != nullptr,
        .folded_topk = folded,
        .gate_q_dispatch = impl_->compilation_profile().nint_q_dispatch,
        .down_q_dispatch = down.impl_->compilation_profile().nint_q_dispatch,
        .shared_gate_q_dispatch = shared_gate_up.has_uniform_q_bits(),
        .shared_down_q_dispatch = shared_down.has_uniform_q_bits(),
        .stage_one_shape = Shape{tokens, stage_one_width},
        .output_shape = Shape{tokens, impl_->neuron_len},
        .tokens = tokens,
        .routes = routes,
        .experts = impl_->experts,
        .hidden = impl_->neuron_len,
        .intermediate = intermediate,
        .descriptor_size = kDescriptorSize,
        .gate_family_mask = static_cast<int>(impl_->family_mask),
        .gate_vq_profile_mask = static_cast<int>(impl_->vq_profile_mask),
        .gate_vq_execution_layout =
            static_cast<int>(impl_->vq_execution_layout),
        .gate_vq_layouts = impl_->vq_layouts,
        .gate_has_nepq_residual =
            static_cast<int>(impl_->has_nepq_residual),
        .gate_npq_grouped_indices =
            static_cast<int>(impl_->npq_grouped_indices),
        .gate_nint_group_sizes = impl_->nint_group_sizes,
        .down_family_mask = static_cast<int>(down.impl_->family_mask),
        .down_vq_profile_mask =
            static_cast<int>(down.impl_->vq_profile_mask),
        .down_vq_execution_layout =
            static_cast<int>(down.impl_->vq_execution_layout),
        .down_vq_layouts = down.impl_->vq_layouts,
        .down_has_nepq_residual =
            static_cast<int>(down.impl_->has_nepq_residual),
        .down_npq_grouped_indices =
            static_cast<int>(down.impl_->npq_grouped_indices),
        .down_nint_group_sizes = down.impl_->nint_group_sizes,
        .shared_gate_group_size = shared_gate_up.group_size(),
        .shared_gate_groups = shared_gate_up.groups(),
        .shared_up_group_size = shared_gate_up.up_group_size(),
        .shared_up_groups = shared_gate_up.up_groups(),
        .shared_down_group_size = shared_down.group_size(),
        .shared_down_groups = shared_down.groups(),
        .k_lanes = decode_k_lanes,
        .down_k_lanes = down_k_lanes,
        .down_rows_per_simd = down_rows_per_simd,
        .stage_one_workgroups = 1 + (routes + 1) * gate_tiles,
        .stage_two_workgroups = down_tiles,
    };

    auto stage_one_inputs = impl_->narrow_bindings;
    stage_one_inputs.reserve(29 + static_cast<int>(route_groups != nullptr));
    stage_one_inputs.push_back(std::move(source));
    stage_one_inputs.push_back(folded ? mlx::core::contiguous(*router_logits) : ids);
    stage_one_inputs.push_back(shared_gate_up.packed_values());
    stage_one_inputs.push_back(shared_gate_up.row_metadata());
    stage_one_inputs.push_back(shared_gate_up.sub_scales());
    stage_one_inputs.push_back(shared_gate_up.sub_mins());
    stage_one_inputs.push_back(std::move(shared_gate));
    if (route_groups != nullptr) stage_one_inputs.push_back(*route_groups);
    auto stage_one_outputs = mfe_nint_decode_dispatch(
        std::move(stage_one_inputs), config, 1);
    auto stage_one = std::move(stage_one_outputs[0]);
    if (folded) {
        ids = stage_one_outputs[1];
        weights = stage_one_outputs[1];
    }

    auto stage_two_inputs = down.impl_->narrow_bindings;
    stage_two_inputs.reserve(29);
    stage_two_inputs.push_back(std::move(stage_one));
    stage_two_inputs.push_back(shared_down.packed_values());
    stage_two_inputs.push_back(shared_down.row_metadata());
    stage_two_inputs.push_back(shared_down.sub_scales());
    stage_two_inputs.push_back(shared_down.sub_mins());
    stage_two_inputs.push_back(std::move(ids));
    stage_two_inputs.push_back(std::move(weights));
    tyloi::profiling::mark("mfe.dispatch.two_stage_nint_shared");
    auto outputs = mfe_nint_decode_dispatch(std::move(stage_two_inputs), config, 2);
    return std::move(outputs.front());
}

bool MlxMfeWeight::supports_fused_routed_reduce(const array* route_groups) const noexcept {
    if (route_groups && (impl_->indirect_sources || impl_->rotations.size()
            || impl_->family_mask != (std::uint32_t{1} << kFamilyMxfp4))) {
        if (!route_groups->is_available() || route_groups->dtype() != mlx::core::int32
            || !route_groups->flags().row_contiguous || route_groups->ndim() != 3
            || route_groups->shape(2) != 8) return false;
        const auto* groups = route_groups->data<std::int32_t>();
        for (std::size_t index = 0; index < route_groups->size() / 8; ++index)
            if (groups[index * 8 + 1] > 1) return false;
    }
    return mxfp4_decode_down_reduce_enabled()
        && !mlx_reference_enabled()
        && impl_->native_primitive
        && impl_->projections == 1
        && impl_->reference_cohorts.empty() && impl_->mxfp4_sq_cohorts.empty()
        && impl_->fp8_sq_cohorts.empty() && !impl_->standalone_owner
        && (impl_->k_lanes_override == 0 || impl_->k_lanes_override == 16)
        && impl_->out_per_expert > 0
        && impl_->neuron_len > 0
        && impl_->neuron_len % 32 == 0
        && impl_->descriptor_values.size()
            == static_cast<std::size_t>(impl_->experts) * kDescriptorSize;
}

array MlxMfeWeight::routed_matmul_reduce(
    const array& input,
    const array& expert_ids,
    const array& route_weights,
    const array* route_groups) const {
    const auto fallback = [&]() {
        return moe_weighted_reduce(
            routed_matmul(input, expert_ids, route_groups),
            route_weights);
    };
    if (!supports_fused_routed_reduce()
        || input.ndim() != 3
        || expert_ids.ndim() != 2
        || route_weights.ndim() != 2
        || input.shape(0) < 1
        || input.shape(0) > 6
        || expert_ids.shape(0) != input.shape(0)
        || route_weights.shape(0) != input.shape(0)
        || input.shape(1) != expert_ids.shape(1)
        || route_weights.shape(1) != expert_ids.shape(1)
        || input.shape(2) != impl_->neuron_len
        || expert_ids.shape(1) <= 0
        || expert_ids.shape(1) > 16) {
        tyloi::profiling::mark("mfe.dispatch.mxfp4_down_reduce_fallback");
        return fallback();
    }

    const bool pure_mxfp4 = !impl_->indirect_sources && impl_->rotations.empty()
        && impl_->family_mask == (std::uint32_t{1} << kFamilyMxfp4);
    if (!pure_mxfp4) {
        if (route_groups) {
            if (route_groups->dtype() != mlx::core::int32
                || route_groups->shape() != Shape{input.shape(0), expert_ids.shape(1), 8}
                || !route_groups->flags().row_contiguous)
                throw std::invalid_argument("Down route groups must be contiguous int32 [tokens,routes,8]");
            if (!route_groups->is_available()) return fallback();
            const auto* groups = route_groups->data<std::int32_t>();
            for (std::size_t index = 0; index < expert_ids.size(); ++index)
                if (groups[index * 8 + 1] > 1) return fallback();
        }
        tyloi::profiling::mark("mfe.dispatch.mixed_down_reduce_native");
        return routed_matmul_impl(input, expert_ids, false, 0.0f, nullptr, false, nullptr, &route_weights);
    }

    tyloi::profiling::mark("mfe.dispatch.mxfp4_down_reduce_native");

    auto source = input;
    if (source.dtype() != mlx::core::float16
        && source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::contiguous(std::move(source));
    auto ids = mlx::core::contiguous(
        mlx::core::astype(expert_ids, mlx::core::int32));
    auto weights = mlx::core::contiguous(
        mlx::core::astype(route_weights, mlx::core::float32));
    const int tokens = ids.shape(0);
    const int routes = ids.shape(1);
    const int rows_per_lane_group =
        mxfp4_decode_down_reduce_rows();
    const int outputs_per_workgroup =
        2 * rows_per_lane_group;
    const int workgroups = checked_int(
        checked_product(
            static_cast<std::size_t>(tokens),
            (static_cast<std::size_t>(impl_->out_per_expert)
                 + static_cast<std::size_t>(outputs_per_workgroup) - 1u)
                / static_cast<std::size_t>(outputs_per_workgroup),
            "MXFP4 decode down-reduce workgroups"),
        "MXFP4 decode down-reduce workgroups");
    return mxfp4_decode_reduce_dispatch(
        {
            impl_->descriptors,
            impl_->mx_values,
            impl_->mx_scales,
            std::move(source),
            std::move(ids),
            std::move(weights),
        },
        Mxfp4DecodeReduceConfig{
            .dtype = input.dtype() == mlx::core::float32
                ? mlx::core::float32
                : mlx::core::float16,
            .output_shape = Shape{tokens, impl_->out_per_expert},
            .tokens = tokens,
            .routes = routes,
            .experts = impl_->experts,
            .output_width = impl_->out_per_expert,
            .input_width = impl_->neuron_len,
            .descriptor_size = kDescriptorSize,
            .rows_per_lane_group = rows_per_lane_group,
            .workgroups = workgroups,
        });
}

array MlxMfeWeight::routed_matmul_reduce_packed(
    const array& input,
    const array& packed_expert_ids,
    const array& route_weights) const {
    // Packed cache routes store the physical arena slot in the upper bits.
    // Decode the slots on device, then reuse the same fused MXFP4
    // down-projection/reduction primitive as the host-remapped path.
    auto physical_ids = mlx::core::subtract(
        mlx::core::right_shift(
            mlx::core::contiguous(mlx::core::astype(
                packed_expert_ids,
                mlx::core::int32)),
            array(8, mlx::core::int32)),
        array(1, mlx::core::int32));
    return routed_matmul_reduce(
        input,
        mlx::core::contiguous(std::move(physical_ids)),
        route_weights);
}

bool MlxMfeWeight::supports_grouped_mmq() const noexcept {
    // The block-list builder uses one Metal thread and one threadgroup-array
    // entry per addressable expert. Large shared SSD arenas can expose more
    // than the Metal 1024-thread limit even though a model routes to only a
    // small subset. They must use the ordinary mapped routed kernel until the
    // builder itself is made multi-threadgroup.
    if (impl_->experts > 1024) return false;
    if (impl_->projections == 1) return impl_->grouped_mmq;
    return !impl_->projection_views.empty()
        && std::all_of(
            impl_->projection_views.begin(),
            impl_->projection_views.end(),
            [](const auto& projection) {
                return projection->grouped_mmq;
            });
}

bool MlxMfeWeight::prefers_mxfp4_smallm_nax(
    const array& expert_ids) const noexcept {
    const int policy_experts = impl_->logical_experts > 0
        ? impl_->logical_experts
        : impl_->experts;
    return impl_->mxfp4_slot_ids.has_value()
        && impl_->projections == 1
        && impl_->rotations.empty()
        && expert_ids.ndim() == 2
        && mxfp4_nax_smallm_preferred(
            expert_ids,
            expert_ids.shape(0),
            impl_->experts,
            policy_experts,
            impl_->neuron_len,
            impl_->out_per_expert);
}

bool MlxMfeWeight::prefers_mxfp4_nax_prefill(
    int route_count) const noexcept {
    return route_count > 0
        && impl_->mxfp4_slot_ids.has_value()
        && impl_->projections == 1
        && impl_->rotations.empty()
        && mxfp4_nax_prefill_enabled(
            route_count,
            impl_->experts,
            impl_->neuron_len,
            impl_->out_per_expert,
            impl_->automatic_mxfp4_nax_prefill);
}

bool MlxMfeWeight::supports_mxfp4_blocks() const noexcept {
    return impl_->mxfp4_slot_ids.has_value()
        && impl_->mxfp4_slot_ids->dtype() == mlx::core::int32
        && impl_->mxfp4_slot_ids->size()
            == static_cast<std::size_t>(impl_->experts)
        && impl_->projections == 1
        && impl_->rotations.empty()
        && impl_->family_mask
            == (std::uint32_t{1} << kFamilyMxfp4);
}

bool MlxMfeWeight::supports_mxfp4_pair_blocks(
    const MlxMfeWeight& other) const noexcept {
    return supports_mxfp4_blocks()
        && other.supports_mxfp4_blocks()
        && impl_->experts == other.impl_->experts
        && impl_->out_per_expert == other.impl_->out_per_expert
        && impl_->neuron_len == other.impl_->neuron_len;
}

array MlxMfeWeight::mxfp4_blocks_sorted_impl(
    const MlxMfeWeight* other,
    const array& sorted_input,
    const MlxGroupedMmqPlan& plan) const {
    const bool paired = other != nullptr;
    if (
        sorted_input.ndim() != 2
        || sorted_input.shape(0) != plan.route_count
        || sorted_input.shape(1) != impl_->neuron_len
        || plan.experts != impl_->experts
        || plan.max_blocks <= 0
        || plan.block_rows != 32
    ) {
        throw std::invalid_argument(
            "MXFP4 block plan or sorted input is incompatible");
    }

    const auto packed_view = [](const Impl& impl) {
        const auto stride = checked_product(
            static_cast<std::size_t>(impl.out_per_expert),
            static_cast<std::size_t>(impl.neuron_len / 2),
            "MXFP4 block packed stride");
        if (stride == 0 || impl.mx_values.size() % stride != 0) {
            throw std::runtime_error(
                "MXFP4 block packed arena geometry is invalid");
        }
        const auto slots = impl.mx_values.size() / stride;
        return mlx::core::reshape(
            mlx::core::view(impl.mx_values, mlx::core::uint32),
            Shape{
                checked_int(slots, "MXFP4 block slot count"),
                impl.out_per_expert,
                impl.neuron_len / 8,
            });
    };
    const auto scale_view = [](const Impl& impl) {
        const auto stride = checked_product(
            static_cast<std::size_t>(impl.out_per_expert),
            static_cast<std::size_t>(impl.neuron_len / 32),
            "MXFP4 block scale stride");
        if (stride == 0 || impl.mx_scales.size() % stride != 0) {
            throw std::runtime_error(
                "MXFP4 block scale arena geometry is invalid");
        }
        const auto slots = impl.mx_scales.size() / stride;
        return mlx::core::reshape(
            impl.mx_scales,
            Shape{
                checked_int(slots, "MXFP4 block scale slot count"),
                impl.out_per_expert,
                impl.neuron_len / 32,
            });
    };
    // Match oMLX's DeepSeek-V4 MXFP4 path: the following SwiGLU stage already
    // consumes FP16, so running the block GEMM in BF16 only adds cost before
    // an unavoidable cast and does not preserve additional output precision.
    auto source = sorted_input.dtype() == mlx::core::float16
        ? sorted_input
        : mlx::core::astype(sorted_input, mlx::core::float16);
    source = mlx::core::expand_dims(
        mlx::core::contiguous(std::move(source)),
        1);
    const auto& second = paired ? *other->impl_ : *impl_;
    return mxfp4_blocks_dispatch(
        {
            std::move(source),
            packed_view(*impl_),
            scale_view(*impl_),
            *impl_->mxfp4_slot_ids,
            packed_view(second),
            scale_view(second),
            *second.mxfp4_slot_ids,
            plan.block_meta,
            plan.block_count,
        },
        Mxfp4BlocksConfig{
            .output_shape = Shape{
                plan.route_count,
                (paired ? 2 : 1) * impl_->out_per_expert,
            },
            .route_count = plan.route_count,
            .max_blocks = plan.max_blocks,
            .experts = impl_->experts,
            .output_width = impl_->out_per_expert,
            .input_width = impl_->neuron_len,
            .block_rows = plan.block_rows,
            .paired = paired,
        });
}

array MlxMfeWeight::mxfp4_block_matmul_sorted(
    const array& sorted_input,
    const MlxGroupedMmqPlan& plan) const {
    if (!supports_mxfp4_blocks()) {
        throw std::invalid_argument(
            "MXFP4 blocks require a pure-MXFP4 weight");
    }
    return mxfp4_blocks_sorted_impl(nullptr, sorted_input, plan);
}

array MlxMfeWeight::mxfp4_pair_swiglu_sorted(
    const MlxMfeWeight& other,
    const array& sorted_input,
    const MlxGroupedMmqPlan& plan,
    float limit) const {
    if (!supports_mxfp4_pair_blocks(other)) {
        throw std::invalid_argument(
            "MXFP4 pair blocks require compatible pure-MXFP4 weights");
    }
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "MXFP4 pair SwiGLU limit must be finite and non-negative");
    }
    auto pair = mxfp4_blocks_sorted_impl(&other, sorted_input, plan);
    return limit > 0.0f
        ? moe_limited_swiglu_split(std::move(pair), limit)
        : moe_swiglu_split(std::move(pair));
}

int MlxMfeWeight::recommended_mxfp4_nax_prefill_tokens(
    int routes_per_token) const noexcept {
    if (routes_per_token <= 0) return 0;
    if (!impl_->projection_views.empty()) {
        int recommendation = 0;
        for (const auto& projection : impl_->projection_views) {
            const int candidate = MlxMfeWeight(projection)
                .recommended_mxfp4_nax_prefill_tokens(routes_per_token);
            if (candidate == 0) return 0;
            recommendation = recommendation == 0
                ? candidate
                : std::min(recommendation, candidate);
        }
        return recommendation;
    }
    if (!impl_->mxfp4_slot_ids.has_value() ||
        impl_->projections != 1 || !impl_->rotations.empty()) {
        return 0;
    }
    // MLX 0.32's sorted NAX kernel is valid through 32768 routed rows.
    // Align token chunks to 32 for the packed prefill tiling. Top-k=6
    // therefore yields a 5440-token recommendation.
    constexpr int kMaximumSortedRows = 32768;
    constexpr int kTokenAlignment = 32;
    const int maximum_tokens =
        (kMaximumSortedRows / routes_per_token / kTokenAlignment) *
        kTokenAlignment;
    if (maximum_tokens <= 0 ||
        maximum_tokens >
            std::numeric_limits<int>::max() / routes_per_token) {
        return 0;
    }
    // The 256-expert 4096<->2048 geometry performs best with balanced 4096
    // token chunks on M3 Ultra; approaching the sorted-row ceiling regresses
    // its second chunk on long prompts. Keep this recommendation based on
    // operator geometry and device capabilities, not on a model identifier.
    const bool m3_ultra_256_geometry = apple_m3_ultra() &&
        impl_->experts == 256 && (
            (impl_->neuron_len == 4096 &&
             (impl_->out_per_expert == 2048 ||
              impl_->out_per_expert == 4096)) ||
            (impl_->neuron_len == 2048 &&
             impl_->out_per_expert == 4096));
    const int recommended_tokens = m3_ultra_256_geometry
        ? std::min(maximum_tokens, 4096)
        : maximum_tokens;
    const int route_count = recommended_tokens * routes_per_token;
    return mxfp4_nax_prefill_enabled(
               route_count,
               impl_->experts,
               impl_->neuron_len,
               impl_->out_per_expert,
               impl_->automatic_mxfp4_nax_prefill)
        ? recommended_tokens
        : 0;
}

int MlxMfeWeight::recommended_grouped_mmq_block_rows(
    int route_count,
    bool fused_swiglu) const noexcept {
    const bool fused_split_projections =
        fused_swiglu
        && impl_->projections == 2
        && impl_->grouped_mmq
        && impl_->rotations.empty();
    if (!impl_->projection_views.empty() && !fused_split_projections) {
        int selected = 0;
        for (const auto& projection : impl_->projection_views) {
            const int candidate = MlxMfeWeight(projection)
                .recommended_grouped_mmq_block_rows(route_count, false);
            if (selected != 0 && selected != candidate) return 32;
            selected = candidate;
        }
        return selected == 0 ? 32 : selected;
    }
    const bool use_nax = !impl_->has_nepq_residual
        && mixed_grouped_nax_enabled(route_count);
    constexpr bool direct_nax = false;
    if (!use_nax || direct_nax || impl_->experts <= 0) {
        if (!use_nax && impl_->experts > 0 && impl_->neuron_len >= 96
            && impl_->out_per_expert >= (fused_swiglu && impl_->projections == 1 ? 128 : 64)) {
            const std::int64_t mean_routes =
                (std::int64_t(route_count) + impl_->experts - 1) / impl_->experts;
            if (mean_routes > 64) return 96;
            if (mean_routes > 32) return 64;
        }
        return 32;
    }
    if (const char* value = std::getenv(
            "TYLOINFER_METAL_GROUPED_MMQ_BLOCK_ROWS")) {
        if (std::string_view(value) == "64") {
            return 64;
        }
        if (std::string_view(value) == "48") {
            return 48;
        }
        if (std::string_view(value) == "96") {
            return 96;
        }
        if (std::string_view(value) == "80") {
            return 80;
        }
        if (std::string_view(value) == "32") {
            return 32;
        }
    }
    const int mean_routes = (route_count + impl_->experts - 1)
        / impl_->experts;
    if (!fused_swiglu && mean_routes > 64
        && impl_->out_per_expert >= 64 && impl_->neuron_len >= 192) {
        return 128;
    }
    // Up to 96 mean routes, use one 16-row-aligned block per expert to avoid
    // wasting SIMD rows. Once multiple blocks are inevitable, choose between
    // the two high-occupancy plans by their actual padded-row cost. This keeps
    // the decision independent of model family, projection role, and packed
    // weight format while still preferring BM64 when both plans do equal work.
    if (mean_routes <= 32) {
        return 32;
    }
    if (mean_routes <= 48) {
        return 48;
    }
    if (mean_routes <= 64) {
        return 64;
    }
    if (mean_routes <= 80) {
        return 80;
    }
    if (mean_routes <= 96) {
        return 96;
    }
    (void)fused_swiglu;
    const auto padded_rows = [mean_routes](int rows) {
        return (
            (static_cast<std::int64_t>(mean_routes) + rows - 1)
            / rows
        ) * rows;
    };
    return padded_rows(80) < padded_rows(64) ? 80 : 64;
}

MlxGroupedMmqPlan MlxMfeWeight::build_grouped_mmq_plan(
    const array& expert_ids,
    const array& route_order,
    int block_rows) const {
    if (!supports_grouped_mmq()) {
        throw std::invalid_argument(
            "weight does not support grouped MMQ");
    }
    if (!impl_->projection_views.empty()) {
        return MlxMfeWeight(impl_->projection_views.front())
            .build_grouped_mmq_plan(
                expert_ids,
                route_order,
                block_rows);
    }
    return make_grouped_mmq_plan(
        expert_ids,
        route_order,
        *impl_->expert_order,
        impl_->experts,
        block_rows);
}

array MlxMfeWeight::routed_matmul_sorted(
    const array& input,
    const array& expert_ids,
    const array& route_order_value,
    bool input_is_sorted,
    bool fused_swiglu,
    float swiglu_limit,
    const MlxGroupedMmqPlan* plan,
    bool force_mxfp4_nax) const {
    const bool fused_split_projections =
        fused_swiglu
        && impl_->projections == 2
        && impl_->grouped_mmq
        && impl_->rotations.empty();
    if (!impl_->projection_views.empty() && !fused_split_projections) {
        if (fused_swiglu && impl_->projection_views.size() != 2) {
            throw std::invalid_argument(
                "split grouped SwiGLU requires exactly two projections");
        }
        if (!std::isfinite(swiglu_limit) || swiglu_limit < 0.0f) {
            throw std::invalid_argument(
                "grouped SwiGLU limit must be finite and non-negative");
        }
        std::vector<array> outputs;
        outputs.reserve(impl_->projection_views.size());
        for (const auto& projection : impl_->projection_views) {
            outputs.push_back(
                MlxMfeWeight(projection).routed_matmul_sorted(
                    input,
                    expert_ids,
                    route_order_value,
                    input_is_sorted,
                    false,
                    0.0f,
                    plan,
                    force_mxfp4_nax));
        }
        if (!fused_swiglu) return mlx::core::concatenate(std::move(outputs), 1);
        return swiglu_limit > 0.0f
            ? moe_limited_swiglu_pair(outputs.at(0), outputs.at(1), swiglu_limit)
            : moe_swiglu_pair(outputs.at(0), outputs.at(1));
    }
    const bool direct_mxfp4_nax =
        force_mxfp4_nax &&
        impl_->mxfp4_slot_ids.has_value() &&
        impl_->projections == 1 &&
        impl_->rotations.empty();
    if (!supports_grouped_mmq() && !direct_mxfp4_nax) {
        throw std::invalid_argument(
            "weight does not support grouped MMQ");
    }
    if (
        fused_swiglu
        && (
            impl_->out_per_expert <= 0
            || (impl_->projections == 1
                && (impl_->out_per_expert & 1) != 0)
            || (impl_->projections != 1
                && impl_->projections != 2)
            || !std::isfinite(swiglu_limit)
            || swiglu_limit < 0.0f
        )
    ) {
        throw std::invalid_argument(
            "fused grouped VQ SwiGLU parameters are invalid");
    }
    auto ids = mlx::core::contiguous(
        mlx::core::astype(
            expert_ids,
            mlx::core::int32));
    if (ids.ndim() != 2) {
        throw std::invalid_argument(
            "routed expert IDs must have [tokens,routes] shape");
    }
    const int tokens = ids.shape(0);
    const int routes = ids.shape(1);
    const auto route_count_size = checked_product(
        static_cast<std::size_t>(tokens),
        static_cast<std::size_t>(routes),
        "route count");
    const int route_count = checked_int(
        route_count_size,
        "route count");
    auto route_order = mlx::core::contiguous(
        mlx::core::astype(
            route_order_value,
            mlx::core::int32));
    if (
        route_order.ndim() != 1
        || route_order.size() != route_count_size
    ) {
        throw std::invalid_argument(
            "route order must contain one index per routed row");
    }

    bool shared_input = false;
    if (input_is_sorted) {
        if (
            input.ndim() != 2
            || input.shape(0) != route_count
            || input.shape(1) != impl_->neuron_len
        ) {
            throw std::invalid_argument(
                "sorted routed input must have [tokens*routes,K] shape");
        }
    } else if (
        input.ndim() == 2
        && input.shape(0) == tokens
        && input.shape(1) == impl_->neuron_len
    ) {
        shared_input = true;
    } else if (
        input.ndim() != 3
        || input.shape(0) != tokens
        || input.shape(1) != routes
        || input.shape(2) != impl_->neuron_len
    ) {
        throw std::invalid_argument(
            "routed input must have [tokens,K] or [tokens,routes,K] shape");
    }

    // This is a native MLX gather-QMM route. The old "NAX" identifiers are
    // retained at the public/internal compatibility boundary, but do not
    // imply that the current device must provide hardware NAX instructions.
    const bool use_mxfp4_nax =
        (mxfp4_nax_prefill_enabled(
             route_count,
             impl_->experts,
             impl_->neuron_len,
             impl_->out_per_expert,
             impl_->automatic_mxfp4_nax_prefill) ||
         force_mxfp4_nax)
        && impl_->mxfp4_slot_ids.has_value()
        && impl_->projections == 1
        && impl_->rotations.empty();
    // MLX's native gather-QMM accepts both BF16 and FP16 activations. Keep
    // BF16 intact on that path so raw-HF DeepSeek-V4 matches the checkpoint's
    // activation dtype. The heterogeneous Metal kernels still consume FP16,
    // so only the exact native MXFP4 path may bypass the existing cast.
    const bool preserve_bfloat16 =
        use_mxfp4_nax && input.dtype() == mlx::core::bfloat16;
    auto source = input.dtype() == mlx::core::float16 || preserve_bfloat16
        ? input
        : mlx::core::astype(input, mlx::core::float16);
    source = mlx::core::contiguous(source);
    const int variant_stride = shared_input ? tokens : route_count;
    if (!impl_->rotations.empty()) {
        source = mlx::core::contiguous(
            mlx::core::reshape(
                source,
                Shape{variant_stride, impl_->neuron_len}));
        std::vector<array> variants;
        variants.reserve(impl_->rotations.size() + 1);
        variants.push_back(source);
        for (const auto& rotation : impl_->rotations) {
            variants.push_back(
                apply_rotation(
                    source,
                    rotation,
                    variant_stride,
                    impl_->neuron_len));
        }
        source = mlx::core::contiguous(
            mlx::core::concatenate(
                std::move(variants),
                0));
    }

    // Prefill-sized pure-MXFP4 projections can use MLX's batched gather-QMM
    // without repacking. The automatic policy is selected from device and
    // tensor geometry; model identity is deliberately irrelevant.
    if (use_mxfp4_nax) {
        array sorted_source = [&]() {
            if (input_is_sorted) {
                return source;
            }
            if (shared_input) {
                auto source_rows = mlx::core::floor_divide(
                    route_order,
                    array(routes, mlx::core::int32));
                return mlx::core::take(
                    source,
                    source_rows,
                    0);
            }
            return mlx::core::take(
                mlx::core::reshape(
                    source,
                    Shape{route_count, impl_->neuron_len}),
                route_order,
                0);
        }();
        sorted_source = mlx::core::expand_dims(
            mlx::core::contiguous(std::move(sorted_source)),
            1);
        auto sorted_expert_ids = mlx::core::take(
            mlx::core::reshape(ids, Shape{route_count}),
            route_order,
            0);
        auto sorted_slot_ids = mlx::core::take(
            *impl_->mxfp4_slot_ids,
            sorted_expert_ids,
            0);
        const auto value_stride = checked_product(
            static_cast<std::size_t>(impl_->out_per_expert),
            static_cast<std::size_t>(impl_->neuron_len / 2),
            "MXFP4 NAX value stride");
        const auto scale_stride = checked_product(
            static_cast<std::size_t>(impl_->out_per_expert),
            static_cast<std::size_t>(impl_->neuron_len / 32),
            "MXFP4 NAX scale stride");
        const auto slots = impl_->mx_values.size() / value_stride;
        if (
            slots == 0
            || impl_->mx_values.size() % value_stride != 0
            || impl_->mx_scales.size() != slots * scale_stride
        ) {
            throw std::runtime_error(
                "MXFP4 NAX arena geometry is invalid");
        }
        auto packed = mlx::core::reshape(
            mlx::core::view(impl_->mx_values, mlx::core::uint32),
            Shape{
                checked_int(slots, "MXFP4 NAX slot count"),
                impl_->out_per_expert,
                impl_->neuron_len / 8,
            });
        auto scales = mlx::core::reshape(
            impl_->mx_scales,
            Shape{
                checked_int(slots, "MXFP4 NAX slot count"),
                impl_->out_per_expert,
                impl_->neuron_len / 32,
            });
        // MLX 0.32.0's sorted NAX kernel has row-offset overflow above 32768
        // routes and assumes K is divisible by 64.  The unsorted dispatch is
        // still valid for the already sorted rows and avoids both defects.
        const bool safe_sorted_indices =
            impl_->mxfp4_slot_ids_sorted
            && impl_->neuron_len % 64 == 0
            && route_count <= 32768;
        auto projected = mlx::core::squeeze(
            mlx::core::gather_qmm(
                sorted_source,
                packed,
                scales,
                std::nullopt,
                std::nullopt,
                sorted_slot_ids,
                true,
                32,
                4,
                "mxfp4",
                std::nullopt,
                safe_sorted_indices),
            1);
        return fused_swiglu
            ? moe_limited_swiglu_split(
                  std::move(projected),
                  swiglu_limit)
            : projected;
    }

    const array params({0.0f}, mlx::core::float32);
    std::vector<array> kernel_inputs = mfe_narrow_bindings_enabled() && impl_->native_primitive
        ? impl_->narrow_bindings
        : std::vector<array>{
              impl_->descriptors,
              impl_->nint_q,
              impl_->nint_sub_scale,
              impl_->nint_sub_min,
              impl_->nint_anchor_scale,
              impl_->nint_anchor_min,
              impl_->q8_q,
              impl_->q8_scales,
              impl_->vq_indices,
              impl_->vq_state,
              impl_->vq_aux,
              impl_->vq_anchors,
              impl_->vq_codebooks,
              impl_->vq_scales,
              impl_->vq_state_to_codebank,
              impl_->vq_banks,
              impl_->vq_parameters,
              impl_->vq_residual_codebooks,
              impl_->vq_residual_first,
              impl_->vq_residual_second,
              impl_->mx_values,
              impl_->mx_scales,
          };
    kernel_inputs.push_back(source);
    kernel_inputs.push_back(ids);
    kernel_inputs.push_back(route_order);
    kernel_inputs.push_back(params);
    const int output_width = fused_swiglu
        ? (impl_->projections == 2
            ? impl_->out_per_expert
            : impl_->out_per_expert / 2)
        : impl_->out_per_expert;
    const bool use_grouped_nax = !impl_->has_nepq_residual
        && mixed_grouped_nax_enabled(route_count);
    constexpr bool use_direct_nax = false;
    auto owned_plan = plan == nullptr
        ? std::optional<MlxGroupedMmqPlan>(
            make_grouped_mmq_plan(
                ids,
                route_order,
                *impl_->expert_order,
                impl_->experts,
                recommended_grouped_mmq_block_rows(
                    route_count,
                    fused_swiglu)))
        : std::nullopt;
    const auto& selected_plan = plan != nullptr
        ? *plan
        : *owned_plan;
    if (
        selected_plan.route_count != route_count
        || selected_plan.experts != impl_->experts
        || (selected_plan.block_rows != 32
            && selected_plan.block_rows != 48
            && selected_plan.block_rows != 64
            && selected_plan.block_rows != 80
            && selected_plan.block_rows != 96
            && selected_plan.block_rows != 128)
        || (!use_grouped_nax && selected_plan.block_rows != 32
            && selected_plan.block_rows != 64 && selected_plan.block_rows != 80
            && selected_plan.block_rows != 96)
        || (use_direct_nax && selected_plan.block_rows != 32)
        || selected_plan.max_blocks <= 0
    ) {
        throw std::invalid_argument(
            "grouped MMQ block plan does not match routed projection");
    }
    kernel_inputs.push_back(selected_plan.block_meta);
    kernel_inputs.push_back(selected_plan.block_count);
    if (impl_->indirect_addresses) {
        kernel_inputs.push_back(*impl_->indirect_addresses);
        kernel_inputs.insert(kernel_inputs.end(), impl_->indirect_owners.begin(), impl_->indirect_owners.end());
    }
    const auto compilation = impl_->compilation_profile();
    return grouped_mmq_dispatch(
        std::move(kernel_inputs),
        GroupedMmqConfig{
            .output_shape = Shape{
                route_count,
                output_width,
            },
            .route_count = route_count,
            .max_blocks = selected_plan.max_blocks,
            .block_rows = selected_plan.block_rows,
            .tile_columns = use_grouped_nax && !use_direct_nax
                ? grouped_mmq_tile_columns(
                    selected_plan.block_rows,
                    output_width,
                    fused_swiglu)
                : 64,
            .tokens = tokens,
            .routes = routes,
            .experts = impl_->experts,
            .output_width = output_width,
            .matrix_output_width = impl_->out_per_expert,
            .projections = impl_->projections,
            .input_width = impl_->neuron_len,
            .descriptor_size = kDescriptorSize,
            .variant_stride = variant_stride,
            .shared_input = static_cast<int>(shared_input),
            .input_sorted = static_cast<int>(input_is_sorted),
            .fused_swiglu = static_cast<int>(fused_swiglu),
            .has_nepq_residual = static_cast<int>(
                impl_->has_nepq_residual),
            .family_mask = static_cast<int>(impl_->family_mask),
            .vq_profile_mask = static_cast<int>(compilation.vq_profile_mask),
            .nint_group_sizes = impl_->nint_group_sizes,
            .cohort_populations = impl_->cohort_populations,
            .use_nax = use_grouped_nax,
            .direct_nax = use_direct_nax,
            .indirect_pages = impl_->indirect_addresses.has_value(),
            .swiglu_limit = swiglu_limit,
        });
}

array MlxMfeWeight::routed_matmul_impl(
    const array& input,
    const array& expert_ids,
    bool fused_swiglu,
    float swiglu_limit,
    const array* expert_map,
    bool packed_expert_ids,
    const array* route_groups,
    const array* reduce_weights) const {
    if (impl_->indirect_sources && mlx_reference_enabled())
        return concatenate_experts(*impl_->indirect_sources).with_kernel_profile(impl_->kernel_profile)
            .routed_matmul_impl(input, expert_ids, fused_swiglu, swiglu_limit, expert_map, packed_expert_ids, route_groups);
    if (impl_->projections > 1 && !impl_->projection_views.empty() &&
        std::any_of(
            impl_->projection_views.begin(),
            impl_->projection_views.end(),
            [](const auto& projection) {
                return !projection->reference_cohorts.empty()
                    || !projection->mxfp4_sq_cohorts.empty()
                    || !projection->fp8_sq_cohorts.empty();
            })) {
        if (fused_swiglu) {
            throw std::logic_error(
                "split standalone-kernel projections must use routed_swiglu");
        }
        std::vector<array> outputs;
        outputs.reserve(impl_->projection_views.size());
        for (const auto& projection : impl_->projection_views) {
            outputs.push_back(MlxMfeWeight(projection).routed_matmul_impl(
                input,
                expert_ids,
                false,
                0.0f,
                expert_map,
                packed_expert_ids, route_groups));
        }
        return mlx::core::concatenate(std::move(outputs), -1);
    }
    if (mlx_reference_enabled()) {
        if (expert_map != nullptr) {
            auto mapped_ids = mlx::core::take(
                mlx::core::contiguous(mlx::core::astype(
                    *expert_map,
                    mlx::core::int32)),
                expert_ids,
                0);
            return routed_bf16_reference(
                input,
                mapped_ids,
                fused_swiglu,
                swiglu_limit);
        }
        if (packed_expert_ids) {
            auto mapped_ids =
                mlx::core::floor_divide(
                    expert_ids,
                    array(256, mlx::core::int32)) -
                array(1, mlx::core::int32);
            return routed_bf16_reference(
                input,
                mapped_ids,
                fused_swiglu,
                swiglu_limit);
        }
        return routed_bf16_reference(
            input,
            expert_ids,
            fused_swiglu,
            swiglu_limit);
    }
    auto ids = mlx::core::contiguous(
        mlx::core::astype(
            expert_ids,
            mlx::core::int32));
    auto map = expert_map != nullptr
        ? mlx::core::contiguous(
              mlx::core::astype(*expert_map, mlx::core::int32))
        : ids;
    int expert_map_size = expert_map != nullptr
        ? checked_int(map.size(), "expert page-table size")
        : 0;
    if (expert_map != nullptr && map.ndim() != 1) {
        throw std::invalid_argument(
            "expert page table must be one-dimensional");
    }
    if (ids.ndim() != 2) {
        throw std::invalid_argument(
            "routed expert IDs must have "
            "[tokens,routes] shape");
    }
    const int tokens = ids.shape(0);
    const int routes = ids.shape(1);
    if (route_groups != nullptr && (tokens < 2 || tokens > 24 ||
            route_groups->dtype() != mlx::core::int32 ||
            route_groups->shape() != Shape{tokens, routes, 8} ||
            !route_groups->flags().row_contiguous)) {
        throw std::invalid_argument("MFE route groups must be contiguous int32 [tokens,routes,8] for M=2..24");
    }
    if (tokens < 0 || routes < 0) {
        throw std::invalid_argument(
            "routed expert dimensions cannot be negative");
    }
    bool shared_input = false;
    if (
        input.ndim() == 2
        && input.shape(0) == tokens
        && input.shape(1) == impl_->neuron_len
    ) {
        shared_input = true;
    } else if (
        input.ndim() != 3
        || input.shape(0) != tokens
        || input.shape(1) != routes
        || input.shape(2) != impl_->neuron_len
    ) {
        throw std::invalid_argument(
            "routed input must have [tokens,K] or "
            "[tokens,routes,K] shape");
    }

    auto source = input;
    if (
        source.dtype() != mlx::core::float16
        && source.dtype() != mlx::core::float32
    ) {
        source =
            mlx::core::astype(
                source,
                mlx::core::float16);
    }
    source = mlx::core::contiguous(source);
    const int logical_output_width = fused_swiglu
        ? (impl_->projections == 2
            ? impl_->out_per_expert
            : impl_->out_per_expert / 2)
        : impl_->out_per_expert;
    const auto output_width = fused_swiglu
        ? static_cast<std::size_t>(logical_output_width)
        : checked_product(
              static_cast<std::size_t>(impl_->projections),
              static_cast<std::size_t>(impl_->out_per_expert),
              "routed output width");
    const Shape output_shape = reduce_weights ? Shape{tokens, impl_->out_per_expert} : Shape{
        tokens,
        routes,
        checked_int(
            output_width,
            "routed output width"),
    };
    if (tokens == 0 || routes == 0) {
        return mlx::core::zeros(
            output_shape,
            source.dtype());
    }

    std::optional<array> standalone_output;
    if (!impl_->mxfp4_sq_cohorts.empty()
        || !impl_->fp8_sq_cohorts.empty()) {
        auto physical_ids = ids;
        if (packed_expert_ids) {
            physical_ids = mlx::core::floor_divide(
                ids,
                array(256, mlx::core::int32)) -
                array(1, mlx::core::int32);
        } else if (expert_map != nullptr) {
            if (expert_map_size <= 0) {
                physical_ids = mlx::core::full(
                    ids.shape(), -1, mlx::core::int32);
            } else {
                auto valid = mlx::core::logical_and(
                    mlx::core::greater_equal(
                        ids, array(0, mlx::core::int32)),
                    mlx::core::less(
                        ids,
                        array(expert_map_size, mlx::core::int32)));
                auto safe = mlx::core::minimum(
                    mlx::core::maximum(
                        ids, array(0, mlx::core::int32)),
                    array(expert_map_size - 1, mlx::core::int32));
                physical_ids = mlx::core::where(
                    valid,
                    mlx::core::take(map, safe, 0),
                    mlx::core::full(
                        ids.shape(), -1, mlx::core::int32));
            }
        }
        physical_ids = mlx::core::contiguous(
            mlx::core::astype(physical_ids, mlx::core::int32));
        const auto append_standalone = [&](array projected) {
            if (fused_swiglu) {
                projected = moe_limited_swiglu_split(
                    projected, swiglu_limit);
            }
            if (projected.dtype() != source.dtype()) {
                projected = mlx::core::astype(
                    projected, source.dtype());
            }
            standalone_output = standalone_output.has_value()
                ? *standalone_output + projected
                : std::move(projected);
        };
        for (const auto& cohort : impl_->mxfp4_sq_cohorts) {
            append_standalone(cohort.weight.routed_matmul(
                source,
                physical_ids,
                cohort.expert_map,
                impl_->out_per_expert));
        }
        for (const auto& cohort : impl_->fp8_sq_cohorts) {
            append_standalone(cohort.weight.routed_matmul(
                source,
                physical_ids,
                cohort.expert_map,
                impl_->out_per_expert));
        }
        if (!impl_->has_generic_cohorts) {
            return std::move(*standalone_output);
        }

        auto valid = mlx::core::logical_and(
            mlx::core::greater_equal(
                physical_ids, array(0, mlx::core::int32)),
            mlx::core::less(
                physical_ids,
                array(impl_->experts, mlx::core::int32)));
        auto safe = mlx::core::minimum(
            mlx::core::maximum(
                physical_ids, array(0, mlx::core::int32)),
            array(impl_->experts - 1, mlx::core::int32));
        auto owner = mlx::core::take(
            *impl_->standalone_owner, safe, 0);
        auto is_standalone = mlx::core::logical_and(
            valid,
            mlx::core::greater_equal(
                owner, array(0, mlx::core::int32)));
        ids = mlx::core::contiguous(mlx::core::where(
            is_standalone,
            mlx::core::full(ids.shape(), -1, mlx::core::int32),
            physical_ids));
        map = ids;
        expert_map = nullptr;
        expert_map_size = 0;
        packed_expert_ids = false;
    }

    const auto merge_standalone = [&](array output) {
        return standalone_output.has_value()
            ? output + *standalone_output
            : output;
    };

    const auto route_count_size = checked_product(
        static_cast<std::size_t>(tokens),
        static_cast<std::size_t>(routes),
        "route count");
    const int route_count = checked_int(
        route_count_size,
        "route count");
    const int variant_stride = shared_input ? tokens : route_count;
    const bool sorted_routes = routed_sort_enabled(tokens);
    auto route_order = mlx::core::zeros(
        Shape{1},
        mlx::core::int32);
    if (sorted_routes) {
        route_order = mlx::core::contiguous(
            mlx::core::astype(
                mlx::core::argsort(
                    mlx::core::reshape(
                        ids,
                        Shape{route_count})),
                mlx::core::int32));
    }
    if (!impl_->rotations.empty()) {
        source = mlx::core::contiguous(
            mlx::core::reshape(
                source,
                Shape{
                    variant_stride,
                    impl_->neuron_len,
                }));
        std::vector<array> variants;
        variants.reserve(
            impl_->rotations.size() + 1);
        variants.push_back(source);
        for (const auto& rotation : impl_->rotations) {
            variants.push_back(
                apply_rotation(
                    source,
                    rotation,
                    variant_stride,
                    impl_->neuron_len));
        }
        source = mlx::core::contiguous(
            mlx::core::concatenate(
                std::move(variants),
                0));
    }

    const bool split_fused_swiglu =
        fused_swiglu && impl_->projections == 2;
    const int k_lanes = impl_->k_lanes_override != 0
        ? impl_->k_lanes_override
        : 16;
    // Two rows reuse activation loads on the measured M5 mixed NINT/VQ
    // 640 -> 2560 decode geometry. Do not apply this to the inverse gate/up
    // projection: its larger K/register footprint regresses some cohorts.
    const auto compilation = impl_->compilation_profile();
    const bool mixed_decode_two_rows =
        impl_->projections == 1 && impl_->neuron_len == 640 &&
        impl_->out_per_expert == 2560 &&
        compilation.family_mask == ((std::uint32_t{1} << kFamilyNint) |
                               (std::uint32_t{1} << kFamilyVq));
    const int automatic_rows = mixed_decode_two_rows ||
        (reduce_weights && impl_->out_per_expert > impl_->neuron_len) ||
        (impl_->projections == 1 && compilation.family_mask ==
            (std::uint32_t{1} << kFamilyMxfp4)) ? 2 : 1;
    const int rows_per_simd =
        split_fused_swiglu
        ? 1
        : (tokens == 1 || (tokens <= 6 && mixed_decode_two_rows))
            && (impl_->rotations.empty() || reduce_weights)
        ? mfe_decode_rows_per_simd(automatic_rows)
        : 1;
    const auto rows_per_workgroup = static_cast<std::size_t>(
        (split_fused_swiglu || reduce_weights ? 32 : 64)
        / k_lanes * rows_per_simd);
    const auto output_tiles =
        (
            static_cast<std::size_t>(
                logical_output_width)
            + rows_per_workgroup - 1
        ) / rows_per_workgroup;
    auto workgroups = reduce_weights ? static_cast<std::size_t>(tokens) : route_count_size;
    workgroups = checked_product(
        workgroups,
        static_cast<std::size_t>(
            fused_swiglu ? 1 : impl_->projections),
        "routed projection count");
    workgroups = checked_product(
        workgroups,
        output_tiles,
        "routed workgroup count");
    const array params(
        {swiglu_limit},
        mlx::core::float32);
    std::vector<array> kernel_inputs = mfe_narrow_bindings_enabled() && impl_->native_primitive
        ? impl_->narrow_bindings
        : std::vector<array>{
              impl_->descriptors,
              impl_->nint_q,
              impl_->nint_sub_scale,
              impl_->nint_sub_min,
              impl_->nint_anchor_scale,
              impl_->nint_anchor_min,
              impl_->q8_q,
              impl_->q8_scales,
              impl_->vq_indices,
              impl_->vq_state,
              impl_->vq_aux,
              impl_->vq_anchors,
              impl_->vq_codebooks,
              impl_->vq_scales,
              impl_->vq_state_to_codebank,
              impl_->vq_banks,
              impl_->vq_parameters,
              impl_->vq_residual_codebooks,
              impl_->vq_residual_first,
              impl_->vq_residual_second,
              impl_->mx_values,
              impl_->mx_scales,
          };
    if (!impl_->native_primitive) {
        constexpr int min_device_elements = 8;
        for (auto& bank : kernel_inputs) {
            if (bank.size() < min_device_elements) {
                const int elements = static_cast<int>(bank.size());
                bank = mlx::core::concatenate({
                    mlx::core::reshape(bank, Shape{elements}),
                    mlx::core::zeros(Shape{min_device_elements - elements}, bank.dtype()),
                });
            }
        }
    }
    kernel_inputs.push_back(source);
    kernel_inputs.push_back(ids);
    kernel_inputs.push_back(route_order);
    kernel_inputs.push_back(params);
    if (
        sorted_routes
        && tokens >= 32
        && source.dtype() == mlx::core::float16
        && (impl_->projection_views.empty()
            || (fused_swiglu
                && impl_->projections == 2
                && impl_->rotations.empty()))
        && expert_map == nullptr
        && !packed_expert_ids
        && supports_grouped_mmq()
    ) {
        const bool use_grouped_nax = !impl_->has_nepq_residual
            && mixed_grouped_nax_enabled(route_count);
        constexpr bool use_direct_nax = false;
        auto plan = build_grouped_mmq_plan(
            ids,
            route_order,
            recommended_grouped_mmq_block_rows(
                route_count,
                fused_swiglu));
        kernel_inputs.push_back(plan.block_meta);
        kernel_inputs.push_back(plan.block_count);
        if (impl_->indirect_addresses) {
            kernel_inputs.push_back(*impl_->indirect_addresses);
            kernel_inputs.insert(kernel_inputs.end(), impl_->indirect_owners.begin(), impl_->indirect_owners.end());
        }
        auto sorted_outputs = grouped_mmq_dispatch(
            std::move(kernel_inputs),
            GroupedMmqConfig{
                .output_shape = Shape{
                    route_count,
                    logical_output_width,
                },
                .route_count = route_count,
                .max_blocks = plan.max_blocks,
                .block_rows = plan.block_rows,
                .tile_columns = use_grouped_nax && !use_direct_nax
                    ? grouped_mmq_tile_columns(
                        plan.block_rows,
                        logical_output_width,
                        fused_swiglu)
                    : 64,
                .tokens = tokens,
                .routes = routes,
                .experts = impl_->experts,
                .output_width = logical_output_width,
                .matrix_output_width = impl_->out_per_expert,
                .projections = impl_->projections,
                .input_width = impl_->neuron_len,
                .descriptor_size = kDescriptorSize,
                .variant_stride = variant_stride,
                .shared_input = static_cast<int>(shared_input),
                .fused_swiglu = static_cast<int>(fused_swiglu),
                .has_nepq_residual = static_cast<int>(
                    impl_->has_nepq_residual),
                .family_mask = static_cast<int>(impl_->family_mask),
                .vq_profile_mask = static_cast<int>(compilation.vq_profile_mask),
                .nint_group_sizes = impl_->nint_group_sizes,
                .cohort_populations = impl_->cohort_populations,
                .use_nax = use_grouped_nax,
                .direct_nax = use_direct_nax,
                .indirect_pages = impl_->indirect_addresses.has_value(),
                .swiglu_limit = swiglu_limit,
            });
        auto restored = moe_restore_routed_rows(sorted_outputs,route_order,ids);
        return merge_standalone(mlx::core::reshape(std::move(restored),output_shape));
    }
    if (impl_->native_primitive) {
        if (expert_map != nullptr) {
            kernel_inputs.push_back(map);
        }
        if (route_groups != nullptr) kernel_inputs.push_back(*route_groups);
        if (reduce_weights) kernel_inputs.push_back(mlx::core::contiguous(
            mlx::core::astype(*reduce_weights, mlx::core::float32)));
        if (impl_->indirect_addresses) {
            kernel_inputs[1] = *impl_->indirect_addresses;
            if (!impl_->page_resources)
                kernel_inputs.insert(kernel_inputs.end(), impl_->indirect_owners.begin(), impl_->indirect_owners.end());
        }
        return merge_standalone(native_moe_dispatch(
            std::move(kernel_inputs),
            NativeMoeConfig{
                .dtype = source.dtype(),
                .output_shape = output_shape,
                .tokens = tokens,
                .routes = routes,
                .experts = impl_->experts,
                .output_width = logical_output_width,
                .matrix_output_width = impl_->out_per_expert,
                .projections = impl_->projections,
                .fused_swiglu = static_cast<int>(fused_swiglu),
                .input_width = impl_->neuron_len,
                .k_lanes = k_lanes,
                .rows_per_simd = rows_per_simd,
                .nint_group_sizes = compilation.nint_group_sizes,
                .descriptor_size = kDescriptorSize,
                .variant_stride = variant_stride,
                .shared_input = static_cast<int>(shared_input),
                .vq_execution_layout = static_cast<int>(compilation.vq_execution_layout),
                .vq_layouts = compilation.vq_layouts,
                .family_mask = static_cast<int>(compilation.family_mask),
                .vq_profile_mask = static_cast<int>(
                    compilation.vq_profile_mask),
                .has_nepq_residual = static_cast<int>(
                    impl_->has_nepq_residual),
                .npq_grouped_indices = static_cast<int>(
                    impl_->npq_grouped_indices),
                .sorted_routes = static_cast<int>(
                    sorted_routes),
                .expert_map_size = expert_map_size,
                .packed_expert_ids = static_cast<int>(packed_expert_ids),
                .route_groups = route_groups != nullptr,
                .nint_q_dispatch = compilation.nint_q_dispatch,
                .indirect_pages = impl_->indirect_addresses.has_value(),
                .weighted_reduce = reduce_weights != nullptr,
                .workgroups = static_cast<int>(workgroups),
                .page_resources = impl_->page_resources,
            }));
    }
    const auto grid = checked_product(
        workgroups,
        64,
        "Metal grid");
    if (
        grid
        > static_cast<std::size_t>(
            std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error(
            "MFE compatibility grid exceeds MLX limits");
    }
    kernel_inputs.push_back(map);
    auto outputs = moe_kernel()(
        std::move(kernel_inputs),
        {output_shape},
        {source.dtype()},
        {
            static_cast<int>(grid),
            1,
            1,
        },
        {64, 1, 1},
        {
            {"T", source.dtype()},
            {"ROUTES", routes},
            {"EXPERTS", impl_->experts},
            {"OUT", logical_output_width},
            {"MATRIX_OUT", impl_->out_per_expert},
            {"PROJECTIONS", impl_->projections},
            {
                "FUSED_SWIGLU",
                static_cast<int>(fused_swiglu),
            },
            {"K", impl_->neuron_len},
            {"K_LANES_VALUE", k_lanes},
            {"ROWS_PER_SIMD_VALUE", rows_per_simd},
            {"DESCRIPTOR_SIZE", kDescriptorSize},
            {
                "SHARED_INPUT",
                static_cast<int>(shared_input),
            },
            {
                "VQ_EXECUTION_LAYOUT",
                static_cast<int>(
                    impl_->vq_execution_layout),
            },
            {
                "FAMILY_MASK",
                static_cast<int>(impl_->family_mask),
            },
            {
                "VQ_PROFILE_MASK",
                static_cast<int>(
                    impl_->vq_profile_mask),
            },
            {
                "HAS_NEPQ_RESIDUAL",
                static_cast<int>(
                    impl_->has_nepq_residual),
            },
            {
                "NPQ_GROUPED_INDICES",
                static_cast<int>(
                    impl_->npq_grouped_indices),
            },
            {
                "SORTED_ROUTES",
                static_cast<int>(sorted_routes),
            },
            {"EXPERT_MAP_SIZE", expert_map_size},
            {
                "PACKED_EXPERT_IDS",
                static_cast<int>(packed_expert_ids),
            },
        },
        std::nullopt,
        false,
        {});
    return merge_standalone(std::move(outputs.front()));
}

array MlxMfeWeight::routed_bf16_reference(
    const array& input,
    const array& expert_ids,
    bool fused_swiglu,
    float swiglu_limit) const {
    auto ids = mlx::core::contiguous(
        mlx::core::astype(
            expert_ids,
            mlx::core::int32));
    if (ids.ndim() != 2) {
        throw std::invalid_argument(
            "unpacked reference expert IDs must be [tokens,routes]");
    }
    const int tokens = ids.shape(0);
    const int routes = ids.shape(1);
    const bool shared_input =
        input.ndim() == 2
        && input.shape(0) == tokens
        && input.shape(1) == impl_->neuron_len;
    if (!shared_input && (
        input.ndim() != 3
        || input.shape(0) != tokens
        || input.shape(1) != routes
        || input.shape(2) != impl_->neuron_len)) {
        throw std::invalid_argument(
            "unpacked reference routed input shape mismatch");
    }
    if (impl_->projections != 1
        || impl_->reference_cohorts.empty()) {
        throw std::runtime_error(
            "unpacked reference MFE cohorts are unavailable");
    }

    ids.eval();
    std::vector<bool> selected(
        static_cast<std::size_t>(impl_->experts),
        false);
    const auto* id_values = ids.data<std::int32_t>();
    for (std::size_t index = 0; index < ids.size(); ++index) {
        const int expert = id_values[index];
        if (expert >= 0 && expert < impl_->experts) {
            selected[static_cast<std::size_t>(expert)] = true;
        }
    }

    const auto reference_dtype = mlx::core::float16;
    auto source = input.dtype() == reference_dtype
        ? input
        : mlx::core::astype(
              input,
              reference_dtype);
    auto output = mlx::core::zeros(
        Shape{
            tokens,
            routes,
            impl_->out_per_expert,
        },
        reference_dtype);
    for (int expert = 0; expert < impl_->experts; ++expert) {
        if (!selected[static_cast<std::size_t>(expert)]) {
            continue;
        }
        const ReferenceMoeCohort* cohort = nullptr;
        int local_expert = -1;
        for (const auto& candidate :
             impl_->reference_cohorts) {
            const auto found = std::find(
                candidate.expert_ids.begin(),
                candidate.expert_ids.end(),
                expert);
            if (found != candidate.expert_ids.end()) {
                cohort = &candidate;
                local_expert = static_cast<int>(
                    found - candidate.expert_ids.begin());
                break;
            }
        }
        if (cohort == nullptr) {
            throw std::runtime_error(
                "dense reference cannot locate selected expert");
        }
        const int row_begin =
            local_expert * impl_->out_per_expert;
        const auto rows = mlx::core::arange(
            row_begin,
            row_begin + impl_->out_per_expert,
            1,
            mlx::core::int32);
        auto dense = std::visit(
            [&](const auto& weight) {
                return mlx::core::astype(
                    weight.embedding(
                        rows,
                        mlx::core::float32),
                    reference_dtype);
            },
            cohort->weight);
        auto projected = mlx::core::matmul(
            source,
            mlx::core::transpose(dense));
        if (shared_input) {
            projected = mlx::core::expand_dims(
                projected,
                1);
        }
        auto mask = mlx::core::expand_dims(
            mlx::core::equal(
                ids,
                array(expert, mlx::core::int32)),
            -1);
        output = mlx::core::where(
            mask,
            projected,
            output);
        output.eval();
        mlx::core::clear_cache();
    }
    return fused_swiglu
        ? moe_limited_swiglu_split(
              output,
              swiglu_limit)
        : output;
}

int MlxMfeWeight::experts() const noexcept {
    return impl_->experts;
}

int MlxMfeWeight::out_per_expert() const noexcept {
    return impl_->out_per_expert;
}

int MlxMfeWeight::neuron_len() const noexcept {
    return impl_->neuron_len;
}

int MlxMfeWeight::projections() const noexcept {
    return impl_->projections;
}

std::size_t MlxMfeWeight::packed_copy_count() noexcept {
    return packed_copy_submissions.load(std::memory_order_relaxed);
}

std::size_t MlxMfeWeight::indirect_dispatch_count() noexcept {
    return indirect_expert_submissions.load(std::memory_order_relaxed);
}
std::size_t MlxMfeWeight::uncached_packed_bytes() noexcept {
    return uncached_packed_storage_bytes.load(std::memory_order_relaxed);
}

std::size_t MlxMfeWeight::pinned_packed_bytes() noexcept {
    return packed_pinned_bytes.load(std::memory_order_relaxed);
}

std::size_t MlxMfeWeight::packed_pin_failures() noexcept {
    return packed_pin_errors.load(std::memory_order_relaxed);
}

std::size_t MlxMfeWeight::pooled_staging_count() noexcept {
    return pooled_expert_staging.load(std::memory_order_relaxed);
}

std::shared_ptr<void> MlxMfeWeight::pin_packed_storage() const noexcept {
    struct Pin {
        explicit Pin(array value) : storage(std::move(value)) {}
        ~Pin() {
            if (!bytes) return;
            munlock(storage.data<std::uint8_t>(), bytes);
            packed_pinned_bytes.fetch_sub(bytes, std::memory_order_relaxed);
        }
        array storage;
        std::size_t bytes = 0;
    };
    try {
        if (!impl_->packed_arena || !impl_->packed_arena->is_available()
            || impl_->packed_arena->nbytes() < (std::size_t{64} << 10)) return {};
        const auto* data = impl_->packed_arena->data<std::uint8_t>();
        if (reinterpret_cast<std::uintptr_t>(data) % getpagesize()) return {};
        auto pin = std::make_shared<Pin>(*impl_->packed_arena);
        const auto bytes = pin->storage.buffer_size();
        if (mlock(data, bytes) != 0) {
            packed_pin_errors.fetch_add(1, std::memory_order_relaxed);
            return {};
        }
        pin->bytes = bytes;
        packed_pinned_bytes.fetch_add(bytes, std::memory_order_relaxed);
        return pin;
    } catch (...) {
        packed_pin_errors.fetch_add(1, std::memory_order_relaxed);
        return {};
    }
}

std::size_t MlxMfeWeight::packed_nbytes() const noexcept {
    return impl_->packed_bytes;
}

std::vector<std::shared_ptr<tyloi::TensorStorage>> MlxMfeWeight::backings() const {
    std::vector<std::shared_ptr<tyloi::TensorStorage>> result;
    std::unordered_set<const void*> seen;
    const auto append = [&](std::vector<std::shared_ptr<tyloi::TensorStorage>> storages) {
        for (auto& storage : storages) {
            if (!storage) continue;
            const void* identity = storage->owner ? storage->owner.get() : storage->base;
            if (identity && seen.insert(identity).second) result.push_back(std::move(storage));
        }
    };
    std::vector<array> roots;
    std::vector<std::shared_ptr<tyloi::TensorStorage>> nested;
    impl_->collect_cache_storage(roots, nested);
    append(std::move(nested));
    for (const auto& value : roots) append(tyloi::metal::backings(value));
    return result;
}

MlxRoutedLinear::MlxRoutedLinear(
    MlxMfeWeight weight)
    : weight_(std::move(weight)) {
    if (weight_.projections() != 1) {
        throw std::invalid_argument(
            "routed linear requires one MFE projection");
    }
}

MlxRoutedLinear MlxRoutedLinear::from_blob(
    std::span<const std::uint8_t> blob) {
    return MlxRoutedLinear(
        MlxMfeWeight::from_blob(blob));
}

array MlxRoutedLinear::forward(
    const array& input,
    const array& expert_ids) const {
    return weight_.routed_matmul(input, expert_ids);
}

array MlxRoutedLinear::forward_mapped(
    const array& input,
    const array& expert_ids,
    const array& expert_map) const {
    return weight_.routed_matmul_mapped(
        input,
        expert_ids,
        expert_map);
}

array MlxRoutedLinear::forward_packed(
    const array& input,
    const array& packed_expert_ids) const {
    return weight_.routed_matmul_packed(
        input,
        packed_expert_ids);
}

array MlxRoutedLinear::swiglu(
    const array& input,
    const array& expert_ids,
    float limit) const {
    return weight_.routed_swiglu(
        input,
        expert_ids,
        limit);
}

array MlxRoutedLinear::swiglu_mapped(
    const array& input,
    const array& expert_ids,
    const array& expert_map,
    float limit) const {
    return weight_.routed_swiglu_mapped(
        input,
        expert_ids,
        expert_map,
        limit);
}

array MlxRoutedLinear::swiglu_packed(
    const array& input,
    const array& packed_expert_ids,
    float limit) const {
    return weight_.routed_swiglu_packed(
        input,
        packed_expert_ids,
        limit);
}

array MlxRoutedLinear::swiglu_pair(
    const MlxRoutedLinear& up,
    const array& input,
    const array& expert_ids,
    float limit) const {
    return weight_.routed_swiglu_pair(
        up.weight_,
        input,
        expert_ids,
        limit);
}

array MlxRoutedLinear::combine(
    const array& input,
    const array& expert_ids,
    const array& route_weights) const {
    return weight_.routed_matmul_reduce(
        input,
        expert_ids,
        route_weights);
}

array MlxRoutedLinear::combine_mapped(
    const array& input,
    const array& expert_ids,
    const array& expert_map,
    const array& route_weights) const {
    return moe_weighted_reduce(
        forward_mapped(input, expert_ids, expert_map),
        route_weights);
}

array MlxRoutedLinear::combine_packed(
    const array& input,
    const array& packed_expert_ids,
    const array& route_weights) const {
    return weight_.routed_matmul_reduce_packed(
        input,
        packed_expert_ids,
        route_weights);
}

MlxRoutedSwiGluFfn::MlxRoutedSwiGluFfn(
    MlxMfeWeight gate,
    MlxMfeWeight up,
    MlxMfeWeight down)
    : gate_up_(
          MlxMfeWeight::concatenate_projections(
              {std::move(gate), std::move(up)})),
      down_(std::move(down)) {
    if (
        gate_up_.experts() != down_.experts()
        || gate_up_.out_per_expert()
            != down_.neuron_len()
        || gate_up_.neuron_len()
            != down_.out_per_expert()
        || down_.projections() != 1
    ) {
        throw std::invalid_argument(
            "routed SwiGLU gate/up/down shapes "
            "are incompatible");
    }
}

MlxRoutedSwiGluFfn
MlxRoutedSwiGluFfn::from_blobs(
    std::span<const std::uint8_t> gate,
    std::span<const std::uint8_t> up,
    std::span<const std::uint8_t> down) {
    return MlxRoutedSwiGluFfn(
        MlxMfeWeight::from_blob(gate),
        MlxMfeWeight::from_blob(up),
        MlxMfeWeight::from_blob(down));
}

array MlxRoutedSwiGluFfn::forward(
    const array& input,
    const array& expert_ids,
    const array& route_weights) const {
    auto gate_up =
        gate_up_.routed_matmul(
            input,
            expert_ids);
    auto hidden =
        moe_swiglu_split(gate_up);
    auto pair_output =
        down_.routed_matmul(
            hidden,
            expert_ids);
    return moe_weighted_reduce(
        pair_output,
        route_weights);
}

MlxRoutedFfnResult
MlxRoutedSwiGluFfn::forward_from_logits(
    const array& input,
    const array& router_logits,
    int top_k,
    bool use_sigmoid,
    bool use_sqrt_softplus,
    bool normalize,
    bool delayed_softmax,
    const std::optional<array>& bias,
    const std::optional<array>& available,
    float norm_floor,
    float scale) const {
    auto routes = moe_topk(
        router_logits,
        top_k,
        use_sigmoid,
        use_sqrt_softplus,
        normalize,
        delayed_softmax,
        bias,
        available,
        norm_floor,
        scale);
    auto output = forward(
        input,
        routes.ids,
        routes.weights);
    return {
        std::move(output),
        std::move(routes.ids),
        std::move(routes.weights),
    };
}

} // namespace tyloi::metal

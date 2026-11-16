#include "engine/weights/cache/metal_mfe.h"
#include "engine/backends/metal/ops/mfq/mlx_moe_cache_internal.h"
#include "engine/weights/cache/lru.h"
#include "engine/weights/source/source.h"
#include "engine/backends/cpu/native/host_parallel.h"
#include "mlx_kernel_prepare.h"
#include "mlx_resource_telemetry.h"
#include "mlx_resident_budget.h"
#include "mlx_nint.h"
#include "mlx_nint_metadata.h"
#include "mlx_moe_ops.h"
#include "mlx_vq.h"
#include "mlx_mx.h"
#include "engine/backends/metal/native/memory.h"
#include "engine/backends/metal/native/env.h"
#include "engine/backends/metal/native/allocation.h"
#include <mlx/mlx.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <coroutine>
#include <cstring>
#include <dispatch/dispatch.h>
#include <exception>
#include <future>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
#include <unistd.h>

namespace tyloi::metal {
namespace {
using mlx::core::Shape;
using mlx::core::array;
using namespace mfe_internal;
std::uint64_t checked_range_add(
    std::uint64_t left,
    std::uint64_t right,
    const char* name) {
    if (
        left
        > std::numeric_limits<std::uint64_t>::max()
            - right
    ) {
        throw std::runtime_error(
            std::string("streamed MFE ") + name
            + " overflows");
    }
    return left + right;
}

std::uint64_t checked_range_product(
    std::uint64_t left,
    std::uint64_t right,
    const char* name) {
    if (
        left != 0
        && right
            > std::numeric_limits<std::uint64_t>::max()
                / left
    ) {
        throw std::runtime_error(
            std::string("streamed MFE ") + name
            + " overflows");
    }
    return left * right;
}

template <typename T>
T read_scalar(
    const std::vector<std::uint8_t>& bytes,
    std::size_t offset,
    const char* name) {
    if (
        offset > bytes.size()
        || sizeof(T) > bytes.size() - offset
    ) {
        throw std::runtime_error(
            std::string("truncated streamed MFE ") + name);
    }
    T value{};
    std::memcpy(
        &value,
        bytes.data() + offset,
        sizeof(T));
    return value;
}

std::string read_ascii(
    const std::vector<std::uint8_t>& bytes,
    std::size_t offset,
    std::size_t count,
    const char* name) {
    if (
        offset > bytes.size()
        || count > bytes.size() - offset
    ) {
        throw std::runtime_error(
            std::string("truncated streamed MFE ") + name);
    }
    const auto begin =
        bytes.begin()
        + static_cast<std::ptrdiff_t>(offset);
    const auto end =
        begin + static_cast<std::ptrdiff_t>(count);
    if (
        std::any_of(
            begin,
            end,
            [](std::uint8_t value) {
                return value > 0x7fu;
            })
    ) {
        throw std::runtime_error(
            std::string("streamed MFE ") + name
            + " is not ASCII");
    }
    return {
        reinterpret_cast<const char*>(
            bytes.data() + offset),
        count,
    };
}

std::uint32_t read_packed(
    const std::vector<std::uint8_t>& bytes,
    std::size_t bit_offset,
    int bits) {
    const auto byte_offset = bit_offset / 8;
    const auto shift =
        static_cast<unsigned>(bit_offset & 7);
    std::uint32_t packed = 0;
    for (unsigned index = 0; index < 3; ++index) {
        const auto source = byte_offset + index;
        if (source < bytes.size()) {
            packed |=
                static_cast<std::uint32_t>(
                    bytes[source])
                << (8 * index);
        }
    }
    return (
        packed >> shift
    ) & ((std::uint32_t{1} << bits) - 1u);
}

void write_packed(
    std::vector<std::uint8_t>& bytes,
    std::size_t value_index,
    int bits,
    std::uint32_t value) {
    const auto bit_offset =
        value_index * static_cast<std::size_t>(bits);
    for (int bit = 0; bit < bits; ++bit) {
        if (((value >> bit) & 1u) == 0u) {
            continue;
        }
        const auto target =
            bit_offset
            + static_cast<std::size_t>(bit);
        bytes[target / 8] |=
            static_cast<std::uint8_t>(
                1u << (target & 7));
    }
}

class MfeStreamUnsupported final
    : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

template <typename T>
struct MfeReadAllocator : std::allocator<T> {
    template <typename U> struct rebind { using other = MfeReadAllocator<U>; };
    MfeReadAllocator() = default;
    template <typename U> MfeReadAllocator(const MfeReadAllocator<U>&) {}
    void construct(T* value) { ::new (static_cast<void*>(value)) T; }
    template <typename U, typename... Args>
    void construct(U* value, Args&&... args) { std::construct_at(value, std::forward<Args>(args)...); }
    template <typename U> bool operator==(const MfeReadAllocator<U>&) const { return true; }
};
using MfeReadBytes = std::vector<std::uint8_t, MfeReadAllocator<std::uint8_t>>;

template <typename T, typename Bytes>
void append_scalar_bytes(
    Bytes& target,
    T value) {
    const auto previous = target.size();
    target.resize(previous + sizeof(T));
    std::memcpy(target.data() + previous, &value, sizeof(T));
}

template <typename T, typename Bytes>
void patch_scalar_bytes(
    Bytes& target,
    std::size_t offset,
    T value,
    const char* name) {
    if (offset > target.size() || sizeof(T) > target.size() - offset) {
        throw std::runtime_error(
            std::string("MFE streamed ") + name + " patch is out of range");
    }
    std::memcpy(target.data() + offset, &value, sizeof(T));
}

std::vector<std::uint8_t> unpack_small_selectors(
    const std::vector<std::uint8_t>& packed,
    std::size_t count,
    int bits,
    const char* name) {
    std::vector<std::uint8_t> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto value = read_packed(
            packed,
            index * static_cast<std::size_t>(bits),
            bits);
        if (value >= (std::uint32_t{1} << bits)) {
            throw std::runtime_error(
                std::string("invalid streamed MFE ") + name);
        }
        result[index] = static_cast<std::uint8_t>(value);
    }
    return result;
}

std::vector<std::uint8_t> pack_small_selectors(
    std::span<const std::uint8_t> values,
    int bits) {
    std::vector<std::uint8_t> result(
        checked_packed_size(values.size(), bits, "selector bytes"),
        0);
    for (std::size_t index = 0; index < values.size(); ++index) {
        write_packed(result, index, bits, values[index]);
    }
    return result;
}

struct MfeReadCounters {
    std::size_t bytes = 0, calls = 0, parallel_body_batches = 0;
    double seconds = 0;
};
struct MfeReadRange {
    std::uint64_t source;
    std::size_t destination;
    std::size_t bytes;
};
using MfeReadPlan = std::vector<MfeReadRange>;
thread_local MfeReadPlan* current_mfe_plan = nullptr;
struct MfeReadPlanScope {
    explicit MfeReadPlanScope(MfeReadPlan* value)
        : previous(std::exchange(current_mfe_plan, value)) {}
    ~MfeReadPlanScope() { current_mfe_plan = previous; }
    MfeReadPlan* previous;
};
thread_local MfeReadCounters* current_mfe_reads = nullptr;
struct MfeReadScope {
    explicit MfeReadScope(MfeReadCounters* value)
        : previous(std::exchange(current_mfe_reads, value)) {}
    ~MfeReadScope() { current_mfe_reads = previous; }
    MfeReadCounters* previous;
};
void record_mfe_read(std::size_t bytes, std::chrono::steady_clock::time_point started) {
    if (!current_mfe_reads || !bytes) return;
    current_mfe_reads->bytes += bytes;
    ++current_mfe_reads->calls;
    current_mfe_reads->seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
}

struct MfeRecord {
    TensorMetadata metadata;
    source::Source::TensorReader reader;
    source::Source::TensorBatchReader batch;
    std::shared_ptr<std::mutex> serial;

    void read_range_into(const std::string&, std::uint64_t offset, std::span<std::byte> destination) const {
        if (offset > metadata.nbytes || destination.size() > metadata.nbytes - offset)
            throw std::out_of_range("MFE record range is out of bounds: " + metadata.name);
        std::unique_lock<std::mutex> lock;
        if (serial) lock = std::unique_lock<std::mutex>(*serial);
        if (reader) reader(offset, destination.data(), destination.size());
        else if (batch) batch({{offset, destination.data(), destination.size()}});
        else throw TensorReaderUnsupported("MFE record has no retained reader: " + metadata.name);
    }
    std::vector<std::uint8_t> read_range(const std::string& name, std::uint64_t offset, std::uint64_t size) const {
        if (size > std::numeric_limits<std::size_t>::max()) throw std::overflow_error("MFE record range is too large");
        std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
        read_range_into(name, offset, std::as_writable_bytes(std::span(result)));
        return result;
    }
    void readv_range(const std::string&, std::uint64_t offset, std::span<const std::span<std::byte>> destinations) const {
        std::vector<ReadSpan> spans;
        spans.reserve(destinations.size());
        for (auto destination : destinations) {
            if (offset > metadata.nbytes || destination.size() > metadata.nbytes - offset)
                throw std::out_of_range("MFE record range is out of bounds: " + metadata.name);
            spans.push_back({offset, destination.data(), destination.size()});
            offset += destination.size();
        }
        std::unique_lock<std::mutex> lock;
        if (serial) lock = std::unique_lock<std::mutex>(*serial);
        if (batch) batch(spans);
        else if (reader) for (const auto& span : spans) reader(span.offset, span.destination, span.size);
        else throw TensorReaderUnsupported("MFE record has no retained reader: " + metadata.name);
    }
};

void read_mfe_direct_targets(const MfeRecord& model, const std::string& name,
    std::span<const MfeDirectReadTarget> targets, bool complete, std::size_t missing) {
    if (missing > kMfeDirectRangeMaxExperts || targets.size() < 2) {
        for (const auto& range : targets) {
            const auto started = std::chrono::steady_clock::now();
            auto* destination = complete ? range.destination : range.wire;
            model.read_range_into(name, range.offset,
                std::as_writable_bytes(std::span(destination, range.bytes)));
            record_mfe_read(range.bytes, started);
        }
        return;
    }
    struct Context {
        const MfeRecord& model;
        const std::string& name;
        std::span<const MfeDirectReadTarget> targets;
        bool complete;
        std::mutex mutex;
        std::exception_ptr failure;
    } context{model, name, targets, complete, {}, {}};
    const auto started = std::chrono::steady_clock::now();
    dispatch_apply_f(targets.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        &context, [](void* raw, std::size_t index) {
            auto& work = *static_cast<Context*>(raw);
            const auto& range = work.targets[index];
            try {
                auto* destination = work.complete ? range.destination : range.wire;
                work.model.read_range_into(work.name, range.offset,
                    std::as_writable_bytes(std::span(destination, range.bytes)));
            } catch (...) {
                std::scoped_lock lock(work.mutex);
                if (!work.failure) work.failure = std::current_exception();
            }
        });
    if (context.failure) std::rethrow_exception(context.failure);
    if (current_mfe_reads) {
        for (const auto& range : targets) {
            current_mfe_reads->bytes += range.bytes;
            ++current_mfe_reads->calls;
        }
        ++current_mfe_reads->parallel_body_batches;
        current_mfe_reads->seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
    }
}

template <typename Bytes>
void append_record_range(
    Bytes& target,
    const MfeRecord& model,
    const std::string& name,
    std::uint64_t offset,
    std::uint64_t nbytes) {
    const auto begin = target.size();
    target.resize(checked_add(begin, nbytes, "streamed appended range"));
    if (current_mfe_plan) {
        if (nbytes) current_mfe_plan->push_back({offset, begin, static_cast<std::size_t>(nbytes)});
        return;
    }
    const auto started = current_mfe_reads ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    model.read_range_into(name, offset,
        std::as_writable_bytes(std::span<std::uint8_t>(target).subspan(begin)));
    record_mfe_read(nbytes, started);
}

void read_mfe_batch(const MfeRecord& model, const std::string& name,
    std::span<MfeReadBytes> blobs, std::span<const MfeReadPlan> plans) {
    struct Range {
        std::uint64_t source;
        std::span<std::uint8_t> target;
    };
    std::vector<Range> ranges;
    for (std::size_t index = 0; index < plans.size(); ++index)
        for (const auto& range : plans[index]) ranges.push_back({range.source,
            std::span(blobs[index]).subspan(range.destination, range.bytes)});
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) { return a.source < b.source; });
    constexpr std::size_t max_read = std::size_t{8} << 20;
    MfeReadBytes scratch;
    std::vector<std::span<std::byte>> destinations;
    for (std::size_t first = 0; first < ranges.size();) {
        const auto offset = ranges[first].source;
        auto end = checked_range_add(offset, ranges[first].target.size(), "MFE batch range");
        auto last = first + 1;
        bool contiguous = true;
        while (last < ranges.size() && ranges[last].source <= end) {
            const auto next = checked_range_add(ranges[last].source, ranges[last].target.size(), "MFE batch range");
            if (next - offset > max_read) break;
            contiguous = contiguous && ranges[last].source == end;
            end = std::max(end, next);
            ++last;
        }
        const auto bytes = static_cast<std::size_t>(end - offset);
        const auto started = std::chrono::steady_clock::now();
        if (last == first + 1) {
            model.read_range_into(name, offset, std::as_writable_bytes(ranges[first].target));
            record_mfe_read(bytes, started);
        } else if (contiguous) {
            destinations.clear();
            for (auto index = first; index < last; ++index)
                destinations.push_back(std::as_writable_bytes(ranges[index].target));
            model.readv_range(name, offset, destinations);
            record_mfe_read(bytes, started);
        } else {
            scratch.resize(bytes);
            model.read_range_into(name, offset, std::as_writable_bytes(std::span(scratch)));
            record_mfe_read(bytes, started);
            for (auto index = first; index < last; ++index)
                std::memcpy(ranges[index].target.data(), scratch.data() + ranges[index].source - offset,
                    ranges[index].target.size());
        }
        first = last;
    }
}

template <typename Bytes>
void append_packed_value_slice(
    Bytes& target,
    const MfeRecord& model,
    const std::string& name,
    std::uint64_t stream_offset,
    std::size_t first_value,
    std::size_t value_count,
    int bits) {
    if (value_count == 0) {
        return;
    }
    const auto source_bit = checked_product(
        first_value,
        static_cast<std::size_t>(bits),
        "streamed source bit offset");
    const auto bit_count = checked_product(
        value_count,
        static_cast<std::size_t>(bits),
        "streamed value bit count");
    if ((source_bit & 7u) == 0u && (bit_count & 7u) == 0u) {
        append_record_range(
            target, model, name,
            checked_range_add(
                stream_offset,
                source_bit / 8,
                "streamed aligned source offset"),
            bit_count / 8);
        return;
    }
    const auto source_byte = source_bit / 8;
    const auto shift = source_bit & 7u;
    const auto source_bytes = checked_add(
        checked_add(shift, bit_count, "streamed source bits"),
        7,
        "streamed source rounding") / 8;
    const auto started = current_mfe_reads ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    const auto raw = model.read_range(
        name,
        checked_range_add(
            stream_offset,
            source_byte,
            "streamed packed source offset"),
        source_bytes);
    record_mfe_read(source_bytes, started);
    const auto begin = target.size();
    const auto bytes = checked_packed_size(value_count, bits, "streamed packed slice");
    target.resize(checked_add(begin, bytes, "streamed packed append"));
    for (std::size_t index = 0; index < bytes; ++index) {
        const auto upper = index + 1 < raw.size() ? std::uint32_t{raw[index + 1]} : 0u;
        target[begin + index] = static_cast<std::uint8_t>(
            (std::uint32_t{raw[index]} | (upper << 8)) >> shift);
    }
    if ((bit_count & 7u) != 0u)
        target.back() &= static_cast<std::uint8_t>((1u << (bit_count & 7u)) - 1u);
}

template <std::size_t Selectors>
std::vector<std::array<std::uint32_t, Selectors>> selector_expert_prefixes(
    std::span<const std::uint8_t> selectors,
    int rows_per_expert) {
    if (rows_per_expert <= 0 || selectors.size() % rows_per_expert != 0) {
        throw std::runtime_error("invalid streamed NINT expert selector geometry");
    }
    std::vector<std::array<std::uint32_t, Selectors>> result(
        selectors.size() / rows_per_expert + 1);
    std::array<std::uint32_t, Selectors> counts{};
    for (std::size_t row = 0; row < selectors.size(); ++row) {
        if (selectors[row] >= Selectors) {
            throw std::runtime_error("invalid streamed NINT selector");
        }
        ++counts[selectors[row]];
        if ((row + 1) % rows_per_expert == 0) {
            result[(row + 1) / rows_per_expert] = counts;
        }
    }
    return result;
}

struct MfeNintStreamLayout {
    int rows = 0;
    int groups = 0;
    int group_size = 0;
    int q_bits = 0;
    int nominal_sub_bits = 0;
    bool adaptive_storage = false;
    bool legacy_unpacked = false;
    std::uint64_t payload_offset = 0;
    std::uint64_t anchor_scale_offset = 0;
    std::uint64_t anchor_min_offset = 0;
    std::uint64_t legacy_sub_scale_offset = 0;
    std::uint64_t legacy_sub_min_offset = 0;
    std::uint64_t legacy_q_offset = 0;
    std::uint64_t k_selector_offset = 0;
    std::array<std::uint64_t, 4> sub_scale_offsets{};
    std::array<std::uint64_t, 4> sub_min_offsets{};
    std::uint64_t q_selector_offset = 0;
    std::array<std::uint64_t, 8> q_offsets{};
    std::vector<std::uint8_t> k_selectors;
    std::vector<std::uint8_t> q_selectors;
    std::vector<std::uint8_t> header;
    std::vector<std::array<std::uint32_t, 4>> k_expert_prefixes;
    std::vector<std::array<std::uint32_t, 8>> q_expert_prefixes;
};

struct MfeNvqJscStreamLayout {
    int rows = 0;
    int groups = 0;
    int vectors = 0;
    int signs = 0;
    int state_bits = 0;
    int index_bits = 0;
    int auxiliary_bits = 7;
    bool group64 = false;
    int decoder_profile = kVqProfileGeneric;
    int execution_layout = 0;
    std::uint64_t payload_offset = 0;
    std::uint64_t prefix_bytes = 0;
    std::uint64_t anchors_offset = 0;
    std::uint64_t state_offset = 0;
    std::uint64_t indices_offset = 0;
    std::uint64_t signs_offset = 0;
    std::vector<std::uint8_t> prefix;
};

struct MfeMxStreamLayout {
    int rows = 0;
    int columns = 0;
    int bits = 0;
    std::uint64_t payload_offset = 0;
    std::uint64_t values_offset = 0;
    std::uint64_t scales_offset = 0;
    std::vector<std::uint8_t> header;
};

using MfeStreamLayout = std::variant<
    MfeNintStreamLayout,
    MfeNvqJscStreamLayout,
    MfeMxStreamLayout>;

struct MfeStreamPool {
    std::string dtype;
    int expert_count = 0;
    std::uint64_t runtime_bytes = 0;
    std::uint64_t payload_offset = 0;
    std::uint64_t payload_bytes = 0;
    MfeStreamLayout layout;
    std::vector<std::uint8_t> runtime;
};

struct MfeStreamExpertLocation {
    std::shared_ptr<const MfeStreamPool> pool;
    int local_expert = 0;
};

struct MfeStreamProjection {
    int experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    std::vector<std::optional<MfeStreamExpertLocation>> experts_by_id;
    std::shared_ptr<const MlxMfeKernelProfile> kernel_profile;
};

std::size_t streamed_expert_storage_bound(
    const MfeStreamProjection& projection, const MfeStreamExpertLocation& location) {
    const auto rows = static_cast<std::size_t>(projection.out_per_expert);
    const auto packed = [&](std::size_t count, int bits) {
        return checked_packed_size(count, bits, "streamed expert working storage");
    };
    const auto product = [&](std::size_t first, std::size_t second) {
        return checked_product(first, second, "streamed expert working storage");
    };
    const auto add = [](std::size_t first, std::size_t second) {
        return checked_add(first, second, "streamed expert working storage");
    };
    auto bytes = std::visit([&](const auto& layout) -> std::size_t {
        using Layout = std::decay_t<decltype(layout)>;
        if constexpr (std::is_same_v<Layout, MfeNintStreamLayout>) {
            const auto groups = static_cast<std::size_t>(layout.groups);
            const auto width = product(groups, layout.group_size);
            auto result = add(layout.header.size(), product(rows, 4));
            if (layout.adaptive_storage) {
                result = add(result, add(packed(rows, 2), packed(rows, 3)));
                for (int index = 0; index < 4; ++index) {
                    const auto count = layout.k_expert_prefixes[location.local_expert + 1][index]
                        - layout.k_expert_prefixes[location.local_expert][index];
                    result = add(result, product(2, packed(product(count, groups),
                        layout.nominal_sub_bits - 1 + index)));
                }
                for (int index = 0; index < 8; ++index) {
                    const auto count = layout.q_expert_prefixes[location.local_expert + 1][index]
                        - layout.q_expert_prefixes[location.local_expert][index];
                    result = add(result, packed(product(count, width), index + 1));
                }
            } else {
                const auto metadata = product(rows, groups);
                const auto values = product(rows, width);
                result = add(result, layout.legacy_unpacked
                    ? add(product(metadata, 2), values)
                    : add(product(2, packed(metadata, layout.nominal_sub_bits)), packed(values, layout.q_bits)));
            }
            // Four uint32 row descriptors and per-row byte/alignment tails.
            return add(result, product(rows, 32));
        } else if constexpr (std::is_same_v<Layout, MfeNvqJscStreamLayout>) {
            auto result = add(layout.prefix_bytes, product(rows, 2));
            if (layout.group64) result = add(result, product(product(rows, layout.groups), 8));
            else {
                result = add(result, packed(product(rows, layout.groups), layout.state_bits));
                result = add(result, packed(product(rows, layout.vectors), layout.index_bits));
                result = add(result, packed(product(rows, layout.signs), layout.auxiliary_bits));
            }
            // Packed indices/states remain packed; anchors and small lookup
            // tables may widen. NVQ1-L's built-in 2048x8 table has no wire bytes.
            return add(add(result, product(rows, 2)), add(
                std::max<std::size_t>(16384, product(layout.prefix_bytes, 2)), 1024));
        } else {
            const auto values = product(rows, layout.bits == 4 ? layout.columns / 2 : layout.columns);
            const auto scales = layout.bits == 4 ? product(rows, layout.columns / 32)
                : product((rows + 127) / 128, layout.columns / 128);
            return add(layout.header.size(), add(values, scales));
        }
    }, location.pool->layout);
    bytes = add(bytes, product(location.pool->runtime_bytes, 2));
    return add(bytes, sizeof(std::int32_t) * kDescriptorSize + 21 * 15
        + static_cast<std::size_t>(getpagesize()));
}

MfeNintStreamLayout parse_streamed_nint_layout(
    const MfeRecord& model,
    const std::string& name,
    std::uint64_t payload_offset,
    std::uint64_t payload_bytes,
    int expected_rows,
    int expected_columns,
    int rows_per_expert,
    int expected_experts = 0) {
    constexpr std::uint64_t kHeaderBytes = 42;
    if (payload_bytes < kHeaderBytes) {
        throw std::runtime_error("truncated streamed NINT payload: " + name);
    }
    auto header = model.read_range(
        name, payload_offset, kHeaderBytes);
    const int raw_bits = header[0];
    const int nominal_sub_bits = header[1];
    const auto group_size = read_scalar<std::int32_t>(
        header, 2, "NINT group size");
    const auto axis = read_scalar<std::int32_t>(
        header, 6, "NINT axis");
    const auto columns = read_scalar<std::int32_t>(
        header, 10, "NINT width");
    const auto dimensions = read_scalar<std::uint32_t>(
        header, 14, "NINT dimensions");
    if (dimensions == 3 && expected_experts > 0) {
        if (payload_bytes < kHeaderBytes + 8)
            throw std::runtime_error("truncated streamed NINT expert header: " + name);
        header = model.read_range(name, payload_offset, kHeaderBytes + 8);
    } else if (dimensions != 2 || expected_experts > 0) {
        throw std::runtime_error("unsupported streamed NINT rank: " + name);
    }
    const auto shape_offset = dimensions == 3 ? 26u : 18u;
    const auto shape_rows = read_scalar<std::int64_t>(header, shape_offset, "NINT rows");
    const auto shape_columns = read_scalar<std::int64_t>(
        header, shape_offset + 8, "NINT columns");
    const auto rows = read_scalar<std::uint32_t>(
        header, shape_offset + 16, "NINT output size");
    const auto groups = read_scalar<std::uint32_t>(
        header, shape_offset + 20, "NINT group count");
    const bool shape_valid = dimensions == 3
        ? read_scalar<std::int64_t>(header, 18, "NINT experts") == expected_experts
            && shape_rows == rows_per_expert
            && std::int64_t(expected_experts) * rows_per_expert == expected_rows
        : shape_rows == expected_rows;
    const bool adaptive_storage = (raw_bits & 0x80) != 0;
    const int q_bits = raw_bits & 0x7f;
    if (q_bits < 1
        || q_bits > 8 || nominal_sub_bits < 1
        || nominal_sub_bits > 8 || group_size <= 0 || axis != 0
        || !shape_valid || columns != expected_columns
        || shape_columns != expected_columns
        || rows != static_cast<std::uint32_t>(expected_rows)
        || groups != static_cast<std::uint32_t>(
            (expected_columns - 1) / group_size + 1)) {
        throw std::runtime_error(
            "unsupported streamed NINTv2 geometry: " + name);
    }

    MfeNintStreamLayout result;
    result.rows = expected_rows;
    result.groups = static_cast<int>(groups);
    result.group_size = group_size;
    result.q_bits = q_bits;
    result.nominal_sub_bits = nominal_sub_bits;
    result.adaptive_storage = adaptive_storage;
    result.header.assign(header.begin(), header.begin() + 18);
    patch_scalar_bytes<std::uint32_t>(result.header, 14, 2, "NINT matrix rank");
    append_scalar_bytes<std::int64_t>(result.header, expected_rows);
    append_scalar_bytes<std::int64_t>(result.header, expected_columns);
    append_scalar_bytes<std::uint32_t>(result.header, rows);
    append_scalar_bytes<std::uint32_t>(result.header, groups);
    result.payload_offset = payload_offset;
    result.anchor_scale_offset = payload_offset + header.size();
    result.anchor_min_offset = result.anchor_scale_offset
        + static_cast<std::uint64_t>(expected_rows) * 2;
    std::uint64_t cursor = result.anchor_min_offset
        + static_cast<std::uint64_t>(expected_rows) * 2;
    if (!adaptive_storage) {
        const auto metadata_count = checked_product(
            static_cast<std::size_t>(expected_rows),
            static_cast<std::size_t>(groups),
            "legacy NINT metadata count");
        const auto values_per_row = checked_product(
            static_cast<std::size_t>(groups),
            static_cast<std::size_t>(group_size),
            "legacy NINT values per row");
        const auto q_count = checked_product(
            static_cast<std::size_t>(expected_rows),
            values_per_row,
            "legacy NINT q count");
        const auto packed_metadata_bytes = checked_packed_size(
            metadata_count, nominal_sub_bits,
            "legacy NINT metadata bytes");
        const auto packed_q_bytes = checked_packed_size(
            q_count, q_bits, "legacy NINT q bytes");
        const auto packed_tail = checked_add(
            checked_product(
                packed_metadata_bytes, std::size_t{2},
                "legacy NINT packed metadata"),
            packed_q_bytes,
            "legacy NINT packed tail");
        const auto unpacked_tail = checked_add(
            checked_product(
                metadata_count, std::size_t{2},
                "legacy NINT unpacked metadata"),
            q_count,
            "legacy NINT unpacked tail");
        const auto payload_end = checked_range_add(
            payload_offset, payload_bytes, "legacy NINT payload end");
        if (cursor > payload_end) {
            throw std::runtime_error(
                "truncated streamed legacy NINT payload: " + name);
        }
        const auto remaining = payload_end - cursor;
        if (remaining == unpacked_tail) {
            result.legacy_unpacked = true;
            result.legacy_sub_scale_offset = cursor;
            result.legacy_sub_min_offset = cursor + metadata_count;
            result.legacy_q_offset =
                result.legacy_sub_min_offset + metadata_count;
        } else if (remaining == packed_tail) {
            result.legacy_sub_scale_offset = cursor;
            result.legacy_sub_min_offset = cursor + packed_metadata_bytes;
            result.legacy_q_offset =
                result.legacy_sub_min_offset + packed_metadata_bytes;
        } else {
            throw std::runtime_error(
                "invalid streamed legacy NINT payload length: " + name);
        }
        return result;
    }
    const auto k_selector_bytes = checked_packed_size(
        static_cast<std::size_t>(expected_rows), 2,
        "NINTv2 k selectors");
    result.k_selector_offset = cursor;
    result.k_selectors = unpack_small_selectors(
        model.read_range(name, cursor, k_selector_bytes),
        static_cast<std::size_t>(expected_rows),
        2,
        "NINTv2 k selector");
    result.k_expert_prefixes = selector_expert_prefixes<4>(
        result.k_selectors, rows_per_expert);
    cursor += k_selector_bytes;
    for (int selector = 0; selector < 4; ++selector) {
        const int row_bits = nominal_sub_bits - 1 + selector;
        const auto selected = result.k_expert_prefixes.back()[selector];
        if (selected != 0 && (row_bits < 1 || row_bits > 8)) {
            throw std::runtime_error(
                "invalid streamed NINTv2 subgroup width: " + name);
        }
        const auto bytes = checked_packed_size(
            checked_product(selected, static_cast<std::size_t>(groups),
                "NINTv2 subgroup values"),
            std::max(row_bits, 1),
            "NINTv2 subgroup bytes");
        result.sub_scale_offsets[selector] = cursor;
        cursor += bytes;
        result.sub_min_offsets[selector] = cursor;
        cursor += bytes;
    }
    const auto q_selector_bytes = checked_packed_size(
        static_cast<std::size_t>(expected_rows), 3,
        "NINTv2 q selectors");
    result.q_selector_offset = cursor;
    result.q_selectors = unpack_small_selectors(
        model.read_range(name, cursor, q_selector_bytes),
        static_cast<std::size_t>(expected_rows),
        3,
        "NINTv2 q selector");
    result.q_expert_prefixes = selector_expert_prefixes<8>(
        result.q_selectors, rows_per_expert);
    cursor += q_selector_bytes;
    const auto values_per_row = checked_product(
        static_cast<std::size_t>(groups),
        static_cast<std::size_t>(group_size),
        "NINTv2 values per row");
    for (int selector = 0; selector < 8; ++selector) {
        const int row_bits = selector + 1;
        const auto selected = result.q_expert_prefixes.back()[selector];
        result.q_offsets[selector] = cursor;
        cursor += checked_packed_size(
            checked_product(selected, values_per_row,
                "NINTv2 selected values"),
            row_bits,
            "NINTv2 q bytes");
    }
    if (cursor != payload_offset + payload_bytes) {
        throw std::runtime_error(
            "invalid streamed NINTv2 payload length: " + name);
    }
    return result;
}

MfeNvqJscStreamLayout parse_streamed_nvq_jsc_layout(
    const MfeRecord& model,
    const std::string& name,
    std::uint64_t payload_offset,
    std::uint64_t payload_bytes,
    int expected_rows,
    int expected_columns) {
    constexpr std::uint64_t kMatrixHeaderBytes = 40;
    constexpr std::uint64_t kJscHeaderBytes = 64;
    if (payload_bytes < kMatrixHeaderBytes + kJscHeaderBytes) {
        throw std::runtime_error("truncated streamed NVQ-JSC payload: " + name);
    }
    const auto header = model.read_range(
        name, payload_offset, kMatrixHeaderBytes + kJscHeaderBytes);
    const std::string_view magic(
        reinterpret_cast<const char*>(header.data()), 4);
    const int profile_flags = header[4];
    const int profile = profile_flags & ~(0x80 | 0x40 | 0x20);
    const int state_bits = header[5];
    const auto group_size = read_scalar<std::uint16_t>(
        header, 6, "NVQ group size");
    const auto axis = read_scalar<std::int32_t>(
        header, 8, "NVQ axis");
    const auto columns = read_scalar<std::int32_t>(
        header, 12, "NVQ width");
    const auto dimensions = read_scalar<std::uint32_t>(
        header, 16, "NVQ dimensions");
    const auto shape_rows = read_scalar<std::int64_t>(
        header, 20, "NVQ rows");
    const auto shape_columns = read_scalar<std::int64_t>(
        header, 28, "NVQ columns");
    const auto rows = read_scalar<std::uint32_t>(
        header, 36, "NVQ output size");
    const int vector_size = profile == 1 || profile == 4 || profile == 5
        ? 8 : profile == 2 || profile == 3 || profile == 6 ? 4 : 0;
    const int index_bits = profile == 1 || profile == 2 ? 8
        : profile == 3 ? 9
        : profile == 4 || profile == 6 ? 10
        : profile == 5 ? 12 : 0;
    const int banks = header[kMatrixHeaderBytes + 1];
    const int states = header[kMatrixHeaderBytes + 2];
    const int storage_layout = header[kMatrixHeaderBytes + 52];
    if ((magic != "NVQ1" && magic != "NIQ1")
        || (profile_flags & 0x20) == 0
        || (profile_flags & (0x80 | 0x40)) != 0
        || state_bits != 4 || group_size != 24 || axis != 0
        || dimensions != 2 || columns != expected_columns
        || shape_rows != expected_rows || shape_columns != expected_columns
        || rows != static_cast<std::uint32_t>(expected_rows)
        || vector_size == 0 || index_bits == 0
        || (banks != 1 && banks != 2 && banks != 4)
        || states != 16 || (storage_layout != 0 && storage_layout != 1)) {
        throw MfeStreamUnsupported(
            "MFE VQ cohort is not a streamable NVQ-JSC layout");
    }
    const auto entries = std::uint64_t{1} << index_bits;
    const auto codebook_bytes = checked_range_product(
        checked_range_product(
            static_cast<std::uint64_t>(banks), entries,
            "NVQ-JSC codebook entries"),
        static_cast<std::uint64_t>(vector_size),
        "NVQ-JSC codebook bytes");
    MfeNvqJscStreamLayout result;
    result.rows = expected_rows;
    result.groups = (expected_columns + group_size - 1) / group_size;
    result.vectors = (expected_columns + vector_size - 1) / vector_size;
    result.signs = (expected_columns + 7) / 8;
    result.state_bits = state_bits;
    result.index_bits = index_bits;
    result.group64 = storage_layout == 1;
    result.decoder_profile = index_bits > 8
        ? (vector_size == 4 ? kVqProfileJscExtended4 : kVqProfileJscExtended8)
        : (vector_size == 4 ? kVqProfileJsc4 : kVqProfileJsc8);
    result.execution_layout = profile == 5 && result.group64 ? 2 : 0;
    result.payload_offset = payload_offset;
    result.prefix_bytes = checked_range_add(
        kMatrixHeaderBytes + kJscHeaderBytes,
        codebook_bytes,
        "NVQ-JSC prefix bytes");
    result.anchors_offset = payload_offset + result.prefix_bytes;
    result.state_offset = result.anchors_offset
        + static_cast<std::uint64_t>(expected_rows) * 2;
    if (result.group64) {
        result.indices_offset = result.state_offset;
        const auto records = checked_range_product(
            static_cast<std::uint64_t>(expected_rows),
            static_cast<std::uint64_t>(result.groups),
            "NVQ-JSC group64 records");
        result.signs_offset = checked_range_add(
            result.indices_offset,
            checked_range_product(records, 8, "NVQ-JSC group64 bytes"),
            "NVQ-JSC group64 end");
    } else {
        const auto state_bytes = checked_packed_size(
            checked_product(
                static_cast<std::size_t>(expected_rows),
                static_cast<std::size_t>(result.groups),
                "NVQ-JSC state count"),
            state_bits,
            "NVQ-JSC state bytes");
        result.indices_offset = result.state_offset + state_bytes;
        const auto index_bytes = checked_packed_size(
            checked_product(
                static_cast<std::size_t>(expected_rows),
                static_cast<std::size_t>(result.vectors),
                "NVQ-JSC index count"),
            index_bits,
            "NVQ-JSC index bytes");
        result.signs_offset = result.indices_offset + index_bytes;
    }
    const auto end = result.group64
        ? result.signs_offset
        : result.signs_offset + checked_packed_size(
              checked_product(
                  static_cast<std::size_t>(expected_rows),
                  static_cast<std::size_t>(result.signs),
                  "NVQ-JSC sign count"),
              7,
              "NVQ-JSC sign bytes");
    if (end != payload_offset + payload_bytes) {
        throw std::runtime_error(
            "invalid streamed NVQ-JSC payload length: " + name);
    }
    result.prefix = model.read_range(name, payload_offset, result.prefix_bytes);
    return result;
}

// NVQ1-S/L have the same row-major anchor/state/index/auxiliary structure as
// planar JSC, but use a one-bit delta per group instead of seven-bit signs.
// Keep the codebook prefix intact and slice packed streams at bit boundaries.
MfeNvqJscStreamLayout parse_streamed_nvq1_layout(
    const MfeRecord& model,
    const std::string& name,
    std::uint64_t payload_offset,
    std::uint64_t payload_bytes,
    int expected_rows,
    int expected_columns) {
    constexpr std::size_t header_bytes = 40;
    if (payload_bytes < header_bytes) {
        throw std::runtime_error("truncated streamed NVQ1 header: " + name);
    }
    const auto header = model.read_range(name, payload_offset, header_bytes);
    const bool short_profile = std::memcmp(header.data(), "NQ1S", 4) == 0;
    const bool long_profile = std::memcmp(header.data(), "NQ1L", 4) == 0;
    const int profile = header[4];
    const int state_bits = header[5];
    const auto group_size = read_scalar<std::uint16_t>(header, 6, "NVQ1 group size");
    if ((!short_profile && !long_profile) ||
        (short_profile && (profile != 1 || state_bits != 4 ||
                           group_size != 24 || expected_columns % 8 != 0)) ||
        (long_profile && ((profile != 1 && profile != 2) ||
                          state_bits < 1 || state_bits > 8 ||
                          group_size == 0 || group_size % 8 != 0)) ||
        read_scalar<std::int32_t>(header, 8, "NVQ1 axis") != 0 ||
        read_scalar<std::int32_t>(header, 12, "NVQ1 width") != expected_columns ||
        read_scalar<std::uint32_t>(header, 16, "NVQ1 dimensions") != 2 ||
        read_scalar<std::int64_t>(header, 20, "NVQ1 rows") != expected_rows ||
        read_scalar<std::int64_t>(header, 28, "NVQ1 columns") != expected_columns ||
        read_scalar<std::uint32_t>(header, 36, "NVQ1 output size") !=
            static_cast<std::uint32_t>(expected_rows)) {
        throw std::runtime_error("invalid streamed NVQ1 geometry: " + name);
    }
    MfeNvqJscStreamLayout result;
    result.rows = expected_rows;
    result.groups = (expected_columns - 1) / group_size + 1;
    result.vectors = (expected_columns - 1) / 8 + 1;
    result.signs = result.groups;
    result.state_bits = state_bits;
    result.index_bits = short_profile ? 9 : 11;
    result.auxiliary_bits = 1;
    if (detail::vq_native_nvq1(0, group_size, 8,
            short_profile ? "NVQ1-S" : "NVQ1-L", result.index_bits, state_bits,
            short_profile ? 2 : 0, 3, 1))
        result.decoder_profile = short_profile ? kVqProfileNvq1S : kVqProfileNvq1L;
    result.payload_offset = payload_offset;
    result.prefix_bytes = header_bytes + (short_profile ? 2048 : profile == 2 ? 4096 : 0);
    auto cursor = checked_range_add(payload_offset, result.prefix_bytes, "NVQ1 prefix");
    result.anchors_offset = cursor;
    cursor = checked_range_add(cursor, checked_range_product(
        expected_rows, 2, "NVQ1 anchors"), "NVQ1 anchors end");
    const auto advance = [&](int values_per_row, int bits) {
        const auto offset = cursor;
        cursor = checked_range_add(cursor, checked_packed_size(
            checked_product(static_cast<std::size_t>(expected_rows),
                static_cast<std::size_t>(values_per_row), "NVQ1 stream values"),
            bits, "NVQ1 packed stream"), "NVQ1 stream end");
        return offset;
    };
    result.state_offset = advance(result.groups, state_bits);
    result.indices_offset = advance(result.vectors, result.index_bits);
    result.signs_offset = advance(result.groups, 1);
    if (cursor != checked_range_add(payload_offset, payload_bytes, "NVQ1 payload end")) {
        throw std::runtime_error("invalid streamed NVQ1 payload length: " + name);
    }
    result.prefix = model.read_range(name, payload_offset, result.prefix_bytes);
    return result;
}

MfeMxStreamLayout parse_streamed_mx_layout(
    const MfeRecord& model,
    const std::string& name,
    std::uint64_t payload_offset,
    std::uint64_t payload_bytes,
    int expected_rows,
    int expected_columns,
    int expected_bits) {
    constexpr std::uint64_t kHeaderBytes = 56;
    if (payload_bytes < kHeaderBytes) {
        throw std::runtime_error("truncated streamed MX payload: " + name);
    }
    const auto header = model.read_range(
        name, payload_offset, kHeaderBytes);
    const std::string_view magic(
        reinterpret_cast<const char*>(header.data()), 4);
    const int version = header[4];
    const int bits = header[5];
    const auto reserved = read_scalar<std::uint16_t>(
        header, 6, "MX reserved");
    const auto rows = read_scalar<std::uint64_t>(header, 8, "MX rows");
    const auto columns = read_scalar<std::uint64_t>(header, 16, "MX columns");
    const auto storage_rows = read_scalar<std::uint64_t>(
        header, 24, "MX storage rows");
    const auto storage_columns = read_scalar<std::uint64_t>(
        header, 32, "MX storage columns");
    const auto scale_rows = read_scalar<std::uint64_t>(
        header, 40, "MX scale rows");
    const auto scale_columns = read_scalar<std::uint64_t>(
        header, 48, "MX scale columns");
    const auto expected_storage_columns = expected_bits == 4
        ? static_cast<std::uint64_t>(expected_columns / 2)
        : static_cast<std::uint64_t>(expected_columns);
    const auto expected_scale_rows = expected_bits == 4
        ? static_cast<std::uint64_t>(expected_rows)
        : static_cast<std::uint64_t>((expected_rows + 127) / 128);
    const auto expected_scale_columns = expected_bits == 4
        ? static_cast<std::uint64_t>(expected_columns / 32)
        : static_cast<std::uint64_t>(expected_columns / 128);
    if (magic != "MXT1" || version != 1 || bits != expected_bits
        || reserved != 0 || rows != static_cast<std::uint64_t>(expected_rows)
        || columns != static_cast<std::uint64_t>(expected_columns)
        || storage_rows != rows || storage_columns != expected_storage_columns
        || scale_rows != expected_scale_rows
        || scale_columns != expected_scale_columns) {
        throw std::runtime_error(
            "unsupported streamed MX geometry: " + name);
    }
    const auto value_bytes = checked_range_product(
        storage_rows, storage_columns, "MX value bytes");
    const auto scale_bytes = checked_range_product(
        scale_rows, scale_columns, "MX scale bytes");
    if (kHeaderBytes + value_bytes + scale_bytes != payload_bytes) {
        throw std::runtime_error(
            "invalid streamed MX payload length: " + name);
    }
    return {
        expected_rows,
        expected_columns,
        expected_bits,
        payload_offset,
        payload_offset + kHeaderBytes,
        payload_offset + kHeaderBytes + value_bytes,
        header,
    };
}

template <typename Bytes>
void append_streamed_nint_expert(
    Bytes& result,
    const MfeRecord& model,
    const std::string& name,
    const MfeNintStreamLayout& layout,
    int local_expert,
    int rows_per_expert) {
    const auto begin = checked_product(
        static_cast<std::size_t>(local_expert),
        static_cast<std::size_t>(rows_per_expert),
        "NINT expert row offset");
    const auto end = checked_add(
        begin,
        static_cast<std::size_t>(rows_per_expert),
        "NINT expert row end");
    if (end > static_cast<std::size_t>(layout.rows)) {
        throw std::out_of_range("streamed NINT expert is out of range");
    }
    const auto payload_begin = result.size();
    result.insert(result.end(), layout.header.begin(), layout.header.end());
    patch_scalar_bytes<std::int64_t>(
        result, payload_begin + 18, rows_per_expert, "NINTv2 shape row");
    patch_scalar_bytes<std::uint32_t>(
        result, payload_begin + 34, static_cast<std::uint32_t>(rows_per_expert),
        "NINTv2 output size");
    append_record_range(
        result, model, name,
        layout.anchor_scale_offset + begin * 2,
        static_cast<std::uint64_t>(rows_per_expert) * 2);
    append_record_range(
        result, model, name,
        layout.anchor_min_offset + begin * 2,
        static_cast<std::uint64_t>(rows_per_expert) * 2);
    if (!layout.adaptive_storage) {
        const auto metadata_begin = checked_product(
            begin,
            static_cast<std::size_t>(layout.groups),
            "legacy NINT metadata prefix");
        const auto metadata_count = checked_product(
            static_cast<std::size_t>(rows_per_expert),
            static_cast<std::size_t>(layout.groups),
            "legacy NINT metadata slice");
        const auto append_legacy = [&](std::uint64_t offset,
                                       std::size_t first,
                                       std::size_t count,
                                       int bits) {
            if (layout.legacy_unpacked) {
                append_record_range(
                    result, model, name, offset + first, count);
            } else {
                append_packed_value_slice(result, model, name, offset, first, count, bits);
            }
        };
        append_legacy(
            layout.legacy_sub_scale_offset,
            metadata_begin,
            metadata_count,
            layout.nominal_sub_bits);
        append_legacy(
            layout.legacy_sub_min_offset,
            metadata_begin,
            metadata_count,
            layout.nominal_sub_bits);
        const auto values_per_row = checked_product(
            static_cast<std::size_t>(layout.groups),
            static_cast<std::size_t>(layout.group_size),
            "legacy NINT values per row");
        append_legacy(
            layout.legacy_q_offset,
            checked_product(begin, values_per_row,
                "legacy NINT q prefix"),
            checked_product(
                static_cast<std::size_t>(rows_per_expert),
                values_per_row,
                "legacy NINT q slice"),
            layout.q_bits);
        return;
    }
    const auto k_slice = std::span<const std::uint8_t>(
        layout.k_selectors).subspan(begin, rows_per_expert);
    const auto packed_k = pack_small_selectors(k_slice, 2);
    result.insert(result.end(), packed_k.begin(), packed_k.end());
    for (int selector = 0; selector < 4; ++selector) {
        const auto selected_before = layout.k_expert_prefixes[local_expert][selector];
        const auto selected_here = layout.k_expert_prefixes[local_expert + 1][selector] - selected_before;
        const int bits = layout.nominal_sub_bits - 1 + selector;
        for (const auto offset : {
                 layout.sub_scale_offsets[selector],
                 layout.sub_min_offsets[selector]}) {
            append_packed_value_slice(
                result, model, name, offset,
                checked_product(selected_before,
                    static_cast<std::size_t>(layout.groups),
                    "NINTv2 subgroup prefix"),
                checked_product(selected_here,
                    static_cast<std::size_t>(layout.groups),
                    "NINTv2 subgroup slice"),
                bits);
        }
    }
    const auto q_slice = std::span<const std::uint8_t>(
        layout.q_selectors).subspan(begin, rows_per_expert);
    const auto packed_q_selectors = pack_small_selectors(q_slice, 3);
    result.insert(
        result.end(),
        packed_q_selectors.begin(),
        packed_q_selectors.end());
    const auto values_per_row = checked_product(
        static_cast<std::size_t>(layout.groups),
        static_cast<std::size_t>(layout.group_size),
        "NINTv2 values per row");
    for (int selector = 0; selector < 8; ++selector) {
        const auto selected_before = layout.q_expert_prefixes[local_expert][selector];
        const auto selected_here = layout.q_expert_prefixes[local_expert + 1][selector] - selected_before;
        append_packed_value_slice(
            result, model, name, layout.q_offsets[selector],
            checked_product(selected_before, values_per_row,
                "NINTv2 q prefix"),
            checked_product(selected_here, values_per_row,
                "NINTv2 q slice"),
            selector + 1);
    }
}

template <typename Bytes>
void append_streamed_nvq_expert(
    Bytes& result,
    const MfeRecord& model,
    const std::string& name,
    const MfeNvqJscStreamLayout& layout,
    int local_expert,
    int rows_per_expert) {
    const auto begin = checked_product(
        static_cast<std::size_t>(local_expert),
        static_cast<std::size_t>(rows_per_expert),
        "NVQ-JSC expert row offset");
    const auto end = checked_add(
        begin,
        static_cast<std::size_t>(rows_per_expert),
        "NVQ-JSC expert row end");
    if (end > static_cast<std::size_t>(layout.rows)) {
        throw std::out_of_range("streamed NVQ-JSC expert is out of range");
    }
    const auto payload_begin = result.size();
    result.insert(result.end(), layout.prefix.begin(), layout.prefix.end());
    patch_scalar_bytes<std::int64_t>(
        result, payload_begin + 20, rows_per_expert, "NVQ-JSC shape row");
    patch_scalar_bytes<std::uint32_t>(
        result, payload_begin + 36, static_cast<std::uint32_t>(rows_per_expert),
        "NVQ-JSC output size");
    {
        MfeReadPlanScope immediate(direct_read_metadata != MfeMetadataRead::Planned ? nullptr : current_mfe_plan);
        append_record_range(
            result, model, name,
            layout.anchors_offset + begin * 2,
            static_cast<std::uint64_t>(rows_per_expert) * 2);
    }
    if (layout.group64) {
        append_record_range(
            result, model, name,
            layout.indices_offset
                + begin * static_cast<std::size_t>(layout.groups) * 8,
            static_cast<std::uint64_t>(rows_per_expert)
                * static_cast<std::uint64_t>(layout.groups) * 8);
        return;
    }
    const auto append_bits = [&](std::uint64_t offset,
                                 std::size_t values_per_row,
                                 int bits) {
        append_packed_value_slice(
            result, model, name, offset,
            checked_product(begin, values_per_row,
                "NVQ-JSC stream prefix"),
            checked_product(
                static_cast<std::size_t>(rows_per_expert),
                values_per_row,
                "NVQ-JSC stream slice"),
            bits);
    };
    append_bits(layout.state_offset, layout.groups, layout.state_bits);
    append_bits(layout.indices_offset, layout.vectors, layout.index_bits);
    append_bits(layout.signs_offset, layout.signs, layout.auxiliary_bits);
}

template <typename Bytes>
void append_streamed_mx_expert(
    Bytes& result,
    const MfeRecord& model,
    const std::string& name,
    const MfeMxStreamLayout& layout,
    int local_expert,
    int rows_per_expert) {
    const auto begin = checked_product(
        static_cast<std::size_t>(local_expert),
        static_cast<std::size_t>(rows_per_expert),
        "MX expert row offset");
    const auto end = checked_add(
        begin,
        static_cast<std::size_t>(rows_per_expert),
        "MX expert row end");
    if (end > static_cast<std::size_t>(layout.rows)) {
        throw std::out_of_range("streamed MX expert is out of range");
    }
    const auto payload_begin = result.size();
    result.insert(result.end(), layout.header.begin(), layout.header.end());
    patch_scalar_bytes<std::uint64_t>(
        result, payload_begin + 8, rows_per_expert, "MX logical rows");
    patch_scalar_bytes<std::uint64_t>(
        result, payload_begin + 24, rows_per_expert, "MX storage rows");
    const auto value_stride = static_cast<std::uint64_t>(
        layout.bits == 4 ? layout.columns / 2 : layout.columns);
    append_record_range(
        result, model, name,
        layout.values_offset + begin * value_stride,
        static_cast<std::uint64_t>(rows_per_expert) * value_stride);
    if (layout.bits == 4) {
        const auto scale_stride = static_cast<std::uint64_t>(
            layout.columns / 32);
        patch_scalar_bytes<std::uint64_t>(
            result, payload_begin + 40, rows_per_expert, "MXFP4 scale rows");
        const bool deferred = direct_read_metadata == MfeMetadataRead::DeferredMxScales;
        MfeReadPlanScope immediate(direct_read_metadata == MfeMetadataRead::Eager ? nullptr : current_mfe_plan);
        const auto scale_begin = result.size();
        append_record_range(
            result, model, name,
            layout.scales_offset + begin * scale_stride,
            static_cast<std::uint64_t>(rows_per_expert) * scale_stride);
        if (deferred) std::fill(result.begin() + scale_begin, result.end(), std::uint8_t{0});
    } else {
        if ((begin & 127u) != 0u) {
            throw MfeStreamUnsupported(
                "MXFP8 expert rows are not block-128 aligned");
        }
        const auto scale_rows =
            (static_cast<std::size_t>(rows_per_expert) + 127) / 128;
        const auto scale_stride = static_cast<std::uint64_t>(
            layout.columns / 128);
        patch_scalar_bytes<std::uint64_t>(
            result, payload_begin + 40, scale_rows, "MXFP8 scale rows");
        append_record_range(
            result, model, name,
            layout.scales_offset + (begin / 128) * scale_stride,
            static_cast<std::uint64_t>(scale_rows) * scale_stride);
    }
}

template <typename Bytes>
void append_streamed_mfe_expert_payload(
    Bytes& result,
    const MfeRecord& model,
    const std::string& name,
    const MfeStreamPool& pool,
    int local_expert,
    int rows_per_expert) {
    std::visit(
        [&](const auto& layout) {
            using Layout = std::decay_t<decltype(layout)>;
            if constexpr (std::is_same_v<Layout, MfeNintStreamLayout>) {
                append_streamed_nint_expert(
                    result, model, name, layout, local_expert, rows_per_expert);
            } else if constexpr (
                std::is_same_v<Layout, MfeNvqJscStreamLayout>) {
                append_streamed_nvq_expert(
                    result, model, name, layout, local_expert, rows_per_expert);
            } else {
                append_streamed_mx_expert(
                    result, model, name, layout, local_expert, rows_per_expert);
            }
        },
        pool.layout);
}

} // namespace

namespace {

struct MfeGroupWork {
    std::size_t count;
    std::function<void(std::int64_t, std::int64_t)> load;
};

struct MfeGroupTask {
    struct promise_type {
        std::optional<MfeGroupWork> work;
        std::optional<MlxMfeWeight> result;
        std::exception_ptr failure;
        MfeGroupTask get_return_object() {
            return MfeGroupTask{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(MfeGroupWork value) {
            work.emplace(std::move(value));
            return {};
        }
        void return_value(MlxMfeWeight value) { result.emplace(std::move(value)); }
        void unhandled_exception() { failure = std::current_exception(); }
    };
    std::coroutine_handle<promise_type> handle;
    explicit MfeGroupTask(std::coroutine_handle<promise_type> value) : handle(value) {}
    MfeGroupTask(MfeGroupTask&& other) noexcept : handle(std::exchange(other.handle, {})) {}
    MfeGroupTask(const MfeGroupTask&) = delete;
    ~MfeGroupTask() { if (handle) handle.destroy(); }
    void resume() {
        handle.promise().work.reset();
        handle.resume();
        if (handle.promise().failure) std::rethrow_exception(handle.promise().failure);
    }
};

}

struct MlxMfeOffloadCache::Impl {
    MfeGroupTask prepare_grouped(const std::string& name,
        const std::vector<std::int32_t>& active_experts,
        std::span<const std::uint32_t> activation_counts, bool prefill_scan);
    struct Key {
        std::string name;
        std::int32_t expert = 0;

        bool operator==(const Key& other) const noexcept {
            return expert == other.expert
                && name == other.name;
        }
    };

    struct KeyHash {
        std::size_t operator()(
            const Key& value) const noexcept {
            const auto first =
                std::hash<std::string>{}(value.name);
            const auto second =
                std::hash<std::int32_t>{}(
                    value.expert);
            return first
                ^ (
                    second
                    + std::size_t{
                        0x9e3779b97f4a7c15ull}
                    + (first << 6)
                    + (first >> 2)
                );
        }
    };

    struct MfeCachedExpert {
        MfeCachedExpert(
            std::int32_t global,
            MlxMfeWeight value,
            bool temporary = false)
            : expert(global),
              weight(std::move(value)),
              packed_nbytes(weight.packed_nbytes()),
              transient(temporary) {}

        std::int32_t expert = 0;
        MlxMfeWeight weight;
        std::size_t packed_nbytes = 0;
        bool transient = false;
    };

    struct MfeCacheValue {
        std::shared_ptr<const MfeCachedExpert> weight;
        std::uint8_t credit = 0;
        std::shared_ptr<void> pin;
        bool pin_attempted = false;
    };

    struct PreparedProjection {
        MlxMfeWeight weight;
        std::unordered_map<std::int32_t, std::int32_t> local_ids;
    };

    struct InFlight {
        bool complete = false;
        std::optional<PreparedProjection> result;
        std::vector<std::shared_ptr<const MfeCachedExpert>> pages;
    };

    struct PinRequest {
        Key key;
        std::weak_ptr<const MfeCachedExpert> page;
    };

    using MfeProjectionCache = std::unordered_map<
        std::string,
        std::shared_ptr<const MfeStreamProjection>>;

    Impl(
        const source::Source& source,
        std::size_t limit,
        int expert_count,
        std::function<void(const std::string&)> observer)
        : cache_limit(limit),
          experts(expert_count), on_prepared(std::move(observer)) {
        if (experts < 0) {
            throw std::invalid_argument(
                "MFE offload expert count cannot be negative");
        }
        auto serial = source.supports_parallel_tensor_reads() ? std::shared_ptr<std::mutex>{} : std::make_shared<std::mutex>();
        for (const auto& metadata : source.tensors()) {
            MfeRecord record{metadata, {}, {}, serial};
            if (metadata.dtype == "MFE" || is_nint_dtype(metadata.dtype)) {
                // Unsupported readers fail when that record is used.
                try { record.batch = source.tensor_batch_reader(metadata.name); }
                catch (const TensorReaderUnsupported&) {}
                try { record.reader = source.tensor_reader(metadata.name); }
                catch (const TensorReaderUnsupported&) {}
            }
            records.emplace(metadata.name, std::move(record));
        }
        prepare_packed_copy();
        resources.bind([this] {
            std::lock_guard lock(mutex);
            std::size_t payload = 0;
            for (const auto& item : backing_records) payload += item.second;
            return MlxResourceUsage{0, 0, resident_bytes, payload, 0};
        });
    }

    ~Impl() {
        {
            std::lock_guard lock(pin_mutex);
            pin_stopping = true;
            pin_queue.clear();
            pin_pending.clear();
        }
        pin_ready.notify_one();
        if (pin_worker.joinable()) pin_worker.join();
    }

    bool enqueue_pin(const Key& key, const std::shared_ptr<const MfeCachedExpert>& page) noexcept {
        if (page->packed_nbytes < (std::size_t{64} << 10)) return true;
        try {
            std::lock_guard lock(pin_mutex);
            if (pin_stopping) return false;
            if (pin_pending.contains(key)) return true;
            if (!pin_worker.joinable()) pin_worker = std::thread([this] { pin_loop(); });
            pin_queue.push_back({key, page});
            try {
                pin_pending.emplace(key, std::prev(pin_queue.end()));
            } catch (...) {
                pin_queue.pop_back();
                throw;
            }
            pin_ready.notify_one();
            return true;
        } catch (...) {
            return false;
        }
    }

    void cancel_pin(const Key& key) {
        std::lock_guard lock(pin_mutex);
        if (const auto found = pin_pending.find(key); found != pin_pending.end()) {
            pin_queue.erase(found->second);
            pin_pending.erase(found);
        }
    }

    void pin_loop() {
        for (;;) {
            PinRequest request;
            {
                std::unique_lock lock(pin_mutex);
                pin_ready.wait(lock, [&] { return pin_stopping || !pin_queue.empty(); });
                if (pin_stopping) return;
                request = std::move(pin_queue.front());
                pin_queue.pop_front();
                pin_pending.erase(request.key);
            }
            std::shared_ptr<const MfeCachedExpert> page;
            {
                std::lock_guard lock(mutex);
                const auto cached = mfe_cache.find(request.key);
                page = request.page.lock();
                if (!page || !cached || cached->weight != page) continue;
            }
            auto pin = page->weight.pin_packed_storage();
            {
                std::lock_guard lock(mutex);
                const auto cached = mfe_cache.find(request.key);
                if (cached && cached->weight == page)
                    cached->pin = std::move(pin);
            }
        }
    }

    static std::shared_ptr<const MfeStreamProjection> make_projection(
        int projection_experts, int rows_per_expert, int columns,
        std::vector<std::optional<MfeStreamExpertLocation>> locations) {
        auto profile = std::make_shared<MlxMfeKernelProfile>();
        profile->nint_q_dispatch = true;
        std::vector<int> group_sizes;
        std::unordered_set<const MfeStreamPool*> visited;
        for (const auto& location : locations) {
            const auto& pool = *location->pool;
            if (!visited.insert(&pool).second) continue;
            if (const auto* nint = std::get_if<MfeNintStreamLayout>(&pool.layout)) {
                profile->family_mask |= 1u << kFamilyNint;
                group_sizes.push_back(nint->group_size);
                if (nint->adaptive_storage) {
                    for (std::size_t first = 0; first < nint->q_selectors.size(); first += rows_per_expert) {
                        const auto q = nint->q_selectors[first] + 1u;
                        profile->nint_q_dispatch &= q == 4u || q == 5u || q == 6u || q == 8u;
                        for (std::size_t row = 1; profile->nint_q_dispatch && row < rows_per_expert; ++row)
                            profile->nint_q_dispatch &= nint->q_selectors[first + row] == nint->q_selectors[first];
                    }
                } else {
                    profile->nint_q_dispatch &= nint->q_bits == 4 || nint->q_bits == 5
                        || nint->q_bits == 6 || nint->q_bits == 8;
                }
            } else if (const auto* mx = std::get_if<MfeMxStreamLayout>(&pool.layout)) {
                profile->family_mask |= 1u << (mx->bits == 4 ? kFamilyMxfp4 : kFamilyMxfp8);
            } else {
                const auto& vq = std::get<MfeNvqJscStreamLayout>(pool.layout);
                profile->family_mask |= 1u << kFamilyVq;
                profile->vq_profile_mask |= 1u << vq.decoder_profile;
                profile->vq_layouts |= std::uint64_t{1}
                    << (vq.decoder_profile * 7 + vq.execution_layout);
                profile->vq_execution_layout |= vq.execution_layout != 0;
            }
        }
        std::sort(group_sizes.begin(), group_sizes.end());
        group_sizes.erase(std::unique(group_sizes.begin(), group_sizes.end()), group_sizes.end());
        profile->nint_group_sizes = std::make_shared<const std::vector<int>>(std::move(group_sizes));
        return std::make_shared<MfeStreamProjection>(MfeStreamProjection{
            projection_experts, rows_per_expert, columns, std::move(locations), std::move(profile)});
    }

    std::shared_ptr<const MfeStreamProjection> parse_nint_projection(const std::string& name) {
        const auto& model = record(name);
        const auto& record = model.metadata;
        if (record.nbytes < 50)
            throw std::runtime_error("truncated streamed NINT expert header: " + name);
        const auto header = model.read_range(name, 0, 50);
        const auto count = read_scalar<std::int64_t>(header, 18, "NINT expert count");
        const auto output = read_scalar<std::int64_t>(header, 26, "NINT expert output");
        const auto input = read_scalar<std::int64_t>(header, 34, "NINT expert input");
        const auto limit = std::numeric_limits<int>::max();
        if (read_scalar<std::uint32_t>(header, 14, "NINT rank") != 3
            || count <= 0 || count > limit || (experts > 0 && count != experts)
            || output <= 0 || output > limit || input <= 0 || input > limit
            || count > limit / output)
            throw std::runtime_error("invalid streamed NINT expert geometry: " + name);
        auto pool = std::make_shared<MfeStreamPool>();
        pool->dtype = std::string(tyloi::canonical_format_dtype(record.dtype));
        pool->expert_count = static_cast<int>(count);
        pool->payload_bytes = record.nbytes;
        pool->layout = parse_streamed_nint_layout(model, name, 0, record.nbytes,
            static_cast<int>(count * output), static_cast<int>(input),
            static_cast<int>(output), static_cast<int>(count));
        std::vector<std::optional<MfeStreamExpertLocation>> locations(static_cast<std::size_t>(count));
        for (int expert = 0; expert < count; ++expert)
            locations[expert] = MfeStreamExpertLocation{pool, expert};
        return make_projection(static_cast<int>(count), static_cast<int>(output),
            static_cast<int>(input), std::move(locations));
    }

    std::shared_ptr<const MfeStreamProjection>
    parse_mfe_projection(
        const std::string& name) {
        const auto& model = record(name);
        const auto& record = model.metadata;
        if (is_nint_dtype(record.dtype)) return parse_nint_projection(name);
        if (record.dtype != "MFE") {
            throw MfeStreamUnsupported(
                "expert record is not MFE: " + name);
        }
        constexpr std::uint64_t header_size = 20;
        constexpr std::uint64_t pool_header_size = 24;
        if (record.nbytes < header_size) {
            throw std::runtime_error(
                "truncated streamed MFE header: " + name);
        }
        const auto header = model.read_range(name, 0, header_size);
        const std::string_view magic(
            reinterpret_cast<const char*>(header.data()), 4);
        if (magic == "NIM1") {
            throw MfeStreamUnsupported(
                "legacy NIM1 records are not streamable");
        }
        if (magic != "MFE1" && magic != "NIM2") {
            throw std::runtime_error(
                "invalid streamed MFE magic: " + name);
        }
        const auto record_experts = read_scalar<std::uint32_t>(
            header, 4, "MFE expert count");
        const auto rows_per_expert = read_scalar<std::uint32_t>(
            header, 8, "MFE output width");
        const auto columns = read_scalar<std::uint32_t>(
            header, 12, "MFE input width");
        const auto pool_count = read_scalar<std::uint32_t>(
            header, 16, "MFE pool count");
        if (record_experts == 0
            || record_experts > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max())
            || (experts > 0
                && record_experts != static_cast<std::uint32_t>(experts))
            || rows_per_expert == 0 || columns == 0
            || rows_per_expert > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max())
            || columns > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max())
            || pool_count == 0 || pool_count > record_experts) {
            throw std::runtime_error(
                "invalid streamed MFE dimensions: " + name);
        }
        const int projection_experts = static_cast<int>(record_experts);

        std::vector<std::optional<MfeStreamExpertLocation>> locations(
            static_cast<std::size_t>(projection_experts));
        std::uint64_t offset = header_size;
        for (std::uint32_t pool_index = 0;
             pool_index < pool_count;
             ++pool_index) {
            if (offset > record.nbytes
                || pool_header_size > record.nbytes - offset) {
                throw std::runtime_error(
                    "truncated streamed MFE pool header: " + name);
            }
            const auto pool_header = model.read_range(
                name, offset, pool_header_size);
            const auto pool_experts = read_scalar<std::uint32_t>(
                pool_header, 0, "MFE pool expert count");
            const auto dtype_bytes = read_scalar<std::uint32_t>(
                pool_header, 4, "MFE pool dtype length");
            const auto payload_bytes = read_scalar<std::uint64_t>(
                pool_header, 8, "MFE pool payload length");
            const auto runtime_bytes = read_scalar<std::uint64_t>(
                pool_header, 16, "MFE pool runtime length");
            if (pool_experts == 0 || pool_experts > record_experts
                || dtype_bytes == 0 || dtype_bytes > 32) {
                throw std::runtime_error(
                    "invalid streamed MFE pool metadata: " + name);
            }
            offset = checked_range_add(
                offset, pool_header_size, "MFE pool metadata offset");
            const auto ids_bytes = checked_range_product(
                pool_experts, sizeof(std::int32_t), "MFE expert IDs");
            const auto metadata_bytes = checked_range_add(
                ids_bytes, dtype_bytes, "MFE pool metadata bytes");
            const auto runtime_offset = checked_range_add(
                offset, metadata_bytes, "MFE runtime offset");
            const auto payload_offset = checked_range_add(
                runtime_offset, runtime_bytes, "MFE payload offset");
            const auto payload_end = checked_range_add(
                payload_offset, payload_bytes, "MFE payload end");
            if (payload_end > record.nbytes) {
                throw std::runtime_error(
                    "truncated streamed MFE pool: " + name);
            }
            const auto metadata = model.read_range(
                name, offset, metadata_bytes);
            const auto dtype = read_ascii(
                metadata,
                static_cast<std::size_t>(ids_bytes),
                dtype_bytes,
                "MFE pool dtype");
            const auto canonical = std::string(
                tyloi::canonical_format_dtype(dtype));
            const auto pool_rows_u64 = checked_range_product(
                pool_experts,
                rows_per_expert,
                "MFE pool rows");
            if (pool_rows_u64 > static_cast<std::uint64_t>(
                    std::numeric_limits<int>::max())) {
                throw std::runtime_error(
                    "streamed MFE pool row count is too large: " + name);
            }
            const int pool_rows = static_cast<int>(pool_rows_u64);
            MfeStreamLayout layout;
            if (is_nint_dtype(canonical)) {
                if (runtime_bytes != 0) {
                    throw std::runtime_error(
                        "streamed NINTv2 has unexpected runtime metadata: "
                        + name);
                }
                layout = parse_streamed_nint_layout(
                    model, name, payload_offset, payload_bytes,
                    pool_rows, static_cast<int>(columns),
                    static_cast<int>(rows_per_expert));
            } else if (canonical == "MXFP4" || canonical == "MXFP8") {
                if (runtime_bytes != 0) {
                    throw std::runtime_error(
                        "streamed MX has unexpected runtime metadata: "
                        + name);
                }
                layout = parse_streamed_mx_layout(
                    model, name, payload_offset, payload_bytes,
                    pool_rows, static_cast<int>(columns),
                    canonical == "MXFP4" ? 4 : 8);
            } else if (is_vq_dtype(canonical)) {
                if (payload_bytes < 4) {
                    throw std::runtime_error("truncated streamed VQ magic: " + name);
                }
                const auto magic = model.read_range(name, payload_offset, 4);
                if (std::memcmp(magic.data(), "NQ1S", 4) == 0 ||
                    std::memcmp(magic.data(), "NQ1L", 4) == 0) {
                    if (canonical != tyloi::kNvqDtype || runtime_bytes != 0) {
                        throw std::runtime_error("invalid streamed NVQ1 cohort metadata: " + name);
                    }
                    layout = parse_streamed_nvq1_layout(
                        model, name, payload_offset, payload_bytes,
                        pool_rows, static_cast<int>(columns));
                } else {
                    layout = parse_streamed_nvq_jsc_layout(
                        model, name, payload_offset, payload_bytes,
                        pool_rows, static_cast<int>(columns));
                }
            } else {
                throw MfeStreamUnsupported(
                    "MFE contains a cohort without expert slicing support: "
                    + dtype);
            }

            auto pool = std::make_shared<MfeStreamPool>();
            pool->dtype = dtype;
            pool->expert_count = static_cast<int>(pool_experts);
            pool->runtime_bytes = runtime_bytes;
            pool->payload_offset = payload_offset;
            pool->payload_bytes = payload_bytes;
            pool->layout = std::move(layout);
            pool->runtime = model.read_range(name, runtime_offset, runtime_bytes);
            for (std::uint32_t local = 0; local < pool_experts; ++local) {
                const auto expert = read_scalar<std::int32_t>(
                    metadata,
                    static_cast<std::size_t>(local) * sizeof(std::int32_t),
                    "MFE global expert ID");
                if (expert < 0 || expert >= projection_experts
                    || locations[static_cast<std::size_t>(expert)].has_value()) {
                    throw std::runtime_error(
                        "invalid or duplicate streamed MFE expert ID: "
                        + name);
                }
                locations[static_cast<std::size_t>(expert)] =
                    MfeStreamExpertLocation{
                        pool,
                        static_cast<int>(local),
                    };
            }
            offset = payload_end;
        }
        if (offset != record.nbytes
            || std::any_of(
                locations.begin(), locations.end(),
                [](const auto& value) { return !value.has_value(); })) {
            throw std::runtime_error(
                "streamed MFE pools do not exactly cover the record: "
                + name);
        }
        return make_projection(projection_experts, static_cast<int>(rows_per_expert),
            static_cast<int>(columns), std::move(locations));
    }

    std::shared_ptr<const MfeStreamProjection>
    mfe_projection_locked(const std::string& name) {
        const auto found = mfe_projections.find(name);
        if (found != mfe_projections.end()) {
            return found->second;
        }
        auto result = parse_mfe_projection(name);
        mfe_projections.emplace(name, result);
        if (on_prepared) on_prepared(name);
        return result;
    }

    MfeReadBytes load_mfe_expert_blob(
        const std::string& name,
        const MfeStreamProjection& projection,
        const MfeStreamExpertLocation& location,
        MfeReadCounters* reads = nullptr, MfeReadPlan* plan = nullptr) {
        MfeReadScope observation(reads);
        MfeReadPlanScope planned(plan);
        const auto& runtime = location.pool->runtime;
        MfeReadBytes blob;
        const auto prefix_bytes = std::visit([](const auto& layout) {
            if constexpr (std::is_same_v<std::decay_t<decltype(layout)>, MfeNvqJscStreamLayout>)
                return layout.prefix.size();
            else return layout.header.size();
        }, location.pool->layout);
        const auto reserve_bytes = checked_add(
            checked_add(location.pool->payload_bytes / location.pool->expert_count,
                checked_add(runtime.size(), prefix_bytes + 8, "streamed expert prefix"),
                "streamed expert payload"),
            checked_add(
                std::size_t{20 + 24 + sizeof(std::int32_t)},
                location.pool->dtype.size(),
                "streamed expert metadata"),
            "streamed expert blob");
        blob.reserve(reserve_bytes);
        blob.insert(blob.end(), {'M', 'F', 'E', '1'});
        append_scalar_bytes<std::uint32_t>(blob, 1);
        append_scalar_bytes<std::uint32_t>(
            blob, static_cast<std::uint32_t>(projection.out_per_expert));
        append_scalar_bytes<std::uint32_t>(
            blob, static_cast<std::uint32_t>(projection.neuron_len));
        append_scalar_bytes<std::uint32_t>(blob, 1);
        append_scalar_bytes<std::uint32_t>(blob, 1);
        append_scalar_bytes<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(location.pool->dtype.size()));
        append_scalar_bytes<std::uint64_t>(blob, 0);
        append_scalar_bytes<std::uint64_t>(blob, runtime.size());
        append_scalar_bytes<std::int32_t>(blob, 0);
        blob.insert(
            blob.end(),
            location.pool->dtype.begin(),
            location.pool->dtype.end());
        blob.insert(blob.end(), runtime.begin(), runtime.end());
        const auto payload_begin = blob.size();
        append_streamed_mfe_expert_payload(blob, record(name), name, *location.pool,
            location.local_expert, projection.out_per_expert);
        patch_scalar_bytes<std::uint64_t>(blob, 28, blob.size() - payload_begin,
            "streamed MFE payload size");
        return blob;
    }

    const MfeRecord& record(const std::string& name) const {
        const auto found = records.find(name);
        if (found == records.end()) throw std::runtime_error("missing MFE record: " + name);
        return found->second;
    }
    std::unordered_map<std::string, MfeRecord> records;
    std::size_t cache_limit = 0;
    const int experts = 0;
    std::function<void(const std::string&)> on_prepared;
    mutable std::mutex mutex;
    std::unordered_map<
        std::string,
        std::shared_ptr<const MfeStreamProjection>> mfe_projections;
    tyloi::weights::Lru<Key, MfeCacheValue, KeyHash> mfe_cache{std::nullopt};
    std::unordered_set<std::string> prefill_priority;
    std::size_t resident_bytes = 0;
    std::unordered_map<std::string, std::size_t> backing_records;
    MlxResourceTelemetry resources;
    tyloi::HostParallelPool io_workers{{}, false};
    std::unordered_map<std::string, PreparedProjection> prepared;
    std::unordered_map<std::string,
        std::vector<std::shared_ptr<const MfeCachedExpert>>> resident_pages;
    std::unordered_map<std::string, std::shared_ptr<const MlxMfeCachedProjection>> cached_views;
    std::size_t page_generation = 0;
    std::unordered_map<std::string, std::uint8_t> route_confidence;
    std::unordered_map<std::string, std::shared_ptr<InFlight>> loading;
    std::condition_variable loaded;
    std::size_t epoch = 0;
    MlxMfeCacheReuseStats stats;
    std::mutex pin_mutex;
    std::condition_variable pin_ready;
    std::list<PinRequest> pin_queue;
    std::unordered_map<Key, std::list<PinRequest>::iterator, KeyHash> pin_pending;
    bool pin_stopping = false;
    std::thread pin_worker;

    void drop_prepared(const std::string& name) {
        const auto found = prepared.find(name);
        if (found == prepared.end()) return;
        resident_bytes -= found->second.weight.packed_nbytes();
        prepared.erase(found);
    }

    void drop_cached_view(const std::string& name) {
        ++page_generation;
        if (const auto found = cached_views.find(name); found != cached_views.end()) {
            resident_bytes -= found->second->metadata_bytes;
            cached_views.erase(found);
        }
    }

    void trim() {
        while (resident_bytes > cache_limit && !prepared.empty())
            drop_prepared(prepared.begin()->first);
        while (resident_bytes > cache_limit && !mfe_cache.empty()) {
            const auto& key = *mfe_cache.oldest();
            auto& victim = mfe_cache.at(key);
            if (victim.credit > 0) {
                --victim.credit;
                mfe_cache.get(key);
                continue;
            }
            drop_cached_view(key.name);
            resident_bytes -= victim.weight->packed_nbytes;
            resident_pages.at(key.name)[key.expert].reset();
            ++stats.evictions;
            cancel_pin(key);
            mfe_cache.erase(key);
        }
    }
};

MlxMfeOffloadCache::MlxMfeOffloadCache(
    const source::Source& model, std::size_t limit, int experts)
    : MlxMfeOffloadCache(model, limit, experts, {}) {}

MlxMfeOffloadCache::MlxMfeOffloadCache(
    const source::Source& model, std::size_t limit, int experts,
    std::function<void(const std::string&)> prepared)
    : impl_(std::make_unique<Impl>(model, limit, experts, std::move(prepared))) {}

MlxMfeOffloadCache::~MlxMfeOffloadCache() =
    default;

bool MlxMfeOffloadCache::can_offload(
    const std::string& name) {
    std::lock_guard lock(impl_->mutex);
    try {
        (void)impl_->mfe_projection_locked(name);
        return true;
    } catch (const MfeStreamUnsupported&) {
        return false;
    }
}

bool MlxMfeOffloadCache::can_group_mfe(
    const std::string& name) {
    std::lock_guard lock(impl_->mutex);
    try {
        (void)impl_->mfe_projection_locked(name);
        return true;
    } catch (const MfeStreamUnsupported&) {
        return false;
    }
}

void MlxMfeOffloadCache::activate_record(const std::string& name) {
    std::lock_guard lock(impl_->mutex);
    (void)impl_->mfe_projection_locked(name);
    impl_->backing_records.emplace(name, impl_->record(name).metadata.nbytes);
}

void MlxMfeOffloadCache::prepare_projection(
    const std::string& name, bool shared_input, bool fused_swiglu) {
    auto* preparation = MlxKernelPreparation::current();
    if (!preparation || preparation->routes() <= 0) return;
    std::shared_ptr<const MfeStreamProjection> projection;
    {
        std::lock_guard lock(impl_->mutex);
        projection = impl_->mfe_projection_locked(name);
    }
    preparation->collect([&] {
        collect_packed_copy();
        const auto blob = impl_->load_mfe_expert_blob(
            name, *projection, *projection->experts_by_id.front());
        const auto weight = MlxMfeWeight::from_blob(blob).with_kernel_profile(projection->kernel_profile);
        weight.collect_cache_preparation(projection->experts, shared_input, fused_swiglu);
    });
}

void MlxMfeOffloadCache::prepare_swiglu_pair(
    const std::string& gate, const std::string& up, int routes) {
    auto* preparation = MlxKernelPreparation::current();
    if (!preparation || routes <= 0 || routes > 16) return;
    std::array<std::shared_ptr<const MfeStreamProjection>, 2> projections;
    {
        std::lock_guard lock(impl_->mutex);
        projections = {impl_->mfe_projection_locked(gate), impl_->mfe_projection_locked(up)};
    }
    if (!uses_indirect_expert_pages(projections[0]->kernel_profile->family_mask
            | projections[1]->kernel_profile->family_mask)) return;
    preparation->collect([&] {
        std::vector<MlxMfeWeight> weights;
        for (std::size_t index = 0; index < projections.size(); ++index) {
            const auto& projection = projections[index];
            const auto blob = impl_->load_mfe_expert_blob(index == 0 ? gate : up,
                *projection, *projection->experts_by_id.front());
            const auto weight = MlxMfeWeight::from_blob(blob).with_kernel_profile(projection->kernel_profile);
            weights.push_back(MlxMfeWeight::concatenate_expert_pages(
                {weight, weight}, projection->kernel_profile).with_kernel_profile(projection->kernel_profile));
        }
        for (const int rows : {1, 2}) {
            const auto input = mlx::core::zeros(Shape{rows, projections[0]->neuron_len}, mlx::core::float16);
            const std::vector<std::int32_t> host_ids(static_cast<std::size_t>(rows) * routes, 0);
            const auto ids = array(host_ids.begin(), Shape{rows, routes});
            const auto groups = rows > 1 ? moe_route_groups_host(host_ids, rows, routes) : std::nullopt;
            preparation->add(weights[0].routed_swiglu_pair(weights[1], input, ids, 0.0f,
                groups ? &*groups : nullptr));
            if (rows == 1) {
                const auto map = array({0}, mlx::core::int32);
                preparation->add(weights[0].routed_swiglu_pair(weights[1], input, ids,
                    0.0f, nullptr, &map, &map));
            }
        }
    });
}

std::size_t MlxMfeOffloadCache::backing_payload_bytes() const {
    std::lock_guard lock(impl_->mutex);
    std::size_t bytes = 0;
    for (const auto& record : impl_->backing_records) bytes += record.second;
    return bytes;
}

MlxMfeProjectionInfo
MlxMfeOffloadCache::projection_info(
    const std::string& name) {
    std::lock_guard lock(impl_->mutex);
    const auto projection = impl_->mfe_projection_locked(name);
    MlxMfeProjectionInfo result;
    result.experts = projection->experts;
    result.out_per_expert = projection->out_per_expert;
    result.neuron_len = projection->neuron_len;
    for (int expert = 0; expert < projection->experts; ++expert) {
        if (projection->experts_by_id[
                static_cast<std::size_t>(expert)].has_value()) {
            result.available_experts.push_back(expert);
        }
    }
    return result;
}

std::vector<std::uint8_t>
MlxMfeOffloadCache::availability(
    const std::string& name) {
    std::lock_guard lock(impl_->mutex);
    const auto projection = impl_->mfe_projection_locked(name);
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(projection->experts), 0);
    for (int expert = 0; expert < projection->experts; ++expert) {
        result[static_cast<std::size_t>(expert)] =
            static_cast<std::uint8_t>(projection->experts_by_id[
                static_cast<std::size_t>(expert)].has_value());
    }
    return result;
}

std::optional<std::vector<MlxMfeWeight>> MlxMfeOffloadCache::layer_mfe(
    std::span<const std::string> names,
    std::span<const std::int32_t> active_experts) {
    if (names.empty()) throw std::invalid_argument("MFE layer records cannot be empty");
    std::size_t bytes = 0;
    std::size_t dense_bytes = 0;
    std::size_t missing_bytes = 0;
    {
        std::lock_guard lock(impl_->mutex);
        for (const auto& name : names) {
            const auto projection = impl_->mfe_projection_locked(name);
            bytes = checked_add(bytes, impl_->record(name).metadata.nbytes, "MFE layer bytes");
            const auto elements = checked_product(checked_product(projection->experts,
                projection->out_per_expert, "MFE layer rows"), projection->neuron_len,
                "MFE layer elements");
            dense_bytes = checked_add(dense_bytes, checked_product(elements, 2, "MFE layer F16 bytes"),
                "MFE layer execution bytes");
            std::unordered_set<std::int32_t> seen;
            for (const auto expert : active_experts) {
                if (expert < 0 || expert >= projection->experts || !projection->experts_by_id[expert])
                    throw std::out_of_range("bulk MFE expert is unavailable");
                if (!seen.insert(expert).second || impl_->mfe_cache.find({name, expert})) continue;
                const auto& pool = projection->experts_by_id[expert]->pool;
                missing_bytes = checked_add(missing_bytes, pool->payload_bytes / pool->expert_count,
                    "bulk MFE missing bytes");
            }
        }
        if (bytes > impl_->cache_limit / 2) return std::nullopt;
        if (!active_experts.empty() && missing_bytes < bytes / 2) return std::nullopt;
    }
    const auto staging = checked_add(checked_add(bytes, bytes, "MFE layer staging"),
        mlx_reference_enabled() ? dense_bytes : 0, "MFE layer staging and execution");
    if (!MlxResidentBudgetScope::permits_optional(staging)) return std::nullopt;
    MlxResidentBudgetScope::reserve(staging);
    std::size_t epoch = 0;
    {
        std::lock_guard lock(impl_->mutex);
        if (bytes > impl_->cache_limit / 2) return std::nullopt;
        epoch = impl_->epoch;
    }
    std::vector<MlxMfeWeight> result;
    result.reserve(names.size());
    constexpr std::size_t chunk = std::size_t{8} << 20;
    const auto read_record = [&](const std::string& name, array storage) {
        const std::span blob(storage.data<std::uint8_t>(), storage.nbytes());
        const auto pieces = (blob.size() + chunk - 1) / chunk;
        std::vector<double> durations(pieces);
        impl_->io_workers.run(0, pieces, 1,
            static_cast<int>(std::min(8u, std::max(1u, std::thread::hardware_concurrency()))),
            [&](std::int64_t begin, std::int64_t end) {
                for (auto index = begin; index < end; ++index) {
                    const auto offset = static_cast<std::size_t>(index) * chunk;
                    const auto started = std::chrono::steady_clock::now();
                    impl_->record(name).read_range_into(name, offset, std::as_writable_bytes(
                        std::span(blob.data() + offset, std::min(chunk, blob.size() - offset))));
                    durations[index] = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started).count();
                }
            });
        std::lock_guard lock(impl_->mutex);
        impl_->stats.bytes_read += blob.size();
        impl_->stats.read_calls += pieces;
        impl_->stats.io_seconds += std::accumulate(durations.begin(), durations.end(), 0.0);
        return storage;
    };
    auto storage = read_record(names.front(), allocate_wire_array(
        static_cast<std::size_t>(impl_->record(names.front()).metadata.nbytes)));
    for (std::size_t index = 0; index < names.size(); ++index) {
        std::future<decltype(storage)> next;
        if (index + 1 < names.size()) {
            auto pending = allocate_wire_array(
                static_cast<std::size_t>(impl_->record(names[index + 1]).metadata.nbytes));
            next = std::async(std::launch::async,
                [&read_record, &names, index, pending = std::move(pending)]() mutable {
                return read_record(names[index + 1], std::move(pending));
            });
        }
        result.push_back(is_nint_dtype(impl_->record(names[index]).metadata.dtype)
            ? MlxMfeWeight::from_nint_blob(storage) : MlxMfeWeight::from_blob(storage));
        if (next.valid()) storage = next.get();
    }
    {
        std::lock_guard lock(impl_->mutex);
        if (epoch == impl_->epoch)
            for (const auto& name : names)
                impl_->backing_records.emplace(name, impl_->record(name).metadata.nbytes);
    }
    return result;
}

MfeGroupTask MlxMfeOffloadCache::Impl::prepare_grouped(
    const std::string& name,
    const std::vector<std::int32_t>& active_experts,
    std::span<const std::uint32_t> activation_counts, bool prefill_scan) {
    auto* const impl_ = this;
    if (active_experts.empty())
        throw std::invalid_argument("streamed MFE active expert set cannot be empty");
    if (!activation_counts.empty() && activation_counts.size() != active_experts.size())
        throw std::invalid_argument("MFE activation counts must match the active experts");
    if (std::any_of(activation_counts.begin(), activation_counts.end(),
        [](auto count) { return count == 0; }))
        throw std::invalid_argument("MFE activation counts must be positive");
    std::optional<MlxResidentBudgetHold> staging;
    std::vector<std::shared_ptr<const Impl::MfeCachedExpert>> protected_pages;
    std::optional<std::size_t> admission_epoch;
    if (MlxResidentBudgetScope::enabled()) {
        std::size_t working_bytes = 0;
        {
            std::lock_guard lock(impl_->mutex);
            admission_epoch = impl_->epoch;
            const auto projection = impl_->mfe_projection_locked(name);
            protected_pages.reserve(std::min(active_experts.size(), static_cast<std::size_t>(projection->experts)));
            const bool indirect_pages = uses_indirect_expert_pages(projection->kernel_profile->family_mask);
            if (indirect_pages) working_bytes = std::size_t{3} * getpagesize();
            std::size_t missing_bytes = 0, largest_missing = 0, missing_count = 0;
            std::unordered_set<std::int32_t> seen;
            for (const auto expert : active_experts) {
                if (!seen.insert(expert).second) continue;
                if (expert < 0 || expert >= projection->experts || !projection->experts_by_id[expert])
                    throw std::out_of_range("streamed MFE expert is unavailable");
                const auto cached = impl_->mfe_cache.find({name, expert});
                const bool missing = !cached;
                if (!missing) protected_pages.push_back(cached->weight);
                const auto bytes = missing
                    ? streamed_expert_storage_bound(*projection, *projection->experts_by_id[expert])
                    : indirect_pages ? sizeof(std::int32_t) * (kDescriptorSize + 1)
                        + 21 * sizeof(std::uint64_t) : cached->weight->packed_nbytes;
                const auto copies = missing && !indirect_pages ? std::size_t{5} : std::size_t{1};
                working_bytes = checked_add(working_bytes, checked_product(bytes, copies,
                    "expert packed staging"), "active expert working set");
                if (missing && indirect_pages) {
                    missing_bytes = checked_add(missing_bytes, bytes, "missing expert storage");
                    largest_missing = std::max(largest_missing, bytes);
                    ++missing_count;
                }
            }
            if (missing_count) {
                const auto workers = std::max(1u, std::thread::hardware_concurrency());
                const auto concurrent = std::min(missing_count,
                    std::size_t(prefill_scan ? workers : std::min(kMfeSmallMReadWorkers, workers)));
                const auto batch_bytes = prefill_scan
                    ? std::max(largest_missing, std::size_t{8} << 20) : largest_missing;
                const auto in_flight = std::min(missing_bytes,
                    checked_product(concurrent, batch_bytes, "expert reader working set"));
                working_bytes = checked_add(working_bytes,
                    checked_product(in_flight, std::size_t{3}, "expert reader staging"),
                    "active expert working set");
            }
        }
        staging.emplace(working_bytes);
    }
    std::unique_lock lock(impl_->mutex);
    std::shared_ptr<Impl::InFlight> completed;
    if (const auto found = impl_->loading.find(name); found != impl_->loading.end()) {
        completed = found->second;
        ++impl_->stats.load_waits;
        const auto started = std::chrono::steady_clock::now();
        impl_->loaded.wait(lock, [&] { return completed->complete; });
        impl_->stats.wait_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
    }
    const auto projection = impl_->mfe_projection_locked(name);

    std::vector<Impl::Key> ordered_keys;
    ordered_keys.reserve(active_experts.size());
    std::unordered_set<Impl::Key, Impl::KeyHash> active_keys;
    active_keys.reserve(active_experts.size());
    std::unordered_map<std::int32_t, std::uint32_t> counts;
    for (std::size_t index = 0; index < active_experts.size(); ++index) {
        const auto expert = active_experts[index];
        if (expert < 0 || expert >= projection->experts) {
            throw std::out_of_range(
                "streamed MFE global expert ID is out of range");
        }
        if (!projection->experts_by_id[
                static_cast<std::size_t>(expert)].has_value()) {
            throw std::runtime_error(
                "streamed MFE global expert is unavailable: "
                + std::to_string(expert));
        }
        Impl::Key key{name, expert};
        if (active_keys.emplace(key).second) {
            ordered_keys.push_back(std::move(key));
        }
        auto& count = counts[expert];
        count = std::min<std::uint64_t>(3, static_cast<std::uint64_t>(count)
            + (activation_counts.empty() ? 1 : activation_counts[index]));
    }

    impl_->backing_records.emplace(name, impl_->record(name).metadata.nbytes);

    const bool priority_scan = prefill_scan && impl_->prefill_priority.contains(name);
    const auto touch = [&](const Impl::Key& key, bool hit) {
        if (prefill_scan && !priority_scan) return;
        const auto cached = impl_->mfe_cache.find(key);
        if (!cached) return;
        auto& value = *cached;
        value.credit = static_cast<std::uint8_t>(std::min<unsigned>(3,
            value.credit + counts.at(key.expert) - (hit ? 0 : 1)));
        impl_->mfe_cache.get(key);
    };
    const auto reuse_packed = [&](const Impl::PreparedProjection& packed)
        -> std::optional<MlxMfeWeight> {
        std::vector<std::int32_t> local;
        local.reserve(ordered_keys.size());
        for (const auto& key : ordered_keys) {
            const auto found = packed.local_ids.find(key.expert);
            if (found == packed.local_ids.end()) return std::nullopt;
            local.push_back(found->second);
        }
        auto weight = packed.weight;
        for (const auto& key : ordered_keys) touch(key, true);
        impl_->stats.expert_hits += ordered_keys.size();
        ++impl_->stats.packed_hits;
        lock.unlock();
        return weight.select_experts(local);
    };
    if (completed && completed->result)
        if (auto result = reuse_packed(*completed->result)) co_return std::move(*result);
    const auto prepared = impl_->prepared.find(name);
    if (prepared != impl_->prepared.end())
        if (auto result = reuse_packed(prepared->second)) co_return std::move(*result);
    impl_->drop_prepared(name);

    struct ActivePage {
        Impl::Key key;
        std::shared_ptr<const Impl::MfeCachedExpert> value;
        bool staged = false;
    };
    struct PendingPage {
        std::size_t page = 0;
        std::int32_t expert = 0;
        MfeReadCounters reads;
        double prepare_seconds = 0;
        std::optional<array> destination;
    };
    std::vector<ActivePage> pages;
    pages.reserve(ordered_keys.size());
    std::vector<PendingPage> pending;
    pending.reserve(ordered_keys.size());
    for (const auto& key : ordered_keys) {
        const auto cached = impl_->mfe_cache.find(key);
        if (cached) {
            pages.push_back({key, cached->weight, false});
            ++impl_->stats.expert_hits;
            continue;
        }
        if (const auto protected_page = std::find_if(protected_pages.begin(), protected_pages.end(),
                [&](const auto& page) { return page->expert == key.expert; }); protected_page != protected_pages.end()) {
            pages.push_back({key, *protected_page, false});
            ++impl_->stats.expert_hits;
            continue;
        }
        if (completed) {
            const auto found = std::find_if(completed->pages.begin(), completed->pages.end(),
                [&](const auto& page) { return page->expert == key.expert; });
            if (found != completed->pages.end()) {
                pages.push_back({key, *found, false});
                ++impl_->stats.expert_hits;
                continue;
            }
        }
        const auto page = pages.size();
        pages.push_back({key, nullptr, true});
        pending.push_back({page, key.expert, {}});
    }

    std::vector<std::shared_ptr<const Impl::MfeCachedExpert>> bank_pages;
    std::size_t bank_bytes = 0;
    const bool indirect_pages = uses_indirect_expert_pages(projection->kernel_profile->family_mask);
    if (!prefill_scan) {
        auto& confidence = impl_->route_confidence[name];
        confidence = pending.empty() ? static_cast<std::uint8_t>(std::min<unsigned>(15, confidence + 1)) : 0;
    }
    const auto resident = impl_->resident_pages.find(name);
    if (!indirect_pages && pending.empty() && resident != impl_->resident_pages.end()) {
        for (const auto& page : resident->second) {
            if (!page) continue;
            bank_pages.push_back(page);
            bank_bytes = checked_add(bank_bytes, page->packed_nbytes + 128,
                "prepared MFE expert bytes");
        }
    }
    if (!indirect_pages && pending.empty()) {
        for (const auto& page : pages) {
            if (impl_->mfe_cache.find(page.key)) continue;
            bank_pages.push_back(page.value);
            bank_bytes = checked_add(bank_bytes, page.value->packed_nbytes + 128,
                "in-flight MFE expert bytes");
        }
    }
    const auto spare = impl_->cache_limit - std::min(impl_->cache_limit, impl_->resident_bytes);
    bool prepare = pending.empty() && bank_pages.size() > 1 && bank_bytes <= spare;
    const auto epoch = admission_epoch.value_or(impl_->epoch);
    auto in_flight = std::make_shared<Impl::InFlight>();
    impl_->loading.emplace(name, in_flight);
    struct FinishLoading {
        Impl& impl;
        const std::string& name;
        std::unique_lock<std::mutex>& lock;
        std::shared_ptr<Impl::InFlight> request;
        ~FinishLoading() {
            if (!lock.owns_lock()) lock.lock();
            request->complete = true;
            impl.loading.erase(name);
            impl.loaded.notify_all();
        }
    } finish{*impl_, name, lock, in_flight};
    lock.unlock();
    if (prepare) {
        prepare = MlxResidentBudgetScope::permits_optional(bank_bytes);
        if (prepare) MlxResidentBudgetScope::reserve(bank_bytes);
    }
    if (!prepare) bank_pages.clear();
    std::optional<array> transient_arena;
    if (prefill_scan && pending.size() > 1 && !mlx_reference_enabled()) {
        std::vector<std::size_t> offsets{0};
        for (const auto& request : pending) {
            const auto bytes = streamed_expert_storage_bound(*projection,
                *projection->experts_by_id[request.expert]);
            offsets.push_back(checked_add(checked_add(offsets.back(), bytes,
                "prefill expert arena"), 15, "prefill expert alignment") & ~std::size_t{15});
        }
        if (offsets.back() > spare && offsets.back() <= std::size_t(std::numeric_limits<int>::max())) {
            transient_arena = offsets.back() >= (std::size_t{64} << 10)
                ? allocate_wire_array(offsets.back())
                : allocate_packed_array(offsets.back(), mlx::core::uint8);
            for (std::size_t index = 0; index < pending.size(); ++index) {
                auto view = array(Shape{static_cast<int>(offsets[index + 1] - offsets[index])}, mlx::core::uint8, nullptr, {});
                view.copy_shared_buffer(*transient_arena, view.strides(), view.flags(), view.size(), offsets[index]);
                view.set_status(array::Status::available);
                pending[index].destination.emplace(std::move(view));
            }
            pooled_expert_staging.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (prefill_scan && pending.size() > 1) {
        std::vector<std::size_t> read_order(pending.size());
        std::iota(read_order.begin(), read_order.end(), 0);
        std::sort(read_order.begin(), read_order.end(), [&](auto a, auto b) {
            const auto& left = *projection->experts_by_id[pending[a].expert];
            const auto& right = *projection->experts_by_id[pending[b].expert];
            return std::tie(left.pool->payload_offset, left.local_expert)
                < std::tie(right.pool->payload_offset, right.local_expert);
        });
        std::vector<std::vector<std::size_t>> batches;
        std::size_t batch_bytes = 0;
        for (const auto index : read_order) {
            const auto& location = *projection->experts_by_id[pending[index].expert];
            const auto bytes = streamed_expert_storage_bound(*projection, location);
            const auto* previous = batches.empty() ? nullptr
                : &*projection->experts_by_id[pending[batches.back().back()].expert];
            if (!previous || previous->pool != location.pool
                || previous->local_expert + 1 != location.local_expert
                || bytes > (std::size_t{8} << 20) - std::min(batch_bytes, std::size_t{8} << 20)) {
                batches.emplace_back();
                batch_bytes = 0;
            }
            batches.back().push_back(index);
            batch_bytes += bytes;
        }
        co_yield MfeGroupWork{batches.size(),
            [&](std::int64_t begin, std::int64_t end) {
                for (auto batch_index = begin; batch_index < end; ++batch_index) {
                    const auto& batch = batches[batch_index];
                    const auto started = std::chrono::steady_clock::now();
                    std::vector<MfeReadBytes> wire;
                    std::vector<MfeReadPlan> plans(batch.size());
                    wire.reserve(batch.size());
                    auto& reads = pending[batch.front()].reads;
                    for (std::size_t index = 0; index < batch.size(); ++index) {
                        const auto& location = *projection->experts_by_id[pending[batch[index]].expert];
                        wire.push_back(impl_->load_mfe_expert_blob(name, *projection, location, &reads,
                            batch.size() > 1 ? &plans[index] : nullptr));
                    }
                    if (batch.size() > 1) {
                        MfeReadScope observation(&reads);
                        read_mfe_batch(impl_->record(name), name, wire, plans);
                    }
                    for (std::size_t index = 0; index < batch.size(); ++index) {
                        auto& request = pending[batch[index]];
                        const std::array blobs{std::span<const std::uint8_t>(wire[index])};
                        pages[request.page].value = std::make_shared<Impl::MfeCachedExpert>(
                            request.expert, request.destination
                                ? MlxMfeWeight::from_projection_blobs_impl(blobs, false, nullptr, &*request.destination)
                                : MlxMfeWeight::from_blob(wire[index]), request.destination.has_value());
                    }
                    pending[batch.front()].prepare_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started).count() - reads.seconds;
                }
            }};
    } else if (!pending.empty()) co_yield MfeGroupWork{pending.size(),
        [&](std::int64_t begin, std::int64_t end) {
            for (auto index = begin; index < end; ++index) {
                auto& request = pending[index];
                const auto& location = *projection->experts_by_id[request.expert];
                const auto started = std::chrono::steady_clock::now();
                const auto eligible = [&] {
                    if (mlx_reference_enabled()) return false;
                    if (const auto* mx = std::get_if<MfeMxStreamLayout>(&location.pool->layout))
                        return mx->bits == 4;
                    const auto* vq = std::get_if<MfeNvqJscStreamLayout>(&location.pool->layout);
                    if (!vq || vq->group64 || vq->execution_layout != 0
                        || vq->decoder_profile == kVqProfileGeneric
                        || vq->decoder_profile == kVqProfileNpqS || vq->decoder_profile == kVqProfileNpqL) return false;
                    const auto rows = static_cast<std::uint64_t>(projection->out_per_expert);
                    return rows * vq->groups * vq->state_bits % 8 == 0
                        && rows * vq->vectors * vq->index_bits % 8 == 0
                        && rows * vq->signs * vq->auxiliary_bits % 8 == 0;
                }();
                const auto* mx = std::get_if<MfeMxStreamLayout>(&location.pool->layout);
                const bool deferred_scales = eligible && mx && !prefill_scan
                    && pending.size() <= kMfeDirectRangeMaxExperts;
                MfeReadPlan plan;
                struct MetadataScope {
                    MfeMetadataRead previous;
                    MetadataScope(MfeMetadataRead value) : previous(std::exchange(direct_read_metadata, value)) {}
                    ~MetadataScope() { direct_read_metadata = previous; }
                } metadata_scope(deferred_scales ? MfeMetadataRead::DeferredMxScales
                    : eligible ? MfeMetadataRead::Eager : MfeMetadataRead::Planned);
                auto blob = impl_->load_mfe_expert_blob(name, *projection, location, &request.reads,
                    eligible ? &plan : nullptr);
                std::vector<MfeDirectReadTarget> targets;
                for (const auto& range : plan)
                    targets.push_back({range.source, blob.data() + range.destination, range.bytes});
                std::optional<MlxMfeWeight> parsed;
                const auto parse = [&] {
                    const std::array blobs{std::span<const std::uint8_t>(blob)};
                    return request.destination
                        ? MlxMfeWeight::from_projection_blobs_impl(blobs, false, nullptr, &*request.destination)
                        : MlxMfeWeight::from_blob(blob);
                };
                if (eligible) {
                    MfeDirectReadScope redirect(targets);
                    parsed.emplace(parse());
                } else parsed.emplace(parse());
                const bool complete = std::all_of(targets.begin(), targets.end(),
                    [](const auto& range) { return range.destination != nullptr; });
                {
                    MfeReadScope observation(&request.reads);
                    read_mfe_direct_targets(impl_->record(name), name, targets, complete, pending.size());
                }
                if (deferred_scales && complete) {
                    const auto scale_offset = mx->scales_offset
                        + static_cast<std::uint64_t>(location.local_expert)
                            * projection->out_per_expert * (mx->columns / 32);
                    const auto scale = std::find_if(targets.begin(), targets.end(),
                        [&](const auto& range) { return range.offset == scale_offset; });
                    if (scale == targets.end()) throw std::logic_error("deferred MX scale range is missing");
                    if (std::find(scale->destination, scale->destination + scale->bytes, std::uint8_t{255})
                        != scale->destination + scale->bytes)
                        throw std::runtime_error("MFE MXFP4 contains an E8M0 NaN scale");
                }
                if (!complete) {
                    parsed.emplace(parse());
                }
                pages[request.page].value = std::make_shared<Impl::MfeCachedExpert>(
                    request.expert, std::move(*parsed), request.destination.has_value());
                request.prepare_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count() - request.reads.seconds;
            }
        }};

    const auto view_started = std::chrono::steady_clock::now();
    std::vector<MlxMfeWeight> weights;
    weights.reserve(pages.size());
    for (const auto& page : pages) {
        weights.push_back(page.value->weight);
    }
    std::optional<Impl::PreparedProjection> bank;
    const auto assemble = [&](const std::vector<MlxMfeWeight>& values) {
        return MlxMfeWeight::concatenate_expert_pages(values, projection->kernel_profile);
    };
    if (prepare) {
        std::vector<MlxMfeWeight> bank_weights;
        bank_weights.reserve(bank_pages.size());
        std::unordered_map<std::int32_t, std::int32_t> local;
        for (const auto& page : bank_pages) {
            local.emplace(page->expert, static_cast<std::int32_t>(bank_weights.size()));
            bank_weights.push_back(page->weight);
        }
        bank.emplace(Impl::PreparedProjection{
            assemble(bank_weights)
                .with_kernel_profile(projection->kernel_profile), std::move(local)});
    }
    auto result = [&] {
        if (!bank) return assemble(weights)
            .with_kernel_profile(projection->kernel_profile);
        std::vector<std::int32_t> local;
        for (const auto& key : ordered_keys) local.push_back(bank->local_ids.at(key.expert));
        return bank->weight.select_experts(local);
    }();
    if (!bank && pages.size() > 1 && !indirect_pages) {
        std::unordered_map<std::int32_t, std::int32_t> local;
        for (std::size_t index = 0; index < pages.size(); ++index)
            local.emplace(pages[index].key.expert, static_cast<std::int32_t>(index));
        bank.emplace(Impl::PreparedProjection{result, std::move(local)});
    }

    const auto view_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - view_started).count();
    lock.lock();
    impl_->stats.view_seconds += view_seconds;
    for (const auto& request : pending) {
        impl_->stats.bytes_read += request.reads.bytes;
        impl_->stats.read_calls += request.reads.calls;
        impl_->stats.parallel_body_read_batches += request.reads.parallel_body_batches;
        impl_->stats.io_seconds += request.reads.seconds;
        impl_->stats.prepare_seconds += request.prepare_seconds;
    }
    std::unordered_map<std::int32_t, std::int32_t> result_ids;
    for (std::size_t index = 0; index < pages.size(); ++index) {
        result_ids.emplace(pages[index].key.expert, static_cast<std::int32_t>(index));
        in_flight->pages.push_back(pages[index].value);
    }
    in_flight->result.emplace(Impl::PreparedProjection{result, std::move(result_ids)});
    impl_->stats.expert_loads += pending.size();
    if (epoch != impl_->epoch) co_return result;

    if (std::any_of(pages.begin(), pages.end(), [&](const auto& page) {
        return !page.value->transient && !impl_->mfe_cache.find(page.key);
    })) impl_->drop_cached_view(name);

    impl_->mfe_cache.reserve_entries(checked_add(
        impl_->mfe_cache.entries().size(),
        static_cast<std::size_t>(std::count_if(
            pages.begin(), pages.end(),
            [](const auto& page) { return page.staged; })),
        "resident MFE cache entries"));
    for (const auto& page : pages) {
        if (page.value->transient) continue;
        if (impl_->mfe_cache.find(page.key)) continue;
        auto& by_id = impl_->resident_pages[name];
        if (by_id.empty()) by_id.resize(projection->experts);
        const auto bytes = checked_add(impl_->resident_bytes,
            page.value->packed_nbytes, "resident MFE expert bytes");
        impl_->mfe_cache.put(page.key, {page.value}, page.value->packed_nbytes,
            prefill_scan && !priority_scan);
        impl_->resident_bytes = bytes;
        by_id[page.key.expert] = page.value;
    }
    for (const auto& page : pages) touch(page.key, !page.staged);
    if (bank && bank->weight.packed_nbytes() <= impl_->cache_limit
        - std::min(impl_->cache_limit, impl_->resident_bytes)) {
        const auto bytes = bank->weight.packed_nbytes();
        impl_->prepared.emplace(name, std::move(*bank));
        impl_->resident_bytes += bytes;
    }
    impl_->trim();
    for (const auto& page : pages) {
        const auto cached = impl_->mfe_cache.find(page.key);
        if (!cached) continue;
        auto& value = *cached;
        if (!value.pin_attempted) value.pin_attempted = impl_->enqueue_pin(page.key, value.weight);
    }
    co_return result;
}

MlxMfeWeight MlxMfeOffloadCache::grouped_mfe(
    const std::string& name,
    const std::vector<std::int32_t>& active_experts,
    std::span<const std::uint32_t> activation_counts, bool prefill_scan) {
    const std::array<std::string, 1> names{name};
    return grouped_mfe(names, active_experts, activation_counts, prefill_scan).front();
}

std::vector<MlxMfeWeight> MlxMfeOffloadCache::grouped_mfe(
    std::span<const std::string> names,
    const std::vector<std::int32_t>& active_experts,
    std::span<const std::uint32_t> activation_counts, bool prefill_scan,
    const std::function<void(const std::vector<MlxMfeWeight>&)>& prefix_ready) {
    if (names.empty()) throw std::invalid_argument("streamed MFE projections cannot be empty");
    std::unordered_set<std::string> unique;
    for (const auto& name : names)
        if (!unique.insert(name).second)
            throw std::invalid_argument("streamed MFE projections must be distinct");
    const auto workers = std::max(1u, std::thread::hardware_concurrency());
    if (prefix_ready) {
        if (names.size() != 3)
            throw std::invalid_argument("streamed MFE prefix submission requires three projections");
        bool parallel = !prefill_scan;
        std::array<std::size_t, 2> missing{};
        if (parallel) {
            std::lock_guard lock(impl_->mutex);
            for (std::size_t index = 0; index < names.size(); ++index) {
                (void)impl_->mfe_projection_locked(names[index]);
                for (const auto expert : active_experts)
                    missing[index == 2] += !impl_->mfe_cache.find({names[index], expert});
            }
            parallel = missing[0] && missing[1] && missing[0] + missing[1] <= std::min(8u, workers);
        }
        if (!parallel) {
            auto result = grouped_mfe(names.first(2), active_experts, activation_counts, prefill_scan);
            prefix_ready(result);
            result.push_back(grouped_mfe(names.back(), active_experts, activation_counts, prefill_scan));
            return result;
        }
    }
    std::vector<MfeGroupTask> tasks;
    std::vector<MfeGroupWork*> jobs;
    std::vector<std::size_t> offsets{0};
    std::vector<std::size_t> order(names.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](auto left, auto right) { return names[left] < names[right]; });
    tasks.reserve(names.size());
    for (const auto index : order) {
        tasks.push_back(impl_->prepare_grouped(names[index], active_experts, activation_counts, prefill_scan));
        tasks.back().resume();
        if (auto& work = tasks.back().handle.promise().work; work) {
            jobs.push_back(&*work);
            offsets.push_back(checked_add(offsets.back(), work->count, "MFE grouped expert tasks"));
        }
    }
    auto* allocation_log = tyloi::metal::AllocationScope::log();
    // Prefill has wide CPU parsing work; small route sets mostly contend on
    // the allocator's resource/residency lock once the I/O queue is saturated.
    if (prefix_ready) {
        std::vector<std::pair<std::size_t, MfeGroupWork*>> prioritized;
        for (std::size_t index = 0; index < tasks.size(); ++index)
            if (auto& work = tasks[index].handle.promise().work; work)
                prioritized.emplace_back(order[index], &*work);
        std::sort(prioritized.begin(), prioritized.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        jobs.clear();
        offsets.assign(1, 0);
        std::size_t prefix_jobs = 0;
        for (const auto& [index, job] : prioritized) {
            jobs.push_back(job);
            offsets.push_back(checked_add(offsets.back(), job->count, "MFE grouped expert tasks"));
            if (index < 2) ++prefix_jobs;
        }
        std::atomic<std::size_t> prefix_left{offsets[prefix_jobs]};
        std::mutex ready_mutex;
        std::condition_variable ready_condition;
        bool io_done = false;
        auto background = std::async(std::launch::async, [&] {
            struct Finish {
                std::mutex& mutex;
                std::condition_variable& condition;
                bool& done;
                ~Finish() {
                    { std::lock_guard lock(mutex); done = true; }
                    condition.notify_all();
                }
            } finish{ready_mutex, ready_condition, io_done};
            impl_->io_workers.run(0, offsets.back(), 1,
                static_cast<int>(std::min<std::size_t>(offsets.back(), std::min(8u, workers))),
                [&](std::int64_t begin, std::int64_t end) {
                    tyloi::metal::AllocationScope allocation_scope(allocation_log);
                    while (begin < end) {
                        const auto index = std::upper_bound(offsets.begin(), offsets.end(),
                            static_cast<std::size_t>(begin)) - offsets.begin() - 1;
                        const auto next = std::min<std::int64_t>(end, offsets[index + 1]);
                        jobs[index]->load(begin - offsets[index], next - offsets[index]);
                        const auto count = static_cast<std::size_t>(next - begin);
                        if (index < prefix_jobs && prefix_left.fetch_sub(count) == count) {
                            std::lock_guard lock(ready_mutex);
                            ready_condition.notify_all();
                        }
                        begin = next;
                    }
                });
        });
        {
            std::unique_lock lock(ready_mutex);
            ready_condition.wait(lock, [&] { return prefix_left.load() == 0 || io_done; });
        }
        if (prefix_left.load() != 0) background.get();
        std::array<std::optional<MlxMfeWeight>, 2> selected;
        for (std::size_t index = 0; index < tasks.size(); ++index) if (order[index] < 2) {
            auto& task = tasks[index];
            if (task.handle.promise().work) task.resume();
            if (!task.handle.promise().result)
                throw std::logic_error("streamed MFE prefix preparation did not finish");
            selected[order[index]] = *task.handle.promise().result;
        }
        const std::vector<MlxMfeWeight> prefix{*selected[0], *selected[1]};
        prefix_ready(prefix);
        bool submitted_early = false;
        { std::lock_guard lock(ready_mutex); submitted_early = !io_done; }
        if (prefix_jobs && prefix_jobs < jobs.size()) {
            std::lock_guard lock(impl_->mutex);
            ++impl_->stats.overlapped_read_batches;
            impl_->stats.overlapped_read_submissions += submitted_early;
        }
        background.get();
    } else impl_->io_workers.run(0, offsets.back(), 1,
        static_cast<int>(std::min(std::max<std::size_t>(1, offsets.back()),
            std::size_t(prefill_scan ? workers : std::min(kMfeSmallMReadWorkers, workers)))),
        [&](std::int64_t begin, std::int64_t end) {
            tyloi::metal::AllocationScope allocation_scope(allocation_log);
            while (begin < end) {
                const auto index = std::upper_bound(offsets.begin(), offsets.end(),
                    static_cast<std::size_t>(begin)) - offsets.begin() - 1;
                const auto next = std::min<std::int64_t>(end, offsets[index + 1]);
                jobs[index]->load(begin - offsets[index], next - offsets[index]);
                begin = next;
            }
        });
    std::vector<std::optional<MlxMfeWeight>> ordered(tasks.size());
    for (std::size_t index = 0; index < tasks.size(); ++index) {
        auto& task = tasks[index];
        if (task.handle.promise().work) task.resume();
        if (!task.handle.promise().result)
            throw std::logic_error("streamed MFE preparation did not finish");
        ordered[order[index]].emplace(std::move(*task.handle.promise().result));
    }
    std::vector<MlxMfeWeight> result;
    result.reserve(ordered.size());
    for (auto& weight : ordered) result.push_back(std::move(*weight));
    return result;
}

std::size_t
MlxMfeOffloadCache::cache_limit_bytes() const noexcept {
    std::lock_guard lock(impl_->mutex);
    return impl_->cache_limit;
}


std::size_t
MlxMfeOffloadCache::resident_packed_bytes() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->resident_bytes;
}

std::size_t
MlxMfeOffloadCache::cached_expert_count() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->mfe_cache.entries().size();
}

MlxMfeCacheReuseStats MlxMfeOffloadCache::reuse_stats() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->stats;
}

std::size_t MlxMfeOffloadCache::set_cache_limit(std::size_t bytes) {
    std::lock_guard lock(impl_->mutex);
    const auto before = impl_->resident_bytes;
    if (impl_->cache_limit != bytes) {
        impl_->cache_limit = bytes;
        ++impl_->epoch;
    }
    impl_->trim();
    return before - impl_->resident_bytes;
}

void MlxMfeOffloadCache::set_prefill_priority(std::span<const std::string> names) {
    std::lock_guard lock(impl_->mutex);
    std::unordered_set<std::string> priority(names.begin(), names.end());
    if (priority == impl_->prefill_priority) return;
    impl_->prefill_priority = std::move(priority);
    if (impl_->prefill_priority.empty()) return;
    const auto& keys = impl_->mfe_cache.keys();
    if (keys.empty()) return;
    auto current = std::prev(keys.end());
    const auto last = keys.begin();
    for (;;) {
        const bool done = current == last;
        const auto next = done ? keys.end() : std::prev(current);
        auto& value = impl_->mfe_cache.at(*current);
        const bool preferred = impl_->prefill_priority.contains(current->name);
        value.credit = preferred ? 3 : 0;
        if (preferred) impl_->mfe_cache.get(*current);
        if (done) break;
        current = next;
    }
}

void MlxMfeOffloadCache::discard_record(
    const std::string& name) noexcept {
    std::lock_guard lock(impl_->mutex);
    ++impl_->epoch;
    impl_->drop_prepared(name);
    impl_->drop_cached_view(name);
    impl_->route_confidence.erase(name);
    const auto& keys = impl_->mfe_cache.keys();
    if (!keys.empty()) {
        auto current = std::prev(keys.end());
        const auto last = keys.begin();
        for (;;) {
            const bool done = current == last;
            const auto next = done ? keys.end() : std::prev(current);
            if (current->name == name) {
                impl_->resident_bytes -= impl_->mfe_cache.at(*current).weight->packed_nbytes;
                impl_->cancel_pin(*current);
                impl_->mfe_cache.erase(*current);
            }
            if (done) break;
            current = next;
        }
    }
    impl_->mfe_projections.erase(name);
    impl_->resident_pages.erase(name);
    impl_->backing_records.erase(name);
}

void MlxMfeOffloadCache::clear() {
    std::lock_guard lock(impl_->mutex);
    ++impl_->epoch;
    impl_->prepared.clear();
    impl_->cached_views.clear();
    ++impl_->page_generation;
    impl_->route_confidence.clear();
    impl_->resident_pages.clear();
    {
        std::lock_guard pin_lock(impl_->pin_mutex);
        impl_->pin_queue.clear();
        impl_->pin_pending.clear();
    }
    impl_->mfe_cache.clear();
    impl_->mfe_projections.clear();
    impl_->resident_bytes = 0;
}

std::shared_ptr<const MlxMfeCachedProjection> MlxMfeOffloadCache::cached_projection(const std::string& name) {
    std::unique_lock lock(impl_->mutex);
    const auto confidence = impl_->route_confidence.find(name);
    if (confidence == impl_->route_confidence.end() || confidence->second < 8) return {};
    if (const auto cached = impl_->cached_views.find(name); cached != impl_->cached_views.end()) return cached->second;
    const auto projection = impl_->mfe_projection_locked(name);
    const auto resident = impl_->resident_pages.find(name);
    if (resident == impl_->resident_pages.end() || !uses_indirect_expert_pages(projection->kernel_profile->family_mask)) return {};
    const auto count = std::count_if(resident->second.begin(), resident->second.end(), [](const auto& page) { return bool(page); });
    if (!count) return {};
    const auto estimate = checked_add(std::size_t{3} * getpagesize(), checked_product(count,
        sizeof(std::int32_t) * (kDescriptorSize + 2) + 21 * sizeof(std::uint64_t), "cached expert metadata"),
        "cached expert metadata");
    lock.unlock();
    if (!MlxResidentBudgetScope::permits_optional(estimate)) return {};
    MlxResidentBudgetScope::reserve(estimate);
    lock.lock();
    if (const auto cached = impl_->cached_views.find(name); cached != impl_->cached_views.end()) return cached->second;
    const auto current = impl_->resident_pages.find(name);
    if (current == impl_->resident_pages.end()) return {};
    const auto epoch = impl_->epoch;
    const auto generation = impl_->page_generation;
    std::vector<MlxMfeWeight> pages;
    std::vector<std::int32_t> map(projection->experts, -1);
    for (std::size_t expert = 0; expert < current->second.size(); ++expert) if (const auto& page = current->second[expert]) {
        map[expert] = static_cast<std::int32_t>(pages.size());
        pages.push_back(page->weight);
    }
    if (pages.empty()) return {};
    lock.unlock();
    auto weight = MlxMfeWeight::concatenate_expert_pages(pages, projection->kernel_profile).with_prepared_page_resources();
    const auto weight_metadata = weight.cached_page_metadata_bytes();
    if (!weight_metadata) return {};
    array expert_map(map.begin(), Shape{projection->experts}, mlx::core::int32);
    const auto metadata = *weight_metadata + expert_map.nbytes();
    auto view = std::make_shared<const MlxMfeCachedProjection>(MlxMfeCachedProjection{
        std::move(weight), std::move(expert_map), std::move(map), name, metadata});
    lock.lock();
    if (epoch == impl_->epoch && generation == impl_->page_generation && metadata <= impl_->cache_limit) {
        if (const auto cached = impl_->cached_views.find(name); cached != impl_->cached_views.end()) return cached->second;
        const auto bytes = checked_add(impl_->resident_bytes, metadata, "cached expert metadata");
        impl_->cached_views.emplace(name, view);
        impl_->resident_bytes = bytes;
        impl_->trim();
    }
    return view;
}

bool MlxMfeOffloadCache::resolve_cached_projections(std::span<const std::shared_ptr<const MlxMfeCachedProjection>> views,
    std::span<const std::int32_t> experts) {
    if (experts.empty() || views.empty()) throw std::invalid_argument("cached expert routes cannot be empty");
    std::unordered_map<std::int32_t, unsigned> counts;
    std::vector<std::int32_t> ordered;
    counts.reserve(experts.size());
    ordered.reserve(experts.size());
    for (const auto expert : experts) {
        const auto [found, inserted] = counts.try_emplace(expert, 0);
        if (inserted) ordered.push_back(expert);
        auto& count = found->second;
        count = std::min(3u, count + 1);
    }
    std::vector<bool> hits(views.size(), true);
    for (std::size_t index = 0; index < views.size(); ++index) {
        if (!views[index]) throw std::invalid_argument("cached projection is missing");
        for (const auto expert : ordered) {
            if (expert < 0 || static_cast<std::size_t>(expert) >= views[index]->host_expert_map.size())
                throw std::out_of_range("cached expert route is out of range");
            hits[index] = hits[index] && views[index]->host_expert_map[expert] >= 0;
        }
    }
    std::lock_guard lock(impl_->mutex);
    if (std::any_of(hits.begin(), hits.end(), [](bool hit) { return !hit; })) {
        for (std::size_t index = 0; index < views.size(); ++index) if (!hits[index]) {
            impl_->route_confidence[views[index]->name] = 0;
        }
        ++impl_->stats.cached_route_misses;
        return false;
    }
    ++impl_->stats.cached_route_hits;
    for (const auto& view : views) {
        auto& confidence = impl_->route_confidence[view->name];
        confidence = static_cast<std::uint8_t>(std::min<unsigned>(15, confidence + 1));
        impl_->stats.expert_hits += counts.size();
        for (const auto expert : ordered) {
            const auto count = counts.at(expert);
            const Impl::Key key{view->name, expert};
            const auto cached = impl_->mfe_cache.find(key);
            if (!cached) continue;
            auto& value = *cached;
            value.credit = static_cast<std::uint8_t>(std::min<unsigned>(3, value.credit + count));
            if (value.credit >= 2 && !value.pin_attempted) value.pin_attempted = impl_->enqueue_pin(key, value.weight);
            impl_->mfe_cache.get(key);
        }
    }
    return true;
}

array MlxMfeCachedRoute::forward(const array& source, const array& routes, float swiglu_limit,
    const array* reduce_weights) {
    ids = routes;
    const auto down = projections.size() - 1;
    auto intermediate = [&]() {
        if (projections.size() != 3)
            return projections[0]->weight.routed_swiglu(source,
                mlx::core::take(projections[0]->expert_map, routes), swiglu_limit);
        return projections[0]->weight.routed_swiglu_pair(projections[1]->weight,
            source, routes, swiglu_limit, nullptr, &projections[0]->expert_map, &projections[1]->expert_map);
    }();
    auto down_ids = mlx::core::take(projections[down]->expert_map, routes);
    return reduce_weights ? projections[down]->weight.routed_matmul_reduce(intermediate, down_ids, *reduce_weights)
        : projections[down]->weight.routed_matmul(intermediate, down_ids);
}

std::span<const std::int32_t> MlxMfeCachedRoute::host_ids() {
    if (!ids) throw std::logic_error("cached route was not evaluated");
    ids->eval();
    if (ids->dtype() != mlx::core::int32 || !ids->flags().row_contiguous) {
        ids = mlx::core::contiguous(mlx::core::astype(*ids, mlx::core::int32));
        ids->eval();
    }
    return {ids->data<std::int32_t>(), ids->size()};
}

bool MlxMfeCachedRoute::all_resident() {
    const auto routes = host_ids();
    for (const auto& projection : projections) for (const auto expert : routes) {
        if (expert < 0 || static_cast<std::size_t>(expert) >= projection->host_expert_map.size())
            throw std::out_of_range("cached expert route is out of range");
        if (projection->host_expert_map[expert] < 0) return false;
    }
    return true;
}

bool MlxMfeCachedRoute::resolve() {
    return cache->resolve_cached_projections(projections, host_ids());
}

} // namespace tyloi::metal

#include "mlx_paged_session_codec.h"
#include "mlx_resident_budget.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <mlx/mlx.h>

namespace mfq::metal {
namespace {

using mlx::core::Dtype;
using mlx::core::Shape;
using mlx::core::array;

constexpr std::array<std::uint8_t, 8> kMagic{
    'M', 'F', 'Q', 'M', 'L', 'X', '1', 0};
constexpr std::uint32_t kVersion = 4;
constexpr std::uint32_t kDFlashVersion = 5;
constexpr std::uint8_t kKvLayer = 1;
constexpr std::uint8_t kRecurrentLayer = 2;
constexpr std::uint8_t kRecurrentUnavailableLayer = 3;
constexpr std::uint32_t kMiniRuntime = 1;
constexpr std::uint32_t kQwen35Runtime = 2;
constexpr std::uint32_t kQwen4Runtime = 3;
constexpr std::uint32_t kDeepseekV41Runtime = 4;
constexpr std::uint32_t kInkling2Runtime = 5;
constexpr std::uint32_t kHy4Runtime = 6;
constexpr std::uint8_t kQsaLayer = 4;
constexpr std::uint8_t kGdnLayer = 5;
constexpr std::uint8_t kGdnUnavailableLayer = 6;

class Writer {
public:
    template <typename T>
    void scalar(T value) {
        ensure_capacity(bytes_.size() + sizeof(T));
        using Unsigned = std::make_unsigned_t<T>;
        const auto converted = static_cast<Unsigned>(value);
        for (std::size_t index = 0; index < sizeof(T); ++index) {
            bytes_.push_back(static_cast<std::uint8_t>(
                converted >> (index * 8)));
        }
    }

    void raw(const void* data, std::size_t size) {
        ensure_capacity(bytes_.size() + size);
        const auto* begin = static_cast<const std::uint8_t*>(data);
        bytes_.insert(bytes_.end(), begin, begin + size);
    }

    void tensor(const array& source) {
        auto value = mlx::core::contiguous(source);
        value.eval();
        scalar<std::uint8_t>(
            static_cast<std::uint8_t>(value.dtype().val()));
        if (value.ndim() == 0 || value.ndim() > 8) {
            throw std::runtime_error("invalid MLX cache tensor rank");
        }
        scalar<std::uint8_t>(static_cast<std::uint8_t>(value.ndim()));
        scalar<std::uint16_t>(0);
        for (const auto dimension : value.shape()) {
            if (dimension <= 0) {
                throw std::runtime_error("invalid MLX cache tensor shape");
            }
            scalar<std::int32_t>(dimension);
        }
        scalar<std::uint64_t>(value.nbytes());
        raw(value.data<std::uint8_t>(), value.nbytes());
    }

    std::uint8_t* tensor_storage(const Shape& shape, Dtype dtype) {
        std::size_t bytes = dtype.size();
        scalar<std::uint8_t>(static_cast<std::uint8_t>(dtype.val()));
        scalar<std::uint8_t>(static_cast<std::uint8_t>(shape.size()));
        scalar<std::uint16_t>(0);
        for (const int dimension : shape) {
            if (dimension <= 0 || bytes > std::numeric_limits<std::size_t>::max() / dimension)
                throw std::runtime_error("invalid offloaded cache tensor shape");
            scalar<std::int32_t>(dimension);
            bytes *= dimension;
        }
        scalar<std::uint64_t>(bytes);
        const auto offset = bytes_.size();
        ensure_capacity(offset + bytes);
        bytes_.resize(offset + bytes);
        return bytes_.data() + offset;
    }

    void offloaded_kv_tensor(const MlxQsaKvSnapshot& state, int start, int count, bool value) {
        const int key_width = state.quantization.enabled() ? mlx_kv_packed_width(state.dimension, state.quantization.key_bits()) : state.dimension;
        const int value_width = state.quantization.enabled() ? mlx_kv_packed_width(state.dimension, state.quantization.value_bits()) : state.dimension;
        const int width = value ? value_width : key_width;
        auto* destination = tensor_storage(Shape{1, state.heads, count, width}, state.dtype);
        const auto lane = static_cast<std::size_t>(width) * state.dtype.size();
        const auto key_bytes = static_cast<std::size_t>(key_width) * state.dtype.size() * state.heads;
        std::vector<std::uint8_t> row(static_cast<std::size_t>(key_width + value_width) * state.dtype.size() * state.heads);
        for (int i = 0; i < count; ++i) {
            state.rows.read_rows(start + i, 1, row.data());
            for (int head = 0; head < state.heads; ++head)
                std::memcpy(destination + (head * static_cast<std::size_t>(count) + i) * lane,
                    row.data() + (value ? key_bytes : 0) + head * lane, lane);
        }
    }

    void offloaded_index_tensor(const MlxQsaIndexSnapshot& state, int start, int count) {
        auto* destination = tensor_storage(Shape{1, count, state.width}, state.dtype);
        state.rows.read_rows(start, count, destination);
    }

    std::shared_ptr<const std::vector<std::uint8_t>> finish() && {
        return std::make_shared<const std::vector<std::uint8_t>>(
            std::move(bytes_));
    }

private:
    void ensure_capacity(std::size_t needed) {
        if (needed <= bytes_.capacity()) return;
        const auto capacity = std::max(needed, bytes_.capacity() * 2);
        auto budget = std::make_unique<MlxResidentBudgetHold>(capacity);
        bytes_.reserve(capacity);
        budget_ = std::move(budget);
    }
    std::vector<std::uint8_t> bytes_;
    std::unique_ptr<MlxResidentBudgetHold> budget_;
};

struct SerializedTensor {
    // Non-owning view into a PagedPrefixPayload. decode() keeps every shared
    // payload alive until reconstruction completes, so block tensors can be
    // copied directly into their final full-prefix allocation instead of
    // allocating one MLX array per block and concatenating a second time.
    Dtype dtype = mlx::core::float16;
    Shape shape;
    const std::uint8_t* data = nullptr;
    std::size_t bytes = 0;
};

class Reader {
public:
    explicit Reader(const std::vector<std::uint8_t>& bytes)
        : cursor_(bytes.data()), end_(bytes.data() + bytes.size()) {}

    template <typename T>
    T scalar(const char* name) {
        if (remaining() < sizeof(T)) {
            throw std::runtime_error(
                std::string("truncated MLX cache ") + name);
        }
        using Unsigned = std::make_unsigned_t<T>;
        Unsigned converted = 0;
        for (std::size_t index = 0; index < sizeof(T); ++index) {
            converted |= static_cast<Unsigned>(cursor_[index]) << (index * 8);
        }
        cursor_ += sizeof(T);
        return static_cast<T>(converted);
    }

    void raw(void* destination, std::size_t size, const char* name) {
        if (remaining() < size) {
            throw std::runtime_error(
                std::string("truncated MLX cache ") + name);
        }
        std::memcpy(destination, cursor_, size);
        cursor_ += size;
    }

    SerializedTensor tensor() {
        const auto dtype_value = scalar<std::uint8_t>("dtype");
        const auto rank = scalar<std::uint8_t>("rank");
        (void)scalar<std::uint16_t>("tensor flags");
        if (rank == 0 || rank > 8 ||
            dtype_value > static_cast<std::uint8_t>(Dtype::Val::complex64)) {
            throw std::runtime_error("invalid MLX cache tensor header");
        }
        Shape shape;
        shape.reserve(rank);
        std::size_t elements = 1;
        for (std::uint8_t index = 0; index < rank; ++index) {
            const auto dimension = scalar<std::int32_t>("shape");
            if (dimension <= 0 ||
                elements > std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(dimension)) {
                throw std::runtime_error("invalid MLX cache tensor shape");
            }
            shape.push_back(dimension);
            elements *= static_cast<std::size_t>(dimension);
        }
        const auto dtype = dtype_from_value(
            static_cast<Dtype::Val>(dtype_value));
        const auto bytes = scalar<std::uint64_t>("tensor size");
        if (elements > std::numeric_limits<std::size_t>::max() / dtype.size() ||
            bytes != elements * dtype.size() || bytes > remaining()) {
            throw std::runtime_error("invalid MLX cache tensor size");
        }
        SerializedTensor result{
            dtype,
            std::move(shape),
            cursor_,
            static_cast<std::size_t>(bytes),
        };
        cursor_ += static_cast<std::size_t>(bytes);
        return result;
    }

    void expect_end() const {
        if (cursor_ != end_) {
            throw std::runtime_error("trailing MLX cache payload bytes");
        }
    }

    std::size_t remaining() const noexcept {
        return static_cast<std::size_t>(end_ - cursor_);
    }

private:
    static Dtype dtype_from_value(Dtype::Val value) {
        switch (value) {
            case Dtype::Val::bool_: return mlx::core::bool_;
            case Dtype::Val::uint8: return mlx::core::uint8;
            case Dtype::Val::uint16: return mlx::core::uint16;
            case Dtype::Val::uint32: return mlx::core::uint32;
            case Dtype::Val::uint64: return mlx::core::uint64;
            case Dtype::Val::int8: return mlx::core::int8;
            case Dtype::Val::int16: return mlx::core::int16;
            case Dtype::Val::int32: return mlx::core::int32;
            case Dtype::Val::int64: return mlx::core::int64;
            case Dtype::Val::float16: return mlx::core::float16;
            case Dtype::Val::float32: return mlx::core::float32;
            case Dtype::Val::float64: return mlx::core::float64;
            case Dtype::Val::bfloat16: return mlx::core::bfloat16;
            case Dtype::Val::complex64: return mlx::core::complex64;
        }
        throw std::runtime_error("unsupported MLX cache tensor dtype");
    }

    const std::uint8_t* cursor_;
    const std::uint8_t* end_;
};

struct DecodedLayer {
    std::uint8_t kind = 0;
    bool exact = false;
    int batch = 0;
    int heads = 0;
    int maximum_sequence = 0;
    int head_dimension = 0;
    int capacity = 0;
    int position = 0;
    int index_start = 0;
    int index_ratio = 0;
    Dtype dtype = mlx::core::float16;
    MlxKvQuantization quantization;
    SerializedTensor first;
    SerializedTensor second;
    std::optional<SerializedTensor> index;
    std::optional<SerializedTensor> pooled;
    std::optional<SerializedTensor> ple;
    std::vector<std::int64_t> ple_context;
};

struct DecodedBlock {
    std::uint32_t runtime = 0;
    std::uint32_t start = 0;
    std::uint32_t count = 0;
    std::vector<DecodedLayer> layers;
    std::vector<DecodedLayer> mtp_layers;
    std::optional<SerializedTensor> last_hidden;
    std::vector<std::pair<SerializedTensor, SerializedTensor>> dflash;
};

void write_header(
    Writer& writer,
    std::uint32_t runtime,
    std::uint32_t start,
    std::uint32_t count,
    std::size_t layers,
    std::uint32_t version = kVersion) {
    writer.raw(kMagic.data(), kMagic.size());
    writer.scalar<std::uint32_t>(version);
    writer.scalar<std::uint32_t>(runtime);
    writer.scalar<std::uint32_t>(start);
    writer.scalar<std::uint32_t>(count);
    writer.scalar<std::uint32_t>(static_cast<std::uint32_t>(layers));
}

void write_kv_layer(
    Writer& writer,
    const MlxKvCacheSnapshot& snapshot,
    int start,
    int count,
    std::uint8_t kind = kKvLayer,
    bool exact = true) {
    if (snapshot.position < start + count || snapshot.batch != 1) {
        throw std::runtime_error("invalid MLX KV cache block range");
    }
    writer.scalar<std::uint8_t>(kind);
    writer.scalar<std::uint8_t>(
        static_cast<std::uint8_t>(snapshot.dtype.val()));
    writer.scalar<std::uint16_t>((exact ? 1 : 0) | (snapshot.quantization.enabled()
        ? 2 | (static_cast<int>(snapshot.quantization.bits * 2) << 8) : 0));
    writer.scalar<std::int32_t>(snapshot.batch);
    writer.scalar<std::int32_t>(snapshot.heads);
    writer.scalar<std::int32_t>(snapshot.maximum_sequence);
    writer.scalar<std::int32_t>(snapshot.head_dimension);
    writer.scalar<std::int32_t>(snapshot.capacity);
    writer.scalar<std::int32_t>(start + count);
    const Shape begin{0, 0, start, 0};
    const Shape end{
        snapshot.batch,
        snapshot.heads,
        start + count,
        snapshot.key.shape(3),
    };
    writer.tensor(mlx::core::slice(snapshot.key, begin, end));
    writer.tensor(mlx::core::slice(snapshot.value, begin, Shape{snapshot.batch, snapshot.heads, start + count, snapshot.value.shape(3)}));
}

void write_recurrent_layer(
    Writer& writer,
    const MlxQwen35LinearAttentionCacheSnapshot& snapshot,
    int boundary,
    std::uint8_t kind = kRecurrentLayer) {
    if (snapshot.batch != 1 || snapshot.position < boundary) {
        throw std::runtime_error("invalid recurrent cache boundary");
    }
    writer.scalar<std::uint8_t>(kind);
    writer.scalar<std::uint8_t>(0);
    writer.scalar<std::uint16_t>(0);
    writer.scalar<std::int32_t>(snapshot.batch);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(boundary);
    writer.tensor(snapshot.convolution_state);
    writer.tensor(snapshot.recurrent_state);
}

void write_recurrent_unavailable_layer(
    Writer& writer,
    const MlxQwen35LinearAttentionCacheSnapshot& snapshot,
    int boundary,
    std::uint8_t kind = kRecurrentUnavailableLayer) {
    if (snapshot.batch != 1 || snapshot.position < boundary) {
        throw std::runtime_error("invalid recurrent cache boundary");
    }
    writer.scalar<std::uint8_t>(kind);
    writer.scalar<std::uint8_t>(0);
    writer.scalar<std::uint16_t>(0);
    writer.scalar<std::int32_t>(snapshot.batch);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(boundary);
}

void write_optional_tensor(Writer& writer, const std::optional<array>& value) {
    writer.scalar<std::uint8_t>(value.has_value());
    if (value) writer.tensor(*value);
}

void write_flash_layer(Writer& writer, const MlxQwen4LayerCacheSnapshot& layer,
    std::size_t start, std::size_t count, bool exact) {
    const int boundary = static_cast<int>(start + count);
    if (layer.batch != 1 || layer.position < boundary ||
        ((layer.kv || layer.offloaded_kv) ? !layer.index_keys : (!layer.convolution || !layer.recurrent)))
        throw std::runtime_error("invalid Flash-Next cache layer");
    if (layer.kv || layer.offloaded_kv) {
        if (layer.offloaded_kv) {
            const auto& kv = *layer.offloaded_kv;
            writer.scalar<std::uint8_t>(kQsaLayer);
            writer.scalar<std::uint8_t>(static_cast<std::uint8_t>(kv.dtype.val()));
            writer.scalar<std::uint16_t>((exact ? 1 : 0) | (kv.quantization.enabled()
                ? 2 | (static_cast<int>(kv.quantization.bits * 2) << 8) : 0));
            writer.scalar<std::int32_t>(1);
            writer.scalar<std::int32_t>(kv.heads);
            writer.scalar<std::int32_t>(kv.maximum);
            writer.scalar<std::int32_t>(kv.dimension);
            writer.scalar<std::int32_t>(boundary);
            writer.scalar<std::int32_t>(boundary);
            writer.offloaded_kv_tensor(kv, start, count, false);
            writer.offloaded_kv_tensor(kv, start, count, true);
        } else write_kv_layer(writer, *layer.kv, static_cast<int>(start), static_cast<int>(count), kQsaLayer, exact);
        const int ratio = layer.index_ratio;
        if (ratio <= 0 || layer.index_start < 0 || layer.index_start % ratio ||
            layer.index_keys->shape() != Shape{1, layer.position - layer.index_start, layer.index_keys->shape(2)})
            throw std::runtime_error("invalid QSA index tail");
        writer.scalar<std::int32_t>(ratio);
        writer.scalar<std::int32_t>(exact ? layer.index_start : boundary);
        write_optional_tensor(writer, exact ? layer.index_keys : std::nullopt);
        const int pool_start = static_cast<int>(start) / ratio;
        const int pool_end = boundary / ratio;
        if (pool_end > pool_start && layer.offloaded_pooled_keys) {
            writer.scalar<std::uint8_t>(1);
            writer.offloaded_index_tensor(*layer.offloaded_pooled_keys, pool_start, pool_end - pool_start);
        } else if (pool_end > pool_start && layer.pooled_keys) {
            write_optional_tensor(writer, mlx::core::slice(*layer.pooled_keys, Shape{0, pool_start, 0},
                Shape{1, pool_end, layer.pooled_keys->shape(2)}));
        } else {
            if (pool_end > pool_start) throw std::runtime_error("missing QSA block keys");
            write_optional_tensor(writer, std::nullopt);
        }
    } else {
        MlxQwen35LinearAttentionCacheSnapshot state{*layer.convolution, *layer.recurrent,
            layer.position, layer.batch};
        if (exact) write_recurrent_layer(writer, state, boundary, kGdnLayer);
        else write_recurrent_unavailable_layer(writer, state, boundary, kGdnUnavailableLayer);
    }
    write_optional_tensor(writer, exact ? layer.ple_convolution : std::nullopt);
    writer.scalar<std::uint32_t>(exact ? static_cast<std::uint32_t>(layer.ple_context.size()) : 0);
    if (exact) for (const auto token : layer.ple_context) writer.scalar<std::int64_t>(token);
}

DecodedLayer read_layer(Reader& reader) {
    DecodedLayer layer;
    layer.kind = reader.scalar<std::uint8_t>("layer kind");
    const auto dtype_value = reader.scalar<std::uint8_t>("layer dtype");
    const auto flags = reader.scalar<std::uint16_t>("layer flags");
    layer.exact = (flags & 1) != 0;
    if (flags & 2) {
        layer.quantization.bits = (flags >> 8) / 2.0;
        if (!layer.quantization.enabled() || !mlx_kv_bits_valid(layer.quantization.bits))
            throw std::runtime_error("invalid TurboQuant KV checkpoint bit width");
    } else if (flags & 0xff00) throw std::runtime_error("invalid KV checkpoint quantization flags");
    layer.batch = reader.scalar<std::int32_t>("batch");
    layer.heads = reader.scalar<std::int32_t>("heads");
    layer.maximum_sequence = reader.scalar<std::int32_t>("maximum sequence");
    layer.head_dimension = reader.scalar<std::int32_t>("head dimension");
    layer.capacity = reader.scalar<std::int32_t>("capacity");
    layer.position = reader.scalar<std::int32_t>("position");
    const bool unavailable = layer.kind == kRecurrentUnavailableLayer || layer.kind == kGdnUnavailableLayer;
    if (!unavailable) { layer.first = reader.tensor(); layer.second = reader.tensor(); }
    if (layer.kind == kKvLayer || layer.kind == kQsaLayer) {
        layer.dtype = layer.first.dtype;
        if (layer.quantization.enabled() && layer.dtype != mlx::core::uint32)
            throw std::runtime_error("invalid TurboQuant KV checkpoint dtype");
        if (dtype_value > static_cast<std::uint8_t>(Dtype::Val::complex64) ||
            layer.dtype.val() != static_cast<Dtype::Val>(dtype_value))
            throw std::runtime_error("MLX KV layer dtype mismatch");
    } else if (unavailable || layer.kind == kRecurrentLayer || layer.kind == kGdnLayer) {
        if (dtype_value != 0 || layer.batch != 1 || layer.heads != 0 || layer.maximum_sequence != 0 ||
            layer.head_dimension != 0 || layer.capacity != 0 || layer.position <= 0 ||
            (!unavailable && (layer.first.dtype != mlx::core::float32 || layer.second.dtype != mlx::core::float32)))
            throw std::runtime_error("invalid recurrent cache checkpoint");
    } else throw std::runtime_error("unknown MLX cache layer kind");
    if (layer.kind == kQsaLayer) {
        layer.index_ratio = reader.scalar<std::int32_t>("index ratio");
        layer.index_start = reader.scalar<std::int32_t>("index tail start");
        if (layer.index_ratio <= 0 || layer.index_start < 0 || layer.index_start > layer.position)
            throw std::runtime_error("invalid QSA index tail geometry");
        if (reader.scalar<std::uint8_t>("index tail presence")) layer.index = reader.tensor();
        if (reader.scalar<std::uint8_t>("pooled presence")) layer.pooled = reader.tensor();
    }
    if (layer.kind == kQsaLayer || layer.kind == kGdnLayer || layer.kind == kGdnUnavailableLayer) {
        if (reader.scalar<std::uint8_t>("PLE presence")) layer.ple = reader.tensor();
        const auto count = reader.scalar<std::uint32_t>("PLE context count");
        if (count > 65536) throw std::runtime_error("invalid PLE context count");
        for (std::uint32_t i = 0; i < count; ++i)
            layer.ple_context.push_back(reader.scalar<std::int64_t>("PLE token"));
    }
    return layer;
}

DecodedBlock read_block(const std::vector<std::uint8_t>& payload) {
    Reader reader(payload);
    std::array<std::uint8_t, 8> magic{};
    reader.raw(magic.data(), magic.size(), "magic");
    const auto version = reader.scalar<std::uint32_t>("version");
    if (magic != kMagic || version < 1 || version > kDFlashVersion) {
        throw std::runtime_error("unsupported MLX cache payload");
    }
    DecodedBlock block;
    block.runtime = reader.scalar<std::uint32_t>("runtime");
    if (block.runtime == kQwen4Runtime && version < 3)
        throw std::runtime_error("obsolete Flash-Next index history cache");
    block.start = reader.scalar<std::uint32_t>("start");
    block.count = reader.scalar<std::uint32_t>("count");
    const auto layer_count = reader.scalar<std::uint32_t>("layer count");
    if (block.count == 0 || layer_count == 0 || layer_count > 4096) {
        throw std::runtime_error("invalid MLX cache block geometry");
    }
    block.layers.reserve(layer_count);
    for (std::uint32_t index = 0; index < layer_count; ++index) {
        block.layers.push_back(read_layer(reader));
    }
    if (version >= 2) {
        if (reader.scalar<std::uint8_t>("MTP hidden presence")) block.last_hidden = reader.tensor();
        const auto count = reader.scalar<std::uint32_t>("MTP layer count");
        if (count > 4096 || (count > 0 && !block.last_hidden))
            throw std::runtime_error("invalid MTP checkpoint");
        for (std::uint32_t i = 0; i < count; ++i) block.mtp_layers.push_back(read_layer(reader));
    }
    if (version == kDFlashVersion) {
        const auto count = reader.scalar<std::uint32_t>("DFlash2 ring layer count");
        if (block.runtime != kQwen4Runtime || count == 0 || count > 64)
            throw std::runtime_error("invalid DFlash2 cache checkpoint");
        for (std::uint32_t i = 0; i < count; ++i) {
            auto key = reader.tensor();
            auto value = reader.tensor();
            if (key.dtype != mlx::core::bfloat16 || value.dtype != key.dtype ||
                key.shape.size() != 4 || key.shape[0] != 1 || value.shape != key.shape)
                throw std::runtime_error("invalid DFlash2 cache ring");
            block.dflash.emplace_back(std::move(key), std::move(value));
        }
    }
    reader.expect_end();
    return block;
}

std::vector<DecodedBlock> decode_blocks(
    const std::vector<MlxPagedPayload>& payloads,
    std::uint32_t expected_runtime,
    std::size_t token_count,
    std::size_t block_size) {
    if (payloads.empty() || block_size == 0 || token_count == 0 ||
        (token_count + block_size - 1) / block_size != payloads.size()) {
        throw std::runtime_error("MLX cache block chain length mismatch");
    }
    std::vector<DecodedBlock> blocks;
    blocks.reserve(payloads.size());
    std::size_t expected_start = 0;
    std::size_t expected_layers = 0;
    for (const auto& payload : payloads) {
        if (!payload) {
            throw std::runtime_error("MLX cache block payload is null");
        }
        auto block = read_block(*payload);
        if (block.runtime != expected_runtime || block.start != expected_start ||
            block.count != std::min(block_size, token_count - expected_start) ||
            (expected_layers != 0 && block.layers.size() != expected_layers)) {
            throw std::runtime_error("incompatible MLX cache block chain");
        }
        const auto boundary = static_cast<std::uint64_t>(block.start) +
            static_cast<std::uint64_t>(block.count);
        if (boundary >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::int32_t>::max()) ||
            std::any_of(
                block.layers.begin(),
                block.layers.end(),
                [boundary](const DecodedLayer& layer) {
                    return layer.position != static_cast<int>(boundary);
                })) {
            throw std::runtime_error("MLX cache layer boundary mismatch");
        }
        expected_start += block.count;
        expected_layers = block.layers.size();
        blocks.push_back(std::move(block));
    }
    return blocks;
}

MlxKvCacheSnapshot rebuild_kv(
    const std::vector<DecodedBlock>& blocks,
    std::size_t layer_index,
    std::size_t token_count) {
    if (token_count > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error("MLX cache token count is too large");
    }
    const auto& final = blocks.back().layers.at(layer_index);
    if (final.batch <= 0 || final.heads <= 0 ||
        final.head_dimension <= 0) {
        throw std::runtime_error("invalid MLX KV cache topology");
    }
    const auto expected_shape = [&](std::size_t count, bool key) {
        return Shape{
            final.batch,
            final.heads,
            static_cast<int>(count),
            final.quantization.enabled() ? mlx_kv_packed_width(final.head_dimension,
                key ? final.quantization.key_bits() : final.quantization.value_bits()) : final.head_dimension,
        };
    };
    for (const auto& block : blocks) {
        const auto& layer = block.layers.at(layer_index);
        if ((layer.kind != kKvLayer && layer.kind != kQsaLayer) || layer.batch != final.batch ||
            layer.heads != final.heads ||
            layer.maximum_sequence != final.maximum_sequence ||
            layer.head_dimension != final.head_dimension ||
            layer.dtype != final.dtype ||
            layer.quantization != final.quantization ||
            layer.first.dtype != final.dtype ||
            layer.second.dtype != final.dtype ||
            layer.first.shape != expected_shape(block.count, true) ||
            layer.second.shape != expected_shape(block.count, false) ||
            layer.first.data == nullptr || layer.second.data == nullptr ||
            static_cast<std::size_t>(block.start) + block.count > token_count) {
            throw std::runtime_error("inconsistent MLX KV cache block topology");
        }
    }

    const auto element_bytes = final.dtype.size();
    const auto batch = static_cast<std::size_t>(final.batch);
    const auto heads = static_cast<std::size_t>(final.heads);
    const auto dimension = static_cast<std::size_t>(std::max(expected_shape(1, true)[3], expected_shape(1, false)[3]));
    if (batch > std::numeric_limits<std::size_t>::max() / heads ||
        token_count > std::numeric_limits<std::size_t>::max() / dimension ||
        token_count * dimension >
            std::numeric_limits<std::size_t>::max() / element_bytes) {
        throw std::runtime_error("MLX KV cache tensor size overflow");
    }
    const auto lanes = batch * heads;
    const auto target_lane_bytes = token_count *
        dimension * element_bytes;
    if (target_lane_bytes != 0 &&
        lanes > std::numeric_limits<std::size_t>::max() / target_lane_bytes) {
        throw std::runtime_error("MLX KV cache allocation size overflow");
    }
    auto rebuild = [&](bool key) {
        const auto target_shape = expected_shape(token_count, key);
        const auto dimension = static_cast<std::size_t>(target_shape[3]);
        const auto target_lane_bytes = token_count * dimension * element_bytes;
        const auto total_bytes = lanes * target_lane_bytes;
        MlxResidentBudgetScope::reserve(total_bytes);
        auto result = array(
            mlx::core::allocator::malloc(total_bytes),
            target_shape,
            final.dtype);
        auto* destination = result.data<std::uint8_t>();
        for (const auto& block : blocks) {
            const auto& tensor = key
                ? block.layers.at(layer_index).first
                : block.layers.at(layer_index).second;
            const auto block_lane_bytes =
                static_cast<std::size_t>(block.count) *
                dimension * element_bytes;
            if (tensor.bytes != lanes * block_lane_bytes) {
                throw std::runtime_error("invalid MLX KV cache tensor size");
            }
            const auto destination_offset =
                static_cast<std::size_t>(block.start) *
                dimension * element_bytes;
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                std::memcpy(
                    destination + lane * target_lane_bytes +
                        destination_offset,
                    tensor.data + lane * block_lane_bytes,
                    block_lane_bytes);
            }
        }
        return result;
    };
    auto key = rebuild(true);
    auto value = rebuild(false);
    const int position = static_cast<int>(token_count);
    return MlxKvCacheSnapshot{
        final.batch,
        final.heads,
        final.maximum_sequence,
        final.head_dimension,
        std::max(final.capacity, position),
        position,
        final.dtype,
        std::move(key),
        std::move(value),
        final.quantization,
    };
}

array copy_tensor(const SerializedTensor& source) {
    if (source.data == nullptr || source.bytes == 0) throw std::runtime_error("empty cache tensor");
    MlxResidentBudgetScope::reserve(source.bytes);
    auto result = array(mlx::core::allocator::malloc(source.bytes), source.shape, source.dtype);
    std::memcpy(result.data<std::uint8_t>(), source.data, source.bytes);
    return result;
}

MlxQwen4LayerCacheSnapshot rebuild_flash(const std::vector<DecodedBlock>& blocks,
    std::size_t index, std::size_t count) {
    const auto& final = blocks.back().layers.at(index);
    MlxQwen4LayerCacheSnapshot state;
    state.batch = final.batch;
    state.position = static_cast<int>(count);
    if (final.kind == kQsaLayer) {
        if (auto store = mlx_qsa_kv_offload_store()) {
            if (final.batch != 1 || final.heads <= 0 || final.head_dimension <= 0 ||
                (final.quantization.enabled() ? final.dtype != mlx::core::uint32 :
                    (final.dtype != mlx::core::float16 && final.dtype != mlx::core::bfloat16)))
                throw std::runtime_error("invalid offloaded QSA checkpoint geometry");
            const int key_width = final.quantization.enabled() ? mlx_kv_packed_width(final.head_dimension, final.quantization.key_bits()) : final.head_dimension;
            const int value_width = final.quantization.enabled() ? mlx_kv_packed_width(final.head_dimension, final.quantization.value_bits()) : final.head_dimension;
            const auto key_lane = static_cast<std::size_t>(key_width) * final.dtype.size();
            const auto value_lane = static_cast<std::size_t>(value_width) * final.dtype.size();
            const auto key_bytes = key_lane * final.heads, value_bytes = value_lane * final.heads;
            QsaKvSequence rows(store, key_bytes + value_bytes, store->config().microblock_rows, 1);
            std::vector<std::uint8_t> row(key_bytes + value_bytes);
            for (const auto& block : blocks) {
                const auto& layer = block.layers.at(index);
                if (block.start != rows.position() || block.start + block.count > count ||
                    layer.kind != kQsaLayer || layer.heads != final.heads ||
                    layer.head_dimension != final.head_dimension || layer.maximum_sequence != final.maximum_sequence ||
                    layer.quantization != final.quantization ||
                    layer.first.shape != Shape{1, final.heads, static_cast<int>(block.count), key_width} ||
                    layer.second.shape != Shape{1, final.heads, static_cast<int>(block.count), value_width} ||
                    layer.first.dtype != final.dtype || layer.second.dtype != final.dtype ||
                    layer.first.bytes != block.count * key_bytes || layer.second.bytes != block.count * value_bytes ||
                    !layer.first.data || !layer.second.data)
                    throw std::runtime_error("offloaded QSA checkpoint block topology mismatch");
                for (std::size_t token = 0; token < block.count; ++token) {
                    for (int head = 0; head < final.heads; ++head) {
                        std::memcpy(row.data() + head * key_lane, layer.first.data + (head * block.count + token) * key_lane, key_lane);
                        std::memcpy(row.data() + key_bytes + head * value_lane, layer.second.data + (head * block.count + token) * value_lane, value_lane);
                    }
                    rows.append(row.data(), 1);
                }
            }
            if (rows.position() != count) throw std::runtime_error("incomplete offloaded QSA checkpoint");
            rows.seal_tail();
            state.offloaded_kv = MlxQsaKvSnapshot{rows.snapshot(), final.heads, final.head_dimension,
                final.maximum_sequence, final.dtype, final.quantization};
        } else state.kv = rebuild_kv(blocks, index, count);
        if (!final.index || final.index->shape.size() != 3)
            throw std::runtime_error("missing QSA index checkpoint");
        const int width = final.index->shape[2];
        const int ratio = final.index_ratio;
        if (width <= 0 || ratio <= 0 || final.index_start < 0 || final.index_start % ratio ||
            final.index_start > static_cast<int>(count) ||
            final.index->shape != Shape{1, static_cast<int>(count) - final.index_start, width} ||
            final.index->dtype != mlx::core::float16 ||
            count - final.index_start > static_cast<std::size_t>(ratio) + MlxQwen4CausalLm::maximum_verification_rows)
            throw std::runtime_error("QSA index tail checkpoint mismatch");
        state.index_start = final.index_start;
        state.index_ratio = ratio;
        state.index_keys = copy_tensor(*final.index);
        const auto pool_count = count / ratio;
        if (pool_count > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(width) / 4)
            throw std::runtime_error("QSA block index checkpoint size overflow");
        std::optional<array> pooled;
        std::unique_ptr<QsaKvSequence> pool_rows;
        if (auto store = mlx_qsa_kv_offload_store())
            pool_rows = std::make_unique<QsaKvSequence>(store, width * 4, 16, 2);
        else if (pool_count) {
            MlxResidentBudgetScope::reserve(pool_count * width * 4);
            pooled = array(mlx::core::allocator::malloc(pool_count * width * 4),
                Shape{1, static_cast<int>(pool_count), width}, mlx::core::float32);
        }
        for (const auto& block : blocks) {
            const auto& layer = block.layers.at(index);
            const auto begin = block.start / ratio;
            const auto rows = (block.start + block.count) / ratio - begin;
            if (layer.index_ratio != ratio || (rows && (!layer.pooled ||
                layer.pooled->dtype != mlx::core::float32 ||
                layer.pooled->shape != Shape{1, static_cast<int>(rows), width})))
                throw std::runtime_error("QSA pooled index block topology mismatch");
            if (!rows) continue;
            if (pool_rows) {
                if (pool_rows->position() != begin) throw std::runtime_error("QSA block index chain mismatch");
                pool_rows->append(layer.pooled->data, rows);
            } else std::memcpy(pooled->data<std::uint8_t>() + begin * width * 4,
                layer.pooled->data, layer.pooled->bytes);
        }
        if (pool_rows && pool_count) {
            pool_rows->seal_tail();
            state.offloaded_pooled_keys = MlxQsaIndexSnapshot{pool_rows->snapshot(), width, mlx::core::float32};
        } else state.pooled_keys = std::move(pooled);
    } else if (final.kind == kGdnLayer) {
        state.convolution = copy_tensor(final.first);
        state.recurrent = copy_tensor(final.second);
    } else throw std::runtime_error("missing exact GDN checkpoint");
    if (final.ple) state.ple_convolution = copy_tensor(*final.ple);
    state.ple_context = final.ple_context;
    return state;
}

MlxQwen35LinearAttentionCacheSnapshot rebuild_recurrent(
    const std::vector<DecodedBlock>& blocks,
    std::size_t layer_index,
    std::size_t token_count) {
    const auto& final = blocks.back().layers.at(layer_index);
    if (final.kind != kRecurrentLayer || final.batch != 1 ||
        final.position != static_cast<int>(token_count) ||
        final.first.data == nullptr || final.second.data == nullptr) {
        throw std::runtime_error("invalid recurrent cache boundary block");
    }
    const auto copy = [](const SerializedTensor& source) {
        MlxResidentBudgetScope::reserve(source.bytes);
        auto result = array(
            mlx::core::allocator::malloc(source.bytes),
            source.shape,
            source.dtype);
        std::memcpy(result.data<std::uint8_t>(), source.data, source.bytes);
        return result;
    };
    return MlxQwen35LinearAttentionCacheSnapshot{
        copy(final.first),
        copy(final.second),
        static_cast<int>(token_count),
        final.batch,
    };
}

struct V41LayerBlock {
    int ratio = 0, capacity = 0, kv_width = 0, index_width = 0;
    Dtype dtype = mlx::core::uint8;
    std::optional<SerializedTensor> kv, index, local, partial, score;
};

struct V41Block {
    std::uint32_t start = 0, count = 0;
    bool exact = false;
    std::vector<V41LayerBlock> layers;
    std::vector<std::int64_t> context;
    std::vector<SerializedTensor> rings;
};

std::optional<SerializedTensor> read_optional_tensor(Reader& reader) {
    const auto present = reader.scalar<std::uint8_t>("tensor presence");
    if (present > 1) throw std::runtime_error("invalid cache tensor presence");
    return present ? std::optional<SerializedTensor>(reader.tensor()) : std::nullopt;
}

V41Block read_v41_block(const MlxPagedPayload& payload) {
    if (!payload) throw std::runtime_error("null DeepSeek-V4.1 cache payload");
    Reader reader(*payload);
    std::array<std::uint8_t, 8> magic{};
    reader.raw(magic.data(), magic.size(), "magic");
    if (magic != kMagic || reader.scalar<std::uint32_t>("version") != kVersion ||
        reader.scalar<std::uint32_t>("runtime") != kDeepseekV41Runtime)
        throw std::runtime_error("incompatible DeepSeek-V4.1 cache payload");
    V41Block block;
    block.start = reader.scalar<std::uint32_t>("start");
    block.count = reader.scalar<std::uint32_t>("count");
    const auto layers = reader.scalar<std::uint32_t>("layers");
    const auto exact = reader.scalar<std::uint8_t>("exact boundary");
    const auto boundary = std::uint64_t(block.start) + block.count;
    if (!block.count || !layers || layers > 4096 || exact > 1 ||
        boundary > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::runtime_error("invalid DeepSeek-V4.1 cache boundary");
    block.exact = exact != 0;
    for (std::uint32_t i = 0; i < layers; ++i) {
        V41LayerBlock layer;
        layer.ratio = reader.scalar<std::int32_t>("compression ratio");
        layer.capacity = reader.scalar<std::int32_t>("compression capacity");
        layer.kv_width = reader.scalar<std::int32_t>("KV width");
        layer.index_width = reader.scalar<std::int32_t>("Indexer width");
        const auto dtype = reader.scalar<std::uint8_t>("CSA2 dtype");
        if (dtype == static_cast<std::uint8_t>(mlx::core::uint8.val())) layer.dtype = mlx::core::uint8;
        else if (dtype == static_cast<std::uint8_t>(mlx::core::float16.val())) layer.dtype = mlx::core::float16;
        else if (dtype == static_cast<std::uint8_t>(mlx::core::bfloat16.val())) layer.dtype = mlx::core::bfloat16;
        else if (dtype == static_cast<std::uint8_t>(mlx::core::float32.val())) layer.dtype = mlx::core::float32;
        else throw std::runtime_error("invalid CSA2 checkpoint dtype");
        if (layer.ratio < 0 || layer.capacity < 0 || layer.kv_width < 0 || layer.index_width < 0 ||
            (layer.ratio == 0 ? (layer.capacity || layer.kv_width || layer.index_width)
                : (!layer.capacity || !layer.kv_width || !layer.index_width || boundary / layer.ratio > std::uint64_t(layer.capacity))))
            throw std::runtime_error("invalid CSA2 checkpoint geometry");
        layer.kv = read_optional_tensor(reader);
        layer.index = read_optional_tensor(reader);
        layer.local = read_optional_tensor(reader);
        layer.partial = read_optional_tensor(reader);
        layer.score = read_optional_tensor(reader);
        const int rows = layer.ratio ? int(boundary / layer.ratio - block.start / layer.ratio) : 0;
        const auto range_valid = [&](const auto& tensor, int width) {
            return tensor.has_value() == (rows > 0) && (!tensor ||
                (tensor->shape == Shape{1, rows, width} && tensor->dtype == layer.dtype));
        };
        if (!range_valid(layer.kv, layer.kv_width) || !range_valid(layer.index, layer.index_width) ||
            layer.local.has_value() != block.exact || (layer.local &&
                (layer.local->shape.size() != 3 || layer.local->shape[0] != 1)) ||
            layer.partial.has_value() != (block.exact && layer.ratio > 1) ||
            layer.score.has_value() != layer.partial.has_value() || (layer.partial &&
                (layer.partial->shape.size() != 3 || layer.partial->shape[0] != 1 ||
                 layer.partial->shape[1] != layer.ratio || layer.partial->shape != layer.score->shape ||
                 layer.partial->dtype != mlx::core::float32 || layer.score->dtype != mlx::core::float32)))
            throw std::runtime_error("invalid CSA2 checkpoint tensors");
        block.layers.push_back(std::move(layer));
    }
    const auto context = reader.scalar<std::uint32_t>("Engram context size");
    if (context > 65536 || (!block.exact && context))
        throw std::runtime_error("invalid Engram checkpoint history");
    for (std::uint32_t i = 0; i < context; ++i)
        block.context.push_back(reader.scalar<std::int64_t>("Engram history"));
    const auto stages = reader.scalar<std::uint32_t>("DSpark stages");
    if (stages > 4096 || (!block.exact && stages))
        throw std::runtime_error("invalid DSpark checkpoint stage count");
    for (std::uint32_t i = 0; i < stages; ++i) block.rings.push_back(reader.tensor());
    reader.expect_end();
    return block;
}

template <typename State, typename LayerWriter>
std::vector<MlxPagedPayload> encode_state(
    const State& state,
    std::size_t block_size,
    std::size_t first_block,
    std::size_t maximum_blocks,
    std::uint32_t runtime,
    LayerWriter write_layer,
    bool allow_tail = false) {
    if (block_size == 0 || state.cache_batch != 1 ||
        state.cache_position <= 0 ||
        state.tokens.size() != static_cast<std::size_t>(state.cache_position) ||
        state.layers.empty()) {
        throw std::runtime_error("invalid MLX session state for paging");
    }
    const auto full_blocks = allow_tail ? (state.tokens.size() + block_size - 1) / block_size
        : state.tokens.size() / block_size;
    if (first_block > full_blocks) {
        throw std::runtime_error("invalid MLX cache first block");
    }
    const auto end_block = maximum_blocks > full_blocks - first_block
        ? full_blocks
        : first_block + maximum_blocks;
    std::vector<MlxPagedPayload> result;
    result.reserve(end_block - first_block);
    for (std::size_t block = first_block; block < end_block; ++block) {
        const auto start = block * block_size;
        const auto count = std::min(block_size, state.tokens.size() - start);
        const bool exact_boundary = start + count == state.tokens.size();
        const bool dflash = [&] {
            if constexpr (requires { state.dflash; }) return exact_boundary && state.dflash.has_value();
            else return false;
        }();
        Writer writer;
        write_header(
            writer,
            runtime,
            static_cast<std::uint32_t>(start),
            static_cast<std::uint32_t>(count),
            state.layers.size(),
            dflash ? kDFlashVersion : kVersion);
        for (const auto& layer : state.layers) {
            write_layer(
                writer,
                layer,
                start,
                count,
                exact_boundary);
        }
        if constexpr (requires { state.mtp_layers; state.last_hidden; }) {
            write_optional_tensor(writer, exact_boundary ? state.last_hidden : std::nullopt);
            writer.scalar<std::uint32_t>(exact_boundary && state.last_hidden
                ? static_cast<std::uint32_t>(state.mtp_layers.size()) : 0);
            if (exact_boundary && state.last_hidden) for (const auto& layer : state.mtp_layers) {
                if constexpr (std::is_same_v<std::decay_t<decltype(layer)>, MlxKvCacheSnapshot>)
                    write_kv_layer(writer, layer, 0, layer.position);
                else write_flash_layer(writer, layer, 0, layer.position, true);
            }
        } else {
            writer.scalar<std::uint8_t>(0);
            writer.scalar<std::uint32_t>(0);
        }
        if constexpr (requires { state.dflash; }) {
            if (dflash) {
                const auto& snapshot = *state.dflash;
                if (snapshot.position != state.cache_position || snapshot.layers.empty() || snapshot.layers.size() > 64)
                    throw std::runtime_error("invalid DFlash2 cache boundary");
                writer.scalar<std::uint32_t>(static_cast<std::uint32_t>(snapshot.layers.size()));
                for (const auto& layer : snapshot.layers) {
                    if (!layer.keys || !layer.values) throw std::runtime_error("missing DFlash2 cache ring");
                    writer.tensor(*layer.keys);
                    writer.tensor(*layer.values);
                }
            }
        }
        result.push_back(std::move(writer).finish());
    }
    return result;
}

} // namespace

MlxPagedPayload MlxPagedSessionCodec<MlxDeepseekV41TextSessionState>::encode_block(
    const MlxDeepseekV41TextSessionState& state, std::size_t block_size, std::size_t index) {
    if (!block_size || state.cache_batch != 1 || state.cache_position <= 0 ||
        state.tokens.size() != std::size_t(state.cache_position) || state.layers.empty() ||
        index > (state.tokens.size() - 1) / block_size ||
        state.engram_hash.batch != 1 || state.engram_hash.position != state.cache_position)
        throw std::runtime_error("invalid DeepSeek-V4.1 session for paging");
    const auto start = index * block_size;
    const auto count = std::min(block_size, state.tokens.size() - start);
    const auto end = start + count;
    const bool exact = end == state.tokens.size();
    Writer writer;
    write_header(writer, kDeepseekV41Runtime, start, count, state.layers.size());
    writer.scalar<std::uint8_t>(exact);
    for (const auto& entry : state.layers) {
        const auto& layer = entry.attention;
        const int ratio = layer.compressed_kv ? (layer.partial_kv ? layer.partial_kv->shape(1) : 1) : 0;
        if (layer.position != state.cache_position || layer.speculation ||
            layer.compressed_kv.has_value() != layer.index_k.has_value() ||
            layer.partial_kv.has_value() != layer.partial_score.has_value() ||
            layer.compressed_length != (ratio ? state.cache_position / ratio : 0) ||
            layer.partial_length != (ratio ? state.cache_position % ratio : 0))
            throw std::runtime_error("inconsistent DeepSeek-V4.1 checkpoint state");
        const int capacity = ratio ? layer.compressed_kv->shape(1) : 0;
        const int kv_width = ratio ? layer.compressed_kv->shape(2) : 0;
        const int index_width = ratio ? layer.index_k->shape(2) : 0;
        const auto dtype = ratio ? layer.compressed_kv->dtype() : mlx::core::uint8;
        if (ratio && (layer.compressed_kv->shape() != Shape{1, capacity, kv_width} ||
            layer.index_k->shape() != Shape{1, capacity, index_width} || layer.index_k->dtype() != dtype))
            throw std::runtime_error("invalid DeepSeek-V4.1 compressed cache shape");
        for (int value : {ratio, capacity, kv_width, index_width}) writer.scalar<std::int32_t>(value);
        writer.scalar<std::uint8_t>(static_cast<std::uint8_t>(dtype.val()));
        const auto range = [&](const std::optional<array>& value) -> std::optional<array> {
            if (!ratio || start / ratio == end / ratio) return std::nullopt;
            return mlx::core::slice(*value, Shape{0, int(start / ratio), 0},
                Shape{1, int(end / ratio), value->shape(2)});
        };
        write_optional_tensor(writer, range(layer.compressed_kv));
        write_optional_tensor(writer, range(layer.index_k));
        write_optional_tensor(writer, exact ? std::optional<array>(layer.local_kv) : std::nullopt);
        write_optional_tensor(writer, exact ? layer.partial_kv : std::nullopt);
        write_optional_tensor(writer, exact ? layer.partial_score : std::nullopt);
    }
    writer.scalar<std::uint32_t>(exact ? state.engram_hash.context.size() : 0);
    if (exact) for (auto token : state.engram_hash.context) writer.scalar<std::int64_t>(token);
    if (exact && state.dspark && state.dspark->position() != state.cache_position)
        throw std::runtime_error("DSpark checkpoint position mismatch");
    writer.scalar<std::uint32_t>(exact && state.dspark ? state.dspark->stages() : 0);
    if (exact && state.dspark) for (std::size_t stage = 0; stage < state.dspark->stages(); ++stage)
        writer.tensor(state.dspark->ring(stage));
    return std::move(writer).finish();
}

MlxDeepseekV41TextSessionState MlxPagedSessionCodec<MlxDeepseekV41TextSessionState>::decode(
    const std::vector<MlxPagedPayload>& payloads, const std::vector<std::int64_t>& tokens,
    std::size_t block_size) {
    if (!block_size || tokens.empty() || tokens.size() > std::size_t(std::numeric_limits<int>::max()) ||
        payloads.size() != (tokens.size() + block_size - 1) / block_size)
        throw std::runtime_error("invalid DeepSeek-V4.1 cache chain length");
    std::vector<V41Block> blocks;
    for (std::size_t i = 0; i < payloads.size(); ++i) {
        auto block = read_v41_block(payloads[i]);
        if (block.start != i * block_size || block.count != std::min(block_size, tokens.size() - i * block_size) ||
            (!blocks.empty() && block.layers.size() != blocks.front().layers.size()))
            throw std::runtime_error("incompatible DeepSeek-V4.1 cache chain");
        blocks.push_back(std::move(block));
    }
    const auto& final = blocks.back();
    if (!final.exact) throw std::runtime_error("DeepSeek-V4.1 cache has no exact checkpoint");
    MlxDeepseekV41TextSessionState state;
    state.tokens = tokens;
    state.cache_batch = 1;
    state.cache_position = static_cast<int>(tokens.size());
    state.engram_hash = {1, state.cache_position, final.context};
    state.bytes = final.context.size() * sizeof(std::int64_t);
    for (std::size_t index = 0; index < final.layers.size(); ++index) {
        const auto& layout = final.layers[index];
        for (const auto& block : blocks) {
            const auto& part = block.layers[index];
            if (part.ratio != layout.ratio || part.capacity != layout.capacity || part.kv_width != layout.kv_width ||
                part.index_width != layout.index_width || part.dtype != layout.dtype)
                throw std::runtime_error("inconsistent CSA2 cache chain layout");
        }
        MlxDeepseekV41AttentionState layer;
        layer.position = state.cache_position;
        layer.compressed_length = layout.ratio ? state.cache_position / layout.ratio : 0;
        layer.partial_length = layout.ratio ? state.cache_position % layout.ratio : 0;
        layer.local_kv = copy_tensor(*layout.local);
        if (layout.partial) layer.partial_kv = copy_tensor(*layout.partial);
        if (layout.score) layer.partial_score = copy_tensor(*layout.score);
        const auto rebuild = [&](bool kv) {
            const int width = kv ? layout.kv_width : layout.index_width;
            const auto lane = std::size_t(width) * layout.dtype.size();
            if (std::size_t(layout.capacity) > std::numeric_limits<std::size_t>::max() / lane)
                throw std::runtime_error("CSA2 checkpoint allocation overflow");
            const auto bytes = lane * layout.capacity;
            MlxResidentBudgetScope::reserve(bytes);
            auto value = array(mlx::core::allocator::malloc(bytes), Shape{1, layout.capacity, width}, layout.dtype);
            auto* destination = value.data<std::uint8_t>();
            std::memset(destination, 0, bytes);
            for (const auto& block : blocks) {
                const auto& part = block.layers[index];
                const auto& source = kv ? part.kv : part.index;
                if (source) std::memcpy(destination + (block.start / layout.ratio) * lane, source->data, source->bytes);
            }
            return value;
        };
        if (layout.ratio) {
            layer.compressed_kv = rebuild(true);
            layer.index_k = rebuild(false);
        }
        state.bytes += layer.nbytes();
        state.layers.push_back({std::move(layer)});
    }
    if (!final.rings.empty()) {
        std::vector<array> rings;
        for (const auto& ring : final.rings) rings.push_back(copy_tensor(ring));
        state.dspark = MlxDeepseekV41DSparkState::from_snapshot(std::move(rings), state.cache_position);
        state.bytes += state.dspark->nbytes();
    }
    return state;
}

bool MlxPagedSessionCodec<MlxDeepseekV41TextSessionState>::has_exact_boundary(const MlxPagedPayload& payload) {
    return read_v41_block(payload).exact;
}

bool MlxPagedSessionCodec<MlxDeepseekV41TextSessionState>::has_mtp(const MlxPagedPayload& payload) {
    const auto block = read_v41_block(payload);
    return block.exact && !block.rings.empty();
}

std::size_t MlxPagedSessionCodec<MlxDeepseekV41TextSessionState>::token_count(const MlxPagedPayload& payload) {
    const auto block = read_v41_block(payload);
    return std::size_t(block.start) + block.count;
}

std::size_t MlxPagedSessionCodec<MlxDeepseekV41TextSessionState>::decodable_blocks(
    const std::vector<MlxPagedPayload>& payloads) {
    for (std::size_t count = payloads.size(); count > 0; --count)
        if (has_exact_boundary(payloads[count - 1])) return count;
    return 0;
}

std::vector<MlxPagedPayload>
MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::encode(
    const MlxMiniCPMO45TextSessionState& state,
    std::size_t block_size,
    std::size_t first_block) {
    return encode_state(
        state,
        block_size,
        first_block,
        std::numeric_limits<std::size_t>::max(),
        kMiniRuntime,
        [](Writer& writer, const MlxKvCacheSnapshot& layer,
           std::size_t start, std::size_t count, bool) {
            write_kv_layer(
                writer,
                layer,
                static_cast<int>(start),
                static_cast<int>(count));
        });
}

MlxPagedPayload
MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::encode_block(
    const MlxMiniCPMO45TextSessionState& state,
    std::size_t block_size,
    std::size_t block_index) {
    auto result = encode_state(
        state,
        block_size,
        block_index,
        1,
        kMiniRuntime,
        [](Writer& writer, const MlxKvCacheSnapshot& layer,
           std::size_t start, std::size_t count, bool) {
            write_kv_layer(
                writer,
                layer,
                static_cast<int>(start),
                static_cast<int>(count));
        });
    if (result.size() != 1) {
        throw std::runtime_error("invalid MiniCPM cache block index");
    }
    return std::move(result.front());
}

MlxMiniCPMO45TextSessionState
MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::decode(
    const std::vector<MlxPagedPayload>& payloads,
    const std::vector<std::int64_t>& tokens,
    std::size_t block_size) {
    auto blocks = decode_blocks(
        payloads, kMiniRuntime, tokens.size(), block_size);
    MlxMiniCPMO45TextSessionState state;
    state.tokens = tokens;
    state.cache_position = static_cast<int>(tokens.size());
    state.cache_batch = 1;
    state.layers.reserve(blocks.front().layers.size());
    for (std::size_t index = 0; index < blocks.front().layers.size(); ++index) {
        auto layer = rebuild_kv(blocks, index, tokens.size());
        state.bytes += layer.nbytes();
        state.layers.push_back(std::move(layer));
    }
    return state;
}

MlxPagedPayload
MlxPagedSessionCodec<MlxInkling2TextSessionState>::encode_block(
    const MlxInkling2TextSessionState& state, std::size_t block_size, std::size_t index) {
    const auto position = state.tokens.size();
    if (!position || position > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        !block_size || index > (position - 1) / block_size || state.window <= 0 ||
        state.visits.empty() || state.visits.size() > 4096 || !state.last_logits)
        throw std::runtime_error("invalid Inkling prefix checkpoint");

    const auto start = index * block_size;
    const auto count = std::min(block_size, position - start);
    const auto end = start + count;
    const bool exact = end == position;
    Writer writer;
    write_header(writer, kInkling2Runtime, static_cast<std::uint32_t>(start),
        static_cast<std::uint32_t>(count), state.visits.size());
    writer.scalar<std::int32_t>(state.window);
    writer.scalar<std::uint8_t>(exact ? 1 : 0);
    for (const auto& layer : state.visits) {
        if (layer.global.has_value() == layer.window.has_value())
            throw std::runtime_error("Inkling prefix visit needs exactly one KV store");
        writer.scalar<std::uint8_t>(layer.global ? 1 : 0);
        if (layer.global) {
            write_kv_layer(writer, *layer.global, static_cast<int>(start), static_cast<int>(count));
        } else {
            const auto& window = *layer.window;
            const auto length = std::min(position, static_cast<std::size_t>(state.window));
            if (window.ndim() != 3 || window.shape(0) != 1 ||
                window.shape(1) != static_cast<int>(length))
                throw std::runtime_error("invalid Inkling prefix window");
            const auto origin = position - length;
            const auto row_start = std::max(origin, start);
            const bool visible = row_start < end;
            writer.scalar<std::uint8_t>(visible ? 1 : 0);
            if (visible) {
                writer.scalar<std::uint32_t>(static_cast<std::uint32_t>(row_start));
                writer.tensor(mlx::core::slice(window,
                    {0, static_cast<int>(row_start - origin), 0},
                    {1, static_cast<int>(end - origin), window.shape(2)}));
            }
        }
        if (exact) for (const auto& history : layer.convolution) writer.tensor(history);
    }
    if (exact) writer.tensor(*state.last_logits);
    return std::move(writer).finish();
}

MlxInkling2TextSessionState
MlxPagedSessionCodec<MlxInkling2TextSessionState>::decode(
    const std::vector<MlxPagedPayload>& payloads, const std::vector<std::int64_t>& tokens,
    std::size_t block_size) {
    if (tokens.empty() || tokens.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        !block_size || (tokens.size() - 1) / block_size + 1 != payloads.size())
        throw std::runtime_error("invalid Inkling prefix chain length");
    MlxInkling2TextSessionState state;
    state.tokens = tokens;
    std::vector<DecodedBlock> blocks;
    std::vector<std::vector<array>> local;
    std::vector<std::size_t> local_next;
    for (std::size_t index = 0; index < payloads.size(); ++index) {
        if (!payloads[index]) throw std::runtime_error("missing Inkling prefix block");
        Reader reader(*payloads[index]);
        std::array<std::uint8_t, 8> magic{};
        reader.raw(magic.data(), magic.size(), "magic");
        if (magic != kMagic || reader.scalar<std::uint32_t>("version") != kVersion ||
            reader.scalar<std::uint32_t>("runtime") != kInkling2Runtime)
            throw std::runtime_error("invalid Inkling prefix format");
        DecodedBlock block;
        block.start = reader.scalar<std::uint32_t>("start");
        block.count = reader.scalar<std::uint32_t>("count");
        const auto layers = reader.scalar<std::uint32_t>("visits");
        const auto window = reader.scalar<std::int32_t>("window");
        const bool exact = reader.scalar<std::uint8_t>("checkpoint") != 0;
        const auto start = index * block_size;
        const auto end = start + std::min(block_size, tokens.size() - start);
        if (block.start != start || block.count != end - start || !layers || layers > 4096 ||
            window <= 0 || (index && (window != state.window || layers != local.size())) ||
            (end == tokens.size() && !exact))
            throw std::runtime_error("inconsistent Inkling prefix geometry");
        const auto need = tokens.size() - std::min(tokens.size(), static_cast<std::size_t>(window));
        if (!index) {
            state.window = window;
            local.resize(layers);
            local_next.assign(layers, need);
        }
        block.layers.resize(layers);
        for (std::uint32_t visit = 0; visit < layers; ++visit) {
            const bool global = reader.scalar<std::uint8_t>("KV kind") != 0;
            if (index && (blocks.front().layers[visit].kind == kKvLayer) != global)
                throw std::runtime_error("Inkling KV kind changed across the prefix");
            if (global) {
                block.layers[visit] = read_layer(reader);
                if (block.layers[visit].kind != kKvLayer || block.layers[visit].position != static_cast<int>(end))
                    throw std::runtime_error("invalid Inkling global KV block");
            } else if (reader.scalar<std::uint8_t>("window rows present")) {
                const auto row_start = reader.scalar<std::uint32_t>("window start");
                auto slice = reader.tensor();
                if (row_start < start || row_start >= end || slice.shape.size() != 3 ||
                    slice.shape[0] != 1 || slice.shape[1] != static_cast<int>(end - row_start))
                    throw std::runtime_error("invalid Inkling window page rows");
                if (end > need) {
                    const auto taken = std::max<std::size_t>(row_start, need);
                    if (taken != local_next[visit]) throw std::runtime_error("Inkling window page gap");
                    auto data = copy_tensor(slice);
                    if (taken != row_start)
                        data = mlx::core::slice(data, {0, static_cast<int>(taken - row_start), 0}, data.shape());
                    local[visit].push_back(std::move(data));
                    local_next[visit] = end;
                }
            }
            if (exact) {
                if (end == tokens.size()) {
                    std::array<array, 4> conv{
                        copy_tensor(reader.tensor()), copy_tensor(reader.tensor()),
                        copy_tensor(reader.tensor()), copy_tensor(reader.tensor())};
                    for (const auto& item : conv) state.bytes += item.nbytes();
                    state.visits.push_back({std::nullopt, std::nullopt, std::move(conv)});
                } else {
                    for (int tap = 0; tap < 4; ++tap) (void)reader.tensor();
                }
            }
        }
        if (exact) {
            const auto logits = reader.tensor();
            if (end == tokens.size()) {
                state.last_logits = copy_tensor(logits);
                state.bytes += state.last_logits->nbytes();
            }
        }
        reader.expect_end();
        blocks.push_back(std::move(block));
    }
    for (std::size_t i = 0; i < state.visits.size(); ++i) {
        auto& visit = state.visits[i];
        if (blocks.front().layers[i].kind == kKvLayer) {
            visit.global = rebuild_kv(blocks, i, tokens.size());
            state.bytes += visit.global->nbytes();
        } else {
            if (local[i].empty() || local_next[i] != tokens.size())
                throw std::runtime_error("Inkling prefix window is incomplete");
            visit.window = local[i].size() == 1 ? local[i].front() : mlx::core::concatenate(local[i], 1);
            state.bytes += visit.window->nbytes();
        }
    }
    return state;
}

namespace {
void write_hy4_rows(Writer& writer, const MlxHy4RowSnapshot& snapshot, int start, int count) {
    if (snapshot.offloaded) {
        writer.offloaded_index_tensor(*snapshot.offloaded, start, count);
    } else if (snapshot.resident) {
        const auto& rows = *snapshot.resident;
        writer.tensor(mlx::core::slice(rows, {0, start, 0}, {1, start + count, rows.shape(-1)}));
    } else {
        throw std::runtime_error("Hy4 prefix page is missing rows");
    }
}

struct Hy4PrefixRows {
    std::size_t position = 0;
    int width = 0;
    std::optional<array> resident;
    std::optional<QsaKvSequence> offloaded;

    void append(const SerializedTensor& tensor, std::size_t offset, std::size_t skip,
        std::size_t count, std::size_t total, Dtype dtype, int priority,
        const std::shared_ptr<QsaKvStore>& store) {
        if (tensor.shape.size() != 3 || tensor.shape[0] != 1 || tensor.dtype != dtype ||
            static_cast<std::size_t>(tensor.shape[1]) != skip + count ||
            position != offset || count > total - offset || (width && width != tensor.shape[2]))
            throw std::runtime_error("Hy4 prefix row chain geometry mismatch");
        width = tensor.shape[2];
        const auto stride = static_cast<std::size_t>(width) * dtype.size();
        const auto* data = tensor.data + skip * stride;
        if (store) {
            if (!offloaded) offloaded.emplace(store, stride, 16, priority);
            offloaded->append(data, count);
        } else {
            if (!resident) resident.emplace(mlx::core::allocator::malloc(total * stride),
                Shape{1, static_cast<int>(total), width}, dtype);
            std::memcpy(resident->data<std::uint8_t>() + offset * stride, data, count * stride);
        }
        position += count;
    }

    MlxHy4RowSnapshot finish(std::size_t total, Dtype dtype, std::size_t& bytes) {
        if (position != total) throw std::runtime_error("incomplete Hy4 prefix row chain");
        if (offloaded) {
            offloaded->seal_tail();
            auto snapshot = offloaded->snapshot();
            bytes += snapshot.tail.size() + snapshot.blocks.size() * sizeof(QsaKvStore::BlockPtr);
            return {std::nullopt, MlxQsaIndexSnapshot{std::move(snapshot), width, dtype}};
        }
        if (resident) bytes += resident->nbytes();
        return {std::move(resident), std::nullopt};
    }
};
}  // namespace

MlxPagedPayload MlxPagedSessionCodec<MlxHy4TextSessionState>::encode_block(
    const MlxHy4TextSessionState& state, std::size_t block_size, std::size_t index) {
    const auto position = state.tokens.size();
    if (!position || position > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        !block_size || index > (position - 1) / block_size || state.window <= 0 || state.group_size <= 1 ||
        state.layers.empty() || state.layers.size() > 4096 || !state.last_hidden || !state.last_logits)
        throw std::runtime_error("invalid Hy4 prefix checkpoint");
    const auto start = index * block_size;
    const auto count = std::min(block_size, position - start);
    const auto end = start + count;
    const bool exact = end == position;
    Writer writer;
    write_header(writer, kHy4Runtime, static_cast<std::uint32_t>(start),
        static_cast<std::uint32_t>(count), state.layers.size());
    writer.scalar<std::int32_t>(state.window);
    writer.scalar<std::int32_t>(state.group_size);
    writer.scalar<std::uint8_t>(exact);
    for (const auto& layer : state.layers) {
        writer.scalar<std::uint8_t>(layer.window.has_value());
        if (layer.window) {
            const auto length = std::min(position, static_cast<std::size_t>(state.window));
            const auto first = position - length;
            const auto visible = std::max(start, first);
            writer.scalar<std::uint8_t>(visible < end);
            if (visible < end) {
                writer.scalar<std::uint32_t>(static_cast<std::uint32_t>(visible));
                writer.tensor(mlx::core::slice(*layer.window,
                    {0, static_cast<int>(visible - first), 0},
                    {1, static_cast<int>(end - first), layer.window->shape(-1)}));
            }
            continue;
        }
        write_hy4_rows(writer, layer.latent, static_cast<int>(start), static_cast<int>(count));
        const auto first = start / state.group_size;
        const auto complete = end / state.group_size - first;
        if (complete) write_hy4_rows(writer, layer.groups, static_cast<int>(first), static_cast<int>(complete));
        if (exact) {
            if (bool(layer.index_tail) != bool(end % state.group_size) || !layer.last_indices)
                throw std::runtime_error("invalid Hy4 prefix selection checkpoint");
            if (layer.index_tail) writer.tensor(*layer.index_tail);
            writer.tensor(*layer.last_indices);
        }
    }
    if (exact) {
        writer.tensor(*state.last_hidden);
        writer.tensor(*state.last_logits);
    }
    return std::move(writer).finish();
}

MlxHy4TextSessionState MlxPagedSessionCodec<MlxHy4TextSessionState>::decode(
    const std::vector<MlxPagedPayload>& payloads, const std::vector<std::int64_t>& tokens, std::size_t block_size) {
    const auto position = tokens.size();
    if (!position || position > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        !block_size || (position - 1) / block_size + 1 != payloads.size())
        throw std::runtime_error("invalid Hy4 prefix chain length");

    MlxHy4TextSessionState state;
    state.tokens = tokens;
    state.bytes = position * sizeof(tokens.front());
    const auto store = mlx_qsa_kv_offload_store();
    struct LayerRows {
        bool sliding = false;
        Hy4PrefixRows latent, groups;
    };
    std::vector<LayerRows> rows;
    for (std::size_t index = 0; index < payloads.size(); ++index) {
        if (!payloads[index]) throw std::runtime_error("missing Hy4 prefix block");
        Reader reader(*payloads[index]);
        std::array<std::uint8_t, 8> magic{};
        reader.raw(magic.data(), magic.size(), "magic");
        if (magic != kMagic || reader.scalar<std::uint32_t>("version") != kVersion ||
            reader.scalar<std::uint32_t>("runtime") != kHy4Runtime)
            throw std::runtime_error("invalid Hy4 prefix format");
        const auto start = index * block_size, end = start + std::min(block_size, position - start);
        const auto page_start = reader.scalar<std::uint32_t>("start");
        const auto page_count = reader.scalar<std::uint32_t>("count");
        const auto layers = reader.scalar<std::uint32_t>("layers");
        const auto window = reader.scalar<std::int32_t>("window");
        const auto ratio = reader.scalar<std::int32_t>("group size");
        const bool exact = reader.scalar<std::uint8_t>("checkpoint");
        if (page_start != start || page_count != end - start || !layers || layers > 4096 ||
            window <= 0 || ratio <= 1 || (end == position && !exact) ||
            (index && (layers != rows.size() || window != state.window || ratio != state.group_size)))
            throw std::runtime_error("inconsistent Hy4 prefix geometry");

        if (!index) {
            state.window = window;
            state.group_size = ratio;
            state.layers.resize(layers);
            rows.resize(layers);
        }
        const auto length = std::min(position, static_cast<std::size_t>(window));
        const auto first_window = position - length;
        for (std::uint32_t i = 0; i < layers; ++i) {
            auto& output = state.layers[i];
            auto& layer = rows[i];
            const bool sliding = reader.scalar<std::uint8_t>("KV kind");
            if (!index) layer.sliding = sliding;
            else if (layer.sliding != sliding) throw std::runtime_error("Hy4 prefix KV kind changed");
            if (sliding) {
                if (reader.scalar<std::uint8_t>("window rows present")) {
                    const auto from = reader.scalar<std::uint32_t>("window row start");
                    const auto tensor = reader.tensor();
                    if (from < start || from >= end)
                        throw std::runtime_error("Hy4 prefix window outside its page");
                    if (end > first_window) {
                        const auto visible = std::max<std::size_t>(first_window, from);
                        layer.latent.append(tensor, visible - first_window, visible - from, end - visible,
                            length, mlx::core::float16, 0, {});
                    }
                }
                continue;
            }
            layer.latent.append(reader.tensor(), start, 0, end - start, position,
                mlx::core::float16, 1, store);
            const auto first_group = start / ratio, groups = end / ratio - first_group;
            if (groups) layer.groups.append(reader.tensor(), first_group, 0, groups, position / ratio,
                mlx::core::float32, 0, store);
            if (exact) {
                std::optional<SerializedTensor> tail;
                if (end % ratio) tail = reader.tensor();
                const auto indices = reader.tensor();
                if (end == position) {
                    if (tail) output.index_tail = copy_tensor(*tail);
                    output.last_indices = copy_tensor(indices);
                    state.bytes += indices.bytes + (tail ? tail->bytes : 0);
                }
            }
        }
        if (exact) {
            const auto hidden = reader.tensor();
            const auto logits = reader.tensor();
            if (end == position) {
                state.last_hidden = copy_tensor(hidden);
                state.last_logits = copy_tensor(logits);
                state.bytes += hidden.bytes + logits.bytes;
            }
        }
        reader.expect_end();
    }
    for (std::size_t i = 0; i < rows.size(); ++i) {
        auto& layer = rows[i];
        auto& output = state.layers[i];
        if (layer.sliding) {
            output.window = layer.latent.finish(std::min(position, static_cast<std::size_t>(state.window)),
                mlx::core::float16, state.bytes).resident;
        } else {
            output.latent = layer.latent.finish(position, mlx::core::float16, state.bytes);
            output.groups = layer.groups.finish(position / state.group_size, mlx::core::float32, state.bytes);
        }
    }
    return state;
}

std::vector<MlxPagedPayload>
MlxPagedSessionCodec<MlxQwen35TextSessionState>::encode(
    const MlxQwen35TextSessionState& state,
    std::size_t block_size,
    std::size_t first_block) {
    return encode_state(
        state,
        block_size,
        first_block,
        std::numeric_limits<std::size_t>::max(),
        kQwen35Runtime,
        [](Writer& writer, const MlxQwen35LayerCacheSnapshot& layer,
           std::size_t start, std::size_t count, bool exact_boundary) {
            std::visit(
                [&](const auto& snapshot) {
                    using Snapshot = std::decay_t<decltype(snapshot)>;
                    if constexpr (std::is_same_v<Snapshot, MlxKvCacheSnapshot>) {
                        write_kv_layer(
                            writer,
                            snapshot,
                            static_cast<int>(start),
                            static_cast<int>(count));
                    } else if (exact_boundary) {
                        write_recurrent_layer(
                            writer,
                            snapshot,
                            static_cast<int>(start + count));
                    } else {
                        write_recurrent_unavailable_layer(
                            writer,
                            snapshot,
                            static_cast<int>(start + count));
                    }
                },
                layer);
        }, true);
}

MlxPagedPayload
MlxPagedSessionCodec<MlxQwen35TextSessionState>::encode_block(
    const MlxQwen35TextSessionState& state,
    std::size_t block_size,
    std::size_t block_index) {
    auto result = encode_state(
        state,
        block_size,
        block_index,
        1,
        kQwen35Runtime,
        [](Writer& writer, const MlxQwen35LayerCacheSnapshot& layer,
           std::size_t start, std::size_t count, bool exact_boundary) {
            std::visit(
                [&](const auto& snapshot) {
                    using Snapshot = std::decay_t<decltype(snapshot)>;
                    if constexpr (std::is_same_v<
                                      Snapshot, MlxKvCacheSnapshot>) {
                        write_kv_layer(
                            writer,
                            snapshot,
                            static_cast<int>(start),
                            static_cast<int>(count));
                    } else if (exact_boundary) {
                        write_recurrent_layer(
                            writer,
                            snapshot,
                            static_cast<int>(start + count));
                    } else {
                        write_recurrent_unavailable_layer(
                            writer,
                            snapshot,
                            static_cast<int>(start + count));
                    }
                },
                layer);
        }, true);
    if (result.size() != 1) {
        throw std::runtime_error("invalid Qwen3.5 cache block index");
    }
    return std::move(result.front());
}

MlxQwen35TextSessionState
MlxPagedSessionCodec<MlxQwen35TextSessionState>::decode(
    const std::vector<MlxPagedPayload>& payloads,
    const std::vector<std::int64_t>& tokens,
    std::size_t block_size) {
    auto blocks = decode_blocks(
        payloads, kQwen35Runtime, tokens.size(), block_size);
    MlxQwen35TextSessionState state;
    state.tokens = tokens;
    state.cache_position = static_cast<int>(tokens.size());
    state.cache_batch = 1;
    state.layers.reserve(blocks.front().layers.size());
    for (std::size_t index = 0; index < blocks.front().layers.size(); ++index) {
        if (blocks.front().layers[index].kind == kKvLayer) {
            auto layer = rebuild_kv(blocks, index, tokens.size());
            state.bytes += layer.nbytes();
            state.layers.emplace_back(std::move(layer));
        } else {
            auto layer = rebuild_recurrent(blocks, index, tokens.size());
            state.bytes += layer.nbytes();
            state.layers.emplace_back(std::move(layer));
        }
    }
    const auto& final = blocks.back();
    if (final.last_hidden) {
        state.last_hidden = copy_tensor(*final.last_hidden);
        state.bytes += state.last_hidden->nbytes();
        for (const auto& layer : final.mtp_layers) {
            if (layer.position != state.cache_position - 1 || layer.kind != kKvLayer)
                throw std::runtime_error("MTP checkpoint position mismatch");
            DecodedBlock head;
            head.count = static_cast<std::uint32_t>(layer.position);
            head.layers.push_back(layer);
            auto snapshot = rebuild_kv({head}, 0, layer.position);
            state.bytes += snapshot.nbytes();
            state.mtp_layers.push_back(std::move(snapshot));
        }
    }
    return state;
}

std::size_t MlxPagedSessionCodec<MlxQwen35TextSessionState>::token_count(const MlxPagedPayload& payload) {
    if (!payload) throw std::runtime_error("null cache payload");
    const auto block = read_block(*payload);
    return block.start + block.count;
}

bool MlxPagedSessionCodec<MlxQwen35TextSessionState>::has_mtp(const MlxPagedPayload& payload) {
    if (!payload) throw std::runtime_error("null cache payload");
    const auto block = read_block(*payload);
    if (block.runtime != kQwen35Runtime) throw std::runtime_error("incompatible Qwen3.5 cache block");
    return block.last_hidden && (block.start + block.count == 1 || !block.mtp_layers.empty());
}

bool MlxPagedSessionCodec<MlxQwen4TextSessionState>::has_mtp(const MlxPagedPayload& payload) {
    if (!payload) throw std::runtime_error("null cache payload");
    const auto block = read_block(*payload);
    if (block.runtime != kQwen4Runtime) throw std::runtime_error("incompatible Flash-Next cache block");
    return block.last_hidden && (block.start + block.count == 1 || !block.mtp_layers.empty());
}

MlxPagedPayload MlxPagedSessionCodec<MlxQwen4TextSessionState>::encode_block(
    const MlxQwen4TextSessionState& state, std::size_t block_size, std::size_t index) {
    auto blocks = encode_state(state, block_size, index, 1, kQwen4Runtime, write_flash_layer, true);
    if (blocks.size() != 1) throw std::runtime_error("invalid Flash-Next cache block index");
    return std::move(blocks.front());
}

MlxQwen4TextSessionState MlxPagedSessionCodec<MlxQwen4TextSessionState>::decode(
    const std::vector<MlxPagedPayload>& payloads, const std::vector<std::int64_t>& tokens,
    std::size_t block_size) {
    const auto blocks = decode_blocks(payloads, kQwen4Runtime, tokens.size(), block_size);
    MlxQwen4TextSessionState state;
    state.tokens = tokens;
    state.cache_position = static_cast<int>(tokens.size());
    state.cache_batch = 1;
    for (std::size_t i = 0; i < blocks.front().layers.size(); ++i)
        state.layers.push_back(rebuild_flash(blocks, i, tokens.size()));
    const auto& final = blocks.back();
    if (final.last_hidden) {
        state.last_hidden = copy_tensor(*final.last_hidden);
        for (const auto& layer : final.mtp_layers) {
            if (layer.position != state.cache_position - 1 || layer.kind != kQsaLayer)
                throw std::runtime_error("Flash-Next MTP checkpoint position mismatch");
            DecodedBlock head;
            head.count = static_cast<std::uint32_t>(layer.position);
            head.layers.push_back(layer);
            state.mtp_layers.push_back(rebuild_flash({head}, 0, layer.position));
        }
    }
    const auto bytes = [&](const MlxQwen4LayerCacheSnapshot& layer) {
        if (layer.kv) state.bytes += layer.kv->nbytes();
        if (layer.offloaded_kv) state.bytes += layer.offloaded_kv->rows.tail.size() +
            layer.offloaded_kv->rows.blocks.size() * sizeof(QsaKvStore::BlockPtr);
        if (layer.offloaded_pooled_keys) state.bytes += layer.offloaded_pooled_keys->rows.tail.size() +
            layer.offloaded_pooled_keys->rows.blocks.size() * sizeof(QsaKvStore::BlockPtr);
        for (const auto* value : {&layer.index_keys, &layer.pooled_keys, &layer.convolution,
            &layer.recurrent, &layer.ple_convolution}) if (*value) state.bytes += (*value)->nbytes();
        state.bytes += layer.ple_context.size() * sizeof(std::int64_t);
    };
    for (const auto& layer : state.layers) bytes(layer);
    for (const auto& layer : state.mtp_layers) bytes(layer);
    if (state.last_hidden) state.bytes += state.last_hidden->nbytes();
    if (!final.dflash.empty()) {
        auto& snapshot = state.dflash.emplace();
        snapshot.position = state.cache_position;
        for (const auto& [key, value] : final.dflash)
            snapshot.layers.push_back({copy_tensor(key), copy_tensor(value)});
        state.bytes += snapshot.bytes();
    }
    return state;
}

bool MlxPagedSessionCodec<MlxQwen4TextSessionState>::has_exact_boundary(const MlxPagedPayload& payload) {
    if (!payload) throw std::runtime_error("null cache payload");
    const auto block = read_block(*payload);
    if (block.runtime != kQwen4Runtime) throw std::runtime_error("incompatible Flash-Next cache block");
    const auto boundary = static_cast<std::uint64_t>(block.start) + block.count;
    if (boundary > std::numeric_limits<std::int32_t>::max() ||
        std::any_of(block.layers.begin(), block.layers.end(), [boundary](const DecodedLayer& layer) {
            return layer.position != static_cast<int>(boundary);
        })) throw std::runtime_error("Flash-Next cache checkpoint boundary mismatch");
    return std::none_of(block.layers.begin(), block.layers.end(), [](const DecodedLayer& layer) {
        return layer.kind == kGdnUnavailableLayer || (layer.kind == kQsaLayer && !layer.exact);
    });
}

std::size_t MlxPagedSessionCodec<MlxQwen4TextSessionState>::decodable_blocks(
    const std::vector<MlxPagedPayload>& payloads) {
    for (std::size_t count = payloads.size(); count > 0; --count)
        if (has_exact_boundary(payloads[count - 1])) return count;
    return 0;
}

std::size_t MlxPagedSessionCodec<MlxQwen4TextSessionState>::token_count(const MlxPagedPayload& payload) {
    if (!payload) throw std::runtime_error("null cache payload");
    const auto block = read_block(*payload);
    return block.start + block.count;
}

std::size_t
MlxPagedSessionCodec<MlxQwen35TextSessionState>::decodable_blocks(
    const std::vector<MlxPagedPayload>& payloads) {
    for (std::size_t count = payloads.size(); count > 0; --count) {
        if (has_exact_boundary(payloads[count - 1])) return count;
    }
    return 0;
}

bool MlxPagedSessionCodec<MlxQwen35TextSessionState>::has_exact_boundary(
    const MlxPagedPayload& payload) {
    if (!payload) {
        throw std::runtime_error("MLX cache block payload is null");
    }
    const auto block = read_block(*payload);
    if (block.runtime != kQwen35Runtime) {
        throw std::runtime_error("incompatible Qwen3.5 cache block");
    }
    const auto boundary = static_cast<std::uint64_t>(block.start) +
        static_cast<std::uint64_t>(block.count);
    if (boundary >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int32_t>::max()) ||
        std::any_of(
            block.layers.begin(),
            block.layers.end(),
            [boundary](const DecodedLayer& layer) {
                return layer.position != static_cast<int>(boundary);
            })) {
        throw std::runtime_error("Qwen3.5 cache checkpoint boundary mismatch");
    }
    return std::none_of(
        block.layers.begin(),
        block.layers.end(),
        [](const DecodedLayer& layer) {
            return layer.kind == kRecurrentUnavailableLayer;
        });
}

} // namespace mfq::metal

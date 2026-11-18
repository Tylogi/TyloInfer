#include "engine/weights/rows/gguf.h"

#include "engine/backends/cuda/native/tensor/tensor.h"
#include "engine/memory/host.h"
#include "engine/weights/encoding/gguf/decode.h"
#include "engine/weights/encoding/gguf_quant.h"
#include "engine/weights/source/mapped.h"
#include "engine/weights/source/row_residual.h"
#include "engine/weights/source/source.h"
#include "engine/weights/source/transform.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace tyloi::weights {
namespace {
constexpr std::size_t kCacheBytes = 8u << 20;

class RowCache final : public HostCache {
    struct Image {
        std::vector<std::byte> bytes;
        Image(const std::byte* source, std::size_t size) : bytes(source, source + size) {
            cuda::charge_tensor_host_bytes(bytes.capacity());
        }
        ~Image() {
            const auto size = bytes.capacity();
            { std::vector<std::byte> released; bytes.swap(released); }
            cuda::tensor_host_bytes.fetch_sub(size);
        }
    };
    using Entry = std::pair<std::uint64_t, std::shared_ptr<const Image>>;
    std::mutex mutex_;
    std::list<Entry> lru_;
    std::unordered_map<std::uint64_t, std::list<Entry>::iterator> entries_;
    std::size_t used_ = 0;

    std::size_t evict() {
        auto it = std::prev(lru_.end());
        const auto bytes = it->second->bytes.capacity();
        const std::weak_ptr<const Image> owner = it->second;
        entries_.erase(it->first);
        used_ -= bytes + 128;
        lru_.erase(it);
        return owner.expired() ? bytes : 0;
    }
public:
    std::shared_ptr<const Image> find(std::uint64_t offset) {
        std::lock_guard lock(mutex_);
        const auto found = entries_.find(offset);
        if (found == entries_.end()) return {};
        lru_.splice(lru_.begin(), lru_, found->second);
        return found->second->second;
    }

    void insert(std::uint64_t offset, const std::byte* source, std::size_t bytes) {
        if (bytes > kCacheBytes - 128) return;
        auto data = std::make_shared<Image>(source, bytes);
        std::lock_guard lock(mutex_);
        const auto found = entries_.find(offset);
        if (found != entries_.end()) {
            lru_.splice(lru_.begin(), lru_, found->second);
            return;
        }
        const auto backing = data->bytes.capacity();
        if (backing > kCacheBytes - 128) return;
        const auto cost = backing + 128;
        while (used_ > kCacheBytes - cost) evict();
        lru_.emplace_front(offset, std::move(data));
        try {
            entries_.emplace(offset, lru_.begin());
        } catch (...) {
            lru_.pop_front();
            throw;
        }
        used_ += cost;
    }

    std::size_t reclaim(std::size_t bytes) override {
        std::lock_guard lock(mutex_);
        std::size_t released = 0;
        while (released < bytes && !lru_.empty()) released += evict();
        return released;
    }
};
} // namespace

struct GgufRows::Cache {
    std::mutex read_mutex;
    std::shared_ptr<RowCache> rows;
    source::Source::TensorReader read;
    source::Source::TensorBatchReader batch;
    std::shared_ptr<const RowResidualFile> correction;
};

GgufRows::GgufRows(const source::Source& source, const std::string& name)
    : cache_(std::make_unique<Cache>()) {
    const auto* record = source.find_tensor(name);
    if (!record) throw std::invalid_argument("missing row table: " + name);
    auto shape = record->shape;
    std::uint64_t payload_offset = 0;
    const bool floating = record->dtype == "F16" || record->dtype == "BF16" || record->dtype == "F32";
    if (!shape && floating) {
        std::array<std::byte, 20> header{};
        if (record->nbytes < header.size()) throw std::invalid_argument("truncated floating row table: " + name);
        source.read_range_into(name, 0, header.data(), header.size());
        const auto integer = [&](std::size_t start, std::size_t bytes) {
            std::uint64_t value = 0;
            for (std::size_t i = 0; i < bytes; ++i)
                value |= std::uint64_t(std::to_integer<unsigned char>(header[start + i])) << (8 * i);
            return value;
        };
        if (integer(0, 4) != 2 || integer(4, 8) > INT64_MAX || integer(12, 8) > INT64_MAX)
            throw std::invalid_argument("floating row table requires a matrix: " + name);
        shape = std::vector<std::int64_t>{std::int64_t(integer(4, 8)), std::int64_t(integer(12, 8))};
        payload_offset = header.size();
    }
    if (!shape || shape->size() != 2) throw std::invalid_argument("row table requires a matrix: " + name);
    type_ = record->dtype == "F16" ? GGML_TYPE_F16 : record->dtype == "BF16" ? GGML_TYPE_BF16 :
        record->dtype == "F32" ? GGML_TYPE_F32 : gguf::quant_type(record->dtype);
    rows_ = shape->at(0);
    width_ = shape->at(1);
    if (const auto transform = source::hadamard(source, name)) {
        if ((!transform->signs.empty() && std::uint64_t(width_) != transform->signs.size()) ||
                transform->block_size < 1 || width_ <= 0 || width_ % transform->block_size ||
                std::any_of(transform->signs.begin(), transform->signs.end(), [](int sign) { return sign != 1 && sign != -1; }))
            throw std::invalid_argument("GGUF Hadamard basis is incompatible with the row table");
        if (transform->block_size != 1 || std::any_of(transform->signs.begin(), transform->signs.end(), [](int sign) { return sign != 1; }))
            throw std::invalid_argument("GGUF streaming row tables do not yet support an inverse Hadamard transform");
    }
    row_bytes_ = gguf::row_bytes(type_, width_);
    if (rows_ <= 0 || row_bytes_ > SIZE_MAX || std::uint64_t(rows_) > UINT64_MAX / row_bytes_ ||
            record->nbytes - payload_offset != std::uint64_t(rows_) * row_bytes_)
        throw std::invalid_argument("invalid GGUF row table payload: " + name);

    auto batch = source.tensor_batch_reader(name);
    auto reader = batch ? source::Source::TensorReader{} : source.mapped_tensor_reader(name);
    const bool mapped = bool(reader);
    if (!batch && !reader) reader = source.tensor_reader(name);
    if (!batch && !reader) throw std::invalid_argument("GGUF row table has no retained reader: " + name);
    if (mapped) {
        if (auto mapped_batch = source.mapped_tensor_batch_reader(name)) batch = std::move(mapped_batch);
    }
    if (reader) cache_->read = [reader = std::move(reader), payload_offset](std::uint64_t offset, std::byte* out, std::size_t bytes) {
        reader(payload_offset + offset, out, bytes);
    };
    if (batch) cache_->batch = [batch = std::move(batch), payload_offset](const std::vector<ReadSpan>& spans) {
        auto adjusted = spans;
        for (auto& span : adjusted) span.offset += payload_offset;
        batch(adjusted);
    };
    cache_->correction = source.row_residual(name);
    if (cache_->correction && !cache_->correction->payload_bytes()) cache_->correction.reset();
    if (cache_->correction && (cache_->correction->row_count() != std::uint64_t(rows_) ||
            cache_->correction->columns() != std::uint64_t(width_) || width_ > INT64_MAX - 127))
        throw std::invalid_argument("invalid GGUF row correction shape: " + name);
    staged_type_ = cache_->correction ? GGML_TYPE_F16 : type_;
    staged_row_bytes_ = cache_->correction ? gguf::row_bytes(GGML_TYPE_F16, width_) : row_bytes_;
    if (staged_row_bytes_ > SIZE_MAX) throw std::invalid_argument("GGUF staged row is too large");
    parallel_ = source.supports_parallel_tensor_reads() || (mapped && !cache_->correction);
    cache_->rows = std::make_shared<RowCache>();
    register_host_cache(cache_->rows);
}

GgufRows::~GgufRows() = default;

void GgufRows::decode_row(std::int64_t id, float* output) const {
    if (id < 0 || id >= rows_ || !output) throw std::out_of_range("GGUF table row id");
    std::vector<std::byte> base(static_cast<std::size_t>(row_bytes_));
    read_rows({{std::uint64_t(id) * row_bytes_, base.data(), base.size()}});
    gguf::decode_row(type_, base.data(), output, width_);
    if (cache_->correction) {
        const auto& row = cache_->correction->rows().at(static_cast<std::size_t>(id));
        if (row.type) {
            auto packed = cache_->correction->read_row(id);
            std::vector<float> values(static_cast<std::size_t>((width_ + 127) / 128 * 128));
            gguf::decode_row(ggml_type(row.type), packed.data(), values.data(), std::int64_t(values.size()));
            for (std::int64_t col = 0; col < width_; ++col) output[col] += values[col];
        }
    }
}

void GgufRows::read_staged_rows(const std::vector<ReadSpan>& rows) const {
    if (!cache_->correction) { read_rows(rows); return; }
    std::vector<float> values(static_cast<std::size_t>(width_));
    for (const auto& row : rows) {
        if (row.offset % row_bytes_ || row.offset / row_bytes_ >= std::uint64_t(rows_) ||
                row.size != staged_row_bytes_ || !row.destination)
            throw std::invalid_argument("invalid corrected GGUF row request");
        decode_row(std::int64_t(row.offset / row_bytes_), values.data());
        for (std::int64_t col = 0; col < width_; ++col) {
            const auto value = ggml_fp32_to_fp16(values[col]);
            std::memcpy(row.destination + col * sizeof(value), &value, sizeof(value));
        }
    }
}

void GgufRows::read_rows(const std::vector<ReadSpan>& requests) const {
    std::lock_guard lock(cache_->read_mutex);
    std::vector<ReadSpan> missing;
    std::unordered_map<std::uint64_t, std::byte*> unique;
    std::vector<std::pair<std::byte*, const std::byte*>> repeated;
    for (const auto& row : requests) {
        if (row.size != row_bytes_ || row.offset % row_bytes_ || row.offset / row_bytes_ >= std::uint64_t(rows_) || !row.destination)
            throw std::out_of_range("invalid GGUF table row range");
        if (const auto found = cache_->rows->find(row.offset))
            std::memcpy(row.destination, found->bytes.data(), row.size);
        else if (auto [first, inserted] = unique.emplace(row.offset, row.destination); inserted)
            missing.push_back(row);
        else repeated.emplace_back(row.destination, first->second);
    }
    std::sort(missing.begin(), missing.end(), [](const auto& a, const auto& b) { return a.offset < b.offset; });
    const MappedPrefetchScope copied(missing.size() > 128, true);
    std::vector<ReadSpan> batch;
    std::size_t batch_bytes = 0;
    const auto flush = [&] {
        if (batch.empty()) return;
        if (cache_->batch) cache_->batch(batch);
        else for (const auto& span : batch) cache_->read(span.offset, span.destination, span.size);
        batch.clear();
        batch_bytes = 0;
    };
    for (const auto& row : missing) {
        std::size_t pos = 0;
        while (pos < row.size) {
            const auto size = std::min(row.size - pos, kCacheBytes - batch_bytes);
            batch.push_back({row.offset + pos, row.destination + pos, size});
            pos += size;
            batch_bytes += size;
            if (batch.size() == 256 || batch_bytes == kCacheBytes) flush();
        }
    }
    flush();
    for (const auto& [destination, source] : repeated) std::memcpy(destination, source, row_bytes_);
    for (const auto& row : requests) {
        if (cache_->rows->find(row.offset)) continue;
        cache_->rows->insert(row.offset, row.destination, row.size);
    }
}

} // namespace tyloi::weights

#include "engine/weights/rows/tq.h"
#include "engine/weights/rows/cache.h"
#include "engine/weights/rows/prefetch.h"
#include "engine/weights/encoding/tq/table.h"
#include "engine/weights/source/mapped.h"
#include "engine/backends/cpu/native/host_parallel.h"
#include "engine/backends/cuda/native/tensor/tensor.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <future>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::weights {

struct TqRead::Source {
    std::vector<std::shared_ptr<TqTable>> tables;
    std::vector<std::shared_ptr<RowCache>> caches;
    std::int64_t rows_per_shard;
    std::size_t stride = 0;
    bool parallel = true;
};

namespace {
struct Packet {
    std::vector<std::uint8_t> bytes;
    TqSelection selection;
    explicit Packet(std::size_t size) : bytes(size) {
        cuda::charge_tensor_host_bytes(bytes.capacity());
    }
    ~Packet() {
        const auto capacity = bytes.capacity();
        { std::vector<std::uint8_t> released; bytes.swap(released); }
        cuda::tensor_host_bytes.fetch_sub(capacity);
    }
};

TqRowSelection select(const TqRead::Source& source, const std::vector<std::int64_t>& ids,
                      const std::vector<std::int64_t>& shape) {
    std::int64_t count = 1;
    for (const auto size : shape) {
        if (size < 0 || (size && count > INT_MAX / size))
            throw std::overflow_error("TQ row shape exceeds bounds");
        count *= size;
    }
    if (static_cast<std::size_t>(count) != ids.size())
        throw std::invalid_argument("TQ row shape differs");
    auto selected = select_tq_rows(ids, source.rows_per_shard, source.tables.size());
    for (const auto id : selected.selected) {
        const auto shard = static_cast<std::size_t>(id / source.rows_per_shard);
        if (static_cast<std::uint64_t>(id % source.rows_per_shard) >= source.tables[shard]->rows())
            throw std::out_of_range("TQ row ID exceeds shard");
    }
    return selected;
}

std::size_t packet_bytes(const TqRead::Source& source, std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / source.stride)
        throw std::overflow_error("TQ row packet exceeds bounds");
    return count * source.stride;
}

TqSelection read(const TqRead::Source& source, TqRowSelection selected,
                 std::uint8_t* target, std::size_t capacity) {
    const auto bytes = packet_bytes(source, selected.selected.size());
    if (bytes > capacity || (bytes && !target))
        throw std::overflow_error("TQ row packet exceeds capacity");
    TqSelection result;
    result.rows.resize(selected.selected.size());
    result.inverse = std::move(selected.inverse);
    const auto prepare = [&](std::int64_t begin, std::int64_t end) {
        for (auto i = begin; i < end; ++i) {
            const auto id = selected.selected[static_cast<std::size_t>(i)];
            const auto shard = static_cast<std::size_t>(id / source.rows_per_shard);
            const auto row = static_cast<std::uint64_t>(id % source.rows_per_shard);
            const auto& table = *source.tables[shard];
            std::size_t first = 0, last = table.segment_count();
            while (first < last) {
                const auto mid = first + (last - first) / 2;
                if (row < table.segment_end(mid)) last = mid;
                else first = mid + 1;
            }
            const auto start = first ? table.segment_end(first - 1) : 0;
            const auto& segment = table.segment(first);
            const auto& metadata = segment.layout();
            auto* image = target + static_cast<std::size_t>(i) * source.stride;
            const auto cached = source.caches[shard]->copy(row, reinterpret_cast<std::byte*>(image), source.stride);
            TqRowImageLayout layout;
            if (cached) {
                const auto scales = metadata.columns ? metadata.scale_bytes : (metadata.groups + 1) / 2;
                const auto packed = (72 + scales + 3) & ~std::size_t(3);
                layout = {packed, cached - sizeof(kTqRelative), cached};
            } else {
                const MappedPrefetchScope copied(false, true);
                layout = segment.stage_row_image(row - start, image, source.stride);
                source.caches[shard]->insert(row, reinterpret_cast<const std::byte*>(image), layout.bytes);
            }
            result.rows[static_cast<std::size_t>(i)] = {layout, metadata.width, metadata.gs, metadata.columns};
        }
    };
    const bool parallel = std::all_of(selected.selected.begin(), selected.selected.end(), [&](auto id) {
        return source.tables[static_cast<std::size_t>(id / source.rows_per_shard)]->parallel_reads();
    });
    if (parallel && selected.selected.size() > 1)
        host_parallel_for(0, static_cast<std::int64_t>(selected.selected.size()), 1,
            static_cast<int>(std::min<std::size_t>(selected.selected.size(), 8)), prepare);
    else prepare(0, static_cast<std::int64_t>(selected.selected.size()));
    return result;
}
}

struct TqRead::Impl {
    std::shared_ptr<const Source> source;
    std::vector<std::int64_t> ids, shape;
    std::shared_ptr<Packet> packet;
    std::future<void> pending;
};

std::shared_ptr<const TqRead::Source> TqRead::source(
        std::vector<std::shared_ptr<TqTable>> tables, std::int64_t rows_per_shard) {
    if (tables.empty() || !tables.front() || rows_per_shard < 1)
        throw std::invalid_argument("invalid TQ row shards");
    auto result = std::make_shared<Source>();
    result->rows_per_shard = rows_per_shard;
    for (const auto& table : tables) {
        if (!table || table->width() != tables.front()->width())
            throw std::invalid_argument("TQ row shard widths differ");
        result->stride = std::max(result->stride, table->row_image_bytes_bound());
        result->parallel = result->parallel && table->parallel_reads();
        result->caches.push_back(make_row_cache());
    }
    result->tables = std::move(tables);
    return result;
}

TqRead::TqRead(std::shared_ptr<const Source> source) : impl_(std::make_unique<Impl>()) {
    if (!source) throw std::invalid_argument("missing TQ row source");
    impl_->source = std::move(source);
}
TqRead::~TqRead() { clear(); }

bool TqRead::issue(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape) {
    if (matches(ids, shape)) return true;
    clear();
    if (ids.empty() || !impl_->source->parallel) return false;
    // shortcut: cap speculative stage images, revisit if large prompt prefetch is needed.
    if (ids.size() > prefetch_limit / impl_->source->stride) return false;
    auto selected = select(*impl_->source, ids, shape);
    const auto bytes = packet_bytes(*impl_->source, selected.selected.size());
    try {
        if (sample_host_memory().headroom() < bytes) return false;
    } catch (const std::exception&) {
        return false;
    }
    auto packet = std::make_shared<Packet>(bytes);
    auto source = impl_->source;
    std::packaged_task<void()> job([source, packet, selected = std::move(selected)]() mutable {
        packet->selection = read(*source, std::move(selected), packet->bytes.data(), packet->bytes.size());
    });
    auto pending = job.get_future();
    impl_->ids = ids;
    impl_->shape = shape;
    submit_row_read(std::move(job));
    impl_->packet = std::move(packet);
    impl_->pending = std::move(pending);
    return true;
}

bool TqRead::matches(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape) const {
    return impl_->pending.valid() && impl_->ids == ids && impl_->shape == shape;
}

TqSelection TqRead::collect(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape,
                            std::uint8_t* target, std::size_t capacity) {
    if (matches(ids, shape)) {
        const auto bytes = impl_->packet->bytes.size();
        if (bytes > capacity || (bytes && !target))
            throw std::overflow_error("TQ row packet exceeds capacity");
        auto pending = std::move(impl_->pending);
        auto packet = std::move(impl_->packet);
        impl_->ids.clear();
        impl_->shape.clear();
        pending.get();
        if (bytes) std::memcpy(target, packet->bytes.data(), bytes);
        return std::move(packet->selection);
    }
    clear();
    return read(*impl_->source, select(*impl_->source, ids, shape), target, capacity);
}

std::size_t TqRead::stride() const { return impl_->source->stride; }
void TqRead::clear() {
    if (impl_->pending.valid()) impl_->pending.wait();
    impl_->pending = {};
    impl_->packet.reset();
    impl_->ids.clear();
    impl_->shape.clear();
}

} // namespace tyloi::weights

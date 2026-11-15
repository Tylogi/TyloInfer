#include "engine/state/paged_kv.h"

#include "engine/ops.h"
#include "engine/ops/kv_storage.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace tyloi::state {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("Paged KV growth overflow");
    return a * b;
}
void add_growth(PagedKv::Growth& growth, Device device, std::size_t bytes) {
    if (!bytes) return;
    auto found = std::find_if(growth.begin(), growth.end(),
        [device](const auto& entry) { return entry.first == device; });
    if (found == growth.end()) growth.emplace_back(device, bytes);
    else {
        if (found->second > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::overflow_error("Paged KV growth overflow");
        found->second += bytes;
    }
}
Tensor host_table(std::vector<std::int32_t>& data, std::int64_t rows, std::int64_t columns) {
    const std::array<std::int64_t, 2> shape{rows, columns};
    const Device cpu{DeviceType::cpu, 0};
    auto storage = std::make_shared<TensorStorage>();
    storage->base = data.data();
    storage->bytes = data.size() * sizeof(std::int32_t);
    storage->device = cpu;
    return Tensor{DenseTensor(std::move(storage), make_contiguous_view(data.data(), shape, kInt32, cpu))};
}
}

PagedKv::Growth PagedKv::storage_growth(const std::vector<Layout>& layers,
        size_t maximum_pages, size_t current_pages, size_t required_pages) {
    require(required_pages <= maximum_pages, "Paged KV physical storage exhausted");
    Growth growth;
    while (current_pages < required_pages) {
        const auto chunk = std::min<size_t>(kPagesPerChunk, maximum_pages - current_pages);
        for (const auto& layer : layers) {
            const auto pages = layer.device.is_metal() ? current_pages + chunk : chunk;
            const auto capacity = static_cast<int64_t>(multiply(pages, kPageSize));
            auto bytes = kv_bytes(1, layer.heads, capacity, layer.head_dim, false,
                                  kFloat16, layer.quantization);
            // Metal grow_kv creates a new zero-filled target and functional slice updates.
            // Both can be live in its lazy graph until the replacement is submitted.
            if (layer.device.is_metal() && current_pages) bytes = multiply(bytes, 2);
            add_growth(growth, layer.device, bytes);
        }
        current_pages += chunk;
    }
    return growth;
}

PagedKv::Growth PagedKv::construction_growth(const std::vector<Layout>& layouts,
        int64_t context, int32_t sequences, bool preallocate) {
    require(context > 0 && sequences > 0, "Paged KV requires positive sequence and context limits");
    const auto pages = context / kPageSize + (context % kPageSize != 0);
    const auto maximum = checked_maximum_pages(sequences, pages);
    require(!layouts.empty(), "Paged KV requires at least one full-attention layer");
    const auto table_bytes = multiply(multiply(static_cast<size_t>(sequences),
        static_cast<size_t>(pages)), sizeof(std::int32_t));
    const auto chunks = (maximum + kPagesPerChunk - 1) / kPagesPerChunk;
    Growth growth;
    add_growth(growth, Device(DeviceType::cpu, 0), table_bytes);
    std::unordered_set<std::int64_t> table_devices;
    for (const auto& layout : layouts) {
        const auto device = layout.device;
        require(layout.heads > 0 && layout.head_dim > 0,
            "Paged KV encountered invalid full-attention geometry");
        if ((!device.is_cuda() || device.index < 0) && (!device.is_metal() || device.index != 0))
            throw std::invalid_argument("Paged KV requires CUDA or Metal 0");
#if !TYLOINFER_HAS_CUDA
        if (device.is_cuda()) throw std::invalid_argument("Paged KV CUDA support is unavailable");
#endif
        // Also validate quantization before any device allocation.
        (void)kv_layout(1, layout.heads, 1, layout.head_dim, false, kFloat16, layout.quantization);
        const auto key = (static_cast<std::int64_t>(device.type) << 32) | static_cast<std::uint32_t>(device.index);
        if (table_devices.insert(key).second) add_growth(growth, device, table_bytes);
        if (device.is_cuda()) add_growth(growth, device, multiply(chunks, 2 * sizeof(std::int64_t)));
    }
    if (preallocate)
        for (const auto& [device, bytes] : storage_growth(layouts, maximum, 0, static_cast<size_t>(pages)))
            add_growth(growth, device, bytes);
    return growth;
}

PagedKv::Growth PagedKv::ensure_growth(
        const std::vector<std::pair<const Sequence*, int64_t>>& sequences) const {
    size_t additional = 0;
    std::unordered_set<const Sequence*> seen;
    for (const auto& [sequence, tokens] : sequences) {
        require(sequence && seen.insert(sequence).second && validate(*sequence),
            "Paged KV sequence handle is invalid");
        require(tokens >= 0 && tokens <= logical_pages_per_sequence_ * kPageSize,
            "Paged KV sequence length exceeds the configured context");
        const auto pages = static_cast<size_t>(tokens / kPageSize + (tokens % kPageSize != 0));
        if (pages > sequence->physical_pages_.size()) {
            const auto count = pages - sequence->physical_pages_.size();
            require(count <= maximum_physical_pages_ - allocator_.live_pages() - additional,
                "Paged KV physical storage exhausted");
            additional += count;
        }
    }
    if (additional) require(generation_ != std::numeric_limits<std::uint64_t>::max(),
        "Paged KV allocation generation exhausted");
    const auto required = std::max(allocator_.high_watermark(), allocator_.live_pages() + additional);
    std::vector<Layout> layouts;
    layouts.reserve(layers_.size());
    for (const auto& pool : layers_) layouts.push_back(pool.layout);
    return storage_growth(layouts, maximum_physical_pages_, storage_pages_, required);
}

PagedKv::Growth PagedKv::update_growth() const {
    Growth growth;
    const auto bytes = multiply(host_page_table_.size(), sizeof(std::int32_t));
    for (const auto& [device, table] : device_page_tables_)
        if (device.first == DeviceType::metal) add_growth(growth, Device(device.first, device.second), bytes);
    return growth;
}

PagedKv::PagedKv(const std::vector<Layout>& layouts, int64_t context,
                 int32_t sequences, bool preallocate)
    : maximum_sequences_(sequences),
      logical_pages_per_sequence_((context + kPageSize - 1) / kPageSize),
      maximum_physical_pages_(checked_maximum_pages(sequences, logical_pages_per_sequence_)),
      allocator_(maximum_physical_pages_),
      host_page_table_(size_t(sequences) * size_t(logical_pages_per_sequence_), -1) {
    require(maximum_sequences_ > 0 &&
        logical_pages_per_sequence_ > 0,
        "Paged KV requires positive sequence and context limits");
    const int64_t maximum_chunks =
        (maximum_physical_pages_ + kPagesPerChunk - 1) /
        kPagesPerChunk;
    for (const auto& layout : layouts) {
        const auto heads = layout.heads;
        const auto head_dim = layout.head_dim;
        require(heads > 0 && head_dim > 0,
            "Paged KV encountered invalid full-attention geometry");
        const auto device = layout.device;
        if ((!device.is_cuda() || device.index < 0) && (!device.is_metal() || device.index != 0))
            throw std::invalid_argument("Paged KV requires CUDA or Metal 0");
#if !TYLOINFER_HAS_CUDA
        if (device.is_cuda()) throw std::invalid_argument("Paged KV CUDA support is unavailable");
#endif
        const auto device_key = std::pair{device.type, device.index};
        if (!device_page_tables_.contains(device_key)) {
            if (device.is_metal()) {
                device_page_tables_.emplace(device_key, host_table(host_page_table_, maximum_sequences_, logical_pages_per_sequence_).relocated(Device{DeviceType::metal, 0}));
            } else {
                auto page_table = ops::empty({maximum_sequences_, logical_pages_per_sequence_},
                    TensorOptions().device(device).dtype(kInt32));
                ops::fill(page_table, -1);
                device_page_tables_.emplace(device_key, std::move(page_table));
            }
        }
        Pool pool;
        pool.layout = layout;
        pool.key_width = layout.quantization.enabled()
            ? kv_packed_width(int(head_dim), layout.quantization.key_bits()) : head_dim;
        pool.value_width = layout.quantization.enabled()
            ? kv_packed_width(int(head_dim), layout.quantization.value_bits()) : head_dim;
        pool.dtype = layout.quantization.enabled()
            ? (device.is_metal() ? ScalarType::uint32 : kInt32) : kFloat16;
        if (device.is_cuda()) {
            auto options = TensorOptions().device(device).dtype(kInt64);
            pool.k_chunk_ptrs = ops::zeros({maximum_chunks}, options);
            pool.v_chunk_ptrs = ops::zeros({maximum_chunks}, options);
        }
        layers_.push_back(std::move(pool));
    }
    require(!layers_.empty(),
        "Paged KV requires at least one full-attention layer");
    if (preallocate)
        ensure_storage(static_cast<size_t>(logical_pages_per_sequence_));
}

bool PagedKv::ensure_tokens(Sequence& sequence, int64_t token_count) {
    require(token_count >= 0 &&
        token_count <= logical_pages_per_sequence_ * kPageSize,
        "Paged KV sequence length exceeds the configured context");
    require(validate(sequence),
        "Paged KV sequence handle is invalid");
    const size_t required_pages = static_cast<size_t>(
        (token_count + kPageSize - 1) / kPageSize);
    if (required_pages <= sequence.physical_pages_.size()) return false;
    const size_t previous_size = sequence.physical_pages_.size();
    const size_t additional_pages = required_pages - previous_size;
    require(generation_ !=
        std::numeric_limits<std::uint64_t>::max(),
        "Paged KV allocation generation exhausted");
    const size_t live_pages = allocator_.live_pages();
    require(additional_pages <= maximum_physical_pages_ - live_pages,
        "Paged KV physical storage exhausted");
    const size_t required_highwatermark = std::max(
        allocator_.high_watermark(), live_pages + additional_pages);
    sequence.physical_pages_.reserve(required_pages);
    sequence.page_generations_.reserve(required_pages);
    page_generations_.resize(required_highwatermark);

    auto pages = allocator_.allocate(additional_pages);
    try {
        ensure_storage(allocator_.high_watermark());
    } catch (...) {
        allocator_.release(pages);
        publish_allocator_metrics();
        throw;
    }
    const auto generation = generation_++;
    for (const auto page : pages)
        page_generations_[static_cast<size_t>(page)] = generation;
    sequence.physical_pages_.insert(
        sequence.physical_pages_.end(), pages.begin(), pages.end());
    sequence.page_generations_.insert(
        sequence.page_generations_.end(), additional_pages, generation);
    sequence.owner_ = identity_;
    publish_allocator_metrics();
    return true;
}

void PagedKv::release(Sequence& sequence) {
    require(validate(sequence),
        "Paged KV sequence handle is invalid");
    if (sequence.physical_pages_.empty()) return;
    allocator_.release(sequence.physical_pages_);
    sequence.physical_pages_.clear();
    sequence.page_generations_.clear();
    sequence.owner_.reset();
    publish_allocator_metrics();
}

void PagedKv::update(const std::vector<const Sequence*>& sequences) {
    require(!sequences.empty() &&
        sequences.size() <= static_cast<size_t>(maximum_sequences_),
        "Paged KV cannot update the requested batch size");
    const size_t rows = sequences.size();
    const size_t row_width =
        static_cast<size_t>(logical_pages_per_sequence_);
    for (const auto * sequence : sequences) {
        require(sequence != nullptr &&
            sequence->physical_pages_.size() <= row_width &&
            validate(*sequence),
            "Paged KV received an invalid sequence handle");
    }
    std::fill(
        host_page_table_.begin(),
        host_page_table_.begin() + rows * row_width, -1);
    for (size_t row = 0; row < rows; ++row) {
        const auto * sequence = sequences[row];
        for (size_t logical_page = 0;
                logical_page < sequence->physical_pages_.size();
                ++logical_page) {
            const auto physical =
                sequence->physical_pages_[logical_page];
            host_page_table_[row * row_width + logical_page] = physical;
        }
    }
    for (auto & [device, table] : device_page_tables_) {
        if (device.first == DeviceType::metal) {
            table = host_table(host_page_table_, maximum_sequences_, logical_pages_per_sequence_).relocated(Device{DeviceType::metal, 0});
        } else {
            auto target = Tensor{table.dense().narrow(0, 0, static_cast<int64_t>(rows))};
            ops::copy_into(target, host_table(host_page_table_, static_cast<int64_t>(rows), logical_pages_per_sequence_));
        }
    }
    page_table_updates_metric_.fetch_add(1, std::memory_order_relaxed);
}

tyloi::state::KvView PagedKv::view(size_t index, int64_t batch) const {
    require(batch > 0 && batch <= maximum_sequences_,
        "Paged KV view batch size is invalid");
    const auto& pool = layers_.at(index);
    tyloi::state::KvView cache;
    if (pool.layout.device.is_metal()) {
        cache.physical_key = pool.storage.physical_key;
        cache.physical_value = pool.storage.physical_value;
    } else {
        cache.key_chunk_ptrs = pool.k_chunk_ptrs;
        cache.value_chunk_ptrs = pool.v_chunk_ptrs;
    }
    cache.page_table = device_page_tables_.at({pool.layout.device.type, pool.layout.device.index});
    cache.batch = batch;
    cache.heads = pool.layout.heads;
    cache.head_dim = pool.layout.head_dim;
    cache.page_size = kPageSize;
    cache.pages_per_chunk = kPagesPerChunk;
    cache.dtype = ::tyloi::kFloat16;
    cache.quantization = pool.layout.quantization;
    return cache;
}

PagedKv::Stats PagedKv::stats() const noexcept {
    return {kPageSize,
        int64_t(live_pages_metric_.load(std::memory_order_relaxed)),
        int64_t(peak_live_pages_metric_.load(std::memory_order_relaxed)),
        int64_t(capacity_pages_metric_.load(std::memory_order_relaxed)),
        int64_t(reserved_bytes_metric_.load(std::memory_order_relaxed)),
        int64_t(allocation_count_metric_.load(std::memory_order_relaxed)),
        int64_t(reuse_count_metric_.load(std::memory_order_relaxed)),
        int64_t(release_count_metric_.load(std::memory_order_relaxed)),
        int64_t(page_table_updates_metric_.load(std::memory_order_relaxed))};
}

size_t PagedKv::checked_maximum_pages(int32_t maximum_sequences, int64_t pages_per_sequence) {
    if (maximum_sequences <= 0 || pages_per_sequence <= 0 ||
            static_cast<uint64_t>(maximum_sequences) >
                static_cast<uint64_t>(
                    std::numeric_limits<int32_t>::max()) /
                static_cast<uint64_t>(pages_per_sequence)) {
        throw std::invalid_argument(
            "Paged KV maximum physical page count exceeds int32");
    }
    return static_cast<size_t>(maximum_sequences) *
        static_cast<size_t>(pages_per_sequence);
}

bool PagedKv::validate(const Sequence& sequence) const noexcept {
    if (sequence.physical_pages_.empty())
        return sequence.page_generations_.empty() && !sequence.owner_;
    if (sequence.owner_.get() != identity_.get() ||
            sequence.page_generations_.size() != sequence.physical_pages_.size())
        return false;
    for (size_t index = 0; index < sequence.physical_pages_.size(); ++index) {
        const auto page = sequence.physical_pages_[index];
        if (page < 0 || static_cast<size_t>(page) >= page_generations_.size() ||
                !allocator_.owns(page) ||
                page_generations_[static_cast<size_t>(page)] !=
                    sequence.page_generations_[index])
            return false;
    }
    return true;
}

void PagedKv::publish_allocator_metrics() noexcept {
    live_pages_metric_.store(
        allocator_.live_pages(), std::memory_order_relaxed);
    peak_live_pages_metric_.store(
        allocator_.peak_live_pages(), std::memory_order_relaxed);
    allocation_count_metric_.store(
        allocator_.allocation_count(), std::memory_order_relaxed);
    reuse_count_metric_.store(
        allocator_.reuse_count(), std::memory_order_relaxed);
    release_count_metric_.store(
        allocator_.release_count(), std::memory_order_relaxed);
}

void PagedKv::ensure_storage(size_t required_pages) {
    while (storage_pages_ < required_pages) {
        const size_t chunk_pages = std::min<size_t>(
            kPagesPerChunk, maximum_physical_pages_ - storage_pages_);
        require(chunk_pages > 0,
            "Paged KV physical storage exhausted");
        std::vector<Chunk> pending;
        pending.reserve(layers_.size());
        for (const auto & layer : layers_) {
            Chunk chunk;
            if (layer.layout.device.is_metal()) {
                const auto capacity = static_cast<int64_t>((storage_pages_ + chunk_pages) * kPageSize);
                if (layer.storage.defined())
                    chunk.storage = grow_kv(layer.storage, capacity, static_cast<int64_t>(storage_pages_ * kPageSize));
                else
                    chunk.storage = allocate_kv(1, layer.layout.heads, capacity, layer.layout.head_dim,
                        false, layer.layout.device, kFloat16, layer.layout.quantization);
            } else {
                auto options = TensorOptions().device(layer.layout.device).dtype(layer.dtype);
                chunk.k = ops::empty(
                    {static_cast<int64_t>(chunk_pages), layer.layout.heads, kPageSize, layer.key_width}, options);
                chunk.v = ops::empty(
                    {static_cast<int64_t>(chunk_pages), layer.layout.heads, kPageSize, layer.value_width}, options);
            }
            pending.push_back(std::move(chunk));
        }
        const size_t chunk_index = storage_pages_ / kPagesPerChunk;
        for (size_t index = 0; index < layers_.size(); ++index) {
            auto & layer = layers_[index];
            auto & chunk = pending[index];
            if (layer.layout.device.is_cuda()) {
                ops::bind_kv_chunk(layer.k_chunk_ptrs, layer.v_chunk_ptrs,
                    chunk_index, chunk.k, chunk.v);
            }
        }
        for (size_t index = 0; index < layers_.size(); ++index) {
            auto & layer = layers_[index];
            auto & chunk = pending[index];
            if (layer.layout.device.is_metal()) {
                if (layer.storage.defined()) reserved_bytes_ -= layer.storage.physical_key.nbytes() + layer.storage.physical_value.nbytes();
                reserved_bytes_ += chunk.storage.physical_key.nbytes() + chunk.storage.physical_value.nbytes();
                layer.storage = std::move(chunk.storage);
            } else {
                reserved_bytes_ += chunk.k.nbytes() + chunk.v.nbytes();
                layer.k_chunks.push_back(std::move(chunk.k));
                layer.v_chunks.push_back(std::move(chunk.v));
            }
        }
        storage_pages_ += chunk_pages;
        capacity_pages_metric_.store(
            storage_pages_, std::memory_order_relaxed);
        reserved_bytes_metric_.store(
            reserved_bytes_, std::memory_order_relaxed);
    }
}

} // namespace tyloi::state

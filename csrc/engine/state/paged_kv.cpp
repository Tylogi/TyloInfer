#include "engine/state/paged_kv.h"

#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/native/tensor/backend.h"
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::state {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
Tensor metal_table(std::vector<std::int32_t>& data, std::int64_t rows, std::int64_t columns) {
    const std::array<std::int64_t, 2> shape{rows, columns};
    const Device cpu{DeviceType::cpu, 0};
    auto storage = std::make_shared<TensorStorage>();
    storage->base = data.data();
    storage->bytes = data.size() * sizeof(std::int32_t);
    storage->device = cpu;
    Tensor host{DenseTensor(std::move(storage), make_contiguous_view(data.data(), shape, kInt32, cpu))};
    // Relocation copies the borrowed host bytes before this view is released.
    return host.relocated(Device{DeviceType::metal, 0});
}
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
#ifndef TYLOINFER_NATIVE_CUDA
        if (device.is_cuda()) throw std::invalid_argument("Paged KV CUDA support is unavailable");
#endif
        const auto device_key = std::pair{device.type, device.index};
        if (!device_page_tables_.contains(device_key)) {
            if (device.is_metal()) {
                device_page_tables_.emplace(device_key, metal_table(host_page_table_, maximum_sequences_, logical_pages_per_sequence_));
            } else {
#ifdef TYLOINFER_NATIVE_CUDA
                MfqCudaGuard guard(device.index);
                auto page_table = ::tyloi::cuda::empty({maximum_sequences_, logical_pages_per_sequence_},
                    TensorOptions().device(device).dtype(kInt32));
                page_table.fill_(-1);
                device_page_tables_.emplace(device_key, std::move(page_table));
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
        if (device.is_cuda()) {
            MfqCudaGuard guard(device.index);
            auto options = TensorOptions().device(device).dtype(kInt64);
            pool.k_chunk_ptrs = ::tyloi::cuda::zeros({maximum_chunks}, options);
            pool.v_chunk_ptrs = ::tyloi::cuda::zeros({maximum_chunks}, options);
        }
#endif
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
    const size_t bytes = rows * row_width * sizeof(std::int32_t);
    for (auto & [device, table] : device_page_tables_) {
        if (device.first == DeviceType::metal) {
            table = metal_table(host_page_table_, maximum_sequences_, logical_pages_per_sequence_);
        } else {
#ifdef TYLOINFER_NATIVE_CUDA
            MfqCudaGuard guard(device.second);
            const auto stream = mfq_get_current_cuda_stream();
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(
                table.dense().data_ptr<std::int32_t>(), host_page_table_.data(),
                bytes, cudaMemcpyHostToDevice, stream));
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
                MfqCudaGuard guard(layer.layout.device.index);
                auto options = TensorOptions().device(layer.layout.device).dtype(layer.dtype);
                chunk.k = ::tyloi::cuda::empty(
                    {static_cast<int64_t>(chunk_pages), layer.layout.heads, kPageSize, layer.key_width}, options);
                chunk.v = ::tyloi::cuda::empty(
                    {static_cast<int64_t>(chunk_pages), layer.layout.heads, kPageSize, layer.value_width}, options);
#endif
            }
            pending.push_back(std::move(chunk));
        }
        const size_t chunk_index = storage_pages_ / kPagesPerChunk;
        for (size_t index = 0; index < layers_.size(); ++index) {
            auto & layer = layers_[index];
            auto & chunk = pending[index];
            if (layer.layout.device.is_cuda()) {
#ifdef TYLOINFER_NATIVE_CUDA
                const int64_t k_pointer = reinterpret_cast<int64_t>(chunk.k.dense().data_ptr());
                const int64_t v_pointer = reinterpret_cast<int64_t>(chunk.v.dense().data_ptr());
                MfqCudaGuard guard(layer.layout.device.index);
                const auto stream = mfq_get_current_cuda_stream();
                TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(
                    layer.k_chunk_ptrs.dense().data_ptr<int64_t>() + chunk_index,
                    &k_pointer, sizeof(k_pointer), cudaMemcpyHostToDevice, stream));
                TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(
                    layer.v_chunk_ptrs.dense().data_ptr<int64_t>() + chunk_index,
                    &v_pointer, sizeof(v_pointer), cudaMemcpyHostToDevice, stream));
                TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
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

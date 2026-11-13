#include "engine/backends/metal/native/tq_pool.h"
#include "engine/backends/metal/native/tensor/buffer.h"

#include <mlx/allocator.h>
#include <algorithm>
#include <array>
#include <climits>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <sys/mman.h>
#include <unistd.h>

namespace tyloi::metal {
using namespace mlx::core;

std::size_t pool_page() {
    static const auto page = std::max(std::size_t{4096}, std::size_t(getpagesize()));
    return page;
}

namespace {
struct PoolBuffer {
    std::shared_ptr<mfq::metal::detail::MlxPinnedPoolStorage> storage;
    allocator::Buffer buffer{nullptr};
    ~PoolBuffer() { if (buffer.ptr()) allocator::release(buffer); }
};
}

void TqPool::insert_free(std::size_t offset, std::size_t bytes) {
    while(bytes) {
        const auto part=std::min(bytes,chunk_bytes-offset%chunk_bytes);
        const auto position=sizes.emplace(part,offset);
        try { free.emplace(offset,FreeRange{part,position}); }
        catch (...) { sizes.erase(position);throw; }
        offset+=part;bytes-=part;
    }
}

std::uint64_t TqPool::address(std::size_t offset) const {
    if(offset>=capacity)throw std::out_of_range("TQ chunk offset exceeds capacity");
    const auto& chunk=chunks.at(offset/chunk_bytes);
    const auto buffer=chunk.buffer();
    const auto address=static_cast<const MTL::Buffer*>(buffer.ptr())->gpuAddress()+offset%chunk_bytes;
    return address;
}

void TqPool::erase_free(std::map<std::size_t, FreeRange>::iterator range) {
    sizes.erase(range->second.size_entry);
    free.erase(range);
}

void TqPool::reset_free(std::size_t offset) {
    free.clear(); sizes.clear();
    if (offset < capacity) insert_free(offset, capacity - offset);
}

void TqPool::allocate_storage(std::size_t bytes) {
    capacity = bytes / pool_page() * pool_page();
    if (!capacity || capacity / 4096 > std::size_t(INT_MAX))
        throw std::invalid_argument("invalid TQ expert pool capacity");
    storage = std::make_shared<mfq::metal::detail::MlxPinnedPoolStorage>();
    storage->bytes = capacity;
    storage->data = mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (storage->data == MAP_FAILED) throw std::bad_alloc();
    bind_storage(capacity);
    reset_free(0);
}

void TqPool::bind_storage(std::size_t bytes) {
    if(!storage||!bytes||bytes>storage->bytes||bytes%pool_page())
        throw std::invalid_argument("invalid TQ expert pool view");
    if(bytes>capacity)madvise(static_cast<std::byte*>(storage->data)+capacity,bytes-capacity,MADV_FREE_REUSE);
    const auto count=(bytes+chunk_bytes-1)/chunk_bytes;
    std::size_t keep=0;
    while(keep<chunks.size()&&keep<count&&chunks[keep].nbytes()==std::min(chunk_bytes,bytes-keep*chunk_bytes))++keep;
    chunks.erase(chunks.begin()+keep,chunks.end());
    for(std::size_t part=keep;part<count;++part) {
        const auto length=std::min(chunk_bytes,bytes-part*chunk_bytes);
        auto view=std::make_shared<PoolBuffer>();view->storage=storage;
        view->buffer=allocator::make_buffer(static_cast<std::byte*>(storage->data)+part*chunk_bytes,length);
        if(!view->buffer.ptr())throw std::runtime_error("cannot register TQ expert chunk with Metal");
        chunks.push_back(mfq::metal::mlx_buffer_view(view->buffer,Shape{int(length/4096),1024},uint32,view));
    }
    lifetime=std::make_shared<TqPoolLifetime>();lifetime->chunks=chunks;
    const std::array<std::uint64_t, 8> handle{};
    lifetime->handle = array(handle.data(), Shape{8}, uint64);
    words = mfq::metal::mlx_buffer_view(lifetime->handle.buffer(), lifetime->handle.shape(), uint64, lifetime);
    if (!storage->pin_prefix(bytes)) {
        if (storage->locked_bytes > bytes)
            throw std::runtime_error("cannot release TQ expert page locks");
        std::fprintf(stderr, "TQ expert page locking unavailable: requested=%zu locked=%zu error=%d\n",
            bytes, storage->locked_bytes, storage->lock_error);
    }
    capacity=bytes;data=static_cast<std::byte*>(storage->data);
}

void TqPool::discard_tail(std::size_t previous_capacity) {
    if (capacity < previous_capacity)
        madvise(data + capacity, previous_capacity - capacity, MADV_FREE_REUSABLE);
}

bool TqPool::fits(std::size_t bytes, std::size_t limit) const {
    auto it = sizes.lower_bound(bytes);
    if (limit == SIZE_MAX) return it != sizes.end();
    for (; it != sizes.end(); ++it)
        if (it->second <= limit && bytes <= limit - it->second) return true;
    return false;
}

TqRange TqPool::take(std::size_t bytes, std::size_t limit) {
    auto it = sizes.lower_bound(bytes);
    if (limit != SIZE_MAX) while (it != sizes.end()
        && (it->second > limit || bytes > limit - it->second)) ++it;
    if (it == sizes.end()) throw std::runtime_error("TQ expert pool has no contiguous range");
    const TqRange result{it->second, bytes};
    const auto remaining = it->first - bytes;
    auto size_node = sizes.extract(it);
    auto range_node = free.extract(result.offset);
    if (remaining) {
        size_node.key() = remaining; size_node.mapped() = result.offset + bytes;
        const auto position = sizes.insert(std::move(size_node));
        range_node.key() = result.offset + bytes;
        range_node.mapped() = FreeRange{remaining, position};
        free.insert(std::move(range_node));
    }
    return result;
}

void TqPool::release(TqRange range) {
    auto next = free.lower_bound(range.offset);
    if (next != free.begin()) {
        const auto previous = std::prev(next);
        if (previous->first + previous->second.bytes == range.offset
            && previous->first/chunk_bytes==range.offset/chunk_bytes) {
            range.offset = previous->first;
            range.bytes += previous->second.bytes;
            erase_free(previous);
        }
    }
    if (next != free.end() && range.offset + range.bytes == next->first
        && range.offset/chunk_bytes==next->first/chunk_bytes) {
        range.bytes += next->second.bytes;
        erase_free(next);
    }
    insert_free(range.offset, range.bytes);
}

std::optional<TqRange> TqPool::relocate(TqRange range, std::size_t limit) {
    if (fits(range.bytes, limit)) {
        const auto target = take(range.bytes, limit);
        std::memmove(data + target.offset, data + range.offset, range.bytes);
        release(range);
        return target;
    }
    const auto next = free.lower_bound(range.offset);
    if (next == free.begin()) return std::nullopt;
    const auto previous = std::prev(next);
    if (previous->first/chunk_bytes!=range.offset/chunk_bytes
        || previous->first + previous->second.bytes != range.offset
        || previous->first > limit || range.bytes > limit - previous->first)
        return std::nullopt;
    const TqRange target{previous->first, range.bytes};
    std::memmove(data + target.offset, data + range.offset, range.bytes);
    erase_free(previous);
    release({target.offset + target.bytes, range.offset - target.offset});
    return target;
}

} // namespace tyloi::metal

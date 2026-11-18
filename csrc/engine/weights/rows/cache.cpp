#include "engine/weights/rows/cache.h"
#include "engine/backends/cuda/native/tensor/tensor.h"

#include <cstring>
#include <iterator>
#include <list>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tyloi::weights {
namespace {
constexpr std::size_t kCacheBytes = 8u << 20;

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
} // namespace

struct RowCache::Impl {
    using Entry = std::pair<std::uint64_t, std::shared_ptr<const Image>>;
    std::mutex mutex;
    std::list<Entry> lru;
    std::unordered_map<std::uint64_t, std::list<Entry>::iterator> entries;
    std::size_t used = 0;

    std::size_t evict() {
        auto it = std::prev(lru.end());
        const auto bytes = it->second->bytes.capacity();
        const std::weak_ptr<const Image> owner = it->second;
        entries.erase(it->first);
        used -= bytes + 128;
        lru.erase(it);
        return owner.expired() ? bytes : 0;
    }

    std::shared_ptr<const Image> find(std::uint64_t key) {
        const auto found = entries.find(key);
        if (found == entries.end()) return {};
        lru.splice(lru.begin(), lru, found->second);
        return found->second->second;
    }
};

RowCache::RowCache() : impl_(std::make_unique<Impl>()) {}
RowCache::~RowCache() = default;

std::size_t RowCache::copy(std::uint64_t key, std::byte* out, std::size_t capacity) {
    std::shared_ptr<const Image> image;
    {
        std::lock_guard lock(impl_->mutex);
        image = impl_->find(key);
    }
    if (!image) return 0;
    const auto size = image->bytes.size();
    if (capacity < size || (size && !out)) throw std::invalid_argument("row cache output is too small");
    std::memcpy(out, image->bytes.data(), size);
    return size;
}

void RowCache::insert(std::uint64_t key, const std::byte* source, std::size_t bytes) {
    if (!bytes || bytes > kCacheBytes - 128) return;
    {
        std::lock_guard lock(impl_->mutex);
        if (impl_->find(key)) return;
    }
    auto data = std::make_shared<Image>(source, bytes);
    std::lock_guard lock(impl_->mutex);
    if (impl_->find(key)) return;
    const auto backing = data->bytes.capacity();
    if (backing > kCacheBytes - 128) return;
    const auto cost = backing + 128;
    while (impl_->used > kCacheBytes - cost) impl_->evict();
    impl_->lru.emplace_front(key, std::move(data));
    try {
        impl_->entries.emplace(key, impl_->lru.begin());
    } catch (...) {
        impl_->lru.pop_front();
        throw;
    }
    impl_->used += cost;
}

std::size_t RowCache::reclaim(std::size_t bytes) {
    std::lock_guard lock(impl_->mutex);
    std::size_t released = 0;
    while (released < bytes && !impl_->lru.empty()) released += impl_->evict();
    return released;
}

std::shared_ptr<RowCache> make_row_cache() {
    auto cache = std::shared_ptr<RowCache>(new RowCache());
    register_host_cache(cache);
    return cache;
}

} // namespace tyloi::weights

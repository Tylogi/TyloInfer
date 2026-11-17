#pragma once
#include "engine/memory/host.h"
#include "engine/weights/encoding/quant/nint_rows.h"
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace tyloi {
// Canonical encoded rows stay immutable while a consumer holds their lease.
class NintRowCache final : public HostCache {
public:
    explicit NintRowCache(std::size_t capacity) : capacity_(capacity) {}
    std::shared_ptr<const NintRowBatch> find(std::uint64_t key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = entries_.find(key);
        if (found == entries_.end()) return {};
        recency_.splice(recency_.begin(), recency_, found->second);
        return found->second->value;
    }
    void insert(std::uint64_t key, std::shared_ptr<const NintRowBatch> value) {
        if (!capacity_ || !value) return;
        std::lock_guard<std::mutex> lock(mutex_);
        if (const auto found = entries_.find(key); found != entries_.end()) {
            recency_.splice(recency_.begin(), recency_, found->second);
            return;
        }
        recency_.push_front({key, std::move(value)});
        try { entries_.emplace(key, recency_.begin()); }
        catch (...) { recency_.pop_front(); throw; }
        if (entries_.size() > capacity_) {
            entries_.erase(recency_.back().key);
            recency_.pop_back();
        }
    }
    std::size_t reclaim(std::size_t bytes) override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t released = 0;
        while (released < bytes && !recency_.empty()) {
            const auto& entry = recency_.back();
            const auto key = entry.key;
            const auto payload_bytes = backing_bytes(*entry.value);
            // Keep only a weak observation while removing the cache owner.
            // This makes the accounting reflect the payload's actual
            // destruction rather than a use-count prediction.
            const std::weak_ptr<const NintRowBatch> payload = entry.value;
            entries_.erase(key);
            recency_.pop_back();
            if (payload.expired()) {
                if (payload_bytes > std::numeric_limits<std::size_t>::max() - released)
                    released = std::numeric_limits<std::size_t>::max();
                else released += payload_bytes;
            }
        }
        return released;
    }
private:
    static std::size_t backing_bytes(const NintRowBatch& batch) noexcept {
        const auto packed = batch.packed().capacity();
        const auto descriptors = batch.descriptors().capacity();
        if (descriptors > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t))
            return std::numeric_limits<std::size_t>::max();
        const auto descriptor_bytes = descriptors * sizeof(std::uint32_t);
        if (packed > std::numeric_limits<std::size_t>::max() - descriptor_bytes)
            return std::numeric_limits<std::size_t>::max();
        return packed + descriptor_bytes;
    }

    struct Entry { std::uint64_t key; std::shared_ptr<const NintRowBatch> value; };
    std::size_t capacity_;
    mutable std::list<Entry> recency_;
    mutable std::unordered_map<std::uint64_t, std::list<Entry>::iterator> entries_;
    mutable std::mutex mutex_;
};
}

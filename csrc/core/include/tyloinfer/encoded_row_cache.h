#pragma once
#include "host_cache_registry.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace mfq {
// Immutable encoded rows share a byte-bounded LRU. Leases keep evicted data
// alive, and accounting follows the last lease rather than map membership.
template<class Value>
class EncodedRowCache final:public ReclaimableHostCache {
    using Accounting=std::function<void(std::size_t,bool)>;
    struct Payload {
        Value value;
        std::size_t bytes;
        Accounting accounting;
        Payload(Value next,std::size_t size,Accounting observer)
            :value(std::move(next)),bytes(size),accounting(std::move(observer)) {
            if(accounting)accounting(bytes,true);
        }
        ~Payload(){if(accounting)accounting(bytes,false);}
    };
    struct Entry {std::uint64_t key;std::shared_ptr<const Payload> payload;};
    static constexpr std::size_t entry_bytes=128;
    const std::size_t capacity_;
    std::function<std::size_t(const Value&)> size_;
    Accounting accounting_;
    std::list<Entry> recent_;
    std::unordered_map<std::uint64_t,typename std::list<Entry>::iterator> entries_;
    std::size_t used_=0;
    mutable std::mutex mutex_;
    static std::shared_ptr<const Value> lease(const std::shared_ptr<const Payload>& payload) {
        return {payload,&payload->value};
    }
public:
    EncodedRowCache(std::size_t capacity,std::function<std::size_t(const Value&)> size,
            Accounting accounting={})
        :capacity_(capacity),size_(std::move(size)),accounting_(std::move(accounting)) {}
    EncodedRowCache(const EncodedRowCache&)=delete;
    EncodedRowCache& operator=(const EncodedRowCache&)=delete;
    std::shared_ptr<const Value> find(std::uint64_t key) {
        std::lock_guard lock(mutex_);
        const auto found=entries_.find(key);
        if(found==entries_.end())return {};
        recent_.splice(recent_.begin(),recent_,found->second);
        return lease(found->second->payload);
    }
    void insert(std::uint64_t key,Value value) {
        const auto bytes=size_(value);
        if(capacity_<entry_bytes || bytes>capacity_-entry_bytes)return;
        const auto cost=bytes+entry_bytes;
        std::lock_guard lock(mutex_);
        if(const auto found=entries_.find(key);found!=entries_.end()) {
            recent_.splice(recent_.begin(),recent_,found->second);return;
        }
        while(used_>capacity_-cost) {
            used_-=recent_.back().payload->bytes+entry_bytes;
            entries_.erase(recent_.back().key);recent_.pop_back();
        }
        auto payload=std::make_shared<Payload>(std::move(value),bytes,accounting_);
        recent_.push_front({key,std::move(payload)});
        try {entries_.emplace(key,recent_.begin());}
        catch(...) {recent_.pop_front();throw;}
        used_+=cost;
    }
    std::size_t reclaim(std::size_t bytes)override {
        std::lock_guard lock(mutex_);
        std::size_t released=0;
        while(released<bytes && !recent_.empty()) {
            const auto& entry=recent_.back();const auto size=entry.payload->bytes;
            const bool last=entry.payload.use_count()==1;
            used_-=size+entry_bytes;entries_.erase(entry.key);recent_.pop_back();
            if(last)released+=size;
        }
        return released;
    }
};
}

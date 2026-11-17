#include "tyloinfer/host_cache_registry.h"
#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace mfq {
namespace {
struct Registry {
    std::mutex mutex;
    std::vector<std::weak_ptr<ReclaimableHostCache>> caches;
};
Registry& registry(){static Registry instance;return instance;}
}
void register_host_cache(const std::shared_ptr<ReclaimableHostCache>& cache) {
    if(!cache)throw std::invalid_argument("cannot register an absent host cache");
    auto& state=registry();std::lock_guard lock(state.mutex);
    state.caches.erase(std::remove_if(state.caches.begin(),state.caches.end(),
        [](const auto& entry){return entry.expired();}),state.caches.end());
    for(const auto& entry:state.caches)if(entry.lock()==cache)return;
    state.caches.push_back(cache);
}
std::size_t reclaim_host_caches(std::size_t bytes)noexcept {
    if(!bytes)return 0;
    std::size_t released=0;
    try {
        std::vector<std::shared_ptr<ReclaimableHostCache>> caches;
        {
            auto& state=registry();std::lock_guard lock(state.mutex);
            for(auto entry=state.caches.begin();entry!=state.caches.end();) {
                if(auto cache=entry->lock()){caches.push_back(std::move(cache));++entry;}
                else entry=state.caches.erase(entry);
            }
        }
        // Retain owners through callbacks without holding the registry lock.
        for(const auto& cache:caches) {
            try {
                const auto reclaimed=cache->reclaim(bytes-released);
                released+=std::min(reclaimed,std::numeric_limits<std::size_t>::max()-released);
            }catch(...) {}
            if(released>=bytes)break;
        }
    }catch(...) {}
    return released;
}
}

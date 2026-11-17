#pragma once
#include <cstddef>
#include <memory>

namespace mfq {
class ReclaimableHostCache {
public:
    virtual ~ReclaimableHostCache()=default;
    // Return bytes actually released; outstanding leases remain valid.
    virtual std::size_t reclaim(std::size_t bytes)=0;
};
void register_host_cache(const std::shared_ptr<ReclaimableHostCache>&);
std::size_t reclaim_host_caches(std::size_t bytes)noexcept;
}

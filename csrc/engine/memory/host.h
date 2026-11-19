#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

namespace tyloi {
struct HostMemorySample {
    std::uint64_t resident_bytes = 0;
    std::uint64_t available_bytes = 0;
    std::optional<std::uint64_t> commit_available_bytes;
    std::uint64_t total_bytes = 0;
    std::optional<std::uint64_t> footprint_bytes;
    std::uint64_t available_host_bytes() const noexcept {
        return commit_available_bytes && *commit_available_bytes < available_bytes
            ? *commit_available_bytes : available_bytes;
    }
    // System/commit availability already excludes live allocations.
    std::size_t headroom(std::uint64_t process_limit = 0) const noexcept {
        auto available = available_host_bytes();
        if (process_limit) {
            const auto remaining = resident_bytes < process_limit ? process_limit - resident_bytes : 0;
            if (remaining < available) available = remaining;
        }
        const auto maximum = std::numeric_limits<std::size_t>::max();
        return static_cast<std::size_t>(available > maximum ? maximum : available);
    }
};
HostMemorySample sample_host_memory();
void trim_host_cache() noexcept;
/** Limit an optional cache while allowing another owner to grow into free RAM. */
inline std::size_t shared_host_cache_limit(std::size_t ordinary, std::size_t used,
        std::size_t available, std::size_t requested) noexcept {
    if (!requested) return ordinary;
    if (available >= requested) {
        const auto retained = std::min(ordinary, used);
        return retained + std::min(ordinary - retained, available - requested);
    }
    return std::min(ordinary, used - std::min(used, requested - available));
}

class HostCache {
public:
    virtual ~HostCache() = default;
    // Return only released backing bytes; outstanding leases remain valid.
    virtual std::size_t reclaim(std::size_t bytes) = 0;
};
void register_host_cache(const std::shared_ptr<HostCache>& cache);
std::size_t reclaim_host_caches(std::size_t bytes) noexcept;
// Reclaim under pressure; the caller's existing admission checks still apply.
void reclaim_host(std::size_t bytes, std::uint64_t process_limit = 0) noexcept;
} // namespace tyloi

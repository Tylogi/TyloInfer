#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace mfq {
struct HostMemorySample {
    std::uint64_t resident_bytes = 0;
    std::uint64_t available_bytes = 0;
    std::optional<std::uint64_t> commit_available_bytes;
    std::uint64_t available_host_bytes() const noexcept {
        return commit_available_bytes && *commit_available_bytes < available_bytes
            ? *commit_available_bytes : available_bytes;
    }
};
// System/commit availability already excludes live allocations. Subtract
// resident bytes only from an explicit process limit.
inline std::size_t host_memory_headroom(const HostMemorySample& sample,
        std::uint64_t process_limit = 0) noexcept {
    auto available = sample.available_host_bytes();
    if (process_limit) {
        const auto remaining = sample.resident_bytes < process_limit
            ? process_limit - sample.resident_bytes : 0;
        if (remaining < available) available = remaining;
    }
    const auto maximum = std::numeric_limits<std::size_t>::max();
    return static_cast<std::size_t>(available > maximum ? maximum : available);
}
HostMemorySample sample_host_memory();
}

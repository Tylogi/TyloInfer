#include "host.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <string>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <malloc/malloc.h>
#include <sys/sysctl.h>
#endif

namespace {
struct Registry {
    std::mutex mutex;
    std::vector<std::weak_ptr<tyloi::HostCache>> caches;
};
Registry& registry() { static Registry instance; return instance; }

#if defined(__linux__) || defined(__APPLE__)
std::uint64_t bytes(std::uint64_t count, std::uint64_t size) {
    if (size && count > std::numeric_limits<std::uint64_t>::max() / size)
        throw std::overflow_error("host memory measurement overflow");
    return count * size;
}
#endif
}

void tyloi::register_host_cache(const std::shared_ptr<HostCache>& cache) {
    if (!cache) throw std::invalid_argument("cannot register an absent host cache");
    auto& state = registry();
    std::lock_guard lock(state.mutex);
    state.caches.erase(std::remove_if(state.caches.begin(), state.caches.end(),
        [](const auto& entry) { return entry.expired(); }), state.caches.end());
    for (const auto& entry : state.caches)
        if (entry.lock() == cache) return;
    state.caches.push_back(cache);
}

std::size_t tyloi::reclaim_host_caches(std::size_t bytes) noexcept {
    if (!bytes) return 0;
    std::size_t released = 0;
    try {
        std::vector<std::shared_ptr<HostCache>> caches;
        {
            auto& state = registry();
            std::lock_guard lock(state.mutex);
            for (auto entry = state.caches.begin(); entry != state.caches.end();) {
                if (auto cache = entry->lock()) { caches.push_back(std::move(cache)); ++entry; }
                else entry = state.caches.erase(entry);
            }
        }
        // Cache reclamation can release other owners; keep the registry unlocked.
        for (const auto& cache : caches) {
            try {
                const auto reclaimed = cache->reclaim(bytes - released);
                released += std::min(reclaimed, std::numeric_limits<std::size_t>::max() - released);
            } catch (...) {}
            if (released >= bytes) break;
        }
    } catch (...) {}
    return released;
}

std::size_t tyloi::reclaim_host(std::size_t bytes, std::uint64_t process_limit) noexcept {
    if (!bytes) return 0;
    try {
        const auto available = sample_host_memory().headroom(process_limit);
        if (available >= bytes) return 0;
        const auto reclaimed = reclaim_host_caches(bytes - available);
        trim_host_cache();
        return reclaimed;
    } catch (...) {}
    return 0;
}

tyloi::HostMemorySample tyloi::sample_host_memory() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS process{};
    process.cb = sizeof(process);
    MEMORYSTATUSEX system{};
    system.dwLength = sizeof(system);
    if (!K32GetProcessMemoryInfo(GetCurrentProcess(), &process, sizeof(process)) ||
        !GlobalMemoryStatusEx(&system))
        throw std::runtime_error("cannot measure host memory");
    return {process.WorkingSetSize, system.ullAvailPhys, system.ullAvailPageFile,
            system.ullTotalPhys, std::nullopt};
#elif defined(__linux__)
    std::uint64_t virtual_pages = 0, resident_pages = 0;
    std::ifstream status("/proc/self/statm");
    const auto page_size = sysconf(_SC_PAGESIZE);
    if (!(status >> virtual_pages >> resident_pages) || page_size <= 0)
        throw std::runtime_error("cannot measure process resident memory");
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    std::optional<std::uint64_t> total, available;
    while (std::getline(meminfo, line)) {
        constexpr auto total_key = "MemTotal:";
        constexpr auto available_key = "MemAvailable:";
        const bool is_total = line.compare(0, sizeof("MemTotal:") - 1, total_key) == 0;
        const bool is_available = line.compare(0, sizeof("MemAvailable:") - 1, available_key) == 0;
        if (!is_total && !is_available) continue;
        std::istringstream entry(line.substr(is_total ? sizeof("MemTotal:") - 1 : sizeof("MemAvailable:") - 1));
        std::uint64_t value = 0;
        std::string unit;
        if (!(entry >> value >> unit) || unit != "kB")
            throw std::runtime_error("invalid physical memory data in /proc/meminfo");
        (is_total ? total : available) = bytes(value, 1024);
        if (total && available) break;
    }
    if (!total || !*total || !available)
        throw std::runtime_error("cannot measure physical memory");
    return {bytes(resident_pages, static_cast<std::uint64_t>(page_size)),
            *available, std::nullopt, *total, std::nullopt};
#elif defined(__APPLE__)
    std::uint64_t total = 0;
    std::size_t total_size = sizeof(total);
    if (sysctlbyname("hw.memsize", &total, &total_size, nullptr, 0) != 0 ||
        total_size != sizeof(total) || !total)
        throw std::runtime_error("cannot measure total physical memory");
    const auto host = mach_host_self();
    if (host == MACH_PORT_NULL) throw std::runtime_error("cannot obtain Mach host port");
    vm_size_t page_size = 0;
    vm_statistics64_data_t statistics{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    const bool valid = host_page_size(host, &page_size) == KERN_SUCCESS &&
        host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&statistics), &count) == KERN_SUCCESS;
    const auto released = mach_port_deallocate(mach_task_self(), host);
    if (!valid || released != KERN_SUCCESS || !page_size || count < HOST_VM_INFO64_COUNT)
        throw std::runtime_error("cannot measure available physical memory");
    task_vm_info_data_t process{};
    mach_msg_type_number_t process_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&process),
                  &process_count) != KERN_SUCCESS || process_count < TASK_VM_INFO_REV1_COUNT)
        throw std::runtime_error("cannot measure process resident memory");
    const auto pages = std::uint64_t(statistics.free_count) + std::uint64_t(statistics.inactive_count);
    return {std::uint64_t(process.resident_size), std::min(total, bytes(pages, page_size)),
            std::nullopt, total, std::uint64_t(process.phys_footprint)};
#else
    throw std::runtime_error("host memory measurement is unavailable on this platform");
#endif
}

void tyloi::trim_host_cache() noexcept {
#if defined(__GLIBC__)
    (void)malloc_trim(0);
#elif defined(__APPLE__)
    (void)malloc_zone_pressure_relief(malloc_default_zone(), 0);
#endif
}

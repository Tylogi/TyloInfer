#include "memory.h"

#include <algorithm>
#include <mach/mach.h>
#include <sys/sysctl.h>

#include <mlx/backend/metal/device.h>
#include <mlx/memory.h>

namespace tyloi::metal {
namespace {

template <typename Device>
static bool available(Device& device) {
    if constexpr (requires { device.residency_sets(); }) {
        return device.residency_sets().enabled();
    } else {
        return device.residency_set().mtl_residency_set() != nullptr;
    }
}

template <typename Device>
static std::size_t allocated_bytes(Device& device) {
    if constexpr (requires { device.residency_sets(); }) {
        return device.residency_sets().wired_size();
    } else if (__builtin_available(macOS 15, *)) {
        if (const auto* set = device.residency_set().mtl_residency_set()) {
            return set->allocatedSize();
        }
    }
    return 0;
}

}  // namespace

Memory memory() {
    std::uint64_t total = 0;
    std::size_t size = sizeof(total);
    if (sysctlbyname("hw.memsize", &total, &size, nullptr, 0) != 0 ||
        size != sizeof(total) || total == 0 ||
        total > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("cannot determine physical memory size");
    }

    const auto host = mach_host_self();
    if (host == MACH_PORT_NULL) {
        throw std::runtime_error("cannot obtain Mach host port");
    }
    vm_size_t page_size = 0;
    vm_statistics64_data_t statistics{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    const bool success = host_page_size(host, &page_size) == KERN_SUCCESS &&
        host_statistics64(host, HOST_VM_INFO64,
            reinterpret_cast<host_info64_t>(&statistics), &count) == KERN_SUCCESS;
    const auto deallocated = mach_port_deallocate(mach_task_self(), host);
    if (!success || deallocated != KERN_SUCCESS || page_size == 0 ||
        count < HOST_VM_INFO64_COUNT) {
        throw std::runtime_error("cannot determine available physical memory");
    }
    const auto pages = static_cast<std::uint64_t>(statistics.free_count) +
        static_cast<std::uint64_t>(statistics.inactive_count);
    if (pages > std::numeric_limits<std::size_t>::max() / page_size) {
        throw std::overflow_error("available physical memory exceeds size_t");
    }
    return {active_memory(), cache_memory(), wired_memory(), memory_limit(),
        std::min(static_cast<std::size_t>(pages) * page_size,
                 static_cast<std::size_t>(total)),
        static_cast<std::size_t>(total)};
}

std::size_t memory_limit() { return mlx::core::get_memory_limit(); }
void set_memory_limit(std::size_t bytes) { mlx::core::set_memory_limit(bytes); }

PackedLayout packed_layout(std::size_t elements) {
    if (elements == 0) {
        throw std::invalid_argument(
            "packed storage must contain at least one element");
    }
    constexpr auto max_axis = static_cast<std::size_t>(
        std::numeric_limits<std::int32_t>::max());
    if (elements <= max_axis) {
        return {1, static_cast<int>(elements)};
    }
    if (elements / max_axis > max_axis) {
        throw std::length_error(
            "packed storage exceeds the MLX multi-axis shape range");
    }

    const auto minimum_rows =
        elements / max_axis + (elements % max_axis != 0);
    for (auto rows = minimum_rows; rows <= elements / rows; ++rows) {
        if (elements % rows != 0) {
            continue;
        }
        const auto columns = elements / rows;
        if (rows <= max_axis && columns <= max_axis) {
            return {
                static_cast<int>(rows),
                static_cast<int>(columns),
            };
        }
    }
    throw std::length_error(
        "packed storage cannot be represented by exact MLX int32 axes");
}

bool residency_available() {
    return available(mlx::core::metal::device(mlx::core::Device::gpu));
}

std::size_t active_memory() { return mlx::core::get_active_memory(); }
std::size_t cache_memory() { return mlx::core::get_cache_memory(); }
std::size_t peak_memory() { return mlx::core::get_peak_memory(); }
std::size_t wired_memory() {
    return allocated_bytes(mlx::core::metal::device(mlx::core::Device::gpu));
}
void set_cache_limit(std::size_t limit) { mlx::core::set_cache_limit(limit); }
void set_wired_limit(std::size_t limit) { mlx::core::set_wired_limit(limit); }
void clear_cache() { mlx::core::clear_cache(); }

}  // namespace tyloi::metal

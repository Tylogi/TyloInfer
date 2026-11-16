#include "memory.h"
#include "engine/memory/host.h"

#include <algorithm>
#include <atomic>

#include <mlx/backend/metal/device.h>
#include <mlx/memory.h>

namespace tyloi::metal {
namespace {

std::atomic<std::size_t> configured_wired_limit{0};

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
    const auto host = tyloi::sample_host_memory();
    const auto footprint = host.footprint_bytes.value();
    if (host.total_bytes > std::numeric_limits<std::size_t>::max() ||
        host.available_host_bytes() > std::numeric_limits<std::size_t>::max() ||
        footprint > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("host memory measurement exceeds size_t");
    }
    return {active_memory(), cache_memory(), wired_memory(), memory_limit(),
        static_cast<std::size_t>(host.available_host_bytes()),
        static_cast<std::size_t>(host.total_bytes), static_cast<std::size_t>(footprint)};
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
std::size_t wired_limit() { return configured_wired_limit.load(std::memory_order_relaxed); }
void set_wired_limit(std::size_t limit) {
    if (limit == 0 || !residency_available()) {
        throw std::runtime_error("Metal memory wiring is unavailable");
    }
    mlx::core::set_wired_limit(limit);
    configured_wired_limit.store(limit, std::memory_order_relaxed);
}
void refresh_residency() {
    const auto limit = wired_limit();
    if (limit == 0) return;
    const auto active = active_memory();
    if (active > limit) {
        throw std::runtime_error(
            "Metal active memory exceeds the wired budget; "
            "reduce resident experts or unload an idle model");
    }
    mlx::core::set_wired_limit(limit - 1);
    mlx::core::set_wired_limit(limit);
    if (wired_memory() < active) {
        throw std::runtime_error(
            "Metal residency does not cover the loaded allocations; "
            "reduce resident experts or unload an idle model");
    }
}
void clear_cache() { mlx::core::clear_cache(); }

}  // namespace tyloi::metal

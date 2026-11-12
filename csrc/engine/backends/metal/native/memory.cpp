#include "memory.h"

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

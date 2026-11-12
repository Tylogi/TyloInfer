#pragma once

#include "mlx_weight_residency.h"
#include "engine/backends/metal/native/memory.h"

#include <atomic>
#include <cstddef>
#include <stdexcept>

namespace mfq::metal {

class MlxMemoryResidency {
public:
    static void configure(std::size_t limit) {
        if (limit == 0 || !tyloi::metal::residency_available()) {
            throw std::runtime_error("Metal memory wiring is unavailable");
        }
        tyloi::metal::set_wired_limit(limit);
        limit_bytes().store(limit, std::memory_order_relaxed);
        MlxWeightResidency::begin_load();
    }

    static void refresh() {
        const auto limit = configured_limit();
        if (limit == 0) return;
        const auto active = tyloi::metal::active_memory();
        if (active > limit) {
            throw std::runtime_error(
                "Metal active memory exceeds the wired budget; "
                "reduce resident experts or unload an idle model");
        }
        tyloi::metal::set_wired_limit(limit - 1);
        tyloi::metal::set_wired_limit(limit);
        if (wired_bytes() < active) {
            throw std::runtime_error(
                "Metal residency does not cover the loaded allocations; "
                "reduce resident experts or unload an idle model");
        }
    }

    static std::size_t configured_limit() {
        return limit_bytes().load(std::memory_order_relaxed);
    }

    static std::size_t wired_bytes() {
        return tyloi::metal::wired_memory();
    }

private:
    static std::atomic<std::size_t>& limit_bytes() {
        static std::atomic<std::size_t> value{0};
        return value;
    }
};

}

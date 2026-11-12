#pragma once

#include <algorithm>
#include <cstdlib>
#include <string_view>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

#include "engine/backends/metal/ops/stream.h"
#include "engine/executor.h"

namespace mfq::metal::detail {

inline bool component_profile_requested() noexcept {
    static const bool requested = [] {
        const char* value =
            std::getenv("TYLOINFER_METAL_PROFILE_COMPONENTS");
        return value != nullptr && std::atoi(value) != 0;
    }();
    return requested;
}

inline int component_profile_steps() noexcept {
    static const int steps = [] {
        const char* value =
            std::getenv("TYLOINFER_METAL_PROFILE_STEPS");
        if (value == nullptr) {
            return 1;
        }
        return std::max(1, std::atoi(value));
    }();
    return steps;
}

inline int component_profile_skip_steps() noexcept {
    static const int steps = [] {
        const char* value =
            std::getenv("TYLOINFER_METAL_PROFILE_SKIP_STEPS");
        return value == nullptr
            ? 0
            : std::max(0, std::atoi(value));
    }();
    return steps;
}

inline void eval_with_timing(mlx::core::array& value) {
    tyloi::profiling::measure([&value]() { tyloi::metal::ops::eval(value); });
}

inline void eval_with_timing(
    std::vector<mlx::core::array> values) {
    tyloi::profiling::measure(
        [values = std::move(values)]() mutable {
            tyloi::metal::ops::eval(std::move(values));
        });
}

inline void profile_eval(
    std::string_view component,
    mlx::core::array& value) {
    tyloi::profiling::component(
        component,
        [&value]() { tyloi::metal::ops::eval(value); });
}

inline void profile_eval(
    std::string_view component,
    std::vector<mlx::core::array> values) {
    tyloi::profiling::component(
        component,
        [values = std::move(values)]() mutable {
            tyloi::metal::ops::eval(std::move(values));
        });
}

}  // namespace mfq::metal::detail

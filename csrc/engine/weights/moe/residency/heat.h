#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace tyloi::engine {

inline constexpr std::uint64_t expert_usage_interval = 4;
inline constexpr float expert_usage_retention = .7f;
inline constexpr float expert_replacement_margin = 1.5f;

inline float expert_usage_decay(std::uint64_t windows) {
    return windows == expert_usage_interval ? expert_usage_retention :
        std::pow(expert_usage_retention, float(windows) / float(expert_usage_interval));
}

inline float expert_prefill_usage(std::uint64_t uses, std::size_t rows) noexcept {
    return rows ? float(uses) * float(expert_usage_interval) / (.3f * float(rows)) : 0.f;
}

} // namespace tyloi::engine

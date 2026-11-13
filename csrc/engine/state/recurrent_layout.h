#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace tyloi::state {

struct RecurrentLayout {
    std::array<std::int64_t, 3> convolution;
    std::array<std::int64_t, 4> recurrent;
    std::size_t bytes;
};

RecurrentLayout recurrent_layout(std::int64_t batch, std::int64_t channels,
    std::int64_t kernel, std::int64_t heads, std::int64_t key_dim,
    std::int64_t value_dim, bool transposed = true);

} // namespace tyloi::state

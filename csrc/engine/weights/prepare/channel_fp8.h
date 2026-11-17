#pragma once

#include "engine/tensor/tensor.h"
#include "engine/weights/encoding/fp8/tensor.h"

#include <cstdint>
#include <vector>

namespace tyloi::weights {

ChannelFp8Tensor load_channel_fp8(
    const std::vector<std::uint8_t>& payload,
    Device target);

} // namespace tyloi::weights

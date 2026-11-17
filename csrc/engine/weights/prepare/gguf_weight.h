#pragma once

#include "engine/weights/source/source.h"
#include "engine/weights/encoding/gguf/tensor.h"

#include <cstdint>
#include <string>

namespace tyloi::weights {

std::int64_t gguf_rotation_block(const source::Source&, const std::string& name);

GgufTensor load_gguf(
    const source::Source& source,
    const std::string& name,
    Device target,
    std::int64_t row_begin = 0,
    std::int64_t row_end = -1,
    std::int64_t column_begin = 0,
    std::int64_t column_end = -1);

} // namespace tyloi::weights

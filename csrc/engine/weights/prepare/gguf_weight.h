#pragma once

#include "engine/weights/source/source.h"
#include "engine/tensor/encoding/ggml.h"

#include <cstdint>
#include <string>

namespace tyloi::weights {

GgmlQuantizedTensor load_gguf(
    const source::Source& source,
    const std::string& name,
    Device target,
    std::int64_t row_begin = 0,
    std::int64_t row_end = -1,
    std::int64_t column_begin = 0,
    std::int64_t column_end = -1);

} // namespace tyloi::weights

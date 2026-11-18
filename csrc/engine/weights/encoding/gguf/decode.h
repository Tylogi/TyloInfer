#pragma once

#include "components/ggml/include/ggml.h"

#include <cstdint>

namespace tyloi::gguf {

void decode_row(ggml_type type, const void* packed, float* output, std::int64_t columns);

} // namespace tyloi::gguf

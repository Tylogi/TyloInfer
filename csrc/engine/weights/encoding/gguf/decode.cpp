#include "decode.h"

#include "components/ggml/src/ggml-quants.h"

#include <cstring>
#include <stdexcept>

namespace tyloi::gguf {

void decode_row(ggml_type type, const void* packed, float* output, std::int64_t columns) {
    if (type == GGML_TYPE_F32)
        std::memcpy(output, packed, static_cast<std::size_t>(columns) * sizeof(float));
    else if (type == GGML_TYPE_Q8_K)
        dequantize_row_q8_K(static_cast<const block_q8_K*>(packed), output, columns);
    else {
        const auto* traits = ggml_get_type_traits(type);
        if (!traits->to_float) throw std::invalid_argument("GGUF format has no CPU decoder");
        traits->to_float(packed, output, columns);
    }
}

} // namespace tyloi::gguf

#include "engine/weights/encoding/format_features.h"
#include "engine/weights/encoding/tensor_dtype.h"
#include "engine/weights/encoding/gguf_quant.h"

namespace tyloi {

const bool mfq_compat_enabled = TYLOINFER_ENABLE_MFQ_COMPAT != 0;

bool supported_tensor_dtype(std::string_view dtype) noexcept {
    if (mfq_compat_enabled &&
        (dtype == "NINT" || dtype == "NINT8-0" || dtype == "NVQ" ||
         dtype == "MFE" || dtype == "MFED" || dtype == "NPQ" ||
         dtype == "NEPQ" || dtype == "MXFP4-SQ" ||
         dtype == "MXFP8-SQ" || dtype == "FP8-128SQ")) return true;
    if (dtype == "TQ" || dtype == "BF16" || dtype == "F16" || dtype == "F32"
        || dtype == "F8_E4M3" || dtype == "I32" || dtype == "I64"
        || dtype == "MXFP4" || dtype == "MXFP8" || dtype == "FP8-E4M3-CHANNEL" || dtype == "BLOB")
        return true;
#define TYLOINFER_GGUF_FEATURE(name) if (dtype == "GGUF_" #name) return true;
    TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_FEATURE)
#undef TYLOINFER_GGUF_FEATURE
    return false;
}

} // namespace tyloi

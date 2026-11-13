#include "engine/state/kv_view.h"
#include "engine/ops.h"

#include <stdexcept>

tyloi::state::KvView tyloi::state::allocate_kv(
        int64_t B, int64_t H, int64_t max_seq, int64_t D, bool ring,
        tyloi::Device device, tyloi::ScalarType dtype, KvQuantization quantization) {
    if (max_seq <= 0) throw std::invalid_argument("invalid KV tensor dimensions");
    const auto layout = kv_layout(B, H, max_seq, D, ring, dtype, quantization);
    auto opts = ::tyloi::TensorOptions().device(device).dtype(layout.storage_dtype);
    KvView result;
    result.physical_key = tyloi::ops::zeros({B,H,max_seq,layout.key_width},opts);
    result.physical_value = tyloi::ops::zeros({B,H,max_seq,layout.value_width},opts);
    result.quantization = quantization;
    result.ring = ring;
    result.batch = B;
    result.heads = H;
    result.head_dim = D;
    result.dtype = dtype;
    return result;
}

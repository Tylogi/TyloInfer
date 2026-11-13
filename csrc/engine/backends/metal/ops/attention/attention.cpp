#include "engine/backends/metal/ops/attention/attention.h"
#include "mlx_staged_attention.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace mfq::metal {
namespace {
using mlx::core::array;
array floating(const array& input, mlx::core::Dtype dtype) {
    auto result = input;
    if (result.dtype() != dtype) {
        result = mlx::core::astype(result, dtype);
    }
    return mlx::core::contiguous(result);
}

void require_attention_shape(const array& value, const char* name) {
    if (value.ndim() != 4) {
        throw std::runtime_error(
            std::string(name) + " must have [batch,heads,tokens,head_dim] shape");
    }
}

} // namespace

array scaled_dot_product_attention(
    const array& query,
    const array& key,
    const array& value,
    bool causal,
    float scale,
    const std::optional<array>& mask) {
    require_attention_shape(query, "attention query");
    require_attention_shape(key, "attention key");
    require_attention_shape(value, "attention value");
    if (key.shape() != value.shape() ||
        query.shape(0) != key.shape(0) ||
        query.shape(3) != key.shape(3) ||
        key.shape(1) <= 0 ||
        query.shape(1) % key.shape(1) != 0) {
        throw std::runtime_error("incompatible attention dimensions");
    }
    auto dtype = query.dtype();
    if (dtype != mlx::core::float16 &&
        dtype != mlx::core::bfloat16 &&
        dtype != mlx::core::float32) {
        dtype = mlx::core::float16;
    }
    auto q = floating(query, dtype);
    auto k = key.dtype() == dtype ? key : mlx::core::astype(key, dtype);
    auto v = value.dtype() == dtype ? value : mlx::core::astype(value, dtype);
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(q.shape(3)));
    }
    if (!std::isfinite(scale) || scale <= 0.0f) {
        throw std::runtime_error("attention scale must be positive");
    }
    if (auto staged = mlx_staged_attention(q, k, v, causal, scale, mask)) return std::move(*staged);
    return mlx::core::fast::scaled_dot_product_attention(
        q,
        k,
        v,
        scale,
        causal ? "causal" : "",
        mask);
}

} // namespace mfq::metal

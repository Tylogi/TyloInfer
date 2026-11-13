#pragma once

#include "backend.h"
#include "engine/backends/cuda/ops/projection/projection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

namespace tyloi::cuda::attention_ops {
using DenseTensor = ::tyloi::DenseTensor;
using Linear = ::tyloi::Tensor;
using Transform = std::function<DenseTensor(const Projector&, const DenseTensor&)>;
using Embedding = std::function<DenseTensor(const DenseTensor&)>;

inline DenseTensor rms_norm(const DenseTensor& value,
#ifdef TYLOINFER_NATIVE_CUDA
                            const ::tyloi::Tensor& weight_tensor, double eps) {
#else
                            const DenseTensor& weight, double eps) {
#endif
    auto f = value.to(::tyloi::kFloat32);
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& weight = weight_tensor.dense();
#endif
    return (f * ::tyloi::cuda::rsqrt((f * f).mean(-1, true) + eps) * weight.to(::tyloi::kFloat32))
        .to(value.scalar_type());
}

inline DenseTensor select_pooled_blocks(const DenseTensor& scores, int64_t query_offset,
    int64_t logical_length, int64_t pool, int64_t budget, bool include_tail) {
    TYLOINFER_RUNTIME_CHECK(scores.is_cuda() && scores.dim() == 3 && pool > 0 && budget > 0 &&
        budget % pool == 0 && query_offset >= 0 && logical_length >= query_offset &&
        scores.size(1) <= logical_length - query_offset && scores.size(2) == logical_length / pool,
        "pooled attention selection geometry mismatch");
    const auto b = scores.size(0), t = scores.size(1), pools = scores.size(2);
    const auto options = scores.options().dtype(::tyloi::kInt64);
    auto absolute = ::tyloi::cuda::arange(t, options) + query_offset;
    auto selected = ::tyloi::cuda::full({b,t,budget}, -1, options);
    const auto count = std::min(budget / pool, pools);
    if (count) {
        auto ends = ::tyloi::cuda::arange(pools, options) * pool + (pool - 1);
        auto visible = (ends.reshape({1,1,pools}) <= absolute.reshape({1,t,1})).expand({b,t,pools});
        auto ranked = ::tyloi::cuda::where(visible, scores.to(::tyloi::kFloat32), ::tyloi::cuda::full_like(scores, -1e30).to(::tyloi::kFloat32));
        auto indices = std::get<1>(::tyloi::cuda::topk(ranked, count, -1, true, false));
        auto valid = visible.gather(-1, indices).unsqueeze(-1).expand({b,t,count,pool});
        auto expanded = indices.unsqueeze(-1) * pool + ::tyloi::cuda::arange(pool, options);
        expanded = ::tyloi::cuda::where(valid, expanded, ::tyloi::cuda::full_like(expanded, -1));
        selected.narrow(-1, 0, count * pool).copy_(expanded.reshape({b,t,count * pool}));
    }
    if (include_tail && pool > 1) {
        auto count = (absolute + 1).remainder(pool);
        auto offsets = ::tyloi::cuda::arange(pool - 1, options).unsqueeze(0);
        auto tail = (absolute + 1 - count).unsqueeze(1) + offsets;
        tail = ::tyloi::cuda::where(offsets < count.unsqueeze(1), tail, ::tyloi::cuda::full_like(tail, -1));
        selected = ::tyloi::cuda::cat({selected, tail.unsqueeze(0).expand({b,t,pool - 1})}, -1);
    }
    return selected.to(::tyloi::kInt32).contiguous();
}

} // namespace tyloi::cuda::attention_ops

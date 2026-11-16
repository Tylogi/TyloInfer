#include "engine/ops/ple.h"
#include "engine/ops/ops.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/ops/qwen4_exp.h"
#include "engine/backends/cuda/kernels/embedding/ple.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include <cmath>
#include <optional>

namespace tyloi::ops {
Tensor ple(const Tensor& input, const Tensor& embeddings, const PleWeights& w,
        Tensor& convolution, bool cache, std::int64_t hidden, std::int64_t streams,
        std::int64_t dilation, double eps, std::vector<Tensor>* checkpoints,
        std::size_t checkpoint_offset) {
    const auto& x = input.dense();
    const auto& rows = embeddings.dense();
    const auto b = x.size(0), t = x.size(1);
    auto conv = convolution.defined() ? convolution.dense() : DenseTensor{};
    const auto save_prefixes = [&](const DenseTensor& normalized) {
        if (!checkpoints) return;
        TYLOINFER_RUNTIME_CHECK(cache && conv.defined(), "PLE prefix history is unavailable");
        const auto lookback = conv.size(1);
        for (std::int64_t count = 1; count < t; ++count) {
            auto& saved = checkpoints->at(checkpoint_offset + std::size_t(count - 1)).dense();
            if (count >= lookback) saved.copy_(normalized.narrow(1, count - lookback, lookback));
            else {
                saved.narrow(1, 0, lookback - count).copy_(conv.narrow(1, count, lookback - count));
                saved.narrow(1, lookback - count, count).copy_(normalized.narrow(1, 0, count));
            }
        }
    };
    const auto native = cuda::options::ple_native_fusion();
    if ((!native || *native) && x.is_contiguous() && t <= 32 && hidden > 128 && hidden <= 8192 &&
        (x.scalar_type() == kFloat16 || x.scalar_type() == kFloat32)) {
        auto key = mfq_qwen4_exp::grouped_rms_norm(project(rows, w.key).dense(), w.key_norm.dense(), hidden, eps);
        auto query = mfq_qwen4_exp::grouped_rms_norm(x, w.query_norm.dense(), hidden, eps);
        auto value = project(rows, w.value).dense();
        auto prepared = mfq_qwen4_exp::ple_gate_norm_cuda(key, query, value,
            w.conv_norm.dense().to(kFloat32).contiguous(), hidden, streams, eps);
        save_prefixes(prepared[1]);
        auto output = mfq_qwen4_exp::ple_conv_add_cuda(prepared[1], w.conv.dense(),
            cache && conv.defined() ? std::optional<DenseTensor>(conv) : std::nullopt,
            dilation, prepared[0], x.scalar_type() == kFloat16);
        if (cache) { if (conv.defined()) conv.copy_(output[1]); else convolution = output[1]; }
        return output[0];
    }
    auto key = mfq_qwen4_exp::grouped_rms_norm(project(rows, w.key).dense(), w.key_norm.dense(), hidden, eps)
        .reshape({b, t, streams, hidden});
    auto query = mfq_qwen4_exp::grouped_rms_norm(x, w.query_norm.dense(), hidden, eps)
        .reshape({b, t, streams, hidden});
    auto score = (key.to(kFloat32) * query.to(kFloat32)).sum(-1) / std::sqrt(double(hidden));
    auto sign = (score > 0).to(kFloat32) - (score < 0).to(kFloat32);
    auto root = sign * cuda::clamp_min(score.abs(), 1e-6).sqrt();
    auto gated = (cuda::sigmoid(root).unsqueeze(-1) *
        project(rows, w.value).dense().to(kFloat32).unsqueeze(-2)).reshape({b, t, streams * hidden});
    auto normalized = mfq_qwen4_exp::grouped_rms_norm(gated, w.conv_norm.dense(), hidden, eps);
    save_prefixes(normalized);
    auto convolved = mfq_qwen4_exp::ple_dilated_conv_silu(normalized, w.conv.dense(),
        cache && conv.defined() ? std::optional<DenseTensor>(conv) : std::nullopt, dilation);
    auto output = (gated + convolved[0]).to(x.scalar_type());
    if (cache) { if (conv.defined()) conv.copy_(convolved[1]); else convolution = convolved[1]; }
    return output;
}
}

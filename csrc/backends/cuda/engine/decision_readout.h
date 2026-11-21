#pragma once

#include "quant_linear.h"
#include "tyloinfer/decision_model.h"
#include "runtime/decision_attention.h"

#include <cstring>

namespace mfq::cuda::internal {

inline QuantLinear load_decision_readout(const mfq::ModelSource& source, std::int64_t hidden,
                                        mfq_tensor_backend::Device device) {
    namespace tb = mfq_tensor_backend;
    if (!source.has_asset(mfq::kDecisionReadoutAsset))
        throw std::invalid_argument("trained decision model is missing readout.safetensors");
    const auto bytes = source.read_asset(mfq::kDecisionReadoutAsset);
    const auto [dtype, values] = parse_decision_readout(bytes, hidden);
    const auto kind = dtype == "F32" ? tb::kFloat32 : dtype == "BF16" ? tb::kBFloat16 : tb::kFloat16;
    const auto count = std::uint64_t(255) * std::uint64_t(hidden);
    auto host = tb::empty({255, hidden}, tb::TensorOptions().device(tb::kCPU).dtype(kind));
    std::memcpy(host.data_ptr(), values.data(), values.size());
    host = host.to(tb::kFloat32).contiguous();
    for (std::uint64_t i = 0; i < count; ++i)
        if (!std::isfinite(host.data_ptr<float>()[i]))
            throw std::invalid_argument("decision readout contains nonfinite weights");
    QuantLinear result;
    result.kind = QuantLinearKind::Dense;
    result.dense = host.to(device).contiguous();
    return result;
}

class DecisionAttentionScope {
    CudaDecisionAttentionScope scope_;
    static std::optional<std::int64_t> boundary(const DecisionModelSpec* spec, std::size_t state_tokens) {
        if (!spec || spec->attention == DecisionAttention::causal) return {};
        return spec->attention == DecisionAttention::bidirectional ? -1 : std::int64_t(state_tokens);
    }
public:
    DecisionAttentionScope(CudaExecutionContext& execution, const DecisionModelSpec* spec,
                           std::size_t state_tokens) : scope_(execution,boundary(spec,state_tokens)) {}
    DecisionAttentionScope(const DecisionAttentionScope&) = delete;
    DecisionAttentionScope& operator=(const DecisionAttentionScope&) = delete;
};

} // namespace mfq::cuda::internal

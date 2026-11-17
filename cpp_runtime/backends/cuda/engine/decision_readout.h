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
    if (hidden <= 0 || bytes.size() < 8 || bytes.size() > (256u << 20))
        throw std::invalid_argument("decision readout size is invalid");
    std::uint64_t header_size = 0;
    for (unsigned i = 0; i < 8; ++i)
        header_size |= std::uint64_t(std::to_integer<unsigned char>(bytes[i])) << (8 * i);
    if (!header_size || header_size > (1u << 20) || header_size > bytes.size() - 8)
        throw std::invalid_argument("decision readout header is invalid");
    const auto header = nlohmann::json::parse(reinterpret_cast<const char*>(bytes.data() + 8),
        reinterpret_cast<const char*>(bytes.data() + 8 + header_size));
    const auto& weight = header.at("weight");
    if (weight.at("shape").get<std::vector<std::int64_t>>() != std::vector<std::int64_t>{255, hidden})
        throw std::invalid_argument("decision readout must have shape [255, hidden_size]");
    const auto dtype = weight.at("dtype").get<std::string>();
    const auto kind = dtype == "F32" ? tb::kFloat32 : dtype == "BF16" ? tb::kBFloat16 : tb::kFloat16;
    if (dtype != "F32" && dtype != "BF16" && dtype != "F16")
        throw std::invalid_argument("unsupported decision readout dtype");
    const auto offsets = weight.at("data_offsets").get<std::vector<std::uint64_t>>();
    const auto count = std::uint64_t(255) * std::uint64_t(hidden);
    const auto size = count * (dtype == "F32" ? 4 : 2);
    if (offsets.size() != 2 || offsets[0] > offsets[1] || offsets[1] > bytes.size() - 8 - header_size ||
        offsets[1] - offsets[0] != size)
        throw std::invalid_argument("decision readout tensor range is invalid");
    auto host = tb::empty({255, hidden}, tb::TensorOptions().device(tb::kCPU).dtype(kind));
    std::memcpy(host.data_ptr(), bytes.data() + 8 + header_size + offsets[0], size);
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

#pragma once

#include <cstdint>
#include <optional>

struct CudaExecutionContext;

namespace mfq::cuda {

struct DecisionAttentionFrame {
    const CudaExecutionContext* execution;
    std::optional<std::int64_t> boundary;
    const DecisionAttentionFrame* previous;
};
inline thread_local const DecisionAttentionFrame* current_decision_attention = nullptr;

inline std::optional<std::int64_t> decision_attention_boundary(const CudaExecutionContext& execution) {
    const auto* frame = current_decision_attention;
    return frame && frame->execution == &execution ? frame->boundary : std::nullopt;
}

class CudaDecisionAttentionScope {
    DecisionAttentionFrame frame_;
public:
    CudaDecisionAttentionScope(const CudaExecutionContext& execution, std::optional<std::int64_t> boundary)
        : frame_{&execution, boundary, current_decision_attention} {
        current_decision_attention = &frame_;
    }
    ~CudaDecisionAttentionScope() { current_decision_attention = frame_.previous; }
    CudaDecisionAttentionScope(const CudaDecisionAttentionScope&) = delete;
    CudaDecisionAttentionScope& operator=(const CudaDecisionAttentionScope&) = delete;
};

} // namespace mfq::cuda

#include "engine/models/qwen35/mtp.h"
#include "engine/ops.h"

namespace tyloi::cuda::qwen35 {

namespace {

std::vector<MtpState::Layout> layout(const tyloi::models::qwen35::Predictor& model, ::tyloi::Device device) {
    std::vector<MtpState::Layout> result;
    for (const auto& layer : model.layers) {
        const auto& a = layer.attention;
        result.push_back({device, a.kv_heads, a.attention_head_dim, a.max_position_embeddings,
            a.official_bf16 ? ::tyloi::kBFloat16 : ::tyloi::kFloat16, a.kv_quantization});
    }
    return result;
}

} // namespace

Mtp::Mtp(tyloi::models::qwen35::Predictor model,
        const int64_t& kv_capacity, tyloi::ops::Trace* const& trace, const int& trace_scope,
        bool preallocate)
    : model_(std::move(model)),
      kv_capacity_(kv_capacity), trace_(trace), trace_scope_(trace_scope), preallocate_(preallocate) {}

std::unique_ptr<tyloi::models::MtpState> Mtp::make_state() const {
    auto result = std::make_unique<MtpState>(*this, layout(model_, model_.device), model_.config.hidden_size);
    if (preallocate_) result->prepare(1, 1);
    return result;
}

void Mtp::reset(tyloi::models::MtpState& value, int64_t batch) const {
    state<MtpState>(value).reset(batch);
}

tyloi::models::MtpStep Mtp::step(tyloi::models::MtpState& value, const ::tyloi::Tensor& hidden,
        const ::tyloi::Tensor& next_ids, const ::tyloi::Tensor& positions) const {
    auto& request = state<MtpState>(value);
    const int device = model_.device.index;
    const tyloi::models::ForwardOptions options{kv_capacity_, trace_, trace_scope_, device};
    auto output = model_.step(hidden, next_ids, positions, request, options);
    return {output, output};
}

void Mtp::trim_cache_to(tyloi::models::MtpState& value, int64_t position) const {
    state<MtpState>(value).trim(position);
}

MtpSessionState Mtp::capture_session_state(const tyloi::models::MtpState& value,
                                           int64_t position, const ::tyloi::Tensor& hidden) const {
    tyloi::ops::DeviceScope guard(model_.device);
    return state<MtpState>(value).capture(position, hidden);
}

} // namespace tyloi::cuda::qwen35

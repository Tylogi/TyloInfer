#include "engine/models/qwen35/mtp.h"
#include "engine/ops/ops.h"

namespace tyloi::models::qwen35 {

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

Mtp::Mtp(tyloi::models::qwen35::Predictor model, bool preallocate)
    : model_(std::move(model)), preallocate_(preallocate) {}

std::unique_ptr<tyloi::models::MtpState> Mtp::make_state() const {
    return std::make_unique<MtpState>(*this, layout(model_, model_.device), model_.config.hidden_size);
}

std::vector<std::pair<Device, std::size_t>> Mtp::reset_growth(
        const tyloi::models::MtpState& value, int64_t batch) const {
    const auto& request = state<MtpState>(value);
    return preallocate_ ? request.growth(batch, 1, 0, true) : std::vector<std::pair<Device, std::size_t>>{};
}

void Mtp::reset(tyloi::models::MtpState& value, int64_t batch) const {
    auto& request = state<MtpState>(value);
    request.reset(batch);
    if (preallocate_) request.prepare(batch, 1);
}

std::vector<std::pair<Device, std::size_t>> Mtp::growth(const tyloi::models::MtpState& value,
        const Tensor& hidden, const Tensor& ids, const Tensor&, const ForwardOptions& options) const {
    if (!hidden || !ids || hidden.dim() != 3 || ids.dim() != 2 ||
        hidden.size(0) != ids.size(0) || hidden.size(1) != ids.size(1))
        throw std::invalid_argument("invalid Qwen MTP input dimensions");
    return state<MtpState>(value).growth(hidden.size(0), hidden.size(1), options.kv_capacity);
}

tyloi::models::MtpStep Mtp::step(tyloi::models::MtpState& value, const ::tyloi::Tensor& hidden,
        const ::tyloi::Tensor& next_ids, const ::tyloi::Tensor& positions,
        const tyloi::models::ForwardOptions& options) const {
    auto& request = state<MtpState>(value);
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

} // namespace tyloi::models::qwen35

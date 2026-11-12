#include "engine/models/qwen35/mtp.h"
#include "engine/models/session/snapshot.h"

#include "engine/models/qwen35/state.h"
#include "engine/models/session/cache.h"
#include "engine/backends/cuda/native/collectives.h"

namespace tyloi::cuda::qwen35 {


namespace {

std::vector<MtpState::Layout> layout(const tyloi::models::qwen35::Predictor& model, int device) {
    std::vector<MtpState::Layout> result;
    for (const auto& layer : model.layers) {
        const auto& a = layer.attention;
        result.push_back({::tyloi::Device(::tyloi::kCUDA, device), a.kv_heads, a.attention_head_dim, a.max_position_embeddings,
            a.official_bf16 ? ::tyloi::kBFloat16 : ::tyloi::kFloat16, a.kv_quantization});
    }
    return result;
}

} // namespace

Mtp::Mtp(tyloi::models::qwen35::Predictor model, ::tyloi::Tensor embedding,
        tyloi::ops::RopeTable rope,
        std::unique_ptr<Executor> executor, Collectives& copies,
        int device, const int64_t& kv_capacity, tyloi::ops::Trace* const& trace, const int& trace_scope,
        bool preallocate)
    : model_(std::move(model)), embedding_(std::move(embedding)), rope_(std::move(rope)),
      executor_(std::move(executor)), copies_(copies), device_(device),
      kv_capacity_(kv_capacity), trace_(trace), trace_scope_(trace_scope), preallocate_(preallocate) {}

std::unique_ptr<tyloi::models::MtpState> Mtp::make_state() const {
    auto result = std::make_unique<MtpState>(*this, layout(model_, device_), model_.config.hidden_size);
    if (preallocate_) result->prepare(1, 1);
    return result;
}

void Mtp::reset(tyloi::models::MtpState& value, int64_t batch) const {
    state<MtpState>(value).reset(batch);
}

tyloi::models::MtpStep Mtp::step(tyloi::models::MtpState& value, const ::tyloi::Tensor& hidden,
        const ::tyloi::Tensor& next_ids, const ::tyloi::Tensor& positions) const {
    auto output = std::make_shared<::tyloi::DenseTensor>(evaluate(
        state<MtpState>(value),
        hidden.dense(),
        next_ids.dense(),
        positions.defined() ? positions.dense()
                                          : ::tyloi::DenseTensor{}));
    return {output, output};
}

::tyloi::DenseTensor Mtp::evaluate(MtpState& state, ::tyloi::DenseTensor hidden,
        ::tyloi::DenseTensor next_ids, ::tyloi::DenseTensor positions) const {
    const auto& config = model_.config;
    TYLOINFER_RUNTIME_CHECK(hidden.dim() == 3 && next_ids.dim() == 2 &&
        hidden.size(0) == next_ids.size(0) && hidden.size(1) == next_ids.size(1) &&
        hidden.size(2) == config.hidden_size && hidden.size(0) > 0 && hidden.size(1) > 0,
        "Qwen MTP inputs must be matching [B,T,H] hidden states and [B,T] next-token IDs");
    const auto batch = hidden.size(0), tokens = hidden.size(1);
    TYLOINFER_RUNTIME_CHECK(state.position() <= config.max_position_embeddings - tokens,
        "Qwen MTP history exceeds context capacity");
    const int device = device_;
    MfqCudaGuard guard(device);
    const FfnMode mode{};
    auto executor_scope = executor_->enter(device, mode);
    const bool explicit_positions = positions.defined();
    auto pos = explicit_positions
        ? copies_.copy(std::move(positions), device).to(::tyloi::kInt64).contiguous()
        : ::tyloi::cuda::arange(state.position(), state.position() + tokens,
            ::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kInt64));
    TYLOINFER_RUNTIME_CHECK((pos.dim() == 1 && pos.numel() == tokens) ||
        (pos.dim() == 2 && pos.size(1) == tokens &&
            (pos.size(0) == batch || (pos.size(0) == 3 && !config.mrope_sections.empty()))),
        "Qwen MTP positions must have shape [T], [B,T], or configured grid-MRoPE [3,T]");

    state.prepare(batch, tokens, kv_capacity_);
    const auto kv = state.views();
    auto normalized_ids = copies_.copy(next_ids.to(::tyloi::kInt64), device_).contiguous();
    auto embedded = executor_->profiler().measure("model.embed", [&] {
        return ::tyloi::ops::embedding(::tyloi::Tensor(normalized_ids), embedding_);
    });
    auto embedded_dense = embedded.dense().to(::tyloi::kFloat16).contiguous()
                              .to(hidden.scalar_type());

    ::tyloi::models::AttentionStep attention;
    attention.positions = pos;
    attention.cache_position = state.position();
    if (tokens == 1 && state.position() > 0)
        attention.sequence_lengths = ::tyloi::cuda::full({batch}, state.position() + 1, pos.options());
    if (explicit_positions)
        attention.cache_positions = ::tyloi::cuda::arange(state.position(), state.position() + tokens, pos.options());

    auto output = model_.forward(::tyloi::Tensor(std::move(hidden)),
        ::tyloi::Tensor(std::move(embedded_dense)), attention, kv, rope_, trace_, trace_scope_);
    auto result = output.dense();
    state.advance(tokens);
    return result;
}

void Mtp::trim_cache_to(tyloi::models::MtpState& value, int64_t position) const {
    state<MtpState>(value).trim(position);
}

MtpSessionState Mtp::capture_session_state(const tyloi::models::MtpState& value,
                                           int64_t position, const ::tyloi::Tensor& hidden) const {
    MfqCudaGuard guard(device_);
    return state<MtpState>(value).capture(position, hidden);
}

void restore_session(State& state, Mtp* mtp, const TextSessionState& snapshot,
        const std::function<void()>& drain) {
    tyloi::cuda::internal::CudaSessionOps::restore(
        static_cast<tyloi::models::State&>(state), snapshot, drain,
        static_cast<tyloi::models::Mtp*>(mtp));
}

} // namespace tyloi::cuda::qwen35

#include "engine/models/qwen35/mtp_state.h"
#include "engine/models/session/snapshot.h"
#include "engine/ops.h"

#include <algorithm>
#include <stdexcept>

namespace tyloi::cuda::qwen35 {

namespace {
void require_state(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool on_device(const ::tyloi::Tensor& tensor, tyloi::Device device) {
    return device.is_cpu() ? tensor.is_cpu() : tensor.device() == device;
}
} // namespace
MtpState::MtpState(const tyloi::models::Mtp& owner, std::vector<Layout> layout, int64_t hidden)
    : tyloi::models::MtpState(owner), layout_(std::move(layout)), kv_(layout_.size()), hidden_(hidden) {}

void MtpState::prepare(int64_t batch, int64_t tokens, int64_t kl_capacity) {
    require_state(batch > 0 && tokens > 0, "invalid MTP state dimensions");
    for (size_t index = 0; index < kv_.size(); ++index) {
        const auto& layout = layout_[index];
        require_state(position_ <= layout.capacity - tokens,
            "Qwen MTP history exceeds context capacity");
        if (kv_[index].defined()) {
            require_state(kv_[index].batch == batch,
                "MTP batch changed without resetting state");
            if (kv_[index].physical_key.size(2) >= position_ + tokens) continue;
        }
        tyloi::ops::DeviceScope guard(layout.device);
        const auto capacity = kl_capacity > 0
            ? std::min(layout.capacity, std::max(position_ + tokens, kl_capacity)) : layout.capacity;
        auto grown = kv_[index].defined()
            ? tyloi::state::grow_kv(kv_[index], capacity, position_)
            : tyloi::state::allocate_kv(batch, layout.heads, capacity, layout.head_dim, false,
                layout.device, layout.dtype, layout.quantization);
        kv_[index] = std::move(grown);
    }
}

void MtpState::reset(int64_t batch) {
    position_ = 0;
    for (auto& cache : kv_)
        if (!cache.defined() || cache.batch != batch) cache = {};
}

void MtpState::advance(int64_t tokens) {
    require_state(tokens > 0, "MTP position must advance by consumed inputs");
    position_ += tokens;
}

void MtpState::trim(int64_t position) {
    require_state(position >= 0 && position <= position_,
        "Qwen MTP cache trim position is invalid");
    position_ = position;
}

std::vector<tyloi::state::KvView> MtpState::views() const {
    std::vector<tyloi::state::KvView> result;
    result.reserve(kv_.size());
    for (const auto& cache : kv_) {
        require_state(cache.defined(), "MTP KV state is unavailable");
        result.push_back(cache);
    }
    return result;
}

MtpSessionState MtpState::capture(int64_t target_position, const ::tyloi::Tensor& hidden) const {
    const auto position = target_position - 1;
    if (position <= 0 || position > position_ || !hidden.defined() || hidden.is_encoded() ||
        hidden.dim() != 3 || hidden.size(0) != 1 || hidden.size(1) != 1 || hidden.size(2) != hidden_)
        throw std::runtime_error("Qwen MTP session boundary is unavailable");
    MtpSessionState saved;
    saved.cache_pos = position;
    auto& blocks = std::get<std::vector<KvSnapshot>>(saved.payload);
    blocks.reserve(kv_.size());
    for (size_t index = 0; index < kv_.size(); ++index) {
        tyloi::ops::DeviceScope guard(layout_[index].device);
        blocks.push_back(capture_kv(kv_[index], position, saved.bytes));
    }
    auto saved_hidden = tyloi::ops::clone(hidden);
    saved.bytes += session_tensor_bytes(saved_hidden);
    saved.last_target_hidden = std::move(saved_hidden);
    return saved;
}

MtpState::Restore MtpState::prepare(const MtpSessionState& saved) const {
    const auto* blocks = std::get_if<std::vector<KvSnapshot>>(&saved.payload);
    const auto& hidden = saved.last_target_hidden;
    if (saved.cache_pos <= 0 || blocks == nullptr || blocks->size() != kv_.size() ||
        !hidden.defined() || hidden.is_encoded() || hidden.dim() != 3 ||
        hidden.size(0) != 1 || hidden.size(1) != 1 || hidden.size(2) != hidden_)
        throw CudaSessionStateError("Qwen MTP session state is incompatible");
    for (size_t index = 0; index < blocks->size(); ++index) {
        if (saved.cache_pos > layout_[index].capacity)
            throw CudaSessionStateError("Qwen MTP session exceeds context capacity");
        const auto& block = (*blocks)[index];
        const auto& layout = layout_[index];
        validate_kv(block);
        const auto key_width = layout.quantization.enabled()
            ? tyloi::state::kv_packed_width(int(layout.head_dim), layout.quantization.key_bits()) : layout.head_dim;
        const auto value_width = layout.quantization.enabled()
            ? tyloi::state::kv_packed_width(int(layout.head_dim), layout.quantization.value_bits()) : layout.head_dim;
        const auto dtype = layout.quantization.enabled() ? ::tyloi::kInt32 : layout.dtype;
        if (block.capacity > layout.capacity || block.ring ||
            block.k.size(1) != layout.heads || block.k.size(2) != saved.cache_pos ||
            block.k.size(3) != key_width || block.v.size(3) != value_width ||
            block.k.scalar_type() != dtype || !on_device(block.k, layout.device) ||
            !on_device(block.v, layout.device))
            throw CudaSessionStateError("Qwen MTP session KV topology changed");
    }
    return {std::vector<tyloi::state::KvView>(kv_.size()), saved.cache_pos};
}

namespace {
class SessionRestore final : public tyloi::models::Restore {
public:
    SessionRestore(MtpState& state, const MtpSessionState& saved)
        : state_(state), saved_(saved), candidate_(state.prepare(saved_)) {}
    void stage() override { state_.stage(candidate_, saved_); }
    void publish() noexcept override { state_.publish(std::move(candidate_)); }

private:
    MtpState& state_;
    MtpSessionState saved_;
    MtpState::Restore candidate_;
};
} // namespace

std::unique_ptr<tyloi::models::Restore> MtpState::prepare_restore(const MtpSessionState& saved) {
    return std::make_unique<SessionRestore>(*this, saved);
}

void MtpState::stage(Restore& restore, const MtpSessionState& saved) const {
    const auto& blocks = std::get<std::vector<KvSnapshot>>(saved.payload);
    for (size_t index = 0; index < restore.kv.size(); ++index) {
        const auto& layout = layout_[index];
        tyloi::ops::DeviceScope guard(layout.device);
        restore_kv(restore.kv[index], blocks[index], layout.head_dim,
            layout.dtype, layout.quantization);
    }
}

void MtpState::publish(Restore&& restore) noexcept {
    kv_.swap(restore.kv);
    position_ = restore.position;
}

} // namespace tyloi::cuda::qwen35

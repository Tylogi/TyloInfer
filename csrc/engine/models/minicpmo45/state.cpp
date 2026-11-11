#include "engine/models/minicpmo45/state.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/minicpmo45/model.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::minicpmo45 {

#ifdef TYLOINFER_NATIVE_CUDA
std::unique_ptr<tyloi::models::State> Model::make_state() const {
    return std::make_unique<State>(*this);
}

std::vector<std::pair<tyloi::Device, std::size_t>> State::growth(
        std::int64_t tokens, const tyloi::models::ForwardOptions& options) const {
    if (batch <= 0 || tokens <= 0 || cache_pos < 0)
        throw std::invalid_argument("MiniCPM TTS KV planning requires positive batch and tokens and a nonnegative position");
    if (cache_pos > std::numeric_limits<std::int64_t>::max() - tokens)
        throw std::overflow_error("MiniCPM TTS KV planning position overflow");
    const auto end = cache_pos + tokens;
    std::vector<std::pair<tyloi::Device, std::size_t>> result;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        if (layers[index].defined()) continue;
        const auto& layout = layout_[index];
        const auto capacity = options.kv_capacity > 0 ? std::max(end, options.kv_capacity) : layout.capacity;
        const auto bytes = tyloi::state::kv_bytes(batch, layout.heads, capacity, layout.head_dim,
                                                   false, tyloi::kFloat16, {});
        const tyloi::Device device(layout.cpu ? tyloi::kCPU : tyloi::kCUDA, layout.cpu ? -1 : layout.device);
        auto entry = std::find_if(result.begin(), result.end(),
                                  [device](const auto& item) { return item.first == device; });
        if (entry == result.end()) result.emplace_back(device, bytes);
        else {
            if (entry->second > std::numeric_limits<std::size_t>::max() - bytes)
                throw std::overflow_error("MiniCPM TTS KV planning byte count overflow");
            entry->second += bytes;
        }
    }
    return result;
}
#endif

State::State(const Model& model)
    : layers(model.layers.size()), maximum_(model.metadata.max_position_embeddings) {
    for (const auto& layer : model.layers)
        layout_.push_back({layer.device, layer.cpu, layer.attention.kv_heads,
            layer.attention.attention_head_dim, layer.attention.max_position_embeddings});
}

void State::preallocate() {
    for (size_t index = 0; index < layers.size(); ++index) {
        const auto& layout = layout_[index];
        if (layout.cpu || layers[index].defined()) continue;
        MfqCudaGuard guard(layout.device);
        layers[index] = tyloi::state::allocate_kv(
            1, layout.heads, layout.capacity, layout.head_dim, false,
            ::tyloi::Device(::tyloi::kCUDA, layout.device), ::tyloi::kFloat16);
    }
}

void State::reset(int64_t next_batch) {
    if (next_batch <= 0) throw std::invalid_argument("MiniCPM TTS batch must be positive");
    cache_pos = decode_position_delta = 0;
    batch = next_batch;
    for (size_t index = 0; index < layers.size(); ++index) {
        auto& cache = layers[index];
        if (cache.defined() && cache.batch != batch) {
            MfqCudaGuard guard(layout_[index].device);
            cache = {};
        }
    }
}

std::vector<::tyloi::Tensor> State::storage() const {
    std::vector<::tyloi::Tensor> result;
    for (const auto& cache : layers) {
        if (cache.physical_key.defined()) result.push_back(cache.physical_key);
        if (cache.physical_value.defined()) result.push_back(cache.physical_value);
    }
    return result;
}

TextSessionStateKind State::text_session_state_kind() const {
    return layers.empty() ? TextSessionStateKind::Unsupported : TextSessionStateKind::FullAttention;
}

bool State::supports_text_session_state() const { return !layers.empty(); }
bool State::supports_paged_text_session_state() const { return !layers.empty(); }

TextSessionState State::capture_text_session_state(const std::vector<int64_t>& tokens) const {
    if (layers.empty() || batch != 1 || cache_pos <= 0 || size_t(cache_pos) != tokens.size())
        throw std::runtime_error("MiniCPM TTS session token count does not match the cache");
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = cache_pos;
    state.decode_position_delta = decode_position_delta;
    state.payload = std::vector<KvSnapshot>{};
    auto& saved = std::get<std::vector<KvSnapshot>>(state.payload);
    for (size_t index = 0; index < layers.size(); ++index) {
        MfqCudaGuard guard(layout_[index].device);
        saved.push_back(capture_kv(layers[index], cache_pos, state.bytes));
    }
    return state;
}

State::Restore State::prepare(const TextSessionState& state) const {
    const auto* saved = std::get_if<std::vector<KvSnapshot>>(&state.payload);
    if (layers.empty() || !saved || saved->size() != layers.size() || state.cache_pos <= 0 ||
        state.cache_pos > maximum_ || size_t(state.cache_pos) != state.tokens.size())
        throw CudaSessionStateError("MiniCPM TTS session state is incompatible");
    for (size_t index = 0; index < saved->size(); ++index) {
        const auto& layout = layout_[index];
        const auto& cache = saved->at(index);
        validate_kv(cache);
        const auto device = ::tyloi::Device(layout.cpu ? ::tyloi::kCPU : ::tyloi::kCUDA,
                                       layout.cpu ? -1 : layout.device);
        if (cache.ring || cache.k.device() != device || cache.v.device() != device ||
            cache.k.scalar_type() != ::tyloi::kFloat16 ||
            cache.v.scalar_type() != ::tyloi::kFloat16 ||
            cache.k.size(1) != layout.heads || cache.v.size(1) != layout.heads ||
            cache.k.size(3) != layout.head_dim || cache.v.size(3) != layout.head_dim ||
            cache.k.size(2) != state.cache_pos || cache.capacity < state.cache_pos)
            throw CudaSessionStateError("MiniCPM TTS session cache geometry changed");
    }
    return {std::vector<tyloi::state::KvView>(layers.size()), state.cache_pos,
            state.decode_position_delta};
}

void State::stage(Restore& restore, const TextSessionState& state) const {
    const auto& saved = std::get<std::vector<KvSnapshot>>(state.payload);
    for (size_t index = 0; index < layers.size(); ++index) {
        MfqCudaGuard guard(layout_[index].device);
        restore_kv(restore.layers[index], saved[index], layout_[index].head_dim, ::tyloi::kFloat16, {});
    }
}

void State::publish(Restore&& restore) noexcept {
    layers.swap(restore.layers);
    cache_pos = restore.cache_pos;
    decode_position_delta = restore.decode_position_delta;
    batch = 1;
}

} // namespace tyloi::cuda::minicpmo45

#include "engine/models/gemma4/state.h"
#include "engine/models/gemma4/model.h"
#include "engine/ops.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::gemma4 {
std::unique_ptr<tyloi::models::State> Model::make_state() const {
    return std::make_unique<State>(*this);
}

State::State(const Model& model) : layers(model.layers.size()), graph_views_(model.layers.size() * 2) {
    layout_.reserve(model.layers.size());
    for (const auto& layer : model.layers) {
        const auto& attention = layer.attention;
        layout_.push_back({layer.device, attention.kv_heads, attention.attention_head_dim,
                           attention.sliding ? attention.attention_window : attention.max_position_embeddings,
                           attention.sliding,
                           attention.official_bf16 ? tyloi::ScalarType::bfloat16 : tyloi::ScalarType::float16,
                           attention.kv_quantization});
    }
}

void State::preallocate() {
    prepare_caches(1, 0, {});
}

void State::prepare_caches(std::int64_t batch, std::int64_t tokens,
        const tyloi::models::ForwardOptions& options) {
    if (batch <= 0 || tokens < 0 || cache_pos < 0)
        throw std::invalid_argument("Gemma4 KV preparation requires a positive batch and a nonnegative position");
    if (cache_pos > std::numeric_limits<std::int64_t>::max() - tokens)
        throw std::overflow_error("Gemma4 KV preparation position overflow");
    const auto end = cache_pos + tokens;
    std::vector<tyloi::state::KvView> prepared;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        if (layers[index].defined()) continue;
        const auto& layout = layout_[index];
        const auto capacity = kv_capacity_for(layout.sliding, layout.capacity, end, options.kv_capacity);
        prepared.push_back(tyloi::state::allocate_kv(batch, layout.heads, capacity, layout.head_dim,
            layout.sliding, tyloi::Device(tyloi::DeviceType::cuda, layout.device), layout.dtype,
            layout.quantization));
    }
    // Publish only once every layer cache exists, so a failed allocation cannot
    // leave a half-populated cache that later forwards treat as allocated.
    std::size_t next = 0;
    for (std::size_t index = 0; index < layers.size(); ++index)
        if (!layers[index].defined()) layers[index] = std::move(prepared[next++]);
}

void State::reset(std::int64_t next_batch) {
    cache_pos = decode_position_delta = 0;
    batch = next_batch;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        auto& cache = layers[index];
        if (!cache.defined() || cache.batch != next_batch) {
            graph_views_[2 * index] = {};
            graph_views_[2 * index + 1] = {};
            cache = {};
        }
    }
}

std::vector<std::pair<tyloi::Device, std::size_t>> State::growth(
    const tyloi::models::Input& input, const tyloi::models::ForwardOptions& options) const {
    const auto [next_batch, tokens] = input.shape();
    if (batch <= 0 || next_batch != batch || cache_pos < 0 || input.confirmed != 0 || input.suffix)
        throw std::invalid_argument("Gemma4 KV planning requires positive batch and tokens and a nonnegative position");
    if (cache_pos > std::numeric_limits<std::int64_t>::max() - tokens)
        throw std::overflow_error("Gemma4 KV planning position overflow");
    const auto end = cache_pos + tokens;
    std::vector<std::pair<tyloi::Device, std::size_t>> result;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        if (layers[index].defined()) continue;
        const auto& layout = layout_[index];
        const auto capacity = kv_capacity_for(layout.sliding, layout.capacity, end, options.kv_capacity);
        const auto bytes = tyloi::state::kv_bytes(batch, layout.heads, capacity, layout.head_dim,
                                                  layout.sliding, layout.dtype, layout.quantization);
        const tyloi::Device device(tyloi::DeviceType::cuda, layout.device);
        auto entry = std::find_if(result.begin(), result.end(),
                                  [device](const auto& item) { return item.first == device; });
        if (entry == result.end()) {
            result.emplace_back(device, bytes);
        } else {
            if (entry->second > std::numeric_limits<std::size_t>::max() - bytes)
                throw std::overflow_error("Gemma4 KV planning byte count overflow");
            entry->second += bytes;
        }
    }
    return result;
}

std::vector<tyloi::Tensor*> State::graph_state() {
    if (cache_pos <= 0 || layers.empty())
        throw std::runtime_error("Gemma4 graph warmup requires a prefilled KV cache");

    std::vector<::tyloi::Tensor> views;
    views.reserve(layers.size() * 2);
    for (const auto& cache : layers) {
        if (cache.paged() || !cache.defined() || cache.physical_key.dim() != 4 ||
            cache.physical_value.dim() != 4 || cache.physical_key.size(2) <= 0 ||
            cache.physical_value.size(2) != cache.physical_key.size(2))
            throw std::runtime_error("Gemma4 graph warmup requires allocated non-paged KV caches with matching capacities");
        const auto capacity = cache.physical_key.size(2);
        if (!cache.ring && cache_pos >= capacity)
            throw std::runtime_error("Gemma4 graph warmup exceeds KV cache capacity");
        const auto position = cache.ring ? cache_pos % capacity : cache_pos;
        views.push_back(tyloi::ops::slice(cache.physical_key, 2, position, position + 1));
        views.push_back(tyloi::ops::slice(cache.physical_value, 2, position, position + 1));
    }
    std::vector<tyloi::Tensor*> result;
    result.reserve(views.size());
    for (std::size_t index = 0; index < views.size(); ++index) {
        graph_views_[index] = std::move(views[index]);
        result.push_back(&graph_views_[index]);
    }
    return result;
}

std::vector<::tyloi::Tensor> State::storage() const {
    std::vector<::tyloi::Tensor> result;
    for (const auto& cache : layers) {
        if (cache.paged()) {
            if (cache.key_chunk_ptrs && cache.key_chunk_ptrs.defined()) result.push_back(cache.key_chunk_ptrs);
            if (cache.value_chunk_ptrs && cache.value_chunk_ptrs.defined()) result.push_back(cache.value_chunk_ptrs);
            if (cache.page_table && cache.page_table.defined()) result.push_back(cache.page_table);
        } else {
            if (cache.physical_key && cache.physical_key.defined()) result.push_back(cache.physical_key);
            if (cache.physical_value && cache.physical_value.defined()) result.push_back(cache.physical_value);
        }
    }
    return result;
}

bool State::supports_text_session_state() const { return false; }
bool State::supports_paged_text_session_state() const { return false; }

} // namespace tyloi::cuda::gemma4

#include "engine/ops/qsa.h"
#include "engine/state/sequence.h"
#include "engine/backends/metal/native/tensor/tensor.h"
#include "engine/backends/metal/ops/attention/attention.h"
#include "engine/backends/metal/ops/attention/kv.h"
#include "engine/backends/metal/ops/common/mlx_sparse_attention.h"

#include <mlx/mlx.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <unistd.h>

namespace tyloi::ops {
namespace mx = mlx::core;
namespace {
using metal::array;
using metal::tensor;

std::size_t add_bytes(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) throw std::overflow_error("QSA cache size overflow");
    return a + b;
}
std::size_t multiply_bytes(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::overflow_error("QSA cache size overflow");
    return a * b;
}
std::size_t allocation_bound(std::size_t bytes) {
    if (!bytes) return 0;
    const auto page = std::size_t(getpagesize());
    if (bytes > SIZE_MAX - page + 1) throw std::overflow_error("QSA cache size overflow");
    const auto request = bytes > page ? (bytes + page - 1) / page * page : bytes;
    const auto doubled = request > SIZE_MAX / 2 ? SIZE_MAX : request * 2;
    const auto plus_pages = request > SIZE_MAX - 2 * page ? SIZE_MAX : request + 2 * page;
    const auto bound = std::min(doubled, plus_pages);
    if (bound > SIZE_MAX - page + 1) throw std::overflow_error("QSA cache size overflow");
    return (bound + page - 1) / page * page;
}
Tensor on_device(const Tensor& value, Device device) {
    return value.device() == device ? value : copy(value, device);
}
mx::array select_blocks(const mx::array& scores, int offset, int pool, int budget) {
    const int batch = scores.shape(0), tokens = scores.shape(1), count = scores.shape(2);
    const int selected = std::min(budget / pool, count);
    if (selected <= 0) throw std::invalid_argument("QSA sparse attention has no complete blocks");
    auto absolute = mx::arange(offset, offset + tokens, 1, mx::int32);
    auto ends = mx::arange(0, count, 1, mx::int32) * mx::array(pool) + mx::array(pool - 1);
    auto visible = mx::less_equal(mx::reshape(ends, {1, 1, count}), mx::reshape(absolute, {1, tokens, 1}));
    auto masked = mx::where(visible, scores, mx::array(-1e30f));
    auto canonical = mx::broadcast_to(mx::reshape(mx::arange(selected, mx::int32), {1, 1, selected}),
                                      {batch, tokens, selected});
    if (count <= selected) return canonical;
    auto ranked = mx::slice(mx::argpartition(masked, count - selected, -1), {0, 0, count - selected},
                            {batch, tokens, count});
    auto complete = mx::floor_divide(absolute + mx::array(1), mx::array(pool));
    auto early = mx::reshape(mx::less_equal(complete, mx::array(selected)), {1, tokens, 1});
    return mx::contiguous(mx::sort(mx::astype(mx::where(early, canonical, ranked), mx::int32), -1));
}
mx::array token_selection(const mx::array& blocks, int offset, int length, int pool, int budget) {
    const int b = blocks.shape(0), t = blocks.shape(1), k = blocks.shape(2);
    auto absolute = mx::reshape(mx::arange(offset, offset + t, 1, mx::int32), {1, t, 1, 1});
    auto positions = mx::expand_dims(blocks * mx::array(pool), -1) +
        mx::reshape(mx::arange(pool, mx::int32), {1, 1, 1, pool});
    auto valid = mx::less_equal(mx::expand_dims(blocks * mx::array(pool) + mx::array(pool - 1), -1), absolute);
    positions = mx::reshape(mx::where(valid, positions, mx::array(-1)), {b, t, k * pool});
    if (k * pool < budget)
        positions = mx::pad(positions, {{0, 0}, {0, 0}, {0, budget - k * pool}}, mx::array(-1), "constant");
    if (pool > 1) {
        auto count = mx::remainder(absolute + mx::array(1), mx::array(pool));
        auto offsets = mx::reshape(mx::arange(pool - 1, mx::int32), {1, 1, 1, pool - 1});
        auto tail = absolute + mx::array(1) - count + offsets;
        tail = mx::where(mx::less(offsets, count), tail, mx::array(-1));
        tail = mx::broadcast_to(mx::reshape(tail, {1, t, pool - 1}), {b, t, pool - 1});
        positions = mx::concatenate({positions, tail}, -1);
    }
    (void)length;
    return mx::astype(positions, mx::int32);
}
mx::array attention_mask(const mx::array* selected, int batch, int tokens, int length, int offset) {
    auto keys = mx::reshape(mx::arange(length, mx::int32), {1, 1, length});
    auto queries = mx::reshape(mx::arange(offset, offset + tokens, 1, mx::int32), {1, tokens, 1});
    auto allowed = mx::less_equal(keys, queries);
    if (selected) {
        auto safe = mx::where(mx::greater_equal(*selected, mx::array(0)), *selected, mx::array(length));
        auto indicator = mx::zeros({batch, tokens, length + 1}, mx::int32);
        auto updates = mx::ones(selected->shape(), mx::int32);
        indicator = mx::put_along_axis(indicator, safe, updates, -1);
        indicator = mx::slice(indicator, {0, 0, 0}, {batch, tokens, length});
        allowed = mx::logical_and(allowed, mx::greater(indicator, mx::array(0)));
    }
    return mx::expand_dims(mx::where(allowed, mx::array(0.f),
        mx::array(-std::numeric_limits<float>::infinity())), 1);
}
}

struct Qsa::Impl {
    QsaWeights weights;
    QsaConfig config;
    RopeTable rotary;
    int key_width, value_width;
    ScalarType cache_type;
    Impl(QsaWeights value, QsaConfig geometry)
        : weights(std::move(value)), config(std::move(geometry)),
          rotary(make_rope(config.maximum, config.rotary, config.rope_base, config.device,
                           config.sections, config.interleaved, false, config.yarn)),
          key_width(config.quantization.enabled() ? tyloi::state::kv_packed_width(int(config.width), config.quantization.key_bits()) : int(config.width)),
          value_width(config.quantization.enabled() ? tyloi::state::kv_packed_width(int(config.width), config.quantization.value_bits()) : int(config.width)),
          cache_type(config.quantization.enabled() ? ScalarType::uint32 : kFloat16) {
        if (!config.device.is_metal() || config.heads <= 0 || config.kv_heads <= 0 || config.heads % config.kv_heads ||
            config.width <= 0 || config.index_heads <= 0 || config.index_width <= 0 || config.pool <= 0 ||
            config.budget <= 0 || config.budget % config.pool || config.maximum <= 0 ||
            !tyloi::state::kv_bits_valid(config.quantization.bits) || weights.input_projection.size() != (weights.has_gate ? 5 : 4))
            throw std::invalid_argument("invalid Metal QSA configuration");
    }
};
struct Qsa::State::Impl {
    tyloi::state::Sequence keys, values, index;
    explicit Impl(const Qsa::Impl& model)
        : keys(model.config.maximum, model.config.kv_heads * model.key_width, model.cache_type),
          values(model.config.maximum, model.config.kv_heads * model.value_width, model.cache_type),
          index(model.config.maximum, model.config.index_width) {}
};
struct Qsa::HostLoan::Impl {};
Qsa::HostLoan::HostLoan(std::span<State* const>, std::size_t base_bytes, std::int64_t, std::int64_t)
    : bytes_(base_bytes) {}
Qsa::HostLoan::~HostLoan() = default;
void Qsa::HostLoan::prepare() {}
void Qsa::HostLoan::collect() {}
Qsa::Qsa(QsaWeights weights, QsaConfig config) : impl_(std::make_unique<Impl>(std::move(weights), std::move(config))) {}
Qsa::~Qsa() = default;
Qsa::Qsa(Qsa&&) noexcept = default;
Qsa& Qsa::operator=(Qsa&&) noexcept = default;
Qsa::State::State(const Qsa& model) : impl_(std::make_unique<Impl>(*model.impl_)) {}
Qsa::State::~State() = default;
Qsa::State::State(State&&) noexcept = default;
Qsa::State& Qsa::State::operator=(State&&) noexcept = default;
std::int64_t Qsa::position(const State& state) const { return state.impl_->keys.position(); }
bool Qsa::managed_memory(const State&) const { return false; }
std::int64_t Qsa::graph_capacity(const State&) const { return 0; }
std::vector<Tensor*> Qsa::graph_state(State&) const { return {}; }
void Qsa::activate_memory(State&) const {}

std::size_t Qsa::allocation_bytes(const State& state, std::int64_t batch, std::int64_t tokens, bool reset) const {
    auto measure = [&](const tyloi::state::Sequence& current, std::int64_t width, ScalarType dtype) {
        tyloi::state::Sequence fresh(impl_->config.maximum, width, dtype);
        const auto& sequence = reset ? fresh : current;
        auto growth = sequence.growth_bytes(batch, tokens);
        if (growth) return multiply_bytes(allocation_bound(growth), sequence.position() ? 3 : 2);
        // Metal's functional slice update allocates a full replacement backing.
        return allocation_bound(sequence.storage().nbytes());
    };
    return add_bytes(add_bytes(measure(state.impl_->keys, impl_->config.kv_heads * impl_->key_width, impl_->cache_type),
                               measure(state.impl_->values, impl_->config.kv_heads * impl_->value_width, impl_->cache_type)),
                     measure(state.impl_->index, impl_->config.index_width, kFloat16));
}
std::size_t Qsa::growth_bytes(const State& state, std::int64_t tokens) const {
    const auto& storage = state.impl_->keys.storage();
    return allocation_bytes(state, storage.defined() ? storage.size(0) : 1, tokens, false);
}
std::size_t Qsa::exit_growth(const State&, std::int64_t) const { return 0; }
std::size_t Qsa::graph_exit_growth_bytes(const State&) const { return 0; }
std::size_t Qsa::graph_growth_bytes(const State&, std::int64_t) const { throw std::logic_error("Metal QSA does not support graphs"); }
bool Qsa::supports_session(const State&) const { return true; }
void Qsa::validate_session(const State&, const Snapshot& snapshot, std::int64_t count, Device device) const {
    const auto check = [&](const Tensor& value, std::int64_t width, ScalarType type) {
        return value.defined() && !value.is_encoded() && value.device() == device && value.scalar_type() == type &&
            value.dim() == 3 && value.size(0) == 1 && value.size(1) == count && value.size(2) == width;
    };
    if (snapshot.ram || count <= 0 || count > impl_->config.maximum ||
        !check(snapshot.keys, impl_->config.kv_heads * impl_->key_width, impl_->cache_type) ||
        !check(snapshot.values, impl_->config.kv_heads * impl_->value_width, impl_->cache_type) ||
        !check(snapshot.index, impl_->config.index_width, kFloat16))
        throw std::invalid_argument("incompatible Metal QSA session state");
}
std::shared_ptr<void> Qsa::reserve_session(State&, const tyloi::state::qsa::Session&) const {
    throw std::logic_error("Metal QSA does not support RAM session state");
}
std::size_t Qsa::fresh_session_growth(const State& state, std::int64_t count) const {
    auto result = allocation_bytes(state, 1, count, true);
    const auto rows = std::size_t(count);
    const auto key = multiply_bytes(multiply_bytes(rows, std::size_t(impl_->config.kv_heads * impl_->key_width)), scalar_size(impl_->cache_type));
    const auto value = multiply_bytes(multiply_bytes(rows, std::size_t(impl_->config.kv_heads * impl_->value_width)), scalar_size(impl_->cache_type));
    const auto index = multiply_bytes(multiply_bytes(rows, std::size_t(impl_->config.index_width)), scalar_size(kFloat16));
    result = add_bytes(result, allocation_bound(key));
    result = add_bytes(result, allocation_bound(value));
    return add_bytes(result, allocation_bound(index));
}
void Qsa::stage_session(State& state, const Snapshot& saved, std::int64_t count, Device device) const {
    if (device != impl_->config.device) throw std::invalid_argument("QSA restore target differs from layer device");
    validate_session(state, saved, count, saved.keys.device());
    auto next = std::make_unique<State::Impl>(*impl_);
    next->keys.append(on_device(saved.keys, device));
    next->values.append(on_device(saved.values, device));
    next->index.append(on_device(saved.index, device));
    state.impl_ = std::move(next);
}
std::size_t Qsa::session_bytes(const State&, std::int64_t count) const {
    if (count <= 0) return 0;
    auto kv = multiply_bytes(std::size_t(impl_->config.kv_heads), std::size_t(impl_->key_width + impl_->value_width));
    kv = multiply_bytes(kv, scalar_size(impl_->cache_type));
    return multiply_bytes(std::size_t(count), add_bytes(kv, multiply_bytes(std::size_t(impl_->config.index_width), 2)));
}
void Qsa::capture_session(const State& state, Snapshot& snapshot, std::int64_t count, std::size_t& bytes) const {
    if (count <= 0 || count > position(state)) throw std::invalid_argument("invalid QSA session boundary");
    const auto save = [count](const tyloi::state::Sequence& value) { return clone(slice(value.storage(), 1, 0, count)); };
    Snapshot result{save(state.impl_->keys), save(state.impl_->values), save(state.impl_->index)};
    bytes = add_bytes(bytes, add_bytes(add_bytes(result.keys.nbytes(), result.values.nbytes()), result.index.nbytes()));
    snapshot = std::move(result);
}
void Qsa::restore_session(State& state, const Snapshot& saved, std::int64_t count) const {
    stage_session(state, saved, count, impl_->config.device);
}
std::vector<Tensor> Qsa::resident_cache_tensors(const State& state) const {
    return {state.impl_->keys.storage(), state.impl_->values.storage(), state.impl_->index.storage()};
}
std::vector<Tensor> Qsa::storage(const State& state) const { return resident_cache_tensors(state); }
void Qsa::leave_graph(State&) const {}
void Qsa::reset(State& state) const { state.impl_ = std::make_unique<State::Impl>(*impl_); }
void Qsa::truncate(State& state, std::int64_t keep) const {
    state.impl_->keys.truncate(keep); state.impl_->values.truncate(keep); state.impl_->index.truncate(keep);
}
void Qsa::prepare_graph(State&, const Tensor&, std::int64_t, std::int64_t) const { throw std::logic_error("Metal QSA does not support graphs"); }
void Qsa::advance_graph(State&, std::int64_t, std::int64_t) const { throw std::logic_error("Metal QSA does not support graphs"); }
std::vector<Tensor> Qsa::project_inputs(const Tensor& input) const {
    auto result = project_group(input, impl_->weights.input_projection);
    if (impl_->weights.has_gate) {
        auto b = input.size(0), t = input.size(1), h = impl_->config.heads, d = impl_->config.width;
        std::array<Tensor, 2> pair{reshape(result[0], {b, t, h, d}), reshape(result[4], {b, t, h, d})};
        result[0] = reshape(cat(pair, -1), {b, t, 2 * h * d});
        result.pop_back();
    }
    return result;
}
Tensor Qsa::project_output(const Tensor& input) const { return project(input, impl_->weights.output); }
Tensor Qsa::forward(State& state, const Tensor& hidden, const Tensor& positions, const Tensor& full_positions,
        bool use_cache, std::vector<Tensor>* selection_trace, std::vector<Tensor> projections,
        bool project_result, bool history_only) const {
    const auto& c = impl_->config;
    const auto& w = impl_->weights;
    if (!hidden.is_metal() || hidden.device() != c.device || hidden.dim() != 3 ||
        hidden.size(0) <= 0 || hidden.size(1) <= 0 || (history_only && !use_cache))
        throw std::invalid_argument("Metal QSA requires nonempty Metal [batch,tokens,width] input");
    const auto b = hidden.size(0), t = hidden.size(1), offset = use_cache ? position(state) : 0;
    if (offset > c.maximum || t > c.maximum - offset || full_positions.size(-1) != offset + t ||
        (use_cache && (state.impl_->values.position() != offset || state.impl_->index.position() != offset)))
        throw std::invalid_argument("QSA cache and position geometry mismatch");
    auto pos = on_device(positions, c.device);
    auto all_positions = on_device(full_positions, c.device);
    const bool grid = !c.sections.empty() && pos.dim() == 2 && pos.size(0) == 3;
    auto normalized = [&](const Tensor& value, const Tensor& weight) {
        return rms_norm(value, weight, c.eps, 1.0, value.scalar_type());
    };
    auto rotate = [&](const Tensor& value, const Tensor& selected, bool multi_axis) {
        return rope(value, selected, impl_->rotary, false, multi_axis);
    };
    if (projections.empty()) {
        if (history_only) {
            projections.resize(4);
            for (int i = 1; i < 4; ++i) projections[i] = project(hidden, w.input_projection[i]);
        } else projections = project_inputs(hidden);
    }
    if (projections.size() != 4) throw std::invalid_argument("QSA requires four projected inputs");
    auto key = reshape(projections[1], {b, t, c.kv_heads, c.width});
    key = transpose(normalized(key, w.key_norm), 1, 2);
    key = transpose(rotate(key, pos, grid), 1, 2);
    auto value = reshape(projections[2], {b, t, c.kv_heads, c.width});
    auto raw = slice(projections[3], -1, c.index_heads * c.index_width,
                     (c.index_heads + 1) * c.index_width);
    auto encode = [&](const Tensor& input, int bits, bool is_value) {
        return c.quantization.enabled() ? tensor(metal::mlx_kv_encode(array(input), bits, is_value)) : cast(input, kFloat16);
    };
    auto encoded_key = encode(key, c.quantization.key_bits(), false);
    auto encoded_value = encode(value, c.quantization.value_bits(), true);
    Tensor keys = reshape(encoded_key, {b, t, c.kv_heads * impl_->key_width});
    Tensor values = reshape(encoded_value, {b, t, c.kv_heads * impl_->value_width});
    Tensor raw_index = raw;
    try {
        if (use_cache) {
            keys = state.impl_->keys.append(keys);
            values = state.impl_->values.append(values);
            raw_index = state.impl_->index.append(raw);
        }
        if (history_only) return {};
        auto query_pair = reshape(projections[0], {b, t, c.heads, 2 * c.width});
        auto gate = slice(query_pair, -1, c.width, 2 * c.width);
        auto query = transpose(normalized(slice(query_pair, -1, 0, c.width), w.query_norm), 1, 2);
        query = rotate(query, pos, grid);
        auto iq = reshape(slice(projections[3], -1, 0, c.index_heads * c.index_width),
                          {b, t, c.index_heads, c.index_width});
        iq = transpose(normalized(iq, w.index_query_norm), 1, 2);
        iq = transpose(rotate(iq, pos, grid), 1, 2);
        const auto length = offset + t;
        auto packed_keys = transpose(reshape(keys, {b, length, c.kv_heads, impl_->key_width}), 1, 2);
        auto packed_values = transpose(reshape(values, {b, length, c.kv_heads, impl_->value_width}), 1, 2);
        std::optional<mx::array> blocks, selected;
        if (length > c.budget) {
            const auto pools = length / c.pool;
            auto pooled_input = reshape(slice(raw_index, 1, 0, pools * c.pool), {b, pools, c.pool, c.index_width});
            auto pooled = tensor(mx::astype(mx::mean(mx::astype(array(pooled_input), mx::float32), 2), array(raw_index).dtype()));
            pooled = normalized(pooled, w.index_key_norm);
            auto starts = tensor(mx::arange(0, int(pools * c.pool), int(c.pool), mx::int32));
            auto block_positions = index_select(all_positions, -1, starts);
            const bool block_grid = !c.sections.empty() && block_positions.dim() == 2 && block_positions.size(0) == 3;
            pooled = reshape(rotate(reshape(pooled, {b, 1, pools, c.index_width}), block_positions, block_grid),
                             {b, pools, c.index_width});
            auto query32 = mx::astype(array(iq), mx::float32);
            auto key32 = mx::astype(array(pooled), mx::float32);
            auto head_scores = mx::matmul(query32, mx::expand_dims(mx::transpose(key32, {0, 2, 1}), 1));
            auto scores = mx::sum(mx::maximum(head_scores, mx::array(0.f)), 2) / float(std::sqrt(double(c.index_width)));
            blocks = select_blocks(scores, int(offset), int(c.pool), int(c.budget));
            selected = token_selection(*blocks, int(offset), int(length), int(c.pool), int(c.budget));
            if (selection_trace) *selection_trace = {tensor(scores), tensor(*selected), iq, pooled};
        }
        mx::array attended = [&] {
            if (c.quantization.enabled())
                return metal::mlx_kv_sparse_attention(array(query), array(packed_keys), array(packed_values),
                    blocks, c.quantization, int(length), int(offset), int(c.pool), int(c.budget));
            if (blocks && c.heads == 24 && c.kv_heads == 2 && c.width == 256)
                return metal::mlx_sparse_block_gqa_attention(array(query), array(packed_keys), array(packed_values),
                                                             *blocks, int(offset), int(c.pool));
            auto mask = mx::astype(attention_mask(selected ? &*selected : nullptr, int(b), int(t), int(length), int(offset)),
                                   array(query).dtype());
            return mx::transpose(metal::scaled_dot_product_attention(array(query), array(packed_keys), array(packed_values),
                                                                       false, 0.f, mask), {0, 2, 1, 3});
        }();
        auto gated = mx::astype(attended, mx::float32) * mx::sigmoid(mx::astype(array(gate), mx::float32));
        auto result = tensor(mx::astype(mx::reshape(gated, {int(b), int(t), int(c.heads * c.width)}), array(hidden).dtype()));
        return project_result ? project_output(result) : result;
    } catch (...) {
        if (use_cache) truncate(state, offset);
        throw;
    }
}
Tensor Qsa::forward_graph(State&, const Tensor&, const Tensor&, const Tensor&, std::vector<Tensor>, bool, bool, bool) const {
    throw std::logic_error("Metal QSA does not support graphs");
}

struct Qsa::Graph::Impl {};
Qsa::Graph::Graph(std::vector<Request>, bool) : impl_(std::make_unique<Impl>()) {}
Qsa::Graph::~Graph() = default;
Qsa::Graph::Graph(Graph&&) noexcept = default;
Qsa::Graph& Qsa::Graph::operator=(Graph&&) noexcept = default;
bool Qsa::Graph::eligible(bool) const { return false; }
bool Qsa::Graph::reusable() const { return false; }
bool Qsa::Graph::warmup_supported() const { return false; }
void Qsa::Graph::admit(std::int64_t, std::int64_t, bool) { throw std::logic_error("Metal QSA does not support graphs"); }
int Qsa::Graph::pending_kind() const { return 0; }
std::pair<int, std::vector<Tensor>> Qsa::Graph::pending(std::int64_t, std::int64_t) const { return {}; }
std::vector<std::pair<Device, std::size_t>> Qsa::Graph::growth() const {
    throw std::logic_error("Metal QSA does not support graphs");
}
void Qsa::Graph::materialize() { throw std::logic_error("Metal QSA does not support graphs"); }
bool Qsa::Graph::clear_required() const { return false; }
void Qsa::Graph::layout() { throw std::logic_error("Metal QSA does not support graphs"); }
const std::vector<Qsa::Graph::Row>& Qsa::Graph::rows() const { throw std::logic_error("Metal QSA does not support graphs"); }
std::vector<Tensor> Qsa::Graph::storage() const { return {}; }
void Qsa::Graph::prepare(std::size_t, std::size_t, const Tensor&) const { throw std::logic_error("Metal QSA does not support graphs"); }
void Qsa::Graph::prepare(std::size_t, const Tensor&) const { throw std::logic_error("Metal QSA does not support graphs"); }
} // namespace tyloi::ops

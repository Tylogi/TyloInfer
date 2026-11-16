#include "engine/ops/gdn.h"
#include "engine/ops/ops.h"
#include "engine/backends/metal/native/tensor/tensor.h"
#include "engine/backends/metal/ops/common/mlx_linear_attention.h"

#include <mlx/mlx.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace tyloi::ops {
namespace mx = mlx::core;
namespace {
thread_local std::int64_t projection_rows = 0;

std::size_t add(std::size_t left, std::size_t right) {
    if (left > std::numeric_limits<std::size_t>::max() - right)
        throw std::overflow_error("GDN growth overflow");
    return left + right;
}
std::size_t multiply(std::size_t value, std::size_t count) {
    if (count && value > std::numeric_limits<std::size_t>::max() / count)
        throw std::overflow_error("GDN growth overflow");
    return value * count;
}
std::size_t bytes(std::initializer_list<std::int64_t> shape, Device target) {
    return make_contiguous_view(nullptr, shape, kFloat32, target).nbytes();
}
Tensor slice_rows(const Tensor& value, std::int64_t first, std::int64_t count) {
    return ops::slice(value, 1, first, first + count);
}
Tensor join(const std::vector<Tensor>& values, std::int64_t dim) {
    if (values.empty()) throw std::invalid_argument("empty GDN projection list");
    return values.size() == 1 ? values.front() : ops::cat(values, dim);
}
void require_metal(const Tensor& value, const char* what) {
    if (!value.defined() || value.device() != Device(DeviceType::metal))
        throw std::invalid_argument(what);
}
}

struct Gdn::State::Impl {};
struct Gdn::Impl {
    GdnWeights weights;
    std::int64_t keys, values, width, kernel;
    double eps;
    bool silu_gate;
    Device target;

    Impl(GdnWeights w, std::int64_t k, std::int64_t v, std::int64_t d,
         std::int64_t conv, double epsilon, bool silu, Device device)
        : weights(std::move(w)), keys(k), values(v), width(d), kernel(conv),
          eps(epsilon), silu_gate(silu), target(device) {
        if (target != Device(DeviceType::metal))
            throw std::invalid_argument("Metal GDN requires Metal device 0");
        if (keys <= 0 || values <= 0 || values % keys ||
            (width != 32 && width != 64 && width != 128) || kernel <= 1 ||
            !std::isfinite(eps) || eps <= 0 || keys > (INT32_MAX - values) / 2 ||
            width > INT32_MAX / (2 * keys + values) || kernel > INT32_MAX ||
            (weights.qkv_count != 1 && weights.qkv_count != 3) ||
            weights.input_projection.size() != weights.qkv_count + 3)
            throw std::invalid_argument("invalid Metal GDN geometry or projections");
        require_metal(weights.conv, "Metal GDN requires Metal convolution weights");
        require_metal(weights.dt_bias, "Metal GDN requires Metal time bias");
        require_metal(weights.a_log, "Metal GDN requires Metal decay weights");
        require_metal(weights.norm, "Metal GDN requires Metal norm weights");
    }
    std::int64_t channels() const { return (2 * keys + values) * width; }
    std::size_t conv_bytes(std::int64_t batch) const { return bytes({batch, kernel - 1, channels()}, target); }
    std::size_t recurrent_bytes(std::int64_t batch) const { return bytes({batch, values, width, width}, target); }
};

Gdn::State::State() : impl_(std::make_unique<Impl>()) {}
Gdn::State::~State() = default;
Gdn::State::State(State&&) noexcept = default;
Gdn::State& Gdn::State::operator=(State&&) noexcept = default;
Gdn::Gdn(GdnWeights weights, std::int64_t key_heads, std::int64_t value_heads,
        std::int64_t width, std::int64_t kernel, double eps, bool silu_gate, Device target)
    : impl_(std::make_unique<Impl>(std::move(weights), key_heads, value_heads, width, kernel, eps, silu_gate, target)) {}
Gdn::~Gdn() = default;
Gdn::Gdn(Gdn&&) noexcept = default;
Gdn& Gdn::operator=(Gdn&&) noexcept = default;
Gdn::ProjectionScope::ProjectionScope(std::int64_t rows) : previous_(projection_rows) { projection_rows = rows; }
Gdn::ProjectionScope::~ProjectionScope() { projection_rows = previous_; }

void Gdn::commit(State& state) const {
    state.saved_conv = {}; state.saved_state = {};
    state.saved_prefix_conv.clear(); state.saved_prefix_state.clear(); state.saved_prefix_position = -1;
}
void Gdn::reset(State& state, bool retain) const {
    commit(state);
    if (retain) {
        if (state.conv.defined()) metal::replace(state.conv, mx::zeros_like(metal::array(state.conv)));
        if (state.recurrent.defined()) metal::replace(state.recurrent, mx::zeros_like(metal::array(state.recurrent)));
    } else { state.conv = {}; state.recurrent = {}; }
}
void Gdn::rollback(State& state, std::int64_t keep) const {
    if (!state.saved_conv.defined() || !state.saved_state.defined())
        throw std::logic_error("GDN has no speculative checkpoint");
    Tensor conv = state.saved_conv, recurrent = state.saved_state;
    if (keep >= 0 && state.saved_prefix_position >= 0) {
        const auto index = keep - state.saved_prefix_position;
        if (index < 0 || std::size_t(index) >= state.saved_prefix_conv.size() ||
            state.saved_prefix_conv.size() != state.saved_prefix_state.size())
            throw std::out_of_range("GDN prefix checkpoint is out of range");
        conv = state.saved_prefix_conv[index]; recurrent = state.saved_prefix_state[index];
    }
    state.conv = std::move(conv); state.recurrent = std::move(recurrent);
    commit(state);
}
void Gdn::initialize_request_cache(State& state, bool) const {
    if (state.conv.defined() != state.recurrent.defined())
        throw std::logic_error("incomplete GDN request state");
    if (state.conv.defined()) return;
    const auto options = TensorOptions().dtype(kFloat32).device(impl_->target);
    auto conv = ops::zeros({1, impl_->kernel - 1, impl_->channels()}, options);
    auto recurrent = ops::zeros({1, impl_->values, impl_->width, impl_->width}, options);
    state.conv = std::move(conv);
    state.recurrent = std::move(recurrent);
}
std::size_t Gdn::growth_bytes(const State& state) const {
    return add(state.conv.defined() ? 0 : impl_->conv_bytes(1),
        state.recurrent.defined() ? 0 : impl_->recurrent_bytes(1));
}
std::pair<std::size_t, std::size_t> Gdn::ordinary_growth_bytes(
        const State& state, std::int64_t batch, std::int64_t tokens) const {
    if (batch <= 0 || tokens <= 0 || tokens > INT32_MAX - impl_->kernel ||
        (state.conv.defined() && state.conv.size(0) != batch) ||
        (state.recurrent.defined() && state.recurrent.size(0) != batch))
        throw std::invalid_argument("invalid Metal GDN execution geometry");
    const auto conv = impl_->conv_bytes(batch), recurrent = impl_->recurrent_bytes(batch);
    const auto retained = add(state.conv.defined() ? 0 : conv, state.recurrent.defined() ? 0 : recurrent);
    // Lazy convolution, gate preparation and recurrence retain their intermediate arrays until submission.
    auto temporary = add(conv, multiply(recurrent, 2));
    temporary = add(temporary, bytes({batch, tokens + impl_->kernel - 1, impl_->channels()}, impl_->target));
    temporary = add(temporary, multiply(bytes({batch, tokens, impl_->channels()}, impl_->target), 3));
    temporary = add(temporary, multiply(bytes({batch, tokens, impl_->values}, impl_->target), 2));
    return {retained, temporary};
}
std::size_t Gdn::checkpoint_growth_bytes(const State& state, std::int64_t batch,
        std::int64_t tokens, std::int64_t confirmed) const {
    if (batch <= 0 || tokens <= 0 || confirmed <= 0 || confirmed >= tokens)
        throw std::invalid_argument("invalid GDN checkpoint geometry");
    if (tokens <= 8 && (!state.conv.defined() || !state.recurrent.defined()))
        throw std::invalid_argument("GDN verification requires committed recurrent state");
    return multiply(add(impl_->conv_bytes(batch), impl_->recurrent_bytes(batch)),
        tokens <= 8 ? std::size_t(tokens - confirmed) : 1);
}
std::size_t Gdn::request_cache_growth_bytes(const State& state) const { return growth_bytes(state); }
void Gdn::restore_session(State& state, const Tensor& conv, const Tensor& recurrent, bool retain) const {
    require_metal(conv, "GDN session convolution must be on Metal");
    require_metal(recurrent, "GDN session recurrence must be on Metal");
    if (conv.dim() != 3 || recurrent.dim() != 4 || conv.size(0) != recurrent.size(0) ||
        conv.size(1) != impl_->kernel - 1 || conv.size(2) != impl_->channels() ||
        recurrent.size(1) != impl_->values || recurrent.size(2) != impl_->width ||
        recurrent.size(3) != impl_->width || conv.scalar_type() != kFloat32 || recurrent.scalar_type() != kFloat32)
        throw std::invalid_argument("GDN session geometry mismatch");
    if (retain) {
        if (!state.conv.defined() || !state.recurrent.defined() ||
            state.conv.size(0) != conv.size(0) || state.recurrent.size(0) != recurrent.size(0))
            throw std::invalid_argument("GDN retained session geometry mismatch");
        commit(state); ops::copy_into(state.conv, conv); ops::copy_into(state.recurrent, recurrent);
    } else {
        commit(state); state.conv = ops::clone(conv); state.recurrent = ops::clone(recurrent);
    }
}
Tensor Gdn::recurrent_state(const State& state) const {
    return state.recurrent.defined() ? ops::transpose(state.recurrent, -1, -2) : Tensor{};
}
std::vector<Tensor*> Gdn::graph_state(State&) const { throw std::logic_error("Metal GDN graph capture is unsupported"); }
std::vector<Tensor> Gdn::storage(const State& state) const {
    std::vector<Tensor> result;
    const auto add_value = [&](const Tensor& value) { if (value.defined()) result.push_back(value); };
    add_value(state.conv); add_value(state.recurrent); add_value(state.saved_conv); add_value(state.saved_state);
    for (const auto& value : state.saved_prefix_conv) add_value(value);
    for (const auto& value : state.saved_prefix_state) add_value(value);
    return result;
}
std::vector<Tensor> Gdn::project_inputs(const Tensor& input) const {
    require_metal(input, "GDN input must be on Metal");
    const auto project = [&](const Tensor& value) {
        std::vector<Tensor> qkv;
        for (std::size_t i = 0; i < impl_->weights.qkv_count; ++i)
            qkv.push_back(ops::project(value, impl_->weights.input_projection[i]));
        std::vector<Tensor> result{join(qkv, -1)};
        for (std::size_t i = impl_->weights.qkv_count; i < impl_->weights.input_projection.size(); ++i)
            result.push_back(ops::project(value, impl_->weights.input_projection[i]));
        return result;
    };
    if (projection_rows <= 0 || input.dim() != 3 || input.size(0) != 1 || input.size(1) <= projection_rows)
        return project(input);
    std::array<std::vector<Tensor>, 4> parts;
    for (std::int64_t offset = 0; offset < input.size(1); offset += projection_rows) {
        auto values = project(slice_rows(input, offset, std::min(projection_rows, input.size(1) - offset)));
        for (std::size_t i = 0; i < parts.size(); ++i) parts[i].push_back(std::move(values[i]));
    }
    std::vector<Tensor> result;
    for (const auto& part : parts) result.push_back(join(part, 1));
    return result;
}
Tensor Gdn::project_output(const Tensor& value) const { return ops::project(value, impl_->weights.output); }
Tensor Gdn::prepare_independent_batch(const std::vector<std::pair<const Gdn*, State*>>&) {
    throw std::logic_error("Metal GDN independent batches are unsupported");
}
Tensor Gdn::forward_independent_batch(State&, const Tensor&, const std::vector<Tensor>&, const Tensor&) const {
    throw std::logic_error("Metal GDN independent batches are unsupported");
}
Tensor Gdn::forward_graph(State&, const Tensor&, std::int64_t, std::vector<Tensor>&, std::vector<Tensor>&) const {
    throw std::logic_error("Metal GDN graph capture is unsupported");
}
void Gdn::bind_graph_checkpoint(State&, const std::vector<Tensor>&, const std::vector<Tensor>&, std::int64_t) const {
    throw std::logic_error("Metal GDN graph capture is unsupported");
}
Tensor Gdn::forward_chunk(State& state, const Tensor& input, bool cache,
        std::vector<Tensor> projections, bool project_result, Tensor checkpoint_conv,
        Tensor checkpoint_state, bool* snapshot_written) const {
    if (snapshot_written) *snapshot_written = false;
    require_metal(input, "GDN input must be on Metal");
    if (input.dim() != 3 || input.size(0) <= 0 || input.size(1) <= 0 ||
        input.size(0) > INT32_MAX || input.size(1) > INT32_MAX)
        throw std::invalid_argument("GDN input must have nonempty [batch,tokens,width] geometry");
    if (checkpoint_conv.defined() != checkpoint_state.defined())
        throw std::invalid_argument("GDN checkpoint tensors must be supplied together");
    if (!cache && checkpoint_conv.defined())
        throw std::invalid_argument("GDN checkpoint requires caching");
    const auto b = int(input.size(0)), t = int(input.size(1));
    const auto nk = int(impl_->keys), nv = int(impl_->values), d = int(impl_->width);
    const auto channels = int(impl_->channels());
    if (cache && (state.conv.defined() != state.recurrent.defined()))
        throw std::logic_error("incomplete GDN request state");
    if (cache && state.conv.defined() &&
        (state.conv.size(0) != b || state.recurrent.size(0) != b))
        throw std::invalid_argument("GDN batch size changed");
    if (projections.empty()) projections = project_inputs(input);
    if (projections.size() != 4)
        throw std::invalid_argument("GDN requires qkv, gate, alpha and beta projections");
    for (const auto& value : projections) require_metal(value, "GDN projections must be on Metal");
    if (projections[0].dim() != 3 || projections[0].size(0) != b || projections[0].size(1) != t ||
        projections[0].size(2) != channels || projections[1].dim() != 3 ||
        projections[1].size(0) != b || projections[1].size(1) != t || projections[1].size(2) != nv * d ||
        projections[2].numel() != std::int64_t(b) * t * nv ||
        projections[3].numel() != std::int64_t(b) * t * nv)
        throw std::invalid_argument("GDN projection geometry mismatch");
    const auto initial_conv = cache && state.conv.defined() ? metal::array(state.conv) :
        mx::zeros({b, int(impl_->kernel - 1), channels}, mx::float32);
    const auto initial_recurrent = cache && state.recurrent.defined() ? metal::array(state.recurrent) :
        mx::zeros({b, nv, d, d}, mx::float32);
    auto pieces = mx::split(metal::array(projections[0]), mx::Shape{2 * nk * d}, -1);
    const auto alpha = mx::reshape(metal::array(projections[2]), {b, t, nv});
    const auto beta = mx::reshape(metal::array(projections[3]), {b, t, nv});
    const auto decay = -mx::exp(metal::array(impl_->weights.a_log)) * mx::array(1.0f);
    auto gates = metal::gated_delta_gates(alpha, beta, metal::array(impl_->weights.dt_bias), decay);
    auto convolved = metal::linear_conv_qkv(initial_conv, pieces[0], pieces[1],
        metal::array(impl_->weights.conv), nk, nv, d, d, std::nullopt, 1e-6f);
    auto recurrent = metal::gated_delta_net(convolved.query, convolved.key, convolved.value,
        gates.gate, gates.beta, initial_recurrent, true, impl_->weights.tiled_heads);
    auto normalized = metal::gated_delta_output_norm(recurrent.output, metal::array(projections[1]),
        metal::array(impl_->weights.norm), float(impl_->eps), impl_->silu_gate, metal::array(input).dtype());
    Tensor output = metal::tensor(std::move(normalized));
    if (project_result) output = project_output(output);
    if (cache) {
        state.conv = metal::tensor(std::move(convolved.state));
        state.recurrent = metal::tensor(std::move(recurrent.state));
        if (checkpoint_conv.defined()) {
            ops::copy_into(checkpoint_conv, state.conv);
            ops::copy_into(checkpoint_state, state.recurrent);
            if (snapshot_written) *snapshot_written = true;
        }
    }
    return output;
}

Tensor Gdn::forward(State& state, const Tensor& input, bool cache,
        std::int64_t confirmed, std::int64_t cache_position) const {
    require_metal(input, "GDN input must be on Metal");
    if (input.dim() != 3 || input.size(0) <= 0 || input.size(1) <= 0 || confirmed < 0 ||
        confirmed > input.size(1) || (confirmed && !cache))
        throw std::invalid_argument("invalid GDN verification window");
    const auto tokens = input.size(1);
    if (!confirmed || confirmed == tokens) return forward_chunk(state, input, cache);
    if (state.saved_conv.defined()) throw std::logic_error("resolve the previous GDN checkpoint");
    const bool prefixes = cache_position >= 0 && tokens <= 8;
    if (prefixes && (!state.conv.defined() || !state.recurrent.defined()))
        throw std::invalid_argument("GDN verification requires committed recurrent state");
    auto projected = project_inputs(input);
    const auto run = [&](std::int64_t first, std::int64_t count) {
        std::vector<Tensor> part;
        part.reserve(projected.size());
        for (const auto& value : projected) part.push_back(slice_rows(value, first, count));
        return forward_chunk(state, slice_rows(input, first, count), true, std::move(part), false);
    };
    std::vector<Tensor> outputs;
    outputs.push_back(run(0, confirmed));
    state.saved_conv = state.conv; state.saved_state = state.recurrent;
    if (prefixes) {
        state.saved_prefix_position = cache_position + confirmed;
        state.saved_prefix_conv.push_back(state.conv); state.saved_prefix_state.push_back(state.recurrent);
    }
    try {
        for (std::int64_t pos = confirmed; pos < tokens;) {
            const auto count = prefixes ? std::int64_t(1) : tokens - pos;
            outputs.push_back(run(pos, count)); pos += count;
            if (prefixes && pos < tokens) {
                state.saved_prefix_conv.push_back(state.conv);
                state.saved_prefix_state.push_back(state.recurrent);
            }
        }
        return project_output(join(outputs, 1));
    } catch (...) { rollback(state); throw; }
}
} // namespace tyloi::ops

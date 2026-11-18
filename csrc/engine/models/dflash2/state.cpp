#include "engine/models/dflash2/model.h"
#include "engine/models/session/snapshot.h"
#include "engine/ops/ops.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tyloi::models::dflash2 {
namespace {

ScalarType ring_dtype(const Draft& model) {
    return model.compact_kv ? kBFloat16 : kFloat32;
}

bool compatible_ring(const DenseTensor& ring, const Draft& model) {
    const auto& c = model.config;
    return ring.defined() && ring.dim() == 3 && ring.size(0) == c.window &&
        ring.size(1) == c.kv_heads && ring.size(2) == c.dimension &&
        ring.scalar_type() == ring_dtype(model) && ring.device() == model.device && ring.is_contiguous();
}

std::size_t ring_bytes(const Draft& model) {
    const auto& c = model.config;
    if (c.window <= 0 || c.kv_heads <= 0 || c.dimension <= 0 || c.layers <= 0)
        throw std::invalid_argument("DFlash2 ring geometry is invalid");
    auto bytes = scalar_size(ring_dtype(model));
    for (const auto dimension : {c.window, c.kv_heads, c.dimension}) {
        if (bytes > std::numeric_limits<std::size_t>::max() / std::size_t(dimension))
            throw std::overflow_error("DFlash2 ring allocation overflow");
        bytes *= std::size_t(dimension);
    }
    return bytes;
}

std::int64_t hidden_width(const Draft& model) {
    const auto count = model.config.target_layers.size();
    if (model.config.hidden <= 0 || count == 0 ||
        count > std::size_t(std::numeric_limits<std::int64_t>::max() / model.config.hidden))
        throw std::invalid_argument("DFlash2 hidden geometry is invalid");
    return model.config.hidden * std::int64_t(count);
}

void add_growth(std::vector<std::pair<Device, std::size_t>>& growth, Device device, std::size_t bytes) {
    auto found = std::find_if(growth.begin(), growth.end(),
        [device](const auto& item) { return item.first == device; });
    if (found == growth.end()) growth.emplace_back(device, bytes);
    else {
        if (found->second > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::overflow_error("DFlash2 session allocation overflow");
        found->second += bytes;
    }
}

class SessionRestore final : public Restore {
public:
    SessionRestore(State& state, const MtpSessionState& saved) : state_(state), saved_(saved) {
        const auto& model = state.model;
        const auto* payload = std::get_if<std::vector<KvSnapshot>>(&saved_.payload);
        const auto& hidden = saved_.last_target_hidden;
        if (saved_.cache_pos <= 0 || !payload || payload->size() != std::size_t(model.config.layers) ||
            !hidden.defined() || hidden.is_encoded() || !hidden.representation<DenseTensor>() ||
            !hidden.is_contiguous() || hidden.dim() != 3 || hidden.size(0) != 1 || hidden.size(1) != 1 ||
            hidden.size(2) != hidden_width(model) ||
            (hidden.scalar_type() != kFloat16 && hidden.scalar_type() != kBFloat16 && hidden.scalar_type() != kFloat32) ||
            (hidden.device() != model.device && !hidden.device().is_cpu()))
            throw CudaSessionStateError("DFlash2 session state is incompatible");
        for (const auto& block : *payload) {
            validate_kv(block);
            if (!block.ring || block.capacity != model.config.window || block.target_device() != model.device ||
                block.k.is_encoded() || block.v.is_encoded() || !block.k.is_contiguous() || !block.v.is_contiguous() ||
                block.k.size(0) != 1 || block.k.size(1) != model.config.kv_heads ||
                block.k.size(2) != model.config.window || block.k.size(3) != model.config.dimension ||
                block.v.size(3) != model.config.dimension || block.k.scalar_type() != ring_dtype(model))
                throw CudaSessionStateError("DFlash2 session ring geometry is incompatible");
        }
    }

    std::vector<std::pair<Device, std::size_t>> growth() const override {
        std::vector<std::pair<Device, std::size_t>> result;
        const auto& payload = std::get<std::vector<KvSnapshot>>(saved_.payload);
        for (const auto& block : payload) for (const auto* ring : {&block.k, &block.v}) {
            add_growth(result, state_.model.device, ring->nbytes());
            if (ring->device().is_cpu()) add_growth(result, state_.model.device, ring->nbytes());
        }
        if (saved_.last_target_hidden.device().is_cpu())
            add_growth(result, state_.model.device, saved_.last_target_hidden.nbytes());
        return result;
    }

    void stage() override {
        const auto& payload = std::get<std::vector<KvSnapshot>>(saved_.payload);
        const auto& model = state_.model;
        const auto device = model.device;
        ops::DeviceScope guard(device);
        keys_.reserve(payload.size());
        values_.reserve(payload.size());
        for (const auto& block : payload) {
            for (const auto& [source, destination] : {
                    std::pair{&block.k, &keys_}, std::pair{&block.v, &values_}}) {
                auto ring = ops::empty({model.config.window, model.config.kv_heads, model.config.dimension},
                    TensorOptions().device(device).dtype(source->scalar_type()));
                auto target = ops::transpose(ops::reshape(ring,
                    {1, model.config.window, model.config.kv_heads, model.config.dimension}), 1, 2);
                Tensor staged = *source;
                if (source->device().is_cpu()) {
                    staged = ops::empty(source->dense().sizes(), TensorOptions().device(device).dtype(source->scalar_type()));
                    ops::copy_into(staged, *source);
                    staging_.push_back(staged);
                }
                ops::copy_into(target, staged);
                destination->push_back(ring.dense());
            }
        }
        if (saved_.last_target_hidden.device().is_cpu()) {
            hidden_ = ops::empty(saved_.last_target_hidden.dense().sizes(),
                TensorOptions().device(device).dtype(saved_.last_target_hidden.scalar_type()));
            ops::copy_into(hidden_, saved_.last_target_hidden);
        } else hidden_ = saved_.last_target_hidden;
    }

    std::vector<Tensor> storage() const override {
        std::vector<Tensor> result;
        for (const auto& ring : keys_) result.emplace_back(ring);
        for (const auto& ring : values_) result.emplace_back(ring);
        if (hidden_) result.push_back(hidden_);
        result.insert(result.end(), staging_.begin(), staging_.end());
        return result;
    }
    Tensor target_hidden() const override { return hidden_; }
    void publish() noexcept override {
        state_.release_graphs();
        state_.keys.swap(keys_);
        state_.values.swap(values_);
        state_.pos = saved_.cache_pos;
        state_.compact = {};
        state_.selector = {};
        state_.successor_rows = {};
        state_.row = 0;
    }

private:
    State& state_;
    MtpSessionState saved_;
    std::vector<DenseTensor> keys_, values_;
    Tensor hidden_;
    std::vector<Tensor> staging_;
};

} // namespace

State::State(const Draft& model) : MtpState(model), model(model) {}

std::vector<Tensor> State::storage() const {
    std::vector<Tensor> result;
    for (const auto& key : keys) if (key.defined()) result.emplace_back(key);
    for (const auto& value : values) if (value.defined()) result.emplace_back(value);
    if (compact.defined()) result.emplace_back(compact);
    if (selector.defined()) result.emplace_back(selector);
    if (successor_rows.defined()) result.emplace_back(successor_rows);
    return result;
}

std::unique_ptr<Restore> State::prepare_restore(const MtpSessionState& saved) {
    return std::make_unique<SessionRestore>(*this, saved);
}

std::unique_ptr<MtpState> Draft::make_state() const { return std::make_unique<State>(*this); }

std::vector<std::pair<Device, std::size_t>> Draft::reset_growth(const MtpState& value, std::int64_t batch) const {
    const auto& current = state<State>(value);
    if (batch != 1) throw std::invalid_argument("DFlash2 requires a single sequence");
    const auto bytes = ring_bytes(*this);
    std::size_t count = 0;
    for (int i = 0; i < config.layers; ++i) {
        if (std::size_t(i) >= current.keys.size() || !compatible_ring(current.keys[i], *this)) ++count;
        if (std::size_t(i) >= current.values.size() || !compatible_ring(current.values[i], *this)) ++count;
    }
    if (count && bytes > std::numeric_limits<std::size_t>::max() / count)
        throw std::overflow_error("DFlash2 ring allocation overflow");
    return count ? std::vector<std::pair<Device, std::size_t>>{{device, bytes * count}} :
        std::vector<std::pair<Device, std::size_t>>{};
}

void Draft::reset(MtpState& value, std::int64_t batch) const {
    auto& current = state<State>(value);
    if (batch != 1) throw std::invalid_argument("DFlash2 requires a single sequence");
    (void)ring_bytes(*this);
    ops::DeviceScope guard(device);
    current.release_graphs();
    current.keys.resize(config.layers);
    current.values.resize(config.layers);
    for (auto* rings : {&current.keys, &current.values}) for (auto& ring : *rings) {
        if (!compatible_ring(ring, *this))
            ring = ops::zeros({config.window, config.kv_heads, config.dimension},
                TensorOptions().device(device).dtype(ring_dtype(*this))).dense();
        else {
            Tensor view = ring;
            ops::fill(view, 0);
        }
    }
    current.pos = 0;
    current.compact = {};
    current.selector = {};
    current.successor_rows = {};
    current.row = 0;
}

MtpStep Draft::step(MtpState&, const Tensor&, const Tensor&, const Tensor&, const ForwardOptions&) const {
    throw std::logic_error("DFlash2 uses block drafting");
}

void Draft::trim_cache_to(MtpState& value, std::int64_t position) const {
    if (position != state<State>(value).pos)
        throw std::invalid_argument("DFlash2 stores only committed target context");
}

int Draft::maximum_draft_depth() const noexcept {
    return std::min(config.block - 1, engine::mtp::kMaximumBlockwiseDraftDepth);
}

MtpSessionState Draft::capture_session_state(const MtpState& value, std::int64_t target_position,
        const Tensor& hidden) const {
    const auto& current = state<State>(value);
    if (!session_boundary(target_position, current.pos) || !hidden.defined() || hidden.is_encoded() ||
        !hidden.representation<DenseTensor>() || hidden.dim() != 3 || hidden.size(0) != 1 ||
        hidden.size(1) != 1 || hidden.size(2) != hidden_width(*this) ||
        !hidden.device().is_cuda() || (hidden.scalar_type() != kFloat16 &&
            hidden.scalar_type() != kBFloat16 && hidden.scalar_type() != kFloat32) ||
        current.keys.size() != std::size_t(config.layers) || current.values.size() != std::size_t(config.layers))
        throw std::invalid_argument("DFlash2 session is outside the committed context boundary");
    MtpSessionState saved;
    saved.cache_pos = current.pos;
    saved.last_target_hidden = ops::clone(hidden);
    saved.bytes = saved.last_target_hidden.nbytes();
    auto& payload = std::get<std::vector<KvSnapshot>>(saved.payload);
    ops::DeviceScope guard(device);
    for (int i = 0; i < config.layers; ++i) {
        if (!compatible_ring(current.keys[i], *this) || !compatible_ring(current.values[i], *this))
            throw std::runtime_error("DFlash2 session ring is unavailable");
        const auto capture = [&](const DenseTensor& ring) {
            return ops::clone(ops::transpose(ops::reshape(ring,
                {1, config.window, config.kv_heads, config.dimension}), 1, 2));
        };
        KvSnapshot block{capture(current.keys[i]), capture(current.values[i]), config.window, true};
        for (const auto* ring : {&block.k, &block.v}) {
            if (saved.bytes > std::numeric_limits<std::size_t>::max() - ring->nbytes())
                throw std::overflow_error("DFlash2 session size overflow");
            saved.bytes += ring->nbytes();
        }
        payload.push_back(std::move(block));
    }
    return saved;
}

} // namespace tyloi::models::dflash2

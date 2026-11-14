#include "engine/backends/metal/ops/moe.h"
#include "engine/backends/metal/native/tensor/tensor.h"
#include "engine/backends/metal/ops/common/mlx_moe_ops.h"
#include "engine/ops.h"
#include "engine/weights/moe/batch.h"
#include "engine/weights/moe/experts.h"
#include "backends/metal/runtime/mlx_packed_expert_cache.h"
#include "backends/metal/runtime/mlx_tq_expert_cache.h"
#include "backends/metal/storage/gguf/mlx_gguf_expert_cache.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace mx = mlx::core;

namespace tyloi::metal {

ExpertBank::ExpertBank(ExpertCache cache, std::size_t layer, ExpertRole role,
                       std::array<std::int64_t, 3> shape)
    : cache_(std::move(cache)), layer_(layer), role_(role), shape_(shape) {
    if (!std::visit([](const auto& value) { return bool(value); }, cache_) ||
        std::any_of(shape_.begin(), shape_.end(), [](auto n) { return n <= 0; }))
        throw std::invalid_argument("invalid Metal expert bank");
    (void)numel();
}

std::int64_t ExpertBank::numel() const {
    std::int64_t result = 1;
    for (auto size : shape_) {
        if (size > std::numeric_limits<std::int64_t>::max() / result)
            throw std::overflow_error("expert bank shape overflow");
        result *= size;
    }
    return result;
}

std::int64_t ExpertBank::size(std::int64_t dimension) const {
    if (dimension < 0) dimension += 3;
    if (dimension < 0 || dimension >= 3) throw std::out_of_range("expert bank dimension");
    return shape_[dimension];
}

void ExpertBank::record_stream(std::uintptr_t, std::shared_ptr<void>) const {
    throw std::runtime_error("Metal tensor stream recording is not supported");
}

} // namespace tyloi::metal

namespace tyloi::weights {

struct ExpertBatch::Impl {
    std::vector<Tensor> gate_up;
    Tensor down;
    Tensor ids;
};

ExpertBatch::ExpertBatch(std::span<const Tensor> gate_up, const Tensor& down, const Tensor& ids)
    : impl_(std::make_unique<Impl>(Impl{{gate_up.begin(), gate_up.end()}, down, ids})) {}
ExpertBatch::~ExpertBatch() = default;
ExpertBatch::ExpertBatch(ExpertBatch&&) noexcept = default;
ExpertBatch& ExpertBatch::operator=(ExpertBatch&&) noexcept = default;

bool cached(const Tensor& bank) { return bank.representation<metal::ExpertBank>() != nullptr; }

} // namespace tyloi::weights

namespace tyloi::ops {
namespace {

mx::array dense_projection(const mx::array& input, const Tensor& bank,
                           const mx::array& ids, bool paired) {
    auto selected = mx::take(metal::array(bank), ids, 0);
    auto matrix = mx::transpose(selected, {0, 1, 3, 2});
    auto source = mx::astype(input, selected.dtype());
    auto rows = paired ? mx::expand_dims(source, 2)
                       : mx::expand_dims(mx::expand_dims(source, 1), 2);
    return mx::squeeze(mx::matmul(rows, matrix), 2);
}

void check_banks(const std::vector<Tensor>& banks, const Tensor& down,
                 std::int64_t input_width, std::int64_t output_width) {
    if ((banks.size() != 1 && banks.size() != 2) || !down || down.dim() != 3)
        throw std::invalid_argument("routed expert banks are missing");
    for (const auto& bank : banks) {
        if (!bank || bank.dim() != 3 || bank.size(0) != down.size(0) ||
            bank.size(2) != input_width || !bank.is_metal())
            throw std::invalid_argument("routed expert bank geometry mismatch");
    }
    if (!down.is_metal() || down.size(1) != output_width || down.size(2) <= 0 ||
        (banks.size() == 1 ? banks[0].size(1) != 2 * down.size(2) :
         banks[0].size(1) != down.size(2) || banks[1].size(1) != down.size(2)))
        throw std::invalid_argument("routed expert bank geometry mismatch");
}

} // namespace

RoutedExperts routed(const Tensor& input, const Tensor& input_f32, const Tensor& route_weights,
                     weights::ExpertBatch& batch, const ExpertSpec& spec) {
    if (!batch.impl_) throw std::invalid_argument("expert batch has been moved");
    const auto& b = *batch.impl_;
    if (!input.is_metal() || input.dim() != 2 || input.scalar_type() != kFloat16 ||
        !input_f32.is_metal() || input_f32.dim() != 2 || input_f32.scalar_type() != kFloat32 ||
        input.size(0) <= 0 || input.size(1) <= 0 || input_f32.size(0) != input.size(0) ||
        input_f32.size(1) != input.size(1) || !b.ids.is_metal() || b.ids.dim() != 2 ||
        b.ids.scalar_type() != kInt32 || b.ids.size(0) != input.size(0) || b.ids.size(1) <= 0 ||
        !route_weights.is_metal() || route_weights.dim() != 2 || route_weights.scalar_type() != kFloat32 ||
        route_weights.size(0) != b.ids.size(0) || route_weights.size(1) != b.ids.size(1) ||
        !spec.shared_gate || spec.shared_gate.dim() != 2 || spec.shared_gate.size(0) != 1 ||
        spec.shared_gate.size(1) != input.size(1) || !std::isfinite(spec.swiglu_limit) || spec.swiglu_limit < 0)
        throw std::invalid_argument("routed expert input geometry mismatch");
    if (bool(spec.latent_in) != bool(spec.latent_out) ||
        (spec.latent_in && (spec.latent_in.dim() != 2 || spec.latent_in.size(1) != input.size(1) ||
         spec.latent_out.dim() != 2 || spec.latent_out.size(0) != input.size(1))))
        throw std::invalid_argument("routed expert latent geometry mismatch");

    auto x = spec.latent_in ? project(input, spec.latent_in) : input;
    const auto output_width = spec.latent_out ? spec.latent_out.size(1) : input.size(1);
    check_banks(b.gate_up, b.down, x.size(1), output_width);
    auto logits = project(input_f32, spec.shared_gate);
    const auto* gate = b.gate_up[0].representation<metal::ExpertBank>();
    const auto* down = b.down.representation<metal::ExpertBank>();
    Tensor result;
    if (gate || down || (b.gate_up.size() == 2 && b.gate_up[1].representation<metal::ExpertBank>())) {
        const auto* up = b.gate_up.size() == 2 ? b.gate_up[1].representation<metal::ExpertBank>() : nullptr;
        if (!gate || !up || !down || gate->role() != metal::ExpertRole::gate ||
            up->role() != metal::ExpertRole::up || down->role() != metal::ExpertRole::down ||
            gate->layer() != up->layer() || gate->layer() != down->layer() ||
            gate->cache() != up->cache() || gate->cache() != down->cache())
            throw std::invalid_argument("incompatible Metal cached expert banks");
        const auto& ids = metal::array(b.ids);
        const auto& weights = metal::array(route_weights);
        auto value = std::visit([&](const auto& cache) -> mx::array {
            using Cache = typename std::decay_t<decltype(cache)>::element_type;
            if constexpr (std::is_same_v<Cache, metal::MlxTqExpertCache>) {
                return cache->apply_ffn(gate->layer(), metal::array(x), ids, weights,
                                        static_cast<float>(spec.swiglu_limit));
            } else {
                if (spec.swiglu_limit > 0) {
                    auto* gguf = dynamic_cast<metal::MlxGgufExpertCache*>(cache.get());
                    if (!gguf) throw std::runtime_error("Metal affine expert cache does not support limited SwiGLU");
                    return gguf->apply_ffn(gate->layer(), metal::array(x), ids, weights, nullptr,
                                           static_cast<float>(spec.swiglu_limit));
                }
                return cache->apply_ffn(gate->layer(), metal::array(x), ids, weights);
            }
        }, gate->cache());
        result = metal::tensor(std::move(value));
    } else {
        if (b.down.is_encoded() || std::any_of(b.gate_up.begin(), b.gate_up.end(),
                                               [](const Tensor& bank) { return bank.is_encoded(); }))
            throw std::invalid_argument("unsupported Metal expert bank representation");
        const auto& ids = metal::array(b.ids);
        auto gate_values = dense_projection(metal::array(x), b.gate_up[0], ids, false);
        auto hidden = b.gate_up.size() == 1
            ? (spec.swiglu_limit > 0 ? metal::moe_limited_swiglu_split(gate_values, spec.swiglu_limit)
                                     : metal::moe_swiglu_split(gate_values))
            : ([&] {
                auto up = dense_projection(metal::array(x), b.gate_up[1], ids, false);
                return spec.swiglu_limit > 0 ? metal::moe_limited_swiglu_pair(gate_values, up, spec.swiglu_limit)
                                             : metal::moe_swiglu_pair(gate_values, up);
            })();
        auto pairs = dense_projection(hidden, b.down, ids, true);
        result = metal::tensor(metal::moe_weighted_reduce(pairs, metal::array(route_weights)));
    }
    if (spec.latent_out) result = project(result, spec.latent_out);
    return {cast(result, kFloat16), cast(logits, kFloat32), true};
}

Tensor combine_experts(const RoutedExperts& experts, const Tensor& route_weights, const Tensor& shared) {
    auto values = experts.reduced ? metal::array(experts.values) :
        metal::moe_weighted_reduce(metal::array(experts.values), metal::array(route_weights));
    return metal::tensor(metal::moe_add_shared_gate(values, metal::array(shared),
                                                   metal::array(experts.shared_logits)));
}

} // namespace tyloi::ops

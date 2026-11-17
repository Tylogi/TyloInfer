#include "engine/backends/metal/ops/moe.h"
#include "engine/backends/metal/native/tensor/tensor.h"
#include "engine/backends/metal/ops/ffn/projection.h"
#include "engine/backends/metal/ops/common/mlx_moe_ops.h"
#include "engine/ops/ops.h"
#include "engine/ops/ffn.h"
#include "engine/weights/moe/batch.h"
#include "engine/weights/moe/experts.h"
#include "backends/metal/runtime/mlx_packed_expert_cache.h"
#include "backends/metal/runtime/mlx_projection_batch.h"
#include "backends/metal/runtime/mlx_tq_expert_cache.h"
#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/tensor/encoding/metal_experts.h"
#include "engine/backends/metal/ops/mfq/mlx_moe.h"
#include "engine/weights/cache/metal_mfe.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
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
    std::int64_t row_offset = 0;
    std::int64_t row_total = 0;
    ops::MlpActivation activation = ops::MlpActivation::silu_gated;
};

ExpertBatch::ExpertBatch(std::span<const Tensor> gate_up, const Tensor& down, const Tensor& ids,
                         std::int64_t row_offset, std::int64_t row_total,
                         ops::MlpActivation activation)
    : impl_(std::make_unique<Impl>(Impl{{gate_up.begin(), gate_up.end()}, down, ids,
                                         row_offset, row_total, activation})) {
    if (row_offset < 0 || row_total <= 0 || row_offset > row_total ||
        ids.size(0) > row_total - row_offset)
        throw std::invalid_argument("invalid packed Metal MoE route row coordinates");
    if (activation == ops::MlpActivation::relu_squared)
        throw std::invalid_argument("Metal routed experts do not support relu_squared activation");
}
ExpertBatch::~ExpertBatch() = default;
ExpertBatch::ExpertBatch(ExpertBatch&&) noexcept = default;
ExpertBatch& ExpertBatch::operator=(ExpertBatch&&) noexcept = default;

bool cached(const Tensor& bank) {
    if (bank.representation<metal::ExpertBank>()) return true;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (const auto* mfe = bank.representation<encoding::MetalExperts>())
        return bool(mfe->cache());
#endif
    return false;
}

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

template<class Work>
Tensor rows(const Tensor& input, std::int64_t count, Work work) {
    if (count <= 0 || input.size(0) <= count) return work(input);
    std::vector<Tensor> output;
    for (std::int64_t start = 0; start < input.size(0); start += count)
        output.push_back(work(slice(input, 0, start,
                                    std::min(start + count, input.size(0)))));
    return cat(output, 0);
}

Tensor shared_ffn(std::span<const Tensor> gate_up, const Tensor& down_weight,
                  const Tensor& input, std::int64_t projection_rows, double limit) {
    if (gate_up.size() != 2 || !gate_up[0] || !gate_up[1] || !down_weight ||
        input.dim() != 2)
        throw std::invalid_argument("Metal shared FFN requires Gate, Up and Down projections");
    auto gate = metal::ffn_projection(gate_up[0]);
    auto up = metal::ffn_projection(gate_up[1]);
    auto down = metal::ffn_projection(down_weight);
    metal::MlxProjectionBatch projections({&gate, &up});
    return rows(input, projection_rows, [&](const Tensor& part) {
        return metal::tensor(down(projections.swiglu(
            metal::array(part), static_cast<float>(limit))));
    });
}

#if TYLOINFER_ENABLE_MFQ_COMPAT
bool paged_mfe(const Tensor& bank) {
    const auto* expert = bank.representation<encoding::MetalExperts>();
    return expert && bool(expert->cache());
}
#endif

bool paged_mfe_or_limited_tq(std::span<const Tensor> gate_up, const Tensor& down) {
    bool paged = false;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    paged = paged_mfe(down) || std::any_of(gate_up.begin(), gate_up.end(), paged_mfe);
#endif
    if (paged) return true;
    const auto* bank = gate_up.empty() ? nullptr : gate_up.front().representation<metal::ExpertBank>();
    if (!bank) return false;
    const auto* cache = std::get_if<std::shared_ptr<metal::MlxTqExpertCache>>(&bank->cache());
    return cache && *cache && (*cache)->limit_bytes() < (*cache)->maximum();
}

#if TYLOINFER_ENABLE_MFQ_COMPAT
struct PagedRoutes {
    mx::array local_ids;
    std::vector<std::int32_t> active;
    std::vector<std::uint32_t> activation_counts;
};

PagedRoutes paged_routes(const Tensor& route_ids, std::int64_t experts) {
    auto ids = mx::contiguous(mx::astype(metal::array(route_ids), mx::int32));
    mx::eval(ids);
    const auto* values = ids.data<std::int32_t>();
    std::vector<std::int32_t> active;
    std::vector<std::uint32_t> counts;
    std::vector<std::int32_t> global_to_local(
        static_cast<std::size_t>(experts) + 1, -1);
    std::vector<std::int32_t> local_ids(ids.size(), -1);
    active.reserve(ids.size());
    counts.reserve(ids.size());
    for (std::size_t index = 0; index < ids.size(); ++index) {
        const auto expert = values[index];
        if (expert == -1) continue;
        if (expert < -1 || static_cast<std::int64_t>(expert) >= experts)
            throw std::invalid_argument("paged MFE route expert ID is out of range");
        auto& local = global_to_local[static_cast<std::size_t>(expert) + 1];
        if (local < 0) {
            local = static_cast<std::int32_t>(active.size());
            active.push_back(expert);
            counts.push_back(0);
        }
        if (counts[static_cast<std::size_t>(local)] == std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("paged MFE route count overflow");
        ++counts[static_cast<std::size_t>(local)];
        local_ids[index] = local;
    }
    if (active.empty())
        throw std::invalid_argument("paged MFE routing selected no experts");
    return {mx::array(local_ids.begin(), ids.shape(), mx::int32),
            std::move(active), std::move(counts)};
}
#endif

} // namespace

RoutedExperts Ffn::routed(const Tensor& input, const Tensor& route_weights,
                          weights::ExpertBatch& batch, const ExpertSpec& spec) {
    if (!batch.impl_) throw std::invalid_argument("expert batch has been moved");
    const auto& b = *batch.impl_;
    if (!input.is_metal() || input.dim() != 2 ||
        (input.scalar_type() != kFloat16 && input.scalar_type() != kBFloat16 && input.scalar_type() != kFloat32) ||
        input.size(0) <= 0 || input.size(1) <= 0 || !b.ids.is_metal() || b.ids.dim() != 2 ||
        b.ids.scalar_type() != kInt32 || b.ids.size(0) != input.size(0) || b.ids.size(1) <= 0 ||
        !route_weights.is_metal() || route_weights.dim() != 2 || route_weights.scalar_type() != kFloat32 ||
        route_weights.size(0) != b.ids.size(0) || route_weights.size(1) != b.ids.size(1) ||
        !std::isfinite(spec.swiglu_limit) || spec.swiglu_limit < 0)
        throw std::invalid_argument("routed expert input geometry mismatch");
    if (spec.activation != MlpActivation::silu_gated)
        throw std::invalid_argument("Metal routed experts support only silu_gated activation");
    check_banks(b.gate_up, b.down, input.size(1), input.size(1));
    const auto* gate = b.gate_up[0].representation<metal::ExpertBank>();
    const auto* down = b.down.representation<metal::ExpertBank>();
    Tensor result;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const auto* resident_gate = b.gate_up[0].representation<encoding::MetalExperts>();
    const auto* resident_up = b.gate_up.size() == 2
        ? b.gate_up[1].representation<encoding::MetalExperts>() : nullptr;
    const auto* resident_down = b.down.representation<encoding::MetalExperts>();
    if (resident_gate || resident_up || resident_down) {
        if (!resident_gate || !resident_down || (b.gate_up.size() == 2 && !resident_up))
            throw std::invalid_argument("incompatible resident Metal expert banks");
        const auto paged = bool(resident_gate->cache()) || bool(resident_down->cache()) ||
            (resident_up && bool(resident_up->cache()));
        if (paged) {
            if (!resident_gate->cache() || !resident_down->cache() ||
                (resident_up && !resident_up->cache()) ||
                resident_gate->cache() != resident_down->cache() ||
                (resident_up && resident_gate->cache() != resident_up->cache()))
                throw std::invalid_argument("incompatible paged Metal expert banks");
            auto& cache = *resident_gate->cache();
            const auto routes = paged_routes(b.ids, b.gate_up[0].size(0));
            const auto& source = metal::array(input);
            const auto limit = static_cast<float>(spec.swiglu_limit);
            const auto hidden = [&]() {
                if (resident_up) {
                    const std::array<std::string, 2> names{
                        resident_gate->record_name(), resident_up->record_name()};
                    auto weights = cache.grouped_mfe(names, routes.active, routes.activation_counts);
                    if (weights.size() != 2)
                        throw std::logic_error("paged MFE Gate/Up grouping returned an invalid count");
                    auto result = weights[0].routed_swiglu_pair(
                        weights[1], source, routes.local_ids, limit);
                    cache.retain_until_drain(
                        std::vector<metal::MlxMfeWeight>{std::move(weights[0]), std::move(weights[1])},
                        {result});
                    return result;
                }
                auto weight = cache.grouped_mfe(
                    resident_gate->record_name(), routes.active, routes.activation_counts);
                auto result = weight.routed_swiglu(source, routes.local_ids, limit);
                cache.retain_until_drain({std::move(weight)}, {result});
                return result;
            }();
            auto down = cache.grouped_mfe(
                resident_down->record_name(), routes.active, routes.activation_counts);
            auto value = down.routed_matmul_reduce(
                hidden, routes.local_ids, metal::array(route_weights));
            cache.retain_until_drain({std::move(down)}, {value});
            result = metal::tensor(std::move(value));
        } else {
            const auto& ids = metal::array(b.ids);
            const auto& source = metal::array(input);
            const auto limit = static_cast<float>(spec.swiglu_limit);
            auto hidden = resident_up
                ? resident_gate->weight().routed_swiglu_pair(resident_up->weight(), source, ids, limit)
                : resident_gate->weight().routed_swiglu(source, ids, limit);
            result = metal::tensor(resident_down->weight().routed_matmul_reduce(
                hidden, ids, metal::array(route_weights)));
        }
    } else
#endif
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
                return cache->apply_ffn(gate->layer(), metal::array(input), ids, weights,
                                        static_cast<float>(spec.swiglu_limit), false);
            } else {
                return cache->apply_ffn(gate->layer(), metal::array(input), ids, weights, nullptr,
                                        static_cast<float>(spec.swiglu_limit));
            }
        }, gate->cache());
        result = metal::tensor(std::move(value));
    } else {
        if (b.down.is_encoded() || std::any_of(b.gate_up.begin(), b.gate_up.end(),
                                               [](const Tensor& bank) { return bank.is_encoded(); }))
            throw std::invalid_argument("unsupported Metal expert bank representation");
        const auto& ids = metal::array(b.ids);
        auto gate_values = dense_projection(metal::array(input), b.gate_up[0], ids, false);
        auto hidden = b.gate_up.size() == 1
            ? (spec.swiglu_limit > 0 ? metal::moe_limited_swiglu_split(gate_values, spec.swiglu_limit)
                                     : metal::moe_swiglu_split(gate_values))
            : ([&] {
                auto up = dense_projection(metal::array(input), b.gate_up[1], ids, false);
                return spec.swiglu_limit > 0 ? metal::moe_limited_swiglu_pair(gate_values, up, spec.swiglu_limit)
                                             : metal::moe_swiglu_pair(gate_values, up);
            })();
        auto pairs = dense_projection(hidden, b.down, ids, true);
        result = metal::tensor(metal::moe_weighted_reduce(pairs, metal::array(route_weights)));
    }
    return {cast(result, input.scalar_type()), true};
}

Tensor Ffn::combine(const RoutedExperts& experts, const Tensor& route_weights,
                    const Tensor& shared, const Tensor& shared_logits) {
    auto values = experts.reduced ? metal::array(experts.values) :
        metal::moe_weighted_reduce(metal::array(experts.values), metal::array(route_weights));
    if (!shared) return metal::tensor(std::move(values));
    if (!shared_logits)
        return metal::tensor(values + metal::array(shared));
    if (shared.dim() != 2 || shared.size(0) != experts.values.size(0) ||
        shared.size(1) != experts.values.size(-1) || shared_logits.dim() != 2 ||
        shared_logits.size(0) != shared.size(0) || shared_logits.size(1) != 1 ||
        shared_logits.scalar_type() != kFloat32)
        throw std::invalid_argument("Metal shared expert combination geometry mismatch");
    return metal::tensor(metal::moe_add_shared_gate(values, metal::array(shared),
                                                    metal::array(shared_logits)));
}

Tensor Ffn::routed_shared(const Tensor& input, const Tensor& route_weights,
                          weights::ExpertBatch& batch, std::span<const Tensor> shared_gate_up,
                          const Tensor& shared_down, const Tensor& gate_input,
                          const Tensor& gate_weight, const ExpertSpec& spec,
                          std::int64_t projection_rows) {
    if (!batch.impl_ || !input || !gate_input || !gate_weight || input.dim() != 2 ||
        gate_input.dim() != 2 || gate_input.size(0) != input.size(0) ||
        gate_input.size(1) != input.size(1) || shared_gate_up.size() != 2 ||
        !shared_gate_up[0] || !shared_gate_up[1] || !shared_down || gate_weight.dim() != 2 ||
        gate_weight.size(0) != 1 || gate_weight.size(1) != input.size(1) ||
        spec.activation != MlpActivation::silu_gated)
        throw std::invalid_argument("invalid Metal routed shared FFN arguments");
    const auto& b = *batch.impl_;
    const auto cache_rows = paged_mfe_or_limited_tq(b.gate_up, b.down) ? 1 : 0;
    auto run = [&](const Tensor& part, const Tensor& part_gate_input,
                   weights::ExpertBatch& part_batch, const Tensor& part_weights) {
        auto shared = shared_ffn(shared_gate_up, shared_down, part, projection_rows,
                                 spec.swiglu_limit);
        auto logits = Ffn::project(part_gate_input, gate_weight, projection_rows);
        auto experts = Ffn::routed(part, part_weights, part_batch, spec);
        return Ffn::combine(experts, part_weights, shared, cast(logits, kFloat32));
    };
    if (cache_rows == 0 || input.size(0) <= 1)
        return run(input, gate_input, batch, route_weights);

    std::vector<Tensor> output;
    output.reserve(static_cast<std::size_t>((input.size(0) + cache_rows - 1) / cache_rows));
    for (std::int64_t start = 0; start < input.size(0); start += cache_rows) {
        const auto end = std::min(start + cache_rows, input.size(0));
        auto part_input = slice(input, 0, start, end);
        auto part_gate = slice(gate_input, 0, start, end);
        auto part_weights = slice(route_weights, 0, start, end);
        auto part_ids = slice(b.ids, 0, start, end);
        weights::ExpertBatch part_batch(
            b.gate_up, b.down, part_ids, b.row_offset + start, b.row_total,
            spec.activation);
        output.push_back(run(part_input, part_gate, part_batch, part_weights));
    }
    return cat(output, 0);
}

} // namespace tyloi::ops

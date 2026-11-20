#include "batch_internal.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/weights/encoding/format_features.h"
#include "engine/weights/moe/ggml_bank.h"
#include "engine/weights/moe/experts.h"
#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/weights/moe/residency/cached_source_internal.h"
#endif

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <utility>

namespace tyloi::weights {
namespace {

std::shared_ptr<const MfeTensor> mfe(const Tensor& weight) {
    return std::dynamic_pointer_cast<const MfeTensor>(weight.implementation());
}

std::shared_ptr<const MfeTensor> cpu_expert(const std::shared_ptr<const MfeTensor>& weight) {
    return weight && !weight->cached_source && weight->device().is_cpu() ? weight : nullptr;
}

} // namespace

ExpertBatch::ExpertBatch(std::span<const Tensor> banks, const Tensor& down, const Tensor& ids,
                         std::int64_t row_offset, std::int64_t row_total,
                         ops::MlpActivation activation, bool prefill)
    : impl_(std::make_unique<Impl>()) {
    if ((banks.size() != 1 && banks.size() != 2) || !down || !ids ||
        ids.dim() != 2 || !ids.representation<DenseTensor>() || !ids.is_cuda() || ids.scalar_type() != kInt32 ||
        !ids.is_contiguous() || ids.size(0) <= 0 || ids.size(1) <= 0 ||
        ids.size(0) > INT_MAX / ids.size(1))
        throw std::invalid_argument("invalid selected expert banks or route IDs");
    if (row_offset < 0 || row_total <= 0 || row_offset > row_total ||
        ids.size(0) > row_total - row_offset)
        throw std::invalid_argument("invalid packed MoE route row coordinates");
    for (const auto& bank : banks)
        if (!bank || bank.dim() != 3 || bank.size(0) <= 0 || bank.size(0) > INT_MAX ||
            bank.size(1) <= 0 || bank.size(2) <= 0)
            throw std::invalid_argument("invalid selected expert bank");
    const bool ungated = activation == ops::MlpActivation::relu_squared && banks.size() == 1;
    if (down.dim() != 3 || down.size(0) != banks[0].size(0) ||
        down.size(1) <= 0 || down.size(2) <= 0 ||
        (banks.size() == 2 && (banks[1].size(0) != banks[0].size(0) ||
                               banks[1].size(1) != banks[0].size(1) ||
                               banks[1].size(2) != banks[0].size(2))) ||
        (banks.size() == 1 ? (ungated ? banks[0].size(1) != down.size(2)
                                      : (banks[0].size(1) % 2 != 0 || banks[0].size(1) / 2 != down.size(2)))
                           : banks[0].size(1) != down.size(2)))
        throw std::invalid_argument("incompatible selected expert bank shapes");

    auto& b = *impl_;
    b.gate_up_weights.assign(banks.begin(), banks.end());
    b.down_weight = down;
    b.ids = ids;
    b.activation = activation;
    b.ungated = ungated;
    b.split = banks.size() == 2;
    if (b.split) {
        b.gate = mfe(banks[0]);
        b.up = mfe(banks[1]);
        b.dense_gate = std::dynamic_pointer_cast<const DenseTensor>(banks[0].implementation());
        b.dense_up = std::dynamic_pointer_cast<const DenseTensor>(banks[1].implementation());
        b.tq = tq_banks(banks[0], banks[1], down);
    } else {
        if (ungated) {
            b.up = mfe(banks[0]);
            b.dense_up = std::dynamic_pointer_cast<const DenseTensor>(banks[0].implementation());
            b.tq = tq_banks({}, banks[0], down);
        } else {
            b.gate_up = mfe(banks[0]);
            b.dense_gate_up = std::dynamic_pointer_cast<const DenseTensor>(banks[0].implementation());
            b.tq = tq_banks({}, {}, down);
        }
    }
    b.down = mfe(down);
    b.dense_down = std::dynamic_pointer_cast<const DenseTensor>(down.implementation());
    b.cpu_gate = cpu_expert(b.gate);
    b.cpu_up = cpu_expert(b.up);
    b.cpu_gate_up = cpu_expert(b.gate_up);
    b.cpu_down = cpu_expert(b.down);
    b.sharded = !b.split && expert_parallel(banks[0], down);
    b.ggml = down.representation<GgmlExpertBank>() != nullptr;
    if (b.ggml && row_total > INT_MAX)
        throw std::invalid_argument("GGML route sample row count exceeds the route history limit");
    for (const auto& bank : banks)
        if ((bank.representation<GgmlExpertBank>() != nullptr) != b.ggml ||
            (b.ggml && (bank.device() != ids.device() || down.device() != ids.device())))
            throw std::invalid_argument("GGUF routed banks must share an encoding family and route device");
    const bool tq_routed = bool(b.tq.gate || b.tq.up || b.tq.down);
    if (!b.ggml && ((tq_routed && (b.ungated ? (b.split || b.tq.gate || !b.tq.up || !b.tq.down)
                                  : (!b.split || !b.tq.gate || !b.tq.up || !b.tq.down))) ||
        (!tq_routed && b.split && !((b.gate && b.up && b.down) ||
                                    (b.dense_gate && b.dense_up && b.dense_down))) ||
        (!tq_routed && !b.split &&
            (b.ungated ? !((b.up && b.down) || (b.dense_up && b.dense_down))
                       : !((b.gate_up && b.down) || (b.dense_gate_up && b.dense_down))))))
        throw std::invalid_argument("unsupported selected expert bank representations");

#if TYLOINFER_ENABLE_MFQ_COMPAT
    for (const auto& weight : {b.gate, b.up, b.gate_up, b.down}) {
        if (!weight) continue;
        if (!cpu_expert(weight)) {
            const auto placement = expert_placement(*weight);
            if (weight->shards.empty()) check_placement(placement, ids.device());
        }
    }
#endif
    for (const auto& weight : {b.dense_gate, b.dense_up, b.dense_gate_up, b.dense_down}) {
        if (weight && (!weight->is_contiguous() || weight->device() != ids.device() ||
                       (weight->scalar_type() != kFloat16 && weight->scalar_type() != kBFloat16 &&
                        weight->scalar_type() != kFloat32)))
            throw std::invalid_argument("dense expert banks must be contiguous CUDA F16/BF16/F32 on the route device");
    }
    MfqCudaGuard guard(ids.device().index);
    auto& executor = cuda::Executor::current();
    if (!tq_routed && !b.ggml) {
        b.route = executor.profiler().measure("moe.route_map", [&] {
            return build_moe_route_plan(ids.dense(), static_cast<int>(banks[0].size(0)),
                prefill && bool(b.down && b.down->cached_source));
        });
    }
    b.route.row_offset = row_offset;
    b.route.row_total = row_total;
    b.route.prefill = prefill;
    if (!cuda::Executor::current_ffn().batch_serial && !b.ggml) b.prepare();
}

ExpertBatch::~ExpertBatch() = default;
ExpertBatch::ExpertBatch(ExpertBatch&&) noexcept = default;
ExpertBatch& ExpertBatch::operator=(ExpertBatch&&) noexcept = default;

void ExpertBatch::Impl::prepare() {
    if (prepared) return;
    if (ggml) {
        const auto resident = [](const Tensor& weight) {
            return weight.representation<GgmlExpertBank>()->resident_addresses.defined();
        };
        const bool device_resident = ids.size(0) <= 32 && resident(down_weight) &&
            std::all_of(gate_up_weights.begin(), gate_up_weights.end(), resident);
        ggml_rows = group_expert_rows(ids.dense(), down_weight.size(0), device_resident);
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const bool tq_routed = bool(tq.gate || tq.up || tq.down);
    if (!sharded && !tq_routed) {
        if (down && down->cached_source) down->cached_source->begin_routes(route);
        const auto& first = split ? gate : (ungated ? up : gate_up);
        if (cuda::Executor::current().options().ffn.delayed_readback) {
            if (first) cuda::prefetch_experts(*first, route, true);
        } else {
            bundle_prefetched = prefetch_bundle();
            if (!bundle_prefetched && first) cuda::prefetch_experts(*first, route);
        }
    }
#endif
    prepared = true;
}

bool ExpertBatch::Impl::prefetch_bundle() {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const auto& options = cuda::Executor::current().options().ffn;
    if (!options.bundle_prefetch || cpu_down || cuda::Executor::current_ffn().batch_serial)
        return false;
    if (split) {
        if (cpu_gate || cpu_up || !gate || !up || !down) return false;
        return cuda::prefetch_cached_moe_projection_bundle(*gate, *up, *down, route);
    }
    if (ungated) {
        if (cpu_up || !up || !down) return false;
        return cuda::prefetch_cached_moe_projection_bundle(*up, *down, route);
    }
    if (cpu_gate_up || !gate_up || !down) return false;
    return cuda::prefetch_cached_moe_projection_bundle(*gate_up, *down, route);
#else
    return false;
#endif
}

void ExpertBatch::Impl::ready() {
    if (readied) return;
    prepare();
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const bool tq_routed = bool(tq.gate || tq.up || tq.down);
    if (!sharded && !tq_routed && cuda::Executor::current().options().ffn.delayed_readback) {
        bundle_prefetched = prefetch_bundle();
        const auto& first = split ? gate : (ungated ? up : gate_up);
        if (!bundle_prefetched && first) cuda::prefetch_experts(*first, route);
    }
#endif
    readied = true;
}

void ExpertBatch::Impl::retire() {
    if (ggml_rows && ggml_rows->host_ids.defined()) {
        const auto* output = down_weight.representation<GgmlExpertBank>();
        const GgmlExpertBank* banks[] = {gate_up_weights.front().representation<GgmlExpertBank>(),
            split ? gate_up_weights.back().representation<GgmlExpertBank>() : output, output};
        observe_ggml_routes(std::span{banks}.first(split ? 3 : 2), ggml_rows->host_ids,
            static_cast<int>(route.row_offset), static_cast<int>(route.row_total), route.prefill);
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!sharded && !tq.gate && !tq.up && !tq.down && down && down->cached_source)
        down->cached_source->finish_routes(route);
#endif
}

std::shared_ptr<const MfeTensor> ExpertBatch::Impl::stage(
        const std::shared_ptr<const MfeTensor>& source, const char* label) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    auto& executor = cuda::Executor::current();
    return executor.profiler().measure(label, [&] {
        return stage_cpu_mixed_moe(source, executor.options().projection.moe.heterogeneous);
    });
#else
    (void)source;
    (void)label;
    throw std::runtime_error("MFE CPU staging requires a compatibility build");
#endif
}

} // namespace tyloi::weights

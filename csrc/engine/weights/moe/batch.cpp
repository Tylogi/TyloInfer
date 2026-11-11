#include "batch_internal.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/weights/encoding/format_features.h"
#include "engine/weights/moe/experts.h"

#include <climits>
#include <stdexcept>
#include <utility>

#ifdef TYLOINFER_NATIVE_CUDA
namespace tyloi::weights {
namespace {

std::shared_ptr<const MfeTensor> mfe(const Tensor& weight) {
    return std::dynamic_pointer_cast<const MfeTensor>(weight.implementation());
}

std::shared_ptr<const MfeTensor> cpu_expert(const std::shared_ptr<const MfeTensor>& weight) {
    return weight && !weight->cached_source && weight->device().is_cpu() ? weight : nullptr;
}

} // namespace

ExpertBatch::ExpertBatch(std::span<const Tensor> banks, const Tensor& down, const Tensor& ids)
    : impl_(std::make_unique<Impl>()) {
    if ((banks.size() != 1 && banks.size() != 2) || !down || !ids ||
        ids.dim() != 2 || !ids.representation<DenseTensor>() || !ids.is_cuda() || ids.scalar_type() != kInt32 ||
        !ids.is_contiguous() || ids.size(0) <= 0 || ids.size(1) <= 0 ||
        ids.size(0) > INT_MAX / ids.size(1))
        throw std::invalid_argument("invalid selected expert banks or route IDs");
    for (const auto& bank : banks)
        if (!bank || bank.dim() != 3 || bank.size(0) <= 0 || bank.size(0) > INT_MAX ||
            bank.size(1) <= 0 || bank.size(2) <= 0)
            throw std::invalid_argument("invalid selected expert bank");
    if (down.dim() != 3 || down.size(0) != banks[0].size(0) ||
        down.size(1) <= 0 || down.size(2) <= 0 ||
        (banks.size() == 2 && (banks[1].size(0) != banks[0].size(0) ||
                               banks[1].size(1) != banks[0].size(1) ||
                               banks[1].size(2) != banks[0].size(2))) ||
        (banks.size() == 1 ? (banks[0].size(1) % 2 != 0 || banks[0].size(1) / 2 != down.size(2))
                           : banks[0].size(1) != down.size(2)))
        throw std::invalid_argument("incompatible selected expert bank shapes");

    auto& b = *impl_;
    b.gate_up_weights.assign(banks.begin(), banks.end());
    b.down_weight = down;
    b.ids = ids;
    b.split = banks.size() == 2;
    if (b.split) {
        b.gate = mfe(banks[0]);
        b.up = mfe(banks[1]);
        b.dense_gate = std::dynamic_pointer_cast<const DenseTensor>(banks[0].implementation());
        b.dense_up = std::dynamic_pointer_cast<const DenseTensor>(banks[1].implementation());
        b.tq = tq_banks(banks[0], banks[1], down);
    } else {
        b.gate_up = mfe(banks[0]);
        b.tq = tq_banks({}, {}, down);
    }
    b.down = mfe(down);
    b.dense_down = std::dynamic_pointer_cast<const DenseTensor>(down.implementation());
    b.cpu_gate = cpu_expert(b.gate);
    b.cpu_up = cpu_expert(b.up);
    b.cpu_gate_up = cpu_expert(b.gate_up);
    b.cpu_down = cpu_expert(b.down);
    b.sharded = !b.split && expert_parallel(banks[0], down);
    const bool tq_routed = bool(b.tq.gate || b.tq.up || b.tq.down);
    if ((tq_routed && (!b.split || !b.tq.gate || !b.tq.up || !b.tq.down)) ||
        (!tq_routed && b.split && !((b.gate && b.up && b.down) ||
                                    (b.dense_gate && b.dense_up && b.dense_down))) ||
        (!tq_routed && !b.split && !(b.gate_up && b.down)))
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
    for (const auto& weight : {b.dense_gate, b.dense_up, b.dense_down}) {
        if (weight && (!weight->is_contiguous() || weight->device() != ids.device() ||
                       (weight->scalar_type() != kFloat16 && weight->scalar_type() != kBFloat16 &&
                        weight->scalar_type() != kFloat32)))
            throw std::invalid_argument("dense expert banks must be contiguous CUDA F16/BF16/F32 on the route device");
    }
    MfqCudaGuard guard(ids.device().index);
    auto& executor = cuda::Executor::current();
    if (!tq_routed) {
        b.route = executor.profiler().measure("moe.route_map", [&] {
            return build_moe_route_plan(ids.dense(), static_cast<int>(banks[0].size(0)));
        });
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!b.sharded && !tq_routed) {
        const auto& first = b.split ? b.gate : b.gate_up;
        if (executor.options().ffn.delayed_readback) {
            if (first) cuda::prefetch_experts(*first, b.route, true);
        } else {
            b.bundle_prefetched = b.prefetch_bundle();
            if (!b.bundle_prefetched && first) cuda::prefetch_experts(*first, b.route);
        }
    }
#endif
}

ExpertBatch::~ExpertBatch() = default;
ExpertBatch::ExpertBatch(ExpertBatch&&) noexcept = default;
ExpertBatch& ExpertBatch::operator=(ExpertBatch&&) noexcept = default;

bool ExpertBatch::Impl::prefetch_bundle() {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const auto& options = cuda::Executor::current().options().ffn;
    if (!options.bundle_prefetch || cpu_down || cuda::Executor::current_ffn().batch_serial)
        return false;
    if (split) {
        if (cpu_gate || cpu_up || !gate || !up || !down) return false;
        return cuda::prefetch_cached_moe_projection_bundle(*gate, *up, *down, route);
    }
    if (cpu_gate_up || !gate_up || !down) return false;
    return cuda::prefetch_cached_moe_projection_bundle(*gate_up, *down, route);
#else
    return false;
#endif
}

void ExpertBatch::Impl::ready() {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const bool tq_routed = bool(tq.gate || tq.up || tq.down);
    if (!sharded && !tq_routed && cuda::Executor::current().options().ffn.delayed_readback) {
        bundle_prefetched = prefetch_bundle();
        const auto& first = split ? gate : gate_up;
        if (!bundle_prefetched && first) cuda::prefetch_experts(*first, route);
    }
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
#endif

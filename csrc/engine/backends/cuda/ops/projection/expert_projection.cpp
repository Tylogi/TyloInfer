#include "projection.h"
#include "engine/weights/encoding/mfe.h"
#include "engine/backends/cuda/ops/ffn/tq/banks.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/ops/ffn/experts/moe.h"
#include "engine/ops/ops.h"

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/backends/cuda/ops/ffn/mfe/projection.h"
#include "engine/backends/cuda/ops/projection/cpu_rows.h"
#include "engine/backends/cpu/executor/expert_pool.h"
#include "engine/weights/format.h"
#include "engine/weights/moe/experts.h"
#include "engine/weights/moe/prepare/projection.h"
#endif

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <utility>

namespace tyloi::cuda {

DenseTensor Projector::run(const ::tyloi::Tensor& weight, DenseTensor input,
                           const MoeRoutePlan& route) const {
    const auto* mfe = weight.representation<MfeTensor>();
    if (!mfe) throw std::invalid_argument("unsupported expert tensor layout");
    return run(*mfe, std::move(input), route);
}

#if TYLOINFER_ENABLE_MFQ_COMPAT
namespace {

std::shared_ptr<const void> mixed_source(const MfeTensor& weight) {
    if (weight.nint_dispatch) return weight.nint_dispatch;
    if (weight.nvq_dispatch) return weight.nvq_dispatch;
    if (!weight.pools.empty() && weight.pools.front().expert_local.storage())
        return weight.pools.front().expert_local.storage();
    throw std::runtime_error("mixed MFE has no backing identity");
}

void validate_projection_shape(const MfeTensor& weight, const DenseTensor& input,
                               const MoeRoutePlan& route, const char* message) {
    if (route.ids.dim() != 2 || route.n_experts != weight.n_experts ||
        (input.dim() != 2 && input.dim() != 3) || input.size(0) != route.ids.size(0) ||
        input.size(-1) != weight.neuron_len ||
        (input.dim() == 3 && input.size(1) != route.ids.size(1)))
        throw std::runtime_error(message);
}

DenseTensor cpu_project_groups(const MfeTensor& weight, DenseTensor input,
                               const MoeRoutePlan& route,
                               ::tyloi::weights::PreparedExperts& prepared,
                               const std::vector<::tyloi::MoeDispatchGroup>& groups) {
    validate_projection_shape(weight, input, route,
                              "CPU cold expert input/route shape mismatch");
    const auto tokens = static_cast<int>(route.ids.size(0));
    const auto routes = static_cast<int>(route.ids.size(1));
    const auto total = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(routes);

    const auto device = input.device();
    auto x = input.to(::tyloi::kCPU).to(::tyloi::kFloat32).contiguous();
    const int width = weight.neuron_len, output = weight.out_per_expert;
    auto result = ::tyloi::cuda::empty({tokens, routes, output},
        ::tyloi::TensorOptions().device(::tyloi::kCPU).dtype(::tyloi::kFloat16));
    struct Job {
        MoeHostExpertCache::Lease lease;
        CpuProjectionRows projection;
        std::vector<std::size_t> positions;
        int offset = 0;
        int count = 0;
    };
    std::vector<Job> jobs;
    auto packed = ::tyloi::cuda::empty({static_cast<std::int64_t>(total), width},
        ::tyloi::TensorOptions().device(::tyloi::kCPU).dtype(::tyloi::kFloat32));
    auto values = ::tyloi::cuda::empty({static_cast<std::int64_t>(total), output},
        ::tyloi::TensorOptions().device(::tyloi::kCPU).dtype(::tyloi::kFloat32));
    int offset = 0;
    for (const auto& group : groups) {
        if (group.positions.empty()) continue;
        const auto& lease = prepared.cpu.at(group.expert);
        if (!lease)
            throw std::runtime_error("prepared CPU expert lease is missing");
        const auto& pool = lease->weights;
        auto projection = pool.family == MixedMoeFamily::Nint
            ? make_cpu_projection_rows(pool.nint)
            : pool.family == MixedMoeFamily::Mxfp4
                ? make_cpu_projection_rows(pool.mxfp4)
                : make_cpu_projection_rows(pool.nvq);
        for (std::size_t row = 0; row < group.positions.size(); ++row) {
            const auto source = input.dim() == 2
                ? group.positions[row] / static_cast<std::size_t>(routes)
                : group.positions[row];
            std::memcpy(
                static_cast<std::byte*>(packed.data_ptr()) +
                    static_cast<std::size_t>(offset + row) * width * sizeof(float),
                static_cast<const std::byte*>(x.data_ptr()) + source * width * sizeof(float),
                static_cast<std::size_t>(width) * sizeof(float));
        }
        jobs.push_back({lease, std::move(projection), group.positions,
            offset, static_cast<int>(group.positions.size())});
        offset += static_cast<int>(group.positions.size());
    }
    ::tyloi::cpu::expert_pool().rows(static_cast<std::int64_t>(jobs.size()) * output,
        [&](std::int64_t begin, std::int64_t end) {
            while (begin < end) {
                const auto index = begin / output;
                const auto first = static_cast<int>(begin % output);
                const auto last = static_cast<int>(std::min<std::int64_t>(
                    output, first + end - begin));
                const auto& job = jobs[static_cast<std::size_t>(index)];
                job.projection.run(
                    packed.data_ptr<float>() + static_cast<std::size_t>(job.offset) * width,
                    job.count, width,
                    values.data_ptr<float>() + static_cast<std::size_t>(job.offset) * output,
                    output, first, last);
                begin += last - first;
            }
        });
    for (const auto& job : jobs) {
        for (int row = 0; row < job.count; ++row) {
            auto* destination = result.data_ptr<mfq_half>() +
                job.positions[static_cast<std::size_t>(row)] * output;
            const auto* source = values.data_ptr<float>() +
                static_cast<std::size_t>(job.offset + row) * output;
            for (int column = 0; column < output; ++column)
                destination[column] = mfq_half(source[column]);
        }
    }
    prepared.record_cpu(jobs.size());
    return result.to(device);
}

struct DispatchSubset {
    std::vector<std::int32_t> ids;
    std::vector<std::int64_t> positions;
    std::vector<::tyloi::MoeDispatchGroup> groups;
};

DispatchSubset dispatch_subset(const ::tyloi::weights::PreparedExperts& prepared,
                               bool cpu_only) {
    DispatchSubset subset;
    for (const auto& group : prepared.dispatch.groups) {
        if (cpu_only != (group.kind == ::tyloi::MoeDispatchKind::Cpu))
            continue;
        ::tyloi::MoeDispatchGroup compact{group.expert, group.kind, {}};
        compact.positions.reserve(group.positions.size());
        for (const auto position : group.positions) {
            compact.positions.push_back(subset.ids.size());
            subset.ids.push_back(group.expert);
            subset.positions.push_back(static_cast<std::int64_t>(position));
        }
        subset.groups.push_back(std::move(compact));
    }
    return subset;
}

DenseTensor hybrid_project(const MfeTensor& weight, DenseTensor input,
                           const MoeRoutePlan& route,
                           ::tyloi::weights::PreparedExperts& prepared,
                           const MoeOptions& options, KlMmqState& mmq,
                           MoeScratch& scratch) {
    validate_projection_shape(weight, input, route,
                              "MFE prepared route/input shape mismatch");
    const auto tokens = route.ids.size(0);
    const auto routes = route.ids.size(1);
    const auto cpu = dispatch_subset(prepared, true);
    const auto gpu = dispatch_subset(prepared, false);

    auto host_x = input.to(::tyloi::kCPU).contiguous();
    DenseTensor cpu_x, gpu_y, cpu_y;
    if (!gpu.ids.empty()) {
        const MfqCudaGuard guard(prepared.device);
        DenseTensor gpu_x, gpu_route;
        if (tokens == 1) {
            if (input.dim() == 3)
                gpu_x = input.index_select(1, ::tyloi::cuda::tensor(gpu.positions).to(input.device()));
            else
                gpu_x = input;
            gpu_route = ::tyloi::cuda::tensor(gpu.ids).reshape({1,
                static_cast<std::int64_t>(gpu.ids.size())}).to(input.device());
        } else {
            std::vector<std::int64_t> input_positions;
            input_positions.reserve(gpu.positions.size());
            for (const auto position : gpu.positions)
                input_positions.push_back(input.dim() == 2
                    ? position / routes : position);
            gpu_x = input.reshape({-1, weight.neuron_len}).index_select(0,
                ::tyloi::cuda::tensor(input_positions).to(input.device()));
            gpu_route = ::tyloi::cuda::tensor(gpu.ids).reshape({
                static_cast<std::int64_t>(gpu.ids.size()), 1}).to(input.device());
        }
        auto gpu_plan = build_moe_route_plan(gpu_route, weight.n_experts);
        gpu_y = mixed_project(*prepared.weight, scratch, options, mmq,
            gpu_x, gpu_plan, false);
        prepared.retire();
    }

    if (!cpu.ids.empty()) {
        const auto width = static_cast<std::size_t>(weight.neuron_len);
        const auto row_bytes = width * host_x.element_size();
        if (tokens == 1 && input.dim() == 3) {
            cpu_x = ::tyloi::cuda::empty({1, static_cast<std::int64_t>(cpu.ids.size()),
                weight.neuron_len}, host_x.options());
            for (std::size_t row = 0; row < cpu.positions.size(); ++row)
                std::memcpy(static_cast<std::byte*>(cpu_x.data_ptr()) + row * row_bytes,
                    static_cast<const std::byte*>(host_x.data_ptr()) +
                        static_cast<std::size_t>(cpu.positions[row]) * row_bytes, row_bytes);
        } else if (tokens == 1) {
            cpu_x = host_x;
        } else {
            cpu_x = ::tyloi::cuda::empty({static_cast<std::int64_t>(cpu.ids.size()),
                weight.neuron_len}, host_x.options());
            for (std::size_t row = 0; row < cpu.positions.size(); ++row) {
                const auto source = input.dim() == 2
                    ? static_cast<std::size_t>(cpu.positions[row]) / static_cast<std::size_t>(routes)
                    : static_cast<std::size_t>(cpu.positions[row]);
                std::memcpy(static_cast<std::byte*>(cpu_x.data_ptr()) + row * row_bytes,
                    static_cast<const std::byte*>(host_x.data_ptr()) + source * row_bytes,
                    row_bytes);
            }
        }
        MoeRoutePlan cpu_plan;
        cpu_plan.ids = ::tyloi::cuda::tensor(cpu.ids);
        cpu_plan.ids = tokens == 1
            ? cpu_plan.ids.reshape({1, static_cast<std::int64_t>(cpu.ids.size())})
            : cpu_plan.ids.reshape({static_cast<std::int64_t>(cpu.ids.size()), 1});
        cpu_plan.n_experts = weight.n_experts;
        cpu_y = cpu_project_groups(weight, std::move(cpu_x), cpu_plan, prepared, cpu.groups);
    }

    DenseTensor result;
    if (tokens == 1) {
        result = ::tyloi::cuda::empty({1, static_cast<std::int64_t>(route.ids.size(1)),
            weight.out_per_expert}, input.options().dtype(::tyloi::kFloat16));
        if (gpu_y.defined()) result.index_copy_(1,
            ::tyloi::cuda::tensor(gpu.positions).to(input.device()), gpu_y);
        if (cpu_y.defined()) result.index_copy_(1,
            ::tyloi::cuda::tensor(cpu.positions).to(input.device()), cpu_y.to(input.device()));
    } else {
        result = ::tyloi::cuda::empty({tokens * routes, weight.out_per_expert},
            input.options().dtype(::tyloi::kFloat16));
        if (gpu_y.defined()) result.index_copy_(0,
            ::tyloi::cuda::tensor(gpu.positions).to(input.device()),
            gpu_y.reshape({-1, weight.out_per_expert}));
        if (cpu_y.defined()) result.index_copy_(0,
            ::tyloi::cuda::tensor(cpu.positions).to(input.device()),
            cpu_y.to(input.device()).reshape({-1, weight.out_per_expert}));
        result = result.reshape({tokens, routes, weight.out_per_expert});
    }
    prepared.record_hybrid();
    return result;
}

DenseTensor project_prepared(const MfeTensor& weight, DenseTensor input,
                             const MoeRoutePlan& route,
                             ::tyloi::weights::PreparedExperts& prepared,
                             const MoeOptions& options, KlMmqState& mmq,
                             MoeScratchCache& moe_scratch,
                             const DenseTensor& scratch_input,
                             bool prequantized) {
    if (prepared.mode == ::tyloi::weights::PreparedExperts::Mode::cpu)
        return cpu_project_groups(weight, std::move(input), route, prepared,
                                  prepared.dispatch.groups);
    if (prepared.mode == ::tyloi::weights::PreparedExperts::Mode::hybrid) {
        auto scratch = moe_scratch.get(weight.cached_source, scratch_input);
        return hybrid_project(weight, std::move(input), route, prepared,
                              options, mmq, *scratch);
    }
    if (!prepared.weight)
        throw std::runtime_error("prepared MFE projection has no active weight");
    if (prepared.mode == ::tyloi::weights::PreparedExperts::Mode::staged) {
        MoeScratch temporary;
        return mixed_project(*prepared.weight, temporary, options, mmq,
            std::move(input), route, false);
    }
    const MfqCudaGuard guard(prepared.device);
    auto scratch = moe_scratch.get(weight.cached_source, scratch_input);
    auto output = mixed_project(*prepared.weight, *scratch, options, mmq,
        std::move(input), route, prequantized);
    prepared.retire();
    return output;
}

} // namespace
#endif

DenseTensor Projector::run(const MfeTensor& weight, DenseTensor input,
                           const MoeRoutePlan& route, bool prequantized, int epilogue) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!weight.shards.empty()) {
        if (prequantized)
            throw std::runtime_error("prequantized MoE activation reuse is unavailable with expert parallelism");
        std::vector<DenseTensor> outputs(weight.shards.size());
        for (size_t launch = 0; launch < weight.shards.size(); ++launch) {
            const auto index = parallel_index(options_.peer_first, launch, weight.shards.size());
            const auto& shard = weight.shards[index];
            if (!shard.weight) throw std::runtime_error("expert-parallel MoE shard is missing");
            MfqCudaGuard guard(shard.device);
            auto local = input.defined() ? collectives_.copy(input, shard.device) : input;
            const auto& local_route = moe_route_to_device(collectives_, route, shard.device);
            outputs[index] = run(*shard.weight, std::move(local), local_route, false, epilogue);
        }
        return collectives_.sum(std::move(outputs), primary_, options_.reduce_to_primary, options_.fp16_reduce);
    }
    if (epilogue < 0 || epilogue > 2 || (epilogue && weight.out_per_expert % 2))
        throw std::runtime_error("invalid MFE NINT projection epilogue");
    if (!weight.cached_source && (weight.compact || weight.pools.empty()))
        return Mfe{weight.n_experts, weight.out_per_expert, weight.neuron_len,
            weight.partial_experts, weight.compact}.forward(mfe_scratch_, input, route, prequantized, epilogue);
    if (epilogue) throw std::runtime_error(prequantized
        ? "mixed prequantized MFE GLU epilogue is unavailable"
        : "mixed MFE GLU epilogue must use its native dispatch");
    const auto options = ::tyloi::cuda::options::moe(options_.moe);
    if (weight.cached_source) {
        auto prepared = ::tyloi::weights::prepare_cached_projection(
            weight, route, input.device(),
            prequantized ? ::tyloi::weights::ExpertProjection::prequantized
                         : ::tyloi::weights::ExpertProjection::project,
            options.heterogeneous);
        if (prequantized &&
            (prepared.complete || prepared.mode == ::tyloi::weights::PreparedExperts::Mode::cpu ||
             prepared.mode == ::tyloi::weights::PreparedExperts::Mode::hybrid) &&
            input.scalar_type() != ::tyloi::kFloat16)
            throw std::runtime_error(prepared.complete
                ? "resident CPU experts require original f16 activations"
                : "CPU cold route requires original f16 activation");
        return project_prepared(weight, input, route, prepared,
            options, mmq_, moe_scratch_, input, prequantized);
    }
    return mixed_project(weight, *moe_scratch_.get(mixed_source(weight), input), options, mmq_, input, route, prequantized);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

DenseTensor Projector::glu(const MfeTensor& weight, DenseTensor input,
                           const MoeRoutePlan& route, bool gelu) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!weight.shards.empty()) {
        std::vector<DenseTensor> outputs(weight.shards.size());
        for (size_t launch = 0; launch < weight.shards.size(); ++launch) {
            const auto index = parallel_index(options_.peer_first, launch, weight.shards.size());
            const auto& shard = weight.shards[index];
            if (!shard.weight) throw std::runtime_error("expert-parallel MoE shard is missing");
            MfqCudaGuard guard(shard.device);
            auto local = input.defined() ? collectives_.copy(input, shard.device) : input;
            const auto& local_route = moe_route_to_device(collectives_, route, shard.device);
            outputs[index] = glu(*shard.weight, std::move(local), local_route, gelu);
        }
        return collectives_.sum(std::move(outputs), primary_, options_.reduce_to_primary, options_.fp16_reduce);
    }
    if (!weight.cached_source && (weight.compact || weight.pools.empty())) return run(weight, input, route, false, gelu ? 2 : 1);
    const auto options = ::tyloi::cuda::options::moe(options_.moe);
    if (weight.cached_source) {
        auto prepared = ::tyloi::weights::prepare_cached_projection(
            weight, route, input.device(), ::tyloi::weights::ExpertProjection::glu,
            options.heterogeneous);
        if (prepared.complete || prepared.mode == ::tyloi::weights::PreparedExperts::Mode::cpu ||
            prepared.mode == ::tyloi::weights::PreparedExperts::Mode::hybrid) {
            auto projected = project_prepared(weight, input, route, prepared,
                options, mmq_, moe_scratch_, input, false);
            return gelu ? moe_geglu_split_cuda(projected) : moe_swiglu_split_cuda(projected);
        }
        if (!prepared.weight)
            throw std::runtime_error("prepared MFE GLU has no active weight");
        if (prepared.mode == ::tyloi::weights::PreparedExperts::Mode::staged) {
            MoeScratch temporary;
            return mixed_glu(*prepared.weight, temporary, options, mmq_,
                std::move(input), route, gelu);
        }
        const MfqCudaGuard guard(prepared.device);
        auto scratch = moe_scratch_.get(weight.cached_source, input);
        auto output = mixed_glu(*prepared.weight, *scratch, options, mmq_,
            std::move(input), route, gelu);
        prepared.retire();
        return output;
    }
    return mixed_glu(weight, *moe_scratch_.get(mixed_source(weight), input), options, mmq_, input, route, gelu);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

DenseTensor Projector::gated(const MfeTensor& weight, DenseTensor input,
                             const MoeRoutePlan& route, bool gelu) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!weight.shards.empty()) {
        std::vector<DenseTensor> outputs(weight.shards.size());
        for (size_t launch = 0; launch < weight.shards.size(); ++launch) {
            const auto index = parallel_index(options_.peer_first, launch, weight.shards.size());
            const auto& shard = weight.shards[index];
            if (!shard.weight) throw std::runtime_error("expert-parallel MoE shard is missing");
            MfqCudaGuard guard(shard.device);
            auto local = input.defined() ? collectives_.copy(input, shard.device) : input;
            const auto& local_route = moe_route_to_device(collectives_, route, shard.device);
            outputs[index] = gated(*shard.weight, std::move(local), local_route, gelu);
        }
        return collectives_.sum(std::move(outputs), primary_, options_.reduce_to_primary, options_.fp16_reduce);
    }
    if (!input.is_cuda() || !input.is_contiguous() || input.scalar_type() != ::tyloi::kFloat16 ||
        input.dim() != 3 || input.size(2) != 2 * weight.neuron_len)
        throw std::runtime_error("fused MFE GLU input has an unsupported layout");
    auto hidden = gelu ? moe_geglu_split_cuda(input) : moe_swiglu_split_cuda(input);
    return run(weight, hidden, route);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

DenseTensor Projector::clamped(const MfeTensor& weight, DenseTensor input,
                               const MoeRoutePlan& route, double limit) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!weight.shards.empty()) {
        std::vector<DenseTensor> outputs(weight.shards.size());
        for (size_t launch = 0; launch < weight.shards.size(); ++launch) {
            const auto index = parallel_index(options_.peer_first, launch, weight.shards.size());
            const auto& shard = weight.shards[index];
            if (!shard.weight) throw std::runtime_error("expert-parallel MoE shard is missing");
            MfqCudaGuard guard(shard.device);
            auto local = input.defined() ? collectives_.copy(input, shard.device) : input;
            const auto& local_route = moe_route_to_device(collectives_, route, shard.device);
            outputs[index] = clamped(*shard.weight, std::move(local), local_route, limit);
        }
        return collectives_.sum(std::move(outputs), primary_, options_.reduce_to_primary, options_.fp16_reduce);
    }
    if (weight.cached_source && supports_clamped_swiglu(weight)) {
        const auto options = ::tyloi::cuda::options::moe(options_.moe);
        auto prepared = ::tyloi::weights::prepare_cached_projection(
            weight, route, input.device(), ::tyloi::weights::ExpertProjection::clamp,
            options.heterogeneous);
        if (prepared.quantized) {
            auto parts = input.split_with_sizes({weight.neuron_len, weight.neuron_len}, -1);
            auto gate = ::tyloi::cuda::clamp_max(parts[0], limit);
            auto up = ::tyloi::cuda::clamp(parts[1], -limit, limit);
            auto activation = (::tyloi::cuda::silu(gate) * up).contiguous();
            return project_prepared(weight, std::move(activation), route, prepared,
                options, mmq_, moe_scratch_, input, false);
        }
        if (prepared.mode == ::tyloi::weights::PreparedExperts::Mode::cpu ||
            prepared.mode == ::tyloi::weights::PreparedExperts::Mode::hybrid)
            throw std::runtime_error("cached clamped MFE CPU path requires quantized experts");
        if (!prepared.weight)
            throw std::runtime_error("prepared clamped MFE has no active weight");
        if (prepared.mode == ::tyloi::weights::PreparedExperts::Mode::staged) {
            MoeScratch temporary;
            return mixed_clamp(*prepared.weight, temporary, options, mmq_,
                std::move(input), route, limit);
        }
        const MfqCudaGuard guard(prepared.device);
        auto scratch = moe_scratch_.get(weight.cached_source, input);
        auto output = mixed_clamp(*prepared.weight, *scratch, options, mmq_,
            std::move(input), route, limit);
        prepared.retire();
        return output;
    }
    if (!weight.compact && !weight.cached_source && supports_clamp(weight)) {
        const auto options = ::tyloi::cuda::options::moe(options_.moe);
        return mixed_clamp(weight, *moe_scratch_.get(mixed_source(weight), input), options, mmq_, input, route, limit);
    }
    return Mfe{weight.n_experts, weight.out_per_expert, weight.neuron_len,
        weight.partial_experts, weight.compact}.clamped(mfe_scratch_, input, route, limit);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

DenseTensor Projector::run(const TqMoeBanks& banks, DenseTensor input, DenseTensor ids,
                      DenseTensor weights, float limit) const {
    if (banks.slices.empty())
        return banks.forward(tq_, std::move(input), std::move(ids), std::move(weights),
                             limit, &profiler_);
    if (!banks.gate)
        throw std::invalid_argument("TQ expert parallelism requires a Gate bank");
    std::vector<DenseTensor> partials(banks.slices.size());
    for (size_t launch = 0; launch < banks.slices.size(); ++launch) {
        const auto index = parallel_index(options_.peer_first, launch, banks.slices.size());
        const auto& slice = banks.slices[index];
        const MfqCudaGuard guard(slice.device);
        auto local_input = collectives_.copy(input, slice.device);
        auto local_ids = collectives_.copy(ids, slice.device);
        auto local_weights = collectives_.copy(weights, slice.device);
        partials[index] = banks.forward(tq_, std::move(local_input), std::move(local_ids),
            std::move(local_weights), limit, &profiler_, slice.begin, slice.end);
    }
    return collectives_.sum(std::move(partials), primary_,
                            options_.reduce_to_primary, options_.fp16_reduce);
}

} // namespace tyloi::cuda

namespace tyloi::ops {
namespace {

void validate_expert_inputs(const Tensor& input, const Tensor& ids,
                            const Tensor& weights,
                            std::span<const Tensor> gate_up,
                            const Tensor& down, std::int64_t width) {
    TYLOINFER_RUNTIME_CHECK(gate_up.size() >= 1 && gate_up.size() <= 2 && down.defined() &&
        input.defined() && ids.defined() && weights.defined() && width > 0 &&
        input.dim() == 2 && ids.dim() == 2 && weights.dim() == 2 &&
        input.size(0) == ids.size(0) && ids.size(0) == weights.size(0) &&
        ids.size(1) == weights.size(1) && input.device() == ids.device() &&
        input.device() == weights.device(), "invalid routed expert inputs");
    for (const auto& weight : gate_up)
        TYLOINFER_RUNTIME_CHECK(weight.defined(), "missing routed expert projection");
}

} // namespace
} // namespace tyloi::ops

tyloi::Tensor tyloi::ops::experts(const Tensor& input, const Tensor& ids, const Tensor& weights,
                                  std::span<const Tensor> gate_up, const Tensor& down, std::int64_t width) {
    validate_expert_inputs(input, ids, weights, gate_up, down, width);
    auto& executor = cuda::Executor::current();
    const auto project = executor.projector(cuda::Executor::current_ffn());
    auto banks = tq_banks(gate_up.front(), gate_up.back(), down);
    if (banks.gate || banks.up || banks.down) {
        TYLOINFER_RUNTIME_CHECK(gate_up.size() == 2 && down && banks.gate && banks.up && banks.down,
                                "TQ MoE requires separate Gate, Up and Down banks");
        return executor.profiler().measure("tq.moe", [&] {
            return project.run(banks, input.dense(), ids.dense(), weights.dense());
        });
    }
    DenseTensor hidden;
    {
        auto gu = project.run(gate_up[0], input.dense(), ids.dense());
        if (gate_up.size() == 2) {
            auto up = project.run(gate_up[1], input.dense(), ids.dense());
            gu = cuda::cat({gu, up}, -1);
        }
        auto gate = gu.narrow(-1, 0, width), up = gu.narrow(-1, width, width);
        hidden = (gate * cuda::sigmoid(gate)) * up;
    }
    auto pairs = project.run(down, hidden, ids.dense());
    hidden = {};
    auto reduced = cuda::zeros({input.size(0), input.size(1)},
                               input.options().dtype(kFloat32));
    for (int64_t r = 0; r < ids.size(1); ++r)
        reduced = reduced + pairs.select(1, r).to(kFloat32) *
                            weights.dense().select(1, r).unsqueeze(-1);
    return reduced.to(pairs.scalar_type());
}

tyloi::Tensor tyloi::ops::moe(const Tensor& input, const Tensor& ids, const Tensor& weights,
                              std::span<const Tensor> gate_up, const Tensor& down,
                              std::span<const Tensor> shared, const Tensor& shared_gate,
                              std::int64_t width, std::int64_t shared_rows) {
    validate_expert_inputs(input, ids, weights, gate_up, down, width);
    TYLOINFER_RUNTIME_CHECK(shared.size() == 3 && shared_gate.defined(),
                            "invalid shared expert inputs");
    for (const auto& weight : shared)
        TYLOINFER_RUNTIME_CHECK(weight.defined(), "missing shared expert projection");

    const auto shared_output = cuda::MoeSharedExpert(
        shared[0], shared[1], shared[2], shared_gate).forward(input.dense(), shared_rows);
    const auto routed = experts(input, ids, weights, gate_up, down, width).dense();
    return routed + shared_output;
}

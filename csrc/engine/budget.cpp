#include "engine/budget.h"

#include "engine/generation/prefill_budget.h"
#include "engine/state/session/sessions.h"
#include "engine/requests/processor.h"
#include "engine/engine.h"
#include "engine/weights/memory.h"
#include "components/error.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

namespace tyloi::engine {
static void require_cuda_device(Device device) {
    if (!device.is_cuda() || device.index < 0)
        throw std::invalid_argument("budget requires a CUDA device");
}

Budget::Budget(::tyloi::Executor& executor,
        ::tyloi::weights::Memory& weight_memory, ::tyloi::weights::Cache& weights,
        state::Sessions& cache, Device primary)
    : executor_(executor), weight_memory_(weight_memory), weights_(weights), cache_(cache),
      primary_(primary), initial_expert_budget_(weights.budget()) {
    require_cuda_device(primary);
    allocation_ceiling_ = memory(primary_).limit;
}

Budget::~Budget() { unbind(); }

void Budget::bind(bool predictor) {
    weights_.bind(*this, primary_, predictor);
}

void Budget::unbind() noexcept {
    weights_.unbind();
}

::tyloi::Executor::Memory Budget::memory(Device device) const {
    return executor_.memory({&device, 1}).front();
}

void Budget::reclaim(std::size_t bytes) {
    const auto available=memory(primary_).allocatable;
    if(available>=bytes)return;
    weights_.release_workspace(bytes, "workspace");
    if(memory(primary_).allocatable>=bytes)return;
    const auto released=trim_prefix(bytes, primary_);
    executor_.reclaim(primary_);
    if(released)std::cerr<<"runtime_workspace_reclaim requested_bytes="<<bytes
        <<" available_bytes="<<memory(primary_).allocatable<<" prefix_released_bytes="<<released<<'\n';
}

void Budget::reclaim_kv(Device device, std::size_t bytes) {
    require_cuda_device(device);
    if (memory(device).allocatable >= bytes) return;
    if (device == primary_) {
        weights_.release_workspace(bytes, "kv_growth");
        if (memory(device).allocatable >= bytes) return;
        trim_prefix(bytes, primary_);
    }
    executor_.reclaim(device);
}

void Budget::reserve_kv(Device device, std::size_t bytes) {
    reclaim_kv(device, bytes);
    if (memory(device).allocatable < bytes)
        throw engine::InferenceInputError(engine::InferenceInputErrorCode::ContextLength,
            "insufficient VRAM for context growth; reduce concurrent requests or prompt length");
}

void Budget::reserve_ram(std::size_t bytes) {
    cache_.reserve_ram(bytes);
    weights_.reserve(bytes);
}

void Budget::reserve(std::span<const std::pair<Device, std::size_t>> growth) {
    for (const auto& [device, bytes] : growth) {
        if (!bytes) continue;
        if (device.is_cpu()) reserve_ram(bytes);
        else reserve_kv(device, prefill_budget_add(bytes, std::size_t(64) << 20));
    }
}

std::size_t Budget::trim_prefix(std::size_t bytes, Device device) {
    std::size_t hot=0;
    for(const auto& [name,value]:cache_.metrics())
        if(name=="prefix_cache_hot_bytes")hot=std::size_t(value);
    const auto available=memory(device).allocatable;
    const auto shortage=bytes>available?bytes-available:0;
    return cache_.trim_hot(hot>shortage?hot-shortage:0);
}

void Budget::workspace(Device device, std::size_t bytes) {
    require_cuda_device(device);
    weights_.workspace(bytes);
    if(memory(device).allocatable>=bytes)return;
    const auto released=trim_prefix(bytes, device);
    executor_.reclaim(device);
    std::cerr<<"runtime_workspace_budget requested_bytes="<<bytes
        <<" available_bytes="<<memory(device).allocatable<<" prefix_released_bytes="<<released<<'\n';
}

PrefillChunk Budget::batch(Device device, std::size_t row_bytes, PrefillChunk chunk,
        std::size_t growth) {
    require_cuda_device(device);
    const auto resources=memory(device);
    const auto transfer=weights_.prefill_budget(false).transfer_bytes;
    const auto fixed=engine::prefill_budget_add((std::size_t(1)<<30)+growth,transfer);
    const auto available=resources.allocatable+resources.workspace;
    const auto selected=engine::select_prefill_chunk(chunk.count,chunk.count,available,fixed,row_bytes);
    if(!selected)throw engine::InferenceInputError(
        engine::InferenceInputErrorCode::ContextLength,
        "insufficient CUDA workspace for batched prefill; reduce concurrent requests");
    chunk.count=selected;
    return chunk;
}

void Budget::request(Device device, bool retire_decode) {
    require_cuda_device(device);
    weights_.request(retire_decode);
    executor_.reclaim(device);
}

void Budget::prepare_prefill(Device device, std::size_t tokens, std::int64_t limit,
        ::tyloi::weights::Cache::Prefill mode, bool retire_decode) {
    require_cuda_device(device);
    if (auto bytes = weights_.prepare_prefill(device, tokens, limit, mode, retire_decode))
        workspace(device, *bytes);
}

std::int64_t Budget::prefill(std::size_t tokens, std::int64_t limit,
        std::size_t row_bytes, bool resumed, bool automatic) {
    prepare_prefill(primary_, tokens, limit, weights_.prefill_mode(tokens, resumed));
    return chunk(primary_, tokens, limit, row_bytes, resumed, automatic);
}

std::int64_t Budget::chunk(Device device, std::size_t tokens, std::int64_t limit,
        std::size_t row_bytes, bool resumed, bool automatic, bool allow_borrow) {
    require_cuda_device(device);
    if (weights_.prefill_mode(tokens, resumed) == ::tyloi::weights::Cache::Prefill::retained)
        return std::min<int64_t>(4, limit);
    if (tokens <= 128) return std::min<int64_t>(std::max<std::size_t>(1, tokens), limit);

    auto budget = weights_.prefill_budget(allow_borrow);
    auto resources = memory(device);
    auto available = engine::prefill_budget_add(engine::prefill_budget_add(resources.allocatable, resources.workspace), budget.lendable_bytes);
    const auto sequence_bytes = allow_borrow
        ? std::size_t(tokens <= 16384 ? tokens : std::min<std::size_t>(tokens, limit)) * row_bytes * 3 / 8 : 0;
    const auto fixed = engine::prefill_budget_add(budget.reserve_bytes + sequence_bytes, budget.transfer_bytes);
    const auto wanted = std::min<int64_t>({int64_t(tokens), limit, engine::max_prefill_compute_chunk});
    const auto required = engine::prefill_budget_add(fixed, std::size_t(wanted) * row_bytes);
    if (required > available) {
        workspace(device, required > budget.lendable_bytes ? required - budget.lendable_bytes : 0);
        budget = weights_.prefill_budget(allow_borrow);
        resources = memory(device);
        available = engine::prefill_budget_add(resources.allocatable + resources.workspace, budget.lendable_bytes);
    }
    const auto chunk_size = engine::select_prefill_chunk(tokens, limit, available, fixed, row_bytes);
    if (!chunk_size) throw engine::InferenceInputError(engine::InferenceInputErrorCode::ContextLength,
        "insufficient CUDA workspace for prefill computation and expert transfer buffers");
    std::cerr << "runtime_prefill_plan mode=" << (automatic ? "auto" : "limited")
              << " tokens=" << tokens << " limit=" << limit << " compute_chunk=" << chunk_size
              << " available_bytes=" << available << " lendable_bytes=" << budget.lendable_bytes
              << " transfer_bytes=" << budget.transfer_bytes << '\n';
    return chunk_size;
}

bool Budget::snapshot(Device device, std::size_t bytes, std::size_t peak) {
    require_cuda_device(device);
    std::size_t hot = 0, limit = 0;
    for (const auto& metric : cache_.metrics()) {
        if (metric.first == "prefix_cache_hot_bytes") hot = std::size_t(metric.second);
        if (metric.first == "prefix_cache_max_bytes") limit = std::size_t(metric.second);
    }
    const auto target = limit > bytes ? limit - bytes : 0;
    if (hot > target) cache_.trim_hot(target);
    const auto needed = peak + (std::size_t(64) << 20);
    workspace(device, needed);
    const auto available = memory(device).allocatable;
    if (available < needed) {
        std::cerr << "runtime_session_cache action=skip reason=workspace_pressure"
            << " requested_bytes=" << needed << " available_bytes=" << available << '\n';
        return false;
    }
    return true;
}

Budget::Change Budget::change(const engine::SessionCommand& command) {
    return Change(*this, command);
}

Budget::Change::Change(Budget& budget, const engine::SessionCommand& command)
    : budget_(budget), bytes(command.bytes) {
    const auto resources = budget.memory(budget.primary_);
    dense = budget.weight_memory_.resident_bytes();
    minimum = budget.weights_.minimum();
    requested_weights = command.weight_bytes.value_or(budget.weight_budget_);
    if (requested_weights && requested_weights < dense + minimum)
        throw std::invalid_argument("weight budget is below dense weights and minimum expert working set");
    limit = command.bytes ? budget.allocation_ceiling_
            ? std::min<std::size_t>(command.bytes, budget.allocation_ceiling_)
            : command.bytes
        : budget.allocation_ceiling_;
    if (limit) {
        limit = std::min(limit, std::min(resources.total, resources.available + resources.resident));
    }
    if (limit && limit < dense + minimum)
        throw std::invalid_argument("resident memory budget is below non-offloadable weights");
    previous_limit = resources.limit;
    previous_experts = budget.weights_.budget();
    before = resources.resident;
}

engine::SessionResult Budget::Change::apply() {
    auto& weights_ = budget_.weights_;
    auto& cache_ = budget_.cache_;
    auto& executor = budget_.executor_;
    const auto device = budget_.primary_;
    weights_.retire_decode();
    executor.reclaim(device);
    auto target = requested_weights ? requested_weights - dense : budget_.initial_expert_budget_;
    std::size_t hot = 0, expert_bytes = 0;
    for (const auto& [name, value] : cache_.metrics())
        if (name == "prefix_cache_hot_bytes") hot = std::size_t(value);
    for (const auto& [name, value] : weights_.metrics())
        if (name == "resident_expert_cache_bytes") expert_bytes = std::size_t(value);
    auto used = budget_.memory(device).resident;
    if (limit && used > limit) {
        cache_.trim_hot(hot > used - limit ? hot - (used - limit) : 0);
        executor.reclaim(device);
        used = budget_.memory(device).resident;
    }
    const auto fixed = used > expert_bytes ? used - expert_bytes : 0;
    if (limit) {
        if (fixed > limit || minimum > limit - fixed)
            throw std::invalid_argument("resident memory budget is below live KV and workspace requirements");
        target = std::min(target, limit - fixed);
    }
    try {
        if (!limit || limit > previous_limit) executor.set_memory_limit(device, limit);
        weights_.resize(std::max(target, minimum));
        executor.set_memory_limit(device, limit);
    } catch (...) {
        executor.set_memory_limit(device, 0);
        weights_.resize(previous_experts);
        executor.set_memory_limit(device, previous_limit);
        throw;
    }
    budget_.resident_budget_ = bytes;
    budget_.weight_budget_ = requested_weights;
    const auto after = budget_.memory(device).resident;
    return {before > after ? before - after : 0, {}};
}

} // namespace tyloi::engine

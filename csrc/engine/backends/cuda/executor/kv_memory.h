#pragma once

#include "engine/kv_budget.h"
#include "engine/state/kv_layout.h"
#include "engine/backends/cuda/kernels/mfq_cuda_kv_hot_cache.h"
#include "engine/weights/workspace.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::engine { class Budget; }
namespace tyloi::cuda {

engine::KvOptions kv_memory_options();

struct KvMemoryAllocation {
    KvMemoryAllocation() = default;
    KvMemoryAllocation(const KvMemoryAllocation&) = delete;
    KvMemoryAllocation& operator=(const KvMemoryAllocation&) = delete;
    KvMemoryAllocation(KvMemoryAllocation&&) noexcept = default;
    KvMemoryAllocation& operator=(KvMemoryAllocation&&) noexcept = default;
    state::QsaLayout layout;
    Tensor keys, values, index, pooled;
    TensorOptions options;
    std::shared_ptr<state::KvOffloadStore> offload;
    HotKvCache hot;
    std::int64_t selected_columns = 0;
    bool host = false, ready = false;
    std::vector<Tensor> storage() const;
    std::size_t gpu_bytes() const;
    std::size_t ram_bytes() const;
    std::size_t growth_bytes(std::int64_t tokens) const;
    void ensure(std::int64_t tokens);

private:
    struct Virtual;
    std::shared_ptr<Virtual> virtual_keys_, virtual_values_;
    std::vector<weights::KvRegion> expert_regions_;
    std::size_t initial_bytes() const;
    void materialize(const engine::KvOptions&, const engine::KvPlacement&,
        std::shared_ptr<state::KvOffloadStore>, weights::Cache* lending);
    friend class KvMemoryManager;
};

// Root-owned registration and storage; forked executors borrow this manager.
class KvMemoryManager {
public:
    explicit KvMemoryManager(engine::KvOptions options = kv_memory_options());
    std::shared_ptr<KvMemoryAllocation> add(std::int64_t maximum, std::int64_t heads,
        std::int64_t key_width, std::int64_t value_width, std::int64_t index_width,
        std::int64_t pool, const TensorOptions& options, std::int64_t selected_columns = 0);
    // Cache outlives the manager and its allocations; call before expert arenas.
    void finalize(weights::Cache& cache, std::size_t minimum_experts = 0);
    std::vector<std::pair<Device, std::size_t>> growth(std::int64_t tokens) const;
    void ensure(std::int64_t tokens);
    void bind_budget(engine::Budget* budget) { budget_ = budget; }
    std::shared_ptr<state::KvOffloadStore> offload() const { return offload_; }
    bool dynamic() const { return options_.dynamic; }
    bool ssd_enabled() const { return options_.placement == "auto_ssd"; }
    std::size_t growth_reserve() const;
    std::size_t expert_budget_charge() const;
    std::vector<std::pair<std::string, double>> metrics() const;

private:
    engine::KvOptions options_;
    std::vector<std::weak_ptr<KvMemoryAllocation>> allocations_;
    std::shared_ptr<state::KvOffloadStore> offload_;
    engine::Budget* budget_ = nullptr;
    bool ready_ = false, automatic_ram_ = false;
    weights::Cache* cache_ = nullptr;
    bool lending_ = false;
    std::size_t gpu_budget_ = 0, ram_required_ = 0;
};

} // namespace tyloi::cuda

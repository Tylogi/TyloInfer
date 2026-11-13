#pragma once

#include "engine/tensor.h"
#include "engine/state/kv_offload.h"
#include <memory>
#include <string>
#include <vector>

namespace tyloi::engine { class Budget; }
namespace tyloi::weights { class Cache; }
namespace tyloi::cuda {
struct HotKvCache;
// Placement applies to raw K/V. Indexer and kernel scratch remain GPU-local.
struct KvMemoryOptions {
    std::string placement = "auto";
    bool dynamic = false;
    bool budget_policy = false, ram_preallocated = true;
    std::size_t gpu_budget = 0;
    std::size_t gpu_limit = 0, ram_limit = 0;
    bool ram_limit_set = false;
    bool additional_request = false;
    int64_t auto_stream_threshold = 65536, resident_cells = 32768;
    std::string directory;
};
KvMemoryOptions kv_memory_options();

struct KvMemoryAllocation {
    KvMemoryAllocation() = default;
    KvMemoryAllocation(const KvMemoryAllocation&) = delete;
    KvMemoryAllocation& operator=(const KvMemoryAllocation&) = delete;
    KvMemoryAllocation(KvMemoryAllocation&&) noexcept = default;
    KvMemoryAllocation& operator=(KvMemoryAllocation&&) noexcept = default;
    ::tyloi::Tensor keys, values, index, pooled;
    std::shared_ptr<KvOffloadStore> offload;
    int64_t maximum = 0, heads = 0, key_width = 0, value_width = 0, index_width = 0, pool = 0, selected_columns = 0;
    ::tyloi::TensorOptions options;
    bool host = false, ready = false;
    std::vector<::tyloi::Tensor> storage() const;
    std::size_t raw_bytes() const;
    std::size_t index_bytes() const;
    std::size_t gpu_bytes() const;
    std::size_t ram_bytes() const;
    std::size_t growth_bytes(int64_t tokens) const;
    void ensure(int64_t tokens);
private:
    struct Native;
    Native& native();
    std::shared_ptr<Native> native_;
    friend class KvMemoryManager;
    friend HotKvCache& kv_hot(KvMemoryAllocation&);
};
class KvMemoryManager {
public:
    explicit KvMemoryManager(KvMemoryOptions options = kv_memory_options());
    std::shared_ptr<KvMemoryAllocation> add(int64_t maximum, int64_t heads,
        int64_t key_width, int64_t value_width, int64_t index_width, int64_t pool,
        const ::tyloi::TensorOptions& options,int64_t selected_columns=0);
    // Cache outlives the manager and its allocations; call before expert arenas.
    void finalize(tyloi::weights::Cache& cache, std::size_t minimum_experts = 0);
    void ensure(int64_t tokens);
    void bind_budget(tyloi::engine::Budget* budget) { budget_ = budget; }
    std::shared_ptr<KvOffloadStore> offload() const { return offload_; }
    bool dynamic() const { return options_.dynamic; }
    bool ssd_enabled() const { return options_.placement=="auto_ssd"; }
    std::size_t growth_reserve() const;
    std::size_t expert_budget_charge() const;
    std::vector<std::pair<std::string,double>> metrics() const;
private:
    KvMemoryOptions options_;
    std::vector<std::weak_ptr<KvMemoryAllocation>> allocations_;
    std::shared_ptr<KvOffloadStore> offload_;
    tyloi::engine::Budget* budget_ = nullptr;
    bool ready_ = false;
    bool automatic_ram_ = false;
    tyloi::weights::Cache* cache_ = nullptr;
    bool lending_ = false;
    std::size_t gpu_budget_=0,ram_required_=0;
};
}

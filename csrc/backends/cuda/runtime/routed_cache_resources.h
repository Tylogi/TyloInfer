#pragma once
#include "storage/moe_expert_cache.h"
#include <array>
#include <optional>

struct CudaExecutionContext;
class TqExpertCache;
class GgufExpertCache;

namespace mfq::cuda {
class RoutedGraphPools;
// The engine coordinates ownership without depending on encoded weight views.
class RoutedCacheLearningScope {
    std::shared_ptr<MoeExpertCache> legacy_;
    std::shared_ptr<GgufExpertCache> gguf_;
    std::shared_ptr<TqExpertCache> tq_;
    bool legacy_previous_,gguf_previous_,tq_previous_;
public:
    explicit RoutedCacheLearningScope(CudaExecutionContext&,bool enabled);
    RoutedCacheLearningScope(const RoutedCacheLearningScope&)=delete;
    RoutedCacheLearningScope& operator=(const RoutedCacheLearningScope&)=delete;
    ~RoutedCacheLearningScope();
};
class RoutedSpeculativeRoutes {
    std::shared_ptr<MoeExpertCache> legacy_;
    std::shared_ptr<GgufExpertCache> gguf_;
    std::shared_ptr<TqExpertCache> tq_;
public:
    RoutedSpeculativeRoutes()=default;
    RoutedSpeculativeRoutes(const RoutedSpeculativeRoutes&)=delete;
    RoutedSpeculativeRoutes& operator=(const RoutedSpeculativeRoutes&)=delete;
    RoutedSpeculativeRoutes(RoutedSpeculativeRoutes&&) noexcept;
    RoutedSpeculativeRoutes& operator=(RoutedSpeculativeRoutes&&)=delete;
    ~RoutedSpeculativeRoutes();
    void discard() noexcept;
    void begin(CudaExecutionContext&,int rows);
    void resolve(int rows,bool cancelled);
};
class RoutedPrefillWorkspaceLoan {
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
public:
    explicit RoutedPrefillWorkspaceLoan(CudaExecutionContext&);
    RoutedPrefillWorkspaceLoan(const RoutedPrefillWorkspaceLoan&)=delete;
    RoutedPrefillWorkspaceLoan& operator=(const RoutedPrefillWorkspaceLoan&)=delete;
    ~RoutedPrefillWorkspaceLoan();
    std::size_t begin(std::size_t bytes);
    void suspend();
    void finish();
};
bool has_routed_expert_cache(const CudaExecutionContext& execution) noexcept;
void finish_routed_expert_exchanges(const CudaExecutionContext& execution);
bool prepare_routed_graph_pool(const CudaExecutionContext& execution);
void prepare_execution_graph_pool(const CudaExecutionContext& execution,std::unique_ptr<RoutedGraphPools>& owned);
void finalize_routed_weight_images(CudaExecutionContext& execution,std::size_t growth);
MoePrefillWorkspaceBudget routed_prefill_workspace_budget(
        const CudaExecutionContext& execution,bool allow_borrow);
std::size_t routed_prefill_transfer_bytes(const CudaExecutionContext& execution);
std::uint64_t routed_expert_streamed_bytes(const CudaExecutionContext& execution);
std::array<double,4> routed_expert_phase_timings(const CudaExecutionContext& execution);
std::vector<std::pair<std::string,double>> routed_expert_memory_metrics(
        const CudaExecutionContext& execution);
std::size_t routed_expert_cache_budget_bytes(const CudaExecutionContext& execution);
void resize_routed_expert_cache(CudaExecutionContext& execution,std::size_t bytes);
void reserve_routed_workspace(CudaExecutionContext& execution,std::size_t bytes);
void constrain_routed_capture_workspace(CudaExecutionContext& execution,std::size_t bytes);
void reserve_routed_capture_workspace(CudaExecutionContext& execution,std::size_t bytes);
void restore_routed_decode_workspace(CudaExecutionContext& execution,std::size_t growth,
        std::optional<std::size_t> transient=std::nullopt,bool rebalance_gguf_cache=true);
void restore_routed_decode_workspace_after_capture(CudaExecutionContext& execution);
std::size_t reclaim_routed_optional_host(CudaExecutionContext& execution,std::size_t allocation_bytes);
std::size_t reclaim_routed_expert_duplicates(CudaExecutionContext&,std::size_t bytes);
std::size_t reclaim_routed_expert_pinned(CudaExecutionContext&,std::size_t bytes);
std::size_t share_routed_expert_host(CudaExecutionContext&,std::size_t allocation_bytes);
} // namespace mfq::cuda

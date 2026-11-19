#pragma once

#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/ops/attention/attention.h"
#include "engine/backends/cuda/ops/ffn/dense.h"
#include "engine/backends/cuda/ops/ffn/options.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/executor.h"

#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

class TqExec;
namespace tyloi::weights { class Cache; }
namespace tyloi::state { class KvOffloadStore; }

namespace tyloi::cuda {

class Branches;
class Collectives;
class RouteStats;
class KvMemoryManager;

class Executor
    : public tyloi::Executor
{
public:
    std::unique_ptr<tyloi::Executor::Scope> enter() override;
    std::unique_ptr<tyloi::Executor::Work> work() override;
    std::unique_ptr<tyloi::Executor::Timer> timer() override;
    std::unique_ptr<tyloi::Executor::Profile> profile(const char* name) override;
    std::unique_ptr<tyloi::Executor::Graph> graph() override;
    std::vector<tyloi::Executor::Memory> memory(std::span<const tyloi::Device> devices) const override;
    void reclaim(tyloi::Device device) override;
    void set_memory_limit(tyloi::Device device, std::size_t bytes) override;
    void bind_budget(tyloi::engine::Budget* budget) override;
    /** Model-thread budget binding shared by the root executor and its forks. */
    struct BudgetLink { tyloi::engine::Budget* value = nullptr; };
    std::weak_ptr<const BudgetLink> budget_link() const noexcept { return budget_; }
    tyloi::engine::Metrics metrics() const override;
    void drain() override;
    void clear() override;
    // Set the default device during loading, before publication or concurrent submissions.
    void set_primary(int primary) noexcept { primary_ = primary; }
    std::unique_ptr<tyloi::Executor::Scope> enter(int device);
    void reset();
    void join();
    static Executor& current();
    static Executor* active() noexcept;
    /** Create the root executor's KV owner before model loading. */
    void init_kv();
    /** Finalize KV mappings after model loading and publish them to the weight cache. */
    void finalize_kv(tyloi::weights::Cache& weights);
    std::shared_ptr<KvMemoryManager> kv_memory() const noexcept { return kv_memory_.lock(); }
    std::shared_ptr<state::KvOffloadStore> kv_offload() const;
    void wait();
    static const FfnMode& current_ffn();
    std::unique_ptr<tyloi::Executor::Scope> enter(int device, const FfnMode& mode);
    // Override only the policy on this active executor; retain its device and stream.
    // Nested policy scopes retain an outer serial-branch requirement.
    std::unique_ptr<tyloi::Executor::Scope> enter(const FfnMode& mode);
    Collectives& collectives() noexcept { return collectives_; }
    const Collectives& collectives() const noexcept { return collectives_; }
    struct Compute {
        std::optional<bool> rotary;
        std::optional<bool> attention_grouped;
        std::optional<bool> residual_two_stage;
        std::optional<bool> residual_dense_injection;
    };

    struct Mode {
        FfnMode ffn;
        struct {
            bool shared_input = true;
            bool fused_conv = true;
        } linear;
        Compute compute;
        std::int64_t kv_capacity = 0;
        bool transposed = true;
        Trace* trace = nullptr;
        int trace_layer = -1;
        int primary = 0;
        bool gemma4_fused_norms = true;
        bool groupwise_output = true;
    };

    struct Options {
        Projector::Options projection;
        AttentionOptions attention{
            .kernels = {.bf16_flash128 = true, .flash128_casts = true,
                .bf16_gqa_decode = true, .decode_aten = false,
                .mma_prefill = true, .mma_decode = true, .split_k = true}};
        DenseOptions dense;
        FfnOptions ffn;
        ResidualOptions residual;
        bool gdn_fused_output = true;
        bool gdn_fused_preparation = true;
        bool gdn_fused_core = true;
        bool tensor_parallel_shared_linear_attention_input = true;
        bool linear_conv_prefill_fused = true;
        bool gdn_transposed_state = true;
        bool gemma4_fused_norms = true;
        bool dsv4_groupwise_output_a = true;
        bool report_memory = false;
        std::string profile_filter;
    };

    static void prepare(std::int64_t vocabulary);

    Executor(Options options, int primary);
    ~Executor();
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    std::unique_ptr<Executor> fork() const;
    std::unique_ptr<Executor> fork(int primary) const;
    ContextSet& contexts() noexcept { return contexts_; }
    const ContextSet& contexts() const noexcept { return contexts_; }

    std::vector<::tyloi::Tensor> storage() const;
    Projector projector(const FfnMode& mode);
    Projector projector(const FfnMode& mode, int primary);
    CudaProfiler& profiler() noexcept { return profiler_; }
    KlMmqState& mmq() noexcept { return mmq_; }
    const KlMmqState& mmq() const noexcept { return mmq_; }
    const std::shared_ptr<TqExec>& tq() const noexcept { return tq_; }
    RouteStats& routes() noexcept { return routes_; }
    const RouteStats& routes() const noexcept { return routes_; }
    Branches& branches() noexcept { return branches_; }
    const Options& options() const noexcept { return options_; }
    Mode& mode() noexcept { return mode_; }
    const Mode& mode() const noexcept { return mode_; }
    Mode forward_mode(int primary) const;
    const AttentionOptions& attention_options() const noexcept;
    ResidualOptions residual_options() const noexcept;
    const Compute& compute() const noexcept;
    std::unique_ptr<tyloi::Executor::Scope> enter(const Compute& policy);
    std::unique_ptr<tyloi::Executor::Scope> enter(const AttentionKernels& kernels);
    bool fp32_residual() const noexcept { return options_.dense.fp32_residual; }
    AttentionScratch& attention_scratch(std::size_t index);
    using AttentionGeometry = std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t>;
    AttentionScratch& attention_scratch(tyloi::Device device, AttentionGeometry geometry);
    MlaScratch& mla_scratch(int device) { return mla_scratch_[device]; }

private:
    void configure_kv(std::shared_ptr<KvMemoryManager> memory, std::shared_ptr<state::KvOffloadStore> offload);
    Executor(ContextSet& contexts, CudaProfiler& profiler, KlMmqState& mmq, const std::shared_ptr<TqExec>& tq,
             Branches& branches, Collectives& collectives, RouteStats& routes,
             const Options& options, Mode& mode, int primary,
             std::weak_ptr<KvMemoryManager>, std::weak_ptr<state::KvOffloadStore>,
             const std::shared_ptr<BudgetLink>&);
    std::unique_ptr<ContextSet> owned_contexts_;
    ContextSet& contexts_;
    std::unique_ptr<Options> owned_options_;
    const Options& options_;
    std::unique_ptr<Mode> owned_mode_;
    Mode& mode_;
    std::unique_ptr<CudaProfiler> owned_profiler_;
    CudaProfiler& profiler_;
    std::unique_ptr<KlMmqState> owned_mmq_;
    KlMmqState& mmq_;
    std::shared_ptr<TqExec> tq_;
    std::unique_ptr<Branches> owned_branches_;
    Branches& branches_;
    std::unique_ptr<Collectives> owned_collectives_;
    Collectives& collectives_;
    std::unique_ptr<RouteStats> owned_routes_;
    RouteStats& routes_;
    std::weak_ptr<KvMemoryManager> kv_memory_;
    std::weak_ptr<state::KvOffloadStore> kv_offload_;
    std::shared_ptr<KvMemoryManager> owned_kv_memory_;
    std::shared_ptr<state::KvOffloadStore> owned_kv_offload_;
    std::shared_ptr<BudgetLink> budget_;
    int primary_;
    std::vector<AttentionScratch> scratch_;
    MfeScratch mfe_scratch_;
    MoeScratchCache moe_scratch_;
    mutable std::mutex attention_scratch_mutex_;
    std::map<std::pair<int, std::uintptr_t>, std::map<AttentionGeometry, AttentionScratch>> attention_scratch_;
    std::map<int, MlaScratch> mla_scratch_;
};

} // namespace tyloi::cuda

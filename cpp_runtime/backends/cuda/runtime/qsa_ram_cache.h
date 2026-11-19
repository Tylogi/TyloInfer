#pragma once
#include "mfq_tensor_backend.h"
#include "tyloinfer/qsa_ram_store.h"
#include <cuda_runtime_api.h>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace mfq::cuda {
inline constexpr std::size_t qsa_shared_vram_cache_default=100ULL<<20;
inline constexpr int qsa_prefill_recall_tile_rows=8;
bool qsa_host_metadata_enabled(bool prefer_host=false);
mfq_tensor_backend::Tensor qsa_descriptor_table(std::int64_t rows,std::int64_t stride,
    const mfq_tensor_backend::TensorOptions& options,bool mapped);
// A decode window and its prefill-owned caches may use different streams on
// the same device. Limit optional reclamation across that entire window.
class QsaRamDecodeBudgetScope {
public:
    explicit QsaRamDecodeBudgetScope(int device,bool defer_shared_cache=false);
    ~QsaRamDecodeBudgetScope();
    QsaRamDecodeBudgetScope(const QsaRamDecodeBudgetScope&)=delete;
    QsaRamDecodeBudgetScope& operator=(const QsaRamDecodeBudgetScope&)=delete;
    static bool claim_optional_reclaim(int device);
    static bool defer_shared_cache(int device);
private:
    static thread_local QsaRamDecodeBudgetScope* active_;
    QsaRamDecodeBudgetScope* previous_;
    int device_;
    bool claimed_=false;
    bool defer_shared_cache_;
};
using QsaRamReclaimer=std::function<std::size_t(std::size_t)>;
// Registration is owned by the engine. Queries retain only the callback for
// their duration; it must in turn use weak ownership of disposable RAM data.
std::shared_ptr<void> register_qsa_ram_reclaimer(int device,cudaStream_t stream,QsaRamReclaimer reclaim);
std::size_t reclaim_qsa_ram(int device,cudaStream_t stream,std::size_t bytes);
// RAM-only KV may reclaim optional expert RAM after compression. SSD-backed
// KV instead spills cold blocks and retains the RAM expert population.
std::shared_ptr<void> register_qsa_ram_capacity_reclaimer(int device,cudaStream_t stream,QsaRamReclaimer reclaim);
std::size_t reclaim_qsa_ram_capacity(int device,cudaStream_t stream,std::size_t bytes);
// Unlike a reclaim request, every reservation callback receives the complete
// free-RAM target. A zero target releases a previous optional-cache limit.
std::shared_ptr<void> register_qsa_ram_budget_reserver(int device,cudaStream_t stream,QsaRamReclaimer reserve);
std::size_t reserve_qsa_ram_budget(int device,cudaStream_t stream,std::size_t free_bytes);
// Locked-page exhaustion is independent of available physical RAM. Invoke
// these callbacks only on the model thread, outside pool/store/graph locks.
std::shared_ptr<void> register_qsa_pinned_reclaimer(int device,cudaStream_t stream,QsaRamReclaimer reclaim);
std::size_t reclaim_qsa_pinned(int device,cudaStream_t stream,std::size_t bytes);
struct QsaRamWorkspaceProgress {std::size_t logical=0,physical=0;};
// A shared budget changes after each layer is reclaimed. Revisit all layers
// until the host allocation can proceed, rather than accepting a local fit.
void prepare_qsa_ram_workspace(std::size_t bytes,
    const std::function<QsaRamWorkspaceProgress()>& reclaim);
struct QsaRamMemory {
    const void* pool=nullptr;
    std::size_t payload=0,allocated=0,target=0,disk=0;
    std::size_t index_disk=0;
    std::array<std::uint64_t,4> tokens{};
    bool ssd_enabled=false;
    bool adaptive_precision=true;
    std::size_t pageable=0;
    std::size_t shared_gpu_cache=0;
};
struct QsaRamPrefillStage {
    mfq_tensor_backend::Tensor payload,views;
    std::size_t blocks=0,used_bytes=0;
};


class QsaRamCache {
public:
    using Tensor=mfq_tensor_backend::Tensor;
    using Encode=std::function<std::vector<tyloinfer::qsa::MicroblockImage>(
        const std::vector<tyloinfer::qsa::MicroblockImage>&,const std::vector<tyloinfer::qsa::Precision>&)>;
    QsaRamCache(int capacity,int rows,int selected,int bits,const mfq_tensor_backend::TensorOptions& options,const Tensor& live,
        bool ssd,const std::string& directory,Encode encode,bool adaptive_precision=true,bool prefer_host_metadata=false,
        bool use_spare_vram=false);
    ~QsaRamCache();
    // Use the cache's owning worker even after loader/request thread handoff.
    std::shared_ptr<void> hold_workspace(std::size_t bytes);
    QsaRamWorkspaceProgress reclaim_shared_workspace();
    // Run only after retiring the batch graph that reads these descriptors.
    std::size_t reclaim_pinned_for_admission(std::size_t bytes);
    void prepare(int queries=8,bool exact_width=false,bool allocate_shared_cache=true);
    void activate_shared_cache();
    // Fixed-cache graphs must be retired first. The spare-VRAM cache is
    // addressed through a device descriptor that stays valid when it is reclaimed.
    std::size_t release_shared_device_cache(bool spare_only=false);
    void begin_prefill_stage();
    void prepare_prefill_stage(int queries,std::uint64_t epoch=0);
    std::size_t fixed_prefill_ram_reservation(std::size_t target_blocks) const;
    void release_prefill_stage();
    void trim_stage_queries(int queries);
    Tensor prepared_views()const;
    bool collect_graph_selection(const Tensor& selected,int queries,int count,std::uint32_t* graph_control=nullptr);
    bool graph_task_supported()const;
    void prepare_graph_task(std::uint64_t epoch,int queries=0);
    Tensor stage_graph_task(const Tensor& selected,int queries,int count,const Tensor& live,const Tensor& positions,int appended);
    void append(std::vector<tyloinfer::qsa::MicroblockImage> images,std::uint64_t epoch,bool fresh=false);
    void drain();
    // Use the cache's owning worker even when execution moved from the loader
    // thread to the request thread. Called after reset, outside graph capture.
    void release_idle();
    bool discard_idle_descriptors();
    void bind_descriptors(const Tensor& live);
    Tensor stage(const Tensor& selected,int queries,int count,const Tensor& live,
        const Tensor& position,std::int64_t host_position,int appended,bool selection_ready=false,
        bool unique_selection=false);
    // Copy one layer's original encoded KV once for an entire prefill chunk.
    // Return no stage when current GPU workspace cannot hold its real bytes.
    std::optional<QsaRamPrefillStage> stage_prefill(std::size_t blocks,std::size_t reserve_blocks=0,
        const QsaRamPrefillStage* previous=nullptr);
    void observe(const std::vector<double>& hits,std::uint64_t epoch);
    void wait();
    void truncate(std::size_t blocks);
    bool graph_safe()const;
    bool encoding_urgent()const;
    std::shared_ptr<tyloinfer::qsa::RamKvSnapshot> snapshot(std::size_t blocks);
    // Hold all incoming layer targets before hydrating the first layer. The
    // lease restores the previous target even if a later layer fails to load.
    std::shared_ptr<void> reserve_restore(const tyloinfer::qsa::RamKvSnapshot& state,bool borrow_prefix=false);
    void enable_prefix_sharing();
    bool restore_can_place_in_ram(std::size_t bytes) const;
    void restore(const tyloinfer::qsa::RamKvSnapshot& state,bool borrow_prefix);
    void restore(const tyloinfer::qsa::RamKvSnapshot& state,bool borrow_prefix,
                 const std::function<void(std::size_t)>& block_read,
                 std::span<const QsaRamCache* const> peers={});
    tyloinfer::qsa::MicroblockImage reconstruction_image(std::size_t id);
    tyloinfer::qsa::AdaptiveCacheStats stats()const;
    QsaRamMemory memory()const;
    std::vector<Tensor> resident_tensors()const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace mfq::cuda

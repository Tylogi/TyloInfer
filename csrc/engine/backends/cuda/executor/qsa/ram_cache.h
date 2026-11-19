#pragma once
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/state/kv/qsa/adaptive_cache.h"
#include "engine/state/kv/qsa/ram_store.h"
#include <cuda_runtime_api.h>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace tyloi::cuda {
struct QsaRamMemory {
    const void* pool=nullptr;
    std::size_t payload=0,allocated=0,target=0,disk=0;
    std::size_t index_disk=0;
    std::array<std::uint64_t,4> tokens{};
    bool ssd_enabled=false;
};
struct QsaRamPrefillStage {
    ::tyloi::DenseTensor payload,views;
};


class QsaRamCache {
public:
    using Tensor=::tyloi::DenseTensor;
    using Encode=std::function<std::vector<tyloinfer::qsa::MicroblockImage>(
        const std::vector<tyloinfer::qsa::MicroblockImage>&,const std::vector<tyloinfer::qsa::Precision>&)>;
    QsaRamCache(int capacity,int rows,int selected,int bits,const ::tyloi::TensorOptions& options,const Tensor& live,
        bool ssd,const std::string& directory,Encode encode);
    ~QsaRamCache();
    // Use the cache's owning worker even after loader/request thread handoff.
    std::shared_ptr<void> hold_workspace(std::size_t bytes);
    const void* workspace_pool()const noexcept;
    // Reserve each worker first; pass one live cache per worker.
    static void prepare_workspace(std::span<QsaRamCache* const> pools,std::size_t bytes);
    void prepare();
    static std::pair<std::size_t,std::size_t> graph_growth(std::span<QsaRamCache* const> caches);
    std::vector<Tensor> graph_tensors() const;
    Tensor prepared_views()const;
    bool collect_graph_selection(const Tensor& selected,int queries,int count);
    bool graph_task_supported()const;
    void prepare_graph_task(std::uint64_t epoch);
    Tensor stage_graph_task(const Tensor& selected,int queries,int count,const Tensor& live,const Tensor& positions,int appended);
    void append(std::vector<tyloinfer::qsa::MicroblockImage> images,std::uint64_t epoch,bool fresh=false);
    void drain();
    // Use the cache's owning worker even when execution moved from the loader
    // thread to the request thread. Called after reset, outside graph capture.
    void release_idle();
    Tensor stage(const Tensor& selected,int queries,int count,const Tensor& live,
        const Tensor& position,std::int64_t host_position,int appended,bool selection_ready=false);
    // Copy one layer's original encoded KV once for an entire prefill chunk.
    // Return no stage when current GPU workspace cannot hold its real bytes.
    std::optional<QsaRamPrefillStage> stage_prefill(std::size_t blocks);
    void observe(const std::vector<double>& hits,std::uint64_t epoch);
    void wait();
    void truncate(std::size_t blocks);
    bool graph_safe()const;
    bool encoding_urgent()const;
    std::shared_ptr<tyloinfer::qsa::RamKvSnapshot> snapshot(std::size_t blocks);
    // Hold all incoming layer targets before hydrating the first layer. The
    // lease restores the previous target even if a later layer fails to load.
    std::shared_ptr<void> reserve_restore(const tyloinfer::qsa::RamKvSnapshot& state);
    void restore(const tyloinfer::qsa::RamKvSnapshot& state,bool borrow_prefix);
    void restore(const tyloinfer::qsa::RamKvSnapshot& state,bool borrow_prefix,
                 const std::function<void(std::size_t)>& block_read);
    tyloinfer::qsa::MicroblockImage reconstruction_image(std::size_t id);
    tyloinfer::qsa::AdaptiveCacheStats stats()const;
    QsaRamMemory memory()const;
    std::vector<Tensor> resident_tensors()const;
private:
    struct WorkspaceProgress {std::size_t logical=0,physical=0;};
    WorkspaceProgress reclaim_shared_workspace();
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace tyloi::cuda

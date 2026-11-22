#pragma once
#include "mfq_tensor_backend.h"
#include "tyloinfer/qsa_adaptive_cache.h"
#include "qsa_ram_cache.h"
#include <functional>
#include <memory>
#include <span>
#include <string>

struct CudaProfiler;
namespace mfq::cuda {
struct QsaTqSession;
// Auto uses RAM precision tiers; numeric modes retain fixed precision.
std::string qsa_cache_mode();
bool qsa_tq_cache_enabled();
class QsaTqCache {
public:
    using Tensor=mfq_tensor_backend::Tensor;
    using Pool=std::function<Tensor(const Tensor&,const Tensor&)>;
    using CompletePool=std::function<Tensor(const Tensor&,const Tensor&,const Tensor&)>;
    QsaTqCache(int64_t maximum,int heads,int kv_heads,int width,int index_heads,int index_width,int budget,
               const mfq_tensor_backend::TensorOptions& options,bool ssd_cache=false,bool prefer_host_indexer=false,
               bool prefer_host_metadata=false,bool use_spare_vram=false);
    ~QsaTqCache();
    int64_t position() const;
    void reset();
    void truncate(int64_t keep);
    void wait();
    // Retire completed GPU append owners without joining background codecs.
    void collect();
    // Indexer buffers can be released after all graphs that use them retire.
    void release_idle(bool release_indexer=false);
    void ensure_descriptors();
    std::size_t discard_idle_descriptors();
    std::size_t indexer_storage_bytes() const;
    std::size_t prefill_scratch_row_bytes() const;
    std::size_t indexer_growth_bytes() const;
    void reserve_indexer();
    bool place_unallocated_indexer_in_ram(bool reserved_restore=false);
    std::size_t migrate_indexer_to_ram();
    std::size_t migrate_indexer_to_device(std::size_t decode_workspace);
    bool indexer_migrated_to_ram() const;
    std::size_t minimum_recall_workspace_bytes() const;
    std::size_t append_attention_workspace_bytes(int64_t tokens) const;
    std::size_t append_packed_history_bytes(int64_t tokens) const;
    std::size_t fixed_prefill_ram_reservation(int64_t target_tokens) const;
    std::size_t codec_workspace_growth_bytes(std::size_t prefix_tokens) const;
    void set_budget(std::size_t bytes);
    // Attention scratch follows the live allocator budget independently of FFN.
    int64_t prefill_rows(int64_t requested) const;
    void begin_prefill_workspace(int64_t end_position);
    void release_prefill_workspace();
    void trim_ram_stage_queries(int queries);
    // Low bits: partial M1, complete M1, or M2..8. Bit 2 requests host
    // staging boundaries; a model applies it to all its QSA layers so the
    // captured boundary count remains stable as individual layers spill.
    int device_graph_kind(int64_t tokens) const;
    bool pending_graph_enabled() const;
    void prepare_decode(bool allow_pending_graph=false);
    void prepare_decode_tokens(int64_t tokens,bool allow_pending_graph=false,int host_mode=0,
        bool share_attention_workspace=false);
    bool graph_task_supported()const;
    void advance_graph(int64_t tokens);
    void activate_graph_cache();
    std::size_t release_shared_device_cache(bool spare_only=false);
    // An empty output appends history without index search or attention.
    Tensor forward_pending_graph(const Tensor& query,const Tensor& key,const Tensor& value,const Tensor& iq,
        const Tensor& raw_index,const Tensor& cache_positions,const Tensor& gate,Tensor output,
        const Pool& pool={},const Tensor& absolute_positions={},const CompletePool& complete_pool={});
    Tensor forward(const Tensor& query,const Tensor& keys,const Tensor& values,const Tensor& iq,
                   const Tensor& raw_index,const Tensor& positions,const Pool& pool,bool history_only=false,
                   const Tensor& gate={},Tensor output={},bool fp16_math=true,CudaProfiler* profiler=nullptr);
    tyloinfer::qsa::AdaptiveCacheStats stats() const;
    // Observe I/O without finishing pending history or changing cache policy.
    std::uint64_t ssd_read_bytes() const;
    std::optional<QsaRamMemory> ram_memory()const;
    std::shared_ptr<void> hold_host_workspace(std::size_t bytes);
    QsaRamWorkspaceProgress reclaim_shared_host_workspace();
    std::size_t reclaim_pinned_for_admission(std::size_t bytes);
    std::vector<Tensor> resident_tensors()const;
    bool supports_session()const;
    std::shared_ptr<QsaTqSession> capture_session(std::int64_t tokens);
    std::shared_ptr<void> reserve_session_restore(const QsaTqSession& state);
    void enable_prefix_sharing();
    bool restore_can_place_in_ram(std::size_t bytes) const;
    void restore_session(const QsaTqSession& state,bool borrow_prefix,std::span<const QsaTqCache* const> peers={});
private:
    std::size_t migrate_indexer(bool host,std::size_t decode_workspace);
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace mfq::cuda

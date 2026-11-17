#pragma once
#include "cuda_execution.h"
#include "tyloinfer/model_source.h"
#include "tyloinfer/kernels/cuda/tq_ops.h"
#include "runtime/row_stage.h"
#include <array>
#include <functional>

namespace mfq {class ModelSource;class TqRows;}
class TqGemmResources;
struct TqWeight {
    // Keep execution streams alive until image storage has fenced its users.
    std::shared_ptr<TqGemmResources> gemm;
    mfq_tensor_backend::Tensor image;
    std::optional<MfqCudaStream> allocation_stream;
    TrellisProjectionView view{};
    std::int64_t out=0,neuron_len=0;
    TqWeight with_image(mfq_tensor_backend::Tensor storage)const;
    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor)const;
    bool can_forward_prefill_residual(std::int64_t tokens,const mfq_tensor_backend::Tensor& residual)const;
    mfq_tensor_backend::Tensor forward_prefill_residual(mfq_tensor_backend::Tensor input,
        const mfq_tensor_backend::Tensor& residual)const;
    mfq_tensor_backend::Tensor try_residual(mfq_tensor_backend::Tensor input,
        const mfq_tensor_backend::Tensor& normalized,std::int64_t streams,bool injection=false)const;
    std::vector<mfq_tensor_backend::Tensor> try_residual_down(mfq_tensor_backend::Tensor input,
        const mfq_tensor_backend::Tensor& injection,std::int64_t streams)const;
    mfq_tensor_backend::Tensor forward_groupwise(mfq_tensor_backend::Tensor,std::int64_t groups)const;
    mfq_tensor_backend::Tensor embedding(mfq_tensor_backend::Tensor)const;
};
TqWeight to_device_tq(const std::vector<std::uint8_t>&,std::int64_t input_columns,int device,bool cuda=true);
std::array<std::int64_t,2> tq_projection_shape(const mfq::ModelSource&,const std::string&);
TqWeight load_tq_weight(const mfq::ModelSource&,const std::string&,int device,bool cuda=true);
bool is_tq_row_table(const mfq::ModelSource&,const std::string&);

// Device row plan only; compressed weights retain their original storage.
class TqProjectionGroup {
    std::vector<TqWeight> weights_;
    std::vector<TrellisProjectionJob> host_plan_;
    mfq_tensor_backend::Tensor plan_;
    std::int64_t input_=0,output_=0;
    std::size_t joined_prefix_=1;
    std::int64_t joined_rows_=0;
    int blocks_=0;
public:
    TqProjectionGroup()=default;
    explicit TqProjectionGroup(std::vector<TqWeight>);
    void join_prefix(std::size_t count);
    bool defined()const noexcept{return !weights_.empty();}
    std::vector<mfq_tensor_backend::Tensor> forward(mfq_tensor_backend::Tensor)const;
    mfq_tensor_backend::Tensor swiglu(mfq_tensor_backend::Tensor)const;
};

// Table rows remain in their source. Only requested rows enter device memory.
class TqRowTable {
    struct RowCache;
    std::shared_ptr<RowCache> row_cache_;
    std::vector<std::shared_ptr<mfq::TqRows>> segments_;
    std::vector<std::uint64_t> ends_;
    struct SourceReader {
        mfq::ModelSource::TensorReader read;
        mfq::ModelSource::TensorBatchReader batch;
    };
    std::vector<SourceReader> readers_;
    std::vector<std::size_t> origins_;
    std::int64_t columns_=0;
    std::uint64_t storage_bytes_=0;
    bool parallel_reads_=true;
public:
    TqRowTable(const mfq::ModelSource&,const std::string&);
    std::uint64_t rows()const noexcept{return ends_.back();}
    std::int64_t width()const noexcept{return columns_;}
    std::uint64_t storage_bytes()const noexcept{return storage_bytes_;}
    bool parallel_reads()const noexcept{return parallel_reads_;}
    std::size_t routed_image_storage_bound(std::uint64_t experts)const;
    std::size_t expert_image_bytes(std::uint64_t expert,std::uint64_t rows_per_expert)const;
    void cache_row_metadata();
    TrellisProjectionView stage_row(std::uint64_t,std::uint8_t*,std::uint8_t*,std::size_t)const;
    std::size_t row_image_bytes_bound(std::uint64_t count=1)const;
    std::vector<std::uint8_t> slice_rows(std::uint64_t begin,std::uint64_t end)const;
    std::vector<std::vector<std::uint8_t>> slice_experts(
        const std::vector<std::int64_t>& experts,std::uint64_t rows_per_expert)const;
    mfq_tensor_backend::Tensor lookup(const std::int64_t*,std::size_t,const mfq_tensor_backend::Device&)const;
    mfq_tensor_backend::Tensor lookup(mfq_tensor_backend::Tensor)const;
};
mfq_tensor_backend::Tensor lookup_tq_shards(const std::vector<std::shared_ptr<TqRowTable>>&,
    std::int64_t rows_per_shard,const std::vector<std::int64_t>& ids,
    const mfq_tensor_backend::Device&);
std::shared_ptr<RowStage> make_tq_row_stage(const std::vector<std::shared_ptr<TqRowTable>>&,
    std::int64_t rows_per_shard,const std::vector<std::int64_t>& shape,
    const mfq_tensor_backend::Device&);
std::function<mfq_tensor_backend::Tensor()> prefetch_tq_shards(const std::vector<std::shared_ptr<TqRowTable>>&,
    std::int64_t rows_per_shard,const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape,
    const mfq_tensor_backend::Device&);

// Immutable predictor experts keep one encoded allocation and a fixed device
// descriptor table alive independently of the evictable routed cache.
struct TqResidentBank {
    // Destruct encoded storage before releasing its GEMM/stream resources.
    std::shared_ptr<TqGemmResources> gemm;
    mfq_tensor_backend::Tensor packed,views;
    std::vector<TqWeight> experts;
};
struct TqRoutedWeight {
    std::shared_ptr<TqRowTable> table;
    std::shared_ptr<TqExpertCache> cache;
    std::shared_ptr<TqResidentBank> resident;
    std::uint64_t source_id=0;
    std::int64_t experts=0,out=0,input=0;
    TqRoutedWeight(CudaExecutionContext&,const mfq::ModelSource&,const std::string&,
                   std::int64_t experts,std::int64_t out,std::int64_t input);
    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor,mfq_tensor_backend::Tensor)const;
};
std::shared_ptr<TqRoutedWeight> load_tq_routed_weight(CudaExecutionContext&,
    const mfq::ModelSource&,const std::string&);
bool prepare_tq_graph_pool(const std::shared_ptr<TqExpertCache>&);
void wait_tq_expert_admission(const std::shared_ptr<TqExpertCache>&);
void finalize_tq_expert_cache(const std::shared_ptr<TqExpertCache>&,std::size_t growth);
bool tq_graph_supports_tokens(std::int64_t tokens)noexcept;
std::size_t tq_expert_cache_budget_bytes(const std::shared_ptr<TqExpertCache>&);
std::size_t tq_expert_source_bytes(const std::shared_ptr<TqExpertCache>&);
std::uint64_t tq_expert_streamed_bytes(const std::shared_ptr<TqExpertCache>&);
std::array<double,2> tq_expert_transfer_timings(const std::shared_ptr<TqExpertCache>&);
void resize_tq_expert_cache(const std::shared_ptr<TqExpertCache>&,std::size_t bytes);
void resize_tq_expert_host(const std::shared_ptr<TqExpertCache>&,std::size_t bytes);
std::size_t tq_prefill_lendable_bytes(const std::shared_ptr<TqExpertCache>&);
void reserve_tq_expert_workspace(const std::shared_ptr<TqExpertCache>&,std::size_t bytes);
void constrain_tq_capture_workspace(const std::shared_ptr<TqExpertCache>&,std::size_t bytes);
void restore_tq_decode_workspace(const std::shared_ptr<TqExpertCache>&,std::size_t growth,
    std::optional<std::size_t> transient=std::nullopt);
void restore_tq_decode_workspace_after_capture(const std::shared_ptr<TqExpertCache>&);
bool set_tq_reuse_learning(const std::shared_ptr<TqExpertCache>&,bool enabled);
void begin_tq_speculative_routes(const std::shared_ptr<TqExpertCache>&,int rows);
void commit_tq_speculative_routes(const std::shared_ptr<TqExpertCache>&,int rows);
void discard_tq_speculative_routes(const std::shared_ptr<TqExpertCache>&) noexcept;
class TqPrefillWorkspaceLoan {
    std::shared_ptr<TqExpertCache> cache_;
    std::uint64_t token_=0;
public:
    explicit TqPrefillWorkspaceLoan(std::shared_ptr<TqExpertCache> cache):cache_(std::move(cache)) {}
    TqPrefillWorkspaceLoan(const TqPrefillWorkspaceLoan&)=delete;
    TqPrefillWorkspaceLoan& operator=(const TqPrefillWorkspaceLoan&)=delete;
    ~TqPrefillWorkspaceLoan();
    std::size_t begin(std::size_t bytes);
    void suspend();
    void finish();
};
std::size_t reclaim_tq_expert_host(const std::shared_ptr<TqExpertCache>&,std::size_t bytes);
std::size_t reclaim_tq_expert_duplicates(const std::shared_ptr<TqExpertCache>&,std::size_t bytes);
std::size_t reserve_tq_expert_host(const std::shared_ptr<TqExpertCache>&,std::size_t allocation_bytes);
std::size_t share_tq_expert_host(const std::shared_ptr<TqExpertCache>&,std::size_t allocation_bytes);
std::vector<std::pair<std::string,double>> tq_expert_memory_metrics(const std::shared_ptr<TqExpertCache>&);

// Three ordinary [experts, rows, columns] TQ tensors. Expert ID selects
// a row range; every row retains its own bit width, scale and encoded data.
struct TqMoeBanks {
    std::shared_ptr<TqRoutedWeight> gate,up,down;
    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor ids,mfq_tensor_backend::Tensor weights,float swiglu_limit=0,
        CudaProfiler* profiler=nullptr,std::int64_t expert_begin=0,std::int64_t expert_end=-1,
        const std::function<void()>& independent_work={},CudaIndependentBranchExecutor* shared_executor=nullptr)const;
    mfq_tensor_backend::Tensor forward_parallel(CudaExecutionContext&,
        mfq_tensor_backend::Tensor input,mfq_tensor_backend::Tensor ids,
        mfq_tensor_backend::Tensor weights,float swiglu_limit=0,const std::function<void()>& independent_work={},CudaIndependentBranchExecutor* shared_executor=nullptr)const;
};

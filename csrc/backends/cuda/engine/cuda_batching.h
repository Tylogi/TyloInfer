#pragma once

#include "continuous_batch.h"
#include "cuda_execution.h"
#include "cuda_runtime_config.h"
#include "cuda_sampling.h"
#include "models/hy4_flash/batch_state.h"
#include "models/inkling2/batch_state.h"
#include "models/qwen35/batch_state.h"
#include "models/qwen4_exp/mtp.h"
#include "models/qwen4_exp/ops.h"
#include <map>
#include <type_traits>
#include <variant>

struct Qwen35Mtp;

namespace mfq::cuda {

bool qwen_continuous_batch_cuda_graph_enabled(const Qwen35CausalLm &,
                                              const CudaContinuousBatchConfig &);
struct QwenContinuousDecodeGraph;

struct CudaBatchMtpState {
    std::unique_ptr<MtpModule> predictor;
    mfq_tensor_backend::Tensor predictor_hidden;
    std::vector<int64_t> verified_tokens;
    size_t verified_drafts=0;
    mfq::engine::mtp::GenerationStats mtp_stats;
    int64_t first_token=0;
};

struct BatchLogitCandidates {
    mfq_tensor_backend::Tensor values,indices;
    int32_t best(int64_t row) const;
    mfq::engine::mtp::CompactDistribution distribution(int64_t row,const MfqSamplingParams&) const;
};
BatchLogitCandidates rank_cuda_batch_logits(mfq_tensor_backend::Tensor logits,int64_t top_k);

mfq::engine::Generation publish_cuda_batch_mtp(std::shared_ptr<mfq::engine::BatchRequest>,CudaBatchMtpState&,
    mfq::engine::InferenceOutput&,std::function<void()>,std::function<bool()>,std::function<void()>);

struct QwenBatchRequest : mfq::engine::BatchRequest,CudaBatchMtpState {
    using BatchRequest::BatchRequest;
    std::optional<Sampler> sampler;
    mfq_tensor_backend::Tensor counts;
    mfq_tensor_backend::Tensor prefill_ids;
    std::vector<std::unique_ptr<internal::PrefillCudaTimer>> prefill_timers;
    qwen35::QwenBatchRequestState cache;
};

// Native operations consumed directly by engine::ContinuousBatch.
class QwenBatchOperations {
  public:
    using Tensor = mfq_tensor_backend::Tensor;
    using Request = QwenBatchRequest;
    using State = mfq::engine::BatchState<Request>;
    struct Sample {
        Tensor next;
        int64_t token = 0;
        MfqPrefillTiming timing;
    };
    struct Decoded {
        Tensor logits, greedy_tokens;
    };

    QwenBatchOperations(Qwen35CausalLm &, CudaExecutionContext &, CudaContinuousBatchConfig,const Qwen35Mtp* =nullptr);
    ~QwenBatchOperations();
    QwenBatchOperations(const QwenBatchOperations &) = delete;
    QwenBatchOperations &operator=(const QwenBatchOperations &) = delete;
    int64_t vocab_size() const;
    int64_t max_context() const;
    mfq::engine::Metrics metrics() const;
    void suspend_decode();
    mfq::engine::PrefillChunk prefill_chunk(const std::shared_ptr<Request>&, mfq::engine::PrefillChunk);
    Sample prefill(const std::shared_ptr<Request> &, mfq::engine::PrefillChunk);
    void activate(const std::shared_ptr<Request> &, const Sample &);
    void resume_decode(int64_t cache_position);
    void discard_prefill(const std::shared_ptr<Request> &);
    void retire(const std::vector<std::shared_ptr<Request>> &, int64_t cache_position);
    std::optional<mfq::engine::Generation> generate_decode(const std::shared_ptr<Request>&,
        mfq::engine::InferenceOutput&,std::function<void()>,std::function<bool()>,std::function<void()>);
    Decoded decode(State &);
    Sample sample(const std::shared_ptr<Request> &, const Decoded &);
    void accept(const std::shared_ptr<Request> &, const Sample &);
    void recover(State &);

  private:
    void initialize_sampling(Request &, const Tensor &prompt_ids);
    void ensure_decode_metadata_buffers(int primary);
    void invalidate_decode_graph();

    Decoded verify_mtp(State&);
    struct MtpGraph;
    const Qwen35Mtp* predictor_;
    std::map<std::pair<int64_t,int64_t>,std::unique_ptr<MtpGraph>> mtp_graphs_;
    int64_t mtp_batches_=0,mtp_maximum_batch_=0;

    Qwen35CausalLm &model_;
    qwen35::Qwen35BatchStateAdapter state_adapter_;
    CudaExecutionContext &execution_;
    const CudaContinuousBatchConfig config_;
    int32_t max_sequences_ = 0;
    bool moe_enabled_ = false;
    bool cached_moe_enabled_ = false;
    int64_t decode_batches_{0};
    int64_t decode_tokens_{0};
    int64_t batched_greedy_batches_{0};
    int64_t packed_metadata_batches_{0};
    int64_t cuda_graph_captures_{0};
    int64_t cuda_graph_replays_{0};
    Tensor decode_metadata_host_;
    Tensor decode_metadata_cuda_;
    std::vector<std::unique_ptr<QwenContinuousDecodeGraph>> decode_graphs_;
};

struct FlashNextBatchRequest : mfq::engine::BatchRequest {
    FlashNextBatchRequest(std::string id, const mfq::engine::InferenceRequest& input)
        : BatchRequest(std::move(id), input), cache_plan(input.cache_plan) {}
    MfqPromptCachePlan cache_plan;
    std::size_t reused_prefix = 0, prefix_boundary = 0;
    bool prefix_stored = false;
    std::optional<Sampler> sampler;
    mfq_tensor_backend::Tensor counts, prefill_ids;
    std::vector<std::unique_ptr<internal::PrefillCudaTimer>> prefill_timers;
    std::unique_ptr<Qwen4BatchRequestState> cache;
    std::unique_ptr<qwen4_exp::Qwen4ExpMtp> predictor;
    mfq_tensor_backend::Tensor predictor_hidden;
    struct Verification {
        std::vector<int64_t> emitted;
        mfq_tensor_backend::Tensor ids, hidden;
        std::size_t accepted = 0;
    };
    std::optional<Verification> verification;
    mfq::engine::mtp::GenerationStats mtp_stats;
    int64_t first_token = 0;
    int64_t decode_row = -1;
    int64_t prefill_compute_chunk = 0;
    double packed_prefill_ms = 0;
};
class FlashNextBatchOperations {
  public:
    using Tensor = mfq_tensor_backend::Tensor;
    using Request = FlashNextBatchRequest;
    using State = mfq::engine::BatchState<Request>;
    using Sample = QwenBatchOperations::Sample;
    struct Decoded { Tensor logits, greedy_tokens; };
    FlashNextBatchOperations(Qwen4CausalLm&, CudaExecutionContext&, CudaContinuousBatchConfig,
        const qwen4_exp::Qwen4ExpMtp* predictor=nullptr);
    ~FlashNextBatchOperations();
    static bool supports_mtp(const MfqSamplingParams&,const MfqTokenConstraintPtr&);
    void set_prefix_cache(internal::TextSessionCache& cache) { prefix_cache_ = &cache; }
    std::size_t prepare(const std::shared_ptr<Request>&);
    std::size_t prefix_tokens(const std::shared_ptr<Request>& request) const { return request->prefix_boundary; }
    void commit_prefix(const std::shared_ptr<Request>&);
    int64_t vocab_size() const;
    int64_t max_context() const;
    mfq::engine::Metrics metrics() const;
    void suspend_decode();
    mfq::engine::PrefillChunk prefill_chunk(const std::shared_ptr<Request>&, mfq::engine::PrefillChunk);
    Sample prefill(const std::shared_ptr<Request>&, mfq::engine::PrefillChunk);
    int64_t solo_prefill_span(const std::shared_ptr<Request>&,int64_t compute_chunk) const;
    std::size_t prefill_batch_limit() const;
    int64_t prefill_batch_token_budget() const;
    std::vector<mfq::engine::PrefillBatchResult<Sample>> prefill_batch(
        const std::vector<std::pair<std::shared_ptr<Request>,mfq::engine::PrefillChunk>>&);
    void activate(const std::shared_ptr<Request>&, const Sample&);
    int64_t cache_length(const std::shared_ptr<Request>& request) const { return request->cache->position(); }
    std::optional<mfq::engine::Generation> generate_decode(const std::shared_ptr<Request>&,
        mfq::engine::InferenceOutput&,std::function<void()>,std::function<bool()>,std::function<void()>);
    void resume_decode(int64_t);
    void discard_prefill(const std::shared_ptr<Request>&);
    void retire(const std::vector<std::shared_ptr<Request>>&, int64_t);
    Decoded decode(State&);
    Sample sample(const std::shared_ptr<Request>&, const Decoded&);
    void accept(const std::shared_ptr<Request>&, const Sample&);
    void recover(State&);
    static constexpr bool isolate_prefill_failures = true;
    static constexpr bool isolate_context_growth_failures = true;
  private:
    mfq::engine::Generation generate_mtp(std::shared_ptr<Request>,mfq::engine::InferenceOutput&,
        std::function<void()>,std::function<bool()>,std::function<void()>);
    Decoded verify_mtp(State&);
    void initialize_prefill(const std::shared_ptr<Request>&,mfq::engine::PrefillChunk);
    bool layer_major_prefill(const Request&, int64_t tokens) const;
    std::size_t release_spare_qsa_cache();
    void borrow_dense_prefill_workspace(std::size_t bytes,int64_t tokens);
    std::size_t reclaim_indexer_for_admission(std::size_t bytes);
    std::size_t reclaim_pinned_for_admission(const Request* incoming,std::size_t bytes);
    Sample finish_prefill(const std::shared_ptr<Request>&,Tensor next);
    Qwen4CausalLm& model_;
    CudaExecutionContext& execution_;
    CudaContinuousBatchConfig config_;
    const qwen4_exp::Qwen4ExpMtp* predictor_ = nullptr;
    internal::TextSessionCache* prefix_cache_ = nullptr;
    std::vector<std::weak_ptr<Request>> requests_;
    std::unique_ptr<Qwen4BatchDecodeGraph> graph_;
    bool routed_workspace_restore_pending_ = false;
    int64_t decode_batches_ = 0, decode_tokens_ = 0, greedy_batches_ = 0;
    int64_t packed_prefill_batches_ = 0, packed_prefill_requests_ = 0, packed_prefill_tokens_ = 0;
    int64_t layer_major_prefill_chunks_ = 0;
    int64_t mtp_batches_ = 0, mtp_maximum_batch_ = 0;
};
mfq::engine::PrefillChunk plan_stateful_batch_prefill(CudaExecutionContext&,
    mfq::engine::PrefillChunk,std::size_t row_bytes,const std::function<void()>& release_decode_graphs);

class DecodeWindow;
class GraphPoolLease;

template<class Cache> struct StatefulCudaBatchRequest : mfq::engine::BatchRequest,CudaBatchMtpState {
    StatefulCudaBatchRequest(std::string id,const mfq::engine::InferenceRequest& input)
        : BatchRequest(std::move(id),input),
          prefix_limit(input.cache_plan.stable_prefix_tokens) {}
    std::optional<Sampler> sampler;
    mfq_tensor_backend::Tensor ids,counts;
    Cache cache;
    double prefill_ms=0;
    int64_t slot=-1;
    std::size_t prefix_limit=0;
    bool prefix_saved=false;
};

struct InklingBatchAdapter {
    using Cache=inkling::BatchCache;
    using Verification=inkling::BatchVerification;
    static constexpr auto set_verification=inkling::set_batch_verification;
    static constexpr auto take_verification=inkling::take_batch_verification;
    static constexpr auto take=inkling::take_cache;
    static constexpr auto bind=inkling::bind_cache;
    static constexpr auto reserve=inkling::reserve_slots;
    static constexpr auto assign=inkling::copy_to_slot;
    static constexpr auto save=inkling::save_convolution;
    static constexpr auto restore=inkling::restore_convolution;
};

struct Hy4BatchAdapter {
    using Cache=hy4::BatchCache;
    using Verification=hy4::BatchVerification;
    static constexpr auto set_verification=hy4::set_batch_verification;
    static constexpr auto take_verification=hy4::take_batch_verification;
    static constexpr auto take=hy4::take_cache;
    static constexpr auto bind=hy4::bind_cache;
    static constexpr auto reserve=hy4::reserve_slots;
    static constexpr auto assign=hy4::copy_to_slot;
    static constexpr auto save=hy4::save_pending_groups;
    static constexpr auto restore=hy4::restore_pending_groups;
};

template<class Model,class Adapter> class StatefulCudaBatchOperations {
public:
    using Tensor=mfq_tensor_backend::Tensor;
    using Cache=typename Adapter::Cache;
    using Request=StatefulCudaBatchRequest<Cache>;
    using State=mfq::engine::BatchState<Request>;
    using Sample=QwenBatchOperations::Sample;
    using Decoded=QwenBatchOperations::Decoded;
    StatefulCudaBatchOperations(Model&,CudaExecutionContext&,CudaContinuousBatchConfig,const MtpModule* predictor=nullptr);
    ~StatefulCudaBatchOperations();
    int64_t vocab_size() const;
    int64_t max_context() const;
    mfq::engine::Metrics metrics() const;
    void set_prefix_cache(internal::TextSessionCache& cache) {prefix_cache_=&cache;}
    std::size_t prepare(const std::shared_ptr<Request>&);
    std::size_t prefix_tokens(const std::shared_ptr<Request>& request) const {return request->prefix_limit;}
    void commit_prefix(const std::shared_ptr<Request>&);
    Sample prefill(const std::shared_ptr<Request>&,mfq::engine::PrefillChunk);
    void activate(const std::shared_ptr<Request>&,const Sample&);
    void discard_prefill(const std::shared_ptr<Request>&);
    void suspend_decode();
    void resume_decode(int64_t cache_position);
    void retire(const std::vector<std::shared_ptr<Request>>&,int64_t cache_position);
    Decoded decode(State&);
    Sample sample(const std::shared_ptr<Request>&,const Decoded&);
    void accept(const std::shared_ptr<Request>&,const Sample&);
    void recover(State&);
    std::optional<mfq::engine::Generation> generate_decode(const std::shared_ptr<Request>&,mfq::engine::InferenceOutput&,
        std::function<void()>,std::function<bool()>,std::function<void()>);
    static constexpr bool isolate_prefill_failures=true;
private:
    Decoded verify_speculative(State&);
    void clear_slots();
    Model& model_;
    CudaExecutionContext& execution_;
    CudaContinuousBatchConfig config_;
    const MtpModule* predictor_;
    internal::TextSessionCache* prefix_cache_=nullptr;
    std::vector<Request*> slots_;
    Cache decode_cache_;
    bool decode_bound_=false,restore_workspace_=false;
    int64_t planned_length_=0;
    int64_t verification_width_=1;
    int64_t decode_steps_=0,decode_tokens_=0,graph_captures_=0,graph_replays_=0;
    Tensor metadata_host_,metadata_,graph_logits_;
    [[no_unique_address]] std::conditional_t<std::is_same_v<Model,Hy4CausalLm>,hy4::EngramRows,std::monostate> staged_;
    std::shared_ptr<GraphPoolLease> graph_pool_;
    std::unique_ptr<DecodeWindow> graph_;
    struct SpeculativeGraph;
    std::map<int64_t,std::unique_ptr<SpeculativeGraph>> speculative_graphs_;
    int64_t speculative_batches_=0,speculative_maximum_batch_=0;
};

using InklingBatchOperations=StatefulCudaBatchOperations<InklingCausalLm,InklingBatchAdapter>;
using Hy4BatchOperations=StatefulCudaBatchOperations<Hy4CausalLm,Hy4BatchAdapter>;
template<class Model> using CudaBatchOperations = std::conditional_t<
    std::is_same_v<Model,Qwen4CausalLm>,FlashNextBatchOperations,
    std::conditional_t<std::is_same_v<Model,InklingCausalLm>,InklingBatchOperations,
    std::conditional_t<std::is_same_v<Model,Hy4CausalLm>,Hy4BatchOperations,QwenBatchOperations>>>;
} // namespace mfq::cuda

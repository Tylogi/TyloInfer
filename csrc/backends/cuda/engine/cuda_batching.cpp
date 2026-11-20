#include "mfq_cuda_sampling_ops.h"
#include "mfq_cuda_resident_allocations.h"
#include "cuda_batching.h"

#include "cuda_execution.h"
#include "core/decode_graph.h"
#include "core/full_block.h"
#include "models/qwen35/ops.h"
#include "storage/moe_expert_cache.h"
#include "storage/text_session_cache.h"
#include "storage/session_state.h"
#include "runtime/execution_options.h"
#include "runtime/qsa_ram_cache.h"
#include <set>
#include "runtime/routed_cache_resources.h"
#include "prefill_budget.h"

#include <algorithm>
#include <exception>
#include <cstdio>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda {

using internal::PrefillCudaTimer;
using Tensor = mfq_tensor_backend::Tensor;
using qwen35::Qwen35BatchStateAdapter;
using qwen35::QwenBatchRequestState;

template<class Model>
static std::size_t batch_prefill_row_bytes(const Model& model) {
    const auto residual=std::size_t(model.hidden_size())*model.hc_mult()*sizeof(float)*11;
    if constexpr (std::is_same_v<Model,Qwen4CausalLm>) {
        if(!model.config.experts) {
            return qwen4_dense_prefill_workspace(model,0,true).row_bytes;
        }
    }
    return residual;
}

template<class Model>
static std::size_t batch_prefill_workspace_bytes(Model& model,CudaExecutionContext& execution,
        int64_t tokens,std::size_t growth=0) {
    const auto transfer=routed_prefill_workspace_budget(execution,false).transfer_bytes;
    auto reserve=std::size_t(1)<<30;
    if constexpr (std::is_same_v<Model,Qwen4CausalLm>) {
        if(!model.config.experts)reserve=0;
    }
    const auto fixed=mfq::engine::prefill_budget_add(mfq::engine::prefill_budget_add(reserve,growth),transfer);
    // Residual inputs and assembled outputs survive projection tiles. Hold
    // their space throughout prefill so expert admission cannot consume it.
    return mfq::engine::PrefillWorkspace{fixed,batch_prefill_row_bytes(model)}.bytes(tokens);
}

template<class Model>
static mfq::engine::PrefillChunk budget_batch_prefill(Model& model,
        CudaExecutionContext& execution,mfq::engine::PrefillChunk chunk,std::size_t growth=0) {
    const auto context=default_context(execution.layer_placement.primary_device());
    const auto fixed=batch_prefill_workspace_bytes(model,execution,0,growth);
    const auto row_bytes=batch_prefill_row_bytes(model);
    const auto wanted=batch_prefill_workspace_bytes(model,execution,chunk.count,growth);
    if(context->available_memory()+context->workspace_available()<wanted)
        reserve_routed_workspace(execution,wanted);
    auto available=context->available_memory()+context->workspace_available();
    auto selected=mfq::engine::select_prefill_chunk(chunk.count,chunk.count,available,fixed,row_bytes);
    if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
        if(!selected && !model.config.experts) {
            // Prefix hydration leaves free pages in the CUDA pool. Return
            // those pages before rejecting a short append under a hard cap.
            context->reclaim_idle_memory();
            available=context->available_memory()+context->workspace_available();
            if(available>fixed)
                selected=std::min<std::size_t>(chunk.count,(available-fixed)/row_bytes);
        }
    }
    if(!selected)throw mfq::engine::InferenceInputError(
        mfq::engine::InferenceInputErrorCode::ContextLength,
        "insufficient CUDA workspace for batched prefill; reduce concurrent requests");
    chunk.count=selected;
    return chunk;
}

mfq::engine::PrefillChunk QwenBatchOperations::prefill_chunk(
        const std::shared_ptr<Request>&,mfq::engine::PrefillChunk chunk) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    return budget_batch_prefill(model_,execution_,chunk);
}

mfq::engine::PrefillChunk FlashNextBatchOperations::prefill_chunk(
        const std::shared_ptr<Request>& request,mfq::engine::PrefillChunk chunk) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    initialize_prefill(request,chunk);
    // The restored dense path can lend weights before choosing its tile.
    if(!model_.config.experts && request->reused_prefix)return chunk;
    return budget_batch_prefill(model_,execution_,chunk,request->cache->prefill_growth_bytes(chunk.count));
}

bool qwen_continuous_batch_cuda_graph_enabled(const Qwen35CausalLm &model,
                                              const CudaContinuousBatchConfig &config) {
    const auto &execution = *model.execution;
    return config.cuda_graph && !Qwen35BatchStateAdapter::has_cached_moe(model) &&
           mfq_cuda_graph_capture_supported() && model_parallel_cuda_graph_enabled(execution);
}

class MoeContinuousBatchCacheScope {
  public:
    MoeContinuousBatchCacheScope(CudaExecutionContext &execution, bool enabled)
        : execution_(execution), previous_(execution.continuous_batch_cache_serial) {
        execution_.continuous_batch_cache_serial = enabled;
    }

    ~MoeContinuousBatchCacheScope() { execution_.continuous_batch_cache_serial = previous_; }

    MoeContinuousBatchCacheScope(const MoeContinuousBatchCacheScope &) = delete;
    MoeContinuousBatchCacheScope &operator=(const MoeContinuousBatchCacheScope &) = delete;

  private:
    CudaExecutionContext &execution_;
    bool previous_ = false;
};

struct QwenContinuousDecodeGraph {
    decltype(mfq_get_stream_from_pool(false)) stream;
    std::vector<MfqCudaStream> compute_streams;
    std::unique_ptr<MfqCudaGraph> graph;
    Tensor static_next;
    std::vector<const void *> state_addresses;
    int64_t batch = 0;
    int64_t planned_len = 0;
    bool valid = false;

    QwenContinuousDecodeGraph() : stream(mfq_get_stream_from_pool(false)) {}

    void ensure_compute_streams(const ParallelConfig &parallel) {
        if (compute_streams.empty()) {
            compute_streams = make_cuda_graph_compute_streams(stream, parallel);
        }
    }

    std::vector<MfqCudaStream>
    participant_streams(const ModelParallelCollectiveRuntime &collectives) const {
        return cuda_graph_participant_streams(compute_streams, collectives);
    }

    bool matches(int64_t candidate_batch, int64_t candidate_len,
                 const std::vector<const void *> &candidate_addresses) const {
        return valid && batch == candidate_batch && planned_len == candidate_len &&
               state_addresses == candidate_addresses;
    }

    void set_key(int64_t candidate_batch, int64_t candidate_len,
                 std::vector<const void *> candidate_addresses) {
        batch = candidate_batch;
        planned_len = candidate_len;
        state_addresses = std::move(candidate_addresses);
        valid = true;
    }

    void invalidate() {
        if (graph)
            graph->reset();
        graph.reset();
        static_next = Tensor();
        state_addresses.clear();
        valid = false;
    }
};

QwenBatchOperations::QwenBatchOperations(Qwen35CausalLm &model, CudaExecutionContext &execution,
                                         CudaContinuousBatchConfig config)
    : model_(model), state_adapter_(model, config.max_sequences, config.paged_kv),
      execution_(execution), config_(std::move(config)),
      max_sequences_(static_cast<int32_t>(config_.max_sequences)),
      moe_enabled_(state_adapter_.has_moe()),
      cached_moe_enabled_(state_adapter_.has_cached_moe()) {}

QwenBatchOperations::~QwenBatchOperations() = default;

int64_t QwenBatchOperations::vocab_size() const { return model_.vocab_size(); }

int64_t QwenBatchOperations::max_context() const { return model_.max_position_embeddings(); }

std::vector<std::pair<std::string, double>> QwenBatchOperations::metrics() const {
    const auto paged = state_adapter_.paged_kv_stats();
    return {
        {"continuous_batching_decode_batches", static_cast<double>(decode_batches_)},
        {"continuous_batching_decode_tokens", static_cast<double>(decode_tokens_)},

        {"continuous_batching_compactions", 0.0},
        {"continuous_batching_stable_slot_releases",
         static_cast<double>(state_adapter_.slot_releases())},
        {"continuous_batching_batched_greedy_batches",
         static_cast<double>(batched_greedy_batches_)},
        {"continuous_batching_packed_metadata_batches",
         static_cast<double>(packed_metadata_batches_)},
        {"continuous_batching_cuda_graph_captures", static_cast<double>(cuda_graph_captures_)},
        {"continuous_batching_cuda_graph_replays", static_cast<double>(cuda_graph_replays_)},
        {"continuous_batching_moe", moe_enabled_ ? 1.0 : 0.0},
        {"continuous_batching_moe_cached_row_serial", cached_moe_enabled_ ? 1.0 : 0.0},
        {"continuous_batching_paged_kv", state_adapter_.paged_kv_enabled() ? 1.0 : 0.0},
        {"paged_kv_page_size", static_cast<double>(paged.page_size)},
        {"paged_kv_live_pages", static_cast<double>(paged.live_pages)},
        {"paged_kv_peak_live_pages", static_cast<double>(paged.peak_live_pages)},
        {"paged_kv_capacity_pages", static_cast<double>(paged.capacity_pages)},
        {"paged_kv_reserved_bytes", static_cast<double>(paged.reserved_bytes)},
        {"paged_kv_page_allocations", static_cast<double>(paged.allocations)},
        {"paged_kv_page_reuses", static_cast<double>(paged.reuses)},
        {"paged_kv_page_releases", static_cast<double>(paged.releases)},
        {"paged_kv_table_updates", static_cast<double>(paged.table_updates)},
    };
}

void QwenBatchOperations::initialize_sampling(Request &request, const Tensor &prompt_ids) {
    request.sampler.emplace(request.sampling);
    if (request.sampler->has_penalties())
        request.counts = SamplingOps::token_counts(prompt_ids, model_.vocab_size());
}

void QwenBatchOperations::suspend_decode() {
    invalidate_decode_graph();
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    state_adapter_.suspend_decode();
}

QwenBatchOperations::Sample QwenBatchOperations::prefill(
    const std::shared_ptr<Request> &request, mfq::engine::PrefillChunk chunk) {
    const int primary = execution_.layer_placement.primary_device();
    MfqCudaGuard guard(primary);
    state_adapter_.prepare_prefill(request->cache, chunk.offset, request->prompt.size());
    if (chunk.offset == 0) {
        request->prefill_ids =
            mfq_tensor_backend::tensor(
                request->prompt,
                mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, primary)))
                .reshape({1, -1})
                .contiguous();
        initialize_sampling(*request, request->prefill_ids);
    }
    Tensor hidden;
    std::unique_ptr<PrefillCudaTimer> final_timer;
    auto chunk_timer = std::make_unique<PrefillCudaTimer>();
    MfqOptional<Tensor> length = mfq_nullopt;
    if (chunk.offset > 0 && chunk.count == 1)
        length =
            mfq_tensor_backend::full({1}, chunk.offset + 1, request->prefill_ids.options());
    hidden = model_.hidden_forward(
        request->prefill_ids.narrow(1, chunk.offset, chunk.count).contiguous(), mfq_nullopt,
        length);
    const auto end = chunk.offset + chunk.count;
    if (end < request->prefill_ids.size(1)) {
        TYLOINFER_CUDA_CHECK(
            cudaEventRecord(chunk_timer->finished_event(), mfq_get_current_cuda_stream()));
        request->prefill_timers.push_back(std::move(chunk_timer));
    } else {
        final_timer = std::move(chunk_timer);
    }
    if (end < static_cast<int64_t>(request->prompt.size())) {
        state_adapter_.pause_prefill(request->cache);
        return {};
    }
    auto logits = state_adapter_.logits_from_last_hidden(std::move(hidden));
    TYLOINFER_RUNTIME_CHECK(final_timer != nullptr, "continuous batching lost the prefill timer");
    TYLOINFER_CUDA_CHECK(
        cudaEventRecord(final_timer->finished_event(), mfq_get_current_cuda_stream()));
    request->prefill_timers.push_back(std::move(final_timer));
    auto next = mfq::cuda::sample_logits(*request->sampler, std::move(logits), request->counts,
                                         request->token_constraint);
    const int64_t token = next.item<int64_t>();
    double prefill_ms = 0.0;
    for (const auto &timer : request->prefill_timers) {
        prefill_ms += timer->elapsed_ms();
    }
    request->prefill_timers.clear();
    request->prefill_ids = Tensor();
    if(model_.restore_decode_workspace)model_.restore_decode_workspace();
    return {std::move(next), token, {request->prompt.size(), prefill_ms, 0.0, prefill_ms}};
}

void QwenBatchOperations::activate(const std::shared_ptr<Request> &request, const Sample &sample) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    if (request->counts.defined())
        sample_token_counts_add_cuda(request->counts, sample.next.contiguous());
    state_adapter_.activate(request->cache);
}

void QwenBatchOperations::resume_decode(int64_t cache_position) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    state_adapter_.resume_decode(cache_position);
}

void QwenBatchOperations::discard_prefill(const std::shared_ptr<Request> &request) {
    state_adapter_.discard_prefill(request->cache);
    request->prefill_timers.clear();
    request->prefill_ids = Tensor();
}

void QwenBatchOperations::retire(const std::vector<std::shared_ptr<Request>> &retired,
                                 int64_t cache_position) {
    for (const auto &request : retired)
        state_adapter_.release(request->cache);
    if (cache_position == 0)
        invalidate_decode_graph();
    state_adapter_.finish_retire(cache_position);
}

void QwenBatchOperations::ensure_decode_metadata_buffers(int primary) {
    if (decode_metadata_host_.defined())
        return;
    const int64_t capacity = 3 * static_cast<int64_t>(max_sequences_);
    decode_metadata_host_ =
        mfq_tensor_backend::empty({capacity}, mfq_tensor_backend::TensorOptions()
                                                  .dtype(mfq_tensor_backend::kInt64)
                                                  .device(mfq_tensor_backend::kCPU)
                                                  .pinned_memory(true));
    decode_metadata_cuda_ = mfq_tensor_backend::empty(
        {capacity},
        mfq_tensor_backend::TensorOptions()
            .dtype(mfq_tensor_backend::kInt64)
            .device(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, primary)));
}

void QwenBatchOperations::invalidate_decode_graph() {
    for (auto &graph : decode_graphs_)
        graph->invalidate();
    decode_graphs_.clear();
}

QwenBatchOperations::Decoded QwenBatchOperations::decode(State &state) {
    const int primary = execution_.layer_placement.primary_device();
    MfqCudaGuard primary_guard(primary);
    for (const auto &request : state.active)
        state_adapter_.ensure_decode_tokens(request->cache, request->cache_length + 1);
    int64_t max_position = 0;
    for (const auto &request : state.active)
        max_position = std::max(max_position, request->cache_length);
    state_adapter_.prepare_decode(max_position);
    const int64_t batch = max_sequences_;
    const bool batch_greedy = std::all_of(
        state.active.begin(), state.active.end(), [](const std::shared_ptr<Request> &request) {
            return request->sampler->greedy() && !request->sampler->has_penalties() &&
                   !request->token_constraint;
        });
    int64_t requested_len = 0;
    int64_t minimum_remaining =
        state.active.front()->generation_limit - state.active.front()->produced;
    for (const auto &request : state.active) {
        const int64_t remaining = request->generation_limit - request->produced;
        minimum_remaining = std::min(minimum_remaining, remaining);
        requested_len = std::max(requested_len, request->cache_length + remaining);
    }
    const int64_t planned_len =
        decode_graph_bucket(requested_len, model_.max_position_embeddings());
    const int64_t graph_attention_parts =
        decode_graph_attention_parts(planned_len, FullBlock::kDecodeAttentionMaxParts);
    std::vector<const void *> graph_state_addresses;
    bool graph_cache_hit = false;
    bool graph_decode = false;
    if (batch >= 2 && batch_greedy && config_.greedy &&
        qwen_continuous_batch_cuda_graph_enabled(model_, config_)) {
        graph_state_addresses = state_adapter_.decode_state_addresses();
        const auto found =
            std::find_if(decode_graphs_.begin(), decode_graphs_.end(), [&](const auto &graph) {
                return graph->matches(batch, planned_len, graph_state_addresses);
            });
        graph_cache_hit = found != decode_graphs_.end();
        graph_decode =
            graph_cache_hit || minimum_remaining >= config_.cuda_graph_minimum_tokens;
    }
    QwenContinuousDecodeGraph *decode_graph = nullptr;
    if (graph_cache_hit) {
        decode_graph =
            std::find_if(decode_graphs_.begin(), decode_graphs_.end(), [&](const auto &graph) {
                return graph->matches(batch, planned_len, graph_state_addresses);
            })->get();
    } else if (graph_decode) {
        if (decode_graphs_.size() >= 8)
            invalidate_decode_graph();
        decode_graphs_.push_back(std::make_unique<QwenContinuousDecodeGraph>());
        decode_graph = decode_graphs_.back().get();
    }
    std::unique_ptr<MfqCudaStreamGuard> graph_stream_guard;
    std::vector<std::unique_ptr<MfqCudaStreamGuard>> graph_compute_stream_guards;
    if (graph_decode) {
        const auto &parallel = execution_.tensor_parallel.enabled()
                                   ? execution_.tensor_parallel
                                   : execution_.expert_parallel;
        decode_graph->ensure_compute_streams(parallel);
        graph_stream_guard = std::make_unique<MfqCudaStreamGuard>(decode_graph->stream);
        graph_compute_stream_guards =
            activate_cuda_graph_compute_streams(decode_graph->compute_streams);
    }
    ensure_decode_metadata_buffers(primary);
    auto *packed_metadata_data = decode_metadata_host_.data_ptr<int64_t>();
    std::fill_n(packed_metadata_data, batch, int64_t{0});
    std::fill_n(packed_metadata_data + batch, batch, int64_t{0});
    std::fill_n(packed_metadata_data + 2 * batch, batch, int64_t{1});
    int64_t max_sequence_length = 0;
    for (const auto &request : state.active) {
        const auto slot = static_cast<int64_t>(request->cache.slot());
        TYLOINFER_RUNTIME_CHECK(slot >= 0 && slot < batch,
                          "continuous batching request lost its stable slot");
        packed_metadata_data[slot] = request->pending_token;
        packed_metadata_data[batch + slot] = request->cache_length;
        packed_metadata_data[2 * batch + slot] = request->cache_length + 1;
        max_sequence_length = std::max(max_sequence_length, request->cache_length + 1);
    }
    const int64_t values = 3 * batch;
    auto metadata_host = decode_metadata_host_.narrow(0, 0, values);
    auto metadata_cuda = decode_metadata_cuda_.narrow(0, 0, values);
    metadata_cuda.copy_(metadata_host);
    auto matrix = metadata_cuda.reshape({3, batch});
    Tensor ids = matrix.narrow(0, 0, 1).reshape({batch, 1});
    Tensor pos = matrix.narrow(0, 1, 1).reshape({batch, 1});
    Tensor lengths = matrix.narrow(0, 2, 1).reshape({batch});
    ++packed_metadata_batches_;
    std::vector<int32_t> paused_slots;
    for (const auto &request : state.active)
        if (!request->eligible)
            paused_slots.push_back(request->cache.slot());
    // ponytail: masked slots still occupy GPU rows; restore recurrent state
    // and overwrite the uncommitted KV position when they become eligible.
    const auto paused_state = state_adapter_.capture_recurrent_slots(paused_slots);
    Tensor logits;
    Tensor graph_tokens;
    MoeContinuousBatchCacheScope moe_cache_scope(execution_, cached_moe_enabled_ && batch > 1);
    try {
        if (graph_decode) {
            const auto invoke = [&]() {
                auto hidden = model_.hidden_forward_static(
                    ids, pos, lengths, {planned_len, graph_attention_parts});
                auto current_logits = state_adapter_.logits_from_last_hidden(std::move(hidden));
                return sample_greedy_cuda(current_logits.contiguous().view({batch, -1}));
            };
            if (!graph_cache_hit) {
                decode_graph->invalidate();
                try {
                    DecodeGraphBranchScope branch_scope(
                        execution_.decode_graph_serial_branches);
                    DecodeGraphTpProjectionScope tp_projection_scope(
                        execution_.decode_graph_tp_projection_major);
                    decode_graph->graph = std::make_unique<MfqCudaGraph>();
                    prepare_decode_graph_memory(
                        model_, *decode_graph->graph, [&]() { (void)invoke(); },
                        decode_graph->participant_streams(
                            execution_.model_parallel_collectives));
                    decode_graph->graph->capture_begin();
                    decode_graph->static_next = invoke();
                    decode_graph->graph->capture_end();
                    decode_graph->set_key(batch, planned_len,
                                          state_adapter_.decode_state_addresses());
                    ++cuda_graph_captures_;
                } catch (...) {
                    decode_graph->invalidate();
                    throw;
                }
            }
            decode_graph->graph->replay();
            graph_tokens = decode_graph->static_next;
            ++cuda_graph_replays_;
        } else {
            auto hidden = model_.hidden_forward(ids, pos, lengths, nullptr, pos);
            logits = state_adapter_.logits_from_last_hidden(std::move(hidden));
        }
        state_adapter_.finish_decode(max_sequence_length);
    } catch (...) {
        invalidate_decode_graph();
        throw;
    }
    state_adapter_.restore_recurrent_slots(paused_slots, paused_state);
    ++decode_batches_;
    decode_tokens_ += std::count_if(state.active.begin(), state.active.end(),
                                    [](const auto &request) { return request->eligible; });
    Tensor batched_greedy_tokens;
    if (graph_tokens.defined()) {
        batched_greedy_tokens = graph_tokens.to(mfq_tensor_backend::kCPU).contiguous();
        TYLOINFER_RUNTIME_CHECK(batched_greedy_tokens.scalar_type() == mfq_tensor_backend::kInt64 &&
                              batched_greedy_tokens.numel() == batch,
                          "continuous batching CUDA Graph sampler returned the wrong shape");
        ++batched_greedy_batches_;
    } else if (config_.greedy && batch_greedy) {
        batched_greedy_tokens = sample_greedy_cuda(logits.contiguous().view({batch, -1}))
                                    .to(mfq_tensor_backend::kCPU)
                                    .contiguous();
        TYLOINFER_RUNTIME_CHECK(batched_greedy_tokens.scalar_type() == mfq_tensor_backend::kInt64 &&
                              batched_greedy_tokens.numel() == batch,
                          "continuous batching greedy sampler returned the wrong shape");
        ++batched_greedy_batches_;
    }
    return {std::move(logits), std::move(batched_greedy_tokens)};
}

QwenBatchOperations::Sample QwenBatchOperations::sample(
    const std::shared_ptr<Request> &request, const Decoded &decoded) {
    const auto slot = static_cast<int64_t>(request->cache.slot());
    if (decoded.greedy_tokens.defined())
        return {{}, decoded.greedy_tokens.data_ptr<int64_t>()[slot], {}};
    auto next = mfq::cuda::sample_logits(*request->sampler, decoded.logits.narrow(0, slot, 1),
                                         request->counts, request->token_constraint);
    const auto token = next.item<int64_t>();
    return {std::move(next), token, {}};
}

void QwenBatchOperations::accept(const std::shared_ptr<Request> &request, const Sample &sample) {
    if (request->counts.defined()) {
        TYLOINFER_RUNTIME_CHECK(sample.next.defined(),
                          "batched greedy sampling cannot update token penalties");
        sample_token_counts_add_cuda(request->counts, sample.next.contiguous());
    }
}

void QwenBatchOperations::recover(State &state) {
    std::vector<QwenBatchRequestState *> caches;
    for (const auto &request : state.prefilling)
        caches.push_back(&request->cache);
    for (const auto &request : state.active)
        caches.push_back(&request->cache);
    state_adapter_.recover(caches);
    invalidate_decode_graph();
}

FlashNextBatchOperations::FlashNextBatchOperations(Qwen4CausalLm& model,
        CudaExecutionContext& execution, CudaContinuousBatchConfig config,
        const qwen4_exp::Qwen4ExpMtp* predictor)
    : model_(model), execution_(execution), config_(std::move(config)), predictor_(predictor) {
    TYLOINFER_RUNTIME_CHECK(config_.max_sequences > 0 && config_.max_sequences <= 32,
                      "Flash-Next continuous batching supports 1..32 requests");
    model_.reset(1);
    if (config_.cuda_graph) graph_ = std::make_unique<Qwen4BatchDecodeGraph>(model_);
}
FlashNextBatchOperations::~FlashNextBatchOperations() = default;
bool FlashNextBatchOperations::supports_mtp(const MfqSamplingParams& sampling,const MfqTokenConstraintPtr& constraint) {
    return sampling.max_tokens>1 && (sampling.temperature<=0.0 || sampling.top_k==1) &&
        sampling.presence_penalty==0.0 && sampling.frequency_penalty==0.0 &&
        sampling.repetition_penalty==1.0 && !constraint;
}
static std::exception_ptr flash_next_prefill_failure(std::exception_ptr failure);
std::size_t FlashNextBatchOperations::prepare(const std::shared_ptr<Request>& request) {
    if(!prefix_cache_ || !prefix_cache_->persistent_prefix_enabled() ||
            !request->cache_plan.stable_prefix_tokens || !model_.supports_text_session_state())return 0;
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    const auto maximum=std::min(request->cache_plan.stable_prefix_tokens,request->prompt.size()-1);
    const auto block=prefix_cache_->prefix_block_tokens();
    // Page-aligned recurrent checkpoints remain reusable across short suffixes.
    request->prefix_boundary=maximum>=block?maximum/block*block:maximum;
    std::size_t peer_tokens=0;
    try {
        suspend_decode();
        // Prefill loans are revocable for a newly admitted prefix. Its KV
        // has priority over the workspace of an unfinished earlier append.
        if(!model_.config.experts && qwen4_borrowed_dense_weight_bytes(model_)) {
            if(release_spare_qsa_cache())default_context(execution_.layer_placement.primary_device())->reclaim_idle_memory();
            qwen4_restore_dense_prefill_weights(model_,nullptr,SIZE_MAX,0,0);
        }
        PinnedAllocationScope pinned(execution_.layer_placement.primary_device(),mfq_current_cuda_stream(),
            [&](std::size_t bytes){return reclaim_pinned_for_admission(request.get(),bytes);});
        initialize_prefill(request,{0,0});
        std::vector<std::pair<std::size_t,std::shared_ptr<Request>>> matching;
        for(const auto& weak:requests_)if(auto active=weak.lock();active && active!=request && active->cache) {
            const auto limit=std::min({active->reused_prefix,maximum,active->prompt.size()});
            const auto first=request->prompt.begin();
            const auto tokens=std::size_t(std::mismatch(first,first+limit,active->prompt.begin()).first-first);
            if(tokens)matching.emplace_back(tokens,std::move(active));
        }
        std::stable_sort(matching.begin(),matching.end(),[](const auto& a,const auto& b){return a.first>b.first;});
        std::vector<const Qwen4BatchRequestState*> peers;
        peers.reserve(matching.size());
        for(const auto& [tokens,active]:matching)peers.push_back(active->cache.get());
        if(!matching.empty())peer_tokens=matching.front().first;
        request->reused_prefix=request->cache->restore_prefix(model_,*prefix_cache_,
            request->cache_plan.session_id,request->prompt,maximum,
            request->predictor.get(),&request->predictor_hidden,peers);
    }catch(...){std::rethrow_exception(flash_next_prefill_failure(std::current_exception()));}
    request->prefix_stored=request->reused_prefix>=request->prefix_boundary;
    std::fprintf(stderr,"native_batch_prefix_restore cached_tokens=%zu prompt_tokens=%zu boundary=%zu peer_tokens=%zu\n",
        request->reused_prefix,request->prompt.size(),request->prefix_boundary,peer_tokens);
    return request->reused_prefix;
}
void FlashNextBatchOperations::commit_prefix(const std::shared_ptr<Request>& request) {
    if(!prefix_cache_ || request->prefix_stored || !request->prefix_boundary ||
            !request->cache || request->cache->position()!=int64_t(request->prefix_boundary))return;
    request->prefix_stored=true;
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    try {
        std::vector<int64_t> tokens(request->prompt.begin(),request->prompt.begin()+request->prefix_boundary);
        auto snapshot=request->cache->capture_prefix(model_,tokens,request->predictor.get(),request->predictor_hidden);
        prefix_cache_->store(request->cache_plan.session_id,std::move(snapshot));
        std::fprintf(stderr,"native_batch_prefix_store tokens=%zu\n",tokens.size());
    } catch(const std::exception& error) {
        std::fprintf(stderr,"native_batch_prefix_store action=skip error=%s\n",error.what());
    }
}
int64_t FlashNextBatchOperations::vocab_size() const { return model_.vocab_size(); }
int64_t FlashNextBatchOperations::max_context() const { return model_.max_position_embeddings(); }
mfq::engine::Metrics FlashNextBatchOperations::metrics() const {
    std::size_t ram_pageable=0,shared_cache=0;
    std::size_t host_recurrence=0;
    std::size_t bytes = 0, contexts = 0,ram_payload=0,ram_allocated=0,ram_target=0,ram_disk=0,auto_layers=0,fixed_layers=0;
    std::set<const void*> pools;
    std::array<std::uint64_t,4> tokens{};
    for (const auto& weak : requests_) if (const auto request = weak.lock(); request && request->cache) {
        bytes += request->cache->resident_bytes();
        host_recurrence += request->cache->host_recurrence_bytes();
        contexts += request->cache->position() > 0;
        for(const auto& memory:request->cache->adaptive_ram_memory()) {
            ram_payload+=memory.payload;ram_target+=memory.target;ram_disk+=memory.disk;
            auto_layers+=memory.adaptive_precision;fixed_layers+=!memory.adaptive_precision;
            if(pools.insert(memory.pool).second){ram_allocated+=memory.allocated;ram_pageable+=memory.pageable;
                bytes+=memory.shared_gpu_cache;shared_cache+=memory.shared_gpu_cache;}
            for(unsigned i=0;i<4;++i)tokens[i]+=memory.tokens[i];
        }
    }
    const auto pinned=pinned_memory_budget_stats();
    const auto graph_time=graph_?graph_->timings():Qwen4BatchDecodeGraph::Timings{};
    return {{"continuous_batching_pinned_live_bytes",double(pinned.live_bytes)},
        {"continuous_batching_pinned_proven_budget_bytes",double(pinned.proven_bytes)},
        {"continuous_batching_pinned_budget_known",pinned.proven?1.:0.},
        {"continuous_batching_qsa_ram_payload_bytes",double(ram_payload)},
        {"continuous_batching_qsa_ram_allocated_bytes",double(ram_allocated)},
        {"continuous_batching_qsa_ram_pageable_bytes",double(ram_pageable)},
        {"continuous_batching_host_recurrence_bytes",double(host_recurrence)},
        {"continuous_batching_qsa_vram_cache_bytes",double(shared_cache)},
        {"continuous_batching_qsa_ram_target_bytes",double(ram_target)},
        {"continuous_batching_qsa_ssd_bytes",double(ram_disk)},
        {"continuous_batching_qsa_auto_layers",double(auto_layers)},
        {"continuous_batching_qsa_fixed_layers",double(fixed_layers)},
        {"continuous_batching_qsa_tier_0_tokens",double(tokens[0])},
        {"continuous_batching_qsa_tier_1_tokens",double(tokens[1])},
        {"continuous_batching_qsa_tier_2_tokens",double(tokens[2])},
        {"continuous_batching_qsa_tier_3_tokens",double(tokens[3])},
        {"continuous_batching_decode_batches",double(decode_batches_)},
        {"continuous_batching_decode_tokens",double(decode_tokens_)},
        {"continuous_batching_batched_greedy_batches",double(greedy_batches_)},
        {"continuous_batching_packed_prefill_batches",double(packed_prefill_batches_)},
        {"continuous_batching_packed_prefill_requests",double(packed_prefill_requests_)},
        {"continuous_batching_packed_prefill_tokens",double(packed_prefill_tokens_)},
        {"continuous_batching_mtp_batches",double(mtp_batches_)},
        {"continuous_batching_mtp_maximum_batch",double(mtp_maximum_batch_)},
        {"continuous_batching_moe_cached_row_serial",0.0},
        {"continuous_batching_cuda_graph_captures",double(graph_ ? graph_->captures() : 0)},
        {"continuous_batching_cuda_graph_replays",double(graph_ ? graph_->replays() : 0)},
        {"continuous_batching_graph_prepare_ms",graph_time.prepare_ms},
        {"continuous_batching_qsa_prepare_ms",graph_time.qsa_prepare_ms},
        {"continuous_batching_ple_prepare_ms",graph_time.ple_prepare_ms},
        {"continuous_batching_graph_capture_ms",graph_time.capture_ms},
        {"continuous_batching_graph_replay_ms",graph_time.replay_ms},
        {"continuous_batching_graph_advance_ms",graph_time.advance_ms},
        {"continuous_batching_request_kv_bytes",double(bytes)},
        {"continuous_batching_request_contexts",double(contexts)}};
}
void FlashNextBatchOperations::suspend_decode() { if (graph_) graph_->clear(); }
std::size_t FlashNextBatchOperations::release_spare_qsa_cache() {
    if(model_.config.experts)return 0;
    std::size_t released=0;
    for(const auto& weak:requests_)if(const auto request=weak.lock();request && request->cache)
        released+=request->cache->release_shared_device_cache(true);
    return released;
}
void FlashNextBatchOperations::borrow_dense_prefill_workspace(std::size_t bytes,int64_t tokens) {
    auto ram=qwen4_pinned_transaction_bytes(model_,tokens);
    for(const auto& weak:requests_)if(const auto active=weak.lock();active && active->cache) {
        const auto target=std::min<std::size_t>(model_.max_position_embeddings(),
            active->prompt.size()+std::size_t(std::max(0,active->sampling.max_tokens)));
        ram=mfq::engine::prefill_budget_add(ram,active->cache->fixed_prefill_ram_reservation(target));
    }
    suspend_decode();
    qwen4_borrow_dense_prefill_weights(model_,nullptr,bytes,[&](std::size_t needed) {
        if(prefix_cache_)prefix_cache_->reserve_ram(needed);
    },ram);
}
void FlashNextBatchOperations::resume_decode(int64_t) {
    if(model_.config.experts)return;
    const auto borrowed=qwen4_borrowed_dense_weight_bytes(model_);
    const auto* placement=std::getenv("TYLOINFER_QSA_CPU_INDEXER");
    const bool automatic=(!placement || !*placement) && !std::getenv("TYLOINFER_QWEN4_BATCH_HOST_INDEXER_BYTES");
    if(!borrowed && !automatic)return;
    std::size_t rows=0,growth=0,attention=0;
    bool speculative=false;
    for(const auto& weak:requests_)if(const auto active=weak.lock();active && active->cache) {
        const auto position=active->cache->position();
        // Unfinished prefills still need the borrowed space.
        if(position<int64_t(active->prompt.size()))return;
        if(position>=model_.max_position_embeddings())continue;
        ++rows;
        speculative|=bool(active->predictor);
        growth=mfq::engine::prefill_budget_add(growth,active->cache->prefill_growth_bytes(1));
        attention=mfq::engine::prefill_budget_add(attention,active->cache->prefill_attention_workspace_bytes(1));
    }
    if(speculative) {
        if(borrowed && release_spare_qsa_cache())
            default_context(execution_.layer_placement.primary_device())->reclaim_idle_memory();
        if(borrowed && model_.restore_decode_workspace)model_.restore_decode_workspace();
        return;
    }
    const auto logits=mfq::engine::prefill_budget_multiply(rows,std::size_t(model_.vocab_size())*sizeof(float)*8);
    const auto workspace=qwen4_dense_prefill_workspace(model_,growth,true).bytes(rows,
        mfq::engine::prefill_budget_add(attention,logits));
    if(borrowed && release_spare_qsa_cache())
        default_context(execution_.layer_placement.primary_device())->reclaim_idle_memory();
    if(borrowed)qwen4_restore_dense_prefill_weights(model_,nullptr,SIZE_MAX,0,workspace);
    if(!rows || !automatic || qwen4_borrowed_dense_weight_bytes(model_))return;
    const auto device=execution_.layer_placement.primary_device();
    MfqCudaGuard guard(device);
    const auto context=default_context(device);
    context->reclaim_idle_memory();
    std::size_t promoted=0;
    for(const auto& weak:requests_)if(const auto active=weak.lock();active && active->cache)
        promoted+=active->cache->promote_indexers(workspace);
    if(promoted && std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))
        std::fprintf(stderr,"native_batch_qsa_indexer_promotion bytes=%zu decode_workspace=%zu available=%zu\n",
            promoted,workspace,context->available_memory());
}
static std::exception_ptr flash_next_prefill_failure(std::exception_ptr failure) {
    try {std::rethrow_exception(failure);}
    catch(const std::exception& error) {
        const std::string message=error.what();
        if(dynamic_cast<const MfqResourceExhausted*>(&error) ||
           message.starts_with("CUDA process VRAM limit exceeded:") ||
           message.starts_with("CUDA VRAM limit exceeded:") ||
           message.starts_with("CUDA pinned allocation failed:")) {
            std::fprintf(stderr,"runtime_prefill_resource_failure reason=%s\n",message.c_str());
            return std::make_exception_ptr(mfq::engine::InferenceInputError(
                mfq::engine::InferenceInputErrorCode::ContextLength,
                "insufficient GPU or pinned RAM for prefill workspace; reduce concurrent requests or the packed prefill token budget"));
        }
    } catch(...) {}
    return failure;
}
void FlashNextBatchOperations::initialize_prefill(
        const std::shared_ptr<Request>& request, mfq::engine::PrefillChunk chunk) {
    if (!request->cache) {
        TYLOINFER_RUNTIME_CHECK(chunk.offset == 0,"Flash-Next batch lost request state");
        std::erase_if(requests_,[](const auto& weak) { return weak.expired(); });
        const auto active=std::size_t(std::count_if(requests_.begin(),requests_.end(),[](const auto& weak) {
            const auto row=weak.lock();return row && row->cache;
        }));
        if(!active) {
            // The preceding exclusive request's global KV/recurrent state is
            // unused while independent batch rows own their own context.
            qwen4_release_decode_graphs(model_,nullptr);
            model_.reset(1);
            if(!model_.config.experts) {
                qwen4_release_idle_qsa(model_,nullptr,true);
                qwen4_discard_idle_qsa_descriptors(model_);
            }
            release_moe_decode_buffers(execution_.moe_expert_cache);
            default_context(execution_.layer_placement.primary_device())->reclaim_idle_memory();
        } else if(release_spare_qsa_cache()) {
            default_context(execution_.layer_placement.primary_device())->reclaim_idle_memory();
        }
        requests_.push_back(request);
        const auto slots=config_.max_sequences-std::min(active,std::size_t(config_.max_sequences)-1);
        std::size_t workspace=0;
        int recall_rows=1;
        if(!model_.config.experts && request->prefix_boundary) {
            const auto suffix=request->prompt.size()-request->prefix_boundary;
            const auto budget=std::size_t(config_.prefill_token_budget);
            const auto tokens=std::min(std::min(suffix,budget)*config_.max_sequences,budget);
            workspace=batch_prefill_workspace_bytes(model_,execution_,tokens);
            recall_rows=int(std::min<std::size_t>(suffix,qsa_prefill_recall_tile_rows));
        }
        request->cache = std::make_unique<Qwen4BatchRequestState>(model_,slots,workspace,recall_rows,request->prefix_boundary,[&](std::size_t bytes) {
            return reclaim_indexer_for_admission(bytes);
        },[&](std::size_t bytes) {
            return reclaim_pinned_for_admission(request.get(),bytes);
        });
        request->sampler.emplace(request->sampling);
        if(predictor_ && request->sampling.enable_mtp && supports_mtp(request->sampling,request->token_constraint))
            request->predictor=predictor_->fork_empty();
    }
    if(chunk.count>0 && !request->prefill_ids.defined()) {
        namespace tb=mfq_tensor_backend;
        const auto options=tb::TensorOptions().dtype(tb::kInt64).device(
            tb::Device(tb::kCUDA,execution_.layer_placement.primary_device()));
        if(request->sampler->has_penalties()) {
            const auto prompt=tb::tensor(request->prompt,options).reshape({1,-1});
            request->counts=SamplingOps::token_counts(prompt,model_.vocab_size());
            const auto reused=static_cast<int64_t>(request->reused_prefix);
            request->prefill_ids=reused?prompt.narrow(1,reused,prompt.size(1)-reused).clone():prompt;
        } else if(request->reused_prefix) {
            const std::vector<int64_t> suffix(request->prompt.begin()+request->reused_prefix,request->prompt.end());
            request->prefill_ids=tb::tensor(suffix,options).reshape({1,-1});
        } else {
            request->prefill_ids=tb::tensor(request->prompt,options).reshape({1,-1});
        }
    }
    TYLOINFER_RUNTIME_CHECK(request->cache->position() == chunk.offset,"Flash-Next batch prefill position mismatch");
}
std::size_t FlashNextBatchOperations::reclaim_indexer_for_admission(std::size_t bytes) {
    suspend_decode();
    std::size_t released=0;
    for(const auto& weak:requests_)if(released<bytes) {
        const auto request=weak.lock();
        if(request && request->cache)released+=request->cache->reclaim_indexer_for_admission(bytes-released);
    }
    if(released && std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))
        std::fprintf(stderr,"native_batch_qsa_indexer_migration required=%zu moved_to_ram=%zu\n",bytes,released);
    return released;
}

std::size_t FlashNextBatchOperations::reclaim_pinned_for_admission(const Request* incoming,std::size_t bytes) {
    std::size_t released=0;
    for(const auto& weak:requests_) {
        if(released>=bytes)break;
        const auto request=weak.lock();
        if(request && request.get()!=incoming && request->cache)
            released+=request->cache->reclaim_pinned_for_admission(bytes-released);
    }
    if(released && std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))
        std::fprintf(stderr,"native_batch_qsa_pinned_reclaim required=%zu released=%zu\n",bytes,released);
    if(released<bytes)released+=reclaim_routed_expert_pinned(execution_,bytes-released);
    return released;
}
FlashNextBatchOperations::Sample FlashNextBatchOperations::finish_prefill(
        const std::shared_ptr<Request>& request,Tensor next) {
    if (!next.defined()) return {};
    const auto token=next.item<int64_t>();
    double milliseconds=request->packed_prefill_ms;
    for (const auto& measured:request->prefill_timers) milliseconds+=measured->elapsed_ms();
    request->prefill_timers.clear(); request->prefill_ids={}; request->packed_prefill_ms=0;
    if(model_.restore_decode_workspace && !qwen4_borrowed_dense_weight_bytes(model_)) {
        bool after_capture=graph_ && !request->predictor && execution_.gguf_expert_cache &&
            !std::getenv("TYLOINFER_CUDA_WORKSPACE_RESERVE_GIB");
#ifdef _WIN32
        if(after_capture)after_capture=bool(physical_device_memory_stats(execution_.layer_placement.primary_device()));
#endif
        // A slot arena fixes its virtual size when first created. Wait until
        // the ordinary graph owns its scratch before using the expert budget.
        if(after_capture)routed_workspace_restore_pending_=true;
        else {
            model_.restore_decode_workspace();
            routed_workspace_restore_pending_=false;
        }
    }
    return {std::move(next),token,{request->prompt.size(),milliseconds,0.0,milliseconds}};
}
FlashNextBatchOperations::Sample FlashNextBatchOperations::prefill(
        const std::shared_ptr<Request>& request, mfq::engine::PrefillChunk chunk) {
    try {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    initialize_prefill(request,chunk);
    RoutedPrefillWorkspaceLoan workspace(execution_);
    const bool restored_dense=!model_.config.experts && request->reused_prefix;
    const auto workspace_bytes=[&](int64_t rows,bool packed_history=false) {
        const auto projections=batch_prefill_workspace_bytes(model_,execution_,rows,request->cache->prefill_growth_bytes(rows));
        return restored_dense?mfq::engine::prefill_budget_add(projections,
            request->cache->prefill_attention_workspace_bytes(rows,packed_history)):projections;
    };
    auto compute_rows=chunk.count;
    auto bytes=workspace_bytes(compute_rows);
    if(!model_.config.experts) {
        const auto context=default_context(execution_.layer_placement.primary_device());
        const auto required=workspace_bytes(compute_rows,restored_dense);
        if(context->available_memory()+context->workspace_available()<required && release_spare_qsa_cache())
            context->reclaim_idle_memory();
    }
    if(restored_dense) {
        const auto loan_target=workspace_bytes(compute_rows,true);
        const auto context=default_context(execution_.layer_placement.primary_device());
        const auto available_workspace=[&] {
            // The row also publishes persistent state into resident pools.
            // Vacant pages in the ordinary pool cannot cover those allocations.
            return mfq::engine::prefill_budget_add(context->available_memory(),context->workspace_available());
        };
        if(available_workspace()<loan_target) {
            context->reclaim_idle_memory();
            auto available=available_workspace();
            if(available<loan_target && std::none_of(requests_.begin(),requests_.end(),[](const auto& weak) {
                    const auto active=weak.lock();
                    return active && active->predictor;
                })) {
                // Let required prefill scratch displace the optional recall
                // cache before spilling any weights or Indexer state.
                suspend_decode();
                if(request->cache->release_shared_device_cache()) {
                    context->reclaim_idle_memory();
                    available=available_workspace();
                }
            }
            if(available<loan_target) {
                // Keep the cache growth promised to every admitted request
                // before lending VRAM-backed weights to this prefill tile.
                borrow_dense_prefill_workspace(loan_target-available,chunk.count);
            }
            const auto* placement=std::getenv("TYLOINFER_QSA_CPU_INDEXER");
            if((!placement || !*placement) && !std::getenv("TYLOINFER_QWEN4_BATCH_HOST_INDEXER_BYTES")) {
                for(available=available_workspace();available<bytes;available=available_workspace()) {
                    if(!reclaim_indexer_for_admission(bytes-available))break;
                    context->reclaim_idle_memory();
                }
            }
            available=available_workspace();
            if(available<bytes && !request->predictor) {
                // Reduce the numerical tile before advancing any layer's KV.
                // The scheduler still consumes the complete requested slice.
                compute_rows=std::max<int64_t>(1,mfq::engine::select_prefill_chunk(
                    chunk.count,chunk.count,available,workspace_bytes,true));
                bytes=workspace_bytes(compute_rows);
                if(compute_rows<chunk.count && std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))
                    std::fprintf(stderr,"native_batch_prefill_tile tokens=%lld rows=%lld available=%zu required=%zu\n",
                        static_cast<long long>(chunk.count),static_cast<long long>(compute_rows),available,bytes);
            }
        }
    }
    workspace.begin(bytes);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    const PrefillAllocationScope prefill_storage(qwen4_borrowed_dense_weight_bytes(model_)!=0);
#endif
    auto timer = std::make_unique<PrefillCudaTimer>();
    Tensor hidden;
    const auto fallback_context=restored_dense && compute_rows<chunk.count?
        default_context(execution_.layer_placement.primary_device()):nullptr;
    for(int64_t done=0;done<chunk.count;) {
        hidden={};
        const auto remaining=chunk.count-done;
        auto rows=std::min(compute_rows,remaining);
        if(fallback_context && done) {
            // Reprice the next tile after retiring the previous tile's
            // scratch. An earlier fit does not cover new live KV or arenas.
            fallback_context->reclaim_idle_memory();
            const auto available=mfq::engine::prefill_budget_add(fallback_context->available_memory(),
                fallback_context->workspace_available());
            rows=std::max<int64_t>(1,mfq::engine::select_prefill_chunk(remaining,rows,available,workspace_bytes,true));
        }
        auto ids=request->prefill_ids.narrow(1,chunk.offset+done-static_cast<int64_t>(request->reused_prefix),
            rows).contiguous();
        Tensor raw;
        hidden=request->cache->prefill(model_,ids,
            [&](std::size_t bytes){return reclaim_pinned_for_admission(request.get(),bytes);},request->predictor?&raw:nullptr);
        if(request->predictor)
            request->predictor_hidden=request->predictor->prime_prompt(MtpTarget(model_),raw,ids,request->predictor_hidden);
        done+=rows;
    }
    const auto finished = chunk.offset + chunk.count == static_cast<int64_t>(request->prompt.size());
    Tensor next;
    if (finished) next = sample_logits(*request->sampler,
        model_.logits_from_hidden(model_.last_hidden(std::move(hidden))).reshape({1,-1}),
        request->counts,request->token_constraint);
    TYLOINFER_CUDA_CHECK(cudaEventRecord(timer->finished_event(),mfq_get_current_cuda_stream()));
    request->prefill_timers.push_back(std::move(timer));
    hidden={};
    workspace.suspend();
    // Only the sampled token crosses the prefill/decoder boundary. Keep it
    // outside the expert-backed workspace while retaining transient logits.
    if(next.defined())next=next.clone();
    workspace.finish();
    return finish_prefill(request,std::move(next));
    } catch(...) {std::rethrow_exception(flash_next_prefill_failure(std::current_exception()));}
}
std::size_t FlashNextBatchOperations::prefill_batch_limit() const {
    return runtime_options::batch_packed_prefill()?config_.max_sequences:1;
}
int64_t FlashNextBatchOperations::prefill_batch_token_budget() const {
    return runtime_options::batch_packed_prefill_token_budget();
}
std::vector<mfq::engine::PrefillBatchResult<FlashNextBatchOperations::Sample>>
FlashNextBatchOperations::prefill_batch(
        const std::vector<std::pair<std::shared_ptr<Request>,mfq::engine::PrefillChunk>>& items) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    qwen4_release_decode_graphs(model_,nullptr);
    model_.adapter_begin_forward(false);
    release_moe_decode_buffers(execution_.moe_expert_cache);
    std::vector<mfq::engine::PrefillBatchResult<Sample>> result(items.size());
    std::vector<std::size_t> ready;
    for (std::size_t row=0;row<items.size();++row) {
        try { initialize_prefill(items[row].first,items[row].second); ready.push_back(row); }
        catch (...) { result[row].failure=flash_next_prefill_failure(std::current_exception()); }
    }
    const auto execute=[&](auto&& self,const std::vector<std::size_t>& rows)->void {
        if (rows.empty()) return;
        std::size_t growth=model_.config.experts?(std::size_t(256)<<20):0;
        std::size_t attention=0;
        int64_t tokens=0;
        for (const auto row:rows) {
            tokens+=items[row].second.count;
            growth+=items[row].first->cache->prefill_growth_bytes(items[row].second.count);
            if(!model_.config.experts)
                attention=mfq::engine::prefill_budget_add(attention,
                    items[row].first->cache->prefill_attention_workspace_bytes(items[row].second.count));
        }
        // Each restored row submits its own score and recall allocations.
        // Their asynchronous frees may still hold physical pages while the
        // next row runs and the packed projections remain live.
        const auto workspace_bytes=mfq::engine::prefill_budget_add(
            batch_prefill_workspace_bytes(model_,execution_,tokens,growth),attention);
        try {
            if(!model_.config.experts && rows.size()>1) {
                const auto context=default_context(execution_.layer_placement.primary_device());
                const auto available_workspace=[&] {
                    return mfq::engine::prefill_budget_add(context->available_memory(),context->workspace_available());
                };
                if(available_workspace()<workspace_bytes) {
                    release_spare_qsa_cache();
                    context->reclaim_idle_memory();
                }
                auto available=available_workspace();
                if(attention && available<workspace_bytes) {
                    borrow_dense_prefill_workspace(workspace_bytes-available,tokens);
                    available=available_workspace();
                }
                if(attention && std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))
                    std::fprintf(stderr,"native_batch_prefill_workspace rows=%zu requested=%zu attention=%zu available=%zu split=%d\n",
                        rows.size(),workspace_bytes,attention,available,int(available<workspace_bytes));
                if(available<workspace_bytes)
                    throw mfq::engine::InferenceInputError(mfq::engine::InferenceInputErrorCode::ContextLength,
                        "insufficient CUDA workspace for packed dense prefill");
            }
            if((rows.size()>1 || model_.config.experts) && model_.reserve_kv_workspace)model_.reserve_kv_workspace(growth);
        }
        catch (const mfq::engine::InferenceInputError& error) {
            if(error.code!=mfq::engine::InferenceInputErrorCode::ContextLength)throw;
            if(rows.size()==1) {result[rows.front()].failure=std::current_exception();return;}
            const auto middle=rows.begin()+rows.size()/2;
            self(self,std::vector<std::size_t>(rows.begin(),middle));
            self(self,std::vector<std::size_t>(middle,rows.end()));
            return;
        }
        try {
            if(rows.size()==1) {
                const auto row=rows.front();
                result[row].sample=prefill(items[row].first,items[row].second);
                return;
            }
            std::vector<Qwen4BatchRequestState*> caches;
            std::vector<Tensor> chunks;
            for (const auto row:rows) {
                const auto& [request,chunk]=items[row];
                caches.push_back(request->cache.get());
                chunks.push_back(request->prefill_ids.narrow(1,
                    chunk.offset-static_cast<int64_t>(request->reused_prefix),chunk.count).contiguous());
            }
            RoutedPrefillWorkspaceLoan workspace(execution_);
            workspace.begin(workspace_bytes);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            const PrefillAllocationScope prefill_storage(qwen4_borrowed_dense_weight_bytes(model_)!=0);
#endif
            PrefillCudaTimer timer;
            const bool prime=std::any_of(rows.begin(),rows.end(),[&](auto row) {return bool(items[row].first->predictor);});
            std::vector<Tensor> raw;
            auto hidden=qwen4_batch_prefill(model_,caches,chunks,prime?&raw:nullptr);
            std::vector<Tensor> last;
            std::vector<std::size_t> finished;
            for(std::size_t i=0;i<rows.size();++i) {
                const auto& [request,chunk]=items[rows[i]];
                if(request->predictor)
                    request->predictor_hidden=request->predictor->prime_prompt(MtpTarget(model_),raw[i],chunks[i],request->predictor_hidden);
                if(chunk.offset+chunk.count==static_cast<int64_t>(request->prompt.size())) {
                    last.push_back(hidden[i]);finished.push_back(rows[i]);
                }
            }
            std::vector<Tensor> next(items.size());
            if(!last.empty()) {
                auto logits=model_.logits_from_hidden(mfq_tensor_backend::cat(last,0)).reshape({int64_t(last.size()),-1});
                for(std::size_t i=0;i<finished.size();++i) {
                    const auto row=finished[i];const auto& request=items[row].first;
                    next[row]=sample_logits(*request->sampler,logits.narrow(0,i,1),request->counts,request->token_constraint);
                }
            }
            TYLOINFER_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
            const auto milliseconds=timer.elapsed_ms();
            hidden.clear();raw.clear();last.clear();workspace.suspend();
            for(auto& token:next)if(token.defined())token=token.clone();
            workspace.finish();
            for(const auto row:rows) {
                items[row].first->packed_prefill_ms+=milliseconds*items[row].second.count/tokens;
                result[row].sample=finish_prefill(items[row].first,std::move(next[row]));
            }
            ++packed_prefill_batches_;packed_prefill_requests_+=rows.size();packed_prefill_tokens_+=tokens;
        } catch (...) {
            // Kernels may have advanced attention state. Fail this group;
            // splitting/retrying is allowed only before numerical execution.
            const auto error=flash_next_prefill_failure(std::current_exception());
            for(const auto row:rows)result[row].failure=error;
        }
    };
    execute(execute,ready);
    return result;
}
void FlashNextBatchOperations::activate(const std::shared_ptr<Request>& request, const Sample& sampled) {
    request->first_token=sampled.token;
    accept(request,sampled);
}
void FlashNextBatchOperations::discard_prefill(const std::shared_ptr<Request>& request) {
    if (graph_) graph_->clear();
    finish_routed_expert_exchanges(execution_);
    request->verification.reset();request->predictor.reset();request->predictor_hidden={};
    std::size_t released=0;
    if(request->cache && std::none_of(requests_.begin(),requests_.end(),[&](const auto& weak) {
            const auto active=weak.lock();
            return active && active.get()!=request.get() && active->cache;
        }))released=request->cache->release_shared_device_cache();
    request->cache.reset(); request->prefill_ids = {}; request->prefill_timers.clear(); request->packed_prefill_ms=0;
    if(released)default_context(execution_.layer_placement.primary_device())->reclaim_idle_memory();
}
void FlashNextBatchOperations::retire(const std::vector<std::shared_ptr<Request>>& retired,int64_t) {
    for (const auto& request : retired) discard_prefill(request);
    const bool active = std::any_of(requests_.begin(),requests_.end(),[](const auto& weak) {
        const auto request = weak.lock();
        return request && request->cache;
    });
    if(!active)resume_decode(0);
    if (!active && !model_.graph_registry) {
        // Batch widths have distinct transfer arenas and graph scratch. Return
        // them at idle while preserving resident experts and adaptation heat.
        release_moe_decode_buffers(execution_.moe_expert_cache);
        default_context(execution_.layer_placement.primary_device())->reclaim_idle_memory();
    }
}
std::optional<mfq::engine::Generation> FlashNextBatchOperations::generate_decode(
        const std::shared_ptr<Request>& request,mfq::engine::InferenceOutput& output,
        std::function<void()> schedule,std::function<bool()> ready,std::function<void()> check) {
    if(!request->predictor)return std::nullopt;
    return generate_mtp(request,output,std::move(schedule),std::move(ready),std::move(check));
}
mfq::engine::Generation FlashNextBatchOperations::generate_mtp(std::shared_ptr<Request> request,
        mfq::engine::InferenceOutput& output,std::function<void()> schedule,std::function<bool()> ready,
        std::function<void()> check) {
    request->mtp_stats.available=request->mtp_stats.used=true;
    output.metrics.mtp=request->mtp_stats;
    request->pending_token=request->first_token;
    auto first=output.append({request->first_token});
    request->produced+=static_cast<int32_t>(first.token_ids.size());
    if(!output.stopped())schedule();
    co_yield std::move(first);
    while(!output.stopped()) {
        while(!ready() && !output.stopped())co_yield mfq::StepState::waiting;
        if(output.stopped())co_return;
        check();
        auto verified=std::exchange(request->verification,std::nullopt).value();
        auto delta=output.append(verified.emitted);
        request->produced+=static_cast<int32_t>(delta.token_ids.size());
        const auto accepted=std::min(verified.accepted,delta.token_ids.size());
        const auto keep=static_cast<int64_t>(accepted)+1;
        const auto rollback_started=mfq::engine::Clock::now();
        request->cache->finish_verification(keep);
        request->mtp_stats.rollback_ms+=std::chrono::duration<double,std::milli>(
            mfq::engine::Clock::now()-rollback_started).count();
        request->cache_length=request->cache->position();
        request->mtp_stats.accepted_tokens+=accepted;
        for(std::size_t pos=0;pos<accepted;++pos)++request->mtp_stats.position_accepted[pos];
        if(!output.stopped()) {
            const auto history_started=mfq::engine::Clock::now();
            namespace tb=mfq_tensor_backend;
            auto history=keep==1?request->predictor_hidden:tb::cat({request->predictor_hidden,
                verified.hidden.narrow(1,0,keep-1)},1);
            (void)request->predictor->batch_graph(MtpTarget(model_),history,verified.ids.narrow(1,0,keep),true);
            request->predictor_hidden=verified.hidden.narrow(1,keep-1,1).clone();
            request->mtp_stats.head_ms+=std::chrono::duration<double,std::milli>(
                mfq::engine::Clock::now()-history_started).count();
            request->pending_token=verified.emitted.back();
            schedule();
        }
        output.metrics.mtp=request->mtp_stats;
        co_yield std::move(delta);
    }
}
FlashNextBatchOperations::Decoded FlashNextBatchOperations::verify_mtp(State& state) {
    namespace tb=mfq_tensor_backend;
    const auto device=tb::Device(tb::kCUDA,execution_.layer_placement.primary_device());
    const auto options=tb::TensorOptions().dtype(tb::kInt64).device(device);
    const MtpTarget target(model_);
    std::vector<Qwen4BatchRequestState*> caches;
    std::vector<std::shared_ptr<Request>> active;
    std::vector<Tensor> windows;
    std::vector<std::vector<int32_t>> drafts;
    std::vector<int64_t> packed_ids,widths;
    std::vector<int64_t> offsets{0};
    bool speculative=false;
    for(const auto& request:state.active) {
        request->decode_row=-1;
        if(!request->eligible)continue;
        TYLOINFER_RUNTIME_CHECK(request->cache && request->cache->position()==request->cache_length,
            "Flash-Next MTP target position mismatch");
        request->decode_row=static_cast<int64_t>(active.size());
        active.push_back(request);
        caches.push_back(request->cache.get());
        std::vector<int64_t> ids{request->pending_token};
        auto& proposals=drafts.emplace_back();
        if(auto* predictor=request->predictor.get()) {
            const auto draft_started=mfq::engine::Clock::now();
            const auto position=predictor->cache_position();
            TYLOINFER_RUNTIME_CHECK(position+1==request->cache_length && request->predictor_hidden.defined(),
                "Flash-Next MTP predictor position mismatch");
            const auto cap=std::clamp(request->sampling.mtp_max_draft_tokens,1,predictor->maximum_draft_depth());
            const auto depth=std::min<int64_t>({cap,request->generation_limit-request->produced-1,
                model_.max_position_embeddings()-request->cache_length-1});
            auto hidden=request->predictor_hidden;
            for(int64_t i=0;i<depth;++i) {
                auto step=predictor->batch_graph(target,hidden,tb::tensor({ids.back()},options).reshape({1,1}));
                const auto token=step.greedy_token;
                TYLOINFER_RUNTIME_CHECK(token>=0,"CUDA Graph returned no MTP draft token");
                ids.push_back(token);
                proposals.push_back(token);
                hidden=std::move(step.chain_hidden);
            }
            if(depth>0)predictor->trim_cache_to(position);
            request->mtp_stats.head_ms+=std::chrono::duration<double,std::milli>(
                mfq::engine::Clock::now()-draft_started).count();
            speculative=speculative || !proposals.empty();
        }
        offsets.push_back(offsets.back()+static_cast<int64_t>(ids.size()));
        widths.push_back(static_cast<int64_t>(ids.size()));
        packed_ids.insert(packed_ids.end(),ids.begin(),ids.end());
        windows.push_back(tb::tensor(ids,options).reshape({1,-1}));
    }
    const auto target_started=mfq::engine::Clock::now();
    Tensor logits,raw;
    {
        RoutedCacheLearningScope learning(execution_,!speculative);
        const auto host_ids=tb::tensor(packed_ids,options.device(tb::kCPU)).reshape({1,-1});
        auto result=graph_?graph_->run(caches,host_ids,false,widths,&raw):std::nullopt;
        TYLOINFER_RUNTIME_CHECK(result && raw.defined(),"CUDA Graph is required for batch MTP verification");
        logits=std::move(*result);
    }
    auto greedy=sample_greedy_cuda(logits.contiguous()).to(tb::kCPU).contiguous();
    const auto target_ms=std::chrono::duration<double,std::milli>(mfq::engine::Clock::now()-target_started).count();
    const auto* selected=greedy.data_ptr<int64_t>();
    std::vector<Tensor> ordinary;
    std::size_t mtp_rows=0;
    for(std::size_t row=0;row<active.size();++row) {
        auto& request=*active[row];
        ordinary.push_back(logits.narrow(0,offsets[row],1));
        if(!request.predictor) {
            request.cache->finish_verification(1);
            continue;
        }
        ++mtp_rows;
        const auto& proposals=drafts[row];
        mfq::engine::mtp::ChainVerification result{0,static_cast<int32_t>(selected[offsets[row]]),true};
        if(!proposals.empty()) {
            std::vector<int32_t> tokens;
            tokens.reserve(widths[row]);
            for(auto i=offsets[row];i<offsets[row+1];++i)tokens.push_back(static_cast<int32_t>(selected[i]));
            result=mfq::engine::mtp::verify_greedy(proposals,tokens);
        }
        std::vector<int64_t> emitted(proposals.begin(),proposals.begin()+result.accepted_drafts);
        emitted.push_back(result.next_token);
        request.verification.emplace(Request::Verification{std::move(emitted),std::move(windows[row]),
            raw.narrow(1,offsets[row],widths[row]),result.accepted_drafts});
        auto& stats=request.mtp_stats;
        stats.target_ms+=target_ms;
        ++stats.cycles;
        stats.drafted_tokens+=proposals.size();
        stats.standard_tokens+=proposals.empty();
        stats.selected_depth=static_cast<int>(proposals.size());
        ++stats.depth_cycles[proposals.size()];
        for(std::size_t pos=0;pos<proposals.size();++pos)++stats.position_drafted[pos];
    }
    ++mtp_batches_;
    mtp_maximum_batch_=std::max(mtp_maximum_batch_,static_cast<int64_t>(mtp_rows));
    ++decode_batches_;
    decode_tokens_+=offsets.back();
    return {tb::cat(ordinary,0),{}};
}
FlashNextBatchOperations::Decoded FlashNextBatchOperations::decode(State& state) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    PinnedAllocationScope required_pinned(execution_.layer_placement.primary_device(),mfq_current_cuda_stream(),[&](std::size_t bytes) {
        if(DecodeWindow::recording())return std::size_t(0);
        return reclaim_routed_expert_pinned(execution_,bytes);
    });
    if(std::any_of(state.active.begin(),state.active.end(),[](const auto& request) {return request->eligible && request->predictor;})) {
        if(std::exchange(routed_workspace_restore_pending_,false))model_.restore_decode_workspace();
        return verify_mtp(state);
    }
    std::vector<Qwen4BatchRequestState*> caches;
    std::vector<int64_t> tokens;
    for (const auto& request : state.active) {
        request->decode_row = -1;
        if (!request->eligible) continue;
        request->decode_row = static_cast<int64_t>(tokens.size());
        TYLOINFER_RUNTIME_CHECK(request->cache && request->cache->position() == request->cache_length,
                          "Flash-Next batch decode position mismatch");
        caches.push_back(request->cache.get()); tokens.push_back(request->pending_token);
    }
    auto host_ids = mfq_tensor_backend::tensor(tokens,
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)).reshape({-1,1});
    Tensor eager_ids;
    MoeContinuousBatchCacheScope moe_cache_scope(execution_,tokens.size() > 8);
    const bool greedy = config_.greedy && std::all_of(state.active.begin(),state.active.end(),[](const auto& request) {
        return !request->eligible || (request->sampler->greedy() && !request->sampler->has_penalties() && !request->token_constraint);
    });
    std::optional<Tensor> captured;
    try {
        if(graph_)captured=graph_->run(caches,host_ids,greedy);
        if(std::exchange(routed_workspace_restore_pending_,false) && !captured)model_.restore_decode_workspace();
        if(!captured)eager_ids=host_ids.to(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA,
            execution_.layer_placement.primary_device()));
    }
    catch(const std::exception& error) {
        const std::string message=error.what();
        if(!dynamic_cast<const MfqResourceExhausted*>(&error) &&
           !message.starts_with("CUDA process VRAM limit exceeded:") &&
           !message.starts_with("CUDA VRAM limit exceeded:") &&
           !message.starts_with("CUDA pinned allocation failed:"))throw;
        if(std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))
            std::fprintf(stderr,"native_batch_decode_resource_failure rows=%zu reason=%s\n",tokens.size(),message.c_str());
        throw mfq::engine::InferenceInputError(mfq::engine::InferenceInputErrorCode::ContextLength,
            "insufficient GPU or pinned RAM for the batch decode workspace; reduce concurrent requests");
    }
    auto logits = captured ? std::move(*captured) : qwen4_batch_decode(model_,caches,std::move(eager_ids),host_ids);
    Tensor greedy_tokens;
    if (greedy) {
        greedy_tokens = captured ? logits.to(mfq_tensor_backend::kCPU).contiguous()
            : sample_greedy_cuda(logits.contiguous()).to(mfq_tensor_backend::kCPU).contiguous();
        ++greedy_batches_;
    }
    ++decode_batches_; decode_tokens_ += tokens.size();
    return {std::move(logits),std::move(greedy_tokens)};
}
FlashNextBatchOperations::Sample FlashNextBatchOperations::sample(
        const std::shared_ptr<Request>& request,const Decoded& decoded) {
    if(request->predictor)return {};
    TYLOINFER_RUNTIME_CHECK(request->decode_row >= 0,"Flash-Next sampled an inactive row");
    if(decoded.greedy_tokens.defined())return {{},decoded.greedy_tokens.data_ptr<int64_t>()[request->decode_row],{}};
    auto next = sample_logits(*request->sampler,decoded.logits.narrow(0,request->decode_row,1),
                             request->counts,request->token_constraint);
    const auto token = next.item<int64_t>();
    return {std::move(next),token,{}};
}
void FlashNextBatchOperations::accept(const std::shared_ptr<Request>& request,const Sample& sampled) {
    if (request->counts.defined()) sample_token_counts_add_cuda(request->counts,sampled.next.contiguous());
}
void FlashNextBatchOperations::recover(State& state) {
    for (const auto& request : state.active) discard_prefill(request);
    for (const auto& request : state.prefilling) discard_prefill(request);
}
} // namespace mfq::cuda

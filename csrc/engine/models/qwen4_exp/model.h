#pragma once
#include "env.h"
#include "engine/backends/cuda/executor/options.h"
#include "resident_allocations.h"
#include "engine/backends/cuda/kernels/mfq_cuda_linear_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/models/qwen4_exp/causal_lm.h"
#include "engine/models/qwen4_exp/ngram.h"
#include "engine/models/common/attention_ops.h"
#include "engine/ops.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/backends/cuda/kernels/qwen4_exp.h"
#include "engine/backends/cuda/kernels/ple_fused.h"
#include "engine/weights/nint_row_pipeline.h"
#include "engine/weights/row_stage.h"
#include "engine/backends/cuda/executor/decode_window.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/qwen4_exp/snapshot.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <numeric>
#include <set>
#include <unordered_map>

namespace tyloi::cuda::qwen4_exp {
using attention_ops::DenseTensor;
using attention_ops::Linear;
using attention_ops::Embedding;
using attention_ops::rms_norm;

// The upstream PLE hash is a CPU uint64 algorithm, including signed remainder
// after bitwise mixing and EOS segment boundaries. Keep shards independent.
class NgramEmbedding {
  public:
    using Context = tyloi::models::qwen4_exp::NgramContext;
    using RowLookup = std::function<::tyloi::Tensor(const std::vector<int64_t>&,const ::tyloi::Device&)>;
    using StageFactory = std::function<std::shared_ptr<RowStage>(const std::vector<int64_t>&,const ::tyloi::Device&)>;
    struct State {
        State() = default;
        State(const State&) = delete;
        State& operator=(const State&) = delete;
        State(State&&) = default;
        State& operator=(State&&) = default;
        Context context;
        std::shared_ptr<NintRowPipeline> pipeline;
        bool hash_ready=false,hash_reported=false;
        int cached_tokens=0;
        std::array<int64_t,8> cached_ids{};
        tyloi::models::qwen4_exp::NgramHashes cached_hashes;
    };
    NgramEmbedding(std::vector<Embedding> shards, int64_t rows, int64_t dimension, int64_t ngram,
                   int64_t heads_per_ngram, int64_t eos, std::vector<int64_t> multipliers,
                   std::vector<int64_t> offsets, std::vector<int64_t> vocab,std::vector<std::shared_ptr<tyloi::NintRows>> ranges = {},
                   RowLookup row_lookup = {},StageFactory stage_factory = {})
        : shards_(std::move(shards)), rows_(rows), dimension_(dimension), ngram_(ngram),
          heads_(heads_per_ngram), eos_(eos), multipliers_(std::move(multipliers)),
          offsets_(std::move(offsets)), vocab_(std::move(vocab)),row_lookup_(std::move(row_lookup)),stage_factory_(std::move(stage_factory)) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if(!ranges.empty())source_=std::make_shared<NintRowSource>(std::move(ranges));
#else
        (void)ranges;
#endif
        TYLOINFER_RUNTIME_CHECK(!shards_.empty() && rows_ > 0 && dimension_ > 0 && ngram_ > 1 &&
                              heads_ > 0 && multipliers_.size() == size_t(ngram_) &&
                              offsets_.size() == size_t((ngram_ - 1) * heads_) &&
                              vocab_.size() == offsets_.size(),
                          "invalid Qwen4 ngram metadata geometry");
        for (size_t h = 0; h < vocab_.size(); ++h)
            TYLOINFER_RUNTIME_CHECK(vocab_[h] > 0 && offsets_[h] >= 0 &&
                                  offsets_[h] <= int64_t(shards_.size()) * rows_ - vocab_[h],
                              "Qwen4 ngram vocabulary lies outside embedding shards");
    }
    void reset(State& state) const { state.context = {};state.hash_ready=false; }
    bool graph_supported() const { return bool(source_) || bool(stage_factory_); }
    std::pair<std::size_t, std::size_t> context_growth_bytes(const State& state, int64_t batch, int64_t tokens) const {
        TYLOINFER_RUNTIME_CHECK(batch > 0 && tokens > 0 && ngram_ > 1 &&
            tokens <= std::numeric_limits<int64_t>::max() - (ngram_ - 1), "invalid ngram growth geometry");
        const std::array<int64_t,2> next{batch, ngram_ - 1};
        const std::array<int64_t,2> history{batch, ngram_ - 1 + tokens};
        const auto retained = ::tyloi::make_contiguous_view(nullptr, next, ::tyloi::kInt64, {}).nbytes();
        auto temporary = ::tyloi::make_contiguous_view(nullptr, history, ::tyloi::kInt64, {}).nbytes();
        if (state.context.tokens.size() > (std::numeric_limits<std::size_t>::max() - temporary) / sizeof(int64_t))
            throw std::overflow_error("ngram context growth overflow");
        temporary += state.context.tokens.size() * sizeof(int64_t);
        return {retained, temporary};
    }
    const Context &context(const State& state) const { return state.context; }
    void restore(State& state,Context context) const { state.context = std::move(context);state.hash_ready=false; }
    Context prefix_context(const State& state,const DenseTensor& host_ids,int64_t count) const {
        auto host=host_ids.narrow(1,0,count).to(::tyloi::kInt64).contiguous();
        return tyloi::models::qwen4_exp::ngram_hashes(host.data_ptr<int64_t>(),host.size(0),count,
            ngram_,heads_,eos_,multipliers_,offsets_,vocab_,state.context,true).next;
    }
    void prefetch(State& state,const DenseTensor& host_ids,bool early_prefix=false) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT

        if(!source_)return;
        if(!state.pipeline)state.pipeline=std::make_shared<NintRowPipeline>(source_);
        TYLOINFER_RUNTIME_CHECK(host_ids.is_cpu() && host_ids.dim()==2,"PLE prefetch requires host token IDs");
        auto host=host_ids.to(::tyloi::kInt64).contiguous();const auto b=host.size(0),t=host.size(1);
        if(cacheable_hashes(b,t)) {
            const auto& hashes=graph_hashes(state,host);
            state.pipeline->issue(hashes.ids,{b,t,(ngram_-1)*heads_},early_prefix);
            return;
        }
        auto hashes=tyloi::models::qwen4_exp::ngram_hashes(host.data_ptr<int64_t>(),b,t,ngram_,heads_,eos_,
            multipliers_,offsets_,vocab_,state.context,true);
        state.pipeline->issue(hashes.ids,{b,t,(ngram_-1)*heads_},early_prefix);

#else
    (void)state; (void)host_ids; (void)early_prefix;
#endif
}
    std::shared_ptr<RowStage> graph_stage(State& state,int64_t b,int64_t t,const ::tyloi::Device& device) const {
        const std::vector<int64_t> shape{b,t,(ngram_-1)*heads_};
        if(stage_factory_)return stage_factory_(shape,device);
        TYLOINFER_RUNTIME_CHECK(bool(source_),"PLE graph requires prepared range rows");
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if(!state.pipeline)state.pipeline=std::make_shared<NintRowPipeline>(source_);
        return std::make_shared<NintStage>(state.pipeline,shape,device);
#else
        throw std::runtime_error("PLE graph requires TQ row stages");
#endif
    }
    void upload_graph(State& state,const DenseTensor& host_ids,RowStage& stage) const {
        TYLOINFER_RUNTIME_CHECK(graph_supported() && host_ids.is_cpu() && host_ids.dim()==2,"PLE graph requires host token IDs");
        auto host=host_ids.to(::tyloi::kInt64).contiguous();const auto b=host.size(0),t=host.size(1);
        if(cacheable_hashes(b,t)) {
            auto& hashes=graph_hashes(state,host);
            stage.upload(hashes.ids,{b,t,(ngram_-1)*heads_});
            restore(state,std::move(hashes.next));
            return;
        }
        auto hashes=tyloi::models::qwen4_exp::ngram_hashes(host.data_ptr<int64_t>(),b,t,ngram_,heads_,eos_,
            multipliers_,offsets_,vocab_,state.context,true);
        stage.upload(hashes.ids,{b,t,(ngram_-1)*heads_});
        restore(state,std::move(hashes.next));
    }
    DenseTensor forward(State& state,const DenseTensor &ids, bool cache, DenseTensor *hashed_ids = nullptr,const DenseTensor& host_ids = {}) const {
        TYLOINFER_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0,
                          "Qwen4 ngram IDs must be [B,T]");
        const auto b = ids.size(0), t = ids.size(1), nh = (ngram_ - 1) * heads_;
        auto host = host_ids.defined() ? host_ids.to(::tyloi::kInt64).contiguous() : ids.to(::tyloi::kInt64).contiguous().cpu();
        TYLOINFER_RUNTIME_CHECK(host.is_cpu() && host.sizes()==ids.sizes(),"PLE host token mirror differs");
        const auto *source = host.data_ptr<int64_t>();
        auto hashes = tyloi::models::qwen4_exp::ngram_hashes(
            source, b, t, ngram_, heads_, eos_, multipliers_, offsets_, vocab_, state.context, cache);
        auto &global = hashes.ids;
        if(row_lookup_) {
            auto result=row_lookup_(global,ids.device());
            if(hashed_ids)*hashed_ids=::tyloi::cuda::tensor(global).reshape({b,t,nh}).to(ids.device());
            if(cache)restore(state,std::move(hashes.next));
            return result.dense().reshape({b,t,nh*dimension_});
        }
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if(bool(source_)) {
            if(!state.pipeline)state.pipeline=std::make_shared<NintRowPipeline>(source_);
            auto result=state.pipeline->collect(global,{b,t,nh},ids.device());
            if(hashed_ids)*hashed_ids=::tyloi::cuda::tensor(global).reshape({b,t,nh}).to(ids.device());
            if(cache)restore(state,std::move(hashes.next));
            return result.dense().reshape({b,t,nh*dimension_});
        }
#endif

        auto global_tensor =
            ::tyloi::cuda::from_blob(global.data(), {b, t, nh}, ::tyloi::TensorOptions().dtype(::tyloi::kInt64))
                .clone()
                .to(ids.device());
        if (hashed_ids)
            *hashed_ids = global_tensor;
        std::set<int64_t> used;
        for (auto value : global)
            used.insert(value / rows_);
        DenseTensor result;
        for (auto shard : used) {
            auto mask = (global_tensor >= shard * rows_) & (global_tensor < (shard + 1) * rows_);
            auto local =
                ::tyloi::cuda::where(mask, global_tensor - shard * rows_, ::tyloi::cuda::zeros_like(global_tensor));
            auto values = shards_.at(shard)(local);
            TYLOINFER_RUNTIME_CHECK(values.sizes().vec() == std::vector<int64_t>({b, t, nh, dimension_}),
                              "Qwen4 ngram embedding shape mismatch");
            values = values * mask.unsqueeze(-1).to(values.scalar_type());
            result = result.defined() ? result + values : values;
        }
        if (cache)
            restore(state,std::move(hashes.next));
        return result.reshape({b, t, nh * dimension_});
    }

  private:
#if TYLOINFER_ENABLE_MFQ_COMPAT
    struct NintStage final:RowStage {
        std::shared_ptr<NintRowPipeline> pipeline;
        std::shared_ptr<NintRowStage> stage;
        NintStage(std::shared_ptr<NintRowPipeline> p,const std::vector<int64_t>& shape,const ::tyloi::Device& device)
            :pipeline(std::move(p)),stage(pipeline->make_stage(shape,device)) {}
        const ::tyloi::Tensor& output()const override{return stage->output();}
        bool mapped_rows()const override{return stage->mapped_rows();}
        void decode()override{stage->decode();}
        void upload(const std::vector<int64_t>& ids,const std::vector<int64_t>& shape)override {
            pipeline->upload(*stage,ids,shape);
        }
    };
#endif

    std::vector<Embedding> shards_;
    int64_t rows_, dimension_, ngram_, heads_, eos_;
    std::vector<int64_t> multipliers_, offsets_, vocab_;
    std::shared_ptr<const NintRowSource> source_;
    RowLookup row_lookup_;
    StageFactory stage_factory_;
    bool reuse_hashes_=[] {
        const auto* option=tyloi::cuda::env_lookup("TYLOINFER_PLE_HASH_REUSE");return !option || option[0]!='0';
    }();
    bool cacheable_hashes(int64_t b,int64_t t) const {return reuse_hashes_ && b==1 && t>0 && t<=8;}
    tyloi::models::qwen4_exp::NgramHashes& graph_hashes(State& state,const DenseTensor& host) const {
        const auto tokens=host.size(1);const auto* ids=host.data_ptr<int64_t>();
        if(state.hash_ready && state.cached_tokens==tokens && std::equal(ids,ids+tokens,state.cached_ids.begin())) {
            if(!state.hash_reported) {
                state.hash_reported=true;
                std::fprintf(stderr,"ple_ngram_hash_reuse enabled=1 tokens=%lld\n",static_cast<long long>(tokens));
            }
            return state.cached_hashes;
        }
        state.cached_hashes=tyloi::models::qwen4_exp::ngram_hashes(ids,1,tokens,ngram_,heads_,eos_,
            multipliers_,offsets_,vocab_,state.context,true);
        std::copy_n(ids,tokens,state.cached_ids.begin());state.cached_tokens=int(tokens);state.hash_ready=true;
        return state.cached_hashes;
    }
};

struct PleWeights {
    Linear key, value;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor key_norm, query_norm, conv_norm, conv;
#else
    DenseTensor key_norm, query_norm, conv_norm, conv;
#endif
};
class Ple {
  public:
    struct State {
        NgramEmbedding::State embedding;
#ifdef TYLOINFER_NATIVE_CUDA
        ::tyloi::Tensor conv;
#else
        DenseTensor conv;
#endif
        DenseTensor saved_conv;
        NgramEmbedding::Context saved_context;
        std::vector<DenseTensor> saved_prefix_conv;
        std::vector<NgramEmbedding::Context> saved_prefix_context;
        int64_t saved_prefix_position=-1;
        bool saved=false;
    };
    Ple(NgramEmbedding embedding, PleWeights weights, int64_t hidden, int64_t streams,
        int64_t dilation, double eps)
        : embedding_(std::move(embedding)), w_(std::move(weights)), hidden_(hidden),
          streams_(streams), dilation_(dilation), eps_(eps) {}
    void reset(State& state,bool retain_storage=false) const {
        embedding_.reset(state.embedding);
#ifdef TYLOINFER_NATIVE_CUDA
        if(retain_storage && state.conv.defined())state.conv.dense().zero_();else state.conv = {};
#else
        if(retain_storage && state.conv.defined())state.conv.zero_();else state.conv = DenseTensor();
#endif
        commit(state);
    }
    void commit(State& state) const {
        state.saved = false;
        state.saved_conv = DenseTensor();
        state.saved_context = {};
        state.saved_prefix_conv.clear();state.saved_prefix_context.clear();state.saved_prefix_position=-1;
    }
    void rollback(State& state,int64_t keep=-1) const {
        TYLOINFER_RUNTIME_CHECK(state.saved, "Qwen4 PLE has no speculative checkpoint");
        if(keep>=0 && state.saved_prefix_position>=0) {
            const auto index=keep-state.saved_prefix_position;
            TYLOINFER_RUNTIME_CHECK(index>=0 && index<int64_t(state.saved_prefix_conv.size()),
                "Qwen4 PLE prefix checkpoint is out of range");
            state.saved_conv=state.saved_prefix_conv[index];state.saved_context=state.saved_prefix_context[index];
        }
#ifdef TYLOINFER_NATIVE_CUDA
        if(state.conv.defined())state.conv.dense().copy_(state.saved_conv);else state.conv=state.saved_conv;
#else
        if(state.conv.defined())state.conv.copy_(state.saved_conv);else state.conv=state.saved_conv;
#endif
        embedding_.restore(state.embedding,state.saved_context);
        commit(state);
    }
#ifdef TYLOINFER_NATIVE_CUDA
    const ::tyloi::Tensor &conv_state(const State& state) const { return state.conv; }
    std::vector<::tyloi::Tensor*> graph_state(State& state) const { return {&state.conv}; }
#else
    const DenseTensor &conv_state(const State& state) const { return state.conv; }
    std::vector<DenseTensor*> graph_state(State& state) const { return {&state.conv}; }
#endif
    std::pair<std::size_t, std::size_t> ordinary_growth_bytes(const State& state, int64_t batch, int64_t tokens) const {
        if (state.conv.defined()) TYLOINFER_RUNTIME_CHECK(state.conv.size(0) == batch, "PLE batch changed");
        const auto kernel = w_.conv.size(-1);
        TYLOINFER_RUNTIME_CHECK(batch > 0 && tokens > 0 && kernel > 0 && dilation_ > 0 &&
            kernel - 1 <= std::numeric_limits<int64_t>::max() / dilation_, "invalid PLE growth geometry");
        const auto history = (kernel - 1) * dilation_;
        TYLOINFER_RUNTIME_CHECK(tokens <= std::numeric_limits<int64_t>::max() - history, "PLE growth overflow");
        const std::array<int64_t,4> joined{batch, history + tokens, streams_, hidden_};
        const std::array<int64_t,4> conv{batch, history, streams_, hidden_};
        const auto device = w_.conv.device();
        const auto a = ::tyloi::make_contiguous_view(nullptr, joined, ::tyloi::kFloat32, device).nbytes();
        const auto b = ::tyloi::make_contiguous_view(nullptr, conv, ::tyloi::kFloat32, device).nbytes();
        if (state.conv.defined()) {
            if (batch > 1 && a > std::numeric_limits<std::size_t>::max() - b) throw std::overflow_error("PLE growth overflow");
            return {0, a + (batch > 1 ? b : 0)};
        }
        if (batch > 1 && a > std::numeric_limits<std::size_t>::max() - b) throw std::overflow_error("PLE growth overflow");
        return batch == 1 ? std::pair{a, b} : std::pair{b, a + b};
    }
    std::size_t checkpoint_growth_bytes(const State& state, int64_t batch, int64_t tokens, int64_t confirmed) const {
        if (tokens <= 8) TYLOINFER_RUNTIME_CHECK(state.conv.defined(), "PLE verification requires committed convolution state");
        const auto kernel = w_.conv.size(-1);
        TYLOINFER_RUNTIME_CHECK(kernel > 0 && dilation_ > 0 && kernel - 1 <= std::numeric_limits<int64_t>::max() / dilation_,
                                "invalid PLE checkpoint geometry");
        const auto count = tokens <= 8 ? tokens - confirmed : 1;
        const std::array<int64_t,4> shape{batch, (kernel - 1) * dilation_, streams_, hidden_};
        const auto bytes = ::tyloi::make_contiguous_view(nullptr, shape, ::tyloi::kFloat32, w_.conv.device()).nbytes();
        if (bytes && std::size_t(count) > std::numeric_limits<std::size_t>::max() / bytes)
            throw std::overflow_error("PLE checkpoint growth overflow");
        return bytes * std::size_t(count);
    }
    std::pair<std::size_t, std::size_t> context_growth_bytes(const State& state, int64_t batch, int64_t tokens) const {
        return embedding_.context_growth_bytes(state.embedding, batch, tokens);
    }
    bool graph_supported() const { return embedding_.graph_supported(); }
    const NgramEmbedding::Context& session_context(const State& state) const { return embedding_.context(state.embedding); }
    void restore_graph_context(State& state,NgramEmbedding::Context context) const { embedding_.restore(state.embedding,std::move(context)); }
    void restore_session(State& state,const DenseTensor& convolution,NgramEmbedding::Context context,bool retain_storage=false) const {
        TYLOINFER_RUNTIME_CHECK(convolution.defined(),"missing Qwen4 PLE session state");
        if(retain_storage) {
#ifdef TYLOINFER_NATIVE_CUDA
            TYLOINFER_RUNTIME_CHECK(state.conv.defined() && state.conv.dense().sizes()==convolution.sizes(),
#else
            TYLOINFER_RUNTIME_CHECK(state.conv.defined() && state.conv.sizes()==convolution.sizes(),
#endif
                "Qwen4 PLE graph session geometry changed");
            commit(state);
#ifdef TYLOINFER_NATIVE_CUDA
            state.conv.dense().copy_(convolution);
#else
            state.conv.copy_(convolution);
#endif
        } else {reset(state);state.conv=convolution.clone();}
        embedding_.restore(state.embedding,std::move(context));
    }
    std::shared_ptr<RowStage> graph_stage(State& state,int64_t tokens,const ::tyloi::Device& device) const {
        return embedding_.graph_stage(state.embedding,1,tokens,device);
    }
    void upload_graph(State& state,const DenseTensor& host_ids,RowStage& stage) const {embedding_.upload_graph(state.embedding,host_ids,stage);}
    NgramEmbedding::Context graph_prefix_context(const State& state,const DenseTensor& host_ids,int64_t count) const {
        return embedding_.prefix_context(state.embedding,host_ids,count);
    }
    void bind_graph_checkpoint(State& state,const std::vector<DenseTensor>& conv,
                               std::vector<NgramEmbedding::Context> context,int64_t position) const {
        TYLOINFER_RUNTIME_CHECK(!state.saved,"resolve previous PLE checkpoint");
        TYLOINFER_RUNTIME_CHECK(!conv.empty() && conv.size()==context.size(),"invalid PLE prefix checkpoints");
        state.saved=true;state.saved_conv=conv.front();state.saved_context=context.front();
        state.saved_prefix_conv=conv;state.saved_prefix_context=std::move(context);state.saved_prefix_position=position;
    }
    DenseTensor forward_graph(State& state,const DenseTensor& x,const DenseTensor& ids,
                         const DenseTensor& rows,int64_t confirmed,std::vector<DenseTensor>& checkpoint_conv) const {
        if(!confirmed)return forward_chunk(state,x,ids,true,{},rows);
        auto prefix=forward_chunk(state,x.narrow(1,0,confirmed),ids.narrow(1,0,confirmed),
            true,{},rows.narrow(1,0,confirmed));
#ifdef TYLOINFER_NATIVE_CUDA
        checkpoint_conv.front().copy_(state.conv.dense());
#else
        checkpoint_conv.front().copy_(state.conv);
#endif
        auto suffix=forward_chunk(state,x.narrow(1,confirmed,x.size(1)-confirmed),
            ids.narrow(1,confirmed,x.size(1)-confirmed),true,{},
            rows.narrow(1,confirmed,x.size(1)-confirmed),&checkpoint_conv,1);
        return ::tyloi::cuda::cat({prefix,suffix},1);
    }
    void prefetch_tokens(State& state,const DenseTensor& ids,bool early_prefix=false) const { embedding_.prefetch(state.embedding,ids,early_prefix); }
    DenseTensor forward(State& state, const DenseTensor &x, const DenseTensor &ids, bool cache,
                   int64_t confirmed = 0,const DenseTensor& host_ids = {},int64_t cache_position = -1) const {
        TYLOINFER_RUNTIME_CHECK(x.dim() == 3 && ids.dim() == 2 && x.size(0) == ids.size(0) &&
                              x.size(1) == ids.size(1) && x.size(2) == hidden_ * streams_ &&
                              confirmed >= 0 && confirmed <= x.size(1) && (!confirmed || cache),
                          "invalid Qwen4 PLE input");
        if(cache && cache_position>=0 && confirmed>0 && confirmed<x.size(1) && x.size(1)<=8) {
            TYLOINFER_RUNTIME_CHECK(!state.saved && state.conv.defined(),"PLE verification requires committed convolution state");
            auto host=host_ids.defined()?host_ids:ids.to(::tyloi::kCPU,::tyloi::kInt64);
            std::vector<DenseTensor> conv;
            std::vector<NgramEmbedding::Context> context;
            for(int64_t i=confirmed;i<x.size(1);++i) {
#ifdef TYLOINFER_NATIVE_CUDA
                conv.push_back(::tyloi::cuda::empty_like(state.conv.dense()));
#else
                conv.push_back(::tyloi::cuda::empty_like(state.conv));
#endif
                context.push_back(embedding_.prefix_context(state.embedding,host,i));
            }
            auto rows=embedding_.forward(state.embedding,ids,true,nullptr,host);
            auto result=forward_graph(state,x,ids,rows,confirmed,conv);
            bind_graph_checkpoint(state,conv,std::move(context),cache_position+confirmed);
            return result;
        }
        return tyloi::models::recurrent_window(
            x.size(1), cache, confirmed, state.saved,
            [&](int64_t start, int64_t count) {
                return forward_chunk(state, x.narrow(1, start, count),
                                     ids.narrow(1, start, count), cache,
                                     host_ids.defined() ? host_ids.narrow(1,start,count) : DenseTensor{});
            },
            [&] {
                state.saved = true;
#ifdef TYLOINFER_NATIVE_CUDA
                state.saved_conv = state.conv.dense().clone();
#else
                state.saved_conv = state.conv.clone();
#endif
                state.saved_context = embedding_.context(state.embedding);
            },
            [&] { rollback(state); },
            [](DenseTensor first, DenseTensor second) { return ::tyloi::cuda::cat({first, second}, 1); });
    }
    DenseTensor forward_chunk(State& state, const DenseTensor &x, const DenseTensor &ids,
                         bool cache,const DenseTensor& host_ids = {},const DenseTensor& graph_embeddings = {},
                         std::vector<DenseTensor>* checkpoints=nullptr,std::size_t checkpoint_offset=0) const {
        auto context = embedding_.context(state.embedding);
#ifdef TYLOINFER_NATIVE_CUDA
        auto conv = state.conv.defined() ? state.conv.dense() : DenseTensor{};
#else
        auto conv = state.conv;
#endif
        const auto b = x.size(0), t = x.size(1);
        const auto save_prefixes=[&](const DenseTensor& normalized) {
            if(!checkpoints)return;
            TYLOINFER_RUNTIME_CHECK(cache && conv.defined(),"PLE prefix history is unavailable");
            const auto lookback=conv.size(1);
            // The FIR state is a tail of normalized inputs. Recover every
            // intermediate tail while keeping key/value projections batched.
            for(int64_t count=1;count<t;++count) {
                auto& saved=checkpoints->at(checkpoint_offset+std::size_t(count-1));
                if(count>=lookback)saved.copy_(normalized.narrow(1,count-lookback,lookback));
                else {
                    saved.narrow(1,0,lookback-count).copy_(conv.narrow(1,count,lookback-count));
                    saved.narrow(1,lookback-count,count).copy_(normalized.narrow(1,0,count));
                }
            }
        };
#ifdef TYLOINFER_NATIVE_CUDA
        const auto native=::tyloi::cuda::options::ple_native_fusion();
        if((!native || *native) && x.is_contiguous() && t<=32 && hidden_>128 && hidden_<=8192 &&
            (x.scalar_type()==::tyloi::kFloat16 || x.scalar_type()==::tyloi::kFloat32)) {
            try {
                auto embeddings=graph_embeddings.defined()?graph_embeddings:
                    embedding_.forward(state.embedding,ids,cache,nullptr,host_ids);
#ifdef TYLOINFER_NATIVE_CUDA
                auto key=mfq_qwen4_exp::grouped_rms_norm(::tyloi::ops::project(embeddings,w_.key).dense(),w_.key_norm.dense(),hidden_,eps_);
                auto query=mfq_qwen4_exp::grouped_rms_norm(x,w_.query_norm.dense(),hidden_,eps_);
#else
                auto key=mfq_qwen4_exp::grouped_rms_norm(::tyloi::ops::project(embeddings,w_.key).dense(),w_.key_norm,hidden_,eps_);
                auto query=mfq_qwen4_exp::grouped_rms_norm(x,w_.query_norm,hidden_,eps_);
#endif
                auto value=::tyloi::ops::project(embeddings,w_.value).dense();
                auto prepared=mfq_qwen4_exp::ple_gate_norm_cuda(key,query,value,
#ifdef TYLOINFER_NATIVE_CUDA
                    w_.conv_norm.dense().to(::tyloi::kFloat32).contiguous(),hidden_,streams_,eps_);
#else
                    w_.conv_norm.to(::tyloi::kFloat32).contiguous(),hidden_,streams_,eps_);
#endif
                save_prefixes(prepared[1]);
#ifdef TYLOINFER_NATIVE_CUDA
                auto output=mfq_qwen4_exp::ple_conv_add_cuda(prepared[1],w_.conv.dense(),
#else
                auto output=mfq_qwen4_exp::ple_conv_add_cuda(prepared[1],w_.conv,
#endif
                    cache && conv.defined()?std::optional<DenseTensor>(conv):std::nullopt,dilation_,prepared[0],
                    x.scalar_type()==::tyloi::kFloat16);
                if(cache) {if(conv.defined())conv.copy_(output[1]);else state.conv=output[1];}
                return output[0];
            }catch(...) {embedding_.restore(state.embedding,std::move(context));throw;}
        }
#endif
        return tyloi::models::qwen4_exp::position_embedding(
            cache, [&] { return graph_embeddings.defined() ? graph_embeddings : embedding_.forward(state.embedding,ids, cache,nullptr,host_ids); },
            [&](const DenseTensor &embeddings) {
                auto key = mfq_qwen4_exp::grouped_rms_norm(::tyloi::ops::project(embeddings,w_.key).dense(),
#ifdef TYLOINFER_NATIVE_CUDA
                                                           w_.key_norm.dense(), hidden_, eps_)
#else
                                                           w_.key_norm, hidden_, eps_)
#endif
                               .reshape({b, t, streams_, hidden_});
                return key;
            },
            [&] {
#ifdef TYLOINFER_NATIVE_CUDA
                auto query = mfq_qwen4_exp::grouped_rms_norm(x, w_.query_norm.dense(), hidden_, eps_)
#else
                auto query = mfq_qwen4_exp::grouped_rms_norm(x, w_.query_norm, hidden_, eps_)
#endif
                                 .reshape({b, t, streams_, hidden_});
                return query;
            },
            [&](const DenseTensor &key, const DenseTensor &query, const DenseTensor &embeddings) {
                auto score = (key.to(::tyloi::kFloat32) * query.to(::tyloi::kFloat32)).sum(-1) /
                             std::sqrt(double(hidden_));
                auto sign = (score > 0).to(::tyloi::kFloat32) - (score < 0).to(::tyloi::kFloat32);
                auto root = sign * ::tyloi::cuda::clamp_min(score.abs(), 1e-6).sqrt();
                auto gated = (::tyloi::cuda::sigmoid(root).unsqueeze(-1) *
                              ::tyloi::ops::project(embeddings,w_.value).dense().to(::tyloi::kFloat32).unsqueeze(-2))
                                 .reshape({b, t, streams_ * hidden_});
                return gated;
            },
            [&](const DenseTensor &gated) {
#ifdef TYLOINFER_NATIVE_CUDA
                return mfq_qwen4_exp::grouped_rms_norm(gated, w_.conv_norm.dense(), hidden_, eps_);
#else
                return mfq_qwen4_exp::grouped_rms_norm(gated, w_.conv_norm, hidden_, eps_);
#endif
            },
            [&](DenseTensor normalized) {
                save_prefixes(normalized);
                return mfq_qwen4_exp::ple_dilated_conv_silu(
#ifdef TYLOINFER_NATIVE_CUDA
                    normalized, w_.conv.dense(),
#else
                    normalized, w_.conv,
#endif
                    cache && conv.defined() ? std::optional<DenseTensor>(conv) : std::nullopt,
                    dilation_);
            },
            [&](DenseTensor gated, DenseTensor convolved) { return (gated + convolved).to(x.scalar_type()); },
            [&](DenseTensor next) { if(conv.defined())conv.copy_(next);else state.conv=std::move(next); },
            [&] { embedding_.restore(state.embedding,std::move(context)); });
    }

  private:
    NgramEmbedding embedding_;
    PleWeights w_;
    int64_t hidden_, streams_, dilation_;
    double eps_;
};

} // namespace tyloi::cuda::qwen4_exp

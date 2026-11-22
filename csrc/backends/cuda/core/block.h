#pragma once

#include "rope.h"
#include "app/studio/kernels/cuda/independent_hc.h"

#include <cstdint>
#include <vector>

struct CudaExecutionContext;

mfq_tensor_backend::Tensor qwen_rms_norm(
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor weight,
    double eps,
    double weight_offset);

mfq_tensor_backend::Tensor qwen_rms_norm_bf16(
    const CudaExecutionConfig& config,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor weight,
    double eps,
    double weight_offset);

struct Block {
    struct ResidualDepthState {
        std::vector<mfq_tensor_backend::Tensor> blocks, prelude_blocks, previous_pass;
        mfq_tensor_backend::Tensor prelude, previous_output;
    };
    struct LoopMhcWeights {
        mfq_tensor_backend::Tensor attention, attention_bias, ffn, ffn_bias;
        float pre_scale = 1.0f;
        int iterations = 0;
        bool fixed_identity = false;
    };
    struct Context {
        mfq_tensor_backend::Tensor token_ids;
        mfq_tensor_backend::Tensor host_token_ids;
        mfq_tensor_backend::Tensor positions;
        mfq_tensor_backend::Tensor full_positions;
        int64_t cache_position = 0;
        int64_t confirmed_prefix = 0;
        int64_t planned_kv_length = 0;
        int64_t decode_attention_parts = 0;
        MfqOptional<mfq_tensor_backend::Tensor> sequence_lengths = mfq_nullopt;
        MfqOptional<mfq_tensor_backend::Tensor> cache_positions = mfq_nullopt;
        MfqOptional<mfq_tensor_backend::Tensor> attention_mask = mfq_nullopt;
        mfq_tensor_backend::Tensor* loop_streams = nullptr;
        ResidualDepthState* residual_depth = nullptr;
    };

    int cuda_device = 0;
    bool cpu_offloaded = false;
    std::shared_ptr<const LoopMhcWeights> loop_mhc;
    bool begins_loop_mhc = false;
    bool ends_loop_mhc = false;

    template<class Attend, class FeedForward>
    mfq_tensor_backend::Tensor forward_loop_mhc(const Context& context,
        const mfq_tensor_backend::Tensor& hidden, Attend attend, FeedForward feed_forward) const {
        namespace tb = mfq_tensor_backend;
        TYLOINFER_RUNTIME_CHECK(context.loop_streams && loop_mhc, "loop mHC requires per-forward stream state");
        auto& streams = *context.loop_streams;
        if (begins_loop_mhc)
            streams = hidden.unsqueeze(2).expand({hidden.size(0), hidden.size(1), 2, hidden.size(2)}).contiguous();
        const auto apply = [&](const tb::Tensor& function, const tb::Tensor& bias, const auto& branch) {
            const auto route = mfq::cuda::two_stream_mhc_prepare(streams, function, bias, 1e-6f,
                loop_mhc->pre_scale, loop_mhc->iterations, loop_mhc->fixed_identity);
            auto output = branch(route.branch).to(streams.scalar_type()).contiguous();
            streams = mfq::cuda::two_stream_mhc_merge(streams, output, route);
        };
        apply(loop_mhc->attention, loop_mhc->attention_bias, attend);
        apply(loop_mhc->ffn, loop_mhc->ffn_bias, feed_forward);
        auto output = streams.to(tb::kFloat32).mean(2).to(hidden.scalar_type());
        if (ends_loop_mhc) streams = {};
        return output;
    }

    virtual ~Block() = default;
    virtual void reset(int64_t batch) = 0;
    virtual void set_token_ids(const mfq_tensor_backend::Tensor&) {}
    virtual void prefetch_token_ids(const mfq_tensor_backend::Tensor&) {}
    virtual bool supports_speculation() const noexcept { return false; }
    virtual std::vector<mfq_tensor_backend::Tensor*> graph_warmup_state() { return {}; }
    virtual void begin_speculative(int64_t) {}
    virtual void commit_speculative() {}
    virtual void rollback_speculative(int64_t) {}
    virtual mfq_tensor_backend::Tensor forward(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor positions,
        int64_t cache_position,
        const MfqOptional<mfq_tensor_backend::Tensor>& sequence_lengths,
        const RopeCache& rope,
        const MfqOptional<mfq_tensor_backend::Tensor>& cache_positions = mfq_nullopt,
        const MfqOptional<mfq_tensor_backend::Tensor>& attention_mask = mfq_nullopt) = 0;
    virtual mfq_tensor_backend::Tensor forward_context(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        const Context& context,
        const RopeCache& rope);
};

using mfq_tensor_backend::indexing::Slice;

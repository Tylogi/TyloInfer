#include "../../runtime/execution_options.h"
#include "../../kernels/gguf_mmq.h"
#include "mfq_cuda_moe_ops.h"
#include "mfq_cuda_process_memory.h"
#include "mfq_cuda_shared_gate.h"
#include "mfq_cuda_sampling_ops.h"
#include "tyloinfer/runtime.h"
#include "tyloinfer/host_memory.h"
#include "runtime/qsa_ram_cache.h"
#include "kernels/qsa_ram.h"
#include "ops.h"
#include "model.h"
#include "mtp.h"
#include "storage/weight_loader.h"
#include "storage/moe_ffn_pipeline.h"
#include "storage/moe_expert_cache.h"
#include "runtime/routed_cache_resources.h"
#include "runtime/routed_graph_pools.h"
#include "prefill_budget.h"
#include "runtime/nint_row_pipeline.h"
#include "runtime/decode_window.h"
#include "runtime/row_packet_prefetch.h"
#include "runtime/moe_pipeline.h"
#include "runtime/router_lookahead_audit.h"
#include <fstream>
#include <iostream>
#include <atomic>
#include "models/common/gated_mlp.h"
#include "models/common/moe.h"
#include "models/common/transformer_layer.h"
#include "storage/session_codec.h"
#include "storage/text_session_cache.h"
#include "gated_residual.h"
#include "gated_residual_fused.h"
#include "../../ops/gguf/gguf_residual.h"
#include "float_projection.h"
#include "../../kernels/gguf.h"
#include "nint_prefill_cache.h"
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "mfq_native_matmul_rows.h"
#endif
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string_view>
#include <thread>

namespace mfq::cuda {
void Qwen4HiddenCapture::begin(bool capture) {
    active = capture && !layers.empty(); values.clear();
    if (active) values.resize(layers.size());
}
bool Qwen4HiddenCapture::contains(int64_t layer) const {
    return active && std::binary_search(layers.begin(), layers.end(), layer);
}
void Qwen4HiddenCapture::capture(int64_t layer, const mfq_tensor_backend::Tensor& hidden, int64_t streams) {
    if (!contains(layer)) return;
    const auto slot = std::lower_bound(layers.begin(), layers.end(), layer) - layers.begin();
    values[slot] = hidden.reshape({hidden.size(0), hidden.size(1), streams, -1}).to(mfq_tensor_backend::kFloat32).mean(2).contiguous();
}
mfq_tensor_backend::Tensor Qwen4HiddenCapture::result(const mfq_tensor_backend::Tensor& fallback) const {
    if (!active) return fallback;
    TYLOINFER_RUNTIME_CHECK(std::all_of(values.begin(), values.end(), [](const auto& value) { return value.defined(); }),
        "DFlash2 target hidden states were not fully captured");
    return mfq_tensor_backend::cat(values, -1).contiguous();
}
static bool streamed_decode_graphs(const CudaExecutionContext& execution) {
    if(!execution.qsa_kv_store)return true;
    const auto* option=std::getenv("TYLOINFER_KV_STREAM_GRAPHS");
    return execution.kv_memory && execution.kv_memory->offload() && (!option || option[0]!='0');
}
static mfq_tensor_backend::Tensor join_request_outputs(
        const std::vector<mfq_tensor_backend::Tensor>& outputs,int64_t axis) {
    const auto option=mfq::cuda::runtime_options::single_output_view();
    if(outputs.size()==1 && outputs.front().is_contiguous() && (!option || *option))
        return outputs.front();
    return mfq_tensor_backend::cat(outputs,axis);
}
static mfq_tensor_backend::Tensor append_decode_positions(
        const mfq_tensor_backend::Tensor& previous,const mfq_tensor_backend::Tensor& current,
        mfq_tensor_backend::Tensor& storage,int64_t maximum,
        const char* option_name="TYLOINFER_QWEN4_POSITION_BUFFER") {
    namespace tb=mfq_tensor_backend;
    if(const auto option=mfq::cuda::runtime_options::position_buffer(option_name))if(!*option)
        return previous.defined()?tb::cat({previous,current},-1):current.clone();
    TYLOINFER_RUNTIME_CHECK(current.dim()>=1 && (!previous.defined() || (previous.dim()==current.dim() &&
        previous.scalar_type()==current.scalar_type() && previous.device()==current.device())),
        "invalid decode position history");
    const int64_t count=previous.defined()?previous.size(-1):0,required=count+current.size(-1);
    TYLOINFER_RUNTIME_CHECK(required<=maximum,"decode position history exceeds context");
    for(int axis=0;previous.defined() && axis+1<current.dim();++axis)
        TYLOINFER_RUNTIME_CHECK(previous.size(axis)==current.size(axis),"decode position history axes changed");
    // Rollback retains a prefix view; restored histories receive a new arena.
    // The same last-axis append serves 1D target and 3D MTP rotary positions.
    const bool owns_prefix=storage.defined() && previous.defined() &&
        previous.data_ptr()==storage.data_ptr();
    if(!owns_prefix || storage.size(-1)<required) {
        int64_t capacity=owns_prefix?storage.size(-1):std::min<int64_t>(256,maximum);
        while(capacity<required)capacity=capacity>maximum/2?maximum:capacity*2;
        auto shape=current.sizes().vec();shape.back()=capacity;
        auto expanded=tb::empty(shape,current.options());
        if(count)expanded.narrow(-1,0,count).copy_(previous);
        storage=std::move(expanded);
    }
    storage.narrow(-1,count,current.size(-1)).copy_(current);
    return storage.narrow(-1,0,required);
}
}

namespace mfq::cuda::qwen4_exp {
using Config = mfq::models::qwen4_exp::Config;

struct Gr {
    Tensor norm;
    weight_loader::ResidualLinear down, up, injection;
    int64_t hidden, streams;
    double eps;
    GatedResidualDownInjectionProjection combined_down;
    int64_t prefill_work_chunk(const Tensor& input)const {
        if(input.dim()!=3 || input.size(0)!=1)return 0;
        const auto option=mfq::cuda::runtime_options::gr_prefill_work_chunk();
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        const int64_t chunk=option?*option:8192;
        // Chat framing just beyond 8k needs no large temporary-memory cut.
        // Avoid extra concatenation there; longer inputs use 8k work tiles.
        if(!option && input.size(1)<=9216)return 0;
#else
        const int64_t chunk=option?*option:0;
#endif
        return chunk>=256 && input.size(1)>chunk?chunk:0;
    }
    template<class Project>
    std::vector<Tensor> bounded_pre(const Tensor& input,int64_t chunk,bool reuse_input,Project project)const {
        const auto tokens=input.size(1);
        std::vector<Tensor> output(3);
        if(reuse_input)output[1]=input;
        for(int64_t begin=0;begin<tokens;) {
            auto count=std::min(chunk,tokens-begin);
            const auto tail=tokens-begin-count;
            // Both pieces must retain the main GR prefill kernels; a tiny
            // remainder selects different projection arithmetic.
            if(tail>0 && tail<128)count-=128-tail;
            auto part=project(begin,count);
            for(size_t index=0;index<part.size();++index) {
                if(!part[index].defined() || (reuse_input && index==1))continue;
                if(!output[index].defined()) {
                    auto shape=part[index].sizes().vec();shape[1]=tokens;
                    output[index]=tb::empty(shape,part[index].options());
                }
                output[index].narrow(1,begin,count).copy_(part[index]);
            }
            begin+=count;
        }
        return output;
    }
    bool can_fuse(CudaExecutionContext& execution,const Tensor& x)const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if(!execution.config.gr_two_stage || !execution.config.gr_fused_projections ||
            !execution.config.gr_native_projection_input || !down.weight || !up.weight || !injection.weight ||
            !down.weight->is_nint() || !up.weight->is_nint() || streams<1 || streams>4 ||
            x.dim()<1 || x.size(-1)!=hidden*streams || x.numel()<hidden*streams ||
            x.numel()/(hidden*streams)>32 || !x.is_cuda() || !x.is_contiguous() ||
            (x.scalar_type()!=tb::kFloat16 && x.scalar_type()!=tb::kFloat32) ||
            !norm.is_contiguous() || norm.scalar_type()!=tb::kFloat32)return false;
        if(!injection.weight->is_nint() && !(execution.config.gr_two_stage_dense_injection &&
            injection.weight->is_dense() && injection.weight->dense.is_contiguous() &&
            injection.weight->dense.scalar_type()==tb::kBFloat16))return false;
        const auto& w=up.weight->nint;
        const auto rows=x.numel()/(hidden*streams);
        const auto continuous=mfq::cuda::runtime_options::gr_continuous_two_stage();
        if(rows>8 && (!execution.continuous_batch_cache_serial || (continuous && !*continuous)))
            return false;
        const auto batch=mfq::cuda::runtime_options::gr_batch_two_stage();
        if(rows>1 && batch && !*batch)return false;
        const bool selected=w.ng==1 || ((w.out+3)/4)*rows>=nint_float_projection_resident_blocks(x.get_device());
        const auto trace=mfq::cuda::runtime_options::trace_gr_dispatch();
        if(selected && trace && *trace) {
            static thread_local uint64_t seen[2]{};
            const int kind=x.scalar_type()==tb::kFloat16?0:1;
            if(!(seen[kind]&(uint64_t(1)<<rows))) {
                seen[kind]|=uint64_t(1)<<rows;
                std::fprintf(stderr,"gr_two_stage rows=%lld half=%d continuous=%d\n",
                    static_cast<long long>(rows),kind==0,int(execution.continuous_batch_cache_serial));
            }
        }
        return selected;
#else
        return false;
#endif
    }
    std::vector<Tensor> fused(const Tensor& branch,const Tensor& residual,const Tensor& prior,bool after,
        const GatedResidualMarker& marker={})const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
        const auto& projection=*injection.weight;
        return gated_residual_two_stage_cuda(branch,residual,prior,norm,down.weight->nint,up.weight->nint,
            projection.is_nint()?&projection.nint:nullptr,streams,eps,after,
            projection.is_dense()?projection.dense:Tensor{},injection.prepared_right,marker,up.up_group_coefficients);
#else
        throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
    }
    std::vector<Tensor> pre(CudaExecutionContext &execution, const Tensor &x,
        const GatedResidualMarker& marker={}) const {
        if(streams==1) {
            auto normalized=grouped_rms_norm_cuda(x.contiguous(),norm,x.size(-1),eps,0.0);
            return {std::move(normalized),x,x.narrow(-1,0,1)};
        }
        if(const auto chunk=prefill_work_chunk(x))
            return bounded_pre(x,chunk,true,[&](int64_t begin,int64_t count) {
                return pre(execution,x.narrow(1,begin,count),marker);
            });
        if(can_fuse(execution,x))return fused({},x,{},false,marker);
        if(auto gguf=gguf_fused(execution,x,{},{},marker);!gguf.empty())return gguf;
        if(auto dense=dense_fused(execution,x,{},{},marker);!dense.empty())return dense;
        return gated_residual_pre_projected(execution, x, norm, down.forward, up.forward, injection.forward,
            hidden, streams, eps,up.mixed,marker,down.activated,injection.activated,combined_down);
    }
    Tensor post(const Tensor &branch, const std::vector<Tensor> &inputs) const {
        if(streams==1)return inputs[1]+branch;
        return mfq_qwen4_exp::gated_residual_post(branch, inputs[1], inputs[2], streams);
    }
    bool post_into(const Tensor& branch,const std::vector<Tensor>& inputs,const Tensor& output)const {
        if(streams==1){auto target=output;target.copy_(inputs[1]+branch);return true;}
        return mfq_qwen4_exp::try_gated_residual_post_out(branch,inputs[1],inputs[2],output,streams);
    }
    std::vector<Tensor> dense_fused(CudaExecutionContext& execution,const Tensor& input,
            const Tensor& branch,const Tensor& prior,const GatedResidualMarker& marker)const {
        const auto option=mfq::cuda::runtime_options::gr_dense_bf16_direct();
        if(!execution.config.gr_two_stage || !execution.config.gr_fused_projections ||
                !execution.config.gr_fused_projection_activation ||
                (branch.defined() && !execution.config.gr_fused_post_norm) || (option && !*option) ||
                !down.weight || !up.weight || !injection.weight || !down.weight->is_dense() ||
                !up.weight->is_dense() || !injection.weight->is_dense())return {};
        return try_gated_residual_bf16_cuda(input,norm,down.weight->dense,up.weight->dense,
            injection.weight->dense,streams,eps,branch,prior,marker);
    }
    std::vector<Tensor> gguf_fused(CudaExecutionContext& execution,const Tensor& input,
            const Tensor& branch,const Tensor& prior,const GatedResidualMarker& marker)const {
        if(!execution.config.gr_fused_projections || !execution.config.gr_fused_projection_activation ||
                (branch.defined() && !execution.config.gr_fused_post_norm) ||
                !down.weight || !up.weight || !injection.weight || !down.weight->is_gguf() ||
                !up.weight->is_gguf() || !injection.weight->is_gguf())return {};
        return try_gguf_gated_residual_pre(input,norm,down.weight->gguf,up.weight->gguf,
            injection.weight->gguf,streams,eps,branch,prior,marker);
    }
    std::vector<Tensor> pre_after(CudaExecutionContext& execution,const Tensor& branch,
        const std::vector<Tensor>& inputs,const GatedResidualMarker& marker={}) const {
        if(streams==1)return pre(execution,inputs[1]+branch,marker);
        if(const auto chunk=prefill_work_chunk(inputs[1]))
            return bounded_pre(inputs[1],chunk,false,[&](int64_t begin,int64_t count) {
                const std::vector<Tensor> part{{},inputs[1].narrow(1,begin,count),inputs[2].narrow(1,begin,count)};
                return pre_after(execution,branch.narrow(1,begin,count),part,marker);
            });
        if(can_fuse(execution,inputs[1]) && branch.is_contiguous() && inputs[2].is_contiguous() &&
            (branch.scalar_type()==tb::kFloat16 || branch.scalar_type()==tb::kFloat32) &&
            (inputs[2].scalar_type()==tb::kFloat16 || inputs[2].scalar_type()==tb::kFloat32))
            return fused(branch,inputs[1],inputs[2],true,marker);
        if(auto gguf=gguf_fused(execution,inputs[1],branch,inputs[2],marker);!gguf.empty())return gguf;
        if(auto dense=dense_fused(execution,inputs[1],branch,inputs[2],marker);!dense.empty())return dense;
        return gated_residual_pre_after_projected(execution,branch,inputs[1],inputs[2],norm,
            down.forward,up.forward,injection.forward,hidden,streams,eps,
            up.mixed,marker,down.activated,injection.activated,combined_down);
    }
};

// Preserve router and shared-expert arithmetic while regrouping routed experts.
static thread_local int64_t qwen_prefill_reference_rows=0;
struct QwenPrefillReferenceRowsScope {
    int64_t previous;
    explicit QwenPrefillReferenceRowsScope(int64_t rows):previous(qwen_prefill_reference_rows) {
        qwen_prefill_reference_rows=rows;
    }
    ~QwenPrefillReferenceRowsScope(){qwen_prefill_reference_rows=previous;}
};
template<class Projection>
static Tensor qwen_prefill_preserve_rows(const Tensor& source,Projection project) {
    const auto rows=qwen_prefill_reference_rows;
    if(rows<=0 || source.size(0)<=rows)return project(source);
    Tensor output;
    for(int64_t begin=0;begin<source.size(0);begin+=rows) {
        const auto count=std::min<int64_t>(rows,source.size(0)-begin);
        auto part=project(source.narrow(0,begin,count));
        if(!output.defined()) {
            auto shape=part.sizes().vec();shape[0]=source.size(0);
            output=tb::empty(shape,part.options());
        }
        output.narrow(0,begin,count).copy_(part);
    }
    return output;
}

// This scope is active only while an entire GDN layer is evaluated. Retain
// each projection's original token count and therefore its GEMM arithmetic.
static thread_local int64_t qwen_gdn_prefill_projection_rows=0;
struct QwenGdnPrefillProjectionRowsScope {
    int64_t previous;
    explicit QwenGdnPrefillProjectionRowsScope(int64_t rows):previous(qwen_gdn_prefill_projection_rows) {
        qwen_gdn_prefill_projection_rows=rows;
    }
    ~QwenGdnPrefillProjectionRowsScope(){qwen_gdn_prefill_projection_rows=previous;}
};
static Linear qwen_gdn_preserve_projection_rows(Linear project) {
    return [project](CudaExecutionContext& execution,const Tensor& source) {
        const auto rows=qwen_gdn_prefill_projection_rows;
        if(rows<=0 || source.dim()!=3 || source.size(0)!=1 || source.size(1)<=rows)
            return project(execution,source);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        const auto parallel_option=mfq::cuda::runtime_options::gdn_prefill_projection_streams();
        const int streams=parallel_option?*parallel_option:2;
        const auto tail=source.size(1)%rows;
        if(streams>0 && (tail==0 || tail>8)) {
            const mfq::cuda::MatmulRowsScope gemm_rows(rows,streams);
            return project(execution,source);
        }
#endif
        Tensor output;
        for(int64_t begin=0;begin<source.size(1);begin+=rows) {
            const auto count=std::min<int64_t>(rows,source.size(1)-begin);
            auto part=project(execution,source.narrow(1,begin,count));
            if(!output.defined()) {
                auto shape=part.sizes().vec();shape[1]=source.size(1);
                output=tb::empty(shape,part.options());
            }
            output.narrow(1,begin,count).copy_(part);
        }
        return output;
    };
}
static LinearGroup qwen_gdn_preserve_projection_rows(LinearGroup project) {
    if(!project)return {};
    return [project](CudaExecutionContext& execution,const Tensor& source) {
        const auto rows=qwen_gdn_prefill_projection_rows;
        if(rows<=0 || source.dim()!=3 || source.size(0)!=1 || source.size(1)<=rows)
            return project(execution,source);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        const auto parallel_option=mfq::cuda::runtime_options::gdn_prefill_projection_streams();
        const int streams=parallel_option?*parallel_option:2;
        const auto tail=source.size(1)%rows;
        if(streams>0 && (tail==0 || tail>8)) {
            const mfq::cuda::MatmulRowsScope gemm_rows(rows,streams);
            return project(execution,source);
        }
#endif
        std::vector<Tensor> output;
        for(int64_t begin=0;begin<source.size(1);begin+=rows) {
            const auto count=std::min<int64_t>(rows,source.size(1)-begin);
            auto part=project(execution,source.narrow(1,begin,count));
            if(output.empty())for(const auto& value:part) {
                auto shape=value.sizes().vec();shape[1]=source.size(1);
                output.push_back(tb::empty(shape,value.options()));
            }
            TYLOINFER_RUNTIME_CHECK(part.size()==output.size(),"GDN projection count changed across chunks");
            for(size_t i=0;i<part.size();++i)output[i].narrow(1,begin,count).copy_(part[i]);
        }
        return output;
    };
}

template<class Projection>
static Projection qwen_gdn_profile_projection(Projection project,const char* begin,const char* end) {
    if(!project)return {};
    return [project,begin,end](CudaExecutionContext& execution,const Tensor& input,auto... arguments) {
        auto* window=mfq::cuda::DecodeWindow::recording();
        if(window)window->mark_current(begin);
        auto output=project(execution,input,arguments...);
        if(window)window->mark_current(end);
        return output;
    };
}

static Linear qwen_moe(CudaExecutionContext& loading,weight_loader::Routed gate_up, weight_loader::Routed down,
                       Linear router, Linear shared_gate, Linear sg, Linear su, Linear sd,
                       const Config &c, MoeFfnPrefetch* prefetch=nullptr) {
    std::shared_ptr<TqProjectionGroup> shared_tq;
    const auto shared_g=weight_loader::linear_weight(sg),shared_u=weight_loader::linear_weight(su),
        shared_d=weight_loader::linear_weight(sd);
    if(shared_g && shared_u && shared_g->is_tq() && shared_u->is_tq() &&
       shared_g->tq.image.is_cuda() && shared_u->tq.image.is_cuda())
        shared_tq=std::make_shared<TqProjectionGroup>(std::vector<TqWeight>{shared_g->tq,shared_u->tq});
    weight_loader::LinearGroup shared_gguf;
    int64_t shared_width=0;
    if(shared_g && shared_u && shared_g->is_gguf() && shared_u->is_gguf() && shared_g->gguf.out==shared_u->gguf.out) {
        std::vector<Linear> projections{sg,su};
        shared_gguf=weight_loader::grouped_linear(loading,projections,2);shared_width=shared_g->gguf.out;
    }
    const auto shared_gate_weight=weight_loader::linear_weight(shared_gate);
    auto shared_scale=[shared_gate,shared_gate_weight](CudaExecutionContext& execution,const Tensor& input) {
        const auto option=mfq::cuda::runtime_options::shared_gate_fused();
        const auto* nint_option=std::getenv("TYLOINFER_SHARED_GATE_NINT");
        if((!option || *option) && (!nint_option || nint_option[0]!='0') && shared_gate_weight &&
                shared_gate_weight->is_nint() && !shared_gate_weight->tensor_parallel())
            if(auto output=try_nint_shared_gate_sigmoid_cuda(input,shared_gate_weight->nint))return *output;
        // Match dense_projection's short-window accumulation exactly. Its
        // larger windows use BLAS with a different reduction tree.
        if((!option || *option) && input.size(0)<=6 && shared_gate_weight && shared_gate_weight->is_dense() &&
                !shared_gate_weight->tensor_parallel())
            if(auto output=try_shared_gate_sigmoid_cuda(input,shared_gate_weight->dense))return *output;
        return tb::sigmoid(shared_gate(execution,input));
    };
    const auto scale_output=[shared_scale,shared_gate_weight](CudaExecutionContext& execution,const Tensor& source,const Tensor& output) {
        const auto option=mfq::cuda::runtime_options::shared_gate_fused();
        if((!option || *option) && shared_gate_weight && shared_gate_weight->is_dense() &&
                !shared_gate_weight->tensor_parallel())
            if(auto result=try_shared_gate_multiply_cuda(source,shared_gate_weight->dense,output))return *result;
        return shared_scale(execution,source)*output;
    };
    auto shared_raw=[sg,su,sd,scale_output,shared_tq,shared_gguf,shared_width,shared_d](CudaExecutionContext& execution,const Tensor& source) {
        if(shared_tq) {
            auto output=sd(execution,shared_tq->swiglu(source));
            return scale_output(execution,source,output);
        }
        if(shared_gguf && source.size(0)<=32 && source.scalar_type()==tb::kFloat16) {
            auto gu=shared_gguf(execution,source).at(0).contiguous();
            if(shared_d && shared_d->is_gguf() && !shared_d->tensor_parallel() &&
                    !shared_d->gguf.prefers_original_input() &&
                    shared_d->gguf.packed.is_cuda() && shared_d->gguf.packed.device()==source.device()) {
                const auto& down=shared_d->gguf;
                const auto tokens=source.size(0);
                const bool position_major=down.prefers_position_major_q8(tokens);
                const auto bytes=position_major?mfq::cuda::gguf::interleaved_activation_bytes(tokens,shared_width):
                    tokens*mfq::cuda::gguf::q8_row_bytes(shared_width);
                auto quantized=tb::empty({int64_t(bytes)},source.options().dtype(tb::kUInt8));
                mfq::cuda::gguf::gate_up_swiglu_q8_f16(gu.data_ptr(),tokens,shared_width,quantized.data_ptr(),
                    position_major,mfq_current_cuda_stream());
                return scale_output(execution,source,down.forward_q8(quantized,tokens,source.scalar_type(),position_major));
            }
            auto shape=source.sizes().vec();shape.back()=shared_width;
            auto hidden=tb::empty(shape,source.options());
            mfq::cuda::gguf::gate_up_swiglu_f16(gu.data_ptr(),source.size(0),shared_width,hidden.data_ptr(),0.f,mfq_current_cuda_stream());
            auto output=sd(execution,hidden);
            return scale_output(execution,source,output);
        }
        auto unfused=[](const auto&...){return std::optional<Tensor>{};};
        auto output=mfq::models::gated_mlp(source,false,0.0,unfused,unfused,
            [&](Tensor input){return std::array<Tensor,2>{sg(execution,input),su(execution,input)};},
            [](Tensor g,Tensor u,mfq::models::GatedActivation,double){return (g*tb::sigmoid(g))*u;},
            [&](Tensor hidden){return sd(execution,hidden);},unfused);
        return scale_output(execution,source,output);
    };
    const std::array<Linear,3> shared_projections{sg,su,sd};
    const bool shared_row_parallel=std::all_of(shared_projections.begin(),shared_projections.end(),
        [](const auto& projection) {
            const auto weight=weight_loader::linear_weight(projection);
            return weight && weight->is_nint() && !weight->tensor_parallel() && !weight->nint.q8_zero;
        });
    auto shared=[shared_raw,shared_projections,shared_gate,shared_row_parallel](CudaExecutionContext& execution,const Tensor& source) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        const auto rows=qwen_prefill_reference_rows;
        const auto parallel_option=mfq::cuda::runtime_options::moe_prefill_shared_streams();
        const int streams=parallel_option?*parallel_option:0;
        const auto chain_option=mfq::cuda::runtime_options::moe_prefill_shared_chain_streams();
        const int chain_streams=chain_option?*chain_option:4;
        const auto tail=rows>0?source.size(0)%rows:0;
        if(chain_streams>0 && shared_row_parallel && rows>8 && source.size(0)>=8192 &&
                source.size(0)>rows && (tail==0 || tail>8)) {
            // The first original chunk prepares all three cached weight
            // matrices. Worker pipelines then read them after the ready fence.
            auto first=shared_raw(execution,source.narrow(0,0,rows));
            auto shape=first.sizes().vec();shape[0]=source.size(0);
            auto output=tb::empty(shape,first.options());
            output.narrow(0,0,rows).copy_(first);
            mfq::cuda::submit_parallel_rows(source.get_device(),rows,source.size(0),rows,chain_streams,
                [&](int64_t begin,int64_t count) {
                    auto part=shared_raw(execution,source.narrow(0,begin,count));
                    output.narrow(0,begin,count).copy_(part);
                });
            return output;
        }
        if(streams>0 && shared_row_parallel && rows>8 && source.size(0)>rows && (tail==0 || tail>8)) {
            // Each GEMM retains its original row count. Only its submission
            // stream changes; activations remain in their canonical FP16 form.
            const auto project=[&](int index,const Tensor& input) {
                const mfq::cuda::MatmulRowsScope gemm_rows(rows,streams);
                return shared_projections[index](execution,input);
            };
            auto gate=project(0,source),up=project(1,source);
            auto output=project(2,(gate*tb::sigmoid(gate))*up);
            auto scale=qwen_prefill_preserve_rows(source,[&](const Tensor& part) {
                return tb::sigmoid(shared_gate(execution,part));
            });
            return scale*output;
        }
#endif
        return qwen_prefill_preserve_rows(source,[&](const Tensor& part){return shared_raw(execution,part);});
    };
    MoeFfnSharedWeights shared_weights;
    shared_weights.projections={weight_loader::linear_weight(sg),weight_loader::linear_weight(su),weight_loader::linear_weight(sd)};
    shared_weights.scalar_gate=shared_gate_weight;
    shared_weights.gate=shared_scale;
    std::shared_ptr<TqMoeBanks> tq;
    if(!gate_up.tq_projections.empty() || !down.tq_projections.empty()) {
        TYLOINFER_RUNTIME_CHECK(gate_up.tq_projections.size()==2 && down.tq_projections.size()==1 &&
            gate_up.projections.empty() && down.projections.empty(),
            "TQ MoE requires separate Gate, Up and Down banks");
        tq=std::make_shared<TqMoeBanks>();
        tq->gate=gate_up.tq_projections[0];tq->up=gate_up.tq_projections[1];tq->down=down.tq_projections[0];
    }
    std::shared_ptr<GgufMoeBanks> gguf;
    std::shared_ptr<CudaIndependentBranchExecutor> shared_executor;
    if(!gate_up.gguf_projections.empty() || !down.gguf_projections.empty()) {
        TYLOINFER_RUNTIME_CHECK(gate_up.gguf_projections.size()==2 && down.gguf_projections.size()==1 &&
            gate_up.projections.empty() && down.projections.empty(),
            "GGUF MoE requires separate Gate, Up and Down banks");
        gguf=std::make_shared<GgufMoeBanks>();
        gguf->gate=gate_up.gguf_projections[0];gguf->up=gate_up.gguf_projections[1];gguf->down=down.gguf_projections[0];
    }
    if((tq || gguf) && loading.config.decode_branch_parallel && !loading.loading_cpu_layer) {
            const int device=active_weight_load_device(loading);
            const MfqCudaGuard guard(device);
            auto& per_device=loading.shared_expert_executors[device];
            if(!per_device)per_device=std::make_shared<CudaIndependentBranchExecutor>();
            shared_executor=per_device;
            shared_executor->ensure(1,mfq_get_current_cuda_stream(device));
    }
    MoeFfnForward pipeline;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if(!tq && !gguf)pipeline=make_moe_ffn_pipeline(gate_up.projections,down.projections,shared,{},std::move(shared_weights),prefetch);
#endif
    return [gate_up, down, router, shared, pipeline, tq, gguf, shared_executor, c](CudaExecutionContext &execution,
                                                               const Tensor &x) {
        auto source = x.reshape({-1, c.hidden}).to(tb::kFloat16);
        const auto layer_prefill = mfq::cuda::runtime_options::moe_prefill_layer();
        const bool combined_prefill = source.size(0)>8 &&
            execution.kl_mmq.mode==KlMmqMode::Default &&
            ((layer_prefill && *layer_prefill!=0 && moe_expert_cache_layer_prefill_ready(execution.moe_expert_cache)) ||
             moe_expert_cache_ssd_experts(execution.moe_expert_cache));
        const bool use_pipeline=pipeline && (gate_up.projections.size()!=1 ||
            combined_prefill ||
            ((source.size(0)<=8 || (source.size(0)<=32 && execution.continuous_batch_cache_serial)) &&
             execution.config.moe_two_stage_ffn &&
             execution.kl_mmq.mode==KlMmqMode::Default));
        const auto routing = c.routing();
        // The three-bank decode path publishes routes and starts missing
        // expert transfers before scheduling this independent shared branch.
        const bool overlap_shared=(gguf || tq) && source.size(0)<=32;
        Tensor deferred_shared;
        return mfq::models::mixture_of_experts(
            [&] { return qwen_prefill_preserve_rows(source,[&](const Tensor& part){return router(execution,part);})
                .to(tb::kFloat32).contiguous(); },
            [&](Tensor logits) {
                return moe_topk_cuda(
                    logits, c.topk, routing.activation == mfq::models::RouterActivation::sigmoid,
                    routing.activation == mfq::models::RouterActivation::sqrt_softplus,
                    routing.normalize, routing.delayed_softmax, mfq_nullopt, 1e-20, routing.scale);
            },
            [&](const auto &) {
                return use_pipeline || overlap_shared ? Tensor{} : shared(execution,source);
            },
            [&](const auto &selected) {
                if(use_pipeline)return pipeline(execution,source,selected[0].to(tb::kInt32).contiguous(),selected[1]);
                if(tq)return execution.profiler.measure("tq.moe",[&] {
                    const std::function<void()> work=overlap_shared?std::function<void()>([&] {
                        deferred_shared=shared(execution,source);
                    }):std::function<void()>{};
                    auto output=tq->forward_parallel(execution,source,selected[0],selected[1],0.f,work,
                        execution.config.decode_branch_parallel?shared_executor.get():nullptr);
                    if(deferred_shared.defined())mfq_cuda_record_stream(deferred_shared,mfq_get_current_cuda_stream(source.get_device()));
                    return output;
                });
                if(gguf)return execution.profiler.measure("gguf.moe",[&] {
                    const std::function<void()> work=overlap_shared?std::function<void()>([&] {
                        deferred_shared=shared(execution,source);
                    }):std::function<void()>{};
                    auto output=gguf->forward(source,selected[0],selected[1],0.f,work,
                        execution.config.decode_branch_parallel?shared_executor.get():nullptr);
                    if(deferred_shared.defined())mfq_cuda_record_stream(deferred_shared,mfq_get_current_cuda_stream(source.get_device()));
                    return output;
                });
                return mfq::models::routed_experts(
                    [&] { return gate_up(execution, source, selected[0]); },
                    [&](Tensor gu) {
                        auto gate = gu.narrow(-1, 0, c.moe_width),
                             up = gu.narrow(-1, c.moe_width, c.moe_width);
                        return (gate * tb::sigmoid(gate)) * up;
                    },
                    [&](Tensor hidden) { return down(execution, hidden, selected[0]); },
                    [&](Tensor pairs) {
                        auto reduced = tb::zeros({source.size(0), c.hidden},
                                                 source.options().dtype(tb::kFloat32));
                        for (int64_t r = 0; r < c.topk; ++r)
                            reduced = reduced + pairs.select(1, r).to(tb::kFloat32) *
                                                    selected[1].select(1, r).unsqueeze(-1);
                        return reduced.to(pairs.scalar_type());
                    });
            },
            [&](Tensor routed, Tensor shared_output, const auto &) {
                if(use_pipeline)return routed.reshape(x.sizes());
                if(overlap_shared)shared_output=std::move(deferred_shared);
                return (routed + shared_output).reshape(x.sizes());
            });
    };
}

struct BlockLoader : weight_loader::Loader {
    using GdnWeights = qwen4_exp::GdnWeights;
    using QsaWeights = qwen4_exp::QsaWeights;
    using PleWeights = qwen4_exp::PleWeights;
    Linear* router_capture=nullptr;
    MoeFfnPrefetch* prefetch_capture=nullptr;
    bool* tq_graph_capture=nullptr;
    int64_t dense_attention_window=0;
    std::shared_ptr<FFN>* dense_ffn_weights=nullptr;
    struct Embedding {
        attention_ops::Embedding function;
        std::shared_ptr<mfq::NintRows> range;
        std::shared_ptr<TqRowTable> tq;
        std::shared_ptr<GgufRowTable> gguf;
        Tensor operator()(const Tensor& ids) const { return function(ids); }
    };

    auto residual_linear(const std::string &name) const {
        return weight_loader::residual_linear(execution, source, name);
    }
    Gr residual(Tensor norm, weight_loader::ResidualLinear down,
        weight_loader::ResidualLinear up, weight_loader::ResidualLinear injection, const Config &c) {
        if(c.streams==1)norm=norm+source.legacy_tensor_compatibility().layout.norm_weight_offset;
        auto combined=weight_loader::residual_down_injection(down,injection);
        return {std::move(norm), std::move(down), std::move(up), std::move(injection),
                c.hidden, c.streams, c.eps,std::move(combined)};
    }
    static auto final_mixer(Gr weights) { return std::make_unique<Gr>(std::move(weights)); }
    auto gdn_weights(const std::string &p) const {
        const bool independent = has(p + ".query.weight") || has(p + ".key.weight") || has(p + ".value.weight");
        TYLOINFER_RUNTIME_CHECK(!independent || !has(p + ".qkv.weight"), "ambiguous Qwen4 GDN input projections");
        GdnWeights w{independent ? Linear{} : linear(p + ".qkv.weight"), linear(p + ".gate.weight"),
            linear(p + ".alpha.weight"), linear(p + ".beta.weight"), linear(p + ".output.weight"),
            dense(p + ".conv.weight"), dense(p + ".dt_bias"), dense(p + ".a"), dense(p + ".norm.weight")};
        if(!source.legacy_tensor_compatibility().layout.linear_attention_a_is_log)w.a_log=tb::log(-w.a_log);
        if (independent)
            for (const auto* name : {"query", "key", "value"})
                w.split_qkv.push_back(linear(p + "." + name + ".weight"));
        return w;
    }
    auto qsa_weights(const std::string &p) const {
        Linear index;
        if(has(p+".indexer.query_key.weight"))index=linear(p+".indexer.query_key.weight");
        else {
            auto query=linear(p+".indexer.query.weight"),key=linear(p+".indexer.key.weight");
            index=[query,key](CudaExecutionContext& e,const Tensor& x) {
                const auto input=x.to(tb::kFloat32);
                return tb::cat({query(e,input),key(e,input)},-1);
            };
        }
        QsaWeights w{linear(p + ".query.weight"), linear(p + ".key.weight"), linear(p + ".value.weight"),
            linear(p + ".output.weight"), std::move(index), dense(p + ".query_norm.weight"),
            dense(p + ".key_norm.weight"), dense(p + ".indexer.query_norm.weight"), dense(p + ".indexer.key_norm.weight")};
        const auto offset=source.legacy_tensor_compatibility().layout.norm_weight_offset-1.;
        if(offset)for(auto* norm:{&w.query_norm,&w.key_norm,&w.index_query_norm,&w.index_key_norm})
            *norm=norm->to(tb::kFloat32)+offset;
        if(has(p+".indexer.pre_key_norm.weight"))
            w.index_pre_key_norm=fp32(dense(p+".indexer.pre_key_norm.weight"));
        if (has(p + ".gate.weight")) w.gate = linear(p + ".gate.weight");
        return w;
    }
    auto gdn(GdnWeights weights, const Config &c) const {
        std::vector<Linear> projections = weights.split_qkv.empty()
            ? std::vector<Linear>{weights.qkv} : weights.split_qkv;
        const bool independent = !weights.split_qkv.empty();
        projections.insert(projections.end(), {weights.gate,weights.alpha,weights.beta});
        const auto packed_qkv_option=mfq::cuda::runtime_options::gdn_packed_qkv();
        const bool packed_qkv=independent && (!packed_qkv_option || *packed_qkv_option);
        auto group = weight_loader::grouped_linear(execution,projections,packed_qkv?3:1);
        if (independent) {
            if(group && packed_qkv)weights.input_projection=std::move(group);
            else if (group)
                weights.input_projection = [group](CudaExecutionContext &e, const Tensor &x) {
                    auto p = group(e,x);
                    return std::vector<Tensor>{tb::cat({p[0],p[1],p[2]},-1),p[3],p[4],p[5]};
                };
            weights.qkv = [q=projections[0],k=projections[1],v=projections[2]](CudaExecutionContext &e, const Tensor &x) {
                return tb::cat({q(e,x),k(e,x),v(e,x)},-1);
            };
        } else {
            weights.input_projection = std::move(group);
            weights.qkv = std::move(projections[0]);
        }
        const auto start = independent ? 3 : 1;
        weights.gate=std::move(projections[start]);weights.alpha=std::move(projections[start+1]);
        weights.beta=std::move(projections[start+2]);
        if(const auto weight=weight_loader::linear_weight(weights.output);weight && weight->is_gguf() &&
                !weight->tensor_parallel() && weight->gguf.packed.is_cuda() && !weight->gguf.prefers_original_input()) {
            weights.output_q8_position_major=[weight](int64_t rows){return weight->gguf.prefers_position_major_q8(rows);};
            weights.output_q8=[weight](CudaExecutionContext&,const Tensor& input,int64_t tokens,tb::ScalarType dtype,bool position_major) {
                return weight->gguf.forward_q8(input,tokens,dtype,position_major).reshape({1,tokens,weight->gguf.out});
            };
        }
        weights.input_projection=qwen_gdn_preserve_projection_rows(std::move(weights.input_projection));
        weights.qkv=qwen_gdn_preserve_projection_rows(std::move(weights.qkv));
        weights.gate=qwen_gdn_preserve_projection_rows(std::move(weights.gate));
        weights.alpha=qwen_gdn_preserve_projection_rows(std::move(weights.alpha));
        weights.beta=qwen_gdn_preserve_projection_rows(std::move(weights.beta));
        weights.output=qwen_gdn_preserve_projection_rows(std::move(weights.output));
        weights.input_projection=qwen_gdn_profile_projection(std::move(weights.input_projection),
            "gdn_input_begin","gdn_input_end");
        weights.output=qwen_gdn_profile_projection(std::move(weights.output),
            "gdn_output_begin","gdn_output_end");
        weights.output_q8=qwen_gdn_profile_projection(std::move(weights.output_q8),
            "gdn_output_begin","gdn_output_end");
        auto gdn=std::make_unique<Gdn>(std::move(weights),c.key_heads,c.value_heads,c.linear_width,
                                      c.kernel,c.eps,c.silu_gate,true,execution.config.gdn_transposed_state,
                                      execution.config.gdn_fused_output,true,execution.config.gdn_fused_preparation,
                                      execution.config.gdn_fused_core,c.linear_tiled_heads);
        // Long-lived dense recurrence should not pin temporary prefill pages.
        if(!c.experts)gdn->initialize_request_cache(false);
        return gdn;
    }
    auto qsa(QsaWeights weights, const Config &c) const {
        std::vector<Linear> projections{weights.query,weights.key,weights.value,weights.index_query_key};
        if (weights.gate) projections.push_back(weights.gate);
        auto group = weight_loader::grouped_linear(execution,projections);
        std::vector<Linear> history{projections[1],projections[2],projections[3]};
        weights.history_projection=weight_loader::grouped_linear(execution,history,1,false);
        if (weights.gate) {
            const auto heads = c.heads, width = c.width;
            const auto combine = [heads,width](const Tensor &q,const Tensor &g) {
                const auto b=q.size(0),t=q.size(1);
                return tb::cat({q.reshape({b,t,heads,width}),g.reshape({b,t,heads,width})},-1)
                    .reshape({b,t,2*heads*width});
            };
            if (group)
                weights.input_projection = [group,combine](CudaExecutionContext &e,const Tensor &x) {
                    auto p=group(e,x);p[0]=combine(p[0],p[4]);p.resize(4);return p;
                };
            projections[0] = [q=projections[0],g=projections[4],combine](CudaExecutionContext &e,const Tensor &x) {
                return combine(q(e,x),g(e,x));
            };
        } else weights.input_projection=std::move(group);
        weights.query=std::move(projections[0]);weights.key=std::move(projections[1]);
        weights.value=std::move(projections[2]);weights.index_query_key=std::move(projections[3]);
        QsaConfig geometry{c.heads, c.kv_heads, c.width, c.index_heads, c.index_width,
                           c.pool, c.budget, c.maximum, c.eps};
        geometry.use_spare_vram=!c.experts;
        if(dense_attention_window) {
            TYLOINFER_RUNTIME_CHECK(c.maximum<=INT64_MAX-(c.pool-1),"dense MTP context budget overflows");
            geometry.budget=(c.maximum+c.pool-1)/c.pool*c.pool;
            geometry.dense_window=std::min(c.maximum,dense_attention_window);
        }
        auto rotary = std::make_shared<RotaryEmbedding>(c.rotary, c.maximum, c.rope_base, c.sections, c.interleaved,std::nullopt,execution.yarn);
        return std::make_unique<Qsa>(std::move(weights), geometry, std::move(rotary), true, true, execution.qsa_kv_store, execution.kv_memory);
    }
    auto dense_ffn(const std::string& p,const Config& c) const {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        std::optional<DenseWeightAllocationScope> residency;
        if(dense_ffn_weights && !execution.loading_cpu_layer && !model_parallel_enabled(execution) &&
                !execution.layer_placement.enabled() && mfq::gguf::is_quantized(require_tensor(source,p+".down.weight").dtype))
            residency.emplace();
#endif
        auto weights=std::make_shared<FFN>();
        weights->down=load_quant_linear(execution,source,p+".down.weight");
        weights->gate_up=load_paired_gate_up(execution,source,{p+".gate.weight",p+".up.weight"},weights->down);
        TYLOINFER_RUNTIME_CHECK(weights->down.out()==c.hidden && weights->down.neuron_len()==c.moe_width,
            "Dense Qwen4 FFN geometry differs from its config");
        prepare_ffn_workspaces(execution,*weights);
        if(dense_ffn_weights)*dense_ffn_weights=weights;
        if(tq_graph_capture)*tq_graph_capture=!execution.loading_cpu_layer && !execution.tensor_parallel.enabled();
        return Linear{[weights](CudaExecutionContext& e,const Tensor& x){return weights->forward(e,x);}};
    }
    auto moe(weight_loader::Routed gate_up, weight_loader::Routed down, Linear router,
                    Linear shared_gate, Linear sg, Linear su, Linear sd, const Config &c) const {
        if(router_capture)*router_capture=router;
        if(tq_graph_capture)*tq_graph_capture=
            (gate_up.tq_projections.size()==2 && down.tq_projections.size()==1) ||
            (gate_up.gguf_projections.size()==2 && down.gguf_projections.size()==1);
        return qwen_moe(execution,std::move(gate_up), std::move(down), std::move(router), std::move(shared_gate),
                        std::move(sg), std::move(su), std::move(sd), c, prefetch_capture);
    }
    auto embedding(const std::string &name) const {
        const auto& dtype=require_tensor(source,name).dtype;
        const bool row_storage=mfq::gguf::is_quantized(dtype) || dtype=="F16" || dtype=="BF16" || dtype=="F32";
        if (row_storage && name.find("position_embedding.ngram.") != std::string::npos) {
            auto table = std::make_shared<GgufRowTable>(source, name);
            Embedding lookup{[table](const Tensor& ids) { return table->lookup(ids); }, {}, {}, table};
            return std::pair{std::move(lookup), std::array<int64_t,2>{table->rows(), table->width()}};
        }
        if(is_tq_row_table(source,name)) {
            auto table=std::make_shared<TqRowTable>(source,name);
            Embedding lookup{[table](const Tensor& ids){return table->lookup(ids);},{},table};
            return std::pair{std::move(lookup),std::array<int64_t,2>{static_cast<int64_t>(table->rows()),table->width()}};
        }
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if (require_tensor(source, name).dtype == "NINT") {
            auto table = load_nint_row_table(source, name);
            Embedding lookup{[table](const Tensor &ids) {
                return nint_row_embedding_lookup(*table, ids);
            },table};
            return std::pair{std::move(lookup), std::array<int64_t, 2>{table->rows(), table->width()}};
        }
#endif

        auto weight = std::make_shared<QuantLinear>(load_quant_linear(execution, source, name));
        Embedding lookup{[weight](const Tensor &ids) { return quant_embedding_lookup(*weight, ids); },{}};
        return std::pair{std::move(lookup), std::array<int64_t, 2>{weight->out(), weight->neuron_len()}};
    }
    std::vector<int64_t> integers(const std::string &name) const {
        const auto &type = require_tensor(source, name).dtype;
        TYLOINFER_RUNTIME_CHECK(type == "I64" || type == "I32",
                          "Qwen4 PLE metadata must be a dense integer array: ", name);
        auto host = load_dense_gpu(execution, source, name).to(tb::kInt64).contiguous().cpu();
        TYLOINFER_RUNTIME_CHECK(host.dim() == 1, "Qwen4 PLE metadata must be a vector");
        return {host.data_ptr<int64_t>(), host.data_ptr<int64_t>() + host.numel()};
    }
    static auto ple(std::vector<Embedding> shards, int64_t rows, int64_t width, const Config &c,
                    std::vector<int64_t> multipliers, std::vector<int64_t> offsets,
                    std::vector<int64_t> sizes, PleWeights weights) {
        std::vector<attention_ops::Embedding> functions;std::vector<std::shared_ptr<mfq::NintRows>> ranges;
        std::vector<std::shared_ptr<TqRowTable>> tq;
        std::vector<std::shared_ptr<GgufRowTable>> gguf;
        for(auto& shard:shards){functions.push_back(std::move(shard.function));ranges.push_back(std::move(shard.range));tq.push_back(std::move(shard.tq));gguf.push_back(std::move(shard.gguf));}
        std::shared_ptr<NintRowPipeline> pipeline;
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if(std::all_of(ranges.begin(),ranges.end(),[](const auto& p){return bool(p);})) {
            pipeline=std::make_shared<NintRowPipeline>(std::move(ranges));
        }
#endif
        NgramEmbedding::RowLookup row_lookup;
        NgramEmbedding::StageFactory stage_factory;
        NgramEmbedding::RowPrepare row_prepare;
        NgramEmbedding::StagePlan stage_plan;
        if (std::all_of(gguf.begin(), gguf.end(), [](const auto& p) { return bool(p); })) {
            row_lookup = [gguf, rows](const auto& ids, const auto& device) {
                return lookup_gguf_shards(gguf, rows, ids, device);
            };
            stage_plan=[gguf](const auto& shape,const auto& device) {return gguf_row_stage_resources(gguf,shape,device);};
            stage_factory = [gguf, rows](const auto& shape, const auto& device) {
                return make_gguf_row_stage(gguf, rows, shape, device);
            };
            row_prepare = [gguf,rows](const auto& ids,const auto& shape,const auto& device) {
                return prefetch_gguf_shards(gguf,rows,ids,shape,device);
            };
        }
        if(std::all_of(tq.begin(),tq.end(),[](const auto& p){return bool(p);})) {
            row_lookup=[tq,rows](const auto& ids,const auto& device) {
                return lookup_tq_shards(tq,rows,ids,device);
            };
            row_prepare=[tq,rows](const auto& ids,const auto& shape,const auto& device) {
                return prefetch_tq_shards(tq,rows,ids,shape,device);
            };
            stage_factory=[tq=std::move(tq),rows](const auto& shape,const auto& device) {
                return make_tq_row_stage(tq,rows,shape,device);
            };
        }
        NgramEmbedding embedding(std::move(functions), rows, width, c.ngram, c.ngram_heads, c.eos,
                                     std::move(multipliers), std::move(offsets), std::move(sizes),std::move(pipeline),std::move(row_lookup),std::move(stage_factory),std::move(row_prepare),std::move(stage_plan));
        return std::make_unique<Ple>(std::move(embedding), std::move(weights), c.hidden, c.streams,
                                     c.ngram, c.eps);
    }
    std::unique_ptr<Qwen4Block> block(const Config &c, int layer, bool predictor);
};

struct RecurrentGraphCheckpoint {
    std::vector<Tensor> gdn_conv,gdn_state,ple_conv;
    bool gdn_replay=false;
    std::vector<Tensor> gdn_inputs;
};

struct Qwen4Block final : Block {
    using Tensor = mfq_tensor_backend::Tensor;
    mfq::models::qwen4_exp::Config config;
    int capture_layer = -1;
    std::shared_ptr<Qwen4HiddenCapture> hidden_capture;
    Gr attention_gr, ffn_gr;
    Linear ffn,router;
    std::shared_ptr<FFN> dense_ffn_weights;
    Linear next_router;
    MoeFfnPrefetch prefetch,next_prefetch;
    bool tq_graph_ffn=false;
    std::shared_ptr<RouterLookaheadInputs> router_audit;
    std::unique_ptr<Gdn> gdn;
    std::unique_ptr<Qsa> qsa;
    std::unique_ptr<Ple> ple;
    Tensor batch_ffn(CudaExecutionContext& execution,const Tensor& input) {
        if(input.size(0)<=8)return ffn(execution,input);
        const auto option=mfq::cuda::runtime_options::continuous_ffn_tile();
        const int64_t rows=option?*option:8;
        TYLOINFER_RUNTIME_CHECK(rows==8 || rows==16,"continuous FFN tile must be 8 or 16 rows");
        auto* window=DecodeWindow::recording();
        const int previous_profile_tile=window ? window->set_profile_tile(-1) : -1;
        struct TileScope {
            CudaExecutionContext& execution;
            int tile,tiles;
            DecodeWindow* window;
            int profile_tile;
            ~TileScope() {execution.continuous_batch_ffn_tile=tile;execution.continuous_batch_ffn_tiles=tiles;
                if(window)window->set_profile_tile(profile_tile);}
        } scope{execution,execution.continuous_batch_ffn_tile,execution.continuous_batch_ffn_tiles,window,previous_profile_tile};
        execution.continuous_batch_ffn_tiles=int((input.size(0)+rows-1)/rows);
        std::vector<Tensor> outputs;
        for(int64_t start=0;start<input.size(0);start+=rows) {
            execution.continuous_batch_ffn_tile=int(start/rows);
            if(window)window->set_profile_tile(int(start/rows));
            outputs.push_back(ffn(execution,input.narrow(0,start,std::min(rows,input.size(0)-start))));
        }
        return tb::cat(outputs,0);
    }
    Qwen4Block(CudaExecutionContext &execution, const mfq::ModelSource &file,
               const Config &c, int layer, bool predictor = false) {
        BlockLoader loader{{execution, file, "qwen4_exp"},&router,&prefetch,&tq_graph_ffn};
        if(!predictor)loader.dense_ffn_weights=&dense_ffn_weights;
        if(predictor)if(const auto window=runtime_options::mtp_attention_window())
            loader.dense_attention_window=*window?*window:c.maximum;
        capture_layer = layer;
        mfq::models::qwen4_exp::load_block(*this, loader, c, layer, predictor);
    }
    Qwen4Block(const Qwen4Block& source, std::unique_ptr<Qsa> attention)
        : Block(source), config(source.config), capture_layer(source.capture_layer),
          attention_gr(source.attention_gr), ffn_gr(source.ffn_gr), ffn(source.ffn),
          prefetch(source.prefetch), tq_graph_ffn(source.tq_graph_ffn), qsa(std::move(attention)) {}
    bool retain_graph_storage=false,retained_prefix=false;
    void reset(int64_t) override {
        if (gdn)
            gdn->reset(retain_graph_storage);
        if (qsa)
            qsa->reset();
        if (ple)
            ple->reset(retain_graph_storage);
    }
    bool supports_speculation() const noexcept override { return true; }
    void prefetch_token_ids(const Tensor& ids) override { if(ple)ple->prefetch_tokens(ids); }
    void commit_speculative() override {
        if (gdn)
            gdn->commit();
        if (ple)
            ple->commit();
    }
    void rollback_speculative(int64_t keep) override {
        if (gdn)
            gdn->rollback(keep);
        if (qsa)
            qsa->truncate(keep);
        if (ple)
            ple->rollback(keep);
    }
    Tensor execute(CudaExecutionContext &execution, Tensor x, const Tensor &ids,
                   const Tensor &positions, const Tensor &full_positions, int64_t confirmed = 0,const Tensor& host_ids = {}) {
        retained_prefix=confirmed>0 && ids.size(1)<=8;
        auto result = mfq::models::qwen4_exp::decoder_layer_chained(
            std::move(x), bool(ple), bool(gdn),
            [&](const Tensor &hidden) {
                return execution.profiler.measure("prefill.ple",[&] {
                    return ple->forward(execution, hidden, ids, true, confirmed,host_ids,
                        full_positions.size(-1)-ids.size(1));
                });
            },
            [](Tensor hidden, Tensor positional) { return hidden + positional; },
            [&](const Tensor &hidden) { return execution.profiler.measure("prefill.gr_attention",[&] {return attention_gr.pre(execution, hidden);}); },
            [&](Tensor branch) { return execution.profiler.measure("prefill.gdn",[&] {return gdn->forward(execution, branch, true, confirmed,
                full_positions.size(-1)-ids.size(1));}); },
            [&](Tensor branch) {
                return execution.profiler.measure("prefill.qsa",[&] {return qsa->forward(execution, branch, positions, full_positions, true);});
            },
            [&](Tensor branch, const auto &mix) { return execution.profiler.measure("prefill.gr_ffn_pre",[&] {return ffn_gr.pre_after(execution, branch, mix);}); },
            [&](Tensor branch) { return execution.profiler.measure("prefill.ffn",[&] {return ffn(execution, branch);}); },
            [&](Tensor branch, const auto &mix) { return execution.profiler.measure("prefill.gr_ffn_post",[&] {return ffn_gr.post(branch, mix);}); });
        if (hidden_capture) hidden_capture->capture(capture_layer, result, config.streams);
        return result;
    }
    Tensor forward(CudaExecutionContext &, Tensor, Tensor, int64_t, const MfqOptional<Tensor> &,
                   const RopeCache &, const MfqOptional<Tensor> & = mfq_nullopt,
                   const MfqOptional<Tensor> & = mfq_nullopt) override {
        throw std::runtime_error(
            "Qwen4 block requires the unified model position/PLE input lifecycle");
    }
    Tensor forward_context(CudaExecutionContext &execution, Tensor x, const Context &context,
                           const RopeCache &) override {
        return execute(execution, std::move(x), context.token_ids, context.positions,
                       context.full_positions, context.confirmed_prefix,context.host_token_ids);
    }
    Tensor prefill_ple(CudaExecutionContext& execution,Tensor x,const Context& context) {
        if(ple)x=x+execution.profiler.measure("prefill.ple",[&] {
            return ple->forward(execution,x,context.token_ids,true,0,context.host_token_ids);
        });
        return x;
    }
    Tensor prefill_attention_core(CudaExecutionContext& execution,const Tensor& input,const Context& context) {
        return gdn?
            execution.profiler.measure("prefill.gdn",[&] {return gdn->forward(execution,input,true,0);}):
            execution.profiler.measure("prefill.qsa",[&] {return qsa->forward(execution,input,context.positions,context.full_positions,true);});
    }
    std::vector<Tensor> prefill_attention(CudaExecutionContext& execution,Tensor x,const Context& context) {
        x=prefill_ple(execution,std::move(x),context);
        auto mix=execution.profiler.measure("prefill.gr_attention",[&] {return attention_gr.pre(execution,x);});
        auto branch=prefill_attention_core(execution,mix[0],context);
        return execution.profiler.measure("prefill.gr_ffn_pre",[&] {return ffn_gr.pre_after(execution,branch,mix);});
    }
    std::vector<Tensor*> graph_warmup_state() override {
        std::vector<Tensor*> result;
        if(gdn)for(auto* value:gdn->graph_state())if(value->defined())result.push_back(value);
        if(ple)for(auto* value:ple->graph_state())if(value->defined())result.push_back(value);
        return result;
    }
    Tensor execute_graph(CudaExecutionContext& execution,Tensor x,const Tensor& ids,
                         const Tensor& positions,const Tensor& cache_positions,const Tensor& ple_rows,int layer,
                         int64_t confirmed=0,RecurrentGraphCheckpoint* checkpoint=nullptr,
                         const std::vector<Tensor>& prior_mix={},std::vector<Tensor>* deferred_mix=nullptr,
                         bool history_tail=false,Tensor* attention_capture=nullptr,Tensor* ffn_input_capture=nullptr,
                         Tensor* attention_input_capture=nullptr) {
        TYLOINFER_RUNTIME_CHECK(!history_tail || (qsa && !gdn && !ple && !confirmed && prior_mix.empty() && !deferred_mix),
            "MTP history tail requires an independent QSA-only predictor block");
        if(auto* window=DecodeWindow::recording())window->set_profile_layer_kind(layer,gdn?"gdn":"qsa");
        const auto mark=[&](const char* phase){if(auto* window=DecodeWindow::recording())window->mark(layer,phase);};
        const bool detailed=mfq::cuda::runtime_options::mhc_timings();
        const auto marker=[&](const char* role)->GatedResidualMarker {
            if(!detailed || !DecodeWindow::recording())return {};
            return [&,role](const char* phase) {
                const auto name=std::string(role)+phase;mark(name.c_str());
            };
        };
        auto result = mfq::models::qwen4_exp::decoder_layer_chained(std::move(x),bool(ple),bool(gdn),
            [&](const Tensor& hidden){return confirmed
                ? ple->forward_graph(execution,hidden,ids,ple_rows,confirmed,checkpoint->ple_conv)
                : ple->forward_chunk(execution,hidden,ids,true,{},ple_rows);},
            [](Tensor hidden,Tensor positional){return hidden+positional;},
            [&](const Tensor& hidden){auto result=prior_mix.empty()?
                attention_gr.pre(execution,hidden,marker("attention_mhc_")):
                attention_gr.pre_after(execution,hidden,prior_mix,marker("attention_mhc_"));
                mark("attention_core_begin");return result;},
            [&](Tensor branch){auto result=confirmed
                ? gdn->forward_graph(execution,branch,confirmed,checkpoint->gdn_conv,checkpoint->gdn_state,
                    checkpoint->gdn_replay?&checkpoint->gdn_inputs:nullptr)
                : gdn->forward_chunk(execution,branch,true);mark("attention_core_end");return result;},
            [&](Tensor branch){
                auto result=qsa->forward_graph(execution,branch,positions,cache_positions,{},!attention_input_capture,false,history_tail);
                if(attention_input_capture){*attention_input_capture=result;result=qsa->project_output(execution,result);}
                if(attention_capture)*attention_capture=result;
                mark("attention_core_end");return result;
            },
            [&](Tensor branch,const auto& mix){
                auto tail=mix;
                if(history_tail)for(auto& value:tail)if(value.defined())value=value.narrow(1,value.size(1)-1,1);
                auto result=ffn_gr.pre_after(execution,branch,tail,marker("ffn_mhc_"));
                if(ffn_input_capture)*ffn_input_capture=result[0];
                mark("ffn_pre_end");return result;
            },
            [&](Tensor branch) {
                DecodeWindow::Task prediction{};
                if(next_prefetch && branch.numel()==config.hidden && DecodeWindow::recording()->prediction_enabled()) {
                    const auto routing=config.routing();
                    auto logits=next_router(execution,branch.reshape({1,config.hidden}).to(tb::kFloat16)).to(tb::kFloat32).contiguous();
                    const int prediction_count=std::min<int64_t>(logits.size(-1),
                        moe_expert_cache_ssd_experts(execution.moe_expert_cache)?4:2);
                    auto selected=moe_topk_cuda(logits,prediction_count,
                        routing.activation==mfq::models::RouterActivation::sigmoid,
                        routing.activation==mfq::models::RouterActivation::sqrt_softplus,
                        routing.normalize,routing.delayed_softmax,mfq_nullopt,1e-20,routing.scale);
                    prediction=next_prefetch(selected[0].to(tb::kInt32).contiguous());
                }
                if(router_audit)router_audit->input(layer,branch);
                const bool early_prediction=prediction.identity && moe_expert_cache_ssd_experts(execution.moe_expert_cache);
                if(early_prediction)DecodeWindow::recording()->enroll(std::move(prediction));
                auto result=ffn(execution,branch);
                // Reserve current-layer readers before selecting speculative
                // slots. The route export itself precedes current FFN math.
                if(prediction.identity && !early_prediction)DecodeWindow::recording()->enroll(std::move(prediction));
                return result;
            },
            [&](Tensor branch,const auto& mix){
                if(deferred_mix){*deferred_mix=mix;return branch;}
                return ffn_gr.post(branch,mix);});
        if (hidden_capture) hidden_capture->capture(capture_layer, result, config.streams);
        return result;
    }
};

std::unique_ptr<Qwen4Block> BlockLoader::block(const Config &c, int layer, bool predictor) {
    return std::make_unique<Qwen4Block>(execution, source, c, layer, predictor);
}

// One word changes the captured readout's work without retaining a second
// graph or its scratch. DecodeWindow waits for the previous replay before the
// host changes this flag. Greedy requests do not consume compact top-k.
struct MtpCompactControl {
    mfq::cuda::HostBuffer host{128};
    Tensor storage=tb::zeros({1},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt32));
    const volatile int32_t* device=nullptr;
    int previous=-1;
    MtpCompactControl() {
        device=storage.data_ptr<int32_t>();
        mfq::cuda::charge_tensor_host_bytes(host.size());
    }
    ~MtpCompactControl(){mfq::cuda::tensor_host_bytes.fetch_sub(host.size());}
    void set(bool enabled,const char* role,int rows) {
        if(previous!=int(enabled)) {
            *static_cast<int32_t*>(host.data())=enabled?1:0;
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(storage.data_ptr(),host.data(),sizeof(int32_t),
                cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
            std::cerr<<"runtime_mtp_compact_control role="<<role<<" rows="<<rows<<" enabled="<<int(enabled)<<'\n';
            previous=int(enabled);
        }
    }
};
struct Qwen4MtpGraph {
    struct Cell {
        bool used_in_request=false;
        int profile_replays=0;
        Tensor ids,hidden,positions,cache_positions,normalized,multi,logits,greedy_host,compact_host,sampling_host,sampling_device;
        Tensor check_attention,check_ffn_input,check_attention_input;
        int64_t* sampling_mapped=nullptr;
        MtpCompactControl compact_control,probability_control;
        DecodeWindow window;
        Cell(const Tensor& source,const Tensor& token_ids,cudaStream_t stream)
            :ids(tb::empty_like(token_ids)),hidden(tb::empty_like(source)),
             positions(tb::empty({3,1,source.size(1)},token_ids.options().dtype(tb::kInt32))),
             cache_positions(tb::empty({source.size(1)},token_ids.options().dtype(tb::kInt64))),
             window(stream,mfq::cuda::runtime_options::layer_timings()) {
            // One packet and one readback serve greedy and stochastic requests.
            // The compact selector remains replay-controlled, without a second
            // unconditional D2H node for every greedy draft.
            sampling_host=tb::zeros({65},token_ids.options().device(tb::kCPU).dtype(tb::kInt64).pinned_memory(true));
            sampling_device=tb::zeros({65},token_ids.options().dtype(tb::kInt64));
            compact_host=sampling_host.narrow(0,0,64).reshape({1,64});
            greedy_host=sampling_host.narrow(0,64,1);
            const auto* mapped_option=std::getenv("TYLOINFER_MTP_MAPPED_PACKET");
            if(!mapped_option || mapped_option[0]!='0') {
                void* mapped=nullptr;
                if(cudaHostGetDevicePointer(&mapped,sampling_host.data_ptr(),0)==cudaSuccess)
                    sampling_mapped=static_cast<int64_t*>(mapped);
                else (void)cudaGetLastError();
            }
            window.set_profile_geometry("mtp_predictor",static_cast<int>(source.size(1)));
        }
    };
    MfqCudaStream stream=mfq_get_current_cuda_stream();
    std::unique_ptr<RoutedGraphPools> memory_pool;
    std::map<std::tuple<int,int,int,int>,std::unique_ptr<Cell>> cells;
    Tensor position_storage;
    int64_t capacity=0;
    bool failed=false;
    Tensor last_compact;
    Tensor queued_host,queued_ids;
    int32_t* queued_mapped=nullptr;
    bool reusable(const Qwen4ExpMtp& predictor) const {
        return predictor.reuse_decode_graphs && !failed && streamed_decode_graphs(*predictor.execution) &&
            !predictor.layers.empty() && predictor.layers[0]->qsa && predictor.layers[0]->qsa->graph_capacity()>0;
    }
    void restore_legacy(Qwen4ExpMtp& predictor) {
        last_compact={};cells.clear();predictor.layers[0]->qsa->leave_graph();
    }
    MtpStep run(Qwen4ExpMtp& predictor,const MtpTarget& target,const Tensor& hidden,const Tensor& ids,
               const Tensor& positions = {},bool history_only=false,bool scalar_history=false,bool queued=false,
               bool history_tail=false) {
        using ProfileTime=std::chrono::steady_clock;
        const auto profile_started=ProfileTime::now();
        const auto profile_transfer_before=routed_expert_streamed_bytes(*predictor.execution);
        TYLOINFER_RUNTIME_CHECK(!failed && stream.stream()==mfq_current_cuda_stream(),"MTP predictor graph is unavailable");
        auto& execution=*predictor.execution;auto& block=*predictor.layers[0];
        last_compact={};
        // Predictor prefill may retain a multi-GiB expert image. Retire it
        // before graph replay so the first draft does not page graph buffers.
        release_moe_prefill_buffers(execution.moe_expert_cache);
        const auto profile_released=ProfileTime::now();
        const int64_t tokens=ids.size(1),start=predictor.lengths[0],required=start+tokens;
        TYLOINFER_RUNTIME_CHECK(required<=predictor.config.maximum,"MTP predictor exceeds context capacity");
        if(block.qsa->graph_capacity()) {
            if(execution.kv_memory)execution.kv_memory->ensure(required);
            capacity=block.qsa->graph_capacity();
        }
        if(required>capacity) {
            finish_routed_expert_exchanges(execution);restore_legacy(predictor);
            capacity=std::min<int64_t>(predictor.config.maximum,std::max<int64_t>(2048,capacity));
            while(capacity<required)capacity=std::min(predictor.config.maximum,capacity*2);
        }
        const bool scalar_chain=history_only && scalar_history && tokens>1;
        // RAM QSA archive publication reads completed GPU work. Queued chains
        // retain their device-only resident path until that commit is deferred.
        int pending_qsa=!(history_tail || queued)?block.qsa->pending_device_graph_kind(tokens):0;
        if(history_only && (scalar_chain || pending_qsa>=4))pending_qsa=0;
        if(pending_qsa>=4 && block.qsa->pending_graph_task_supported())pending_qsa=(pending_qsa&3)+8;
        const auto key=std::make_tuple(int(tokens),int(hidden.scalar_type()),history_only?(scalar_chain?2:1):(history_tail?3:0),pending_qsa);
        auto& owner=cells[key];
        if((!owner || !owner->window.valid()) && !predictor.config.experts &&
                block.qsa->release_shared_device_cache(true))
            default_context(mfq_current_cuda_device())->reclaim_idle_memory();
        bool returned_expert_workspace=false;
        if((!owner || !owner->window.valid()) && (execution.gguf_expert_cache || execution.tq_expert_cache)) {
            // History cells project only the bounded draft/history rows. Their
            // already allocated KV must not be priced as another target graph.
            const auto transient=(std::size_t(64)<<20)+std::size_t(tokens)*
                (predictor.config.hidden*predictor.config.streams*sizeof(float)*32+
                 (history_only?0:predictor.config.vocab*sizeof(float)*8));
            if(default_context(mfq_current_cuda_device())->available_memory()<transient) {
                const auto before=routed_expert_cache_budget_bytes(execution);
                reserve_routed_capture_workspace(execution,transient);
                returned_expert_workspace=routed_expert_cache_budget_bytes(execution)<before;
            }
        }
        if(!owner)owner=std::make_unique<Cell>(hidden,ids,stream.stream());
        auto& cell=*owner;cell.hidden.copy_(hidden);cell.ids.copy_(ids);
        if(!history_only)cell.compact_control.set(predictor.compact_graph_sampling,"predictor",int(tokens));
        if(!history_only)cell.probability_control.set(!queued,"predictor_probability",int(tokens));
        cell.window.set_profile_geometry(history_tail?"mtp_history_tail":scalar_chain?"mtp_history_scalar_chain":history_only?"mtp_history":"mtp_predictor",int(tokens));
        predictor.graph_initialized=predictor.graph_initialized || !cell.used_in_request;
        mfq_qwen4_exp::fill_decode_positions(cell.cache_positions,cell.positions,start);
        if (positions.defined()) cell.positions.copy_(positions);
        try {
            const auto profile_exchange_begin=ProfileTime::now();
            // A warmed history graph touches attention KV and dense weights
            // only. Let the expert exchange proceed until the full draft.
            if(!cell.window.valid())
                finish_routed_expert_exchanges(execution);
            else if(!history_only)prepare_moe_predictor_exchange(execution.moe_expert_cache);
            const auto profile_exchange_end=ProfileTime::now();
            block.qsa->prepare_graph_tokens(predictor.positions[0],capacity,pending_qsa,tokens);
            if(!cell.window.valid()) {
                prepare_execution_graph_pool(execution,memory_pool);
                DecodeGraphBranchScope branch_scope(execution.decode_graph_serial_branches);
                predictor.graph_initialized=true;
                const auto capture_started=std::chrono::steady_clock::now();
                cell.window.capture([&] {
                    cell.window.mark(-1,"window_begin");
                    const auto make_input=[&](const Tensor& hidden_rows,const Tensor& token_rows,int64_t rows) {
                        auto embeddings=target.embed(token_rows);
                        auto e=predictor.embedding_fusion(execution,rms_norm(embeddings,predictor.embedding_norm+1,predictor.config.eps));
                        auto h=predictor.hidden_fusion(execution,rms_norm(hidden_rows,predictor.hidden_norm+1,predictor.config.eps)
                            .reshape({1,rows,predictor.config.streams,predictor.config.hidden}));
                        return (h+e.unsqueeze(-2)).reshape({1,rows,predictor.config.streams*predictor.config.hidden});
                    };
                    if(scalar_chain) {
                        // Keep scalar projection/reduction order while replaying
                        // all accepted history rows with one set of submissions.
                        for(int64_t row=0;row<tokens;++row) {
                            auto input=make_input(cell.hidden.narrow(1,row,1),cell.ids.narrow(1,row,1),1);
                            auto mix=block.attention_gr.pre(execution,input);
                            block.qsa->forward_graph(execution,mix[0],cell.positions.narrow(-1,row,1).contiguous(),
                                cell.cache_positions.narrow(0,row,1),{},true,true);
                        }
                    } else if(history_only) {
                        auto input=make_input(cell.hidden,cell.ids,tokens);
                        auto mix=block.attention_gr.pre(execution,input);
                        block.qsa->forward_graph(execution,mix[0],cell.positions,cell.cache_positions,{},true,true);
                    } else {
                        auto input=make_input(cell.hidden,cell.ids,tokens);
                        cell.window.mark(-1,"input_fusion_end");
                        const auto* verify=std::getenv("TYLOINFER_VERIFY_MTP_HISTORY_CHAIN");
                        const bool check=verify && verify[0]=='1';
                        cell.multi=block.execute_graph(execution,std::move(input),cell.ids,cell.positions,cell.cache_positions,{},0,
                            0,nullptr,{},nullptr,history_tail,check?&cell.check_attention:nullptr,check?&cell.check_ffn_input:nullptr,
                            check?&cell.check_attention_input:nullptr);
                        cell.window.mark(-1,"head_begin");
                        cell.normalized=predictor.final_mixer->pre(execution,cell.multi)[0];
                        cell.window.mark(-1,"head_norm_end");
                        cell.logits=predictor.logits_for(target,cell.normalized.narrow(1,cell.normalized.size(1)-1,1).contiguous());
                        cell.window.mark(-1,"head_projection_end");
                        if(cell.logits.size(-1)>=64) {
                            (void)sample_top_k_compact_if_cuda(cell.logits.reshape({1,-1}).contiguous(),64,
                                cell.compact_control.device,cell.sampling_device.narrow(0,0,64).reshape({1,64}));
                        }
                        (void)sample_greedy_probability_cuda(cell.logits.reshape({1,-1}).contiguous(),
                            cell.sampling_device.narrow(0,64,1),cell.probability_control.device);
                        if(cell.sampling_mapped) {
                            sample_publish_packets_cuda(cell.sampling_device,cell.sampling_mapped);
                            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                                static std::atomic<bool> reported{false};
                                if(!reported.exchange(true))std::fprintf(stderr,
                                    "native_mtp_packet_publication actual_match=1 mapped_store=1 words=65\n");
                            }
                        } else TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(cell.sampling_host.data_ptr(),cell.sampling_device.data_ptr(),65*sizeof(int64_t),
                            cudaMemcpyDeviceToHost,stream.stream()));
                    }
                    cell.window.mark(-1,"window_end");
                },[]{},[&]{cell.multi={};cell.normalized={};cell.logits={};cell.check_attention={};cell.check_ffn_input={};cell.check_attention_input={};});
                ++predictor.graph_captures;
                std::cerr<<"runtime_qwen4_graph_capture role=predictor rows="<<tokens
                    <<" history_only="<<history_only<<" scalar_chain="<<scalar_chain<<" history_tail="<<history_tail<<" segments="<<cell.window.segments()
                    <<" qsa_pending_device="<<pending_qsa<<" count="<<predictor.graph_captures<<" ms="
                    <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-capture_started).count()<<'\n';
            }
            if(returned_expert_workspace) {
                if(predictor.decode_workspace_changed)predictor.decode_workspace_changed();
                else restore_routed_decode_workspace_after_capture(execution);
            }
            const auto profile_run_begin=ProfileTime::now();
            if(queued) {
                TYLOINFER_RUNTIME_CHECK(cell.window.enqueue_if_device_only(),"queued MTP cell acquired a host dependency");
            } else cell.window.run();
            const auto profile_run_end=ProfileTime::now();
            cell.used_in_request=true;++predictor.graph_replays;block.qsa->advance_graph(tokens);
            if(history_tail) {
                ++predictor.history_tail_replays;
                if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                    static std::atomic<bool> reported{false};
                    if(!reported.exchange(true))std::fprintf(stderr,
                        "native_mtp_history_tail actual_match=1 rows=%lld tail_rows=1 batched_projections=1\n",static_cast<long long>(tokens));
                }
            }
            predictor.positions[0]=append_decode_positions(predictor.positions[0],cell.positions,
                position_storage,predictor.config.maximum,"TYLOINFER_MTP_POSITION_BUFFER");
            predictor.lengths[0]=required;predictor.batch=1;
            const auto* profile_option=std::getenv("TYLOINFER_PROFILE_MTP");
            if(profile_option && profile_option[0]=='1' && ++cell.profile_replays<=(history_only?2:16)) {
                const auto ms=[](auto begin,auto end){return std::chrono::duration<double,std::milli>(end-begin).count();};
                std::fprintf(stderr,"runtime_mtp_predictor_profile rows=%lld history=%d replay=%d release_ms=%.4f setup_ms=%.4f exchange_ms=%.4f prepare_ms=%.4f run_ms=%.4f tail_ms=%.4f transfer_mb=%.4f\n",
                    static_cast<long long>(tokens),int(history_only),cell.profile_replays,
                    ms(profile_started,profile_released),ms(profile_released,profile_exchange_begin),
                    ms(profile_exchange_begin,profile_exchange_end),ms(profile_exchange_end,profile_run_begin),
                    ms(profile_run_begin,profile_run_end),ms(profile_run_end,ProfileTime::now()),
                    (routed_expert_streamed_bytes(execution)-profile_transfer_before)/1048576.0);
            }
            if(history_only)return {};
            if(queued)return {cell.normalized,cell.multi,cell.logits};
            if(predictor.compact_graph_sampling && cell.logits.size(-1)>=64)last_compact=cell.compact_host;
            const auto packet=std::uint64_t(*cell.greedy_host.data_ptr<int64_t>());
            const auto bits=std::uint32_t(packet>>32);float probability;
            std::memcpy(&probability,&bits,sizeof(bits));
            return {cell.normalized.clone(),cell.multi.clone(),cell.logits.clone(),
                static_cast<int32_t>(std::uint32_t(packet)),probability};
        } catch(...) {failed=true;throw;}
    }
    std::optional<Tensor> enqueue_greedy_chain(Qwen4ExpMtp& predictor,const MtpTarget& target,
            const Tensor& hidden,const Tensor& ids,const Tensor& positions,int depth,
            bool publish_host) {
        const auto history=ids.size(1),prefix=history-1;
        if(depth<2 || !reusable(predictor) || history<1 || history>128 ||
            predictor.lengths[0]+history+depth-1>capacity)return {};
        const auto ready=[&](int64_t tokens,tb::ScalarType dtype,int mode)->Cell* {
            const auto found=cells.find({int(tokens),int(dtype),mode,0});
            if(found==cells.end() || !found->second || !found->second->window.can_enqueue_device_only())return nullptr;
            return found->second.get();
        };
        const bool history_chain=predictor.history_chain_enabled();
        const bool history_tail=history>1 && history_chain &&
            !predictor.layers[0]->gdn && !predictor.layers[0]->ple && predictor.layers[0]->qsa->supports_history_tail_graph();
        auto* first=ready(history_tail?history:1,hidden.scalar_type(),history_tail?3:0);
        if(!first || !first->multi.defined())return {};
        auto* next=ready(1,first->multi.scalar_type(),0);
        if(!next || !next->multi.defined() || next->multi.scalar_type()!=first->multi.scalar_type())return {};
        const bool scalar_history=prefix>1 && history_chain;
        for(int64_t offset=0;!history_tail && offset<prefix;) {
            const auto rows=scalar_history?std::min<int64_t>(4,prefix-offset):int64_t(1);
            if(!ready(rows,hidden.scalar_type(),scalar_history && rows>1?2:1))return {};
            offset+=rows;
        }
        // Check every needed cell before changing KV or emitting a proposal.
        // Missing, profiled or host-assisted cells use the ordinary step path.
        try {
            if(publish_host && (!queued_host.defined() || queued_host.numel()<depth)) {
                auto host=tb::empty({depth},ids.options().device(tb::kCPU).dtype(tb::kInt32).pinned_memory(true));
                void* mapped=nullptr;
                const auto status=cudaHostGetDevicePointer(&mapped,host.data_ptr(),0);
                if(status!=cudaSuccess){(void)cudaGetLastError();return {};}
                queued_host=std::move(host);queued_mapped=static_cast<int32_t*>(mapped);
            }
            if(!queued_ids.defined() || queued_ids.size(1)<depth+1)
                queued_ids=tb::empty({1,depth+1},ids.options());
        }catch(const MfqResourceExhausted&){return {};}
        if(publish_host) {
            std::fill_n(queued_host.data_ptr<int32_t>(),depth,int32_t(-1));
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        auto verification=queued_ids.narrow(1,0,depth+1);
        verification.narrow(1,0,1).copy_(ids.narrow(1,prefix,1));
        const bool previous_compact=predictor.compact_graph_sampling;
        predictor.compact_graph_sampling=false;
        try {
            for(int64_t offset=0;!history_tail && offset<prefix;) {
                const auto rows=scalar_history?std::min<int64_t>(4,prefix-offset):int64_t(1);
                (void)run(predictor,target,hidden.narrow(1,offset,rows),ids.narrow(1,offset,rows),
                    positions.defined()?positions.narrow(-1,offset,rows):Tensor{},true,scalar_history,true);
                offset+=rows;
            }
            auto current_hidden=history_tail?hidden:hidden.narrow(1,prefix,1);
            auto current_ids=history_tail?ids:ids.narrow(1,prefix,1);
            for(int step=0;step<depth;++step) {
                const auto pos=positions.defined()?(history_tail && step==0?positions:positions.narrow(-1,prefix,1)+step):Tensor{};
                auto result=run(predictor,target,current_hidden,current_ids,pos,false,false,true,history_tail && step==0);
                auto* cell=step==0?first:next;
                auto next_ids=verification.narrow(1,step+1,1);
                sample_export_greedy_ids_cuda(cell->sampling_device.narrow(0,64,1),next_ids,
                    publish_host?queued_mapped+step:nullptr);
                current_hidden=std::move(result.chain_hidden);current_ids=std::move(next_ids);
            }
            predictor.compact_graph_sampling=previous_compact;
            return verification;
        }catch(...) {
            predictor.compact_graph_sampling=previous_compact;failed=true;
            (void)cudaStreamSynchronize(stream.stream());throw;
        }
    }
    std::vector<int32_t> collect_greedy_chain(int depth,
            const std::function<void(const std::vector<int32_t>&)>& publish) {
        const auto* tokens=queued_host.data_ptr<int32_t>();
        try {
            check_stream_progress(stream.stream());
            std::vector<int32_t> result;result.reserve(depth);
            for(int step=0;step<depth;++step) {
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(60);
                unsigned spins=0;
                while(*reinterpret_cast<const volatile int32_t*>(tokens+step)<0) {
                    std::atomic_signal_fence(std::memory_order_seq_cst);
                    if((++spins&1023u)==0) {
                        check_stream_progress(stream.stream());
                        if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("queued MTP token publication timed out");
                        std::this_thread::yield();
                    }
                }
                std::atomic_thread_fence(std::memory_order_acquire);
                result.push_back(tokens[step]);
                if(publish)publish(result);
            }
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                static std::atomic<bool> reported{false};
                if(!reported.exchange(true))std::fprintf(stderr,
                    "native_mtp_queued_chain actual_match=1 depth=%d host_waits=1 gpu_token_feedback=1\n",depth);
            }
            return result;
        }catch(...) {
            failed=true;
            (void)cudaStreamSynchronize(stream.stream());throw;
        }
    }
};
Tensor Qwen4ExpMtp::graph_compact_topk() const {
    return graph_registry?graph_registry->last_compact:Tensor{};
}
std::optional<std::size_t> Qwen4ExpMtp::extra_decode_workspace_bytes(int) const {
    if(!graph_registry || !graph_registry->reusable(*this))return {};
    const bool ready=std::any_of(graph_registry->cells.begin(),graph_registry->cells.end(),[](const auto& entry) {
        return std::get<0>(entry.first)==1 && std::get<2>(entry.first)==0 && entry.second && entry.second->window.valid();
    });
    // The target budget includes bounded native history merges and sampling.
    return ready?std::optional<std::size_t>(0):std::nullopt;
}

Qwen4ExpMtp::Qwen4ExpMtp() = default;
Qwen4ExpMtp::~Qwen4ExpMtp() = default;
Qwen4ExpMtp::Qwen4ExpMtp(Qwen4ExpMtp&&) noexcept = default;
Qwen4ExpMtp& Qwen4ExpMtp::operator=(Qwen4ExpMtp&&) noexcept = default;

std::unique_ptr<Qwen4ExpMtp> Qwen4ExpMtp::fork_empty() const {
    auto result=std::make_unique<Qwen4ExpMtp>();
    result->config=config;
    result->execution=execution;
    result->embedding_norm=embedding_norm;
    result->hidden_norm=hidden_norm;
    result->embedding_fusion=embedding_fusion;
    result->hidden_fusion=hidden_fusion;
    result->draft_output=draft_output;
    result->draft_token_ids=draft_token_ids;
    result->final_mixer=std::make_unique<Gr>(*final_mixer);
    result->dense_attention_window=dense_attention_window;
    result->reuse_decode_graphs=reuse_decode_graphs;
    result->positions.resize(layers.size());
    result->lengths.resize(layers.size());
    result->batch=1;
    for(const auto& layer:layers) {
        TYLOINFER_RUNTIME_CHECK(layer->qsa && !layer->gdn && !layer->ple,
            "independent Qwen4 predictors require QSA-only layers");
        result->layers.push_back(std::make_unique<Qwen4Block>(*layer,layer->qsa->fork_empty()));
    }
    return result;
}

std::optional<Qwen4ExpMtp> Qwen4ExpMtp::load_if_present(CudaExecutionContext &execution,
                                                      const mfq::ModelSource &file,
                                                      const mfq::models::qwen4_exp::Config &main) {
    Qwen4ExpMtp result;
    result.execution = &execution;
    BlockLoader loader{{execution, file, "qwen4_exp"}};
    if (!mfq::models::qwen4_exp::load_predictor(result, loader, main))
        return std::nullopt;
    if(const auto window=runtime_options::mtp_attention_window()) {
        result.dense_attention_window=*window?std::min(*window,main.maximum):main.maximum;
        std::cerr<<"native_mtp_attention_policy mode=dense_window window="<<result.dense_attention_window
            <<" target_sparse_budget="<<main.budget<<'\n';
    }
    const bool head=has_tensor(file,"predictor.output.weight");
    const bool tokens=has_tensor(file,"predictor.output.token_ids");
    if(head!=tokens)throw std::invalid_argument("MTP draft output requires both weights and vocabulary IDs");
    if(head) {
        auto ids=load_dense_cpu(execution,file,"predictor.output.token_ids").to(tb::kInt64).contiguous();
        const auto* weight=file.find_tensor("predictor.output.weight");
        TYLOINFER_RUNTIME_CHECK(ids.dim()==1 && ids.numel()>0 && weight && weight->shape &&
            *weight->shape==std::vector<int64_t>({ids.numel(),main.hidden}),"MTP draft output geometry differs from target");
        std::vector<int64_t> ordered(ids.data_ptr<int64_t>(),ids.data_ptr<int64_t>()+ids.numel());
        std::sort(ordered.begin(),ordered.end());
        TYLOINFER_RUNTIME_CHECK(ordered.front()>=0 && ordered.back()<main.vocab &&
            std::adjacent_find(ordered.begin(),ordered.end())==ordered.end(),"MTP draft vocabulary contains invalid or duplicate IDs");
        result.draft_output=loader.linear("predictor.output.weight");
        result.draft_token_ids=ids.to(result.embedding_norm.device());
    }
    return result;
}

Tensor Qwen4ExpMtp::logits_for(const MtpTarget& target,Tensor hidden) const {
    if(!draft_output)return target.logits(std::move(hidden));
    auto subset=draft_output(*execution,std::move(hidden));
    auto shape=subset.sizes().vec();shape.back()=config.vocab;
    auto logits=tb::full(shape,-std::numeric_limits<float>::infinity(),subset.options());
    logits.index_copy_(-1,draft_token_ids,subset);
    return logits;
}

void Qwen4ExpMtp::reset(int64_t next_batch) {
    TYLOINFER_RUNTIME_CHECK(next_batch > 0, "Qwen4-Exp MTP batch must be positive");
    const bool retain=next_batch==1 && graph_registry && graph_registry->reusable(*this);
    if(!retain)graph_registry.reset();
    else for(auto& [key,cell]:graph_registry->cells) {cell->used_in_request=false;cell->profile_replays=0;}
    graph_initialized=false;
    for (auto &layer : layers) {
        layer->retain_graph_storage=retain;
        layer->reset(next_batch);
    }
    for (auto &pos : positions)
        pos = Tensor();
    std::fill(lengths.begin(), lengths.end(), 0);
    batch = next_batch;
}

std::pair<Tensor, Tensor> Qwen4ExpMtp::evaluate(const MtpTarget &target, const Tensor &hidden,
                                       const Tensor &ids, int64_t depth, bool cache,
                                       const Tensor &supplied_positions,
                                       const Tensor &supplied_embeddings, bool history_only) {
    namespace tb = mfq_tensor_backend;
    TYLOINFER_RUNTIME_CHECK(!history_only || (cache && layers.size()==1),
        "cache-only MTP priming requires a single predictor layer");
    if(graph_registry) {
        if(graph_registry->reusable(*this) && ids.size(0)==1 && cache)layers[0]->qsa->leave_graph();
        else {graph_registry->restore_legacy(*this);graph_registry.reset();}
    }
    TYLOINFER_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 &&
                          hidden.dim() == 3 && hidden.size(0) == ids.size(0) &&
                          hidden.size(1) == ids.size(1) &&
                          hidden.size(2) == hidden_norm.numel(),
                      "Qwen4-Exp MTP input geometry mismatch");
    const auto b = ids.size(0), t = ids.size(1);
    return mfq::models::projected_predictor(
        *this, b, t, depth, cache, [&](int64_t next) { reset(next); },
        [&] {
            auto embeds =
                supplied_embeddings.defined() ? supplied_embeddings : target.embed(ids);
            TYLOINFER_RUNTIME_CHECK(embeds.sizes().vec() ==
                                  std::vector<int64_t>({b, t, config.hidden}),
                              "Qwen4-Exp MTP embedding shape mismatch");
            return embeds;
        },
        [&](mfq::models::PredictorCursor cursor) {
            // Full positions persist across predictor priming tiles and must
            // not occupy expert storage borrowed by this prefill phase.
            mfq::cuda::ResidentAllocationScope persistent_positions;
            auto current = supplied_positions.defined()
                               ? supplied_positions
                               : tb::arange(cursor.start, cursor.start + t,
                                            ids.options().dtype(tb::kInt32));
            if (current.dim() == 1)
                current = current.reshape({1, 1, t}).expand({3, 1, t}).contiguous();
            else if (current.dim() == 2)
                current = current.unsqueeze(1);
            if (current.dim() == 3 && current.size(0) == 4)
                current = current.narrow(0, 1, 3);
            TYLOINFER_RUNTIME_CHECK(current.dim() == 3 && current.size(0) == 3 &&
                                  current.size(-1) == t &&
                                  (current.size(1) == 1 || current.size(1) == b),
                              "Qwen4 MTP positions require [T]/[3,T]/[3,B,T]");
            current = current.to(tb::kInt32).expand({3, b, t}).contiguous();
            auto full = cache && positions[cursor.layer].defined()
                            ? tb::cat({positions[cursor.layer], current}, -1)
                            : current;
            return std::array<Tensor, 2>{current, full};
        },
        [&](Tensor embeds, const auto &pos) {
            return embedding_fusion(*execution,
                                    rms_norm(embeds, embedding_norm + 1, config.eps));
        },
        [&] {
            return hidden_fusion(*execution,
                                 rms_norm(hidden, hidden_norm + 1, config.eps)
                                     .reshape({b, t, config.streams, config.hidden}));
        },
        [&](Tensor e, Tensor streams) {
            return (streams + e.unsqueeze(-2)).reshape({b, t, config.streams * config.hidden});
        },
        [&](Tensor x, int64_t layer, const auto &pos) {
            auto &block = *layers[layer];
            // Teacher-forced rows each receive the target hidden state. Only
            // the attention input contributes persistent predictor history;
            // attention readout, FFN and final mixer have no consumer here.
            if(history_only) {
                auto mix=block.attention_gr.pre(*execution,x);
                block.qsa->forward(*execution,mix[0],pos[0],pos[1],true,nullptr,{},true,true);
                return Tensor{};
            }
            return mfq::models::qwen4_exp::decoder_layer(
                std::move(x), false, false, [](const Tensor &) { return Tensor{}; },
                [](Tensor value, Tensor) { return value; },
                [&](const Tensor &value) { return block.attention_gr.pre(*execution, value); },
                [](Tensor) -> Tensor { throw std::logic_error("Qwen4 MTP requires QSA"); },
                [&](Tensor branch) {
                    return block.qsa->forward(*execution, branch, pos[0], pos[1], cache);
                },
                [&](Tensor branch, const auto &mix) {
                    return block.attention_gr.post(branch, mix);
                },
                [&](const Tensor &value) { return block.ffn_gr.pre(*execution, value); },
                [&](Tensor branch) { return block.ffn(*execution, branch); },
                [&](Tensor branch, const auto &mix) { return block.ffn_gr.post(branch, mix); });
        },
        [&](const Tensor &multi) { return history_only?Tensor{}:final_mixer->pre(*execution, multi)[0]; },
        [&](int64_t layer, const auto &pos) { positions[layer] = pos[1]; });
}

Tensor Qwen4ExpMtp::forward(const MtpTarget &target, Tensor hidden, Tensor ids) {
    return evaluate(target, hidden, ids).first;
}

MtpStep Qwen4ExpMtp::step(const MtpTarget &target, Tensor hidden, Tensor ids) {
    return step_positioned(target,std::move(hidden),std::move(ids),{});
}

Tensor Qwen4ExpMtp::prime_prompt(const MtpTarget& target,const Tensor& hidden,const Tensor& ids,
        const Tensor& previous) {
    TYLOINFER_RUNTIME_CHECK(ids.dim()==2 && ids.size(0)==1 && ids.size(1)>0 &&
        hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==ids.size(1),"invalid MTP prompt history");
    const bool history_only=layers.size()==1;
    if(previous.defined())
        (void)evaluate(target,previous,ids.narrow(1,0,1),0,true,{},{},history_only);
    const auto pairs=ids.size(1)-1;
    if(pairs)
        (void)evaluate(target,hidden.narrow(1,0,pairs),ids.narrow(1,1,pairs),0,true,{},{},history_only);
    const ResidentAllocationScope persistent_hidden;
    return hidden.narrow(1,pairs,1).clone();
}

MtpStep Qwen4ExpMtp::batch_graph(const MtpTarget& target,const Tensor& hidden,const Tensor& ids,bool history_only) {
    const auto graph=mfq::cuda::runtime_options::mtp_predictor_graph();
    TYLOINFER_RUNTIME_CHECK((!graph || *graph) && execution && streamed_decode_graphs(*execution) &&
        layers.size()==1 && layers[0]->qsa && (cache_position()>0 || layers[0]->qsa->graph_capacity()>0) &&
        batch==1 && ids.is_cuda() && ids.dim()==2 && ids.size(0)==1 && ids.size(1)>0 && ids.size(1)<=8 &&
        hidden.is_cuda() && hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==ids.size(1) &&
        hidden.size(2)==hidden_norm.numel(),"CUDA Graph is required for the batch MTP predictor");
    if(!graph_registry)graph_registry=std::make_unique<Qwen4MtpGraph>();
    return graph_registry->run(*this,target,hidden,ids,{},history_only);
}

bool Qwen4ExpMtp::supports_graph_window(int64_t tokens) const {
    if(!execution || layers.size()!=1 || !layers[0]->qsa || tokens<1 || tokens>8 ||
            execution->kl_mmq.mode!=KlMmqMode::Default || !streamed_decode_graphs(*execution))return false;
    if(!config.experts)return layers[0]->tq_graph_ffn;
    const bool row_banks=bool(execution->tq_expert_cache || execution->gguf_expert_cache);
    if(row_banks)return layers[0]->tq_graph_ffn &&
        (!execution->tq_expert_cache || tq_graph_supports_tokens(tokens));
    return execution->config.moe_pipeline && execution->config.moe_two_stage_ffn &&
        execution->moe_expert_cache && layers[0]->prefetch;
}

bool Qwen4ExpMtp::history_chain_enabled() const {
    const auto* setting=std::getenv("TYLOINFER_MTP_HISTORY_CHAIN");
    // Dense predictors keep the same per-row history order as eager priming.
    return config.experts && (!setting || setting[0]!='0');
}

MtpStep Qwen4ExpMtp::step_positioned(const MtpTarget &target, Tensor hidden, Tensor ids, Tensor positions) {
    if (positions.defined()) {
        if (positions.dim()==1) positions=positions.reshape({1,1,-1}).expand({3,ids.size(0),ids.size(1)});
        else if (positions.dim()==2) positions=positions.unsqueeze(1);
        if (positions.dim()!=3 || positions.size(0)!=3 || positions.size(1)!=ids.size(0) || positions.size(2)!=ids.size(1))
            throw std::invalid_argument("Qwen4 MTP positions require [T]/[3,T]/[3,B,T]");
        positions=positions.to(tb::kInt32).contiguous();
    }
    const auto graph=mfq::cuda::runtime_options::mtp_predictor_graph();
    if((!graph || *graph) && batch==1 && ids.dim()==2 && ids.size(0)==1 && supports_graph_window(ids.size(1)) &&
        hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==ids.size(1) && hidden.size(2)==hidden_norm.numel() &&
        ids.is_cuda() && hidden.is_cuda() && !lengths.empty() && lengths[0]>0 &&
        this->positions[0].dim()==3 && this->positions[0].size(1)==1) {
        if(!graph_registry)graph_registry=std::make_unique<Qwen4MtpGraph>();
        return graph_registry->run(*this,target,hidden,ids,positions);
    }
    auto result = evaluate(target, hidden, ids, 0, true, positions);
    auto logits=draft_output?logits_for(target,result.first.narrow(1,result.first.size(1)-1,1).contiguous()):Tensor{};
    return {std::move(result.first), std::move(result.second),std::move(logits)};
}

void Qwen4ExpMtp::history_step(const MtpTarget& target,Tensor hidden,Tensor ids,Tensor positions,bool scalar_chain) {
    if(ids.dim()==1)ids=ids.unsqueeze(0);
    if(positions.defined()) {
        if(positions.dim()==1)positions=positions.reshape({1,1,-1}).expand({3,ids.size(0),ids.size(1)});
        else if(positions.dim()==2)positions=positions.unsqueeze(1);
        TYLOINFER_RUNTIME_CHECK(positions.dim()==3 && positions.size(0)==3 &&
            positions.size(1)==ids.size(0) && positions.size(2)==ids.size(1),"invalid MTP history positions");
        positions=positions.to(tb::kInt32).contiguous();
    }
    const auto graph=mfq::cuda::runtime_options::mtp_predictor_graph();
    if(reuse_decode_graphs && (!graph || *graph) && batch==1 &&
            ids.dim()==2 && ids.size(0)==1 && supports_graph_window(ids.size(1)) &&
            hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==ids.size(1) &&
            hidden.size(2)==hidden_norm.numel() && ids.is_cuda() && hidden.is_cuda() &&
            !lengths.empty() && lengths[0]>0 &&
            this->positions[0].dim()==3 && this->positions[0].size(1)==1) {
        if(!graph_registry)graph_registry=std::make_unique<Qwen4MtpGraph>();
        (void)graph_registry->run(*this,target,hidden,ids,positions,true,scalar_chain);
    } else if(scalar_chain && ids.size(1)>1) {
        for(int64_t row=0;row<ids.size(1);++row)
            (void)step_positioned(target,hidden.narrow(1,row,1),ids.narrow(1,row,1),
                positions.defined()?positions.narrow(-1,row,1):Tensor{});
    } else (void)step_positioned(target,std::move(hidden),std::move(ids),std::move(positions));
}

std::optional<MtpStep> Qwen4ExpMtp::history_tail_step(const MtpTarget& target,Tensor hidden,Tensor ids,Tensor positions) {
    const auto graph=mfq::cuda::runtime_options::mtp_predictor_graph();
    if(!reuse_decode_graphs || (graph && !*graph) || !history_chain_enabled() || batch!=1 ||
        ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<2 || !supports_graph_window(ids.size(1)) ||
        layers[0]->gdn || layers[0]->ple || !layers[0]->qsa->supports_history_tail_graph() ||
        hidden.dim()!=3 || hidden.size(0)!=1 || hidden.size(1)!=ids.size(1) ||
        hidden.size(2)!=hidden_norm.numel() || !ids.is_cuda() || ids.scalar_type()!=tb::kInt64 || !hidden.is_cuda() ||
        lengths.empty() || lengths[0]<=0 || this->positions[0].dim()!=3 || this->positions[0].size(1)!=1)return {};
    if(positions.defined()) {
        if(positions.dim()==1)positions=positions.reshape({1,1,-1}).expand({3,ids.size(0),ids.size(1)});
        else if(positions.dim()==2)positions=positions.unsqueeze(1);
        TYLOINFER_RUNTIME_CHECK(positions.dim()==3 && positions.size(0)==3 &&
            positions.size(1)==1 && positions.size(2)==ids.size(1),"invalid MTP history-tail positions");
        positions=positions.to(tb::kInt32).contiguous();
    }
    if(!graph_registry)graph_registry=std::make_unique<Qwen4MtpGraph>();
    return graph_registry->run(*this,target,hidden,ids,positions,false,false,false,true);
}

int64_t Qwen4ExpMtp::cache_position() const noexcept {
    return lengths.empty() ? 0 : lengths.front();
}

void Qwen4ExpMtp::trim_cache_to(int64_t position) {
    TYLOINFER_RUNTIME_CHECK(position >= 0 && position <= cache_position(),
                      "Qwen4-Exp MTP cache trim position is invalid");
    constexpr size_t index = 0;
    if (index < layers.size() && layers[index]->qsa)
        layers[index]->qsa->truncate(position);
    if (positions[index].defined())
        positions[index] = positions[index].narrow(-1, 0, position);
    lengths[index] = position;
}

bool Qwen4ExpMtp::teacher_forced_prompt_prime() const noexcept { return true; }

bool Qwen4ExpMtp::target_bootstrap_decode() const noexcept { return true; }

bool Qwen4ExpMtp::preserve_output_dtype() const noexcept { return true; }

Qwen4BlockSessionState capture_block_session(const Qwen4Block& block,int64_t position,std::size_t& bytes) {
    Qwen4BlockSessionState state;
    const auto save=[&](const Tensor& tensor) {
        TYLOINFER_RUNTIME_CHECK(tensor.defined(),"Qwen4 recurrent session state is unavailable");
        auto saved=tensor.clone();bytes+=session_tensor_bytes(saved);return saved;
    };
    if(block.qsa)block.qsa->capture_session(state,position,bytes);
    if(block.gdn) {
        const auto tensors=block.gdn->graph_state();
        state.convolution=save(*tensors[0]);state.recurrent=save(*tensors[1]);
    }
    if(block.ple) {
        state.ple_convolution=save(block.ple->conv_state());
        const auto& context=block.ple->session_context();
        state.ple_batch=context.batch;state.ple_tokens=context.tokens;
        bytes+=state.ple_tokens.size()*sizeof(int64_t);
    }
    return state;
}

static std::size_t map_request_recurrence(int device_index,const std::vector<Gdn*>& layers,
        std::size_t desired,bool live_only) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    namespace tb=mfq_tensor_backend;
    const auto budget=std::min<std::uint64_t>(desired,mfq::sample_host_memory().available_host_bytes());
    if(!budget)return 0;
    int map_host=0,unified=0;
    TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&map_host,cudaDevAttrCanMapHostMemory,device_index));
    TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&unified,cudaDevAttrUnifiedAddressing,device_index));
    if(!map_host || !unified)return 0;
    std::vector<Gdn*> mapped;
    std::size_t bytes=0,released=0;
    const auto aligned=[](std::size_t size){return (size+255)&~std::size_t(255);};
    for(auto* gdn:layers)if(!gdn->host_recurrent_state_bytes()) {
        if(live_only && !gdn->has_device_request_state())continue;
        const auto size=gdn->recurrent_state_bytes();
        if(aligned(size)<=budget-bytes){mapped.push_back(gdn);bytes+=aligned(size);released+=size;}
    }
    if(!bytes)return 0;
    const auto elements=int64_t(bytes/sizeof(float));
    const auto device=tb::Device(tb::kCUDA,device_index);
    auto host=tb::empty({elements},tb::TensorOptions().dtype(tb::kFloat32).device(tb::kCPU).pinned_memory(true));
    void* address=nullptr;
    TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&address,host.data_ptr(),0));
    auto storage=std::make_shared<tb::TensorStorage>();
    storage->owner=std::make_shared<tb::Tensor>(host);
    storage->base=address;storage->bytes=bytes;storage->device=device;storage->mapped_host=true;
    const auto view=tb::Tensor(storage,make_contiguous_view(address,
        std::array<int64_t,1>{elements},tb::kFloat32,device));
    int64_t offset=0;
    for(auto* layer:mapped) {
        const auto size=layer->recurrent_state_bytes();
        const auto length=int64_t(size/sizeof(float));
        layer->bind_request_host_state(view.narrow(0,offset,length),host.narrow(0,offset,length));
        offset+=int64_t(aligned(size)/sizeof(float));
    }
    if(std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))std::fprintf(stderr,
        "native_qwen4_host_recurrence bytes=%zu layers=%zu moved_live=%d\n",bytes,mapped.size(),int(live_only));
    return released;
#else
    return 0;
#endif
}

void restore_block_session(Qwen4Block& block,const Qwen4BlockSessionState& state,int64_t position,
        const std::function<bool(std::size_t)>& admit_host_recurrence={}) {
    block.reset(1);
    if(block.qsa)block.qsa->restore_session(state,position);
    if(block.gdn) {
        auto& gdn=*block.gdn;
        try {gdn.restore_session(state.convolution,state.recurrent,block.retain_graph_storage);}
        catch(const MfqResourceExhausted&) {
            const auto bytes=(gdn.recurrent_state_bytes()+255)&~std::size_t(255);
            if(block.retain_graph_storage || !admit_host_recurrence || !admit_host_recurrence(bytes))throw;
            gdn.reset();
            default_context(block.cuda_device)->reclaim_idle_memory();
            if(!map_request_recurrence(block.cuda_device,{&gdn},bytes,false))throw;
            gdn.restore_session(state.convolution,state.recurrent);
        }
    }
    if(block.ple)block.ple->restore_session(state.ple_convolution,{state.ple_batch,state.ple_tokens},block.retain_graph_storage);
}

bool Qwen4ExpMtp::supports_session_state() const noexcept {
    if(!execution || layers.empty())return false;
    for(const auto& layer:layers)if(!layer || (layer->qsa && !layer->qsa->supports_session()))return false;
    return true;
}

MtpSessionState Qwen4ExpMtp::capture_session_state(int64_t target_position,const Tensor& hidden) const {
    const auto position=target_position-1;
    TYLOINFER_RUNTIME_CHECK(supports_session_state() && position>0 && position<=cache_position() &&
        hidden.defined() && hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==1 && hidden.size(2)==hidden_norm.numel(),
        "Qwen4 MTP session boundary is unavailable");
    MtpSessionState state;state.cache_pos=position;
    for(std::size_t i=0;i<layers.size();++i) {
        const auto count=std::min(position,lengths[i]);
        state.qwen4_lengths.push_back(count);
        state.qwen4_blocks.push_back(count?capture_block_session(*layers[i],count,state.bytes):Qwen4BlockSessionState{});
        auto pos=count?positions[i].narrow(-1,0,count).clone():Tensor{};
        state.bytes+=session_tensor_bytes(pos);state.qwen4_positions.push_back(std::move(pos));
    }
    state.last_target_hidden=hidden.clone();state.bytes+=session_tensor_bytes(state.last_target_hidden);
    return state;
}

void Qwen4ExpMtp::restore_session_state(const MtpSessionState& state) {
    TYLOINFER_RUNTIME_CHECK(supports_session_state() && state.cache_pos>0 &&
        state.qwen4_blocks.size()==layers.size() && state.qwen4_positions.size()==layers.size() &&
        state.qwen4_lengths.size()==layers.size() && state.qwen4_lengths.front()==state.cache_pos &&
        state.last_target_hidden.defined() && state.last_target_hidden.dim()==3 &&
        state.last_target_hidden.size(0)==1 && state.last_target_hidden.size(1)==1 &&
        state.last_target_hidden.size(2)==hidden_norm.numel(),"incompatible Qwen4 MTP session state");
    reset(1);
    std::vector<std::shared_ptr<void>> reservations;
    for(std::size_t i=0;i<layers.size();++i)if(state.qwen4_lengths[i] && layers[i]->qsa)
        reservations.push_back(layers[i]->qsa->reserve_session_restore(state.qwen4_blocks[i]));
    for(std::size_t i=0;i<layers.size();++i)if(state.qwen4_lengths[i]) {
        restore_block_session(*layers[i],state.qwen4_blocks[i],state.qwen4_lengths[i]);
        positions[i]=state.qwen4_positions[i].clone();lengths[i]=state.qwen4_lengths[i];
    }
}

} // namespace mfq::cuda::qwen4_exp

namespace mfq::cuda {
bool Qwen4Model::adapter_prefetch_token_rows(Qwen4CausalLm& model,const mfq_tensor_backend::Tensor& ids) {
    return reuse_decode_graphs && graph_registry && qwen4_prefetch_decode_ple(model,ids);
}
bool qwen4_prefetch_decode_ple(Qwen4CausalLm& model,const mfq_tensor_backend::Tensor& host_ids) {
    bool issued=false;
    for(const auto& block:model.blocks) {
        auto* qwen4=dynamic_cast<qwen4_exp::Qwen4Block*>(block.get());
        if(qwen4 && qwen4->ple && qwen4->ple->graph_supported()) {
            qwen4->ple->prefetch_tokens(host_ids,true);
            issued=true;
        }
    }
    return issued;
}
struct Qwen4DecodeGraph {
    static inline thread_local bool startup_warmup=false;
    struct PleStage {
        qwen4_exp::Ple* ple;
        mfq_tensor_backend::Tensor rows;
        std::shared_ptr<RowStage> transfer;
        std::vector<qwen4_exp::NgramEmbedding::Context> prefix_context;
        HostBuffer flags{128,true};
        uint32_t* device=nullptr;
        bool profile=false;
        uint64_t calls=0,route_wait_ns=0,upload_ns=0,notify_ns=0;
        void prefetch_rows(const Tensor& host_ids) {ple->prefetch_graph_tokens(host_ids,transfer);}
        void upload_rows(const Tensor& host_ids,int64_t confirmed) {
            const auto started=profile?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
            if(confirmed) {
                prefix_context.clear();
                for(int i=confirmed;i<host_ids.size(1);++i)
                    prefix_context.push_back(ple->graph_prefix_context(host_ids,i));
            }
            ple->upload_graph(host_ids,*transfer);
            if(profile) {
                upload_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now()-started).count();++calls;
            }
        }
        PleStage(qwen4_exp::Ple* p,int tokens,int hidden):ple(p) {
            const auto timing=mfq::cuda::runtime_options::trace_ple_timings();profile=timing && *timing;
            charge_tensor_host_bytes(128);
            TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device),flags.data(),0));
            std::memset(flags.data(),0,128);publish_mapped_flag(host()+16);
            transfer=ple->graph_stage(tokens,mfq_tensor_backend::Device{mfq_tensor_backend::kCUDA,mfq_current_cuda_device()});
            rows=transfer->output().reshape({1,tokens,hidden});
        }
        ~PleStage(){
            if(profile)std::cerr<<"ple_stage_host mapped_rows="<<transfer->mapped_rows()<<" calls="<<calls
                <<" route_wait_ns="<<route_wait_ns<<" upload_ns="<<upload_ns<<" notify_ns="<<notify_ns<<'\n';
            tensor_host_bytes.fetch_sub(128);
        }
        uint32_t* host(){return static_cast<uint32_t*>(flags.data());}
    };
    struct ReplayStorage {
        Tensor descriptors;
        std::vector<Tensor> retained;
        int layers=0,maximum_heads=0,maximum_channels=0,tokens=0;
        bool recorded_updates=true;
        template<class GetGdn>
        ReplayStorage(const std::vector<qwen4_exp::RecurrentGraphCheckpoint>& checkpoints,
                std::vector<Tensor>& scratch,GetGdn get_gdn) {
            const ResidentAllocationScope resident;
            std::vector<GdnReplayLayer> meta;
            for(std::size_t i=0;i<checkpoints.size();++i) {
                if(!checkpoints[i].gdn_replay)continue;
                if(scratch.size()<=meta.size())scratch.emplace_back();
                auto layer=get_gdn(i).replay_layer(checkpoints[i].gdn_inputs,scratch[meta.size()],retained);
                recorded_updates=recorded_updates && layer.updates!=nullptr;
                maximum_heads=std::max(maximum_heads,layer.value_heads);maximum_channels=std::max(maximum_channels,layer.channels);
                if(!tokens)tokens=layer.tokens;
                TYLOINFER_RUNTIME_CHECK(tokens==layer.tokens,"GDN replay layers have different windows");
                meta.push_back(layer);
            }
            TYLOINFER_RUNTIME_CHECK(!meta.empty(),"GDN replay group is empty");layers=int(meta.size());
            const auto options=checkpoints[0].gdn_inputs.empty()?scratch.front().options():checkpoints[0].gdn_inputs[0].options();
            auto host=mfq_tensor_backend::empty({int64_t(meta.size()*sizeof(GdnReplayLayer))},
                options.device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kUInt8));
            std::memcpy(host.data_ptr(),meta.data(),meta.size()*sizeof(GdnReplayLayer));
            descriptors=host.to(scratch.front().device());retained.push_back(std::move(host));
        }
    };
    struct ReplayCommit final : qwen4_exp::GdnReplayCommitGroup {
        std::shared_ptr<ReplayStorage> storage;int committed=-1;
        explicit ReplayCommit(std::shared_ptr<ReplayStorage> value):storage(std::move(value)){}
        void commit(int64_t keep) override {
            TYLOINFER_RUNTIME_CHECK(keep>0 && keep<=storage->tokens && (committed<0 || committed==keep),"GDN replay layer verdicts differ");
            if(committed>=0)return;
            const auto stream=reinterpret_cast<std::uintptr_t>(mfq_current_cuda_stream());
            for(const auto& tensor:storage->retained)if(tensor.is_cuda())tensor.record_stream(stream);
            gdn_replay_commit_layers_cuda(storage->descriptors,storage->layers,storage->maximum_heads,storage->maximum_channels,int(keep),storage->recorded_updates);
            committed=int(keep);
        }
    };
    struct Cell {
        mfq_tensor_backend::Tensor ids,positions,cache_positions,host_ids,output,raw_hidden,logits,greedy_host,compact_host;
        int64_t* greedy_mapped=nullptr;
        int64_t* compact_mapped=nullptr;
        qwen4_exp::MtpCompactControl compact_control;
        unsigned used_kinds=0;
        std::vector<std::unique_ptr<PleStage>> ple;
        std::vector<qwen4_exp::RecurrentGraphCheckpoint> checkpoints;
        std::shared_ptr<ReplayStorage> replay_storage;
        DecodeWindow window;
        Cell(int tokens,int hidden,const std::vector<std::unique_ptr<Block>>& blocks,cudaStream_t stream,int confirmed,
                std::vector<qwen4_exp::RecurrentGraphCheckpoint>& shared_checkpoints,bool prediction)
            :window(stream,mfq::cuda::runtime_options::layer_timings()) {
            window.set_prediction_enabled(prediction);
            namespace tb=mfq_tensor_backend;
            const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64);
            ids=tb::zeros({1,tokens},options);positions=tb::zeros({tokens},options);cache_positions=tb::zeros({tokens},options);
            host_ids=tb::zeros({1,tokens},options.device(tb::kCPU));
            greedy_host=tb::empty({tokens},options.device(tb::kCPU).pinned_memory(true));
            compact_host=tb::empty({tokens,64},options.device(tb::kCPU).pinned_memory(true));
            // Like the drafter packet, verifier results can be published by
            // GPU stores without waiting behind expert copies on a DMA engine.
            const auto* mapped_option=std::getenv("TYLOINFER_MTP_MAPPED_PACKET");
            if(!mapped_option || mapped_option[0]!='0') {
                const auto map=[](const Tensor& host)->int64_t* {
                    void* alias=nullptr;
                    if(cudaHostGetDevicePointer(&alias,host.data_ptr(),0)==cudaSuccess)
                        return static_cast<int64_t*>(alias);
                    (void)cudaGetLastError();return nullptr;
                };
                greedy_mapped=map(greedy_host);compact_mapped=map(compact_host);
            }
            if(confirmed && shared_checkpoints.empty())shared_checkpoints.resize(blocks.size());
            std::size_t layer=0;
            for(const auto& block:blocks) {
                auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);
                if(b.ple)ple.push_back(std::make_unique<PleStage>(b.ple.get(),tokens,hidden));
                if(confirmed) {
                    auto& shared=shared_checkpoints[layer];
                    qwen4_exp::RecurrentGraphCheckpoint checkpoint;
                    if(b.gdn) {
                        checkpoint.gdn_replay=b.gdn->replay_checkpoint_supported();
                        if(!checkpoint.gdn_replay) {
                        const auto state=b.gdn->graph_state();
                        while(shared.gdn_conv.size()<std::size_t(tokens-confirmed)) {
                            shared.gdn_conv.push_back(tb::empty_like(*state[0]));
                            shared.gdn_state.push_back(tb::empty_like(*state[1]));
                        }
                        checkpoint.gdn_conv.assign(shared.gdn_conv.begin(),shared.gdn_conv.begin()+tokens-confirmed);
                        checkpoint.gdn_state.assign(shared.gdn_state.begin(),shared.gdn_state.begin()+tokens-confirmed);
                        }
                    }
                    if(b.ple) {
                        while(shared.ple_conv.size()<std::size_t(tokens-confirmed))
                            shared.ple_conv.push_back(tb::empty_like(b.ple->conv_state()));
                        checkpoint.ple_conv.assign(shared.ple_conv.begin(),shared.ple_conv.begin()+tokens-confirmed);
                    }
                    checkpoints.push_back(std::move(checkpoint));
                }
                ++layer;
            }
        }
    };
    MfqCudaStream stream,ple_stream;
    std::unique_ptr<RoutedGraphPools> memory_pool;
    std::map<std::tuple<int,int,int,bool,int>,std::unique_ptr<Cell>> cells;
    // Verification widths execute serially and commit/rollback before the next
    // window. Their snapshots can share stable allocations across graph shapes.
    std::vector<qwen4_exp::RecurrentGraphCheckpoint> shared_checkpoints;
    std::vector<Tensor> replay_convolutions;
    std::vector<qwen4_exp::Qwen4Block*> blocks;
    std::vector<qwen4_exp::Qsa*> qsa;
    mfq_tensor_backend::Tensor position_storage;
    int64_t kv_capacity=0;
    bool failed=false;
    mfq_tensor_backend::Tensor last_compact;
    explicit Qwen4DecodeGraph(Qwen4CausalLm& model):stream(mfq_get_current_cuda_stream()),ple_stream(mfq_get_stream_from_pool()) {
        const bool enabled=model.router_lookahead.value_or(mfq::cuda::runtime_options::router_lookahead_override().value_or(
            moe_expert_cache_ssd_experts(model.execution->moe_expert_cache)));
        for(std::size_t i=0;i<model.blocks.size();++i) {
            auto& current=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]);
            blocks.push_back(&current);
            current.next_router={};current.next_prefetch={};
            if(enabled && i+1<model.blocks.size()) {
                auto& next=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i+1]);
                TYLOINFER_RUNTIME_CHECK(next.router && next.prefetch,"router lookahead requires the tiered FFN pipeline");
                current.next_router=next.router;current.next_prefetch=next.prefetch;
            }
        }
        for(const auto& block:model.blocks)if(auto* attention=static_cast<qwen4_exp::Qwen4Block&>(*block).qsa.get())qsa.push_back(attention);
    }
    ~Qwen4DecodeGraph() {(void)cudaStreamSynchronize(ple_stream.stream());cells.clear();}
    void restore_legacy() {
        last_compact={};
        cells.clear();shared_checkpoints.clear();replay_convolutions.clear();for(auto* attention:qsa)attention->leave_graph();
        for(auto* block:blocks)block->retain_graph_storage=false;
    }
    bool reusable() const {
        return !failed && !qsa.empty() && std::all_of(qsa.begin(),qsa.end(),
            [](const auto* attention){return attention->graph_capacity()>0;});
    }

    mfq_tensor_backend::Tensor run(Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,int kind,
                                  mfq_tensor_backend::Tensor* raw_hidden=nullptr,int confirmed=0,bool* initialized=nullptr,
                                  const mfq_tensor_backend::Tensor& host_ids={},mfq_tensor_backend::Tensor* logits=nullptr,
                                  std::vector<int32_t>* greedy_tokens=nullptr,bool compact_sampling=true) {
        namespace tb=mfq_tensor_backend;
        using HostClock=std::chrono::steady_clock;
        static const bool profile_host=[] {
            const auto* setting=std::getenv("TYLOINFER_PROFILE_MTP");return setting && setting[0]=='1';
        }();
        const auto host_begin=profile_host?HostClock::now():HostClock::time_point{};
        if(failed)throw std::runtime_error("MFQ model session requires recovery after its previous error");
        last_compact={};
        if(stream.stream()!=mfq_current_cuda_stream())throw std::runtime_error("MFQ model window changed execution stream");
        const QsaRamDecodeBudgetScope ram_budget(model.execution->layer_placement.primary_device(),true);
        const int tokens=static_cast<int>(ids.size(1));
        // These consumers use the same single-row backbone. Keep its hidden
        // state, logits and selected token in one graph instead of reserving
        // a complete set of layer buffers for each output representation.
        const int result_kind=kind;
        if(tokens==1 && kind>=1 && kind<=4)kind=4;
        release_moe_prefill_buffers(model.execution->moe_expert_cache);
        const auto required=model.cache_pos+tokens;
        TYLOINFER_RUNTIME_CHECK(required<=model.max_position_embeddings(),"Qwen4 graph exceeds context capacity");
        if(!qsa.empty() && qsa.front()->graph_capacity()) {
            if(model.execution->kv_memory)model.execution->kv_memory->ensure(required);
            kv_capacity=qsa.front()->graph_capacity();
        }
        if(required>kv_capacity) {
            finish_routed_expert_exchanges(*model.execution);
            restore_legacy();
            // Small growing chunks preserve dynamic KV allocation without
            // recapturing all MTP batch shapes at every few decode tokens.
            const int64_t chunk=raw_hidden?2048:256;
            kv_capacity=std::min<int64_t>(model.max_position_embeddings(),std::max<int64_t>(chunk,kv_capacity));
            const auto planned=std::max(required,model.decode_reserve_until);
            while(kv_capacity<planned)
                kv_capacity=std::min(model.max_position_embeddings(),kv_capacity>model.max_position_embeddings()/2
                    ? model.max_position_embeddings() : kv_capacity*2);
            if(const auto* trace=std::getenv("TYLOINFER_RUNTIME_TRACE_CUDA_GRAPH");trace && trace[0]=='1')
                std::cerr<<"runtime_qwen4_graph_capacity position="<<model.cache_pos
                    <<" required="<<required<<" planned="<<planned<<" capacity="<<kv_capacity<<'\n';
        }
        const bool prediction=tokens==1 && !model.execution->tq_expert_cache && model.router_lookahead.value_or(
            mfq::cuda::runtime_options::router_lookahead_override().value_or(
                moe_expert_cache_ssd_router_prefetch(model.execution->moe_expert_cache)));
        // Recurrent checkpoints can confirm the first verification row while
        // QSA still appends the complete batch and rolls back rejected rows.
        int pending_qsa=0;
        for(auto* attention:qsa) {
            const int kind=attention->pending_device_graph_kind(tokens);
            if(!kind){pending_qsa=0;break;}
            pending_qsa=std::max(pending_qsa,kind);
        }
        if(pending_qsa>=4) {
            bool all_qsa_tasks=true;
            for(auto* attention:qsa)if(!attention->pending_graph_task_supported()){all_qsa_tasks=false;break;}
            if(all_qsa_tasks)pending_qsa=(pending_qsa&3)+8;
        }
        auto& owner=cells[{tokens,kind,confirmed,prediction,pending_qsa}];
        if((!owner || !owner->window.valid()) && !model.config.experts) {
            std::size_t released=0;
            for(auto* attention:qsa)released+=attention->release_shared_device_cache(true);
            if(released)default_context(mfq_current_cuda_device())->reclaim_idle_memory();
        }
        bool returned_expert_workspace=false;
        // Prepared decode may lend its unused capture allowance to experts.
        // Return it before constructing a new width/output/checkpoint cell.
        if((!owner || !owner->window.valid()) && (model.execution->gguf_expert_cache || model.execution->tq_expert_cache) && !startup_warmup) {
            const auto reserve=runtime_options::workspace_reserve_bytes();
            if(default_context(mfq_current_cuda_device())->available_memory()<reserve) {
                const auto before=routed_expert_cache_budget_bytes(*model.execution);
                // The capture allowance is a reclaim target. Actual graph
                // allocations are checked against the device budget below.
                reserve_routed_capture_workspace(*model.execution,reserve);
                returned_expert_workspace=routed_expert_cache_budget_bytes(*model.execution)<before;
            }
        }
        if(!owner)owner=std::make_unique<Cell>(tokens,model.config.hidden,model.blocks,stream.stream(),confirmed,shared_checkpoints,prediction);
        owner->window.set_profile_geometry(raw_hidden?"mtp_target":"target_decode",tokens);
        auto& cell=*owner;
        if(kind==4 && tokens>1)cell.compact_control.set(compact_sampling,"target",tokens);
        // A retained graph's first replay can still pay cold transfer/cache
        // costs after prefill. Exclude that replay from steady MTP policy data.
        if(initialized)*initialized=!cell.window.valid() || !(cell.used_kinds&(1u<<result_kind));
        if(!cell.ple.empty() && host_ids.defined()) {
            TYLOINFER_RUNTIME_CHECK(!host_ids.is_cuda() && host_ids.scalar_type()==tb::kInt64 &&
                host_ids.sizes()==ids.sizes(),"Qwen4 graph host token geometry changed");
            cell.host_ids.copy_(host_ids);
        } else if(!cell.ple.empty())cell.host_ids.copy_(ids.to(tb::kCPU,tb::kInt64));
        cell.ids.copy_(ids);
        mfq_qwen4_exp::fill_decode_positions(cell.cache_positions,cell.positions,model.cache_pos,model.decode_position_delta);
        try {
            // PLE depends only on token IDs and the committed ngram context.
            // Start its read before waiting for adaptive expert exchanges,
            // rather than leaving that CPU interval outside the I/O overlap.
            // Task reset repeats issue() idempotently; it never advances the
            // context, which is still committed by upload_graph().
            const auto* early_ple=std::getenv("TYLOINFER_PLE_EARLY_ISSUE");
            if(!early_ple || early_ple[0]!='0')
                for(const auto& stage:cell.ple)stage->prefetch_rows(cell.host_ids);
            finish_routed_expert_exchanges(*model.execution);
            // Prefix restoration and eager prefill update the same KV/index
            // storage; refresh pooled keys even when this graph is already valid.
            for(auto* attention:qsa)attention->prepare_graph_tokens(model.positions,kv_capacity,pending_qsa,tokens,true);
            if(!cell.window.valid()) {
                cell.replay_storage.reset();
                prepare_execution_graph_pool(*model.execution,memory_pool);
                DecodeGraphBranchScope branch_scope(
                    model.execution->decode_graph_serial_branches);
                const auto capture_started=std::chrono::steady_clock::now();
                struct State {tb::Tensor* target;tb::Tensor saved;const void* address;};
                std::vector<State> states;
                std::size_t warmup_saved_bytes=0,readonly_gdn_bytes=0;
                if(!startup_warmup) {
                    for(std::size_t layer=0;layer<model.blocks.size();++layer) {
                        auto& block=*model.blocks[layer];
                        auto& b=static_cast<qwen4_exp::Qwen4Block&>(block);
                        // Replay verification records updates without changing
                        // the committed GDN tensors. Preserve only state that
                        // the captured body can actually mutate, including PLE.
                        const auto readonly_gdn=confirmed && cell.checkpoints[layer].gdn_replay && b.gdn
                            ? b.gdn->graph_state():std::vector<tb::Tensor*>{};
                        for(auto* state:block.graph_warmup_state()) {
                            if(std::find(readonly_gdn.begin(),readonly_gdn.end(),state)!=readonly_gdn.end()) {
                                readonly_gdn_bytes+=state->nbytes();
                                continue;
                            }
                            warmup_saved_bytes+=state->nbytes();
                            states.push_back({state,{},state->data_ptr()});
                        }
                    }
                }
                // Capture-only backups must leave room for the recorded body.
                // The live tensors keep their original device addresses.
                const bool host_snapshots=!states.empty() &&
                    warmup_saved_bytes>=default_context(mfq_current_cuda_device())->available_memory();
                for(auto& state:states)
                    state.saved=host_snapshots?state.target->to(tb::kCPU):state.target->clone();
                auto restore=[&] {
                    for(const auto& state:states) {
                        TYLOINFER_RUNTIME_CHECK(state.target->data_ptr()==state.address,"MFQ capture changed recurrent storage");
                        state.target->copy_(state.saved);
                    }
                };
                cell.window.capture([&] {
                    cell.window.mark(-1,"window_begin");
                    auto hidden=model.adapter_prepare_hidden(model.embed_tokens(cell.ids),1,tokens);
                    if (model.hidden_capture) model.hidden_capture->begin(kind >= 3);
                    std::size_t ple_index=0;int layer=0;
                    std::vector<tb::Tensor> prior_mix;
                    const auto chain_setting=mfq::cuda::runtime_options::gr_chain_ffn_post();
                    const bool chain_post=(!chain_setting || *chain_setting) && tokens<=8;
                    for(const auto& block:model.blocks) {
                        cell.window.mark(layer,"block_begin");
                        auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);tb::Tensor rows;
                        if(b.ple) {
                            cell.window.mark(layer,"ple_begin");
                            auto* stage=cell.ple[ple_index++].get();rows=stage->rows;
                            if(stage->transfer->mapped_rows()) {
                                // Token IDs and ngram history are already known.
                                // Reset runs after the previous graph is fenced
                                // and before this graph launches, so mapped rows
                                // need no round trip to the host during replay.
                                cell.window.enroll({stage,[&cell,stage,confirmed] {
                                    stage->prefetch_rows(cell.host_ids);
                                    stage->upload_rows(cell.host_ids,confirmed);
                                },[]{},{},{}});
                            } else {
                            cell.window.enroll({stage,
                                [&cell,stage] {std::memset(stage->flags.data(),0,128);stage->prefetch_rows(cell.host_ids);},
                                [this,&cell,stage,confirmed] {
                                    using Clock=std::chrono::steady_clock;
                                    auto point=stage->profile?Clock::now():Clock::time_point{};
                                    const auto sample=[&](uint64_t& total) {
                                        if(!stage->profile)return;
                                        const auto now=Clock::now();
                                        total+=std::chrono::duration_cast<std::chrono::nanoseconds>(now-point).count();point=now;
                                    };
                                    wait_route_publication(stage->host(),stream.stream());
                                    sample(stage->route_wait_ns);
                                    MfqCudaStreamGuard guard(ple_stream);
                                    stage->upload_rows(cell.host_ids,confirmed);
                                    if(stage->profile)point=Clock::now();
                                    TYLOINFER_CUDA_CHECK(cudaLaunchHostFunc(ple_stream.stream(),[](void* p){publish_mapped_flag(static_cast<uint32_t*>(p));},stage->host()+16));
                                    check_stream_progress(ple_stream.stream());
                                    sample(stage->notify_ns);
                                },{},[stage]{publish_mapped_flag(stage->host()+16);}});
                            signal_mapped_flag(stage->device,stream.stream());wait_mapped_flag(stage->device+16,stream.stream());
                            }
                            stage->transfer->decode();
                            cell.window.mark(layer,"ple_end");
                        }
                        cell.window.mark(layer,"attention_begin");
                        std::vector<tb::Tensor> next_mix;
                        bool defer=false;
                        if(chain_post && size_t(layer+1)<model.blocks.size()) {
                            auto& next=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[layer+1]);
                            defer=!next.ple && next.attention_gr.can_fuse(*model.execution,
                                prior_mix.empty()?hidden:prior_mix[1]);
                            if (model.hidden_capture && model.hidden_capture->contains(layer)) defer = false;
                        }
                        hidden=b.execute_graph(*model.execution,std::move(hidden),cell.ids,cell.positions,
                            cell.cache_positions,rows,layer,confirmed,confirmed?&cell.checkpoints[layer]:nullptr,
                            prior_mix,defer?&next_mix:nullptr);
                        prior_mix=std::move(next_mix);
                        cell.window.mark(layer++,"block_end");
                    }
                    if(kind>=3)cell.raw_hidden=model.adapter_raw_hidden(hidden, {}).clone();
                    hidden=model.adapter_finalize_hidden(std::move(hidden),model.output_norm,1,tokens);
                    if(kind>=3) {
                        cell.output=std::move(hidden);
                        if(kind==4) {
                            cell.logits=model.logits_from_hidden(cell.output);
                            if(tokens>1 && cell.logits.size(-1)>=64) {
                                auto compact=sample_top_k_compact_if_cuda(cell.logits.reshape({tokens,-1}).contiguous(),64,
                                    cell.compact_control.device);
                                if(cell.compact_mapped)sample_publish_packets_cuda(compact,cell.compact_mapped);
                                else TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(cell.compact_host.data_ptr(),compact.data_ptr(),tokens*64*sizeof(int64_t),
                                    cudaMemcpyDeviceToHost,stream.stream()));
                            }
                            auto greedy=sample_greedy_cuda(cell.logits.reshape({tokens,-1}).contiguous());
                            if(cell.greedy_mapped)sample_publish_packets_cuda(greedy,cell.greedy_mapped);
                            else TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(cell.greedy_host.data_ptr(),greedy.data_ptr(),tokens*sizeof(int64_t),
                                cudaMemcpyDeviceToHost,stream.stream()));
                            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1')
                                std::fprintf(stderr,"native_target_packet_publication rows=%d greedy_mapped=%d compact_mapped=%d\n",
                                    tokens,int(cell.greedy_mapped!=nullptr),int(cell.compact_mapped!=nullptr));
                        }
                    }
                    else if(kind==2)cell.output=model.adapter_next_token(model.lm_head,model.last_hidden(std::move(hidden)));
                    else if(kind==1)cell.output=model.adapter_last_logits(model.lm_head,model.last_hidden(std::move(hidden)));
                    else cell.output=model.logits_from_hidden(std::move(hidden));
                    cell.window.mark(-1,"window_end");
                    auto audit=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks.front()).router_audit;
                    if(audit)cell.window.enroll({audit.get(),[]{},[]{},[audit]{audit->finish_sample();},{}});
                },restore,[&]{cell.output={};cell.raw_hidden={};cell.logits={};});
                states.clear();
                ++model.decode_graph_captures;
                std::cerr<<"runtime_qwen4_graph_capture role=target rows="<<tokens<<" kind="<<kind
                    <<" confirmed="<<confirmed<<" segments="<<cell.window.segments()<<" count="<<model.decode_graph_captures
                    <<" qsa_pending_device="<<pending_qsa
                    <<" warmup_saved_bytes="<<warmup_saved_bytes<<" host_snapshots="<<host_snapshots
                    <<" readonly_gdn_bytes="<<readonly_gdn_bytes
                    <<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-capture_started).count()<<'\n';
            }
            if(returned_expert_workspace && model.restore_decode_workspace)model.restore_decode_workspace();
            // Warmup has several capture widths and discards its synthetic
            // history. Delay spare cache allocation until a request uses it.
            if(!startup_warmup)for(auto* attention:qsa)attention->activate_graph_cache();
            const auto host_prepared=profile_host?HostClock::now():HostClock::time_point{};
            cell.window.run();cell.used_kinds|=1u<<result_kind;
            const auto host_replayed=profile_host?HostClock::now():HostClock::time_point{};
            if(confirmed) {
                // Captured copies save the confirmed prefix. Bind their owned
                // storage only after replay, so generic commit/rollback keeps
                // exactly the same transaction lifecycle as eager execution.
                std::shared_ptr<ReplayCommit> replay;
                if(std::any_of(cell.checkpoints.begin(),cell.checkpoints.end(),[](const auto& c){return c.gdn_replay;})) {
                    if(!cell.replay_storage)cell.replay_storage=std::make_shared<ReplayStorage>(
                        cell.checkpoints,replay_convolutions,[&](std::size_t i) -> qwen4_exp::Gdn& {
                            return *static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]).gdn;
                        });
                    replay=std::make_shared<ReplayCommit>(cell.replay_storage);
                }
                std::size_t ple_index=0,layer=0;
                for(const auto& block:model.blocks) {
                    auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);
                    const auto& checkpoint=cell.checkpoints[layer++];
                    const auto position=model.cache_pos+confirmed;
                    if(b.gdn) {
                        if(checkpoint.gdn_replay)b.gdn->bind_graph_replay(checkpoint.gdn_inputs,model.cache_pos,replay);
                        else b.gdn->bind_graph_checkpoint(checkpoint.gdn_conv,checkpoint.gdn_state,position);
                    }
                    if(b.ple)b.ple->bind_graph_checkpoint(checkpoint.ple_conv,cell.ple[ple_index++]->prefix_context,position);
                }
                model.speculative_start=model.cache_pos;model.speculative_confirmed=confirmed;
                model.speculative_prefix_retention=true;
            }
            for(const auto& block:model.blocks) {
                auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);if(b.qsa)b.qsa->advance_graph(tokens);
            }
            model.cache_pos+=tokens;
            model.positions=append_decode_positions(model.positions,cell.positions,position_storage,
                model.max_position_embeddings());model.batch=1;
            const auto host_bound=profile_host?HostClock::now():HostClock::time_point{};
            // MTP can retain history until a later probe. Each step must own its
            // raw residual rather than alias the next replay's capture buffer.
            if(raw_hidden)*raw_hidden=cell.raw_hidden.clone();
            if(kind==4 && tokens>1 && compact_sampling && cell.logits.size(-1)>=64)last_compact=cell.compact_host;
            if(logits)*logits=cell.logits.clone();
            if(greedy_tokens) {
                const auto* first=cell.greedy_host.data_ptr<int64_t>();
                greedy_tokens->assign(first,first+tokens);
            }
            if(profile_host) {
                const auto ms=[](auto duration){return std::chrono::duration<double,std::milli>(duration).count();};
                std::cerr<<"runtime_target_window_host rows="<<tokens
                    <<" prepare_ms="<<ms(host_prepared-host_begin)<<" replay_ms="<<ms(host_replayed-host_prepared)
                    <<" bind_ms="<<ms(host_bound-host_replayed)<<" output_ms="<<ms(HostClock::now()-host_bound)<<'\n';
            }
            if(tokens==1 && result_kind==1)return cell.logits.reshape({1,-1});
            if(tokens==1 && result_kind==2)return cell.greedy_host;
            return cell.output;
        } catch(...) {failed=true;throw;}
    }
};
mfq_tensor_backend::Tensor qwen4_decode_graph_compact_topk(const Qwen4CausalLm& model) {
    return model.graph_registry?model.graph_registry->last_compact:mfq_tensor_backend::Tensor{};
}

std::optional<mfq_tensor_backend::Tensor> qwen4_decode_graph_logits(Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,int kind,
        const mfq_tensor_backend::Tensor& host_ids) {
    if (!streamed_decode_graphs(*model.execution)) return {};
    const bool row_banks=bool(model.execution->tq_expert_cache || model.execution->gguf_expert_cache);
    if((model.config.experts && !row_banks && (!model.execution->config.moe_pipeline || !model.execution->moe_expert_cache)) ||
            model.cache_pos==0 || model.speculative_start>=0)return {};
    if(ids.dim()==1)ids=ids.unsqueeze(0);
    if(ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 || ids.size(1)>8 || model.positions.dim()!=1)return {};
    if(row_banks || !model.config.experts) {
        if(model.execution->tq_expert_cache && !tq_graph_supports_tokens(ids.size(1)))return {};
        for(const auto& block:model.blocks) {
            const auto& b=static_cast<const qwen4_exp::Qwen4Block&>(*block);
            if(!b.tq_graph_ffn || (b.ple && !b.ple->graph_supported()))return {};
        }
    }
    TYLOINFER_RUNTIME_CHECK(model.cache_pos+ids.size(1)<=model.max_position_embeddings(),"MFQ window exceeds context");
    if(!model.graph_registry)model.graph_registry=std::make_unique<Qwen4DecodeGraph>(model);
    // The graph owns its device input. A sampled host token can be copied
    // directly into it without allocating another device tensor per replay.
    auto graph_ids=host_ids.defined()?host_ids:model.device_ids(std::move(ids));
    return model.graph_registry->run(model,std::move(graph_ids),kind,nullptr,0,nullptr,host_ids);
}

void qwen4_report_decode_graph(const Qwen4CausalLm& model,std::ostream& output) {
    if(!model.graph_registry)return;
    for(const auto& [key,cell]:model.graph_registry->cells)
        for(const auto& phase:cell->window.gpu_phases())if(phase.samples)
            output<<"mtp_graph_phase tokens="<<std::get<0>(key)<<" confirmed="<<std::get<2>(key)
                <<" prediction="<<std::get<3>(key)
                <<" layer="<<phase.layer<<" from="<<phase.from<<" to="<<phase.to
                <<" samples="<<phase.samples<<" total_ns="<<phase.total_ns<<'\n';
}

std::vector<std::pair<std::string,double>> qwen4_decode_graph_profile_metrics(
        const Qwen4CausalLm& model,const MtpModule* mtp) {
    std::vector<std::pair<std::string,double>> result;
    const auto* predictor=dynamic_cast<const qwen4_exp::Qwen4ExpMtp*>(mtp);
    if(predictor)result.emplace_back("cuda_mtp_history_tail_replays",double(predictor->history_tail_replays));
    if(predictor)result.emplace_back("cuda_mtp_dense_attention_window",double(predictor->dense_attention_window));
    if(!mfq::cuda::runtime_options::layer_timings())return result;
    const auto append=[&](const auto& cells,const char* role,const char* attribute,const char* state) {
        for(const auto& [key,cell]:cells) {
            auto prefix=std::string("cuda_graph_profile_")+role+"_rows_"+
                std::to_string(std::get<0>(key))+"_"+attribute+"_"+std::to_string(std::get<1>(key))+
                "_"+state+"_"+std::to_string(std::get<2>(key))+"_";
            if constexpr(std::tuple_size_v<std::remove_cvref_t<decltype(key)>> >= 4)
                prefix+="prediction_"+std::to_string(std::get<3>(key))+"_";
            if constexpr(std::tuple_size_v<std::remove_cvref_t<decltype(key)>> >= 5)
                prefix+="qsa_pending_device_"+std::to_string(std::get<4>(key))+"_";
            struct Sample {int64_t total_ns=0,max_ns=0;uint64_t samples=0;};
            std::map<std::string,Sample> phases;
            uint64_t windows=0;
            for(const auto& phase:cell->window.gpu_phases())if(phase.samples) {
                auto& sample=phases[phase.from+"_to_"+phase.to];
                sample.total_ns+=phase.total_ns;sample.samples+=phase.samples;
                sample.max_ns=std::max(sample.max_ns,phase.max_ns);
                windows=std::max(windows,phase.samples);
            }
            if(!windows)continue;
            result.emplace_back(prefix+"windows",double(windows));
            for(const auto& [name,sample]:phases) {
                result.emplace_back(prefix+name+"_ms",sample.total_ns/1e6);
                result.emplace_back(prefix+name+"_samples",double(sample.samples));
                result.emplace_back(prefix+name+"_max_ms",sample.max_ns/1e6);
            }
        }
    };
    if(model.graph_registry)append(model.graph_registry->cells,"target","kind","confirmed");
    if(predictor && predictor->graph_registry)append(predictor->graph_registry->cells,"predictor","dtype","mode");
    return result;
}

bool qwen4_can_verify_device_drafts(const Qwen4CausalLm& model) {
    return model.reuse_decode_graphs && model.graph_registry && !model.config.experts &&
        model.cache_pos>0 && model.positions.dim()==1 && model.speculative_start<0 &&
        streamed_decode_graphs(*model.execution) &&
        std::all_of(model.blocks.begin(),model.blocks.end(),[](const auto& block) {
            const auto& b=static_cast<const qwen4_exp::Qwen4Block&>(*block);
            return !b.ple && (b.prefetch || b.tq_graph_ffn);
        });
}

std::optional<mfq_tensor_backend::Tensor> qwen4_decode_graph_hidden(
        Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,mfq_tensor_backend::Tensor* raw_hidden,int64_t confirmed,
        bool* initialized,const mfq_tensor_backend::Tensor& host_ids,mfq_tensor_backend::Tensor* logits,
        std::vector<int32_t>* greedy_tokens,bool compact_sampling) {
    if(initialized)*initialized=false;
    const bool row_banks=bool(model.execution->tq_expert_cache || model.execution->gguf_expert_cache);
    if(!streamed_decode_graphs(*model.execution) ||
            (model.config.experts && !row_banks && (!model.execution->config.moe_pipeline || !model.execution->moe_expert_cache)) ||
            model.speculative_start>=0)return {};
    // Retained windows own initialized, reset recurrent/KV storage. The
    // first row of a new prompt can replay them without an eager FFN copy.
    if(model.cache_pos==0 && (!model.graph_registry || !model.graph_registry->reusable()))return {};
    if(ids.dim()==1)ids=ids.unsqueeze(0);
    if(ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 || ids.size(1)>8 ||
        (model.cache_pos!=0 && model.positions.dim()!=1) ||
        confirmed<0 || (confirmed && confirmed>=ids.size(1)))return {};
    if(model.execution->tq_expert_cache && !tq_graph_supports_tokens(ids.size(1)))return {};
    for(const auto& block:model.blocks) {
        const auto& b=static_cast<const qwen4_exp::Qwen4Block&>(*block);
        if((!b.prefetch && !b.tq_graph_ffn) || (b.ple && !b.ple->graph_supported()))return {};
    }
    TYLOINFER_RUNTIME_CHECK(model.cache_pos+ids.size(1)<=model.max_position_embeddings(),"MFQ window exceeds context");
    if(!model.graph_registry)model.graph_registry=std::make_unique<Qwen4DecodeGraph>(model);
    return model.graph_registry->run(model,model.device_ids(std::move(ids)),(logits || greedy_tokens)?4:3,raw_hidden,
        static_cast<int>(confirmed),initialized,host_ids,logits,greedy_tokens,compact_sampling);
}

std::optional<std::size_t> qwen4_decode_transient_reserve(
        const Qwen4CausalLm& model,const MtpModule* mtp,int maximum_draft_depth) {
    if(!model.reuse_decode_graphs || !model.graph_registry || !model.graph_registry->reusable())return {};
#ifdef _WIN32
    // The smaller allowance requires device-wide accounting on WDDM. If the
    // optional driver query is unavailable, retain the original capture
    // allowance instead of treating this process's free figure as card-wide.
    if(!physical_device_memory_stats(model.execution->layer_placement.primary_device()))return {};
#endif
    const auto extra=mtp?mtp->extra_decode_workspace_bytes(maximum_draft_depth):std::optional<std::size_t>(0);
    if(!extra)return {};
    const auto rows=mtp?std::min(mtp->maximum_draft_depth(),maximum_draft_depth)+1:1;
    // Wider verification currently uses eager forwards, whose scratch has not
    // been captured or priced by this graph-based planner.
    if(rows<1 || rows>8)return {};
    const auto& graphs=model.graph_registry->cells;
    for(int n=1;n<=rows;++n) {
        const bool ready=std::any_of(graphs.begin(),graphs.end(),[&](const auto& entry) {
            return std::get<0>(entry.first)==n && std::get<1>(entry.first)==4 &&
                std::get<2>(entry.first)==(n>1?1:0) && entry.second && entry.second->window.valid();
        });
        if(!ready)return {};
    }
    // Logit clones, full-vocabulary probability/rejection work, token counts
    // and bounded predictor-history merges remain outside the captured body.
    // Price them from actual vocabulary/hidden/window sizes. The allocator
    // margin also covers pool page granularity and driver/library bookkeeping;
    // it is independent of device/model identity. KV growth is added by caller,
    // and PP, prefix snapshots and new graph cells reclaim their own workspace.
    const auto logits=std::size_t(rows)*model.vocab_size()*sizeof(float)*8;
    const auto history=std::size_t(128)*model.hidden_size()*model.hc_mult()*sizeof(float)*4;
    return mfq::engine::prefill_budget_add((std::size_t(256)<<20)+logits+history,*extra);
}

std::optional<mfq_tensor_backend::Tensor> qwen4_exp::Qwen4ExpMtp::enqueue_greedy_chain(const MtpTarget& target,
        Tensor hidden,Tensor ids,Tensor positions,int depth,bool publish_host) {
    namespace tb=mfq_tensor_backend;
    const auto graph=mfq::cuda::runtime_options::mtp_predictor_graph();
    if(!reuse_decode_graphs || (graph && !*graph) || !graph_registry || batch!=1 || !supports_graph_window(1) ||
        ids.dim()!=2 || ids.size(0)!=1 || hidden.dim()!=3 || hidden.size(0)!=1 || hidden.size(1)!=ids.size(1) ||
        hidden.size(2)!=hidden_norm.numel() || !ids.is_cuda() || ids.scalar_type()!=tb::kInt64 || !hidden.is_cuda() ||
        lengths.empty() || lengths[0]<=0 || this->positions[0].dim()!=3 || this->positions[0].size(1)!=1)return {};
    if(positions.defined()) {
        if(positions.dim()==1)positions=positions.reshape({1,1,-1}).expand({3,ids.size(0),ids.size(1)});
        else if(positions.dim()==2)positions=positions.unsqueeze(1);
        if(positions.dim()!=3 || positions.size(0)!=3 || positions.size(1)!=1 || positions.size(2)!=ids.size(1))return {};
        positions=positions.to(tb::kInt32).contiguous();
    }
    return graph_registry->enqueue_greedy_chain(*this,target,hidden,ids,positions,depth,publish_host);
}

std::optional<std::vector<int32_t>> qwen4_exp::Qwen4ExpMtp::queued_greedy_chain(const MtpTarget& target,
        Tensor hidden,Tensor ids,Tensor positions,int depth,
        const std::function<void(const std::vector<int32_t>&)>& publish) {
    if(!enqueue_greedy_chain(target,std::move(hidden),std::move(ids),std::move(positions),depth,true))return {};
    return graph_registry->collect_greedy_chain(depth,publish);
}

void qwen4_release_decode_graphs(Qwen4CausalLm& model,MtpModule* mtp) {
    finish_routed_expert_exchanges(*model.execution);
    if(model.graph_registry)model.graph_registry->restore_legacy();
    model.graph_registry.reset();
    for(auto& block:model.blocks)static_cast<qwen4_exp::Qwen4Block&>(*block).retain_graph_storage=false;
    if(auto* predictor=dynamic_cast<qwen4_exp::Qwen4ExpMtp*>(mtp)) {
        if(predictor->graph_registry)predictor->graph_registry->restore_legacy(*predictor);
        predictor->graph_registry.reset();
        for(auto& layer:predictor->layers)layer->retain_graph_storage=false;
    }
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
namespace {
namespace tb=mfq_tensor_backend;
std::shared_ptr<tb::TensorStorage> whole_prefill_weight(const tb::Tensor& weight) {
    auto storage=std::static_pointer_cast<tb::TensorStorage>(weight.weak_storage_owner().lock());
    // A replacement invalidates raw device addresses. Only move complete
    // allocations with one tensor view, after retiring every graph reader.
    if(!storage || storage.use_count()!=2 || weight.byte_offset() || storage->bytes!=weight.nbytes())return {};
    return storage;
}
void rebind_prefill_weight(tb::Tensor& weight,const std::shared_ptr<tb::TensorStorage>& storage,
        void* address,std::shared_ptr<void> owner,
        std::function<void(std::uintptr_t,std::shared_ptr<void>)> record_stream,bool mapped) {
    storage->base=address;
    storage->mapped_host=mapped;
    storage->record_stream=std::move(record_stream);
    storage->owner=std::move(owner);
    weight=tb::Tensor(storage,weight.view_descriptor());
}
}
#endif

std::size_t qwen4_borrow_dense_prefill_weights(Qwen4CausalLm& model,MtpModule* mtp,std::size_t bytes,
        const std::function<void(std::size_t)>& reserve_host,std::size_t reserved_ram,bool reserve_live_state) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    const auto& execution=*model.execution;
    if(!bytes || model.config.experts || execution.dense_cpu_layer_count || model_parallel_enabled(execution) ||
            execution.layer_placement.enabled() || model.layer_weights || model.speculative_start>=0)return 0;
    const int device=execution.layer_placement.primary_device();
    const MfqCudaGuard guard(device);
    int mapped=0;
    TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&mapped,cudaDevAttrCanMapHostMemory,device));
    if(!mapped)return 0;
    const auto candidate=[&](QuantLinear& projection) -> tb::Tensor* {
        if(!projection.is_gguf() || projection.tensor_parallel())return nullptr;
        auto& weight=projection.gguf;
        if(weight.rotation || weight.correction || weight.projection_correction || !weight.packed.is_cuda() ||
                weight.packed.get_device()!=device || weight.packed.is_mapped_host() || !weight.packed.is_contiguous())return nullptr;
        return &weight.packed;
    };
    std::vector<tb::Tensor*> candidates;
    for(auto block=model.blocks.rbegin();block!=model.blocks.rend();++block) {
        const auto& dense=static_cast<qwen4_exp::Qwen4Block&>(**block).dense_ffn_weights;
        if(!dense)continue;
        if(auto* weight=candidate(dense->down))candidates.push_back(weight);
        for(auto& projection:dense->gate_up.layers)if(auto* weight=candidate(projection))candidates.push_back(weight);
    }
    if(candidates.empty())return 0;
    model.borrowed_dense_weights.reserve(model.borrowed_dense_weights.size()+candidates.size());
    qwen4_release_decode_graphs(model,mtp);
    const auto context=default_context(device);
    context->reclaim_idle_memory();
    auto headroom=context->available_memory();
    if(reserve_live_state && !mtp && !model.cache_pos && model.borrowed_dense_weights.empty()) {
        const auto needs_indexer=std::any_of(model.loaded_blocks.begin(),model.loaded_blocks.end(),
            [](const auto* layer){return layer->qsa && layer->qsa->indexer_growth_bytes();});
        const auto state_bytes=qwen4_prefill_state_growth_bytes(model,1);
        if(needs_indexer && state_bytes<=headroom) {
            // Sample decode headroom with live state on device, as in prefix
            // restoration. Logical tensor sizes omit allocator page charges.
            try {
                for(auto* layer:model.loaded_blocks) {
                    if(layer->gdn)layer->gdn->reserve_request_state();
                    if(layer->qsa && layer->qsa->indexer_growth_bytes())layer->qsa->reserve_indexer();
                }
            } catch(const MfqResourceExhausted&) {
                (void)cudaGetLastError();
            }
            context->reclaim_idle_memory();
            const auto available=context->available_memory();
            const auto charged=headroom-std::min(headroom,available);
            const auto remaining=qwen4_prefill_state_growth_bytes(model,1);
            const auto reserved=state_bytes-std::min(state_bytes,remaining);
            if(charged>reserved)bytes=mfq::engine::prefill_budget_add(bytes,charged-reserved);
            headroom=available;
        }
    }
    const auto stream=mfq_current_cuda_stream();
    const auto target=mfq::engine::prefill_budget_add(headroom,bytes);
    auto next_check=bytes;
    std::size_t released=0,reclaimed_at=0;
    for(auto* weight:candidates) {
        const auto storage=whole_prefill_weight(*weight);
        if(!storage)continue;
        try {
            const auto required=mfq::engine::prefill_budget_add(reserved_ram,storage->bytes);
            if(reserve_host)reserve_host(required);
            if(required>mfq::sample_host_memory().available_host_bytes())continue;
            auto host=std::make_shared<tb::Tensor>(tb::empty({int64_t(storage->bytes)},
                tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true)));
            void* alias=nullptr;
            const auto status=cudaHostGetDevicePointer(&alias,host->data_ptr(),0);
            if(status==cudaErrorNotSupported || status==cudaErrorInvalidValue) {
                (void)cudaGetLastError();break;
            }
            TYLOINFER_CUDA_CHECK(status);
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(host->data_ptr(),storage->base,storage->bytes,cudaMemcpyDeviceToHost,stream));
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
            rebind_prefill_weight(*weight,storage,alias,std::move(host),{},true);
            if(model.borrowed_dense_weights.empty()) {
                model.dense_prefill_restore_headroom=headroom;
                model.dense_prefill_initial_state_growth=qwen4_prefill_state_growth_bytes(model,1);
            }
            model.borrowed_dense_weights.push_back(weight);
            released+=storage->bytes;
            if(released>=next_check) {
                context->reclaim_idle_memory();
                reclaimed_at=released;
                const auto available=context->available_memory();
                if(available>=target)break;
                // Replacing tensors can leave fragmented pages in the weight
                // pool. Check the physical headroom before ending the loan.
                next_check=mfq::engine::prefill_budget_add(released,target-available);
            }
        } catch(const MfqResourceExhausted&) {
            (void)cudaGetLastError();
        }
    }
    if(released!=reclaimed_at) {
        context->reclaim_idle_memory();
    }
    if(released || std::getenv("TYLOINFER_TRACE_RESIDENCY_MEMORY"))
        std::fprintf(stderr,"native_dense_prefill_weight_loan bytes=%zu required=%zu available=%zu reserved_ram=%zu\n",
            released,bytes,context->available_memory(),reserved_ram);
    return released;
#else
    (void)model;(void)mtp;(void)bytes;(void)reserve_host;(void)reserved_ram;(void)reserve_live_state;return 0;
#endif
}

std::size_t qwen4_borrowed_dense_weight_bytes(const Qwen4CausalLm& model) {
    std::size_t bytes=0;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    for(const auto* weight:model.borrowed_dense_weights)bytes+=weight->nbytes();
#else
    (void)model;
#endif
    return bytes;
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
static void trace_dense_decode_memory(const Qwen4CausalLm& model,const MtpModule* mtp,
        const Context& context,const char* phase) {
    const auto [pool_used,pool_reserved]=context.pool_memory_stats();
    std::size_t cache_bytes=0;
    for(const auto& [name,value]:qwen4_cache_memory_metrics(model,mtp))
        if(name=="kv_cache_bytes")cache_bytes=std::size_t(value);
    std::fprintf(stderr,
        "native_dense_decode_memory phase=%s position=%lld available=%zu local=%zu allocated=%zu "
        "pool_used=%llu pool_reserved=%llu cache=%zu ram_weights=%zu\n",
        phase,static_cast<long long>(model.cache_pos),context.available_memory(),context.local_memory_usage(),
        context.memory_stats().allocated,static_cast<unsigned long long>(pool_used),
        static_cast<unsigned long long>(pool_reserved),cache_bytes,qwen4_borrowed_dense_weight_bytes(model));
}
#endif

void qwen4_restore_dense_prefill_weights(Qwen4CausalLm& model,MtpModule* mtp,std::size_t target,
        std::size_t prefill_workspace,std::optional<std::size_t> batch_decode_workspace) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if(!target)return;
    const auto* trace_setting=std::getenv("TYLOINFER_TRACE_RESIDENCY_MEMORY");
    const bool trace=trace_setting && trace_setting[0]=='1';
    if(model.borrowed_dense_weights.empty() && (!trace || target!=SIZE_MAX))return;
    const auto device=model.execution->layer_placement.primary_device();
    const MfqCudaGuard guard(device);
    const auto context=default_context(device);
    if(model.borrowed_dense_weights.empty()) {
        context->reclaim_idle_memory();
        trace_dense_decode_memory(model,mtp,*context,"decode_ready");
        return;
    }
    qwen4_release_decode_graphs(model,mtp);
    const auto stream=mfq_current_cuda_stream();
    if(target==SIZE_MAX && model.positions.defined()) {
        // The final position tensor survives prefill. Move its small payload
        // before trimming so its original allocation cannot pin scratch pages.
        const ResidentAllocationScope decode_storage;
        model.positions=model.positions.clone();
    }
    if(target==SIZE_MAX)
        for(auto* block:model.loaded_blocks)if(block->qsa)block->qsa->collect_adaptive();
    context->reclaim_idle_memory();
    if(trace && target==SIZE_MAX)trace_dense_decode_memory(model,mtp,*context,"before_return");
    std::size_t restored=0;
    auto& borrowed=model.borrowed_dense_weights;
    auto floor=target==SIZE_MAX?batch_decode_workspace.value_or(model.dense_prefill_restore_headroom):prefill_workspace;
    if(target==SIZE_MAX && !mtp && !batch_decode_workspace) {
        // Inference state allocated during prefill already reduces free VRAM.
        // Keep only the portion of the original headroom that it did not use.
        const auto before=model.dense_prefill_initial_state_growth;
        const auto after=qwen4_prefill_state_growth_bytes(model,1);
        if(before>after)floor-=std::min(floor,before-after);
        const auto attention=qwen4_append_attention_workspace_bytes(model,1);
        if(attention) {
            // Reserve the next decode step, including attention and sampling.
            const auto logits=std::size_t(model.vocab_size())*sizeof(float)*8;
            const auto workspace=qwen4_dense_prefill_workspace(model,after);
            floor=workspace.bytes(1,mfq::engine::prefill_budget_add(attention,logits));
        }
    }
    if(trace)std::fprintf(stderr,"native_dense_prefill_weight_restore_budget available=%zu reusable=%zu floor=%zu ram_bytes=%zu\n",
        context->available_memory(),context->reusable_dense_weight_bytes(),floor,qwen4_borrowed_dense_weight_bytes(model));
    const auto restore_available=[&] {
        for(auto it=borrowed.begin();it!=borrowed.end() && restored<target;) {
            auto& weight=**it;
            const auto storage=whole_prefill_weight(weight);
            const auto available=context->available_memory();
            const auto headroom=available>floor?available-floor:0;
            if(!storage || (storage->bytes>headroom && storage->bytes-headroom>context->reusable_dense_weight_bytes())) {
                ++it;continue;
            }
            try {
                const DenseWeightAllocationScope residency;
                auto image=tb::empty({int64_t(storage->bytes)},weight.options().dtype(tb::kUInt8));
                const auto host=std::static_pointer_cast<tb::Tensor>(storage->owner);
                TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(image.data_ptr(),host->data_ptr(),storage->bytes,cudaMemcpyHostToDevice,stream));
                TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
                context->check_memory_limit();
                if(context->available_memory()<std::min(available,floor)) {
                    image={};
                    context->reclaim_idle_memory();
                    ++it;continue;
                }
                const auto device_storage=std::static_pointer_cast<tb::TensorStorage>(image.weak_storage_owner().lock());
                rebind_prefill_weight(weight,storage,device_storage->base,device_storage->owner,device_storage->record_stream,false);
                restored+=storage->bytes;
                it=borrowed.erase(it);
            } catch(const MfqResourceExhausted&) {
                (void)cudaGetLastError();
                context->reclaim_idle_memory();
                ++it;
            }
        }
    };
    restore_available();
    const auto* indexer_placement=std::getenv("TYLOINFER_QSA_CPU_INDEXER");
    if(target==SIZE_MAX && !batch_decode_workspace && !borrowed.empty() && (!indexer_placement || !*indexer_placement)) {
        // A small number of CPU Indexer layers can keep the streamed dense
        // matrices on device. Preserve the full RAM-only KV reservation.
        const auto next=std::min(model.cache_pos+1,model.max_position_embeddings());
        const auto reserved_ram=qwen4_fixed_prefill_ram_reservation(model,next,1);
        const auto usage_before=context->local_memory_usage();
        std::size_t migrated=0;
        for(auto* block:model.loaded_blocks) {
            if(borrowed.empty())break;
            if(!block->qsa)continue;
            const auto bytes=block->qsa->indexer_storage_bytes();
            const auto free=mfq::sample_host_memory().available_host_bytes();
            if(!bytes || free<reserved_ram || bytes>free-reserved_ram)continue;
            const auto moved=block->qsa->migrate_indexer_to_ram();
            if(!moved)continue;
            migrated+=moved;
            // Pages can stay charged until neighboring allocations retire.
            context->reclaim_idle_memory();
            restore_available();
        }
        if(migrated && trace)
            std::fprintf(stderr,"native_dense_decode_indexer_migration bytes=%zu local_before=%zu local_after=%zu\n",
                migrated,usage_before,context->local_memory_usage());
    }
    std::fprintf(stderr,"native_dense_prefill_weight_restore restored=%zu ram_bytes=%zu\n",
        restored,qwen4_borrowed_dense_weight_bytes(model));
    if(trace && target==SIZE_MAX) {
        context->reclaim_idle_memory();
        trace_dense_decode_memory(model,mtp,*context,"decode_ready");
    }
#else
    (void)model;(void)mtp;(void)target;(void)prefill_workspace;(void)batch_decode_workspace;
#endif
}

std::size_t qwen4_mtp_graph_reserve(const Qwen4CausalLm& model,const MtpModule* mtp,int maximum_draft_depth) {
    const auto* predictor=dynamic_cast<const qwen4_exp::Qwen4ExpMtp*>(mtp);
    const auto* reuse=std::getenv("TYLOINFER_QWEN4_REUSE_GRAPHS");
    const auto* warmup=std::getenv("TYLOINFER_QWEN4_WARMUP_GRAPHS");
    if(!predictor || (reuse && reuse[0]=='0') || (warmup && warmup[0]=='0') ||
            !model.execution->config.moe_pipeline || !streamed_decode_graphs(*model.execution))return 0;
    // The common decode allowance covers verification through four rows.
    // Wider windows retain extra shared recurrent checkpoints and per-shape
    // graph/staging allocations. Price those before admitting expert arenas.
    const int extra_rows=std::max(0,std::min({8,predictor->maximum_draft_depth()+1,maximum_draft_depth+1})-4);
    const auto& c=model.config;
    std::size_t recurrent_bytes=0;
    for(int64_t layer=0;layer<c.layers;++layer)if(c.linear_layer(layer))
        recurrent_bytes+=(std::size_t(c.kernel-1)*(2*c.key_heads+c.value_heads)*c.linear_width+
            std::size_t(c.value_heads)*c.linear_width*c.linear_width)*sizeof(float);
    return std::size_t(extra_rows)*(recurrent_bytes+(std::size_t(256)<<20));
}

void qwen4_warmup_decode_graphs(Qwen4CausalLm& model,MtpModule* mtp,int maximum_draft_depth) {
    const auto* reuse=std::getenv("TYLOINFER_QWEN4_REUSE_GRAPHS");
    model.reuse_decode_graphs=!reuse || reuse[0]!='0';
    auto* predictor=dynamic_cast<qwen4_exp::Qwen4ExpMtp*>(mtp);
    if(predictor)predictor->reuse_decode_graphs=model.reuse_decode_graphs;
    const bool row_banks=bool(model.execution->tq_expert_cache || model.execution->gguf_expert_cache);
    // Row banks bootstrap only one backbone row. Predictor widths still need
    // explicit preparation: a stochastic startup loop may never select some
    // accepted-history lengths, leaving request-time captures in those paths.
    const auto* warmup=std::getenv("TYLOINFER_QWEN4_WARMUP_GRAPHS");
    if(!model.reuse_decode_graphs || (warmup && warmup[0]=='0') ||
        model.max_position_embeddings()<32 || !streamed_decode_graphs(*model.execution) ||
        (model.config.experts && !row_banks && (!model.execution->config.moe_pipeline || !model.execution->moe_expert_cache)))return;
    for(const auto& block:model.blocks) {
        const auto& b=static_cast<const qwen4_exp::Qwen4Block&>(*block);
        if((!b.prefetch && !b.tq_graph_ffn) || (b.qsa && !b.qsa->graph_capacity()) || (b.ple && !b.ple->graph_supported()))return;
    }
    // Startup state contains synthetic tokens and is discarded before serving.
    // Saving every recurrent state during each capture needlessly doubles its
    // transient footprint. Request-time captures still restore their real state.
    struct WarmupScope {
        mfq::cuda::RoutedCacheLearningScope learning;
        std::shared_ptr<MoeExpertCache> cache;
        bool previous;
        explicit WarmupScope(CudaExecutionContext& execution)
            :learning(execution,false),cache(execution.moe_expert_cache),
             previous(suspend_moe_residency(cache,true))
            {Qwen4DecodeGraph::startup_warmup=true;}
        ~WarmupScope(){Qwen4DecodeGraph::startup_warmup=false;
            suspend_moe_residency(cache,previous);}
    } startup_scope(*model.execution);
    namespace tb=mfq_tensor_backend;
    const auto start=std::chrono::steady_clock::now();
    auto scope=model.execution_scope();
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64);
    // Diverse synthetic IDs exercise more routed experts than sixteen copies
    // of one token. Only loading uses these rows; all request state is reset.
    std::vector<int64_t> warm_ids(16);
    for(std::size_t i=0;i<warm_ids.size();++i)warm_ids[i]=int64_t((i*7919+17)%model.vocab_size());
    auto ids=tb::tensor(warm_ids,options.device(tb::kCPU)).to(tb::kCUDA).reshape({1,16});
    model.reset(1);
    if(std::getenv("TYLOINFER_TRACE_RESIDENCY_MEMORY"))std::cerr<<"warmup_memory phase=bootstrap\n";
    // Bootstrap with single-token rows. A sixteen-row eager forward otherwise
    // materializes a whole expert layer just to initialize graph state, before
    // the prompt's workspace loan exists, and defeats tight expert residency.
    std::vector<tb::Tensor> raw_rows;
    // Eager FFN cells retain their transfer wire per layer until this pass
    // completes; eight-row bootstrap cells multiply that transient footprint.
    for(int row=0;row<(row_banks?1:16);++row) {
        auto token=ids.narrow(1,row,1);
        tb::Tensor hidden;
        (void)model.hidden_forward(token,mfq_nullopt,mfq_nullopt,nullptr,mfq_nullopt,&hidden);
        raw_rows.push_back(hidden.clone());
    }
    auto raw=tb::cat(raw_rows,1);
    raw_rows.clear();
    if(row_banks && predictor)raw=raw.expand({1,16,raw.size(2)}).contiguous();
    const int verification_rows=mtp?std::min({8,mtp->maximum_draft_depth()+1,maximum_draft_depth+1}):1;
    const auto warm_verification=[&] {
        // A short output tail or a confidence gate can select any narrower
        // width. Prepare these at load time, independently of draft acceptance.
        // The widest window goes first so narrower cells reuse its arenas.
        for(int rows=verification_rows;rows>=1;--rows) {
            if(model.cache_pos+rows+2>model.max_position_embeddings())model.reset(1);
            tb::Tensor hidden,logits;
            (void)qwen4_decode_graph_hidden(model,ids.narrow(1,0,rows),&hidden,
                rows>1?1:0,nullptr,{},&logits);
            if(rows>1)model.commit_speculative();
        }
    };
    // Bootstrap cells are not the retained whole-model windows. Retire their
    // staging and captured scratch before preparing predictor windows.
    release_moe_decode_buffers(model.execution->moe_expert_cache);
    default_context(model.execution->layer_placement.primary_device())->reclaim_idle_memory();
    if(std::getenv("TYLOINFER_TRACE_RESIDENCY_MEMORY"))std::cerr<<"warmup_memory phase=predictor\n";
    // Capture the normal greedy path and the common MTP verification widths.
    // Synthetic state is cleared afterwards; no prompt or sampling history is saved.
    if(predictor) {
        MtpTarget target(model);
        predictor->reset(1);
        // Full predictor drafts consume one row. Bootstrap the same shape so
        // startup does not retain multi-row expert and pinned transfer arenas.
        for(int row=0;row<16;++row)
            (void)predictor->evaluate(target,raw.narrow(1,row,1),ids.narrow(1,row,1));
        release_moe_decode_buffers(model.execution->moe_expert_cache);
        default_context(model.execution->layer_placement.primary_device())->reclaim_idle_memory();
        // Production drafts run one full predictor row. Committed history
        // appends KV without an FFN/head; multi-row full-predictor graphs
        // retain scratch and driver metadata with no production consumer.
        predictor->trim_cache_to(16);
        (void)predictor->step(target,raw.narrow(1,0,1),ids.narrow(1,0,1));
        for(int rows=4;rows>=1;--rows) {
            predictor->trim_cache_to(16);
            predictor->history_step(target,raw.narrow(1,0,rows),ids.narrow(1,0,rows));
        }
        const bool history_chain=predictor->history_chain_enabled();
        const auto* chain_check=std::getenv("TYLOINFER_VERIFY_MTP_HISTORY_CHAIN");
        const bool verify_chain=chain_check && chain_check[0]=='1' && predictor->layers[0]->qsa->supports_session();
        if(history_chain)for(int rows=2;rows<=std::min(4,verification_rows-1);++rows) {
            Qwen4BlockSessionState expected;
            if(verify_chain) {
                predictor->trim_cache_to(16);
                for(int row=0;row<rows;++row)
                    predictor->history_step(target,raw.narrow(1,row,1),ids.narrow(1,row,1));
                std::size_t bytes=0;
                predictor->layers[0]->qsa->capture_session(expected,16+rows,bytes);
            }
            predictor->trim_cache_to(16);
            predictor->history_step(target,raw.narrow(1,0,rows),ids.narrow(1,0,rows),{},true);
            if(verify_chain) {
                Qwen4BlockSessionState actual;std::size_t bytes=0;
                predictor->layers[0]->qsa->capture_session(actual,16+rows,bytes);
                for(const auto& pair:{std::pair{actual.keys,expected.keys},std::pair{actual.values,expected.values},
                                     std::pair{actual.index,expected.index}}) {
                    const auto a=pair.first.cpu().contiguous(),b=pair.second.cpu().contiguous();
                    TYLOINFER_RUNTIME_CHECK(a.sizes()==b.sizes() && a.scalar_type()==b.scalar_type() &&
                        std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes())==0,"MTP scalar history chain changed KV/index bytes");
                }
                std::cerr<<"runtime_mtp_history_chain_check rows="<<rows<<" KV/index exact PASS\n";
            }
        }
        // Teacher-forced history has no consumers after its attention KV.
        // Batch those projections, retaining a complete FFN/readout only for
        // the final row. Capture all common widths before lending expert VRAM.
        if(history_chain)for(int rows=2;rows<=verification_rows;++rows) {
            for(int axes=0;axes<(verify_chain?2:1);++axes) {
                Tensor custom_positions;
                if(axes)custom_positions=(tb::arange(rows,ids.options().dtype(tb::kInt32))+23)
                    .reshape({1,1,rows}).expand({3,1,rows}).contiguous();
                Qwen4BlockSessionState expected_kv;
                MtpStep expected;
                Tensor expected_attention,expected_ffn_input,expected_attention_input;
                if(verify_chain) {
                    predictor->trim_cache_to(16);
                    expected=predictor->step_positioned(target,raw.narrow(1,axes*4,rows),ids.narrow(1,axes*4,rows),custom_positions);
                    const auto& cell=*predictor->graph_registry->cells.at({rows,int(raw.scalar_type()),0,0});
                    expected_attention=cell.check_attention.narrow(1,rows-1,1).clone();
                    expected_ffn_input=cell.check_ffn_input.narrow(1,rows-1,1).clone();
                    expected_attention_input=cell.check_attention_input.narrow(1,rows-1,1).clone();
                    std::size_t bytes=0;
                    predictor->layers[0]->qsa->capture_session(expected_kv,16+rows,bytes);
                }
                predictor->trim_cache_to(16);
                const auto actual=predictor->history_tail_step(target,raw.narrow(1,axes*4,rows),ids.narrow(1,axes*4,rows),custom_positions);
                if(!actual)break;
                if(verify_chain) {
                    TYLOINFER_RUNTIME_CHECK(predictor->cache_position()==16+rows && actual->chain_hidden.size(1)==1 &&
                        actual->sample_hidden.size(1)==1,"MTP history tail advanced an incorrect number of rows");
                    const auto& cell=*predictor->graph_registry->cells.at({rows,int(raw.scalar_type()),3,0});
                    const auto stage_error=[](const Tensor& actual,const Tensor& expected) {
                        const auto a=actual.to(tb::kCPU,tb::kFloat32).contiguous();
                        const auto b=expected.to(tb::kCPU,tb::kFloat32).contiguous();
                        TYLOINFER_RUNTIME_CHECK(a.sizes()==b.sizes(),"MTP history-tail intermediate shape differs");
                        double maximum=0;
                        for(int64_t i=0;i<a.numel();++i)
                            maximum=std::max(maximum,std::abs(double(a.data_ptr<float>()[i])-b.data_ptr<float>()[i]));
                        return maximum;
                    };
                    const auto raw_error=stage_error(cell.check_attention_input,expected_attention_input);
                    const auto raw_a=cell.check_attention_input.to(tb::kCPU,tb::kFloat32).contiguous();
                    const auto raw_b=expected_attention_input.to(tb::kCPU,tb::kFloat32).contiguous();
                    bool raw_close=true;
                    for(int64_t i=0;i<raw_a.numel();++i) {
                        const auto error=std::abs(double(raw_a.data_ptr<float>()[i])-raw_b.data_ptr<float>()[i]);
                        raw_close=raw_close && std::isfinite(error) && error<=.001+.001*std::abs(raw_b.data_ptr<float>()[i]);
                    }
                    const auto q8_packet=[](const Tensor& value) {
                        auto input=value.to(tb::kFloat32).contiguous();
                        auto output=tb::empty({int64_t(mfq::cuda::gguf::q8_row_bytes(input.size(-1)))},input.options().dtype(tb::kUInt8));
                        mfq::cuda::gguf::quantize_input(input.data_ptr<float>(),1,input.numel(),output.data_ptr(),mfq_current_cuda_stream());
                        return output.cpu().contiguous();
                    };
                    const auto packet_a=q8_packet(cell.check_attention_input),packet_b=q8_packet(expected_attention_input);
                    int changed_bytes=0;
                    for(int64_t i=0;i<packet_a.numel();++i)changed_bytes+=packet_a.data_ptr<uint8_t>()[i]!=packet_b.data_ptr<uint8_t>()[i];
                    std::cerr<<"runtime_mtp_history_tail_stage_check rows="<<rows<<" axes="<<axes
                        <<" pre_output_max_error="<<raw_error<<" pre_output_close="<<raw_close<<" output_q8_changed_bytes="<<changed_bytes
                        <<" attention_max_error="<<stage_error(cell.check_attention,expected_attention)
                        <<" ffn_input_max_error="<<stage_error(cell.check_ffn_input,expected_ffn_input)<<'\n';
                    Qwen4BlockSessionState actual_kv;std::size_t bytes=0;
                    predictor->layers[0]->qsa->capture_session(actual_kv,16+rows,bytes);
                    for(const auto& pair:{std::pair{actual_kv.keys,expected_kv.keys},std::pair{actual_kv.values,expected_kv.values},
                                         std::pair{actual_kv.index,expected_kv.index}}) {
                        const auto a=pair.first.cpu().contiguous(),b=pair.second.cpu().contiguous();
                        TYLOINFER_RUNTIME_CHECK(a.sizes()==b.sizes() && a.scalar_type()==b.scalar_type() &&
                            std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes())==0,"MTP batched history tail changed KV/index bytes");
                    }
                    const auto compare=[&](const MtpStep& reference) {
                        double maximum=0;bool close=true;
                        for(const auto& pair:{std::pair{actual->sample_hidden,reference.sample_hidden.narrow(1,reference.sample_hidden.size(1)-1,1)},
                                             std::pair{actual->chain_hidden,reference.chain_hidden.narrow(1,reference.chain_hidden.size(1)-1,1)}}) {
                            const auto a=pair.first.to(tb::kCPU,tb::kFloat32).contiguous();
                            const auto b=pair.second.to(tb::kCPU,tb::kFloat32).contiguous();
                            TYLOINFER_RUNTIME_CHECK(a.sizes()==b.sizes(),"MTP history-tail output shape differs");
                            for(int64_t i=0;i<a.numel();++i) {
                                const auto error=std::abs(double(a.data_ptr<float>()[i])-b.data_ptr<float>()[i]);
                                maximum=std::max(maximum,error);
                                close=close && std::isfinite(error) && error<=.003+.003*std::abs(b.data_ptr<float>()[i]);
                            }
                        }
                        return std::pair{maximum,close};
                    };
                    const auto [batch_error,batch_close]=compare(expected);
                    // The production reference drafts one query at a time.
                    // Batched GEMVs may regroup FP32 reductions; near router
                    // ties, a tiny query change can select a different expert.
                    // Keep exact batched KV checks and compare the complete
                    // output against both existing projection/reduction orders.
                    predictor->trim_cache_to(16);MtpStep scalar;
                    for(int row=0;row<rows;++row)
                        scalar=predictor->step_positioned(target,raw.narrow(1,axes*4+row,1),ids.narrow(1,axes*4+row,1),
                            custom_positions.defined()?custom_positions.narrow(-1,row,1):Tensor{});
                    const auto [scalar_error,scalar_close]=compare(scalar);
                    std::cerr<<"runtime_mtp_history_tail_reference rows="<<rows<<" axes="<<axes
                        <<" scalar_max_error="<<scalar_error<<" scalar_close="<<scalar_close
                        <<" batch_max_error="<<batch_error<<" batch_close="<<batch_close<<'\n';
                    // GGUF activation quantization is discontinuous at code
                    // boundaries. Small query reduction differences may change
                    // Q8 codes before the unchanged output/FFN projections.
                    // Require a bounded pre-quantization result and observed
                    // packet changes instead of hiding this behind a larger
                    // whole-predictor output tolerance.
                    TYLOINFER_RUNTIME_CHECK(batch_close || scalar_close ||
                        (model.execution->gguf_expert_cache && raw_close && changed_bytes>0),
                        "MTP batched history-tail output differs from both existing predictor reduction orders");
                    std::cerr<<"runtime_mtp_batched_history_tail_check rows="<<rows<<" axes="<<axes
                        <<" KV/index exact scalar_max_error="<<scalar_error<<" batch_max_error="<<batch_error
                        <<" batch_close="<<batch_close<<" PASS\n";
                    // Complete multi-row FFNs were reference-only. They have
                    // no production consumer and must not reduce expert VRAM.
                    for(auto it=predictor->graph_registry->cells.begin();it!=predictor->graph_registry->cells.end();)
                        if(std::get<0>(it->first)==rows && std::get<2>(it->first)==0)it=predictor->graph_registry->cells.erase(it);
                        else ++it;
                }
            }
        }
        // A depth-D draft verifies D+1 target rows. Prepare every supported
        // width before requests lend spare VRAM to resident experts; capturing
        // widths five/six afterwards also needs live recurrent-state clones.
        warm_verification();
        // Capturing and replaying once leaves cold per-width staging paths.
        // Exercise both single-row drafting and batched history/verification
        // again with different IDs before accepting the first request.
        for(int pass=0;pass<2;++pass)for(int rows=1;rows<=verification_rows;++rows) {
            if(model.cache_pos+rows+2>model.max_position_embeddings())break;
            const auto offset=4+pass*4;
            auto tokens=ids.narrow(1,offset,rows);
            predictor->trim_cache_to(16);
            const auto tail=predictor->history_tail_step(target,raw.narrow(1,offset,rows),tokens);
            if(!tail && history_chain && rows>2) {
                for(int first=0;first<rows-1;first+=4) {
                    const auto count=std::min(4,rows-1-first);
                    predictor->history_step(target,raw.narrow(1,offset+first,count),tokens.narrow(1,first,count),{},true);
                }
            }
            else if(!tail)for(int row=0;row+1<rows;++row)
                predictor->history_step(target,raw.narrow(1,offset+row,1),tokens.narrow(1,row,1));
            if(!tail)(void)predictor->step(target,raw.narrow(1,offset+rows-1,1),tokens.narrow(1,rows-1,1));
            tb::Tensor hidden,logits;
            (void)qwen4_decode_graph_hidden(model,tokens,&hidden,rows>1?1:0,nullptr,{},&logits);
            if(rows>1)model.commit_speculative();
        }
        predictor->reset(1);
    } else if(mtp)warm_verification();
    (void)qwen4_decode_graph_logits(model,ids.narrow(1,0,1),2);
    (void)qwen4_decode_graph_logits(model,ids.narrow(1,0,1),1);
    if(moe_expert_cache_ssd_experts(model.execution->moe_expert_cache) &&
            !model.router_lookahead.has_value() && !runtime_options::router_lookahead_override().has_value()) {
        struct Restore {
            Qwen4CausalLm& model;std::optional<bool> previous;
            ~Restore(){model.router_lookahead=previous;}
        } restore{model,model.router_lookahead};
        model.router_lookahead=true;
        if(model.cache_pos+4>=model.max_position_embeddings())model.reset(1);
        (void)qwen4_decode_graph_logits(model,ids.narrow(1,0,1),1);
    }
    model.reset(1);
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    std::cerr<<"runtime_qwen4_graph_warmup target_captures="<<model.decode_graph_captures
        <<" maximum_draft_depth="<<(mtp?maximum_draft_depth:0)
        <<" predictor_captures="<<(predictor?predictor->graph_captures:0)<<" ms="
        <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<'\n';
    release_moe_prefill_buffers(model.execution->moe_expert_cache);
    default_context(model.execution->layer_placement.primary_device())->reclaim_idle_memory();
}

void qwen4_set_rotary_fusion(Qwen4CausalLm& model, bool enabled) {
    TYLOINFER_RUNTIME_CHECK(!model.graph_registry && model.cache_pos == 0,
                      "reset the model before switching rotary execution");
    mfq_cuda_synchronize();
    for (const auto& block : model.blocks) {
        auto* attention = static_cast<qwen4_exp::Qwen4Block&>(*block).qsa.get();
        if (attention) attention->set_rotary_fused(enabled);
    }
}

void qwen4_set_attention_grouping(Qwen4CausalLm& model, bool enabled) {
    TYLOINFER_RUNTIME_CHECK(!model.graph_registry && model.cache_pos == 0,
                      "reset the model before switching attention projection groups");
    mfq_cuda_synchronize();
    for (const auto& block : model.blocks) {
        auto& attention = static_cast<qwen4_exp::Qwen4Block&>(*block);
        if (attention.gdn) attention.gdn->set_grouped_projection(enabled);
        if (attention.qsa) attention.qsa->set_grouped_projection(enabled);
    }
}

void qwen4_begin_router_lookahead_audit(Qwen4CausalLm& model,int samples) {
    if(model.blocks.size()<2 || model.graph_registry || samples<=0)
        throw std::invalid_argument("router lookahead audit requires a fresh multilayer decode session");
    auto audit=std::make_shared<RouterLookaheadInputs>(static_cast<int>(model.blocks.size()),
        static_cast<int>(model.config.hidden),samples);
    for(const auto& block:model.blocks) {
        auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);
        if(!b.router)throw std::runtime_error("router lookahead audit has no retained router");
        b.router_audit=audit;
    }
}
void qwen4_finish_router_lookahead_audit(Qwen4CausalLm& model,const std::string& prefix) {
    auto audit=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks.front()).router_audit;
    if(!audit || !audit->samples())throw std::runtime_error("router lookahead audit captured no decode inputs");
    std::ofstream file(prefix+"-logits.f32",std::ios::binary);
    if(!file)throw std::runtime_error("cannot write router lookahead audit logits");
    for(std::size_t layer=1;layer<model.blocks.size();++layer) {
        auto& b=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[layer]);
        for(int input_layer:{static_cast<int>(layer)-1,static_cast<int>(layer)}) {
            auto input=audit->rows(input_layer).to(mfq_tensor_backend::kCUDA);
            auto logits=b.router(*model.execution,input).to(mfq_tensor_backend::kFloat32).cpu().contiguous();
            if(logits.dim()!=2 || logits.size(0)!=audit->samples() || logits.size(1)!=model.config.experts)
                throw std::runtime_error("router lookahead audit router shape mismatch");
            file.write(reinterpret_cast<const char*>(logits.data_ptr()),logits.numel()*logits.element_size());
            if(!file)throw std::runtime_error("router lookahead audit logits write failed");
        }
    }
    std::cout<<"router_lookahead_audit_samples="<<audit->samples()<<" layers="<<model.blocks.size()
        <<" hidden="<<model.config.hidden<<" experts="<<model.config.experts<<" topk="<<model.config.topk
        <<" scores=predicted_then_same_layer_oracle row_dtype=FP32\n";
}

std::size_t qwen4_prefix_snapshot_bytes(const Qwen4CausalLm& model,const MtpModule* mtp) {
    std::size_t bytes=session_tensor_bytes(model.positions);
    const auto count_block=[&](const qwen4_exp::Qwen4Block& block,int64_t count) {
        if(count<=0)return;
        if(block.qsa)bytes+=block.qsa->session_bytes(count);
        if(block.gdn)for(const auto* tensor:block.gdn->graph_state())bytes+=session_tensor_bytes(*tensor);
        if(block.ple)bytes+=session_tensor_bytes(block.ple->conv_state());
    };
    for(const auto& block:model.blocks)count_block(static_cast<const qwen4_exp::Qwen4Block&>(*block),model.cache_pos);
    if(const auto* predictor=dynamic_cast<const qwen4_exp::Qwen4ExpMtp*>(mtp)) {
        for(std::size_t i=0;i<predictor->layers.size();++i) {
            const auto count=std::min(model.cache_pos-1,predictor->lengths[i]);
            count_block(*predictor->layers[i],count);
            if(count>0)bytes+=std::size_t(count)*3*predictor->positions[i].element_size();
        }
        bytes+=std::size_t(model.config.hidden)*model.config.streams*sizeof(float);
    }
    return bytes;
}

std::vector<std::pair<std::string,double>> qwen4_cache_memory_metrics(
        const Qwen4CausalLm& model,const MtpModule* mtp) {
    // Shared checkpoint views and recurrent state aliases count once. These
    // are existing cache tensors; collecting telemetry never copies GPU data.
    std::map<std::pair<int,std::uintptr_t>,std::size_t> allocations;
    const auto add=[&](const mfq_tensor_backend::Tensor& tensor) {
        if(!tensor.defined() || !tensor.is_cuda())return;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        const auto storage=std::static_pointer_cast<mfq::cuda::TensorStorage>(tensor.weak_storage_owner().lock());
        auto& bytes=allocations[{tensor.get_device(),reinterpret_cast<std::uintptr_t>(storage->base)}];
        bytes=std::max(bytes,storage->bytes);
#else
        auto& bytes=allocations[{tensor.get_device(),reinterpret_cast<std::uintptr_t>(tensor.data_ptr())}];
        bytes=std::max(bytes,std::size_t(tensor.numel())*tensor.element_size());
#endif
    };
    std::set<const void*> ram_pools;
    std::size_t ram_payload=0,ram_allocated=0,ram_target=0,ram_disk=0,index_disk=0,ram_layers=0,ssd_layers=0,auto_layers=0,shared_gpu=0;
    std::array<std::uint64_t,4> ram_tokens{};
    const auto block_cache=[&](const qwen4_exp::Qwen4Block& block) {
        if(block.qsa)if(auto memory=block.qsa->adaptive_ram_memory()) {
            ++ram_layers;ssd_layers+=memory->ssd_enabled;auto_layers+=memory->adaptive_precision;
            ram_payload+=memory->payload;ram_target+=memory->target;ram_disk+=memory->disk;index_disk+=memory->index_disk;
            if(ram_pools.insert(memory->pool).second){ram_allocated+=memory->allocated;shared_gpu+=memory->shared_gpu_cache;}
            for(unsigned i=0;i<4;++i)ram_tokens[i]+=memory->tokens[i];
        }
        if(block.qsa)for(const auto& tensor:block.qsa->resident_cache_tensors())add(tensor);
        if(block.gdn)for(const auto* tensor:block.gdn->graph_state())add(*tensor);
        if(block.ple)for(const auto* tensor:block.ple->graph_state())add(*tensor);
    };
    for(const auto& block:model.blocks)block_cache(static_cast<const qwen4_exp::Qwen4Block&>(*block));
    add(model.positions);
    if(model.graph_registry)add(model.graph_registry->position_storage);
    if(model.graph_registry) {
        for(const auto& tensor:model.graph_registry->replay_convolutions)add(tensor);
        for(const auto& [key,cell]:model.graph_registry->cells)if(cell->replay_storage)add(cell->replay_storage->descriptors);
    }
    if(model.graph_registry)for(const auto& [key,cell]:model.graph_registry->cells)
        for(const auto& checkpoint:cell->checkpoints) {
            for(const auto& tensor:checkpoint.gdn_conv)add(tensor);
            for(const auto& tensor:checkpoint.gdn_state)add(tensor);
            for(const auto& tensor:checkpoint.gdn_inputs)add(tensor);
            for(const auto& tensor:checkpoint.ple_conv)add(tensor);
        }
    if(const auto* predictor=dynamic_cast<const qwen4_exp::Qwen4ExpMtp*>(mtp)) {
        for(const auto& block:predictor->layers)block_cache(*block);
        for(const auto& positions:predictor->positions)add(positions);
        if(predictor->graph_registry)add(predictor->graph_registry->position_storage);
    }
    std::size_t bytes=0;
    std::map<int,std::size_t> devices;
    for(const auto& [address,size]:allocations) { bytes+=size; devices[address.first]+=size; }
    bytes+=shared_gpu;devices[model.execution->layer_placement.primary_device()]+=shared_gpu;
    if(model.execution->kv_memory)
        for(const auto& metric:model.execution->kv_memory->metrics())
            if(metric.first=="kv_memory_gpu_bytes") {
                bytes+=std::size_t(metric.second);
                devices[model.execution->layer_placement.primary_device()]+=std::size_t(metric.second);
            }
    if(model.execution->qsa_kv_store)
        for(const auto& metric:model.execution->qsa_kv_store->metrics())
            if(metric.first=="qsa_kv_resident_bytes") {
                bytes+=std::size_t(metric.second);
                devices[model.execution->layer_placement.primary_device()]+=std::size_t(metric.second);
            }
    std::vector<std::pair<std::string,double>> result{{"kv_cache_bytes",double(bytes)},
        {"kv_cache_contexts",model.cache_pos>0?double(model.batch):0.0}};
    result.insert(result.end(),{{"qsa_cache_auto_enabled",auto_layers?1.:0.},
        {"qsa_cache_fixed_enabled",ram_layers>auto_layers?1.:0.},
        {"qsa_adaptive_ram_enabled",ram_layers?1.:0.},
        {"qsa_adaptive_ssd_enabled",ssd_layers?1.:0.},
        {"qsa_adaptive_ram_payload_bytes",double(ram_payload)},
        {"qsa_adaptive_ram_allocated_bytes",double(ram_allocated)},
        {"qsa_adaptive_ram_target_bytes",double(ram_target)},
        {"qsa_adaptive_ssd_bytes",double(ram_disk)},
        {"qsa_adaptive_index_archive_ssd_bytes",double(index_disk)}});
    for(unsigned i=0;i<4;++i)result.emplace_back("qsa_adaptive_tier_"+std::to_string(i)+"_tokens",double(ram_tokens[i]));
    for(const auto& [device,size]:devices)result.emplace_back("gpu_"+std::to_string(device)+"_live_kv_bytes",double(size));
    return result;
}

Qwen4Model::Qwen4Model() = default;
void qwen4_wait_adaptive_qsa(Qwen4CausalLm& model) {
    for(auto* block:model.loaded_blocks)if(block->qsa)block->qsa->wait_adaptive();
}
void qwen4_release_idle_qsa(Qwen4CausalLm& model,MtpModule* mtp,bool release_indexer) {
    const auto migrated=std::count_if(model.loaded_blocks.begin(),model.loaded_blocks.end(),[](const auto* block) {
        return block->qsa && block->qsa->indexer_migrated_to_ram();
    });
    // A loan during cold prefill must not fix the Indexer placement of the
    // next request. Retire graphs before releasing their mapped pointers.
    if(migrated)qwen4_release_decode_graphs(model,mtp);
    std::size_t released=0;
    for(auto* block:model.loaded_blocks)if(block->qsa) {
        if(!model.config.experts)released+=block->qsa->release_shared_device_cache(true);
        block->qsa->release_idle_adaptive(release_indexer || block->qsa->indexer_migrated_to_ram());
    }
    if(migrated && std::getenv("TYLOINFER_TRACE_RESIDENCY_MEMORY"))
        std::fprintf(stderr,"native_dense_idle_indexer_reset layers=%zu\n",std::size_t(migrated));
    if(auto* predictor=dynamic_cast<qwen4_exp::Qwen4ExpMtp*>(mtp))
        for(const auto& block:predictor->layers)if(block->qsa) {
            if(!model.config.experts)released+=block->qsa->release_shared_device_cache(true);
            block->qsa->release_idle_adaptive(release_indexer);
        }
    if(released)default_context(model.execution->layer_placement.primary_device())->reclaim_idle_memory();
}
void qwen4_discard_idle_qsa_descriptors(Qwen4CausalLm& model) {
    std::size_t bytes=0;
    for(auto* block:model.loaded_blocks)if(block->qsa)bytes+=block->qsa->discard_idle_descriptors();
    if(bytes && std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))
        std::fprintf(stderr,"native_qsa_idle_device_descriptors released_bytes=%zu\n",bytes);
}
void qwen4_set_adaptive_qsa_budget(Qwen4CausalLm& model,std::size_t bytes) {
    for(auto* block:model.loaded_blocks)if(block->qsa)block->qsa->set_adaptive_budget(bytes);
}
std::vector<tyloinfer::qsa::AdaptiveCacheStats> qwen4_adaptive_qsa_stats(const Qwen4CausalLm& model) {
    std::vector<tyloinfer::qsa::AdaptiveCacheStats> result;
    for(auto* block:model.loaded_blocks)if(block->qsa)if(auto stats=block->qsa->adaptive_stats())result.push_back(*stats);
    return result;
}
std::uint64_t qwen4_qsa_ssd_read_bytes(const Qwen4CausalLm& model,const MtpModule* mtp) {
    std::uint64_t bytes=0;
    for(const auto* block:model.loaded_blocks)
        if(block->qsa)bytes+=block->qsa->adaptive_ssd_read_bytes();
    if(const auto* predictor=dynamic_cast<const qwen4_exp::Qwen4ExpMtp*>(mtp))
        for(const auto& block:predictor->layers)
            if(block->qsa)bytes+=block->qsa->adaptive_ssd_read_bytes();
    return bytes;
}
void qwen4_set_hidden_capture_layers(Qwen4CausalLm& model, const std::vector<int64_t>& layers) {
    if (!std::is_sorted(layers.begin(), layers.end()) ||
        std::adjacent_find(layers.begin(), layers.end()) != layers.end() ||
        (!layers.empty() && (layers.front() < 0 || layers.back() >= model.num_hidden_layers())))
        throw std::invalid_argument("invalid Qwen4 hidden capture layers");
    if (model.graph_registry) qwen4_release_decode_graphs(model, nullptr);
    model.hidden_capture = std::make_shared<Qwen4HiddenCapture>();
    model.hidden_capture->layers = layers;
    for (auto& block : model.blocks) static_cast<qwen4_exp::Qwen4Block&>(*block).hidden_capture = model.hidden_capture;
}
Qwen4Model::~Qwen4Model() = default;
Qwen4Model::Qwen4Model(Qwen4Model&&) noexcept = default;
Qwen4Model& Qwen4Model::operator=(Qwen4Model&&) noexcept = default;

struct Qwen4BatchRequestState::Impl {
    struct Layer {
        std::unique_ptr<qwen4_exp::Gdn> gdn;
        std::unique_ptr<qwen4_exp::Qsa> qsa;
        std::unique_ptr<qwen4_exp::Ple> ple;
    };
    std::vector<Layer> layers;
    mfq_tensor_backend::Tensor positions;
    mfq_tensor_backend::Tensor position_storage;
    int64_t cache_position = 0, batch = 1;
    int64_t verification_tokens = 0;
    void swap(Qwen4CausalLm& model) {
        for (std::size_t i = 0; i < layers.size(); ++i) {
            auto& block = static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]);
            layers[i].gdn.swap(block.gdn);
            layers[i].qsa.swap(block.qsa);
            layers[i].ple.swap(block.ple);
        }
        std::swap(positions, model.positions);
        std::swap(cache_position, model.cache_pos);
        std::swap(batch, model.batch);
    }
    std::size_t growth_bytes(int64_t tokens) const {
        std::size_t bytes = 0;
        for (const auto& layer : layers) {
            if (layer.qsa) bytes += layer.qsa->growth_bytes(tokens);
            if (layer.gdn) bytes += layer.gdn->growth_bytes();
        }
        return bytes;
    }
};

Qwen4BatchRequestState::Qwen4BatchRequestState(Qwen4CausalLm& model,std::size_t available_request_slots,
        std::size_t prefill_workspace,int prefill_recall_rows,std::size_t prefix_tokens,
        const std::function<std::size_t(std::size_t)>& reclaim_indexer,
        const std::function<std::size_t(std::size_t)>& reclaim_host_kv)
    : state_(std::make_unique<Impl>()) {
    const auto primary = model.execution->layer_placement.primary_device();
    bool host_metadata=false;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    const bool spill_kv=model.execution->kv_memory?model.execution->kv_memory->ssd_enabled():
        kv_memory_options().placement=="auto_ssd";
    std::size_t codec_growth=0;
    if(!model.config.experts && spill_kv && prefix_tokens) {
        for(const auto* block:model.loaded_blocks)if(block->qsa)
            codec_growth=std::max(codec_growth,block->qsa->codec_workspace_growth_bytes(prefix_tokens));
        // Prefix restoration can start the shared codec while another row
        // appends. Existing codec allocations are already in the free sample.
        prefill_workspace=mfq::engine::prefill_budget_add(prefill_workspace,codec_growth);
    }
    if(!model.config.experts && !std::getenv("TYLOINFER_QSA_HOST_METADATA")) {
        std::size_t state_bytes=0,recall=0;
        for(const auto* block:model.loaded_blocks) {
            if(block->gdn)state_bytes=mfq::engine::prefill_budget_add(state_bytes,block->gdn->recurrent_state_bytes());
            if(block->qsa) {
                state_bytes=mfq::engine::prefill_budget_add(state_bytes,block->qsa->indexer_storage_bytes());
                recall=std::max(recall,block->qsa->minimum_recall_workspace_bytes());
            }
        }
        const auto workspace=mfq::engine::prefill_budget_add(prefill_workspace,recall*prefill_recall_rows);
        const auto available=default_context(primary)->available_memory();
        const auto per_request=available>workspace?(available-workspace)/available_request_slots:0;
        host_metadata=recall && state_bytes>per_request && qsa_host_metadata_enabled(true);
        if(host_metadata && std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))
            std::fprintf(stderr,"native_qsa_metadata_placement tier=host state_bytes=%zu device_share=%zu\n",state_bytes,per_request);
    }
#endif
    std::size_t host_indexer_remaining=0,host_indexer_bytes=0,host_indexer_layers=0;
    if(const auto* limit=std::getenv("TYLOINFER_QWEN4_BATCH_HOST_INDEXER_BYTES");limit && !model.config.experts) {
        int mapped=0;
        TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&mapped,cudaDevAttrCanMapHostMemory,primary));
        if(mapped)host_indexer_remaining=std::min<std::uint64_t>(std::strtoull(limit,nullptr,10),
            mfq::sample_host_memory().available_host_bytes());
    }
    for (const auto& base : model.blocks) {
        TYLOINFER_RUNTIME_CHECK(!base->cpu_offloaded && base->cuda_device == primary,
                          "Flash-Next continuous batching requires one CUDA device");
        const auto& block = static_cast<const qwen4_exp::Qwen4Block&>(*base);
        Impl::Layer layer;
        if (block.gdn) {
            layer.gdn = std::make_unique<qwen4_exp::Gdn>(*block.gdn);
            layer.gdn->reset();
        }
        if (block.qsa) {
            const auto bytes=block.qsa->indexer_storage_bytes();
            const bool host=bytes && bytes<=host_indexer_remaining;
            if(host){host_indexer_remaining-=bytes;host_indexer_bytes+=bytes;++host_indexer_layers;}
            layer.qsa=block.qsa->fork_empty(host,host_metadata);
        }
        if (block.ple) { layer.ple = std::make_unique<qwen4_exp::Ple>(*block.ple); layer.ple->reset(); }
        state_->layers.push_back(std::move(layer));
    }
    if(host_indexer_layers && std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))
        std::fprintf(stderr,"native_qwen4_host_indexer bytes=%zu layers=%zu\n",host_indexer_bytes,host_indexer_layers);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if(const auto* limit=std::getenv("TYLOINFER_QWEN4_BATCH_HOST_RECURRENCE_BYTES");limit && !model.config.experts)
        map_recurrence_to_host(primary,std::strtoull(limit,nullptr,10),false);
#endif
    std::size_t fixed=0;
    for(const auto& layer:state_->layers)if(layer.gdn)fixed+=layer.gdn->request_cache_growth_bytes();
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    const auto* indexer_placement=std::getenv("TYLOINFER_QSA_CPU_INDEXER");
    const bool ram_attention=std::any_of(state_->layers.begin(),state_->layers.end(),[](const auto& layer) {
        return layer.qsa && layer.qsa->minimum_recall_workspace_bytes();
    });
    if(!model.config.experts && ram_attention && (!indexer_placement || !*indexer_placement) &&
            !std::getenv("TYLOINFER_QWEN4_BATCH_HOST_INDEXER_BYTES")) {
        std::size_t indexer_bytes=0,recall_bytes=0;
        for(const auto& layer:state_->layers)if(layer.qsa) {
            indexer_bytes=mfq::engine::prefill_budget_add(indexer_bytes,layer.qsa->indexer_growth_bytes());
            recall_bytes=std::max(recall_bytes,layer.qsa->minimum_recall_workspace_bytes());
        }
        const auto workspace=mfq::engine::prefill_budget_add(prefill_workspace,recall_bytes*prefill_recall_rows);
        const auto minimum=mfq::engine::prefill_budget_add(fixed,workspace);
        auto required=mfq::engine::prefill_budget_add(minimum,indexer_bytes);
        const auto context=default_context(primary);
        // The request slots share the append and attention recall workspaces.
        const auto request_budget=[&](std::size_t bytes) {
            return bytes>workspace?workspace+(bytes-workspace)/available_request_slots:bytes;
        };
        if(required>request_budget(context->available_memory())) {
            context->reclaim_idle_memory();
            auto available=context->available_memory();
            while(reclaim_indexer && available<minimum) {
                const auto before=available;
                if(!reclaim_indexer(minimum-available))break;
                context->reclaim_idle_memory();
                available=context->available_memory();
                if(std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))
                    std::fprintf(stderr,"native_batch_qsa_indexer_device_headroom before=%zu after=%zu\n",before,available);
            }
            int mapped=0;
            TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&mapped,cudaDevAttrCanMapHostMemory,primary));
            const auto protected_ram=spill_kv?0:fixed_prefill_ram_reservation(prefix_tokens);
            const auto host_available=[&]() -> std::size_t {
                if(!mapped)return 0;
                const auto free=mfq::sample_host_memory().available_host_bytes();
                return free>protected_ram?free-protected_ram:0;
            };
            auto host=host_available();
            const auto budget=request_budget(available);
            const auto movable=mfq::engine::prefill_budget_add(indexer_bytes,fixed);
            const auto needed=required>budget?std::min(required-budget,movable):0;
            while(mapped && reclaim_host_kv && needed>host) {
                const auto before=host;
                const auto released=reclaim_host_kv(needed-host);
                host=host_available();
                if(!released || host<=before)break;
            }
            std::size_t placed=0;
            for(auto& layer:state_->layers)if(required>budget && layer.qsa) {
                const auto bytes=layer.qsa->indexer_growth_bytes();
                if(bytes && bytes<=host && layer.qsa->place_unallocated_indexer_in_ram()) {
                    host-=bytes;required-=bytes;placed+=bytes;
                }
            }
            if(placed && std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))
                std::fprintf(stderr,"native_qsa_indexer_placement tier=host reason=request_admission bytes=%zu "
                    "headroom=%zu workspace=%zu codec_growth=%zu slots=%zu host_remaining=%llu\n",placed,available,workspace,codec_growth,
                    available_request_slots,static_cast<unsigned long long>(host));
            if(host && fixed && minimum>budget && !std::getenv("TYLOINFER_QWEN4_BATCH_HOST_RECURRENCE_BYTES")) {
                const auto& c=model.config;
                const auto layer_bytes=std::size_t(c.value_heads)*c.linear_width*c.linear_width*sizeof(float);
                if(layer_bytes) {
                    const auto needed=(1+(minimum-budget-1)/layer_bytes)*layer_bytes;
                    fixed-=map_recurrence_to_host(primary,std::min<std::uint64_t>(needed,host),false);
                }
            }
        }
    }
#endif
    if(fixed && model.reserve_kv_workspace)
        model.reserve_kv_workspace(fixed+(model.config.experts?(std::size_t(64)<<20):0));
    for(auto& layer:state_->layers)if(layer.gdn)layer.gdn->initialize_request_cache(false);
}
std::size_t Qwen4BatchRequestState::map_recurrence_to_host(int device_index,std::size_t desired,bool live_only) {
    std::vector<qwen4_exp::Gdn*> layers;
    for(auto& layer:state_->layers)if(layer.gdn)layers.push_back(layer.gdn.get());
    return qwen4_exp::map_request_recurrence(device_index,layers,desired,live_only);
}
Qwen4BatchRequestState::~Qwen4BatchRequestState() = default;
std::size_t Qwen4BatchRequestState::restore_prefix(Qwen4CausalLm& model,
        internal::TextSessionCache& cache, const std::string& session,
        const std::vector<int64_t>& prompt, std::size_t maximum, MtpModule* predictor,
        mfq_tensor_backend::Tensor* predictor_hidden,std::span<const Qwen4BatchRequestState* const> peers) {
    struct ClearPeers {
        Impl& state;
        ~ClearPeers(){for(auto& layer:state.layers)if(layer.qsa)layer.qsa->set_restore_peers({});}
    } clear_peers{*state_};
    std::vector<const qwen4_exp::Qsa*> layer_peers;
    layer_peers.reserve(peers.size());
    if(!peers.empty())for(std::size_t i=0;i<state_->layers.size();++i)if(auto& qsa=state_->layers[i].qsa) {
        layer_peers.clear();
        for(const auto* peer:peers)
            if(peer && peer!=this && peer->state_->layers.size()==state_->layers.size())
                layer_peers.push_back(peer->state_->layers[i].qsa.get());
        qsa->set_restore_peers(layer_peers);
    }
    state_->swap(model);
    try {
        auto restored=cache.restore_best(model,predictor,session,prompt,maximum);
        state_->swap(model);
        state_->position_storage={};
        if(predictor_hidden)*predictor_hidden=std::move(restored.mtp_last_target_hidden);
        return restored.tokens;
    } catch (...) { state_->swap(model);throw; }
}
TextSessionState Qwen4BatchRequestState::capture_prefix(Qwen4CausalLm& model,
        const std::vector<int64_t>& tokens, MtpModule* predictor,
        const mfq_tensor_backend::Tensor& predictor_hidden) {
    state_->swap(model);
    try {
        auto snapshot=model.capture_text_session_state(tokens);
        if(predictor) {
            snapshot.mtp=predictor->capture_session_state(snapshot.cache_pos,predictor_hidden);
            snapshot.bytes+=snapshot.mtp->bytes;
        }
        state_->swap(model);
        return snapshot;
    } catch (...) { state_->swap(model); throw; }
}
int64_t Qwen4BatchRequestState::position() const { return state_->cache_position; }
void Qwen4BatchRequestState::finish_verification(int64_t accepted_tokens) {
    auto& state=*state_;
    TYLOINFER_RUNTIME_CHECK(state.verification_tokens>0 && accepted_tokens>0 &&
        accepted_tokens<=state.verification_tokens,"invalid packed verification acceptance");
    if(state.verification_tokens>1) {
        const auto keep=state.cache_position-state.verification_tokens+accepted_tokens;
        if(accepted_tokens==state.verification_tokens) {
            for(auto& layer:state.layers) {
                if(layer.gdn)layer.gdn->commit();
                if(layer.ple)layer.ple->commit();
            }
        } else {
            for(auto& layer:state.layers) {
                if(layer.gdn)layer.gdn->rollback(keep);
                if(layer.qsa)layer.qsa->truncate(keep);
                if(layer.ple)layer.ple->rollback(keep);
            }
            state.positions=state.positions.narrow(-1,0,keep);
            state.cache_position=keep;
        }
    }
    state.verification_tokens=0;
}
std::size_t Qwen4BatchRequestState::host_recurrence_bytes() const {
    std::size_t bytes=0;
    for(const auto& layer:state_->layers)if(layer.gdn)bytes+=layer.gdn->host_recurrent_state_bytes();
    return bytes;
}
std::size_t Qwen4BatchRequestState::prefill_growth_bytes(int64_t tokens) const {
    return state_->growth_bytes(tokens);
}

std::size_t Qwen4BatchRequestState::prefill_attention_workspace_bytes(int64_t tokens,bool packed_history) const {
    std::size_t bytes=0;
    for(const auto& layer:state_->layers)if(layer.qsa)
        bytes=std::max(bytes,mfq::engine::prefill_budget_add(layer.qsa->append_attention_workspace_bytes(tokens),
            packed_history?layer.qsa->append_packed_history_bytes(tokens):0));
    return bytes;
}
std::size_t Qwen4BatchRequestState::fixed_prefill_ram_reservation(int64_t target_tokens) const {
    std::size_t bytes=0;
    for(const auto& layer:state_->layers)if(layer.qsa)
        bytes=mfq::engine::prefill_budget_add(bytes,layer.qsa->fixed_prefill_ram_reservation(target_tokens));
    return bytes;
}
std::size_t Qwen4BatchRequestState::reclaim_indexer_for_admission(std::size_t bytes) {
    std::size_t released=0;
    for(auto& layer:state_->layers)if(released<bytes && layer.qsa)released+=layer.qsa->migrate_indexer_to_ram();
    return released;
}
std::size_t Qwen4BatchRequestState::promote_indexers(std::size_t decode_workspace) {
    std::size_t bytes=0;
    for(auto& layer:state_->layers)if(layer.qsa)bytes+=layer.qsa->migrate_indexer_to_device(decode_workspace);
    return bytes;
}
std::size_t Qwen4BatchRequestState::release_shared_device_cache(bool spare_only) {
    std::size_t released=0;
    for(auto& layer:state_->layers)if(layer.qsa)released+=layer.qsa->release_shared_device_cache(spare_only);
    return released;
}
std::size_t Qwen4BatchRequestState::reclaim_pinned_for_admission(std::size_t bytes) {
    std::size_t released=0;
    for(auto& layer:state_->layers) {
        if(released>=bytes)break;
        if(layer.qsa)released+=layer.qsa->reclaim_pinned_for_admission(bytes-released);
    }
    return released;
}
std::size_t Qwen4BatchRequestState::resident_bytes() const {
    std::map<std::pair<int,std::uintptr_t>,std::size_t> allocations;
    const auto add = [&](const mfq_tensor_backend::Tensor& tensor) {
        if (!tensor.defined() || !tensor.is_cuda()) return;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if(tensor.is_mapped_host())return;
#endif
        auto& bytes = allocations[{tensor.get_device(),reinterpret_cast<std::uintptr_t>(tensor.data_ptr())}];
        bytes = std::max(bytes, std::size_t(tensor.numel()) * tensor.element_size());
    };
    add(state_->positions);
    add(state_->position_storage);
    for (const auto& layer : state_->layers) {
        if (layer.qsa) for (const auto& tensor : layer.qsa->resident_cache_tensors()) add(tensor);
        if (layer.gdn) for (const auto* tensor : layer.gdn->graph_state()) add(*tensor);
        if (layer.ple) for (const auto* tensor : layer.ple->graph_state()) add(*tensor);
    }
    std::size_t bytes = 0;
    for (const auto& [address,size] : allocations) bytes += size;
    return bytes;
}
std::vector<QsaRamMemory> Qwen4BatchRequestState::adaptive_ram_memory() const {
    std::vector<QsaRamMemory> result;
    for(const auto& layer:state_->layers)if(layer.qsa)
        if(auto memory=layer.qsa->adaptive_ram_memory())result.push_back(*memory);
    return result;
}
std::size_t qwen4_pinned_transaction_bytes(const Qwen4CausalLm& model,int64_t tokens) {
    std::size_t layers=0;
    for(const auto& base:model.blocks) {
        const auto* attention=static_cast<const qwen4_exp::Qwen4Block&>(*base).qsa.get();
        if(attention && attention->adaptive_ram_memory())++layers;
    }
    const auto quantum=tyloinfer::qsa::RamPressureConfig{}.quantize_tokens;
    const auto count=std::size_t((tokens+3)/4*4)+2*std::size_t(quantum);
    const auto row=std::size_t(model.config.kv_heads)*std::size_t(model.config.width)*4;
    if(layers && (row>SIZE_MAX/layers || count>SIZE_MAX/(row*layers)))
        throw std::overflow_error("Pinned append transaction exceeds addressable storage");
    return count*row*layers;
}
std::size_t qwen4_fixed_prefill_ram_reservation(const Qwen4CausalLm& model,int64_t target_tokens,int64_t tile_rows) {
    std::size_t bytes=0;
    for(const auto* layer:model.loaded_blocks)if(layer->qsa)
        bytes=mfq::engine::prefill_budget_add(bytes,layer->qsa->fixed_prefill_ram_reservation(target_tokens));
    return bytes?mfq::engine::prefill_budget_add(bytes,qwen4_pinned_transaction_bytes(model,tile_rows)):0;
}
int64_t qwen4_rebalance_dense_prefill_weights(Qwen4CausalLm& model,MtpModule* mtp,
        int64_t target_tokens,int64_t rows) {
    const auto borrowed=qwen4_borrowed_dense_weight_bytes(model);
    if(!borrowed)return rows;
    const auto reserved=qwen4_fixed_prefill_ram_reservation(model,target_tokens,rows);
    if(!reserved)return rows;
    auto free=mfq::sample_host_memory().available_host_bytes();
    if(free<reserved) {
        mfq_release_host_allocator_cache();
        free=mfq::sample_host_memory().available_host_bytes();
    }
    if(free>=reserved)return rows;
    const auto workspace=qwen4_dense_prefill_workspace(model,qwen4_prefill_state_growth_bytes(model,rows));
    const auto bytes=[&](int64_t count) {
        return workspace.bytes(count,qwen4_append_attention_workspace_bytes(model,count));
    };
    qwen4_restore_dense_prefill_weights(model,mtp,reserved-free,bytes(1));
    if(qwen4_borrowed_dense_weight_bytes(model)>=borrowed)return rows;
    const auto context=default_context(model.execution->layer_placement.primary_device());
    const auto selected=mfq::engine::select_prefill_chunk(rows,rows,context->available_memory(),bytes,true);
    return std::max<int64_t>(1,selected);
}
mfq_tensor_backend::Tensor Qwen4BatchRequestState::prefill(
        Qwen4CausalLm& model, mfq_tensor_backend::Tensor ids,
        const std::function<std::size_t(std::size_t)>& reclaim_pinned,mfq_tensor_backend::Tensor* raw_hidden) {
    namespace tb=mfq_tensor_backend;
    TYLOINFER_RUNTIME_CHECK(ids.size(0) == 1 && ids.size(1) > 0 &&
        ids.size(1) <= model.max_position_embeddings() - position(), "invalid Flash-Next request prefill");
    if(ids.size(1)>1024 && !default_context(mfq_current_cuda_device())->workspace_loan_active())
        reserve_routed_workspace(*model.execution,runtime_options::workspace_reserve_bytes());
    if (model.reserve_kv_workspace) {
        const auto growth=state_->growth_bytes(ids.size(1));
        const auto reserve=model.config.experts?(std::size_t(256)<<20):0;
        if(growth || reserve)model.reserve_kv_workspace(growth+reserve);
    }
    const auto device=tb::Device(tb::kCUDA,model.execution->layer_placement.primary_device());
    std::size_t host_bytes=0;
    for(const auto& layer:state_->layers)if(layer.ple)
        host_bytes=mfq::engine::prefill_budget_add(host_bytes,layer.ple->workspace_plan(ids.size(1),device).host_bytes);
    std::vector<std::shared_ptr<void>> host_loans;std::set<const void*> pools;
    for(const auto& layer:state_->layers)if(layer.qsa)if(auto memory=layer.qsa->adaptive_ram_memory();
            memory && pools.insert(memory->pool).second)
        host_loans.push_back(layer.qsa->hold_adaptive_workspace(host_bytes));
    RequiredWorkspaceLease workspace(device.index,mfq_current_cuda_stream());
    const auto pinned_transaction=mfq::engine::prefill_budget_add(
        qwen4_pinned_transaction_bytes(model,ids.size(1)),host_bytes);
    reserve_pinned_workspace(pinned_transaction,pinned_transaction,reclaim_pinned);
    if(host_bytes)reclaim_routed_optional_host(*model.execution,host_bytes);
    for(const auto& layer:state_->layers)if(layer.ple)layer.ple->prepare_workspace(ids.size(1),device,workspace);
    PinnedAllocationScope required_pinned(device.index,mfq_current_cuda_stream(),[&](std::size_t bytes) {
        if(DecodeWindow::recording())return std::size_t(0);
        return reclaim_routed_expert_pinned(*model.execution,bytes);
    });
    state_->swap(model);
    try {
        qwen4_exp::Qsa::PrefillWorkspaceScope attention_workspace;
        auto hidden = model.hidden_forward(std::move(ids),mfq_nullopt,mfq_nullopt,nullptr,mfq_nullopt,raw_hidden);
        state_->swap(model);
        return hidden;
    } catch (...) { state_->swap(model); throw; }
}

struct Qwen4BatchDecodeGraph::Impl {
    using Tensor = mfq_tensor_backend::Tensor;
    using Checkpoints=std::vector<std::vector<qwen4_exp::RecurrentGraphCheckpoint>>;
    using ReplayStorage=Qwen4DecodeGraph::ReplayStorage;
    Qwen4CausalLm& model;
    MfqCudaStream stream;
    Event ready;
    std::unique_ptr<RoutedGraphPools> memory_pool;
    std::unique_ptr<DecodeWindow> window;
    struct Variant {
        std::unique_ptr<DecodeWindow> window;
        Tensor logits,raw_hidden;
        Checkpoints checkpoints;
        std::vector<std::shared_ptr<ReplayStorage>> replay_storage;
        std::uint64_t touched;
    };
    std::map<std::vector<int>,Variant> variants;
    std::vector<int> active_variant;
    std::uint64_t variant_clock=0;
    std::vector<Qwen4BatchRequestState*> requests;
    std::vector<int64_t> capacities,widths;
    std::vector<std::vector<int>> qsa_pending;
    std::vector<Tensor> positions, cache_positions;
    std::vector<Tensor> gdn_addresses;
    std::vector<std::vector<std::shared_ptr<RowStage>>> ple;
    Tensor ids, logits,raw_hidden;
    Checkpoints checkpoints;
    std::vector<std::shared_ptr<ReplayStorage>> replay_storage;
    std::vector<std::vector<Tensor>> replay_convolutions;
    bool greedy = false;
    int64_t captures = 0, replays = 0;
    Timings timings;
    static int pending_kind(qwen4_exp::Qsa* attention,int64_t width) {
        if(!attention)return 0;
        const auto kind=attention->pending_device_graph_kind(width);
        return kind>=4 && attention->pending_graph_task_supported()?(kind&3)+8:kind;
    }
    explicit Impl(Qwen4CausalLm& owner) : model(owner), stream(mfq_get_current_cuda_stream()) {}
    void select_variant(std::vector<int> signature,int rows,bool verify) {
        if(window && signature==active_variant)return;
        if(window)variants.insert_or_assign(active_variant,
            Variant{std::move(window),std::move(logits),std::move(raw_hidden),std::move(checkpoints),
                std::move(replay_storage),++variant_clock});
        if(const auto found=variants.find(signature);found!=variants.end()) {
            window=std::move(found->second.window);logits=std::move(found->second.logits);
            raw_hidden=std::move(found->second.raw_hidden);checkpoints=std::move(found->second.checkpoints);
            replay_storage=std::move(found->second.replay_storage);
            variants.erase(found);
        } else {
            // Retain recent host-staging combinations while pressure changes
            // which request caches need the mapped graph task.
            while(variants.size()>=4) {
                const auto oldest=std::min_element(variants.begin(),variants.end(),
                    [](const auto& a,const auto& b){return a.second.touched<b.second.touched;});
                variants.erase(oldest);
            }
            // Keep CPU Indexer submission separate from routed-expert host work.
            window=std::make_unique<DecodeWindow>(stream.stream(),runtime_options::layer_timings(),!model.config.experts);
            window->set_profile_geometry(verify?"batch_mtp_target":"batch_decode",rows);
        }
        active_variant=std::move(signature);
    }
    void clear() {
        if (requests.empty() && !window) return;
        finish_routed_expert_exchanges(*model.execution);
        window.reset();variants.clear();active_variant.clear();
        replay_storage.clear();replay_convolutions.clear();
        memory_pool.reset();
        if(!model.graph_registry) {
            // Membership changes retire every batch graph consumer. Release
            // its scratch before converting caches or capturing a new width.
            // Otherwise arrivals retain scratch for every previous row count.
            release_moe_decode_buffers(model.execution->moe_expert_cache);
            default_context(model.execution->layer_placement.primary_device())->reclaim_idle_memory();
        }
        for (auto* request : requests)
            for (auto& layer : request->state_->layers) if (layer.qsa) layer.qsa->leave_graph();
        requests.clear(); capacities.clear();widths.clear();qsa_pending.clear(); positions.clear(); cache_positions.clear(); ple.clear();gdn_addresses.clear();
        ids = {}; logits = {};raw_hidden={};checkpoints.clear();
    }
};
Qwen4BatchDecodeGraph::Qwen4BatchDecodeGraph(Qwen4CausalLm& model)
    : graph_(std::make_unique<Impl>(model)) {}
Qwen4BatchDecodeGraph::~Qwen4BatchDecodeGraph() = default;
void Qwen4BatchDecodeGraph::clear() { graph_->clear(); }
int64_t Qwen4BatchDecodeGraph::captures() const { return graph_->captures; }
int64_t Qwen4BatchDecodeGraph::replays() const { return graph_->replays; }
Qwen4BatchDecodeGraph::Timings Qwen4BatchDecodeGraph::timings() const { return graph_->timings; }
std::optional<mfq_tensor_backend::Tensor> Qwen4BatchDecodeGraph::run(
        const std::vector<Qwen4BatchRequestState*>& requests,
        const mfq_tensor_backend::Tensor& host_ids,bool greedy,const std::vector<int64_t>& widths,
        mfq_tensor_backend::Tensor* raw_hidden) {
    auto& model=graph_->model;
    const auto& geometry=model.config;
    const auto device=model.execution->layer_placement.primary_device();
    const auto recurrence_bytes=std::size_t(geometry.value_heads)*geometry.linear_width*geometry.linear_width*sizeof(float);
    auto target=recurrence_bytes;
    for(;;) {
        bool retryable=true;
        try {return run_once(requests,host_ids,greedy,retryable,widths,raw_hidden);}
        catch(const MfqResourceExhausted& failure) {
            // The request state must still represent the same pending tokens.
            if(!retryable ||
                    std::string_view(failure.what()).starts_with("CUDA pinned allocation failed:"))throw;
            const auto* limit=dynamic_cast<const CudaMemoryLimitExceeded*>(&failure);
            const auto shortfall=limit?limit->shortfall():0;
            if(geometry.experts) {
                auto& execution=*model.execution;
                const auto before=routed_expert_cache_budget_bytes(execution);
                if(!shortfall || !before)throw;
                graph_->clear();
                const auto context=default_context(device);
                context->reclaim_idle_memory();
                const auto available=context->available_memory();
                reserve_routed_capture_workspace(execution,mfq::engine::prefill_budget_add(available,shortfall));
                if(routed_expert_cache_budget_bytes(execution)>=before || context->available_memory()<=available)throw;
                continue;
            }
            if(!geometry.ple_layers.empty() || !target)throw;
            graph_->clear();
            const auto context=shortfall?default_context(device):nullptr;
            if(context)context->reclaim_idle_memory();
            const auto before=context?context->local_memory_usage():0;
            const auto* placement=std::getenv("TYLOINFER_QSA_CPU_INDEXER");
            const bool move_indexer=shortfall && (!placement || !*placement) &&
                !std::getenv("TYLOINFER_QWEN4_BATCH_HOST_INDEXER_BYTES");
            target=std::max(target,shortfall);
            std::size_t released=0,indexer_bytes=0,physical=0;
            for(;;) {
                std::size_t moved=0;
                if(move_indexer)for(auto it=requests.rbegin();it!=requests.rend() && moved<target;++it)
                    moved+=(*it)->reclaim_indexer_for_admission(target-moved);
                indexer_bytes+=moved;
                for(auto it=requests.rbegin();it!=requests.rend() && moved<target;++it)
                    moved+=(*it)->map_recurrence_to_host(device,std::max(recurrence_bytes,target-moved),true);
                if(!moved)break;
                released+=moved;
                target=mfq::engine::prefill_budget_add(target,target);
                if(!context)break;
                // Moving a tensor may leave the driver page charged to this
                // process. Verify physical progress before recapturing.
                context->reclaim_idle_memory();
                const auto now=context->local_memory_usage();
                physical=before-std::min(before,now);
                if(physical>=shortfall)break;
            }
            if(!released)throw;
            if(std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))
                std::fprintf(stderr,"native_batch_decode_graph_retry moved_vram_bytes=%zu indexer_bytes=%zu "
                    "physical_bytes=%zu shortfall=%zu rows=%zu reason=%s\n",
                    released,indexer_bytes,physical,shortfall,requests.size(),failure.what());
        }
    }
}
std::optional<mfq_tensor_backend::Tensor> Qwen4BatchDecodeGraph::run_once(
        const std::vector<Qwen4BatchRequestState*>& requests,
        const mfq_tensor_backend::Tensor& host_ids,bool greedy,bool& retryable,const std::vector<int64_t>& widths,
        mfq_tensor_backend::Tensor* raw_hidden) {
    namespace tb = mfq_tensor_backend;
    auto& graph = *graph_;
    using Clock=std::chrono::steady_clock;
    const auto started=Clock::now();
    const auto milliseconds=[](auto before,auto after){return std::chrono::duration<double,std::milli>(after-before).count();};
    auto& model = graph.model;
    if (model.execution->qsa_kv_store || !mfq_cuda_graph_capture_supported()) return {};
    const bool verify=!widths.empty();
    const auto batch=static_cast<int64_t>(requests.size());
    const auto axis=verify?1:0;
    TYLOINFER_RUNTIME_CHECK(!verify || (widths.size()==requests.size() && raw_hidden && !greedy),
        "invalid CUDA batch verification layout");
    const auto width=[&](int64_t row) {return verify?widths[row]:int64_t(1);};
    std::vector<int64_t> offsets{0};
    for(int64_t row=0;row<batch;++row) {
        TYLOINFER_RUNTIME_CHECK(width(row)>0 && width(row)<=8 && !requests[row]->state_->verification_tokens,
            "invalid or unresolved CUDA batch verification window");
        offsets.push_back(offsets.back()+width(row));
    }
    const auto host_row=[&](int64_t row) {return host_ids.narrow(axis,offsets[row],width(row));};
    for(auto* request:requests)if(verify && request->position()==0)return {};
    for (auto* request : requests) for (const auto& layer : request->state_->layers)
        if (layer.ple && !layer.ple->graph_supported()) return {};
    const QsaRamDecodeBudgetScope ram_budget(model.execution->layer_placement.primary_device(),true);
    // Whole-layer prefill snapshots the GPU/RAM expert maps. Adaptive decode
    // can exchange those experts, so retire the snapshots before replay just
    // as the single-request window does.
    release_moe_prefill_buffers(model.execution->moe_expert_cache);
    if(model.execution->kv_memory) {
        int64_t required=0;
        for(int64_t row=0;row<batch;++row)required=std::max(required,requests[row]->position()+width(row));
        model.execution->kv_memory->ensure(required);
    }
    graph.ready.record(mfq_get_current_cuda_stream());
    MfqCudaStreamGuard stream_guard(graph.stream);
    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(graph.stream.stream(),graph.ready.get(),0));
    bool reuse = graph.requests == requests && graph.window && graph.greedy == greedy && graph.widths==widths;
    if (reuse) for (int64_t row = 0; row < batch; ++row) {
        reuse = reuse && requests[row]->position() + width(row) <= graph.capacities[row];
    }
    if (!reuse) {
        graph.clear();
        if(!model.config.experts)for(auto* request:requests)request->release_shared_device_cache(true);
        // Every row of the replacement graph decodes one token. Retire wide
        // eager staging across all rows before the first new arena is needed.
        for(int64_t row=0;row<batch;++row)for(auto& layer:requests[row]->state_->layers)
            if(layer.qsa)layer.qsa->trim_batch_decode_staging(static_cast<int>(width(row)));
        // Prefill and previous widths leave asynchronously freed pool pages
        // charged to WDDM. Retire transfers and trim before preparing a graph.
        finish_routed_expert_exchanges(*model.execution);
        default_context(model.execution->layer_placement.primary_device())->reclaim_idle_memory();
    }
    graph.greedy = greedy;
    if (graph.requests.empty()) {
        graph.requests = requests;
        graph.widths=widths;
        std::size_t growth = 0;
        for (int64_t row=0;row<batch;++row) {
            auto* request=requests[row];
            const auto maximum = model.max_position_embeddings();
            const auto required = request->position() + width(row);
            TYLOINFER_RUNTIME_CHECK(required <= maximum,"Flash-Next batch graph exceeds max_context");
            // Keep the initial capture interval. Large independent batches
            // subsequently grow by 128 tokens instead of doubling every
            // request's KV allocation at once under a tight device budget.
            int64_t capacity = std::min<int64_t>(batch>8 ?
                runtime_options::continuous_batch_kv_initial_capacity() : 256,maximum);
            if(batch>16 && capacity<required) {
                const auto chunks=1+(required-1)/128;
                capacity=chunks>maximum/128?maximum:chunks*128;
            } else while (capacity < required)
                capacity = std::min(maximum,capacity > maximum/2 ? maximum : capacity*2);
            for(const auto& layer:request->state_->layers)
                if(layer.qsa && layer.qsa->graph_capacity())capacity=layer.qsa->graph_capacity();
            graph.capacities.push_back(capacity);
            for (const auto& layer : request->state_->layers) {
                if (layer.qsa) growth += layer.qsa->graph_growth_bytes(capacity);
                if (layer.gdn) growth += layer.gdn->request_cache_growth_bytes();
            }
        }
        const auto reserve=model.config.experts?(std::size_t(64)<<20):0;
        if ((growth || reserve) && model.reserve_kv_workspace)model.reserve_kv_workspace(growth+reserve);
        const auto device=tb::Device(tb::kCUDA,model.execution->layer_placement.primary_device());
        graph.ids = tb::empty(host_ids.sizes(),host_ids.options().device(device));
        graph.ple.resize(batch);
        if(verify)graph.replay_convolutions.resize(batch);
        graph.qsa_pending.resize(batch);
        for (int64_t row = 0; row < batch; ++row) {
            auto& state = *requests[row]->state_;
            graph.positions.push_back(tb::empty({width(row)},graph.ids.options()));
            graph.cache_positions.push_back(tb::empty({width(row)},graph.ids.options()));
            graph.ple[row].resize(state.layers.size());
            graph.qsa_pending[row].resize(state.layers.size());
            for (std::size_t i = 0; i < state.layers.size(); ++i) {
                auto& layer = state.layers[i];
                if (layer.gdn) layer.gdn->initialize_request_cache();
                if (layer.ple) graph.ple[row][i] = layer.ple->graph_stage(width(row),device);
            }
        }
        graph.gdn_addresses.resize(model.blocks.size());
        if(!verify && batch>1 && runtime_options::batch_attention_projections() && runtime_options::gdn_independent_batch())
            for(std::size_t i=0;i<model.blocks.size();++i) {
                std::vector<qwen4_exp::Gdn*> rows;
                for(auto* request:requests)if(request->state_->layers[i].gdn)
                    rows.push_back(request->state_->layers[i].gdn.get());
                if(rows.size()==requests.size())graph.gdn_addresses[i]=qwen4_exp::Gdn::prepare_independent_batch(rows);
            }
        prepare_execution_graph_pool(*model.execution,graph.memory_pool);
    }
    std::vector<int> signature;
    for(int64_t row=0;row<batch;++row)for(std::size_t i=0;i<requests[row]->state_->layers.size();++i) {
        const auto kind=Qwen4BatchDecodeGraph::Impl::pending_kind(requests[row]->state_->layers[i].qsa.get(),width(row));
        graph.qsa_pending[row][i]=kind;signature.push_back(kind);
    }
    graph.select_variant(std::move(signature),static_cast<int>(offsets.back()),verify);
    // Upload directly into the address retained by every graph variant.
    graph.ids.copy_(host_ids);
    // Queue immutable table rows before cache preparation. Every stage owns
    // one bounded CPU packet; upload commits its matching ngram context only
    // after the preceding graph has finished reading the published packet.
    RowPacketPrefetch::Group table_reads;
    for(int64_t row=0;row<batch;++row) {
        const auto token=host_row(row);
        table_reads.start([&,row,token] {
            const auto& layers=requests[row]->state_->layers;
            for(std::size_t i=0;i<layers.size();++i)if(layers[i].ple)
                layers[i].ple->prefetch_graph_tokens(token,graph.ple[row][i]);
        });
    }
    std::vector<std::pair<qwen4_exp::Ple*,qwen4_exp::NgramEmbedding::Context>> ple_contexts;
    using PleContexts=std::vector<std::vector<qwen4_exp::NgramEmbedding::Context>>;
    std::vector<PleContexts> prefix_contexts(verify?batch:0);
    for (int64_t row = 0; row < batch; ++row) {
        auto& state = *requests[row]->state_;
        mfq_qwen4_exp::fill_decode_positions(graph.cache_positions[row],graph.positions[row],
                                            state.cache_position,model.decode_position_delta);
        const auto qsa_started=Clock::now();
        for(std::size_t i=0;i<state.layers.size();++i)if(auto* attention=state.layers[i].qsa.get()) {
            try {attention->prepare_graph_tokens(state.positions,graph.capacities[row],graph.qsa_pending[row][i],width(row),true);}
            catch(const std::out_of_range& error) {
                throw std::out_of_range("QSA batch graph preparation at row "+std::to_string(row)+
                    " layer "+std::to_string(i)+": "+error.what());
            }
        }
        graph.timings.qsa_prepare_ms+=milliseconds(qsa_started,Clock::now());
        const auto ple_started=Clock::now();
        table_reads.wait(std::size_t(row));
        if(verify)prefix_contexts[row].resize(state.layers.size());
        for (std::size_t i = 0; i < state.layers.size(); ++i) if (state.layers[i].ple) {
            if(!graph.window->valid())ple_contexts.emplace_back(state.layers[i].ple.get(),state.layers[i].ple->session_context());
            const auto ids=host_row(row);
            if(verify)for(int64_t k=1;k<width(row);++k)
                prefix_contexts[row][i].push_back(state.layers[i].ple->graph_prefix_context(ids,k));
            state.layers[i].ple->upload_graph(ids,*graph.ple[row][i]);
        }
        graph.timings.ple_prepare_ms+=milliseconds(ple_started,Clock::now());
    }
    const auto prepared=Clock::now();graph.timings.prepare_ms+=milliseconds(started,prepared);
    const bool capture_graph=!graph.window->valid();
    if(capture_graph) {
        struct Saved { tb::Tensor* tensor; tb::Tensor host_alias, snapshot; };
        std::vector<Saved> recurrent;
        std::size_t snapshot_bytes=0,mapped_bytes=0;
        const auto collect=[&](tb::Tensor* tensor,tb::Tensor host_alias={}) {
            if(tensor->defined()) {
                (host_alias.defined()?mapped_bytes:snapshot_bytes)+=tensor->nbytes();
                recurrent.push_back({tensor,std::move(host_alias),{}});
            }
        };
        if(verify)graph.checkpoints.resize(batch);
        for(int64_t row=0;row<batch;++row) {
            auto& layers=requests[row]->state_->layers;
            if(verify)graph.checkpoints[row].resize(layers.size());
            const auto drafts=width(row)-1;
            for(std::size_t i=0;i<layers.size();++i) {
                auto& layer=layers[i];
                auto* checkpoint=drafts?&graph.checkpoints[row][i]:nullptr;
                if(layer.gdn) {
                    const auto state=layer.gdn->graph_state();
                    if(checkpoint)checkpoint->gdn_replay=layer.gdn->replay_checkpoint_supported();
                    if(!checkpoint || !checkpoint->gdn_replay) {
                        for(auto* tensor:state)collect(tensor,layer.gdn->graph_host_alias(tensor));
                        if(checkpoint)for(int64_t k=0;k<drafts;++k) {
                            checkpoint->gdn_conv.push_back(tb::empty_like(*state[0]));
                            checkpoint->gdn_state.push_back(tb::empty_like(*state[1]));
                        }
                    }
                }
                if(layer.ple) {
                    for(auto* tensor:layer.ple->graph_state())collect(tensor);
                    if(checkpoint)for(int64_t k=0;k<drafts;++k)
                        checkpoint->ple_conv.push_back(tb::empty_like(layer.ple->conv_state()));
                }
            }
        }
        // These copies are used during capture warmup only. Keep them in host
        // memory when a full duplicate would consume the available VRAM.
        const auto context=default_context(model.execution->layer_placement.primary_device());
        // Mapped recurrence signals that this batch already needed device
        // space for live state. Preserve that space for capture scratch.
        const bool host_snapshots=mapped_bytes || batch>8 || snapshot_bytes>=context->available_memory();
        if(mapped_bytes)TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph.stream.stream()));
        for(auto& saved:recurrent) {
            if(saved.host_alias.defined()) {
                // Save mapped recurrence in RAM without sending it through PCIe.
                auto options=tb::TensorOptions().device(tb::kCPU).dtype(saved.host_alias.scalar_type());
                saved.snapshot=tb::empty(saved.host_alias.sizes(),options);
                std::memcpy(saved.snapshot.data_ptr(),saved.host_alias.data_ptr(),saved.snapshot.nbytes());
            } else saved.snapshot=host_snapshots?saved.tensor->to(tb::kCPU):saved.tensor->clone();
        }
        if(mapped_bytes && std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))
            std::fprintf(stderr,"native_qwen4_batch_capture_host_snapshot bytes=%zu\n",mapped_bytes);
        const auto restore_recurrent=[&] {
            for(const auto& saved:recurrent) {
                if(saved.host_alias.defined())
                    std::memcpy(saved.host_alias.data_ptr(),saved.snapshot.data_ptr(),saved.snapshot.nbytes());
                else saved.tensor->copy_(saved.snapshot);
            }
        };
        DecodeGraphBranchScope branch_scope(model.execution->decode_graph_serial_branches);
        int64_t capture_layer=-1;
        const char* capture_phase="embedding";
        const bool trace=verify && std::getenv("TYLOINFER_TRACE_MOE_SERVE");
        const auto check_capture=[&] {
            if(!trace)return;
            cudaStreamCaptureStatus status{};
            TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(graph.stream.stream(),&status));
            if(status==cudaStreamCaptureStatusInvalidated)
                throw std::runtime_error("CUDA batch MTP capture invalidated at "+std::to_string(capture_layer)+" "+capture_phase);
        };
        retryable=false;
        try { graph.window->capture([&] {
            auto& execution = *model.execution;
            graph.window->mark(-1,"window_begin");
            capture_layer=-1;capture_phase="embedding";
            const auto packed_batch=verify?1:batch;
            const auto packed_tokens=verify?offsets.back():1;
            auto hidden = model.adapter_prepare_hidden(model.embed_tokens(graph.ids),packed_batch,packed_tokens);
            std::vector<tb::Tensor> prior_mix;
            const auto chain_setting=mfq::cuda::runtime_options::gr_chain_ffn_post();
            const bool chain_post=(!chain_setting || *chain_setting) && offsets.back()<=8;
            for (std::size_t i = 0; i < model.blocks.size(); ++i) {
                capture_layer=static_cast<int64_t>(i);capture_phase="PLE";
                graph.window->mark(static_cast<int>(i),"block_begin");
                auto& block = static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]);
                graph.window->set_profile_layer_kind(static_cast<int>(i),block.gdn?"gdn":"qsa");
                const auto marker=[&](const char* role)->GatedResidualMarker {
                    if(!runtime_options::mhc_timings())return {};
                    return [&,role](const char* phase) {
                        graph.window->mark(static_cast<int>(i),(std::string(role)+phase).c_str());
                    };
                };
                if (block.ple) {
                    graph.window->mark(static_cast<int>(i),"ple_begin");
                    std::vector<tb::Tensor> positional;
                    for (int64_t row = 0; row < batch; ++row) {
                        capture_phase="PLE row decode";
                        auto x = hidden.narrow(axis,offsets[row],width(row));
                        graph.ple[row][i]->decode();
                        check_capture();capture_phase="PLE convolution and projection";
                        auto rows = graph.ple[row][i]->output().reshape({1,width(row),model.hidden_size()});
                        auto ids=graph.ids.narrow(axis,offsets[row],width(row));
                        auto& ple=*requests[row]->state_->layers[i].ple;
                        auto output=width(row)>1
                            ?ple.forward_graph(execution,x,ids,rows,1,graph.checkpoints[row][i].ple_conv)
                            :ple.forward_chunk(execution,x,ids,true,{},rows);
                        check_capture();capture_phase="PLE residual";
                        positional.push_back(x+output);
                        check_capture();
                    }
                    capture_phase="PLE output";
                    hidden = join_request_outputs(positional,axis);
                    graph.window->mark(static_cast<int>(i),"ple_end");
                }
                check_capture();capture_phase="attention projections";
                auto mix = prior_mix.empty()?block.attention_gr.pre(execution,hidden,marker("attention_mhc_")):
                    block.attention_gr.pre_after(execution,hidden,prior_mix,marker("attention_mhc_"));
                graph.window->mark(static_cast<int>(i),"attention_core_begin");
                const bool project_batch=verify || (batch>1 && runtime_options::batch_attention_projections());
                std::vector<tb::Tensor> projections;
                if(project_batch && block.gdn)projections=block.gdn->project_inputs(execution,mix[0]);
                else if(verify && block.qsa)projections=block.qsa->project_inputs(execution,mix[0]);
                auto qsa_inputs=project_batch && block.qsa && !verify?
                    block.qsa->project_requests(execution,mix[0],graph.positions):std::vector<qwen4_exp::Qsa::Projection>{};
                if(project_batch)graph.window->mark(static_cast<int>(i),block.qsa?"qsa_projection_end":"gdn_projection_end");
                check_capture();capture_phase="attention";
                std::vector<tb::Tensor> attended;
                Tensor attention;
                bool attention_projected=false;
                if(project_batch && block.gdn && graph.gdn_addresses[i].defined()) {
                    attention=requests[0]->state_->layers[i].gdn->forward_independent_batch(execution,mix[0],projections,graph.gdn_addresses[i],true);
                    attention_projected=attention.defined();
                }
                if(!attention.defined())for (int64_t row = 0; row < batch; ++row) {
                    const DecodeWindow::ProfileTileScope request_profile(*graph.window,int(row));
                    auto& layer = requests[row]->state_->layers[i];
                    auto branch = mix[0].narrow(axis,offsets[row],width(row));
                    std::vector<tb::Tensor> projected;
                    for(const auto& value:projections)projected.push_back(value.narrow(axis,offsets[row],width(row)));
                    if(layer.gdn && width(row)>1) {
                        auto& checkpoint=graph.checkpoints[row][i];
                        attended.push_back(layer.gdn->forward_graph(execution,branch,1,checkpoint.gdn_conv,
                            checkpoint.gdn_state,checkpoint.gdn_replay?&checkpoint.gdn_inputs:nullptr,
                            std::move(projected),false));
                    } else {
                        attended.push_back(layer.gdn?layer.gdn->forward_chunk(execution,branch,true,std::move(projected),!project_batch)
                            :layer.qsa->forward_graph(execution,branch,graph.positions[row],graph.cache_positions[row],
                                std::move(projected),!project_batch,false,false,qsa_inputs.empty()?nullptr:&qsa_inputs[row]));
                    }
                }
                if(!attention.defined())attention=join_request_outputs(attended,axis);
                if(project_batch && !attention_projected)attention=block.gdn ? block.gdn->project_output(execution,attention)
                    : block.qsa->project_output(execution,attention);
                check_capture();capture_phase="FFN";
                graph.window->mark(static_cast<int>(i),"attention_core_end");
                auto ffn_mix = block.ffn_gr.pre_after(execution,std::move(attention),mix,marker("ffn_mhc_"));
                // Finish attention branches' tensor fences before the routed
                // FFN can end this capture segment and start a new one.
                prior_mix.clear();mix.clear();projections.clear();qsa_inputs.clear();attended.clear();
                graph.window->mark(static_cast<int>(i),"ffn_pre_end");
                auto branch=block.batch_ffn(execution,ffn_mix[0]);
                bool defer=false;
                if(chain_post && i+1<model.blocks.size()) {
                    auto& next=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i+1]);
                    defer=!next.ple && next.attention_gr.can_fuse(execution,ffn_mix[1]);
                }
                if(defer){hidden=std::move(branch);prior_mix=std::move(ffn_mix);}
                else {hidden=block.ffn_gr.post(branch,ffn_mix);prior_mix.clear();}
                graph.window->mark(static_cast<int>(i),"block_end");
                check_capture();
            }
            capture_phase="output head";
            if(verify)graph.raw_hidden=hidden;
            auto finalized=model.finalize_hidden(std::move(hidden),packed_batch,packed_tokens);
            graph.logits = greedy ? model.adapter_next_token(model.lm_head,model.last_hidden(std::move(finalized)))
                .reshape({batch,1}) : model.logits_from_hidden(std::move(finalized)).reshape({offsets.back(),-1});
            graph.window->mark(-1,"window_end");
            check_capture();
        },restore_recurrent,
        [&] { graph.logits = {};graph.raw_hidden={}; }); }
        catch(const std::out_of_range& error) {
            throw std::out_of_range(std::string("Batch decode graph capture: ")+error.what());
        }
        catch(const std::exception& error) {
            const std::string message=error.what();
            if(!dynamic_cast<const MfqResourceExhausted*>(&error) &&
               !message.starts_with("CUDA process VRAM limit exceeded:") &&
               !message.starts_with("CUDA VRAM limit exceeded:") &&
               !message.starts_with("CUDA pinned allocation failed:")) {
                if(!verify)throw;
                throw std::runtime_error("Batch MTP graph capture at layer "+std::to_string(capture_layer)+
                    " "+capture_phase+": "+message);
            }
            // Finish restoring state before a retry can move it on another stream.
            graph.clear();
            (void)cudaGetLastError();
            restore_recurrent();
            for(auto& [layer,context]:ple_contexts)layer->restore_graph_context(std::move(context));
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph.stream.stream()));
            retryable=true;
            throw;
        }
        if(verify) {
            graph.replay_storage.resize(batch);
            for(int64_t row=0;row<batch;++row) {
                const auto& checkpoints=graph.checkpoints[row];
                if(std::none_of(checkpoints.begin(),checkpoints.end(),[](const auto& checkpoint){return checkpoint.gdn_replay;}))continue;
                const auto& layers=requests[row]->state_->layers;
                graph.replay_storage[row]=std::make_shared<Qwen4BatchDecodeGraph::Impl::ReplayStorage>(
                    checkpoints,graph.replay_convolutions[row],[&](std::size_t i) -> qwen4_exp::Gdn& {return *layers[i].gdn;});
            }
        }
        ++graph.captures;
        graph.timings.capture_ms+=milliseconds(prepared,Clock::now());
    }
    for(auto* request:requests)for(auto& layer:request->state_->layers)
        if(layer.qsa)layer.qsa->activate_graph_cache();
    if(capture_graph && !verify && model.config.experts &&
            (model.execution->gguf_expert_cache || model.execution->tq_expert_cache)) {
        // The new graph has already claimed its activations and transfer
        // staging. Reserve only the sampling work outside ordinary replay.
        // Concurrent rows can already have emitted their first tokens. Keep
        // the existing GGUF cache and admit experts through decode demand.
        const auto growth=model.execution->kv_memory?model.execution->kv_memory->growth_reserve():0;
        const auto logits=mfq::engine::prefill_budget_multiply(std::size_t(batch),
            std::size_t(model.vocab_size())*sizeof(float)*8);
#ifdef _WIN32
        if(physical_device_memory_stats(model.execution->layer_placement.primary_device()))
#endif
            restore_routed_decode_workspace(*model.execution,growth,logits,/*rebalance_gguf_cache=*/batch==1);
    }
    const auto replay_started=Clock::now();
    retryable=false;
    try {graph.window->run();}
    catch(const std::out_of_range& error) {
        throw std::out_of_range(std::string("Batch decode graph replay: ")+error.what());
    }
    const auto replayed=Clock::now();graph.timings.replay_ms+=milliseconds(replay_started,replayed);
    ++graph.replays;
    for (std::size_t row=0;row<requests.size();++row) {
        auto& state = *requests[row]->state_;
        const auto count=width(row);
        std::shared_ptr<Qwen4DecodeGraph::ReplayCommit> replay;
        if(verify && graph.replay_storage[row])replay=std::make_shared<Qwen4DecodeGraph::ReplayCommit>(graph.replay_storage[row]);
        for(std::size_t i=0;i<state.layers.size();++i) {
            auto& layer=state.layers[i];
            if(layer.qsa)layer.qsa->advance_graph(count);
            if(count==1)continue;
            const auto& checkpoint=graph.checkpoints[row][i];
            if(layer.gdn) {
                if(checkpoint.gdn_replay)layer.gdn->bind_graph_replay(checkpoint.gdn_inputs,state.cache_position,replay);
                else layer.gdn->bind_graph_checkpoint(checkpoint.gdn_conv,checkpoint.gdn_state,state.cache_position+1);
            }
            if(layer.ple)layer.ple->bind_graph_checkpoint(checkpoint.ple_conv,
                std::move(prefix_contexts[row][i]),state.cache_position+1);
        }
        state.positions=append_decode_positions(state.positions,graph.cache_positions[row],
            state.position_storage,model.max_position_embeddings());
        state.cache_position+=count;
        state.verification_tokens=verify?count:0;
    }
    graph.timings.advance_ms+=milliseconds(replayed,Clock::now());
    if(raw_hidden)*raw_hidden=graph.raw_hidden;
    return graph.logits;
}

std::vector<mfq_tensor_backend::Tensor> qwen4_batch_prefill(Qwen4CausalLm& model,
        const std::vector<Qwen4BatchRequestState*>& requests,
        const std::vector<mfq_tensor_backend::Tensor>& chunks,
        std::vector<mfq_tensor_backend::Tensor>* raw_hidden) {
    namespace tb = mfq_tensor_backend;
    TYLOINFER_RUNTIME_CHECK(!requests.empty() && requests.size()==chunks.size(),"invalid packed prefill rows");
    qwen4_exp::Qsa::PrefillWorkspaceScope attention_workspace;
    auto& execution = *model.execution;
    model.adapter_begin_forward(false);
    std::vector<int64_t> offsets{0};
    std::vector<tb::Tensor> current,full;
    for (std::size_t row=0;row<requests.size();++row) {
        auto& state=*requests[row]->state_;
        const auto& ids=chunks[row];
        TYLOINFER_RUNTIME_CHECK(ids.dim()==2 && ids.size(0)==1 && ids.size(1)>0 &&
            !state.verification_tokens && ids.size(1)<=model.max_position_embeddings()-state.cache_position,
            "invalid packed prefill chunk");
        // Each request retains its full positions after this temporary
        // prefill workspace is returned to the expert cache.
        const ResidentAllocationScope persistent_positions;
        auto positions=tb::arange(ids.size(1),ids.options())+state.cache_position;
        current.push_back(positions);
        full.push_back(append_decode_positions(state.positions,positions,state.position_storage,
            model.max_position_embeddings()));
        // Keep the old logical boundary until the computation succeeds, while
        // releasing the old prefix allocation before reserving layer scratch.
        if(state.positions.defined())state.positions=full.back().narrow(-1,0,state.positions.size(-1));
        offsets.push_back(offsets.back()+ids.size(1));
    }
    auto ids=tb::cat(chunks,1);
    auto host_ids=ids.to(tb::kCPU);
    const auto tokens=offsets.back();
    // Batch operations already reserve their working set before opening a
    // loan. A second generic reserve must not retire its borrowed storage.
    if(tokens>1024 && !default_context(mfq_current_cuda_device())->workspace_loan_active())
        reserve_routed_workspace(*model.execution,runtime_options::workspace_reserve_bytes());
    auto hidden=model.adapter_prepare_hidden(model.embed_tokens(ids),1,tokens);
    for (std::size_t i=0;i<model.blocks.size();++i) {
        NintPrefillDecodeCacheScope decode_reuse(true);
        auto& block=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]);
        if (block.ple) {
            std::vector<tb::Tensor> positional;
            for (std::size_t row=0;row<requests.size();++row) {
                auto& state=*requests[row]->state_;
                auto x=hidden.narrow(1,offsets[row],chunks[row].size(1));
                positional.push_back(x+state.layers[i].ple->forward(execution,x,chunks[row],true,
                    0,host_ids.narrow(1,offsets[row],chunks[row].size(1)),state.cache_position));
            }
            hidden=join_request_outputs(positional,1);
        }
        auto mix=block.attention_gr.pre(execution,hidden);
        const auto projections=block.gdn?block.gdn->project_inputs(execution,mix[0])
                                        :block.qsa->project_inputs(execution,mix[0]);
        std::vector<tb::Tensor> attended;
        for (std::size_t row=0;row<requests.size();++row) {
            auto& state=*requests[row]->state_;
            auto& layer=state.layers[i];
            auto branch=mix[0].narrow(1,offsets[row],chunks[row].size(1));
            std::vector<tb::Tensor> projected;
            for (const auto& value:projections)
                projected.push_back(value.narrow(1,offsets[row],chunks[row].size(1)));
            attended.push_back(layer.gdn?layer.gdn->forward_chunk(execution,branch,true,std::move(projected),false)
                :layer.qsa->forward(execution,branch,current[row],full[row],true,nullptr,std::move(projected),false));
        }
        auto attention=join_request_outputs(attended,1);
        attention=block.gdn?block.gdn->project_output(execution,attention):block.qsa->project_output(execution,attention);
        auto ffn_mix=block.ffn_gr.pre_after(execution,std::move(attention),mix);
        // All prompt tokens share each expert transfer and compressed decode.
        hidden=block.ffn_gr.post(block.ffn(execution,ffn_mix[0]),ffn_mix);
    }
    const auto raw=raw_hidden?hidden:tb::Tensor{};
    auto finalized=model.finalize_hidden(std::move(hidden),1,tokens);
    std::vector<tb::Tensor> result;
    if(raw_hidden)raw_hidden->clear();
    for (std::size_t row=0;row<requests.size();++row) {
        auto& state=*requests[row]->state_;
        const auto count=chunks[row].size(1);
        if(raw_hidden)raw_hidden->push_back(raw.narrow(1,offsets[row],count));
        result.push_back(finalized.narrow(1,offsets[row+1]-1,1));
        state.positions=std::move(full[row]);
        state.cache_position+=count;
    }
    return result;
}

mfq_tensor_backend::Tensor qwen4_batch_decode(Qwen4CausalLm& model,
        const std::vector<Qwen4BatchRequestState*>& requests, mfq_tensor_backend::Tensor ids,
        const mfq_tensor_backend::Tensor& host_ids) {
    namespace tb = mfq_tensor_backend;
    const auto batch = static_cast<int64_t>(requests.size());
    TYLOINFER_RUNTIME_CHECK(batch > 0 && ids.dim() == 2 && ids.size(0) == batch && ids.size(1) == 1,
                      "invalid Flash-Next batch input");
    std::size_t growth = 0;
    std::vector<tb::Tensor> current, full;
    for (auto* request : requests) {
        auto& state = *request->state_;
        TYLOINFER_RUNTIME_CHECK(state.cache_position < model.max_position_embeddings(), "batch request exceeds max_context");
        growth += state.growth_bytes(1);
        auto position = tb::full({1},state.cache_position,ids.options());
        current.push_back(position);
        full.push_back(append_decode_positions(state.positions,position,state.position_storage,model.max_position_embeddings()));
    }
    if (growth && model.reserve_kv_workspace)
        model.reserve_kv_workspace(growth + (model.config.experts?(std::size_t(64)<<20):0));
    auto& execution = *model.execution;
    auto hidden = model.adapter_prepare_hidden(model.embed_tokens(ids),batch,1);
    for (std::size_t i = 0; i < model.blocks.size(); ++i) {
        auto& block = static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]);
        if (block.ple) {
            std::vector<tb::Tensor> positional;
            for (int64_t row = 0; row < batch; ++row) {
                auto& layer = requests[row]->state_->layers[i];
                auto x = hidden.narrow(0,row,1);
                positional.push_back(x + layer.ple->forward(execution,x,ids.narrow(0,row,1),true,0,
                                                           host_ids.narrow(0,row,1)));
            }
            hidden = join_request_outputs(positional,0);
        }
        auto mix = block.attention_gr.pre(execution,hidden);
        const bool project_batch=batch>1 && block.gdn && runtime_options::batch_attention_projections();
        const auto projections=project_batch ? block.gdn->project_inputs(execution,mix[0]) : std::vector<tb::Tensor>{};
        std::vector<tb::Tensor> attended;
        for (int64_t row = 0; row < batch; ++row) {
            auto& layer = requests[row]->state_->layers[i];
            auto branch = mix[0].narrow(0,row,1);
            std::vector<tb::Tensor> projected;
            for(const auto& value:projections)projected.push_back(value.narrow(0,row,1));
            attended.push_back(layer.gdn ? layer.gdn->forward_chunk(execution,branch,true,std::move(projected),!project_batch)
                : layer.qsa->forward(execution,branch,current[row],full[row],true));
        }
        auto attention=join_request_outputs(attended,0);
        if(project_batch)attention=block.gdn->project_output(execution,attention);
        auto ffn_mix = block.ffn_gr.pre_after(execution,std::move(attention),mix);
        // Tiles share the expert transfer cache across all eligible rows.
        hidden = block.ffn_gr.post(block.batch_ffn(execution,ffn_mix[0]),ffn_mix);
    }
    auto logits = model.logits_from_hidden(model.finalize_hidden(std::move(hidden),batch,1)).reshape({batch,-1});
    for (int64_t row = 0; row < batch; ++row) {
        requests[row]->state_->positions = std::move(full[row]);
        ++requests[row]->state_->cache_position;
    }
    return logits;
}

void Qwen4Model::adapter_validate_load_options() const {
    weight_loader::validate_load_options(*execution);
}

void Qwen4Model::adapter_load_final_state(const mfq::ModelSource &source,
                                          mfq_tensor_backend::Tensor &output_norm) {
    qwen4_exp::BlockLoader loader{{*execution, source, "qwen4_exp"}};
    final_mixer = loader.final_mixer(
        mfq::models::qwen4_exp::load_residual(loader, config, "model.mhc.pre", false));
    output_norm = mfq_tensor_backend::Tensor();
}

std::unique_ptr<Block> Qwen4Model::adapter_load_block(const mfq::ModelSource &source, int layer,
                                                      int, const std::string &) {
    auto block=std::make_unique<qwen4_exp::Qwen4Block>(*execution, source, config, layer);
    loaded_blocks.push_back(block.get());
    return block;
}

void Qwen4Model::adapter_reset(int64_t new_batch) {
    speculative_prefix_retention=false;
    if(execution)finish_routed_expert_exchanges(*execution);
    const bool retain=reuse_decode_graphs && new_batch==1 && graph_registry && graph_registry->reusable();
    if(graph_registry)for(auto* block:graph_registry->blocks)block->retain_graph_storage=retain;
    if(!retain)graph_registry.reset();
    else for(auto& [key,cell]:graph_registry->cells)cell->used_kinds=0;
    positions = mfq_tensor_backend::Tensor();
    batch = new_batch;
}
void Qwen4Model::adapter_begin_forward(bool capture_raw) {
    if (hidden_capture) hidden_capture->begin(capture_raw);
    for(auto* block:loaded_blocks)block->retained_prefix=false;
    if(graph_registry) {
        if(reuse_decode_graphs && graph_registry->reusable()) {
            for(auto* attention:graph_registry->qsa)attention->leave_graph();
        } else {graph_registry->restore_legacy();graph_registry.reset();}
    }
}

mfq_tensor_backend::Tensor
Qwen4Model::adapter_block_positions(const mfq_tensor_backend::Tensor &full_positions,
                                    const mfq_tensor_backend::Tensor &, int device) const {
    return tensor_to_cuda_device(execution->model_parallel_collectives, full_positions, device);
}

void Qwen4Model::adapter_finish_forward(const mfq_tensor_backend::Tensor &full_positions,
                                        int64_t new_batch, int64_t tokens) {
    positions = full_positions;
    batch = new_batch;
    // Short eager verification retains the same prefix states as its graph.
    // Accepted drafts can be kept without rerunning them as scalar projections.
    speculative_prefix_retention=tokens<=8 && !loaded_blocks.empty() &&
        std::all_of(loaded_blocks.begin(),loaded_blocks.end(),[](const auto* block) {
            return block->retained_prefix;
        });
}

std::size_t qwen4_prefill_state_growth_bytes(const Qwen4CausalLm& model,int64_t tokens) {
    std::size_t growth=0;
    for(const auto* block:model.loaded_blocks) {
        if(block->gdn)growth=mfq::engine::prefill_budget_add(growth,block->gdn->growth_bytes());
        if(block->qsa)growth=mfq::engine::prefill_budget_add(growth,block->qsa->growth_bytes(tokens));
    }
    return growth;
}

mfq::engine::PrefillWorkspace qwen4_dense_prefill_workspace(const Qwen4CausalLm& model,
        std::size_t state_growth,bool packed) {
    std::size_t attention=0;
    for(const auto* block:model.loaded_blocks)if(block->qsa)
        attention=std::max(attention,block->qsa->prefill_scratch_row_bytes());
    // Packed requests also retain assembled residuals from earlier rows.
    const auto residual=std::size_t(model.hidden_size())*model.hc_mult()*sizeof(float)*(packed?11:8);
    const auto ffn=3*std::size_t(model.config.moe_width)*sizeof(float);
    return {state_growth,mfq::engine::prefill_budget_add(mfq::engine::prefill_budget_add(residual,ffn),attention)};
}

std::size_t qwen4_append_attention_workspace_bytes(const Qwen4CausalLm& model,int64_t tokens) {
    std::size_t bytes=0;
    for(const auto* block:model.loaded_blocks)if(block->qsa)
        bytes=std::max(bytes,block->qsa->append_attention_workspace_bytes(tokens));
    return bytes;
}

std::size_t qwen4_append_packed_history_bytes(const Qwen4CausalLm& model,int64_t tokens) {
    std::size_t bytes=0;
    for(const auto* block:model.loaded_blocks)if(block->qsa)
        bytes=std::max(bytes,block->qsa->append_packed_history_bytes(tokens));
    return bytes;
}

// Keep each compressed expert image resident while advancing all prompt chunks
// through that layer. Attention and PLE still advance in causal token order.
bool qwen4_layer_major_prefill_eligible(const Qwen4CausalLm& model, int64_t tokens) {
    const auto enabled=mfq::cuda::runtime_options::moe_prefill_layer_major();
    const auto whole=mfq::cuda::runtime_options::moe_prefill_layer();
    const bool row_banks=model.execution && bool(model.execution->tq_expert_cache || model.execution->gguf_expert_cache);
    // TQ and GGUF group ordinary bank rows by expert. Their layer-major path
    // does not require registration in the legacy MFE transfer pipeline.
    // Keep explicit user overrides and the existing legacy selection policy.
    if((enabled && *enabled==0) || (whole && *whole==0) || (!row_banks && !whole) || tokens<=128 ||
       tokens>16384 || model.speculative_start>=0 || !model.execution ||
       (!row_banks && (!model.execution->config.moe_pipeline || !model.execution->moe_expert_cache)) ||
       model.execution->dense_cpu_layer_count || model.execution->tensor_parallel.enabled() ||
       model.execution->expert_parallel.enabled())return false;
    const auto primary=model.execution->layer_placement.primary_device();
    for(const auto& block:model.blocks)
        if(block->cpu_offloaded || block->cuda_device!=primary)return false;
    return true;
}

int64_t qwen4_dense_layer_major_prefill_rows(const Qwen4CausalLm& model,int64_t tokens,int64_t limit) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    const auto* enabled=std::getenv("TYLOINFER_DENSE_LAYER_MAJOR_PREFILL");
    const bool forced=enabled && enabled[0]=='1';
    if((enabled && enabled[0]=='0') || tokens<=limit || limit<=8 || model.config.experts ||
        model.speculative_start>=0 || model.layer_weights || model.execution->dense_cpu_layer_count ||
        model_parallel_enabled(*model.execution) || model.execution->layer_placement.enabled() ||
        (model.hidden_capture && model.hidden_capture->active))return 0;
    // The initial cold tile already has its K/V available. There is no
    // history staging to share across calls until two later tiles need it.
    if(!forced && !model.cache_pos && tokens<=limit*2)return 0;
    std::size_t ram_qsa_layers=0, staging=0;
    for(const auto* block:model.loaded_blocks)if(block->qsa && block->qsa->adaptive_ram_memory()) {
        ++ram_qsa_layers;
        staging=std::max(staging,block->qsa->prefill_attention_workspace_bytes(tokens));
    }
    if(!ram_qsa_layers)return 0;
    namespace engine=mfq::engine;
    const auto context=default_context(model.execution->layer_placement.primary_device());
    const auto device_available=engine::prefill_budget_add(context->available_memory(),context->workspace_available());
    const auto workspace=qwen4_dense_prefill_workspace(model,qwen4_prefill_state_growth_bytes(model,tokens));
    if(workspace.bytes(std::min<int64_t>(limit,256),staging)>device_available) {
        if(!forced)return 0;
        staging=0;
    }
    const auto rows=engine::select_prefill_chunk(tokens,limit,device_available,[&](int64_t count) {
        return workspace.bytes(count,staging);
    },false);
    if(rows<=8)return 0;
    const auto row=std::size_t(model.hidden_size())*model.hc_mult()*sizeof(float);
    const auto activation_traffic=row*2*model.blocks.size();
    const auto padded=model.config.width<=256?256:model.config.width<=512?512:1024;
    // Q8 keys and values are staged together. Count each RAM-backed layer's
    // payload against the input/output activation transfers of all layers.
    const auto canonical_row=std::size_t(padded)*model.config.kv_heads*2;
    const auto history_traffic=canonical_row*ram_qsa_layers;
    const auto encoded_layer=engine::prefill_budget_multiply(model.cache_pos+tokens,canonical_row);
    // At the normal tile width, losing the encoded staging buffer sends each
    // query group through the host gather path. A smaller tile can retain the
    // layer once and reuse the staged blocks across those groups.
    const bool grouped_recall=staging && workspace.bytes(limit,encoded_layer)>device_available;
    if(!forced && !grouped_recall && !engine::prefill_layer_major_saves_transfers(
            tokens,model.cache_pos,rows,history_traffic,activation_traffic))return 0;
    const auto required=qwen4_fixed_prefill_ram_reservation(model,
        std::max(model.cache_pos+tokens,model.decode_reserve_until),rows);
    const auto available=mfq::sample_host_memory().available_host_bytes();
    if(!row || available<required || std::size_t(tokens+rows)>(available-required)/row)return 0;
    if(const auto trace=runtime_options::prefill_layer_major_trace();trace && *trace)
        std::cerr<<"runtime_dense_prefill_traversal mode=layer_major tokens="<<tokens
            <<" compute_chunk="<<rows<<" staging_bytes="<<staging
            <<" device_available="<<device_available<<" host_available="<<available
            <<" grouped_recall="<<grouped_recall<<'\n';
    return rows;
#else
    (void)model;(void)tokens;(void)limit;return 0;
#endif
}

mfq_tensor_backend::Tensor qwen4_layer_major_prefill_hidden(
        Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,int64_t compute_chunk,
        bool last_only,const std::function<bool()>& cancelled,mfq_tensor_backend::Tensor* raw_hidden,
        bool raw_hidden_host) {
    return qwen4_layer_major_prefill_hidden(model,std::move(ids),compute_chunk,last_only,
        cancelled,raw_hidden,raw_hidden_host,{});
}

mfq_tensor_backend::Tensor qwen4_layer_major_prefill_hidden(
        Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,int64_t compute_chunk,
        bool last_only,const std::function<bool()>& cancelled,mfq_tensor_backend::Tensor* raw_hidden,
        bool raw_hidden_host,const std::function<void()>& computation_finished,bool host_activations) {
    namespace tb=mfq_tensor_backend;
    using Tensor=tb::Tensor;
    TYLOINFER_RUNTIME_CHECK(ids.dim()==2 && ids.size(0)==1 &&
        (host_activations ? last_only && !raw_hidden :
            qwen4_layer_major_prefill_eligible(model,ids.size(1))),
        "layer-major prefill requires a supported single-device Qwen4 request");
    TYLOINFER_RUNTIME_CHECK(compute_chunk>8,"layer-major compute chunk must exceed decode width");
    auto scope=model.execution_scope();
    ids=model.device_ids(std::move(ids));
    const auto tokens=ids.size(1);
    const auto cache_start=model.cache_pos;
    auto& profiler=model.execution->profiler;
    const auto* profile_once=std::getenv("TYLOINFER_PROFILE_PREFILL_ONCE");
    static std::atomic<uint64_t> profile_requests{0};
    const bool profile_enabled=profile_once && profile_once[0]=='1';
    const auto* profile_skip=std::getenv("TYLOINFER_PROFILE_PREFILL_SKIP");
    const auto skipped=profile_skip?std::max<long long>(0,std::atoll(profile_skip)):0;
    const auto profile_request=profile_enabled?profile_requests.fetch_add(1):0;
    const bool collect=profile_enabled && profile_request==uint64_t(skipped);
    if(profile_enabled)std::cerr<<"prefill_profile_request index="<<profile_request
        <<" tokens="<<tokens<<" collect="<<int(collect)<<'\n';
    const bool previous_profile=profiler.enabled;
    if(collect){profiler.reset();profiler.enabled=true;}
    struct ProfileRestore {
        CudaProfiler& profiler;bool enabled,previous,collect;
        ~ProfileRestore() {
            if(collect && std::uncaught_exceptions()==0) {
                try {profiler.report("prefill_regression_once");profiler.reset();}
                catch(const std::exception& error){std::cerr<<"prefill_profile_error "<<error.what()<<'\n';}
            }
            profiler.enabled=previous;
        }
    } restore{profiler,profiler.enabled,previous_profile,collect};
    compute_chunk=std::min(compute_chunk,tokens);
    // Host hidden exports, PLE packets and replacement descriptors overlap
    // live KV. Reserve their actual geometry before reclaiming cold blocks.
    const auto hidden_host=raw_hidden && raw_hidden_host ?
        std::size_t(tokens)*model.config.hidden*model.config.streams*sizeof(float):0;
    const auto ple_host=model.config.ple_layers.empty()?0:
        2*std::size_t(compute_chunk)*model.config.hidden*sizeof(float);
    const auto blocks=std::size_t(cache_start+tokens+3)/4;
    const auto descriptors=2*blocks*(sizeof(tyloinfer::qsa::RamBlockView)+sizeof(std::int32_t));
    const auto codec=2*std::size_t(compute_chunk)*model.config.kv_heads*model.config.width*2*sizeof(float);
    const auto host_workspace=mfq::engine::prefill_budget_add(
        std::max(hidden_host,ple_host),mfq::engine::prefill_budget_add(descriptors,codec));
    std::vector<std::shared_ptr<void>> host_loans;
    std::unordered_set<const void*> reserved_host_pools;
    for(auto* block:model.loaded_blocks)if(block->qsa)
        if(const auto memory=block->qsa->adaptive_ram_memory();
                memory && reserved_host_pools.insert(memory->pool).second)
            host_loans.push_back(block->qsa->hold_adaptive_workspace(host_workspace));
    RoutedPrefillWorkspaceLoan workspace_loan(*model.execution);
    bool borrowed_workspace=false;
    if(!host_activations) {
        // Persistent decode graphs and prefill now coexist. Bound the compute
        // tile by actual remaining workspace instead of discarding those graphs
        // or allocating an 8K projection tile on top of their live buffers.
        const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
        auto reserve=std::size_t(1)<<30;
        std::size_t attention_workspace=0;
        for(const auto* block:model.loaded_blocks)if(block->qsa)
            attention_workspace=std::max(attention_workspace,block->qsa->prefill_attention_workspace_bytes(tokens));
        auto row_bytes=std::size_t(model.config.hidden)*model.config.streams*sizeof(float)*8;
        if(model.execution->gguf_expert_cache) {
            // Gate/Up can use different Q8 layouts. Budget both whole-layer
            // routed input packets even when a matching format shares one.
            namespace mmq=mfq::cuda::gguf::mmq;
            const auto input_row=mmq::activation_bytes(2,model.config.hidden)-mmq::activation_bytes(1,model.config.hidden);
            row_bytes=mfq::engine::prefill_budget_add(row_bytes,2*std::size_t(model.config.topk)*input_row);
        }
        // Whole-prompt residual and FFN injection remain live across tiles.
        // Their previous hidden input and the assembled FFN output can overlap
        // them at a layer boundary. Include that third sequence-sized buffer.
        const auto sequence_reserve=std::size_t(tokens)*model.config.hidden*model.config.streams*sizeof(float)*3;
        auto budget=routed_prefill_workspace_budget(*model.execution,true);
        const auto total_available=mfq::engine::prefill_budget_add(
            context->available_memory()+context->workspace_available(),budget.lendable_bytes);
        const auto minimum=mfq::engine::prefill_budget_add(reserve+sequence_reserve+budget.transfer_bytes,
            std::size_t(std::min<int64_t>(tokens,128))*row_bytes);
        if(minimum<total_available && attention_workspace<=total_available-minimum)
            reserve=mfq::engine::prefill_budget_add(reserve,attention_workspace);
        const auto planned_fixed=mfq::engine::prefill_budget_add(reserve+sequence_reserve,budget.transfer_bytes);
        auto planned_available=mfq::engine::prefill_budget_add(
            context->available_memory()+context->workspace_available(),budget.lendable_bytes);
        const auto required=mfq::engine::PrefillWorkspace{planned_fixed,row_bytes}.bytes(compute_chunk);
        if(required>planned_available) {
            // Later prompt chunks meet an expert cache populated by earlier
            // chunks. Both fixed-slot loans and ordinary revocable GPU images
            // must return workspace before the compute planner reduces a tile.
            reserve_routed_workspace(*model.execution,
                required>budget.lendable_bytes?required-budget.lendable_bytes:0);
            budget=routed_prefill_workspace_budget(*model.execution,true);
            planned_available=mfq::engine::prefill_budget_add(
                context->available_memory()+context->workspace_available(),budget.lendable_bytes);
        }
        compute_chunk=mfq::engine::select_prefill_chunk(tokens,compute_chunk,planned_available,planned_fixed,row_bytes);
        TYLOINFER_RUNTIME_CHECK(compute_chunk>0,"insufficient CUDA prefill workspace before expert loan");
        borrowed_workspace=workspace_loan.begin(mfq::engine::PrefillWorkspace{planned_fixed,row_bytes}.bytes(compute_chunk))>0;
        const auto available=context->available_memory()+context->workspace_available();
        const auto requested=compute_chunk;
        const auto fixed=reserve+sequence_reserve+routed_prefill_transfer_bytes(*model.execution);
        compute_chunk=mfq::engine::select_prefill_chunk(tokens,compute_chunk,available,fixed,row_bytes);
        TYLOINFER_RUNTIME_CHECK(compute_chunk>0,"insufficient CUDA prefill workspace after expert loan");
        if(compute_chunk!=requested)std::cerr<<"runtime_prefill_chunk_budget requested="<<requested
            <<" compute="<<compute_chunk<<" available_bytes="<<available<<'\n';
    }
    const auto pinned_transaction=mfq::engine::prefill_budget_add(
        qwen4_pinned_transaction_bytes(model,compute_chunk),host_workspace);
    reserve_pinned_workspace(pinned_transaction,pinned_transaction,[&](std::size_t bytes) {
        return reclaim_routed_expert_pinned(*model.execution,bytes);
    });
    RequiredWorkspaceLease required_workspace(mfq_current_cuda_device(),mfq_current_cuda_stream());
    const auto stage_device=tb::Device(tb::kCUDA,mfq_current_cuda_device());
    for(auto* block:model.loaded_blocks)if(block->ple) {
        block->ple->prepare_workspace(compute_chunk,stage_device,required_workspace);
        if(tokens%compute_chunk)block->ple->prepare_workspace(tokens%compute_chunk,stage_device,required_workspace);
    }
    PinnedAllocationScope required_pinned(mfq_current_cuda_device(),mfq_current_cuda_stream(),[&](std::size_t bytes) {
        if(DecodeWindow::recording())return std::size_t(0);
        return reclaim_routed_expert_pinned(*model.execution,bytes);
    });
    mfq::cuda::ResidentAllocationScope persistent_preamble;
    model.adapter_validate_forward(1,tokens,cache_start,false,false,false);
    if(cache_start==0)model.reset(1);
    model.adapter_begin_forward(raw_hidden!=nullptr);
    auto current=tb::arange(cache_start,cache_start+tokens,ids.options());
    if(model.decode_position_delta)current=current+model.decode_position_delta;
    auto prepared=model.adapter_prepare_positions(std::move(current),1,tokens);
    model.adapter_validate_positions(prepared.positions,1,tokens,false);
    auto cpu_ids=ids.to(tb::kCPU,tb::kInt64).contiguous();
    // Expert-cache restoration can consume RAM after the preceding chunk's
    // last attention. Reclaim cold KV before the next PLE packet is pinned.
    const auto host_before=mfq::sample_host_memory().available_host_bytes();
    mfq::cuda::prepare_qsa_ram_workspace(host_workspace,[&] {
        for(auto* block:model.loaded_blocks)if(block->qsa)block->qsa->wait_adaptive();
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        mfq::cuda::QsaRamWorkspaceProgress progress;
        std::unordered_set<const void*> pools;
        for(auto* block:model.loaded_blocks)if(block->qsa) {
            if(const auto memory=block->qsa->adaptive_ram_memory()) {
                if(pools.insert(memory->pool).second) {
                    const auto shared=block->qsa->reclaim_shared_adaptive_workspace();
                    progress.logical+=shared.logical;progress.physical+=shared.physical;
                }
            }
            block->qsa->collect_adaptive();
        }
        return progress;
    });
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1')
        std::fprintf(stderr,"native_prefill_host_pressure actual_match=1 before_bytes=%llu after_bytes=%llu workspace_bytes=%zu before_ple=1\n",
            static_cast<unsigned long long>(host_before),
            static_cast<unsigned long long>(mfq::sample_host_memory().available_host_bytes()),host_workspace);
    // Each pending PLE read owns pinned input and decoded GPU output. Keep
    // only the next layer ahead so these buffers do not scale with depth.
    const auto prefetch_ids=cpu_ids.narrow(1,0,std::min(tokens,compute_chunk));
    if(!model.blocks.empty())model.blocks.front()->prefetch_token_ids(prefetch_ids);
    persistent_preamble.finish();
    Tensor hidden;
    std::vector<Tensor> host_tiles;
    const auto cpu=tb::Device(tb::kCPU);
    if(host_activations)host_tiles.reserve(std::size_t((tokens+compute_chunk-1)/compute_chunk));
    else hidden=tb::empty({1,tokens,model.hidden_size()*model.hc_mult()},ids.options().dtype(tb::kFloat16));
    for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
        if(cancelled && cancelled())return {};
        const auto count=std::min(compute_chunk,tokens-begin);
        auto input=model.adapter_prepare_hidden(model.embed_tokens(ids.narrow(1,begin,count)),1,count);
        if(host_activations){host_tiles.push_back(input.to(cpu));continue;}
        TYLOINFER_RUNTIME_CHECK(input.scalar_type()==hidden.scalar_type() &&
            input.size(-1)==hidden.size(-1),"layer-major embedding geometry changed");
        hidden.narrow(1,begin,count).copy_(input);
    }
    const auto trace_option=mfq::cuda::runtime_options::prefill_layer_major_trace();
    const bool trace=trace_option && *trace_option!=0;
    const auto batch_option=mfq::cuda::runtime_options::moe_prefill_expert_batch();
    const bool batch_experts=!host_activations && (!batch_option || *batch_option!=0);
    const auto bulk_gr_option=mfq::cuda::runtime_options::gr_prefill_layer_batch();
    // Long prompts retain the established memory footprint unless explicitly
    // requested; the combined residual tensors have been verified to 16k.
    const bool bulk_gr=batch_experts && (bulk_gr_option?*bulk_gr_option:tokens<=16384);
    const auto bulk_gdn_option=mfq::cuda::runtime_options::gdn_prefill_layer_batch();
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    // Allow 8k chat framing while keeping the extra whole-sequence QKV and
    // gate tensors out of the 16k service path, which also retains MTP weights.
    const bool bulk_gdn=bulk_gr && (bulk_gdn_option?*bulk_gdn_option:tokens<=9216);
#else
    const bool bulk_gdn=bulk_gr && bulk_gdn_option && *bulk_gdn_option;
#endif
    const qwen4_exp::QwenPrefillReferenceRowsScope reference_rows(batch_experts?std::min<int64_t>(1024,compute_chunk):0);
    const auto batch_tokens_option=mfq::cuda::runtime_options::moe_prefill_expert_batch_tokens();
    const int64_t batch_tokens=batch_tokens_option?*batch_tokens_option:8192;
    const auto append_tail_option=mfq::cuda::runtime_options::moe_prefill_append_tail();
    const bool append_tail=append_tail_option && *append_tail_option;
    const auto expert_batch_count=[&](int64_t remaining) {
        // Chat framing can leave a few tokens beyond the batch boundary.
        // Absorb at most one reference chunk instead of running a second
        // almost-empty routed GEMM for every layer.
        return append_tail && remaining<=batch_tokens+1024?remaining:std::min(batch_tokens,remaining);
    };
    TYLOINFER_RUNTIME_CHECK(!batch_experts || batch_tokens>=1024,"expert prefill batch must cover at least 1024 tokens");
    const auto audit_option=mfq::cuda::runtime_options::prefill_expert_batch_audit();
    const bool audit_batch=audit_option && *audit_option!=0;
    const auto started=std::chrono::steady_clock::now();
    int layer=0;
    Tensor final_host_path_tile;
    for(auto& block:model.blocks) {
        auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
        const bool last_layer=std::size_t(layer+1)==model.blocks.size();
        const qwen4_exp::Qsa::LayerPrefillWorkspaceScope layer_kv(
            host_activations?qwen_block.qsa.get():nullptr,cache_start+tokens);
        if(std::size_t(layer+1)<model.blocks.size())
            model.blocks[std::size_t(layer+1)]->prefetch_token_ids(prefetch_ids);
        const auto decode_reuse_option=mfq::cuda::runtime_options::nint_prefill_decode_reuse();
        NintPrefillDecodeCacheScope decode_reuse(!decode_reuse_option || *decode_reuse_option);
        Tensor next_hidden=hidden;
        Tensor ffn_inputs,ffn_injection;
        std::vector<Tensor> full_mix;
        Tensor full_branch;
        if(bulk_gr) {
            auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
            Tensor source=hidden;
            if(qwen_block.ple) {
                for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                    if(cancelled && cancelled())return {};
                    const auto count=std::min(compute_chunk,tokens-begin);
                    Block::Context context;
                    context.token_ids=ids.narrow(1,begin,count);
                    context.host_token_ids=cpu_ids.narrow(1,begin,count);
                    block->set_token_ids(context.token_ids);
                    block->prefetch_token_ids(context.host_token_ids);
                    auto input=qwen_block.prefill_ple(*model.execution,hidden.narrow(1,begin,count),context);
                    if(begin==0)source=tb::empty(hidden.sizes(),input.options());
                    source.narrow(1,begin,count).copy_(input);
                }
            }
            full_mix=profiler.measure("prefill.gr_attention",[&] {
                return qwen_block.attention_gr.pre(*model.execution,source);
            });
            if(bulk_gdn && qwen_block.gdn) {
                if(cancelled && cancelled())return {};
                const qwen4_exp::QwenGdnPrefillProjectionRowsScope projection_rows(compute_chunk);
                full_branch=profiler.measure("prefill.gdn",[&] {
                    return qwen_block.gdn->forward(*model.execution,full_mix[0],true,0);
                });
            }
        }
        for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
            if(cancelled && cancelled())return {};
            if(bulk_gdn && static_cast<qwen4_exp::Qwen4Block&>(*block).gdn)continue;
            const auto count=std::min(compute_chunk,tokens-begin);
            profiler.enabled=restore.enabled;
            Block::Context context;
            context.token_ids=ids.narrow(1,begin,count);
            context.host_token_ids=cpu_ids.narrow(1,begin,count);
            context.positions=prepared.positions.narrow(-1,begin,count);
            context.full_positions=prepared.full_positions.narrow(-1,0,cache_start+begin+count);
            context.cache_position=cache_start+begin;
            block->set_token_ids(context.token_ids);
            block->prefetch_token_ids(context.host_token_ids);
            const auto tile=std::size_t(begin/compute_chunk);
            auto input=host_activations?host_tiles[tile].to(stage_device):hidden.narrow(1,begin,count);
            Tensor output;
            if(bulk_gr) {
                auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
                auto branch=qwen_block.prefill_attention_core(*model.execution,full_mix[0].narrow(1,begin,count),context);
                if(!full_branch.defined()) {
                    auto shape=branch.sizes().vec();shape[1]=tokens;
                    full_branch=tb::empty(shape,branch.options());
                }
                full_branch.narrow(1,begin,count).copy_(branch);
                continue;
            } else if(batch_experts) {
                auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
                auto mix=qwen_block.prefill_attention(*model.execution,input,context);
                if(!ffn_inputs.defined()) {
                    ffn_inputs=tb::empty({1,tokens,model.hidden_size()},mix[0].options().dtype(tb::kFloat16));
                    ffn_injection=tb::empty({1,tokens,mix[2].size(-1)},mix[2].options());
                }
                // qwen_moe casts the router/FFN source to half at this boundary.
                ffn_inputs.narrow(1,begin,count).copy_(mix[0].to(tb::kFloat16));
                ffn_injection.narrow(1,begin,count).copy_(mix[2]);
                output=std::move(mix[1]);
            } else output=block->forward_context(*model.execution,input,context,model.rope);
            // Retain the original rounding at every layer boundary. The next
            // chunk is independent of this input once its cache writes retire.
            TYLOINFER_RUNTIME_CHECK(output.sizes()==input.sizes(),"layer-major decoder changed hidden shape");
            if(host_activations) {
                if(last_layer) {
                    host_tiles[tile]={};
                    if(begin+count==tokens)final_host_path_tile=std::move(output);
                } else host_tiles[tile]=output.to(cpu);
                continue;
            }
            if(begin==0 && output.scalar_type()!=hidden.scalar_type()) {
                next_hidden=tb::empty(hidden.sizes(),output.options());
                if(trace)std::cout<<"prefill_layer_major_hidden layer="<<layer
                    <<" input_bytes="<<hidden.nbytes()<<" output_bytes="<<next_hidden.nbytes()<<std::endl;
            }
            TYLOINFER_RUNTIME_CHECK(output.scalar_type()==next_hidden.scalar_type(),
                "layer-major decoder output dtype changed within a layer");
            next_hidden.narrow(1,begin,count).copy_(output);
        }
        // Attention's decoded matrices are no longer needed by FFN. Keep
        // reuse within each phase, without retaining both sets at peak memory.
        if(tokens>9216)decode_reuse.clear();
        if(bulk_gr) {
            auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
            auto mix=profiler.measure("prefill.gr_ffn_pre",[&] {
                return qwen_block.ffn_gr.pre_after(*model.execution,full_branch,full_mix);
            });
            ffn_inputs=mix[0].to(tb::kFloat16);ffn_injection=std::move(mix[2]);
            next_hidden=std::move(mix[1]);
            full_mix.clear();full_branch={};
        }
        if(tokens>9216)decode_reuse.clear();
        // The new residual owns the next layer's values. Prior hidden readers
        // have already been submitted before this stream-ordered release.
        if(tokens>9216)hidden={};
        if(batch_experts) {
            if(cancelled && cancelled())return {};
            auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
            profiler.enabled=restore.enabled;
            auto branch=profiler.measure("prefill.ffn",[&] {
                if(expert_batch_count(tokens)==tokens)return qwen_block.ffn(*model.execution,ffn_inputs);
                Tensor result;
                for(int64_t begin=0;begin<tokens;) {
                    if(cancelled && cancelled())return Tensor{};
                    auto count=expert_batch_count(tokens-begin);
                    const auto tail=tokens-begin-count;
                    // A tiny final batch would enter the decode FFN path and
                    // release the shared prefill images halfway through every
                    // layer. Keep both batches in prefill, including chat inputs
                    // just above the expert-batch boundary.
                    if(tail>0 && tail<=8)count-=16-tail;
                    auto part=qwen_block.ffn(*model.execution,ffn_inputs.narrow(1,begin,count));
                    if(!result.defined()) {
                        auto shape=part.sizes().vec();shape[1]=tokens;
                        result=tb::empty(shape,part.options());
                    }
                    result.narrow(1,begin,count).copy_(part);
                    begin+=count;
                }
                return result;
            });
            if(!branch.defined())return {};
            if(audit_batch) {
                auto expected=tb::empty(branch.sizes(),branch.options());
                for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                    const auto count=std::min(compute_chunk,tokens-begin);
                    expected.narrow(1,begin,count).copy_(
                        qwen_block.ffn(*model.execution,ffn_inputs.narrow(1,begin,count)));
                }
                auto a=branch.to(tb::kCPU,tb::kFloat32).contiguous();
                auto b=expected.to(tb::kCPU,tb::kFloat32).contiguous();
                double error=0,scale=0,maximum=0;
                for(int64_t i=0;i<a.numel();++i) {
                    const double delta=double(a.data_ptr<float>()[i])-b.data_ptr<float>()[i];
                    error+=delta*delta;scale+=double(b.data_ptr<float>()[i])*b.data_ptr<float>()[i];
                    maximum=std::max(maximum,std::abs(delta));
                }
                std::cout<<"prefill_expert_batch_layer_oracle layer="<<layer<<" tokens="<<tokens
                    <<" relative_rms="<<std::sqrt(error/std::max(1e-30,scale))
                    <<" maximum_absolute="<<maximum<<std::endl;
                // Diagnostic only: continue with the reference so later
                // layers are tested with their original input distribution.
                branch=std::move(expected);
            }
            Tensor post_hidden=next_hidden;
            profiler.measure("prefill.gr_ffn_post",[&] {
                for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                    const auto count=std::min(compute_chunk,tokens-begin);
                    std::vector<Tensor> mix{{},next_hidden.narrow(1,begin,count),ffn_injection.narrow(1,begin,count)};
                    if(qwen_block.ffn_gr.post_into(branch.narrow(1,begin,count),mix,post_hidden.narrow(1,begin,count)))continue;
                    auto output=qwen_block.ffn_gr.post(branch.narrow(1,begin,count),mix);
                    if(begin==0 && output.scalar_type()!=next_hidden.scalar_type())
                        post_hidden=tb::empty(next_hidden.sizes(),output.options());
                    TYLOINFER_RUNTIME_CHECK(output.scalar_type()==post_hidden.scalar_type(),
                        "expert-batched residual dtype changed");
                    post_hidden.narrow(1,begin,count).copy_(output);
                }
                return true;
            });
            next_hidden=std::move(post_hidden);
        }
        hidden=std::move(next_hidden);
        if (!host_activations && model.hidden_capture) model.hidden_capture->capture(layer, hidden, model.hc_mult());
        ++layer;
        if(trace && (layer%4==0 || layer==int(model.blocks.size()))) {
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_get_current_cuda_stream()));
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            const auto memory=mfq::cuda::default_context(ids.get_device())->memory_stats();
            const auto local=mfq::cuda::default_context(ids.get_device())->local_memory_usage();
            cudaMemPool_t pool=nullptr;uint64_t reserved=0,used=0;
            TYLOINFER_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool,ids.get_device()));
            TYLOINFER_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReservedMemCurrent,&reserved));
            TYLOINFER_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrUsedMemCurrent,&used));
            std::cout<<"prefill_layer_major_memory tokens="<<tokens<<" chunk="<<compute_chunk
                <<" completed_layers="<<layer<<" owned_bytes="<<memory.allocated
                <<" process_local_bytes="<<local<<" pool_used_bytes="<<used<<" pool_reserved_bytes="<<reserved<<std::endl;
#endif
            if(layer==int(model.blocks.size())) {
                const auto host=mfq::sample_host_memory();
                std::cout<<"prefill_host_memory rss_bytes="<<host.resident_bytes
                    <<" available_bytes="<<host.available_host_bytes();
                for(const auto& [name,value]:mfq::cuda::routed_expert_memory_metrics(*model.execution))
                    if(name=="host_tensor_bytes" || name=="ram_expert_payload_bytes" ||
                            name=="ram_expert_cache_allocated_bytes" || name=="ram_expert_duplicate_payload_bytes" ||
                            name=="ram_expert_unique_payload_bytes" || name=="ram_expert_pinned_reserved_bytes" ||
                            name=="ram_expert_pageable_payload_bytes" || name=="resident_expert_cache_bytes" ||
                            name=="gguf_decode_dma_buffer_bytes" || name=="ssd_expert_bytes_read" ||
                            name=="gguf_prefill_host_bypass_bytes" || name=="expert_host_fetch_ms")
                        std::cout<<' '<<name<<'='<<value;
                for(const auto& [name,value]:qwen4_cache_memory_metrics(model,nullptr))
                    if(name.starts_with("qsa_adaptive_"))std::cout<<' '<<name<<'='<<std::uint64_t(value);
                std::cout<<std::endl;
            }
            std::cout<<"prefill_layer_major_progress tokens="<<tokens<<" chunk="<<compute_chunk
                <<" completed_layers="<<layer<<" total_layers="<<model.blocks.size()
                <<" elapsed_ms="<<std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-started).count()<<std::endl;
        }
    }
    profiler.enabled=restore.enabled;
    const auto finish_workspace=[&] {
        if(!borrowed_workspace)return;
        // Canonical append images can retire after their layer has returned.
        // Release completed GPU owners before restoring the borrowed experts;
        // ongoing CPU quantization keeps its independent RAM image leases.
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        for(auto* block:model.loaded_blocks)if(block->qsa)block->qsa->collect_adaptive();
        workspace_loan.finish();
    };
    if(last_only && !raw_hidden) {
        const auto begin=((tokens-1)/compute_chunk)*compute_chunk;
        Tensor result;
        {
            auto last_tile=host_activations?std::move(final_host_path_tile):hidden.narrow(1,begin,tokens-begin);
            auto finalized=model.finalize_hidden(std::move(last_tile),1,tokens-begin);
            result=model.last_hidden(std::move(finalized)).unsqueeze(1);
        }
        if(computation_finished)computation_finished();
        if(borrowed_workspace) {
            workspace_loan.suspend();
            result=result.clone();
            hidden={};
            finish_workspace();
        }
        model.adapter_finish_forward(prepared.full_positions,1,tokens);
        model.cache_pos=cache_start+tokens;
        return result;
    }
    Tensor finalized;
    if(last_only) {
        // Use the same final GR dispatch and rounding as chunk-major prefill.
        const auto begin=((tokens-1)/compute_chunk)*compute_chunk;
        finalized=model.finalize_hidden(hidden.narrow(1,begin,tokens-begin),1,tokens-begin);
        finalized=model.last_hidden(std::move(finalized)).unsqueeze(1);
    } else {
        for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
            const auto count=std::min(compute_chunk,tokens-begin);
            auto part=model.finalize_hidden(hidden.narrow(1,begin,count),1,count);
            if(!finalized.defined())finalized=tb::empty({1,tokens,part.size(-1)},part.options());
            finalized.narrow(1,begin,count).copy_(part);
        }
    }
    Tensor captured_raw;
    if (raw_hidden && model.hidden_capture && model.hidden_capture->active)
        captured_raw = model.hidden_capture->result(hidden);
    if(raw_hidden && raw_hidden_host) {
        if (captured_raw.defined()) captured_raw = captured_raw.cpu();
        else hidden=hidden.to(tb::kCPU);
    }
    if(computation_finished)computation_finished();
    if(borrowed_workspace) {
        // The final mixer also needs sequence-sized scratch. Finish it inside
        // the loan, then retain only the outputs consumed by MTP/the caller.
        workspace_loan.suspend();
        if (captured_raw.defined() && !raw_hidden_host) captured_raw = captured_raw.clone();
        if (model.hidden_capture) model.hidden_capture->begin(false);
        if(raw_hidden) {if(!raw_hidden_host)hidden=hidden.clone();}else hidden={};
        finalized=finalized.clone();
        finish_workspace();
    }
    model.adapter_finish_forward(prepared.full_positions,1,tokens);
    model.cache_pos=cache_start+tokens;
    if(raw_hidden) {
        *raw_hidden = captured_raw.defined() ? captured_raw : hidden;
        if (raw_hidden_host && raw_hidden->is_cuda()) *raw_hidden = raw_hidden->cpu();
    }
    return finalized;
}

mfq_tensor_backend::Tensor Qwen4Model::adapter_finalize_hidden(mfq_tensor_backend::Tensor hidden,
                                                               const mfq_tensor_backend::Tensor &,
                                                               int64_t, int64_t) const {
    auto* window=DecodeWindow::recording();
    GatedResidualMarker marker;
    if(window && runtime_options::mhc_timings())marker=[window](const char* phase) {
        window->mark(-1,(std::string("final_mhc_")+phase).c_str());
    };
    return final_mixer->pre(*execution, hidden,marker)[0];
}
mfq_tensor_backend::Tensor Qwen4Model::adapter_raw_hidden(const mfq_tensor_backend::Tensor& hidden,
    const mfq_tensor_backend::Tensor&) const {
    return hidden_capture ? hidden_capture->result(hidden) : hidden;
}

mfq_tensor_backend::Tensor
Qwen4Model::adapter_last_logits(const QuantLinear &lm_head,
                                mfq_tensor_backend::Tensor hidden) const {
    return adapter_logits(lm_head, std::move(hidden));
}

mfq_tensor_backend::Tensor Qwen4Model::adapter_next_token(const QuantLinear &lm_head,
                                                          mfq_tensor_backend::Tensor hidden) const {
    auto* window=DecodeWindow::recording();
    if(window)window->mark(-1,"lm_head_begin");
    auto logits=adapter_logits(lm_head,std::move(hidden));
    if(window)window->mark(-1,"lm_head_projection_end");
    auto output=mfq_tensor_backend::argmax(logits,-1).to(mfq_tensor_backend::kInt64);
    if(window)window->mark(-1,"lm_head_sample_end");
    return output;
}

void Qwen4Model::adapter_rollback_speculative(int64_t keep) {
    speculative_prefix_retention=false;
    if (positions.defined())
        positions = positions.narrow(-1, 0, keep);
}

} // namespace mfq::cuda

namespace mfq::cuda {

TextSessionStateKind CudaSessionCodec<Qwen4Model>::kind(const Qwen4CausalLm& model) {
    if(!model.execution || model.blocks.empty())return TextSessionStateKind::Unsupported;
    for(const auto& base:model.blocks) {
        const auto& block=static_cast<const qwen4_exp::Qwen4Block&>(*base);
        if(block.qsa && !block.qsa->supports_session())return TextSessionStateKind::Unsupported;
    }
    return TextSessionStateKind::Qwen4;
}
bool CudaSessionCodec<Qwen4Model>::supports_paged(const Qwen4CausalLm& model) { return kind(model)==TextSessionStateKind::Qwen4; }

TextSessionState CudaSessionCodec<Qwen4Model>::capture(const Qwen4CausalLm& model,const std::vector<int64_t>& tokens) {
    TYLOINFER_RUNTIME_CHECK(kind(model)==TextSessionStateKind::Qwen4 && model.batch==1 && model.cache_pos>0 &&
        std::size_t(model.cache_pos)==tokens.size() && model.speculative_start<0,
        "Qwen4 text session boundary is unavailable");
    TextSessionState state;state.tokens=tokens;state.cache_pos=model.cache_pos;
    std::vector<Qwen4BlockSessionState> blocks;
    for(const auto& block:model.blocks)blocks.push_back(qwen4_exp::capture_block_session(
        static_cast<const qwen4_exp::Qwen4Block&>(*block),model.cache_pos,state.bytes));
    state.payload=std::move(blocks);
    state.qwen4_positions=model.positions.clone();state.bytes+=session_tensor_bytes(state.qwen4_positions);
    return state;
}

void CudaSessionCodec<Qwen4Model>::restore(Qwen4CausalLm& model,const TextSessionState& state) {
    const auto* blocks=std::get_if<std::vector<Qwen4BlockSessionState>>(&state.payload);
    TYLOINFER_RUNTIME_CHECK(kind(model)==TextSessionStateKind::Qwen4 && blocks && blocks->size()==model.blocks.size() &&
        state.cache_pos>0 && std::size_t(state.cache_pos)==state.tokens.size() &&
        state.qwen4_positions.defined() && state.qwen4_positions.size(-1)==state.cache_pos,
        "incompatible Qwen4 text session state");
    model.reset(1);
    std::vector<std::shared_ptr<void>> reservations;
    for(std::size_t i=0;i<blocks->size();++i) {
        auto& block=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]);
        if(block.qsa)reservations.push_back(block.qsa->reserve_session_restore((*blocks)[i]));
    }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    const auto* placement=std::getenv("TYLOINFER_QSA_CPU_INDEXER");
    if((!placement || !*placement) && !std::getenv("TYLOINFER_QWEN4_BATCH_HOST_INDEXER_BYTES")) {
        // With a shared prefix, the remaining RAM demand is now known. Price
        // all request state before interleaving GPU Indexers and recurrent
        // allocations, so an earlier Indexer cannot consume the last GDN pages.
        const auto device=model.execution->layer_placement.primary_device();
        const auto context=default_context(device);
        auto needed=mfq::engine::prefill_budget_add(qwen4_prefill_state_growth_bytes(model,1),
            state.qwen4_positions.nbytes());
        if(needed>context->available_memory()) {
            context->reclaim_idle_memory();
            const auto available=context->available_memory();
            std::size_t placed=0;
            for(auto* block:model.loaded_blocks)if(needed>available && block->qsa) {
                const auto bytes=block->qsa->indexer_growth_bytes();
                if(!bytes || !block->qsa->place_unallocated_indexer_in_ram(true))continue;
                block->qsa->reserve_indexer();
                needed-=bytes;placed+=bytes;
            }
            if(placed && std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH"))
                std::fprintf(stderr,"native_qwen4_prefix_indexer_placement bytes=%zu device_demand=%zu available=%zu\n",
                    placed,needed,available);
        }
    }
#endif
    const auto admit_host_recurrence=[&](std::size_t bytes) {
        if(std::getenv("TYLOINFER_QWEN4_BATCH_HOST_RECURRENCE_BYTES"))return false;
        return std::any_of(model.loaded_blocks.begin(),model.loaded_blocks.end(),[&](const auto* block) {
            return block->qsa && block->qsa->restore_can_place_in_ram(bytes);
        });
    };
    for(std::size_t i=0;i<blocks->size();++i)qwen4_exp::restore_block_session(
        static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]),(*blocks)[i],state.cache_pos,admit_host_recurrence);
    model.positions=state.qwen4_positions.clone();model.cache_pos=state.cache_pos;
}

} // namespace mfq::cuda

namespace mfq::models {
template struct qwen4_exp::CausalLm<cuda::CudaCausalOps<cuda::Qwen4Model>>;
} // namespace mfq::models
